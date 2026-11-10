"""Documented reconstruction of dynamic convolution, SDConv-ADown and MPDIoU."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from models.common import Conv


class DynConvLayer(nn.Module):
    """Per-image softmax mixture of four learned kernels; BN and SiLU afterwards."""
    def __init__(self, c1, c2, k=3, s=1, experts=4):
        super().__init__()
        self.k, self.s, self.c2 = k, s, c2
        self.weight = nn.Parameter(torch.empty(experts, c2, c1, k, k))
        for w in self.weight:
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        self.router = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                    nn.Linear(c1, max(8,c1//4)), nn.ReLU(),
                                    nn.Linear(max(8,c1//4), experts))
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        b,c,h,w = x.shape
        attention = self.router(x).softmax(-1)
        kernel = (attention @ self.weight.flatten(1)).reshape(b*self.c2,c,self.k,self.k)
        y = F.conv2d(x.reshape(1,b*c,h,w),kernel,stride=self.s,padding=self.k//2,groups=b)
        return self.act(self.bn(y.reshape(b,self.c2,y.shape[-2],y.shape[-1])))


class SDConvADown(nn.Module):
    """ADown split branches; space-to-depth + stride-one conv replaces strided conv."""
    def __init__(self,c1,c2):
        super().__init__()
        assert c1%2==0 and c2%2==0
        self.cv1=Conv(c1*2,c2//2,3,1)
        self.cv2=Conv(c1//2,c2//2,1,1)

    def forward(self,x):
        x=F.avg_pool2d(x,2,1,0,False,True)
        a,b=x.chunk(2,1)
        # ADown average pooling produces odd H/W for even input; pad to even
        # before space-to-depth so both branches have ceil(H/2),ceil(W/2).
        a=F.pad(a,(0,a.shape[-1]%2,0,a.shape[-2]%2),mode='replicate')
        a=torch.cat((a[...,::2,::2],a[...,1::2,::2],a[...,::2,1::2],a[...,1::2,1::2]),1)
        b=F.max_pool2d(b,3,2,1)
        return torch.cat((self.cv1(a),self.cv2(b)),1)


def mpdiou(pred,target,image_hw):
    """Pixel xyxy boxes and actual input (height,width), including rectangular batches."""
    pred,target=pred.float(),target.float()
    inter=(torch.minimum(pred[...,2:],target[...,2:])-torch.maximum(pred[...,:2],target[...,:2])).clamp_min(0).prod(-1)
    area1=(pred[...,2:]-pred[...,:2]).clamp_min(0).prod(-1)
    area2=(target[...,2:]-target[...,:2]).clamp_min(0).prod(-1)
    iou=inter/(area1+area2-inter+1e-7)
    hw=torch.as_tensor(image_hw,device=pred.device,dtype=torch.float32)
    penalty=(pred-target).square().sum(-1)/hw.square().sum().clamp_min(1)
    return (iou-penalty).unsqueeze(-1)
