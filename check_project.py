"""CPU structural/gradient checks; synthetic values are never research results."""
import os
os.environ['OMP_NUM_THREADS']='2'
import copy, gc, tempfile
from pathlib import Path
import torch, yaml
from models.yolo import Model
from models.ld import DynConvLayer, SDConvADown, mpdiou
from utils.loss_tal_dual import ComputeLoss

def main():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    layer=DynConvLayer(3,8,3,2).eval()
    x=torch.randn(2,3,32,32)
    assert torch.allclose(layer(x),torch.cat([layer(t[None]) for t in x]),atol=1e-6)
    for hw in ((32,32),(33,35)):
        assert SDConvADown(8,16)(torch.randn(2,8,*hw)).shape[-2:]==(hw[0]//2,hw[1]//2)
    boxes=torch.tensor([[1.,2.,10.,20.]],requires_grad=True)
    assert torch.allclose(mpdiou(boxes,boxes,(640,640)),torch.ones(1,1))
    assert torch.allclose(mpdiou(boxes*2,boxes.detach()*2+4,(1280,1280)),mpdiou(boxes,boxes.detach()+2,(640,640)),atol=1e-6)
    for variant in ('baseline','dyn','dyn-sd','full','sd-only','mpdiou-only'):
        model=Model(f'configs/{variant}.yaml',ch=3,nc=12).train()
        model.hyp=yaml.safe_load(Path(f'configs/{variant}-hyp.yaml').read_text())
        criterion=ComputeLoss(model)
        targets=torch.tensor([[0,1,.5,.5,.3,.2],[1,4,.3,.4,.2,.3]])
        predictions=model(torch.rand(2,3,64,96))
        loss,items=criterion(predictions,targets)
        assert torch.isfinite(loss)
        loss.backward()
        for name,p in model.named_parameters():
            if p.requires_grad and ('router' in name or 'cv1.conv.weight' in name):
                assert p.grad is not None and torch.isfinite(p.grad).all(),name
        assert criterion.bbox_loss.mpdiou == (variant in ('full','mpdiou-only'))
        if variant=='full':
            model.eval()
            with torch.no_grad():
                before=model(torch.rand(1,3,64,96))[0]
                assert isinstance(before,list) and before[-1].shape[1]==16
                model.fuse()
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'test.pt';torch.save({'model':model},path)
                loaded=torch.load(path,map_location='cpu',weights_only=False)['model']
                with torch.no_grad(): assert torch.isfinite(loaded(torch.rand(1,3,64,96))[0][-1]).all()
        print('PASS',variant,'parameters',sum(p.numel() for p in model.parameters()),flush=True)
        del model,criterion,loss,predictions;gc.collect()
    print('ALL MODEL/LOSS CHECKS PASSED (CPU synthetic; not training results)')

if __name__=='__main__':main()
