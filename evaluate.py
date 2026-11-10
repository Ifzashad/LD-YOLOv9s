import argparse, hashlib, json
from pathlib import Path
from val_dual import run
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--weights',required=True);p.add_argument('--output',required=True);p.add_argument('--metrics',required=True);a=p.parse_args()
    output=Path(a.output)
    r,m,t=run(data='datasets/local.yaml',weights=a.weights,batch_size=8,imgsz=640,task='test',device='0',workers=4,half=False,conf_thres=.001,iou_thres=.7,project=str(output.parent),name=output.name,plots=True,verbose=True,save_txt=True,save_conf=True,exist_ok=False)
    result=dict(zip(('precision','recall','map50','map50_95'),map(float,r[:4])))
    result.update(per_class_map50_95=list(map(float,m)),timing_ms=list(map(float,t)),weights_sha256=hashlib.sha256(Path(a.weights).read_bytes()).hexdigest(),split='test',conf_thres=.001,nms_iou=.7)
    Path(a.metrics).write_text(json.dumps(result,indent=2))
