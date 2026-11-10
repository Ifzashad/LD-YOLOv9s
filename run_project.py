"""Persistent sequential experiments; evaluate separately after training is frozen."""
import argparse, csv, hashlib, json, os, subprocess, sys
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.chdir(ROOT)
for k,v in {'TORCH_HOME':'.cache/torch','MPLCONFIGDIR':'.cache/matplotlib','YOLOV5_CONFIG_DIR':'.cache/yolo','XDG_CACHE_HOME':'.cache'}.items():
    os.environ[k]=str(ROOT/v);(ROOT/v).mkdir(parents=True,exist_ok=True)
VARIANTS=['baseline','dyn','dyn-sd','full','sd-only','mpdiou-only']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def call(cmd,log):
    print('RUN',cmd,'LOG',log,flush=True)
    with log.open('a') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
def gpu_env(gpu):
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    for row in csv.reader(rows.splitlines()):
        if row[0].strip()==str(gpu):
            if int(row[2])<12000:raise SystemExit('Selected GPU has less than 12GB free; nothing started.')
            os.environ['CUDA_VISIBLE_DEVICES']=row[1].strip();print('Physical GPU',gpu,'UUID',row[1].strip(),'visible device 0',flush=True);return
    raise SystemExit('GPU not found')
def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['train','evaluate','report']);p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--run',type=Path);p.add_argument('--variants',nargs='+',choices=VARIANTS,default=VARIANTS[:4]);p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if a.action=='report':
        from report_results import report
        if not a.run:p.error('--run required')
        report(a.run.resolve());return
    audit=ROOT/'results/dataset-audit.json'
    if not audit.exists():raise SystemExit('Run prepare_data.py first.')
    data=json.loads(audit.read_text())
    if data['status']!='PASS':raise SystemExit('Dataset audit needs review.')
    if sha(ROOT/'datasets/manifest.json')!=data['manifest_sha256']:raise SystemExit('Manifest changed; audit again.')
    if a.action=='evaluate' and not a.run:p.error('--run required')
    run=(a.run or ROOT/'runs'/datetime.now().strftime('study-%Y%m%d-%H%M%S')).resolve()
    if not a.dry_run:
        gpu_env(a.gpu)
        # Keep this lock alive for the whole queue, on Linux servers.
        import fcntl
        lock=(ROOT/'.cache/queue.lock').open('w')
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Another project queue is running.')
        run.mkdir(parents=True,exist_ok=True)
        metadata=run/'study.json'
        signature={f:sha(ROOT/f) for f in ['models/ld.py','models/yolo.py','utils/loss_tal_dual.py','train_dual.py','val_dual.py']+[f'configs/{v}{suffix}.yaml' for v in VARIANTS for suffix in ('','-hyp')]}
        new={'manifest_sha256':data['manifest_sha256'],'code_config_sha256':signature}
        if metadata.exists() and json.loads(metadata.read_text())!=new:raise SystemExit('Code/config/data changed; use a new run.')
        metadata.write_text(json.dumps(new,indent=2))
        (run/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    print('STUDY',run,flush=True)
    for variant in a.variants:
        completed=run/f'{variant}-train-complete.json';best=run/variant/'weights/best.pt'
        if a.action=='train':
            if completed.exists() and best.exists() and json.loads(completed.read_text())['best_sha256']==sha(best):
                print('REUSING completed',variant,flush=True);continue
            if (run/variant).exists():raise SystemExit('Incomplete run exists. Preserve it and choose a new study; do not silently restart.')
            cmd=[sys.executable,'-u','train_dual.py','--cfg',f'configs/{variant}.yaml','--hyp',f'configs/{variant}-hyp.yaml','--data',str(ROOT/'datasets/local.yaml'),'--weights','','--epochs','200','--batch-size','8','--img','640','--optimizer','SGD','--patience','50','--close-mosaic','15','--workers','4','--seed','0','--device','0','--project',str(run),'--name',variant]
            if a.dry_run:print(cmd);continue
            call(cmd,run/f'{variant}-train.log')
            if not best.exists():raise SystemExit('Training returned without checkpoint')
            completed.write_text(json.dumps({'best_sha256':sha(best),'command':cmd},indent=2))
        else:
            if not completed.exists() or not best.exists():raise SystemExit('Training not confirmed complete: '+variant)
            metrics=run/f'{variant}-test.json'
            if metrics.exists():
                if json.loads(metrics.read_text())['weights_sha256']!=sha(best):raise SystemExit('Checkpoint changed')
                print('REUSING evaluation',variant);continue
            cmd=[sys.executable,'-u','evaluate.py','--weights',str(best),'--output',str(run/f'{variant}-test'),'--metrics',str(metrics)]
            if a.dry_run:print(cmd);continue
            call(cmd,run/f'{variant}-test.log')
    print(a.action.upper(),'QUEUE COMPLETE',run,flush=True)
    if a.action=='train':print('Test untouched. Evaluate fixed checkpoints when all planned training is complete.')
    elif not a.dry_run:
        from report_results import report
        report(run)
if __name__=='__main__':main()
