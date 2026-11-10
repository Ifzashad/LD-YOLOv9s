"""Plot only measured results; export source and aggregate figures without dataset files."""
import csv,json,shutil
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
def report(run):
    out=ROOT/'results';out.mkdir(exist_ok=True)
    rows=[]
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for name in ('baseline','dyn','dyn-sd','full','sd-only','mpdiou-only'):
        metric=run/f'{name}-test.json'
        if not metric.exists():continue
        m=json.loads(metric.read_text());rows.append({'variant':name,**{k:m[k] for k in ('precision','recall','map50','map50_95')}})
        shutil.copy2(metric,out/metric.name)
        hist=pd.read_csv(run/name/'results.csv');hist.columns=hist.columns.str.strip()
        axes[0].plot(hist['epoch'],hist['train/box_loss'],label=name)
        axes[1].plot(hist['epoch'],100*hist['metrics/mAP_0.5:0.95'],label=name)
        for f in ('confusion_matrix.png','PR_curve.png','F1_curve.png','P_curve.png','R_curve.png'):
            src=run/f'{name}-test'/f
            if src.exists():shutil.copy2(src,out/f'{name}-{f}')
    if not rows:raise ValueError('No measured test results')
    pd.DataFrame(rows).to_csv(out/'comparison.csv',index=False)
    axes[0].set(title='Training box loss',xlabel='Epoch',ylabel='Loss')
    axes[1].set(title='Validation mAP50–95',xlabel='Epoch',ylabel='Percent')
    for ax in axes:ax.legend();ax.grid(alpha=.15)
    fig.tight_layout();fig.savefig(out/'training-curves.png',dpi=160);plt.close(fig)
    df=pd.DataFrame(rows).set_index('variant')*100
    ax=df[['map50','map50_95']].plot.bar(color=['#16818d','#8058b5'],figsize=(10,4),rot=0)
    ax.set(ylabel='Percent',ylim=(0,105),title='Held-out test: measured detection performance')
    ax.figure.tight_layout();ax.figure.savefig(out/'comparison.png',dpi=160);plt.close(ax.figure)
    (out/'README.md').write_text('# Measured results\n\n'+pd.DataFrame(rows).to_csv(index=False)+'\nMetrics are newly measured, not transcribed paper scores.\n')
    readme=ROOT/'README.md';text=readme.read_text()
    text=text.replace('**Status: implementation checks only. Training and test results pending.**','**Status: measured test results available in results/.**')
    text=text.split('<!-- measured-results -->')[0].rstrip()
    text+='\n\n<!-- measured-results -->\n## Measured results\n\n![Test comparison](results/comparison.png)\n\n![Training and validation](results/training-curves.png)\n'
    for name in ('baseline','full'):
        if (out/f'{name}-confusion_matrix.png').exists():text+=f'\n### {name} test confusion matrix\n\n![{name} confusion matrix](results/{name}-confusion_matrix.png)\n'
    readme.write_text(text)
    print('RESULT FIGURES READY',out)

def export():
    import datetime,hashlib
    dest=ROOT/'exports'/datetime.datetime.now().strftime('github-%Y%m%d-%H%M%S')
    dest.mkdir(parents=True)
    # Explicit allowlist: no datasets, environments, private URL, logs or checkpoints.
    for name in ('models','utils','configs','results'):
        if (ROOT/name).exists():
            shutil.copytree(ROOT/name,dest/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('README.md','REPRODUCTION.md','requirements.txt','LICENSE.md','.gitignore','env.sh','setup_server.sh','prepare_data.py','check_project.py','run_project.py','evaluate.py','report_results.py','train_dual.py','val_dual.py','detect.py','detect_dual.py'):
        shutil.copy2(ROOT/name,dest/name)
    files={str(p.relative_to(dest)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.rglob('*') if p.is_file()}
    (dest/'SHA256SUMS.json').write_text(json.dumps(files,indent=2))
    print('CODE + RESULTS EXPORT',dest,'No dataset or checkpoint included.')
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path);p.add_argument('--export',action='store_true');a=p.parse_args()
    if a.run:report(a.run.resolve())
    if a.export:export()
