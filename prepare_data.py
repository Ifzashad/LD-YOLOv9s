"""Import Roboflow v11 YOLO ZIP, validate labels, retain split, audit leakage."""
import argparse, hashlib, json, re, time, zipfile
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from PIL import Image
import requests, yaml

ROOT=Path(__file__).resolve().parent
EXPECTED={'aluminum','battery','can','cap or lid','cardboard boxes and cartons','food & others','glass','paper','plastic bag or wrapper','plastic container','styrofoam','utensils and straw'}
SOURCE='https://universe.roboflow.com/taco-ihjgk/yolov8-trash-detections-kgnug/dataset/11'

def download():
    secret=ROOT/'.secrets/download-url'
    if not secret.exists():raise SystemExit('Save Roboflow YOLOv9 ZIP download URL in .secrets/download-url first.')
    url=secret.read_text().strip()
    if not url.startswith('https://'):raise SystemExit('HTTPS download URL required.')
    target=ROOT/'datasets/download.zip';target.parent.mkdir(exist_ok=True)
    if target.exists():return target
    part=target.with_suffix('.zip.part')
    for attempt in range(12):
        try:
            offset=part.stat().st_size if part.exists() else 0
            with requests.get(url,headers={'Range':f'bytes={offset}-'} if offset else {},stream=True,timeout=(30,120)) as r:
                if r.status_code==416 and zipfile.is_zipfile(part):
                    with zipfile.ZipFile(part) as z:assert z.testzip() is None
                    part.rename(target);return target
                if r.status_code not in (200,206):raise RuntimeError('HTTP '+str(r.status_code))
                append=r.status_code==206 and offset>0
                if append and not r.headers.get('Content-Range','').startswith(f'bytes {offset}-'):raise RuntimeError('Unexpected range')
                with part.open('ab' if append else 'wb') as f:
                    for chunk in r.iter_content(1024*1024):
                        f.write(chunk)
                with zipfile.ZipFile(part) as z:assert z.testzip() is None
                part.rename(target)
                print('DOWNLOAD VERIFIED',target,flush=True);return target
        except Exception as e:
            # Never print exception URL: signed URL can contain credentials.
            print('Download retry',attempt+1,type(e).__name__,flush=True);time.sleep(10)
    raise SystemExit('Download failed. Partial file preserved; refresh private URL and retry.')

def unpack(archive):
    target=ROOT/'datasets/roboflow-v11'
    target.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        if z.testzip():raise ValueError('Corrupt ZIP')
        for info in z.infolist():
            dest=(target/info.filename).resolve()
            if not dest.is_relative_to(target.resolve()) or (info.external_attr>>16)&0o170000==0o120000:raise ValueError('Unsafe ZIP entry')
        z.extractall(target)
    return target

def audit(root):
    files=list(root.rglob('data.yaml'))
    if len(files)!=1:raise ValueError('Expected exactly one data.yaml')
    base=files[0].parent
    data=yaml.safe_load(files[0].read_text())
    names=data['names'];names=[names[i] for i in range(len(names))] if isinstance(names,dict) else names
    if len(names)!=len(EXPECTED) or {('food & others' if n == 'food - others' else n) for n in names}!=EXPECTED:raise ValueError('Unexpected class names: '+repr(names))
    import os, shutil, tempfile
    view = Path(tempfile.mkdtemp(prefix="detection-", dir=ROOT/"datasets"))
    converted = 0
    rows=[];groups=defaultdict(set);families=defaultdict(set);counts={};errors=[]
    for split in ('train','valid','test'):
        directory=base/split/'images';count=Counter();n=0
        if not directory.is_dir():raise ValueError('Missing split '+split)
        for p in sorted(directory.iterdir()):
            if p.suffix.lower() not in ('.jpg','.jpeg','.png','.bmp','.webp'):continue
            n+=1
            with Image.open(p) as im:
                im=im.convert('RGB');pixels=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
            label=base/split/'labels'/(p.stem+'.txt')
            if not label.exists():errors.append('Missing label '+str(label));continue
            image_dir = view/split/"images"
            label_dir = view/split/"labels"
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            try:
                os.link(p, image_dir/p.name)
            except OSError:
                shutil.copy2(p, image_dir/p.name)
            original_label = label
            detection_rows = []
            for line_no, line in enumerate(label.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    v = list(map(float, line.split()))
                except ValueError:
                    errors.append(f"Invalid number {label}:{line_no}")
                    continue
                if len(v) >= 7 and len(v) % 2 == 1:
                    xy = np.asarray(v[1:]).reshape(-1, 2)
                    if not np.isfinite(xy).all() or xy.min() < -1e-6 or xy.max() > 1 + 1e-6:
                        errors.append(f"Invalid polygon coordinates {label}:{line_no}")
                        continue
                    xy = xy.clip(0, 1)
                    lo, hi = xy.min(axis=0), xy.max(axis=0)
                    v = [v[0], *((lo + hi) / 2), *(hi - lo)]
                    converted += 1
                if len(v)!=5 or not np.isfinite(v).all() or v[0]!=int(v[0]) or not 0<=v[0]<12 or not all(0<=x<=1 for x in v[1:]) or min(v[3:])<=0:
                    errors.append('Invalid box '+str(label));continue
                if v[1]-v[3]/2 < -1e-4 or v[1]+v[3]/2>1.0001 or v[2]-v[4]/2 < -1e-4 or v[2]+v[4]/2>1.0001:errors.append('Out-of-image box '+str(label))
                count[names[int(v[0])]]+=1
                detection_rows.append(" ".join([str(int(v[0])), *(format(x, ".12g") for x in v[1:])]))
            label = label_dir/original_label.name
            label.write_text("".join(row + "\n" for row in detection_rows))
            family=p.name.split('.rf.')[0]
            groups[pixels].add(split);families[family].add(split)
            rows.append(dict(split=split,image=str(p.relative_to(base)),pixel_sha256=pixels,label_sha256=hashlib.sha256(label.read_bytes()).hexdigest(),family=family))
            if n%1000==0:print('Audit',split,n,flush=True)
        counts[split]={'images':n,'objects':dict(count)}
    overlaps={k:sorted(v) for k,v in groups.items() if len(v)>1}
    family_overlap={k:sorted(v) for k,v in families.items() if len(v)>1}
    expected_counts={'train':11076,'valid':891,'test':103}
    for split,n in expected_counts.items():
        if counts[split]['images']!=n:errors.append(f'{split} expected {n} images, found {counts[split]["images"]}')
    report=dict(source=SOURCE,counts=counts,converted_polygons=converted,exact_cross_split=overlaps,family_cross_split=family_overlap,errors=errors,near_duplicate_independence='Not established',status='PASS' if not(errors or overlaps) else 'REVIEW REQUIRED')
    out=ROOT/'results';out.mkdir(exist_ok=True)
    (out/'dataset-audit.json').write_text(json.dumps(report,indent=2))
    manifest=ROOT/'datasets/manifest.json';manifest.write_text(json.dumps(rows,indent=2))
    report['manifest_sha256']=hashlib.sha256(manifest.read_bytes()).hexdigest()
    (out/'dataset-audit.json').write_text(json.dumps(report,indent=2))
    local={'path':str(view.resolve()),'train':'train/images','val':'valid/images','test':'test/images','nc':12,'names':names}
    (ROOT/'datasets/local.yaml').write_text(yaml.safe_dump(local,sort_keys=False))
    print(json.dumps(report,indent=2))
    if report['status']!='PASS':raise SystemExit('Data needs review before training; no files removed or splits changed.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--download',action='store_true');p.add_argument('--zip',type=Path);p.add_argument('--root',type=Path)
    a=p.parse_args()
    root=unpack(download() if a.download else a.zip) if (a.download or a.zip) else (a.root or ROOT/'datasets/roboflow-v11')
    audit(root)
