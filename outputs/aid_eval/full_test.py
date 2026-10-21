"""Frozen AID 2,000-image test evaluation with resumable predictions and image publication."""
import concurrent.futures as cf
import hashlib
import io
import json
import os
from pathlib import Path
import time
import uuid
import requests
import numpy as np
from PIL import Image
import braintrust as bt
import eval as pilot

BASE=Path(__file__).resolve().parent
ROOT=BASE/'test-2000'
ROOT.mkdir(exist_ok=True)
(ROOT/'images').mkdir(exist_ok=True)
CONFIG=json.loads((BASE/'config.json').read_text())
LABELS=CONFIG['labels']
SPLITS=json.loads((BASE/'split-manifest.json').read_text())
NAMES=sorted(SPLITS['test'])
assert len(NAMES)==2000 and len(set(NAMES))==2000
assert not set(NAMES)&set(SPLITS['train']) and not set(NAMES)&set(SPLITS['val'])
KEYS={}
for line in Path('.env').read_text().splitlines():
 if line.strip() and not line.lstrip().startswith('#') and '=' in line:
  k,v=line.split('=',1);KEYS[k.strip()]=v.strip().strip(chr(34)).strip(chr(39))
for k in ['OPENAI_API_KEY','TYPESAFE_API_KEY','BRAINTRUST_API_KEY']:assert KEYS.get(k)
pilot.ROOT=ROOT
MANIFEST={**CONFIG,'split':'test','images':2000,'dataset_name':'aid-test-2000',
 'instructions':pilot.INSTRUCTIONS,'frozen_label_order':LABELS,
 'instructions_sha256':hashlib.sha256(pilot.INSTRUCTIONS.encode()).hexdigest(),
 'test_filenames_sha256':hashlib.sha256('\n'.join(NAMES).encode()).hexdigest(),
 'probability_policy':'retain raw response; normalize sum within 0.02 of 1',
 'inference_workers':6,'primary_metric':'accuracy_all_attempts',
 'api_prices_per_million_input_tokens':{'luna':0.1,'jev':0.042},
 'estimated_total_api_cost_usd':0.29,'baseline_status':CONFIG['baseline_status']}
manifest_path=ROOT/'manifest.json'
if manifest_path.exists():assert json.loads(manifest_path.read_text())==MANIFEST
else:manifest_path.write_text(json.dumps(MANIFEST,indent=2))

def progress(stage,**kwargs):
 print(json.dumps({'stage':stage,**kwargs}),flush=True)
 (ROOT/'progress.json').write_text(json.dumps({'stage':stage,**kwargs,'updated_unix':time.time()},indent=2))

def get_image(name):
 label=next(x for x in LABELS if x.lower()==name.split('_')[0].lower())
 path=ROOT/'images'/name
 if not path.exists():
  url=f"https://huggingface.co/datasets/{CONFIG['dataset']}/resolve/{CONFIG['revision']}/data/{label}/{name}"
  for attempt in range(4):
   try:
    r=requests.get(url,timeout=90);r.raise_for_status();raw=r.content
    im=Image.open(io.BytesIO(raw));im.load();assert im.size==(600,600)
    path.write_bytes(raw);break
   except Exception:
    if attempt==3:raise
    time.sleep(2**attempt)
 raw=path.read_bytes();im=Image.open(io.BytesIO(raw));assert im.size==(600,600)
 return {'id':name,'image':name,'label':label,'sha256':hashlib.sha256(raw).hexdigest(),'dimensions':[600,600]}

cases=[]
with cf.ThreadPoolExecutor(max_workers=12) as pool:
 futures=[pool.submit(get_image,n) for n in NAMES]
 for f in cf.as_completed(futures):
  cases.append(f.result())
  if len(cases)%100==0:progress('download',completed=len(cases),total=2000)
cases=sorted(cases,key=lambda c:c['id'])
(ROOT/'cases.json').write_text(json.dumps(cases,indent=2))

for arm in ['luna','jev']:
 path=ROOT/(arm+'.jsonl')
 existing=[json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []
 done={x['id'] for x in existing};assert len(done)==len(existing) and done<=set(NAMES)
 for r in existing:assert r['input_sha256']==next(c['sha256'] for c in cases if c['id']==r['id'])
 pending=[c for c in cases if c['id'] not in done]
 with path.open('a') as out,cf.ThreadPoolExecutor(max_workers=6) as pool:
  futures=[pool.submit(pilot.predict,arm,c,KEYS,LABELS) for c in pending]
  for f in cf.as_completed(futures):
   r=f.result();out.write(json.dumps(r)+'\n');out.flush();existing.append(r)
   if len(existing)%100==0:progress('inference',arm=arm,completed=len(existing),total=2000,failures=sum(x['status']!='ok' for x in existing))
 summary=pilot.summarize(cases,existing,LABELS)
 summary['note']='Full frozen 2,000-image community test split; unchanged pilot instructions and label order.'
 summary['input_tokens']=sum(r.get('usage',r.get('response',{}).get('usage',{})).get('input_tokens',0) for r in existing)
 summary['estimated_api_cost_usd']=summary['input_tokens']*MANIFEST['api_prices_per_million_input_tokens'][arm]/1e6
 summary['request_attempts']=sum(r['attempts'] for r in existing)
 summary['latency_ms_p50']=float(np.percentile([r['latency_ms'] for r in existing],50))
 summary['latency_ms_p95']=float(np.percentile([r['latency_ms'] for r in existing],95))
 # Wilson interval for all-attempt accuracy.
 p=summary['accuracy_all_attempts'];n=len(cases);z=1.95996398454;den=1+z*z/n
 center=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 summary['accuracy_95pct_wilson']=[float(center-half),float(center+half)]
 (ROOT/(arm+'-summary.json')).write_text(json.dumps(summary,indent=2))
 progress('inference_complete',arm=arm,accuracy=p,valid=summary['valid'],estimated_api_cost_usd=summary['estimated_api_cost_usd'])

# Paired bootstrap of accuracy difference: same images, failures count incorrect.
results={arm:{r['id']:r for r in map(json.loads,(ROOT/(arm+'.jsonl')).read_text().splitlines())} for arm in ['luna','jev']}
def correct(arm,c):
 r=results[arm][c['id']];return int(r['status']=='ok' and r['label']==c['label'])
diff=np.array([correct('jev',c)-correct('luna',c) for c in cases],dtype=float)
rng=np.random.default_rng(20261016)
boot=np.array([diff[rng.integers(0,len(diff),len(diff))].mean() for _ in range(10000)])
comparison={'images':2000,'jev_minus_luna_accuracy':float(diff.mean()),'paired_bootstrap_95pct_ci':np.percentile(boot,[2.5,97.5]).tolist(),
 'both_correct':sum(correct('jev',c) and correct('luna',c) for c in cases),
 'jev_only_correct':int((diff==1).sum()),'luna_only_correct':int((diff==-1).sum()),
 'bootstrap_seed':20261016,'bootstrap_replicates':10000}
(ROOT/'comparison.json').write_text(json.dumps(comparison,indent=2))

bt.login(api_key=KEYS['BRAINTRUST_API_KEY'])
ds=bt.init_dataset(project=CONFIG['project'],name='aid-test-2000',description='Frozen 2,000-image community test split. Single-label 30-way aerial scene classification.',metadata=MANIFEST)
# Attachment uploads are resumable via durable references, and are checked for success.
refpath=ROOT/'attachment-references.jsonl'
refs={r['source_id']:r['reference'] for r in map(json.loads,refpath.read_text().splitlines())} if refpath.exists() else {}
def upload(c):
 for attempt in range(5):
  try:
   a=bt.Attachment(data=str(ROOT/'images'/c['image']),filename=c['image'],content_type='image/jpeg')
   status=a.upload()
   if status['upload_status']=='done':return c['id'],a.reference
  except Exception:
   pass
  time.sleep(min(30,2**attempt))
 return c['id'],None
pending=[c for c in cases if c['id'] not in refs]
with refpath.open('a') as out,cf.ThreadPoolExecutor(max_workers=6) as pool:
 for f in cf.as_completed([pool.submit(upload,c) for c in pending]):
  name,ref=f.result()
  if ref is None:
   progress('upload_retry_exhausted',image=name);continue
  refs[name]=ref;out.write(json.dumps({'source_id':name,'reference':ref})+'\n');out.flush()
  if len(refs)%100==0:progress('upload_images',completed=len(refs),total=2000)
assert len(refs)==2000, 'Some uploads failed; rerun to resume without repeating model calls'
ids={}
for c in cases:
 rid=str(uuid.uuid5(uuid.NAMESPACE_URL,CONFIG['revision']+c['id']));ids[c['id']]=rid
 ds.insert(id=rid,input={'image':refs[c['id']]},expected=c['label'],metadata={'source_id':c['id'],'sha256':c['sha256'],'split':'test','dimensions':[600,600]})
ds.flush();s=ds.summarize();links={'project':s.project_url,'dataset':s.dataset_url,'experiments':{}}
(ROOT/'braintrust-links.json').write_text(json.dumps(links,indent=2))
for arm in ['luna','jev']:
 ex=bt.init(project=CONFIG['project'],experiment='test-2000__'+arm,dataset=ds,update=True,description='Full frozen 2,000-image AID test run; identical original 600x600 images.',metadata={'protocol':MANIFEST,'aggregate_metrics':json.loads((ROOT/(arm+'-summary.json')).read_text()),'paired_comparison':comparison})
 for c in cases:
  r=results[arm][c['id']];ok=r['status']=='ok'
  ex.log(id=ids[c['id']],dataset_record_id=ids[c['id']],input={'image':refs[c['id']]},expected=c['label'],output={'label':r.get('label'),'probabilities':r.get('probabilities'),'status':r['status']},scores={'accuracy':correct(arm,c),'valid_output':int(ok)},metrics={'latency_ms':r['latency_ms'],'input_tokens':r.get('usage',{}).get('input_tokens',0)},metadata={k:v for k,v in r.items() if k not in ('response','probabilities')})
 ex.flush();links['experiments'][arm]=ex.summarize(summarize_scores=False).experiment_url
 (ROOT/'braintrust-links.json').write_text(json.dumps(links,indent=2));progress('published_experiment',arm=arm,url=links['experiments'][arm])

# Count every remote attachment; download a deterministic sample and hash-check bytes.
fresh=bt.init_dataset(project=CONFIG['project'],name='aid-test-2000');remote=list(fresh.fetch());assert len(remote)==2000
for row in remote:
 a=row['input']['image'];ref=a.reference if hasattr(a,'reference') else a
 assert ref['type']=='braintrust_attachment' and ref['content_type']=='image/jpeg'
sample_ids={ids[c['id']] for c in cases[::100]}
for row in remote:
 if row['id'] in sample_ids:
  a=row['input']['image'];a=bt.ReadonlyAttachment(a) if isinstance(a,dict) else a
  assert hashlib.sha256(a.data).hexdigest()==row['metadata']['sha256']
links.update(remote_image_rows=2000,verified_download_samples=len(sample_ids))
(ROOT/'braintrust-links.json').write_text(json.dumps(links,indent=2));progress('complete',**links)
