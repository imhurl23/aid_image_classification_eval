"""Publish a shared held-out comparison; reuse cached predictions and images."""
import json,sys,uuid
from pathlib import Path
import braintrust as bt
ROOT=Path(__file__).resolve().parent
AID=ROOT.parents[1]
sys.path.insert(0,str(AID))
from eval import summarize

cases=json.loads((ROOT/'overlap-cases.json').read_text());names={c['id'] for c in cases}
labels=json.loads((AID/'config.json').read_text())['labels']
provenance=json.loads((ROOT/'provenance.json').read_text())
provenance['arena_repository_revision']='b640d13cd645e72d6017223ec584a3e30095fdf8'
provenance['preprocessing_source_revision']='8f7ebfa6d02499e276b37a618b76670595f0aae4'
provenance['preprocessing']='Resize(256), CenterCrop(224), ToTensor; no Normalize'
(ROOT/'provenance.json').write_text(json.dumps(provenance,indent=2))
results={}
for arm in ['luna','jev','resnet50']:
 path=ROOT/'resnet50.jsonl' if arm=='resnet50' else AID/'test-2000'/(arm+'.jsonl')
 rows={r['id']:r for r in map(json.loads,path.read_text().splitlines()) if r['id'] in names}
 assert set(rows)==names
 for c in cases:assert rows[c['id']]['input_sha256']==c['sha256']
 results[arm]=rows
summaries={arm:summarize(cases,list(rows.values()),labels) for arm,rows in results.items()}
for summary in summaries.values():summary['note']='382 images in both our frozen test and AiTLAS published test; same subset for all three models; no new paid API calls.'
(ROOT/'comparison.json').write_text(json.dumps(summaries,indent=2))
for line in Path('.env').read_text().splitlines():
 if line.startswith('BRAINTRUST_API_KEY='):key=line.split('=',1)[1].strip().strip(chr(34)).strip(chr(39))
bt.login(api_key=key)
project='aid-single-label-comparison'
source=bt.init_dataset(project=project,name='aid-test-2000')
images={}
for row in source.fetch():
 name=row['metadata']['source_id']
 if name in names:
  a=row['input']['image'];images[name]=a.reference if hasattr(a,'reference') else a
assert set(images)==names
name='aid-aitlas-heldout-overlap-382'
ds=bt.init_dataset(project=project,name=name,description='Shared held-out intersection: 382 images from both community and AiTLAS test splits. All 30 classes. Same images for specialist, Luna, and Jev.',metadata=provenance)
ids={}
for c in cases:
 rid=str(uuid.uuid5(uuid.NAMESPACE_URL,'aid-aitlas-overlap:'+c['id']));ids[c['id']]=rid
 ds.insert(id=rid,input={'image':images[c['id']]},expected=c['label'],metadata={'source_id':c['id'],'sha256':c['sha256'],'our_split':'test','aitlas_split':'test','dimensions':[600,600]})
ds.flush();s=ds.summarize();links={'project':s.project_url,'dataset':s.dataset_url,'experiments':{}}
for arm in ['resnet50','luna','jev']:
 ex=bt.init(project=project,experiment='aitlas-heldout-382__'+arm,dataset=ds,update=True,description='Exactly the same 382 held-out images for all three arms; do not compare subset accuracy against full 2,000-image run.',metadata={'provenance':provenance,'aggregate_metrics':summaries[arm],'cached_api_predictions':arm!='resnet50','new_paid_api_calls':0})
 for c in cases:
  r=results[arm][c['id']];ok=r['status']=='ok'
  ex.log(id=ids[c['id']],dataset_record_id=ids[c['id']],input={'image':images[c['id']]},expected=c['label'],output={'label':r.get('label'),'probabilities':r.get('probabilities'),'status':r['status']},scores={'accuracy':int(ok and r['label']==c['label']),'valid_output':int(ok)},metadata={'source_id':c['id'],'input_sha256':c['sha256'],'model':r.get('served_model',r.get('model')),'heldout':True,'checkpoint_sha256':provenance['checkpoint_sha256'] if arm=='resnet50' else None})
 ex.flush();links['experiments'][arm]=ex.summarize(summarize_scores=False).experiment_url
 print(arm,summaries[arm]['accuracy_all_attempts'],links['experiments'][arm],flush=True)
(ROOT/'braintrust-links.json').write_text(json.dumps(links,indent=2))
for arm in results:
 ex=bt.init(project=project,experiment='aitlas-heldout-382__'+arm,open=True);rows=list(ex.fetch());assert len(rows)==382
 assert sum(r['scores']['accuracy'] for r in rows)==sum(results[arm][c['id']]['label']==c['label'] for c in cases)
print('Verified all three remote experiment row counts and correct-label totals.',flush=True)
