"""Run the verified AiTLAS checkpoint on the predeclared shared held-out images.
Uses only local inference; does not call paid APIs.
"""
import hashlib,json,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torchvision import models,transforms

ROOT=Path(__file__).resolve().parent
WORKSPACE=ROOT.parents[3]
CHECKPOINT=WORKSPACE/'work/aid-baseline/aid_pretrained_imagenet1K_resnet50.pth.tar'
IMAGE_DIR=ROOT.parents[1]/'test-2000/images'
config=json.loads((ROOT/'provenance.json').read_text())
assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()==config['checkpoint_sha256']
labels=json.loads((ROOT.parents[1]/'config.json').read_text())['labels']
cases=json.loads((ROOT/'overlap-cases.json').read_text())
checkpoint=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
assert checkpoint['id']=='aid_resnet50_pretrained'
model=models.resnet50(weights=None,num_classes=30)
model.load_state_dict(checkpoint['state_dict'],strict=True)
torch.set_num_threads(4)
device='mps' if torch.backends.mps.is_available() else 'cpu'
model=model.eval().to(device)
# Published preprocessing from before this checkpoint was trained, no Normalize.
transform=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor()])
results=[];start=time.perf_counter()
with (ROOT/'resnet50.jsonl').open('w') as out:
 for offset in range(0,len(cases),16):
  chunk=cases[offset:offset+16];inputs=[]
  for c in chunk:
   path=IMAGE_DIR/c['image'];assert hashlib.sha256(path.read_bytes()).hexdigest()==c['sha256']
   with Image.open(path) as im:inputs.append(transform(im.convert('RGB')))
  with torch.inference_mode():
   probs=model(torch.stack(inputs).to(device)).softmax(-1).cpu().numpy()
  assert probs.shape==(len(chunk),30) and np.isfinite(probs).all()
  assert np.allclose(probs.sum(-1),1,atol=1e-5)
  for c,p in zip(chunk,probs):
   r={'id':c['id'],'status':'ok','label':labels[int(p.argmax())],'probabilities':{x:float(v) for x,v in zip(labels,p)},'input_sha256':c['sha256'],'model':'aitlas-resnet50-aid','checkpoint_sha256':config['checkpoint_sha256'],'heldout_split':'published AiTLAS test; intersection with our frozen community test','device':device}
   results.append(r);out.write(json.dumps(r)+'\n')
  out.flush();print('predicted',len(results),'/',len(cases),flush=True)
summary={'images':len(cases),'correct':sum(c['label']==r['label'] for c,r in zip(cases,results)),'device':device,'elapsed_seconds':time.perf_counter()-start,'checkpoint_id':checkpoint['id'],'checkpoint_epoch':checkpoint['epoch'],'checkpoint_loss':checkpoint['loss'],'preprocessing':'PIL Resize(256), CenterCrop(224), ToTensor; no Normalize; historical published transform','preprocessing_source_revision':'8f7ebfa6d02499e276b37a618b76670595f0aae4'}
summary['accuracy']=summary['correct']/len(cases)
(ROOT/'resnet50-inference-summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
