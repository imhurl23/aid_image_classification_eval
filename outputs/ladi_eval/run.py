"""LADI validation/test runner. API keys come from environment or explicit --env-file.

Run --help for commands. Default is a five-image validation pilot.
"""
import argparse
import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time
import uuid

import requests
from PIL import Image, ImageOps
from score import CONFIG, LABELS, score

# Initial operational rubric; freeze after validation review, before test inference.
QUESTIONS = dict(zip(LABELS, [
    'Is any bridge visible?', 'Is any building visible?',
    'Does any building show FEMA affected-level damage or worse (affected, minor, major, or destroyed)?',
    'Does any building show FEMA minor-level damage or worse (minor, major, or destroyed), beyond superficial affected-level damage?',
    'Is disaster-related debris visible?', 'Is flooding visible?',
    'Are any structures affected by flooding?', 'Is any road visible?',
    'Is any road visibly damaged?', 'Are any trees visible?',
    'Are any trees visibly damaged?', 'Is any body or accumulation of water visible?'
]))
PREFIX = 'Assess only visible evidence in this aerial image. Each question is independent. '
PROMPTS = {k: PREFIX+v for k,v in QUESTIONS.items()}
RUBRIC_HASH = hashlib.sha256(json.dumps(PROMPTS, sort_keys=True).encode()).hexdigest()


def prepare(image, *, jev_limits=False):
    image = ImageOps.exif_transpose(image).convert('RGB')
    if jev_limits:
        image.thumbnail((1568,1568), Image.Resampling.LANCZOS)
    if jev_limits and min(image.size) < 280:
        image = ImageOps.pad(image, (max(280,image.width), max(280,image.height)), color='black')
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    return buf.getvalue()


def request_for(arm, png):
    b64 = base64.b64encode(png).decode('ascii')
    if arm == 'decisions_luna':
        return 'https://api.openai.com/v1/decisions', 'OPENAI_API_KEY', {}, {
            'model':'gpt-6-luna',
            'input':[{'role':'user','content':[{'type':'input_image','image_url':'data:image/png;base64,'+b64}]}],
            'questions':[{'type':'predicate','name':k,'instructions':v} for k,v in PROMPTS.items()]}
    if arm != 'jev_multimodal':
        raise ValueError(arm)
    ref = str(uuid.uuid4())
    return 'https://api.typesafe.ai/v1/systemone', 'TYPESAFE_API_KEY', {'TypeSafe-Danger-Zone-Images':'true'}, {
        'model':'jev-1.14.0', 'state':{'image':ref},
        'assets':{ref:{'type':'image','format':'png','data_base64':b64}},
        'questions':{k:{'type':'noul','instructions':v} for k,v in PROMPTS.items()}}


def parse(arm, payload):
    answers = payload['answers']
    if arm == 'decisions_luna':
        if len(answers) != len(LABELS) or len({a['name'] for a in answers}) != len(LABELS):
            raise ValueError('Wrong answer count or duplicate answer names')
        answers = {a['name']:a for a in answers}
    if set(answers) != set(LABELS):
        raise ValueError('Wrong answer names')
    expected_type, field = ('predicate','probability') if arm == 'decisions_luna' else ('noul','noul')
    probs = {}
    for k,a in answers.items():
        if a['type'] != expected_type:
            raise ValueError('Refusal or unexpected answer type')
        p = a[field]
        if type(p) not in (float,int) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError('Invalid probability')
        probs[k] = p
    return probs


def api_predict(arm,png,keys):
    url,keyname,headers,body = request_for(arm,png)
    key = keys[keyname]
    headers['Authorization'] = 'Bearer '+key
    start = time.perf_counter()
    result = {'status':'api_error','requested_model':body['model']}
    for attempt in range(3):
        try:
            response = requests.post(url,headers=headers,json=body,timeout=90)
            result['http_status'] = response.status_code
            if response.status_code in (429,500,502,503,504,529) and attempt < 2:
                time.sleep(2**attempt)
                continue
            payload = response.json()
            # Store response only; never credentials or the base64 request.
            result['response'] = payload
            if response.ok:
                result['status'] = 'invalid_output'
                result['probabilities'] = parse(arm,payload)
                result['served_model'] = payload.get('model')
                result['usage'] = payload.get('usage')
                result['status'] = 'ok'
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            result['error_type'] = type(exc).__name__
        break
    result['attempts'] = attempt+1
    result['latency_ms'] = (time.perf_counter()-start)*1000
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='Downloaded Hugging Face snapshot directory containing data/<split>/*.arrow')
    parser.add_argument('--split',choices=['validation','test'],default='validation')
    parser.add_argument('--limit',type=int,default=5)
    parser.add_argument('--image-condition', choices=['native','matched'], default='native',
                        help='native: supplied resolution, Jev resized as required; matched: every arm receives Jev-sized pixels')
    parser.add_argument('--arms',nargs='+',choices=['decisions_luna','jev_multimodal','mitll_small_reference','mitll_large_reference'],default=['decisions_luna','jev_multimodal'])
    parser.add_argument('--env-file',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.limit < 1: parser.error('--limit must be positive')
    args.out.mkdir(parents=True,exist_ok=True)
    if any(args.out.iterdir()): parser.error('--out must be empty; existing results are never overwritten')
    keys=dict(os.environ)
    if args.env_file:
        for line in args.env_file.read_text().splitlines():
            if line.strip() and not line.lstrip().startswith('#') and '=' in line:
                name, value = line.split('=', 1)
                keys[name.strip()] = value.strip().strip(chr(34)).strip(chr(39))
    for arm,key in [('decisions_luna','OPENAI_API_KEY'),('jev_multimodal','TYPESAFE_API_KEY')]:
        if arm in args.arms and not keys.get(key): parser.error('Missing '+key)
    from datasets import Dataset
    paths=sorted((Path(args.source)/'data'/args.split).glob('*.arrow'))
    if not paths: parser.error('No Arrow shards found')
    # Only contiguous leading shards preserve global row indices in a partial download.
    for i,path in enumerate(paths):
        if path.name != f'data-{i:05d}-of-00040.arrow': parser.error('Source shards must be contiguous from zero')
    manifest={'dataset':CONFIG['dataset'],'dataset_revision':CONFIG['dataset_revision'],
              'split':args.split,'requested_images':args.limit,'arms':args.arms,
              'rubric':PROMPTS,'rubric_hash':RUBRIC_HASH,'image_format':'PNG',
              'image_condition':args.image_condition, 'jev_max_dimension':1568,
              'created_unix':time.time(),'checkpoint_revisions':{},
              'source_shards':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    cases=[]
    images=args.out/'images'; images.mkdir()
    for path in paths:
        for row in Dataset.from_file(str(path)):
            i=len(cases); png=prepare(row['image']); image_path=images/f'{i:05d}.png'; image_path.write_bytes(png)
            cases.append({'id':f'{args.split}:{i}', 'labels':{k:row[k] for k in LABELS},
                          'image_sha256':hashlib.sha256(png).hexdigest(),'image_path':str(image_path)})
            if len(cases)==args.limit: break
        if len(cases)==args.limit: break
    if len(cases)<args.limit: raise ValueError('Not enough source rows; download more shards')
    truth={row['id']:row for row in cases}
    (args.out/'truth.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in cases))
    for arm in args.arms:
        pipeline=None
        if arm.startswith('mitll_'):
            import torch
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            from huggingface_hub import HfApi
            repo=next(a['model'] for a in CONFIG['primary_arms'] if a['name']==arm)
            revision=HfApi().model_info(repo).sha
            processor=AutoImageProcessor.from_pretrained(repo,revision=revision)
            model=AutoModelForImageClassification.from_pretrained(repo,revision=revision).eval()
            if set(model.config.id2label.values()) != set(LABELS): raise ValueError('Checkpoint label mismatch')
            manifest['checkpoint_revisions'][arm]=revision
            (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
        predictions={}
        with (args.out/(arm+'.jsonl')).open('w') as output:
            for case in cases:
                source_png=Path(case['image_path']).read_bytes()
                supplied=Image.open(io.BytesIO(source_png))
                needs_resize=args.image_condition=='matched' or arm=='jev_multimodal'
                png=prepare(supplied, jev_limits=needs_resize)
                sent=Image.open(io.BytesIO(png))
                if arm.startswith('mitll_'):
                    start=time.perf_counter()
                    with torch.inference_mode():
                        logits=model(**processor(images=Image.open(io.BytesIO(png)),return_tensors='pt')).logits[0]
                        probs=logits.sigmoid().tolist()
                    result={'status':'ok','probabilities':{model.config.id2label[i]:p for i,p in enumerate(probs)},
                            'latency_ms':(time.perf_counter()-start)*1000,'served_model':repo,'revision':revision}
                else: result=api_predict(arm,png,keys)
                result.update(id=case['id'],source_image_sha256=case['image_sha256'],
                              input_image_sha256=hashlib.sha256(png).hexdigest(),
                              supplied_dimensions=list(supplied.size), input_dimensions=list(sent.size),
                              image_condition=args.image_condition, rubric_hash=RUBRIC_HASH)
                predictions[case['id']]=result
                output.write(json.dumps(result,allow_nan=False)+'\n'); output.flush()
                print(arm,case['id'],result['status'],flush=True)
        (args.out/(arm+'-scores.json')).write_text(json.dumps(score(truth,predictions),indent=2))
    print('Completed',args.out)

if __name__=='__main__': main()
