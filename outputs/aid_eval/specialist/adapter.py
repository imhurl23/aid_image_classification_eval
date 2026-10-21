"""Load the recovered AID specialist; does not claim a held-out evaluation.

Example from workspace root:
work/aid-baseline/venv/bin/python outputs/aid_eval/specialist/adapter.py \
 --checkpoint work/aid-baseline/checkpoint/neuralNet/models/PiattiVL_v0.69.pth \
 --model-source work/aid-baseline/aerial-vision/NeuralNets/model.py \
 --cases outputs/aid_eval/cases.json --images outputs/aid_eval/images \
 --output outputs/aid_eval/specialist/validation-smoke.jsonl

Dependencies: torch torchvision pillow. Source has been inspected; only model.py
is imported. Checkpoint is loaded with weights_only=True, strict state dict matching.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import torch
from torchvision import transforms
from PIL import Image

CHECKPOINT_SHA256 = 'e083f80edbe96661995d4b0d01ccd94085e50788fa3040124c53d2b66a535054'
MODEL_SOURCE_SHA256 = 'd3cb56af8380110f48f9c6225edba11abfa35631b1d9f1b236de975d55427578'

class AIDSpecialist:
    def __init__(self, checkpoint, model_source):
        checkpoint,model_source=Path(checkpoint),Path(model_source)
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=CHECKPOINT_SHA256:
            raise ValueError('Checkpoint differs from audited pinned artifact')
        if hashlib.sha256(model_source.read_bytes()).hexdigest()!=MODEL_SOURCE_SHA256:
            raise ValueError('Model source differs from inspected pinned source')
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
        self.class_names=payload['class_names']
        if len(self.class_names)!=30 or len(set(self.class_names))!=30:
            raise ValueError('Expected exactly 30 distinct classes')
        spec=importlib.util.spec_from_file_location('aid_piatti_model',model_source)
        source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
        self.model=source.PiattiCNN(num_classes=payload['num_classes'])
        self.model.load_state_dict(payload['model_state_dict'],strict=True)
        self.model.eval()
        # Pinned NeuralNets/test.py evaluation recipe. The demo uses a different resize;
        # do not select a recipe by inspecting evaluation performance.
        self.transform=transforms.Compose([
            transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    def predict(self,path):
        started=time.perf_counter()
        with Image.open(path) as image:
            tensor=self.transform(image.convert('RGB')).unsqueeze(0)
        with torch.inference_mode():
            probabilities=self.model(tensor).softmax(-1)[0].tolist()
        return {'label':self.class_names[max(range(30),key=lambda i:probabilities[i])],
                'probabilities':dict(zip(self.class_names,probabilities)),
                'latency_ms':(time.perf_counter()-started)*1000,
                'status':'ok','training_overlap':'unknown',
                'eligible_for_heldout_ranking':False}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--model-source',required=True,type=Path)
    p.add_argument('--cases',required=True,type=Path)
    p.add_argument('--images',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    if args.output.exists():p.error('Output exists; choose a new path')
    torch.set_num_threads(4)
    model=AIDSpecialist(args.checkpoint,args.model_source)
    cases=json.loads(args.cases.read_text())
    with args.output.open('w') as out:
        for case in cases:
            path=args.images/case['image']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=case['sha256']:
                raise ValueError('Image hash mismatch')
            result=model.predict(path)
            if set(result['probabilities'])!=set(model.class_names):
                raise ValueError('Class mapping mismatch')
            if abs(sum(result['probabilities'].values())-1)>1e-5:
                raise ValueError('Probability sum mismatch')
            out.write(json.dumps({'id':case['id'],**result})+'\n');out.flush()
    print(f'Loaded audited checkpoint strictly; {len(cases)} valid single-label predictions. Training overlap unknown.')
