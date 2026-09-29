"""Evaluate a class-folder ImageNet subset with explicit labels and configuration."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import cv2
import numpy as np
import torch
from kta_attn.model import build_model
from kta_attn.preprocessing import resize_nearest_neighbor, bgr_to_rgb, transform_image


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True, help='Root with one directory per class')
    p.add_argument('--labels', type=Path, required=True, help='ImageNet index JSON: index -> [synset, name]')
    p.add_argument('--model', choices=['tiny', 'small', 'base'], default='base')
    p.add_argument('--checkpoint', type=Path, help='Local timm state_dict; avoids downloading weights')
    p.add_argument('--kta-block', type=int, default=-1, help='-1: before encoder; 0..11: after that block, requires empty ATS blocks')
    p.add_argument('--kta-drop', type=float, default=.15)
    p.add_argument('--ats-drop', type=float, default=.05)
    p.add_argument('--blocks', type=int, nargs='*', default=[3,4,5,6,7,8])
    p.add_argument('--tau', choices=['notebook','tau-b'], default='notebook')
    p.add_argument('--precision', choices=['fp32','fx16'], default='fx16')
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--max-images', type=int, default=None)
    p.add_argument('--manifest', type=Path, help='Optional ordered image paths relative to --data')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output', type=Path, default=Path('runs/evaluation.json'))
    return p


def main():
    args = parser().parse_args()
    if args.max_images is not None and args.max_images <= 0:
        raise ValueError('--max-images must be positive')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    model = build_model(f'vit_{args.model}_patch16_224', 1-args.kta_drop,
                        args.ats_drop, args.blocks, args.tau, pretrained=args.checkpoint is None).to(args.device)
    if args.kta_block not in range(-1,12): raise ValueError("KTA block must be -1..11")
    if args.kta_block != -1 and args.blocks: raise ValueError("Delayed KTA requires --blocks with no values")
    model.kta_block = args.kta_block
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=args.device, weights_only=True), strict=True)
    pretrained_cfg = model.pretrained_cfg
    token_counts = []
    current_counts = []
    handles = [block.register_forward_pre_hook(lambda module, inputs: current_counts.append(inputs[0].shape[1])) for block in model.blocks]
    if args.precision == 'fx16':
        from kta_attn.quantization import quantize
        model = quantize(model)
    labels = json.loads(args.labels.read_text())
    lookup = {name: int(index) for index, names in labels.items() for name in names}
    files = ([args.data / s.strip() for s in args.manifest.read_text().splitlines() if s.strip()]
             if args.manifest else sorted(p for p in args.data.glob('*/*') if p.suffix.lower() in {'.jpg','.jpeg','.png'}))
    files = files[:args.max_images]
    if not files: raise ValueError('No images found')
    for path in files:
        if path.parent.name not in lookup: raise ValueError(f'Unknown class directory: {path.parent.name}')
    correct = 0
    with torch.inference_mode():
        for count, path in enumerate(files, 1):
            image = cv2.imread(str(path))
            if image is None: raise ValueError(f'Cannot decode {path}')
            image = bgr_to_rgb(resize_nearest_neighbor(image, (224,224))).copy()
            current_counts.clear()
            logits = model(transform_image(image, args.device), mask=True)
            token_counts.append(list(current_counts))
            correct += int(logits.argmax(1).item() == lookup[path.parent.name])
            if count % 100 == 0: print(f'{count}/{len(files)}: {100*correct/count:.2f}%')
    manifest = '\n'.join(str(p.relative_to(args.data)) for p in files)+'\n'
    result = dict(config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  pretrained_cfg=pretrained_cfg,
                  checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint else None,
                  mean_tokens_entering_blocks=np.mean(token_counts,axis=0).tolist(),
                  correct=correct, num_images=len(files), accuracy_percent=100*correct/len(files),
                  manifest_sha256=hashlib.sha256(manifest.encode()).hexdigest(),
                  versions={k:importlib.metadata.version(k) for k in ['torch','torchvision','timm','numpy','qtorch']})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    args.output.with_suffix('.manifest.txt').write_text(manifest)
    print(json.dumps(result,indent=2))

if __name__ == '__main__': main()
