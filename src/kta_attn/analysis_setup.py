"""Initialize the explicit shared context required by the original revision analyses."""
import argparse
from pathlib import Path
import random
import json
import hashlib
import numpy as np
import torch
from .model import build_model, pearson_corr
from .quantization import quantize
from .preprocessing import resize_nearest_neighbor, bgr_to_rgb, transform_image


def setup():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--max-images',type=int,default=1000)
    p.add_argument('--keep-ratio',type=float,default=.5)
    p.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--model',choices=['tiny','small','base'],default='base')
    p.add_argument('--seed',type=int,default=0)
    args=p.parse_args()
    if not args.data.is_dir() or args.max_images <= 0: p.error('Supply a valid data directory and positive max-images')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    model=build_model(f'vit_{args.model}_patch16_224',keep_ratio=args.keep_ratio,ats_ratio=0,blocks=[],pretrained=args.checkpoint is None).to(args.device)
    if args.checkpoint: model.load_state_dict(torch.load(args.checkpoint,map_location=args.device,weights_only=True))
    model=quantize(model)
    args.output.mkdir(parents=True,exist_ok=True)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config["checkpoint_sha256"]=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint else None
    (args.output/"run_config.json").write_text(json.dumps(config,indent=2)+"\n")
    def main_processing(image):
        image=bgr_to_rgb(resize_nearest_neighbor(image,(224,224))).copy()
        return transform_image(image,args.device), np.ones(197)
    return dict(args=args,device=args.device,img_folder=str(args.data),drop_rate=args.keep_ratio,
                quantized_model=model,pearson_corr=pearson_corr,main_processing=main_processing,
                resize_nearest_neighbor=resize_nearest_neighbor,bgr_to_rgb=bgr_to_rgb,
                transform_image=transform_image)
