"""Render gradient×token attribution, KTA scores, and progressive token masks for an image."""
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from kta_attn.model import build_model
from kta_attn.ranking import kendall_scores
from kta_attn.preprocessing import resize_nearest_neighbor,bgr_to_rgb,transform_image


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('image',type=Path);p.add_argument('--output',type=Path,default=Path('runs/visualization'))
    p.add_argument('--model',choices=['tiny','small','base'],default='base')
    p.add_argument('--kta-drop',type=float,default=.2);p.add_argument('--ats-drop',type=float,default=.15)
    p.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--checkpoint',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    raw=cv2.imread(str(a.image))
    if raw is None:raise ValueError('Cannot decode input image')
    rgb=bgr_to_rgb(resize_nearest_neighbor(raw,(224,224))).copy()
    tensor=transform_image(rgb,a.device)
    model=build_model(f'vit_{a.model}_patch16_224',1-a.kta_drop,a.ats_drop,pretrained=a.checkpoint is None).to(a.device)
    if a.checkpoint:model.load_state_dict(torch.load(a.checkpoint,map_location=a.device,weights_only=True))
    # Gradient reference uses all tokens, with the same weights and preprocessing.
    captured={}
    def capture(module,inputs,output):
        output.retain_grad();captured['tokens']=output
    hook=model.patch_embed.register_forward_hook(capture)
    # Disable ATS temporarily by using its ordinary transformer operations.
    ats_forwards={}
    from kta_attn.model import CustomBlock
    for b in model.blocks:
        if isinstance(b,CustomBlock):
            ats_forwards[b]=b.forward
            b.forward=lambda x,apply_patch_drop=True,b=b: ats_forwards[b](x,apply_patch_drop=False)
    logits=model(tensor)
    logits[0,logits.argmax(1).item()].backward()
    gradient=(captured['tokens'].grad*captured['tokens']).sum(-1).abs()[0].detach().cpu().numpy()
    hook.remove()
    for b,f in ats_forwards.items():b.forward=f
    model.zero_grad(set_to_none=True)
    states=[];indices=torch.arange(196,device=a.device)
    def stage0(module,inputs,output):
        nonlocal indices
        x=inputs[0];scores=kendall_scores(x[:,0],x[:,1:])[0]
        threshold=scores.kthvalue(min(int(196*(1-a.kta_drop))+1,196)).values
        indices=indices[scores<=threshold]
        captured['kta']=(-scores).cpu().numpy();states.append(('KTA',indices.cpu().numpy().copy()))
    hooks=[model.patch_drop.register_forward_hook(stage0)]
    def ats(module,inputs,output):
        nonlocal indices
        indices=indices[output[1][0,1:]]
        states.append((f'ATS {len(states)}',indices.cpu().numpy().copy()))
    for b in model.blocks:
        if isinstance(b,CustomBlock):hooks.append(b.attn.register_forward_hook(ats))
    with torch.inference_mode():model(tensor,mask=True)
    for hook in hooks:hook.remove()
    fig,axes=plt.subplots(1,3,figsize=(10,3))
    for ax in axes:ax.imshow(rgb);ax.axis('off')
    axes[0].set_title('Original')
    for ax,title,values in [(axes[1],'Gradient × token',gradient),(axes[2],'KTA importance (−tau)',captured['kta'])]:
        ax.imshow(cv2.resize(values.reshape(14,14),(224,224),interpolation=cv2.INTER_NEAREST),alpha=.5,cmap='viridis');ax.set_title(title)
    fig.tight_layout();fig.savefig(a.output/'attribution.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,len(states),figsize=(3*len(states),3),squeeze=False)
    for ax,(name,idx) in zip(axes[0],states):
        mask=np.zeros(196);mask[idx]=1;mask=np.repeat(np.repeat(mask.reshape(14,14),16,0),16,1)
        ax.imshow((rgb*mask[...,None]).astype(np.uint8));ax.set_title(f'{name}: {len(idx)} patches');ax.axis('off')
    fig.tight_layout();fig.savefig(a.output/'progressive_pruning.png',dpi=160);plt.close(fig)
    np.savez_compressed(a.output/'attribution_values.npz',gradient=gradient,kta_importance=captured['kta'])

if __name__=='__main__':main()
