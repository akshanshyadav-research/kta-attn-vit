"""Regenerate lightweight result plots without ImageNet, checkpoints, or a GPU."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def main():
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=root/'results')
    p.add_argument('--output',type=Path,default=root/'figures/generated')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    def save(name):
        plt.tight_layout();plt.savefig(a.output/f'{name}.png',dpi=160);plt.close()
    table=pd.read_csv(a.results/'paper_table_iv.csv')
    for model, frame in table.groupby('model'):
        plt.figure(figsize=(6,4))
        plt.scatter(frame.gflops,frame.accuracy_percent)
        for r in frame.itertuples():
            plt.annotate(f'{r.kta_drop_percent}/{r.ats_drop_percent}',(r.gflops,r.accuracy_percent),fontsize=7,xytext=(3,3),textcoords='offset points')
        plt.xlabel('GFLOPs (reported)');plt.ylabel('Top-1 accuracy (%)')
        plt.title(f'ViT-{model}: paper Table IV');plt.grid(alpha=.2)
        save(f'table_iv_{model}')
    ablation=pd.read_csv(a.results/'paper_table_iii.csv')
    for model, frame in ablation.groupby('model'):
        plt.figure(figsize=(5,3))
        for method, group in frame.groupby('method'):
            plt.plot(group.savings_percent,group.accuracy_percent,'o-',label=method)
        plt.xlabel('Reported computational savings (%)');plt.ylabel('Top-1 accuracy (%)')
        plt.title(f'ViT-{model}: paper Table III');plt.legend();save(f'table_iii_{model}')
    t=pd.read_csv(a.results/'translation_summary.csv')
    for column,label in [('mean_center_distance','Mean normalized center distance'),('center_patch_fraction','Central patch fraction'),('mean_translation_iou','Translation-compensated IoU'),('mean_recall','Translation-compensated recall')]:
        plt.figure(figsize=(5,3));plt.plot(t.translation_pixels,t[column],'o-')
        plt.xlabel('Horizontal translation (pixels)');plt.ylabel(label);plt.grid(alpha=.2)
        save(column)
    t=pd.read_csv(a.results/'center_bias_summary.csv').iloc[0]
    plt.figure(figsize=(5,3));plt.bar(['KTA','Random','Attention'],[t[f'{m}_mean_center_distance'] for m in ['kta','random','attention']])
    plt.ylabel('Mean normalized center distance');save('center_bias')

if __name__=='__main__':main()
