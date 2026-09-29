"""Run Table IV configurations or Fig. 6 pruning-location experiments."""
import argparse
import csv
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--experiment',choices=['table-iv','locations'],default='table-iv')
p.add_argument('--output',type=Path,default=Path('runs/sweep'))
a,extra=p.parse_known_args()
if a.experiment=='table-iv':
    rows=list(csv.DictReader((root/'results/paper_table_iv.csv').open()))
    configs=[(f"{r['model']}_{r['kta_drop_percent']}_{r['ats_drop_percent']}",
         ['--model',r['model'],'--kta-drop',str(int(r['kta_drop_percent'])/100),
          '--ats-drop',str(int(r['ats_drop_percent'])/100)]) for r in rows]
else:
    configs=[]
    for block in range(-1,12):
        configs.append((f'kta_block_{block}',['--kta-drop','.5','--ats-drop','0','--kta-block',str(block),'--blocks']))
        if block>=0:configs.append((f'ats_block_{block}',['--kta-drop','0','--ats-drop','.5','--blocks',str(block)]))
for name,opts in configs:
    subprocess.run([sys.executable,str(root/'scripts/evaluate.py'),*extra,*opts,'--output',str(a.output/f'{name}.json')],check=True)
