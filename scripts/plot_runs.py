"""Plot measured pruning-location accuracies and token trajectories from evaluation JSON."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(description=__doc__);p.add_argument('runs',type=Path);p.add_argument('--output',type=Path,default=Path('runs/plots'));a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
rows=[json.loads(f.read_text()) for f in sorted(a.runs.glob('*.json'))]
rows=[r for r in rows if isinstance(r,dict) and 'config' in r and 'mean_tokens_entering_blocks' in r]
if not rows:raise ValueError('No evaluation JSON files')
plt.figure(figsize=(8,4))
for r in rows:
 c=r['config'];plt.plot(range(12),r['mean_tokens_entering_blocks'],label=f"{c['model']} {c['kta_drop']}/{c['ats_drop']} KTA@{c['kta_block']}")
plt.xlabel('Encoder block (tokens entering)');plt.ylabel('Mean token count');plt.legend(fontsize=6);plt.tight_layout();plt.savefig(a.output/'token_trajectories.png',dpi=160);plt.close()
plt.figure(figsize=(6,4))
for method in ['kta','ats']:
 points=[]
 for r in rows:
  c=r['config']
  if method=='kta' and not c['blocks']:points.append((c['kta_block'],r['accuracy_percent']))
  if method=='ats' and len(c['blocks'])==1 and c['kta_drop']==0:points.append((c['blocks'][0],r['accuracy_percent']))
 if points:
  points.sort();plt.plot(*zip(*points),'o-',label=method.upper())
plt.xlabel('Pruning after block (-1 = before encoder)');plt.ylabel('Measured top-1 accuracy (%)')
if plt.gca().lines:plt.legend()
plt.tight_layout();plt.savefig(a.output/'pruning_locations.png',dpi=160);plt.close()
