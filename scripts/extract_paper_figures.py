"""Extract the software figures (1, 2, 3, 6, 7); requires Poppler's pdftoppm executable."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

# PDF points, top-left coordinates; includes figure artwork, excludes captions.
CROPS = {1:(1,311,195,565,314), 2:(2,311,48,565,193),
         3:(3,48,48,565,174), 6:(6,48,175,303,281),
         7:(7,48,201,303,323)}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('pdf',type=Path)
    p.add_argument('--output',type=Path,default=Path('figures/paper'))
    p.add_argument('--dpi',type=int,default=160)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    for number,(page,x0,y0,x1,y1) in CROPS.items():
        scale=a.dpi/72
        subprocess.run(['pdftoppm','-f',str(page),'-l',str(page),'-singlefile',
            '-r',str(a.dpi),'-x',str(round(x0*scale)),'-y',str(round(y0*scale)),
            '-W',str(round((x1-x0)*scale)),'-H',str(round((y1-y0)*scale)),
            '-png',str(a.pdf),str(a.output/f'figure_{number:02}')],check=True)
    (a.output/'provenance.json').write_text(json.dumps(dict(
        paper='KTA-Attn ViT: A Hierarchical Token Pruning Framework for Accelerating ViTs',
        pdf_sha256=hashlib.sha256(a.pdf.read_bytes()).hexdigest(),dpi=a.dpi,
        crops_pdf_points=CROPS),indent=2)+'\n')

if __name__=='__main__':main()
