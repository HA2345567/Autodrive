from pathlib import Path
import json,sys
import pymupdf as fitz
from PIL import Image,ImageDraw
sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'tmp/presentation_rebuild';OUT.mkdir(parents=True,exist_ok=True)
sources={'synopsis':Path('D:/Telegram Desktop/Minor_synopsis PPT (2).pdf'),'paper':Path('C:/Users/Harsh/Documents/AutoDrive_Research_Paper.pdf')}
for key,path in sources.items():
 d=fitz.open(path);texts=[];thumbs=[];meta=[]
 for i,p in enumerate(d):
  text=p.get_text();texts.append(f'\n===== PAGE {i+1} =====\n'+text)
  meta.append({'page':i+1,'size':[p.rect.width,p.rect.height],'fonts':sorted({(s['font'],round(s['size'],1),s['color']) for b in p.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans']})})
  if key=='synopsis':
   png=OUT/f'reference-{i+1}.png';p.get_pixmap(matrix=fitz.Matrix(1.5,1.5),alpha=False).save(png)
   im=Image.open(png);im.thumbnail((400,240));tile=Image.new('RGB',(420,270),'#dddddd');tile.paste(im,((420-im.width)//2,22));ImageDraw.Draw(tile).text((10,4),str(i+1),fill='black');thumbs.append(tile)
 (OUT/(key+'.txt')).write_text(''.join(texts),encoding='utf-8')
 (OUT/(key+'_metadata.json')).write_text(json.dumps(meta,indent=2),encoding='utf-8')
 if thumbs:
  for start in range(0,len(thumbs),9):
   sheet=Image.new('RGB',(1260,810),'white')
   for j,im in enumerate(thumbs[start:start+9]):sheet.paste(im,((j%3)*420,(j//3)*270))
   sheet.save(OUT/f'reference-contact-{start//9+1}.png')
 print(key,'pages',len(d),'first page',meta[0])
 if key=='synopsis':print(''.join(texts))
