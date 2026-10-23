from pathlib import Path
from zipfile import ZipFile
from lxml import etree
from PIL import Image,ImageDraw
import json
ROOT=Path(__file__).resolve().parent.parent
TMP=ROOT/'tmp/presentation_rebuild'
images=sorted((TMP/'renders').glob('slide-*.png'))
thumbs=[]
for i,p in enumerate(images):
 im=Image.open(p);im.thumbnail((400,225));tile=Image.new('RGB',(420,252),'#DDDDDD');tile.paste(im,((420-im.width)//2,21));ImageDraw.Draw(tile).text((8,4),str(i+1),fill='black');thumbs.append(tile)
for start in range(0,len(thumbs),9):
 sheet=Image.new('RGB',(1260,756),'white')
 for j,t in enumerate(thumbs[start:start+9]):sheet.paste(t,((j%3)*420,(j//3)*252))
 sheet.save(TMP/f'final-contact-{start//9+1}.png')
f=ROOT/'output/presentation/AutoDrive_Final_Presentation.pptx'
if f.exists():
 with ZipFile(f) as z:
  names=z.namelist();slides=[n for n in names if n.startswith('ppt/slides/slide') and n.endswith('.xml') and '/_rels/' not in n]
  notes=[n for n in names if n.startswith('ppt/notesSlides/notesSlide') and n.endswith('.xml') and '/_rels/' not in n]
  chart=[n for n in names if n.startswith('ppt/charts/chart') and n.endswith('.xml')]
  print(json.dumps({'slides':len(slides),'speaker_notes':len(notes),'editable_charts':len(chart),'workbooks':len([n for n in names if n.startswith('ppt/embeddings/')]),'rendered_slides':len(images)},indent=2))
