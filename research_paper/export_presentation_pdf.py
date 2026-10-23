"""Create the presentation PDF from the checked high-resolution slide renders."""
from pathlib import Path
import json
from zipfile import ZipFile
from PIL import Image,ImageDraw
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
ROOT=Path(__file__).resolve().parent.parent
TMP=ROOT/'tmp/presentation_rebuild'
images=sorted((TMP/'ready_renders').glob('slide-*.png'))
if len(images)!=31:raise RuntimeError(f'Expected 31 checked slide renders, found {len(images)}')
manifest=json.loads((TMP/'slide_manifest.json').read_text())
out=ROOT/'output/presentation/AutoDrive_Final_Presentation_Ready.pdf'
c=canvas.Canvas(str(out),pagesize=(960,540),pageCompression=1)
c.setTitle('AutoDrive Final Presentation')
c.setAuthor('Harsh Bhardwaj, Shrishti Singh and Rajat Kandpal')
thumbs=[]
for i,p in enumerate(images):
 c.drawImage(ImageReader(str(p)),0,0,width=960,height=540)
 c.bookmarkPage(f'slide{i+1}');c.addOutlineEntry(manifest[i]['title'],f'slide{i+1}',level=0,closed=False)
 if i==0:c.linkURL('https://github.com/HA2345567/Autodrive',(493.5,24,912,49),relative=0,thickness=0)
 if i==21:c.linkURL('https://github.com/HA2345567/Autodrive',(54,83.25,906,127.5),relative=0,thickness=0)
 c.showPage()
 im=Image.open(p);im.thumbnail((400,225));tile=Image.new('RGB',(420,252),'#DDDDDD');tile.paste(im,((420-im.width)//2,21));ImageDraw.Draw(tile).text((8,4),str(i+1),fill='black');thumbs.append(tile)
c.save()
for start in range(0,len(thumbs),9):
 sheet=Image.new('RGB',(1260,756),'white')
 for j,t in enumerate(thumbs[start:start+9]):sheet.paste(t,((j%3)*420,(j//3)*252))
 sheet.save(TMP/f'ready-contact-{start//9+1}.png')
with ZipFile(ROOT/'output/presentation/AutoDrive_Final_Presentation_Ready.pptx') as z:
 names=z.namelist()
 print('PPTX charts:',len([n for n in names if '/charts/chart' in n and n.endswith('.xml') and '/_rels/' not in n]))
 print('Workbooks:',len([n for n in names if n.startswith('ppt/embeddings/')]))
 print('Speaker notes:',len([n for n in names if n.startswith('ppt/notesSlides/notesSlide') and n.endswith('.xml') and '/_rels/' not in n]))
print(out)
