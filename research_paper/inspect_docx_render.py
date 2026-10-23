from pathlib import Path
import pymupdf as fitz
from PIL import Image, ImageOps, ImageDraw
import sys
sys.stdout.reconfigure(encoding='utf-8')
folder=Path(__file__).resolve().parent.parent/'tmp/docx_latest/render'
pdf=fitz.open(folder/'AutoDrive_Latest_QA.pdf')
thumbs=[]
for i,page in enumerate(pdf):
 path=folder/f'page-{i+1}.png'
 page.get_pixmap(matrix=fitz.Matrix(2,2),alpha=False).save(path)
 im=Image.open(path);im.thumbnail((300,424))
 tile=Image.new('RGB',(320,450),'#dddddd');tile.paste(im,((320-im.width)//2,20))
 ImageDraw.Draw(tile).text((10,3),str(i+1),fill='black');thumbs.append(tile)
 blocks=page.get_text('blocks')
 print(i+1,'blocks',len(blocks),'bottom',round(max((b[3] for b in blocks),default=0),1),'tail',page.get_text()[-80:].replace('\n',' | '))
for start in range(0,len(thumbs),8):
 sheet=Image.new('RGB',(1280,900),'white')
 for j,im in enumerate(thumbs[start:start+8]):sheet.paste(im,((j%4)*320,(j//4)*450))
 sheet.save(folder/f'contact-{start//8+1}.png')
print('PAGES',len(pdf))
