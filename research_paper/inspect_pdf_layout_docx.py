from pathlib import Path
from collections import Counter
import re, json, sys
import pymupdf as fitz
from PIL import Image, ImageDraw, ImageChops
sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parent.parent
TMP=ROOT/'tmp/docx_pdf_layout'
source=fitz.open(ROOT/'output/pdf/AutoDrive_Research_Paper_Reference_Style.pdf')
render=fitz.open(TMP/(sys.argv[1] if len(sys.argv)>1 else 'final_render.pdf'))
def tokens(s):return Counter(re.findall(r'[A-Za-z0-9]+',s))
thumbs=[];report=[]
for i,page in enumerate(render):
 path=TMP/f'page-{i+1}.png'
 pix=page.get_pixmap(matrix=fitz.Matrix(2,2),alpha=False);pix.save(path)
 im=Image.open(path);im.thumbnail((300,424));tile=Image.new('RGB',(320,450),'#dddddd');tile.paste(im,((320-im.width)//2,20));ImageDraw.Draw(tile).text((8,3),str(i+1),fill='black');thumbs.append(tile)
 a=tokens(source[i].get_text());b=tokens(page.get_text())
 expected=source[i].get_text('words');actual=page.get_text('words')
 # Native Word adds rounding at its 1/20-point layout grid. Report placement for shared words.
 byword={}
 for w in actual:byword.setdefault(w[4],[]).append(w)
 errors=[]
 for w in expected:
  candidates=byword.get(w[4],[])
  if candidates:errors.append(min(abs(v[0]-w[0])+abs(v[1]-w[1]) for v in candidates))
 report.append({'page':i+1,'source_tokens':sum(a.values()),'word_tokens':sum(b.values()),'missing':dict(a-b),'extra':dict(b-a),'median_position_error_pt':sorted(errors)[len(errors)//2] if errors else None})
for start in range(0,len(thumbs),8):
 sheet=Image.new('RGB',(1280,900),'white')
 for j,t in enumerate(thumbs[start:start+8]):sheet.paste(t,((j%4)*320,(j//4)*450))
 sheet.save(TMP/f'contact-{start//8+1}.png')
result={'source_pages':len(source),'word_pages':len(render),'pages':report}
(TMP/'layout_check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
