from pathlib import Path
import sys,json,hashlib
import pymupdf as fitz
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'presentation/assets'
OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT))
ns={'__file__':str(ROOT/'build_reference_style.py'),'__name__':'presentation_source'}
code=(ROOT/'build_reference_style.py').read_text(encoding='utf-8').split('doc=Article(); doc.build(S)')[0]
exec(compile(code,str(ROOT/'build_reference_style.py'),'exec'),ns)
tables=[]
for n,y,kind,data in sorted(ns['EVENTS']):
 if kind=='table':
  label,caption,rows,widths=data
  tables.append({'label':label,'caption':caption,'rows':rows,'widths':widths})
(OUT/'tables.json').write_text(json.dumps(tables,indent=2,ensure_ascii=False),encoding='utf-8')
ref=fitz.open('D:/Telegram Desktop/Minor_synopsis PPT (2).pdf')
for page,key in [(0,'cover'),(2,'agenda_car'),(10,'prototype_display')]:
 images=ref[page].get_images(full=True)
 image=max(images,key=lambda v:v[2]*v[3]);data=ref.extract_image(image[0]);(OUT/(key+'.'+data['ext'])).write_bytes(data['image'])
print('Extracted',len(tables),'tables and 3 source images')
