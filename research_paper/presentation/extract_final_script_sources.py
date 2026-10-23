from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as E
import json

root=Path(__file__).resolve().parents[2]
deck=root/'output/presentation/AutoDrive_Final_Presentation_With_Smooth_Video.pptx'
ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
rows=[]
with ZipFile(deck) as z:
    for number in range(1,32):
        slide=E.fromstring(z.read(f'ppt/slides/slide{number}.xml'))
        notes=E.fromstring(z.read(f'ppt/notesSlides/notesSlide{number}.xml'))
        paragraphs=lambda doc: [''.join(p.itertext()) for p in doc.findall('.//a:p',ns) if ''.join(p.itertext()).strip()]
        row={'slide':number,'text':paragraphs(slide),'notes':paragraphs(notes)}
        rows.append(row)
        print(json.dumps(row,ensure_ascii=True))
(root/'tmp/presentation_video/script-sources.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
