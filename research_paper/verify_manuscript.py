"""Structural QA for the manuscript. Visual page inspection is also required."""
import json
import re
from pathlib import Path
import pdfplumber
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parent
PDF=ROOT.parent/'output/pdf/AutoDrive_Research_Paper_Final.pdf'
manifest=json.loads((ROOT/'layout_manifest.json').read_text(encoding='utf-8'))
r=PdfReader(str(PDF))
text=' '.join(' '.join((page.extract_text() or '').split()) for page in r.pages)
assert manifest['title'] in text
assert len(r.pages)==manifest['pages']
for name in manifest['authors']:
    assert name in text
for email in ['bhardwajharsh0312@gmail.com','shrisht2723@gmail.com','rajatkandpal12@gmail.com']:
    assert email in text
assert 'All three authors reviewed and approved the final manuscript' in text
assert 'no external funding' in text and 'declare no competing interests' in text
for obsolete in ['appendix','Corresponding author','Individual roles were not supplied',
                 'Restored research figures','[fill','[replace','\ufffd']:
    assert obsolete.lower() not in text.lower(), obsolete
assert len(manifest['figures'])==11
for i,record in enumerate(manifest['figures'],1):
    assert record['caption'].startswith(f'Figure {i}.')
    assert f'Figure {i}.' in text
    assert record['section'].startswith(('3.','4.','5.','7.'))
refs=text.split(' References ')[-1]
numbers=[int(x) for x in re.findall(r'\[(\d+)\]',refs)]
assert numbers==list(range(1,manifest['reference_count']+1)),numbers
unembedded=[]
for i,page in enumerate(r.pages,1):
    for ref in page['/Resources']['/Font'].get_object().values():
        font=ref.get_object()
        descriptor=font.get('/FontDescriptor')
        if descriptor is None or not any(k in descriptor.get_object() for k in ['/FontFile','/FontFile2','/FontFile3']):
            unembedded.append((i,str(font.get('/BaseFont'))))
assert not unembedded,unembedded
outside=[]
with pdfplumber.open(str(PDF)) as pdf:
    for i,page in enumerate(pdf.pages,1):
        assert len(page.chars)>50
        for ch in page.chars:
            if ch['x0'] < -0.5 or ch['top'] < -0.5 or ch['x1'] > page.width+.5 or ch['bottom'] > page.height+.5:
                outside.append((i,ch['text']))
assert not outside,outside
print(f'PASS: {len(r.pages)} pages; 11 figures within their subject sections; '
      f'{manifest["reference_count"]} numbered references; all fonts embedded; '
      'all author details and declarations present; no appendix or text outside page bounds.')
