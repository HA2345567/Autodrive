from pathlib import Path
from docx import Document
from docx.oxml.ns import qn
p=Path(__file__).resolve().parent.parent/'output/docx/AutoDrive_Research_Paper_Latest.docx'
d=Document(p)
removed=0
for root in [d.styles.element,d.element]:
 for node in list(root.iter(qn('w:pBdr'))):
  node.getparent().remove(node);removed+=1
for style in d.styles:
 for fonts in style.element.iter(qn('w:rFonts')):
  for attr in list(fonts.attrib):
   if attr.endswith('Theme'):del fonts.attrib[attr]
d.save(p)
print('Removed paragraph borders:',removed)
