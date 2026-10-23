"""Correct native table spacing and connector ends without changing slide data."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from lxml import etree as E
ROOT=Path(__file__).resolve().parent.parent
src=ROOT/'output/presentation/AutoDrive_Final_Presentation.pptx'
dst=ROOT/'tmp/presentation_rebuild/candidate-ready.pptx'
ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main','p':'http://schemas.openxmlformats.org/presentationml/2006/main'}
q=lambda tag:'{'+ns['a']+'}'+tag
tables=0;connectors=0
with ZipFile(src) as z,ZipFile(dst,'w',ZIP_DEFLATED) as out:
 for name in z.namelist():
  raw=z.read(name)
  if name.startswith('ppt/slides/slide') and name.endswith('.xml') and '/_rels/' not in name:
   root=E.fromstring(raw)
   for table in root.findall('.//a:tbl',ns):
    tables+=1
    for tc in table.findall('.//a:tc',ns):
     pr=tc.find('a:tcPr',ns)
     if pr is None:pr=E.SubElement(tc,q('tcPr'))
     for k,v in [('marT',19050),('marB',19050),('marL',57150),('marR',57150)]:pr.set(k,str(v))
     pr.set('anchor','ctr')
     for p in tc.findall('.//a:p',ns):
      pp=p.find('a:pPr',ns)
      if pp is None:pp=E.Element(q('pPr'));p.insert(0,pp)
      for tag in ['lnSpc','spcBef','spcAft']:
       old=pp.find(q(tag))
       if old is not None:pp.remove(old)
       el=E.SubElement(pp,q(tag));val=E.SubElement(el,q('spcPct') if tag=='lnSpc' else q('spcPts'));val.set('val','100000' if tag=='lnSpc' else '0')
   for cx in root.findall('.//p:cxnSp',ns):
    ln=cx.find('.//a:ln',ns)
    if ln is None:continue
    head=ln.find('a:headEnd',ns);tail=ln.find('a:tailEnd',ns)
    ha=dict(head.attrib) if head is not None else {'type':'none'}
    ta=dict(tail.attrib) if tail is not None else {'type':'none'}
    if head is None:head=E.SubElement(ln,q('headEnd'))
    if tail is None:tail=E.SubElement(ln,q('tailEnd'))
    head.attrib.clear();head.attrib.update(ta);tail.attrib.clear();tail.attrib.update(ha);connectors+=1
   if name=='ppt/slides/slide26.xml':
    ts=root.findall('.//a:tbl',ns)
    if len(ts)>1:
     row=ts[1].find('a:tr',ns)
     for cell,label in zip(row.findall('a:tc',ns),['Metric','Epoch 1','Training end','Re-evaluation']):
      runs=cell.findall('.//a:t',ns)
      if runs:runs[0].text=label
      for r in runs[1:]:r.text=''
   raw=E.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
  out.writestr(name,raw)
print('Repaired tables:',tables,'connectors:',connectors)
print(dst)
