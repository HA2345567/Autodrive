"""Keep the original deck package and graft only the native video slide."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from xml.etree import ElementTree as E
import posixpath
import json

root=Path(__file__).resolve().parents[2]
build=root/'tmp/presentation_video'
source=root/'output/presentation/AutoDrive_Final_Presentation_Ready.pptx'
native=build/'candidate-with-video.pptx'
result=build/'candidate-preserved-with-video.pptx'
rel_ns='http://schemas.openxmlformats.org/package/2006/relationships'
ct_ns='http://schemas.openxmlformats.org/package/2006/content-types'
with ZipFile(source) as z:
    parts={name:z.read(name) for name in z.namelist()}
with ZipFile(native) as z:
    slide_rels='ppt/slides/_rels/slide19.xml.rels'
    original_rels=E.fromstring(parts[slide_rels])
    original_targets={item.get('Type'):item.get('Target') for item in original_rels}
    rels=E.fromstring(z.read(slide_rels))
    added=[]
    for item in rels:
        kind=item.get('Type').rsplit('/',1)[-1]
        if kind in ('slideLayout','notesSlide'):
            item.set('Target',original_targets[item.get('Type')])
        elif kind in ('video','media','image'):
            old_target=posixpath.normpath(posixpath.join('ppt/slides',item.get('Target')))
            filename='autodrive_smooth_steering_poster.png' if kind=='image' else 'autodrive_smooth_steering.mp4'
            new_target='ppt/media/'+filename
            parts[new_target]=z.read(old_target)
            item.set('Target','../media/'+filename)
            added.append(new_target)
        else:
            raise RuntimeError('Unexpected relationship on video slide: '+kind)
    E.register_namespace('',rel_ns)
    parts[slide_rels]=E.tostring(rels,xml_declaration=True,encoding='utf-8')
    parts['ppt/slides/slide19.xml']=z.read('ppt/slides/slide19.xml')
    parts['ppt/notesSlides/notesSlide19.xml']=z.read('ppt/notesSlides/notesSlide19.xml')
    # Notes inherit the existing notes master and retain the existing relationship IDs.
    # Native notes XML does not refer to those IDs directly.
types=E.fromstring(parts['[Content_Types].xml'])
defaults={item.get('Extension').lower() for item in types if item.tag.endswith('Default')}
for extension,content_type in [('mp4','video/mp4'),('png','image/png')]:
    if extension not in defaults:
        E.SubElement(types,'{'+ct_ns+'}Default',Extension=extension,ContentType=content_type)
E.register_namespace('',ct_ns)
parts['[Content_Types].xml']=E.tostring(types,xml_declaration=True,encoding='utf-8')
with ZipFile(result,'w',ZIP_DEFLATED) as z:
    for name,content in parts.items():
        z.writestr(name,content)
with ZipFile(source) as z:
    changed=[name for name in z.namelist() if z.read(name)!=parts[name]]
report={'candidate':str(result),'changed_original_parts':changed,'new_media_parts':sorted(set(added))}
(build/'preserved-package-inspection.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
