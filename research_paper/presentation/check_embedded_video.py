"""Inspect native PowerPoint media and normalize optional empty hyperlink IDs."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from xml.etree import ElementTree as ET
import hashlib
import json
import re

root = Path(__file__).resolve().parents[2]
build = root / 'tmp/presentation_video'
candidate = build / 'candidate-with-video.pptx'
source = root / 'output/presentation/AutoDrive_Final_Presentation_Ready.pptx'
video = root / 'output/video/AutoDrive_20s_Demo_Smooth_Steering.mp4'
ns = {'a':'http://schemas.openxmlformats.org/drawingml/2006/main',
      'p':'http://schemas.openxmlformats.org/presentationml/2006/main',
      'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
with ZipFile(candidate) as z:
    parts = {name:z.read(name) for name in z.namelist()}

# PowerPoint writes r:id="" for the internal ppaction://media action.
# The relationship is optional for this action. Omit the empty attribute.
slide = parts['ppt/slides/slide19.xml']
slide, count = re.subn(rb'(<a:hlinkClick\b[^>]*?)\s+r:id=""', rb'\1', slide)
parts['ppt/slides/slide19.xml'] = slide
temporary = build / 'normalized-with-video.pptx'
with ZipFile(temporary, 'w', ZIP_DEFLATED) as z:
    for name, content in parts.items():
        z.writestr(name, content)
temporary.replace(candidate)

video_digest = hashlib.sha256(video.read_bytes()).hexdigest()
media = []
for name, content in parts.items():
    if name.startswith('ppt/media/') and name.endswith('.mp4'):
        media.append({'part':name,'bytes':len(content),'matches_requested_video':hashlib.sha256(content).hexdigest()==video_digest})
if not media or not all(item['matches_requested_video'] for item in media):
    raise RuntimeError('The embedded video does not match the requested MP4')
text_changes = []
with ZipFile(source) as original:
    for i in range(1,32):
        name=f'ppt/slides/slide{i}.xml'
        before=[''.join(p.itertext()) for p in ET.fromstring(original.read(name)).findall('.//a:p',ns)]
        after=[''.join(p.itertext()) for p in ET.fromstring(parts[name]).findall('.//a:p',ns)]
        if before != after:
            text_changes.append(i)
            print(json.dumps({'slide':i,'before':before,'after':after}))
if text_changes != [19]:
    raise RuntimeError(f'Unexpected slide text changes: {text_changes}')
report={'embedded_videos':media,'slide_count':31,'slides_with_text_changes':text_changes,
        'optional_empty_media_hyperlink_ids_omitted':count}
(build/'embedded-video-inspection.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
