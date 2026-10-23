from pathlib import Path
from PIL import Image
from zipfile import ZipFile
import hashlib
import json

root=Path(__file__).resolve().parents[2]
build=root/'tmp/presentation_video'
renders=build/'final_renders'
# Four slides per sheet keep layouts legible for the visual scan.
for start in range(1,32,4):
    page=Image.new('RGB',(1280,720),'#dddddd')
    for j,i in enumerate(range(start,min(start+4,32))):
        with Image.open(renders/f'Slide{i}.PNG') as im:
            page.paste(im.convert('RGB').resize((640,360)),((j%2)*640,(j//2)*360))
    page.save(build/f'overview-{start:02d}.png')
source=root/'output/presentation/AutoDrive_Final_Presentation_Ready.pptx'
final=root/'output/presentation/AutoDrive_Final_Presentation_With_Smooth_Video.pptx'
with ZipFile(source) as original, ZipFile(final) as updated:
    changed=[name for name in original.namelist() if original.read(name)!=updated.read(name)]
    movie=updated.read('ppt/media/autodrive_smooth_steering.mp4')
requested=(root/'output/video/AutoDrive_20s_Demo_Smooth_Steering.mp4').read_bytes()
report={'changed_original_parts':changed,'embedded_video_matches':hashlib.sha256(movie).digest()==hashlib.sha256(requested).digest(),'bytes':final.stat().st_size}
(build/'final-package-inspection.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
