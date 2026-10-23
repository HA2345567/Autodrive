"""Package the current runtime and small assets without the driving images."""
from pathlib import Path
import hashlib
import json
import zipfile

root = Path(__file__).resolve().parents[1]
output = root/'output/transfer'
output.mkdir(parents=True,exist_ok=True)
target = output/'AutoDrive_GPU_Handoff.zip'
paths = [*sorted((root/'src').rglob('*.py'))]
paths += [root/p for p in [
    'requirements.txt','requirements-cpu.txt','requirements-video.txt','setup.py','pyproject.toml','.gitignore',
    'docs/NVIDIA_GPU_SETUP.md','docs/OFFLINE_VIDEO_EXPORT.md','tools/build_gpu_handoff.py',
    'saved_models/lane_detection_model/lane_detection_best.pt',
    'saved_models/object_detection_model/object_detection_best.pt',
    'saved_models/regression_model/checkpoint',
    'saved_models/regression_model/model.ckpt.index',
    'saved_models/regression_model/model.ckpt.meta',
    'saved_models/regression_model/model.ckpt.data-00000-of-00001',
    'output/steering_audit/improved_readout.npz',
    'output/steering_audit/improved_readout_audit.json',
    'output/steering_audit/comparison_summary.json',
    'output/steering_audit/STEERING_EXPERIMENT.md',
    'data/driving_dataset/data.txt','data/sterring_wheel_image.png',
]]
manifest = {'source':'Current local working tree, including uncommitted runtime fixes',
            'prepared_date':'2026-10-23','archive_root':'AutoDrive',
            'driving_images_included':False,
            'required_separate_copy':'All images under data/driving_dataset, preserving data.txt paths',
            'files':[]}
for path in paths:
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest['files'].append({'path':path.relative_to(root).as_posix(),'bytes':path.stat().st_size,
                              'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as bundle:
    for path in paths:
        bundle.write(path,'AutoDrive/'+path.relative_to(root).as_posix())
    bundle.writestr('AutoDrive/README.md',(root/'docs/NVIDIA_GPU_SETUP.md').read_text(encoding='utf-8'))
    bundle.writestr('AutoDrive/GPU_TRANSFER_MANIFEST.json',json.dumps(manifest,indent=2))
(output/'GPU_TRANSFER_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({'archive':str(target),'bytes':target.stat().st_size,'source_files_and_assets':len(paths),
                  'driving_images_included':False}))
