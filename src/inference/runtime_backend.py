"""Choose native CUDA or an optional Intel OpenVINO CPU export."""
import hashlib
import importlib.util
import json
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def prepare_weights(path, imgsz, backend, device):
    source = Path(path).resolve()
    if not source.exists():
        raise FileNotFoundError(f'Trained model missing: {source}')
    # OpenVINO is opt-in: it was slower than PyTorch on the project's older
    # Pentium CPU. Presence of a backend is not evidence that it is faster.
    if device != 'cpu' or backend in ('auto', 'torch'):
        return str(source), 'torch'
    if importlib.util.find_spec('openvino') is None:
        if backend == 'openvino':
            raise RuntimeError('Install requirements-cpu.txt to use OpenVINO.')
        LOGGER.warning('OpenVINO unavailable; using PyTorch CPU. Install requirements-cpu.txt for Intel acceleration.')
        return str(source), 'torch'
    from ultralytics import YOLO
    expected = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'imgsz': imgsz, 'precision': 'fp32', 'batch': 1}
    folder = source.with_name(source.stem + '_openvino_model')
    receipt = folder / 'autodrive_export.json'
    if receipt.exists() and list(folder.glob('*.xml')) and list(folder.glob('*.bin')):
        if json.loads(receipt.read_text()) == expected:
            return str(folder), 'openvino'
    try:
        LOGGER.info('Exporting %s for OpenVINO CPU (one-time preparation)', source.name)
        exported = Path(YOLO(str(source)).export(format='openvino', imgsz=imgsz,
                        batch=1, dynamic=False, half=False, device='cpu'))
        (exported / 'autodrive_export.json').write_text(json.dumps(expected, indent=2))
        return str(exported), 'openvino'
    except Exception:
        if backend == 'openvino':
            raise
        LOGGER.exception('OpenVINO preparation failed; falling back to original PyTorch weights')
        return str(source), 'torch'
