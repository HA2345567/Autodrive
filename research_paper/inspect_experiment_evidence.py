"""Extract notebook experiment records without executing notebook cells."""
import json, re, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'tmp/pdfs/experiment_evidence'
OUT.mkdir(parents=True,exist_ok=True)
for path in (ROOT/'model_training').rglob('*.ipynb'):
 nb=json.loads(path.read_text(encoding='utf-8'))
 log=[]
 print('\nNOTEBOOK',path.relative_to(ROOT))
 for i,cell in enumerate(nb['cells']):
  code=''.join(cell.get('source',[]))
  outputs=[]
  for o in cell.get('outputs',[]):
   v=o.get('text',o.get('data',{}).get('text/plain',''))
   outputs.append(''.join(v) if isinstance(v,list) else str(v))
  cleaned=re.sub(r'\x1b\[[0-9;?]*[a-zA-Z]','','\n'.join(outputs))
  if cell['cell_type']=='code':
   summary=' '.join(code.split())[:210]
   print(f'Cell {i}: {summary}; text output={len(cleaned)} chars')
  if cleaned:
   log.extend([f'===== CELL {i} SOURCE =====',code,'===== OUTPUT =====',cleaned])
 (OUT/(path.stem+'_records.txt')).write_text('\n'.join(log),encoding='utf-8')
 for line in '\n'.join(log).splitlines():
  if re.search(r'^\s*(1|50|60)/\d+|^\s*all\s+\d|Starting training|optimizer:|mAP50-95|mAP50|Speed:|Validating ',line):
   print(line[:500])
