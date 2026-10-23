"""Derive empirical comparisons from retained notebook outputs, without training.

Epoch comparisons, repeated evaluation and split comparisons are distinct from
controlled component ablations. Their provenance and printed precision are kept.
"""
from pathlib import Path
import hashlib, json, re, sys
from decimal import Decimal

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent

def output_text(cell):
 parts=[]
 for out in cell.get('outputs',[]):
  value=out.get('text',out.get('data',{}).get('text/plain',''))
  parts.append(''.join(value) if isinstance(value,list) else str(value))
 return re.sub(r'\x1b\[[0-9;?]*[a-zA-Z]','','\n'.join(parts))

def read_run(path):
 nb=json.loads(path.read_text(encoding='utf-8'))
 recorded=[]
 cells=[]
 for index,cell in enumerate(nb['cells']):
  code=''.join(cell.get('source',[]))
  log=output_text(cell)
  cells.append({'cell_index':index,'code':code,'output':log})
  if 'Starting training for ' not in log: continue
  epoch=None
  for line in log.splitlines():
   m=re.match(r'^\s*(\d+)/(\d+)\s+',line)
   if m: epoch=int(m[1])
   m=re.match(r'^\s*all\s+(\d+)\s+(\d+)\s+(.+?)\s*$',line)
   if m and epoch is not None:
    values=m[3].split()
    if len(values) not in (4,8): continue
    recorded.append({'epoch':epoch,'images':int(m[1]),'instances':int(m[2]),'metrics':values,'raw_line':line.strip(),'cell_index':index})
    epoch=None
 return {'path':str(path.relative_to(PROJECT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'training_epochs':recorded,'cells':cells}

def extract():
 det=read_run(PROJECT/'model_training/train_object_detection/coco_object_detection_yolo11.ipynb')
 lane=read_run(PROJECT/'model_training/train_lane_detection/tusimple_lane_yolo11.ipynb')
 d1=next(r for r in det['training_epochs'] if r['epoch']==1)
 d50=next(r for r in det['training_epochs'] if r['epoch']==50)
 l1=next(r for r in lane['training_epochs'] if r['epoch']==1)
 l60=next(r for r in lane['training_epochs'] if r['epoch']==60)
 ltrain=next(c for c in lane['cells'] if 'Starting training for ' in c['output'])
 # The final aggregate line in the training cell is its best.pt validation.
 lend=re.findall(r'^\s*all\s+726\s+2732\s+(.+)$',ltrain['output'],re.M)[-1].split()
 lval=next(c for c in lane['cells'] if re.search(r'Mask mAP50-95:\s*0\.\d+',c['output']))
 lstand=re.findall(r'^\s*all\s+726\s+2732\s+(.+)$',lval['output'],re.M)[-1].split()
 dval=next(c for c in det['cells'] if '=== val ===' in c['output'] and '=== test ===' in c['output'])
 dvalmap=re.search(r'=== val ===\s+mAP50-95:\s*([\d.]+)',dval['output'])[1]
 dtestmap=re.search(r'=== test ===\s+mAP50-95:\s*([\d.]+)',dval['output'])[1]
 dval50=re.search(r'=== val ===\s+mAP50-95:[^\n]+\s+mAP50:\s*([\d.]+)',dval['output'])[1]
 dtest50=re.search(r'=== test ===\s+mAP50-95:[^\n]+\s+mAP50:\s*([\d.]+)',dval['output'])[1]
 comparisons=[]
 def add(label,metric,a,b,kind,notebook,cell_indices):
  delta=Decimal(b)-Decimal(a)
  comparisons.append({'comparison':label,'metric':metric,'reference':a,'observed':b,'delta':f'{delta:+f}','comparison_type':kind,'notebook':notebook['path'],'cell_indices':cell_indices})
 add('Detection: epoch 1 to 50','Validation mAP50-95',d1['metrics'][3],d50['metrics'][3],'training progress',det,[d1['cell_index']])
 add('Lane: epoch 1 to 60','Box mAP50-95',l1['metrics'][3],l60['metrics'][3],'training progress',lane,[l1['cell_index']])
 add('Lane: epoch 1 to 60','Mask mAP50-95',l1['metrics'][7],l60['metrics'][7],'training progress',lane,[l1['cell_index']])
 add('Lane: checkpoint re-evaluation','Mask mAP50-95',lend[7],lstand[7],'evaluation variation',lane,[ltrain['cell_index'],lval['cell_index']])
 add('Detection: validation to test','mAP50-95',dvalmap,dtestmap,'split comparison',det,[dval['cell_index']])
 add('Detection: validation to test','mAP50',dval50,dtest50,'split comparison',det,[dval['cell_index']])
 result={'method':'Read saved outputs from project notebooks; no new model execution. Deltas use printed output precision.','controlled_ablation_results_found':False,'sources':[{'path':r['path'],'sha256':r['sha256'],'recorded_epoch_count':len(r['training_epochs'])} for r in [det,lane]],'comparisons':comparisons,'selected_epoch_records':{'detection_epoch_1':d1,'detection_epoch_50':d50,'lane_epoch_1':l1,'lane_epoch_60':l60}}
 (ROOT/'recorded_comparisons.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 return result

if __name__=='__main__':
 sys.stdout.reconfigure(encoding='utf-8')
 result=extract()
 print(json.dumps({'sources':result['sources'],'comparisons':result['comparisons']},indent=2))
