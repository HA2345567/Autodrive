"""Reproducible, frame-aligned steering audit and preprocessing comparison.

Checkpoint and channel-order selection use training rows only. Validation labels
never select a candidate or fit calibration. Original checkpoint files are read
without modification. This is internal validation, not closed-loop evaluation.
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
import argparse, csv, hashlib, json, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))

def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--full',action='store_true')
 parser.add_argument('--calibration-size',type=int,default=256)
 args=parser.parse_args()
 print('Loading steering runtime...',flush=True)
 import numpy as np
 import cv2
 import tensorflow.compat.v1 as tf
 tf.disable_v2_behavior()
 from src.models import model
 print('TensorFlow',tf.__version__,'devices',tf.config.list_physical_devices(),flush=True)
 cv2.setNumThreads(1)
 folder=ROOT/'output/steering_audit'
 folder.mkdir(parents=True,exist_ok=True)
 records=[]
 index=ROOT/'data/driving_dataset/data.txt'
 for row,line in enumerate(index.read_text().splitlines()):
  name,angle=line.split()[:2]
  records.append((row,index.parent/name,float(angle)))
 split=int(.8*len(records))
 calibration_ids=np.linspace(0,split-1,args.calibration_size,dtype=int)
 validation_ids=np.arange(split,len(records),1 if args.full else 18)
 cal=[records[i] for i in calibration_ids]
 val=[records[i] for i in validation_ids]
 def load(record):
  image=cv2.imread(str(record[1]))
  if image is None: raise ValueError('Unreadable image '+str(record[1]))
  return cv2.resize(image[-150:],(200,66)).astype(np.float32)/255.
 def load_samples(samples):
  with ThreadPoolExecutor(max_workers=6) as pool: return np.stack(list(pool.map(load,samples)))
 print('Loading',len(cal),'training calibration and',len(val),'validation frames.',flush=True)
 xc=load_samples(cal)
 yc=np.array([r[2] for r in cal])
 xv=load_samples(val)
 yv=np.array([r[2] for r in val])
 def metrics(y,pred):
  err=pred-y
  return {'n':len(y),'mae_deg':float(np.mean(np.abs(err))),'rmse_deg':float(np.sqrt(np.mean(err**2))),'bias_deg':float(np.mean(err)),'r_squared':float(1-np.sum(err**2)/np.sum((y-y.mean())**2)),'pearson_r':float(np.corrcoef(y,pred)[0,1]) if np.std(pred)>1e-10 else None,'within_5_deg_pct':float(100*np.mean(np.abs(err)<=5)),'actual_min_deg':float(y.min()),'actual_max_deg':float(y.max()),'predicted_min_deg':float(pred.min()),'predicted_max_deg':float(pred.max())}
 def predict(sess,images,order):
  result=[]
  for start in range(0,len(images),64):
   batch=images[start:start+64]
   if order=='RGB': batch=batch[:,:,:,::-1].copy()
   result.append(sess.run(model.y_pred,feed_dict={model.x:batch,model.keep_prob:1.}).reshape(-1))
  return np.concatenate(result).astype(np.float64)*180/np.pi
 checkpoints=[ROOT/'saved_models/regression_model/model.ckpt',ROOT/'model_training/train_steering_angle/save/model.ckpt']
 candidates=[]; seen={}
 config=tf.ConfigProto(intra_op_parallelism_threads=4,inter_op_parallelism_threads=1)
 saver=tf.train.Saver()
 start=time.time()
 with tf.Session(config=config) as sess:
  for ckpt in checkpoints:
   fingerprint=digest(str(ckpt)+'.data-00000-of-00001')
   if fingerprint in seen:
    print('Duplicate checkpoint:',ckpt.relative_to(ROOT),'matches',seen[fingerprint],flush=True)
    continue
   seen[fingerprint]=str(ckpt.relative_to(ROOT))
   saver.restore(sess,str(ckpt))
   for order in ['BGR','RGB']:
    pred=predict(sess,xc,order)
    case={'checkpoint':str(ckpt.relative_to(ROOT)),'sha256':fingerprint,'channel_order':order,'calibration_metrics':metrics(yc,pred),'calibration_predictions':pred}
    candidates.append(case)
    print('Training calibration:',case['checkpoint'],order,json.dumps(case['calibration_metrics']),flush=True)
  selected=min(candidates,key=lambda c:c['calibration_metrics']['mae_deg'])
  # Freeze selection before any validation predictions or metrics are produced.
  print('Selected using training calibration MAE:',selected['checkpoint'],selected['channel_order'],flush=True)
  for case in candidates:
   saver.restore(sess,str(ROOT/case['checkpoint']))
   pred=predict(sess,xv,case['channel_order'])
   case['validation_metrics']=metrics(yv,pred)
   case['validation_predictions']=pred
   print('Validation:',case['checkpoint'],case['channel_order'],json.dumps(case['validation_metrics']),flush=True)
  # Calibration is fitted only on the training calibration subset.
  pc=selected['calibration_predictions']; pv=selected['validation_predictions']
  bias=float(np.mean(pc-yc))
  slope,intercept=np.linalg.lstsq(np.column_stack([pc,np.ones(len(pc))]),yc,rcond=None)[0]
  methods={'raw_selected':pv,'training_bias_corrected':pv-bias,'training_affine_calibrated':slope*pv+intercept}
  train_y=np.array([r[2] for r in records[:split]])
  methods['training_median_baseline']=np.full(len(yv),np.median(train_y))
  methods['training_mean_baseline']=np.full(len(yv),np.mean(train_y))
  evaluated={name:metrics(yv,pred) for name,pred in methods.items()}
  for name,result in evaluated.items(): print('Method:',name,json.dumps(result),flush=True)
  with open(folder/('full_predictions.csv' if args.full else 'pilot_predictions.csv'),'w',newline='') as f:
   writer=csv.writer(f)
   writer.writerow(['dataset_row','image','actual_deg']+list(methods))
   for j,record in enumerate(val): writer.writerow([record[0],record[1].name,record[2]]+[float(v[j]) for v in methods.values()])
  report={'protocol':{'total_rows':len(records),'training_rows':split,'validation_rows':len(records)-split,'calibration_rows':calibration_ids.tolist(),'validation_stride':1 if args.full else 18,'validation_indices':validation_ids.tolist(),'selection':'Minimum MAE on training calibration rows only; validation labels are not used for selection or calibration.','normalization':'bottom 150 rows; resize 200x66; float32 /255; no smoothing','limitations':'Internal validation on the existing ordered split. Historical checkpoint provenance and training exposure are not independently established. No closed-loop evaluation.'},'dataset_index_sha256':digest(index),'candidates':[{k:v for k,v in c.items() if not k.endswith('_predictions')} for c in candidates],'selected':{k:v for k,v in selected.items() if k in ['checkpoint','sha256','channel_order']},'training_calibration':{'bias_deg':bias,'affine_slope':float(slope),'affine_intercept_deg':float(intercept)},'methods':evaluated,'elapsed_seconds':time.time()-start}
  path=folder/('full_audit.json' if args.full else 'pilot_audit.json')
  path.write_text(json.dumps(report,indent=2),encoding='utf-8')
  print('Saved',path,'elapsed',round(time.time()-start,1),'s',flush=True)

if __name__=='__main__': main()
