"""Train a regularized linear steering readout on frozen CNN image features.

Architecture: existing five convolutions -> 1,152 features -> ridge regression
in steering-wheel degrees. The unbounded head supports labels outside +/-180.
Hyperparameters are selected within the original training partition. The final
20% validation partition supplies results only after model selection is frozen.
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
import csv, hashlib, json, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))

def main():
 print('Loading frozen-feature steering training runtime...',flush=True)
 import numpy as np
 import cv2
 import tensorflow.compat.v1 as tf
 tf.disable_v2_behavior()
 from src.models import model
 cv2.setNumThreads(1)
 folder=ROOT/'output/steering_audit'
 folder.mkdir(parents=True,exist_ok=True)
 index=ROOT/'data/driving_dataset/data.txt'
 records=[(i,index.parent/line.split()[0],float(line.split()[1])) for i,line in enumerate(index.read_text().splitlines())]
 split=int(.8*len(records))
 train_ids=np.arange(0,split,4)
 val_ids=np.arange(split,len(records))
 checkpoint=ROOT/'saved_models/regression_model/model.ckpt'
 fingerprint=hashlib.sha256(Path(str(checkpoint)+'.data-00000-of-00001').read_bytes()).hexdigest()
 ids=np.concatenate([train_ids,val_ids])
 cache=folder/'frozen_rgb_features.npz'
 start=time.time()
 def load(i):
  image=cv2.imread(str(records[i][1]))
  if image is None: raise ValueError('Unreadable image '+str(records[i][1]))
  return cv2.resize(image[-150:],(200,66))[:,:,::-1].astype(np.float32)/255.
 features=None; original=None
 if cache.exists():
  with np.load(cache,allow_pickle=False) as data:
   if str(data['checkpoint_sha256'])==fingerprint and np.array_equal(data['ids'],ids):
    features=data['features']; original=data['original_deg']
    print('Using matching cached image features.',flush=True)
 if features is None:
  chunks=[]; predictions=[]
  saver=tf.train.Saver()
  config=tf.ConfigProto(intra_op_parallelism_threads=4,inter_op_parallelism_threads=1)
  with tf.Session(config=config) as sess, ThreadPoolExecutor(max_workers=4) as pool:
   saver.restore(sess,str(checkpoint))
   for offset in range(0,len(ids),64):
    batch=np.stack(list(pool.map(load,ids[offset:offset+64])))
    feat,pred=sess.run([model.h_conv5_flat,model.y_pred],feed_dict={model.x:batch,model.keep_prob:1.})
    chunks.append(feat); predictions.append(pred.reshape(-1)*180/np.pi)
    if offset%640==0 or offset+64>=len(ids): print(f'Feature extraction {min(offset+64,len(ids))}/{len(ids)}; elapsed {time.time()-start:.0f}s',flush=True)
  features=np.concatenate(chunks)
  original=np.concatenate(predictions)
  np.savez_compressed(cache,ids=ids,features=features,original_deg=original,checkpoint_sha256=fingerprint)
 n=len(train_ids)
 xt=features[:n].astype(np.float64); xv=features[n:].astype(np.float64)
 yt=np.array([records[i][2] for i in train_ids]); yv=np.array([records[i][2] for i in val_ids])
 dev_start=int(.8*split)
 fit_mask=train_ids<dev_start
 def metrics(y,pred):
  err=pred-y
  return {'n':len(y),'mae_deg':float(np.mean(np.abs(err))),'rmse_deg':float(np.sqrt(np.mean(err**2))),'bias_deg':float(np.mean(err)),'r_squared':float(1-np.sum(err**2)/np.sum((y-y.mean())**2)),'pearson_r':float(np.corrcoef(y,pred)[0,1]) if np.std(pred)>1e-10 else None,'within_5_deg_pct':float(100*np.mean(np.abs(err)<=5)),'predicted_min_deg':float(pred.min()),'predicted_max_deg':float(pred.max())}
 def prepare(x,y):
  mean=x.mean(0); scale=x.std(0); scale[scale<1e-5]=1.
  z=(x-mean)/scale
  target_mean=y.mean()
  return mean,scale,target_mean,z.T@z,z.T@(y-target_mean)
 def fit(parameters,alpha):
  mean,scale,target_mean,gram,rhs=parameters
  regularized=gram.copy(); regularized.flat[::regularized.shape[0]+1]+=alpha
  coeff=np.linalg.solve(regularized,rhs)
  return mean,scale,target_mean,coeff
 def predict(fitted,x):
  mean,scale,target_mean,coeff=fitted
  return ((x-mean)/scale)@coeff+target_mean
 print('Selecting ridge regularization on an inner training development block...',flush=True)
 prepared=prepare(xt[fit_mask],yt[fit_mask])
 tuning=[]
 for alpha in [1.,10.,100.,1000.,10000.]:
  fitted=fit(prepared,alpha)
  score=metrics(yt[~fit_mask],predict(fitted,xt[~fit_mask]))
  tuning.append({'alpha':alpha,'development_metrics':score})
  print('Development alpha',alpha,json.dumps(score),flush=True)
 alpha=min(tuning,key=lambda t:t['development_metrics']['mae_deg'])['alpha']
 print('Frozen selection: alpha',alpha,'; refitting on sampled original training rows.',flush=True)
 fitted=fit(prepare(xt,yt),alpha)
 prediction=predict(fitted,xv)
 final=metrics(yv,prediction)
 original_score=metrics(yv,original[n:].astype(np.float64))
 constant=metrics(yv,np.full(len(yv),yt.mean()))
 mean,scale,target_mean,coeff=fitted
 model_path=folder/'frozen_cnn_ridge_readout.npz'
 np.savez_compressed(model_path,feature_mean=mean,feature_scale=scale,target_mean_deg=target_mean,coefficients=coeff,alpha=alpha,checkpoint_sha256=fingerprint,channel_order='RGB')
 with open(folder/'readout_validation_predictions.csv','w',newline='') as f:
  writer=csv.writer(f); writer.writerow(['dataset_row','image','actual_deg','original_rgb_deg','ridge_readout_deg'])
  for j,i in enumerate(val_ids): writer.writerow([int(i),records[i][1].name,records[i][2],float(original[n+j]),float(prediction[j])])
 report={'method':'Frozen five-convolution CNN backbone, standardized 1,152 features, ridge-regression readout in degrees. Existing checkpoint is not fine-tuned.','protocol':{'total_rows':len(records),'original_training_rows':split,'training_stride':4,'training_samples':n,'inner_fit_samples':int(fit_mask.sum()),'inner_development_samples':int((~fit_mask).sum()),'inner_development_start_row':dev_start,'validation_start_row':split,'validation_samples':len(val_ids),'selection':'Minimum inner-development MAE; validation labels do not fit or select the readout. RGB fixed from prior training-only calibration.','limitations':'Existing ordered internal validation split, not an independently collected test set. Checkpoint training exposure and data provenance are not established. No closed-loop vehicle evaluation.'},'checkpoint':str(checkpoint.relative_to(ROOT)),'checkpoint_sha256':fingerprint,'dataset_index_sha256':hashlib.sha256(index.read_bytes()).hexdigest(),'tuning':tuning,'selected_alpha':alpha,'validation_metrics':final,'original_rgb_validation_metrics':original_score,'training_mean_baseline_validation_metrics':constant,'model_file':str(model_path.relative_to(ROOT)),'elapsed_seconds':time.time()-start}
 (folder/'readout_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print('FULL VALIDATION',json.dumps(final),flush=True)
 print('ORIGINAL RGB',json.dumps(original_score),flush=True)
 print('CONSTANT BASELINE',json.dumps(constant),flush=True)
 print('Saved model and per-frame predictions.',flush=True)

if __name__=='__main__': main()
