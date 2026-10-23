"""Add Huber-loss candidates to the predeclared training-only readout search."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
from pathlib import Path
import csv, json, sys, time
import numpy as np
from scipy.optimize import minimize
ROOT=Path(__file__).resolve().parent.parent
FOLDER=ROOT/'output/steering_audit'

def metrics(y,pred):
 err=pred-y
 return {'n':len(y),'mae_deg':float(np.abs(err).mean()),'rmse_deg':float(np.sqrt((err**2).mean())),'bias_deg':float(err.mean()),'r_squared':float(1-np.sum(err**2)/np.sum((y-y.mean())**2)),'pearson_r':float(np.corrcoef(y,pred)[0,1]) if np.std(pred)>1e-10 else None,'within_5_deg_pct':float(100*np.mean(np.abs(err)<=5)),'predicted_min_deg':float(pred.min()),'predicted_max_deg':float(pred.max())}

def fit_huber(x,y,alpha,delta,maxiter):
 mean=x.mean(0); scale=x.std(0); scale[scale<1e-5]=1.
 z=(x-mean)/scale
 n,p=z.shape
 def objective(w):
  err=z@w[:-1]+w[-1]-y
  clipped=np.clip(err,-delta,delta)
  loss=np.where(np.abs(err)<=delta,.5*err**2,delta*(np.abs(err)-.5*delta)).mean()+alpha/(2*n)*np.sum(w[:-1]**2)
  gradient=np.concatenate([z.T@clipped/n+alpha/n*w[:-1],[clipped.mean()]])
  return loss,gradient
 initial=np.zeros(p+1); initial[-1]=np.median(y)
 result=minimize(objective,initial,method='L-BFGS-B',jac=True,options={'maxiter':maxiter,'ftol':1e-8,'gtol':1e-5,'maxls':30})
 return {'mean':mean,'scale':scale,'coef':result.x[:-1],'intercept':result.x[-1],'optimizer_success':bool(result.success),'iterations':int(result.nit),'optimizer_message':str(result.message)}

def predict(model,x): return ((x-model['mean'])/model['scale'])@model['coef']+model['intercept']

def main():
 sys.stdout.reconfigure(encoding='utf-8')
 start=time.time()
 base=json.loads((FOLDER/'readout_audit.json').read_text())
 plan=json.loads((ROOT/'research_paper/steering_readout_plan.json').read_text())
 with np.load(FOLDER/'frozen_rgb_features.npz',allow_pickle=False) as data:
  features=data['features'].astype(np.float64); ids=data['ids']; original=data['original_deg']
 labels=np.array([float(l.split()[1]) for l in (ROOT/'data/driving_dataset/data.txt').read_text().splitlines()])
 n=base['protocol']['training_samples']; xt=features[:n]; xv=features[n:]
 yt=labels[ids[:n]]; yv=labels[ids[n:]]
 mask=ids[:n]<base['protocol']['inner_development_start_row']
 tuning=[dict(t,loss='squared') for t in base['tuning']]
 for candidate in plan['huber_candidates']:
  print('Fitting inner-training Huber candidate',candidate,flush=True)
  fitted=fit_huber(xt[mask],yt[mask],candidate['alpha'],candidate['delta_deg'],plan['huber_max_iterations'])
  score=metrics(yt[~mask],predict(fitted,xt[~mask]))
  record=dict(candidate,loss='huber',development_metrics=score,optimizer_success=fitted['optimizer_success'],iterations=fitted['iterations'])
  tuning.append(record)
  print('Development result',json.dumps(record),flush=True)
 selected=min(tuning,key=lambda c:c['development_metrics']['mae_deg'])
 print('Selected only by inner-development MAE:',json.dumps(selected),flush=True)
 if selected['loss']=='huber':
  fitted=fit_huber(xt,yt,selected['alpha'],selected['delta_deg'],plan['huber_max_iterations'])
  prediction=predict(fitted,xv)
 else:
  with np.load(FOLDER/'frozen_cnn_ridge_readout.npz',allow_pickle=False) as data:
   fitted={'mean':data['feature_mean'],'scale':data['feature_scale'],'coef':data['coefficients'],'intercept':float(data['target_mean_deg'])}
  prediction=predict(fitted,xv)
 path=FOLDER/'improved_readout.npz'
 np.savez_compressed(path,feature_mean=fitted['mean'],feature_scale=fitted['scale'],coefficients=fitted['coef'],target_mean_deg=fitted['intercept'],channel_order='RGB',loss=selected['loss'],checkpoint_sha256=base['checkpoint_sha256'])
 with open(FOLDER/'improved_validation_predictions.csv','w',newline='') as f:
  writer=csv.writer(f); writer.writerow(['dataset_row','image','actual_deg','original_rgb_deg','candidate_readout_deg'])
  names=[l.split()[0] for l in (ROOT/'data/driving_dataset/data.txt').read_text().splitlines()]
  for j,i in enumerate(ids[n:]): writer.writerow([int(i),names[i],float(yv[j]),float(original[n+j]),float(prediction[j])])
 report=dict(base)
 report.update(method='Frozen five-convolution backbone with standardized 1152 features and a regularized linear degree head. Loss and penalty selected within training partition.',tuning=tuning,selected_loss=selected['loss'],selected_alpha=selected['alpha'],selected_huber_delta_deg=selected.get('delta_deg'),validation_metrics=metrics(yv,prediction),model_file=str(path.relative_to(ROOT)),predictions_file='improved_validation_predictions.csv',candidate_label='Frozen CNN + Huber' if selected['loss']=='huber' else 'Frozen CNN + ridge',optimizer={k:fitted[k] for k in ['optimizer_success','iterations','optimizer_message'] if k in fitted},elapsed_seconds=time.time()-start)
 (FOLDER/'improved_readout_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print('FINAL VALIDATION',json.dumps(report['validation_metrics']),flush=True)

if __name__=='__main__': main()
