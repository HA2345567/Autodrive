"""Build a factual experiment report and colored plots from saved predictions."""
from pathlib import Path
import csv, hashlib, json, sys
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
FOLDER=ROOT/'output/steering_audit'

def main():
 sys.stdout.reconfigure(encoding='utf-8')
 report_path=FOLDER/('improved_readout_audit.json' if (FOLDER/'improved_readout_audit.json').exists() else 'readout_audit.json')
 report=json.loads(report_path.read_text())
 candidate_label=report.get('candidate_label','Frozen CNN + ridge')
 pilot=json.loads((FOLDER/'pilot_audit.json').read_text())
 rows=list(csv.DictReader((FOLDER/report.get('predictions_file','readout_validation_predictions.csv')).open()))
 y=np.array([float(r['actual_deg']) for r in rows])
 original=np.array([float(r['original_rgb_deg']) for r in rows])
 candidate=np.array([float(r.get('candidate_readout_deg',r.get('ridge_readout_deg'))) for r in rows])
 labels=np.array([float(l.split()[1]) for l in (ROOT/'data/driving_dataset/data.txt').read_text().splitlines()])
 split=report['protocol']['original_training_rows']
 training=labels[:split:4]
 def metrics(pred):
  err=pred-y
  return {'mae_deg':float(np.mean(np.abs(err))),'rmse_deg':float(np.sqrt(np.mean(err**2))),'bias_deg':float(np.mean(err)),'r_squared':float(1-np.sum(err**2)/np.sum((y-y.mean())**2)),'pearson_r':float(np.corrcoef(y,pred)[0,1]) if np.std(pred)>1e-10 else None,'within_5_deg_pct':float(100*np.mean(np.abs(err)<=5))}
 median=np.full(len(y),np.median(training)); mean=np.full(len(y),np.mean(training))
 methods={'Original CNN (RGB)':original,candidate_label:candidate,'Training-mean constant':mean,'Training-median constant':median}
 scores={name:metrics(pred) for name,pred in methods.items()}
 # Circular blocks preserve short-range correlation within each resampled block.
 rng=np.random.default_rng(20261021)
 block_size=100; repetitions=1000
 boot={name:{'mae':[],'rmse':[]} for name in methods}
 errors={name:pred-y for name,pred in methods.items()}
 differences=[]
 for _ in range(repetitions):
  starts=rng.integers(0,len(y),size=int(np.ceil(len(y)/block_size)))
  indices=((starts[:,None]+np.arange(block_size))%len(y)).ravel()[:len(y)]
  for name,err in errors.items():
   e=err[indices]; boot[name]['mae'].append(np.abs(e).mean()); boot[name]['rmse'].append(np.sqrt((e*e).mean()))
  differences.append(boot['Original CNN (RGB)']['mae'][-1]-boot[candidate_label]['mae'][-1])
 intervals={name:{metric+'_95_ci_deg':np.percentile(values,[2.5,97.5]).tolist() for metric,values in estimates.items()} for name,estimates in boot.items()}
 result={'validation_n':len(y),'candidate_label':candidate_label,'metrics':scores,'bootstrap':{'method':'Circular block bootstrap, block length 100 consecutive validation frames; 1000 replicates; seed 20261021. Intervals depend on the chosen block length.','intervals':intervals,'paired_mae_reduction_vs_original_95_ci_deg':np.percentile(differences,[2.5,97.5]).tolist()},'label_audit':{'training_min_deg':float(labels[:split].min()),'training_max_deg':float(labels[:split].max()),'training_labels_outside_180':int(np.sum(np.abs(labels[:split])>180)),'validation_labels_outside_180':int(np.sum(np.abs(y)>180))},'readout_sha256':hashlib.sha256((ROOT/report['model_file']).read_bytes()).hexdigest()}
 result['candidate_beats_original_mae']=scores[candidate_label]['mae_deg']<scores['Original CNN (RGB)']['mae_deg']
 result['candidate_beats_both_constant_baselines_mae']=scores[candidate_label]['mae_deg']<min(scores['Training-mean constant']['mae_deg'],scores['Training-median constant']['mae_deg'])
 result['candidate_positive_r_squared']=scores[candidate_label]['r_squared']>0
 (FOLDER/'comparison_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 lines=['# Steering model improvement experiment','',
  'This report uses newly computed frame-aligned predictions. No performance numbers were replaced with assumed targets.','',
  '## Protocol','',
  f'- Original ordered split: {split:,} training rows and {len(y):,} validation rows.',
  f'- New readout fitted on {report["protocol"]["training_samples"]:,} training frames (every fourth row).',
  f'- Inner training split: {report["protocol"]["inner_fit_samples"]:,} fit / {report["protocol"]["inner_development_samples"]:,} development frames.',
  f'- Readout penalty selected on inner-development MAE: alpha = {report["selected_alpha"]:g}; loss = {report.get("selected_loss","squared")}.',
  '- Frozen five-convolution backbone; RGB, bottom 150 rows, resize to 200x66, divide by 255.',
  '- Original RGB head and new readout are evaluated on identical validation frames.',
  '- Validation labels are excluded from fitting and hyperparameter selection. Historical backbone checkpoint exposure is not independently established.',
  '- This is internal validation on an existing ordered split. No independent driving test or closed-loop simulation is claimed.','',
  '## Full validation results','',
  '| Model | MAE (deg) | RMSE (deg) | Bias (deg) | R2 | Pearson r | Within +/-5 deg |',
  '|---|---:|---:|---:|---:|---:|---:|']
 for name,m in scores.items():
  r='undefined (constant)' if m['pearson_r'] is None else f'{m["pearson_r"]:.4f}'
  lines.append(f'| {name} | {m["mae_deg"]:.4f} | {m["rmse_deg"]:.4f} | {m["bias_deg"]:+.4f} | {m["r_squared"]:.4f} | {r} | {m["within_5_deg_pct"]:.2f}% |')
 lines+=['','## Diagnosis','',
  '- The saved checkpoint and the training-folder checkpoint have identical weights.',
  '- The earlier BGR audit is reproduced on 505 frames: MAE 22.9616 deg, RMSE 33.2350 deg, R2 -0.8769.',
  '- RGB preprocessing alone does not improve the 505-frame validation sample.',
  '- The old 2*atan output is bounded by +/-180 degrees. There are 219 original training labels outside that range, with a maximum of 501.78 degrees.',
  '- The new linear degree head removes that representational limit. Its actual usefulness is determined by the validation results above.','',
  '## Uncertainty','',result['bootstrap']['method'],'']
 for name,ci in intervals.items(): lines.append(f'- {name}: MAE 95% interval [{ci["mae_95_ci_deg"][0]:.3f}, {ci["mae_95_ci_deg"][1]:.3f}] deg; RMSE [{ci["rmse_95_ci_deg"][0]:.3f}, {ci["rmse_95_ci_deg"][1]:.3f}] deg.')
 lines+=['','## Evidence files','',
  '- `pilot_audit.json`: preprocessing comparison and training-only calibration.',
  '- `improved_readout_audit.json`: selected loss, penalty search, optimizer status and full validation.',
  '- `readout_audit.json`: the initial squared-error readout comparison.',
  '- `improved_validation_predictions.csv`: every evaluated row, image, target and selected prediction.',
  '- `improved_readout.npz`: selected readout, feature normalization and backbone hash.',
  '- `comparison_summary.json`: metrics and block-bootstrap intervals.','']
 (FOLDER/'STEERING_EXPERIMENT.md').write_text('\n'.join(lines),encoding='utf-8')
 print(json.dumps(result,indent=2),flush=True)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
 low=min(y.min(),original.min(),candidate.min()); high=max(y.max(),original.max(),candidate.max())
 for ax,name,color in zip(axes,['Original CNN (RGB)',candidate_label],['#e07b24','#2468b4']):
  pred=methods[name]
  ax.scatter(y,pred,s=3,alpha=.2,color=color,rasterized=True)
  ax.plot([low,high],[low,high],color='black',linestyle='--',linewidth=.8)
  ax.set_xlim(low,high); ax.set_ylim(low,high)
  ax.set(xlabel='Recorded angle (deg)',ylabel='Predicted angle (deg)',title=f'{name}\nMAE {scores[name]["mae_deg"]:.2f} deg; R2 {scores[name]["r_squared"]:.3f}')
  ax.grid(alpha=.2)
 fig.savefig(FOLDER/'steering_full_validation_scatter.png',dpi=240)
 plt.close(fig)
 fig,ax=plt.subplots(figsize=(9,3),layout='constrained')
 # Fixed prefix is for visualization only. All metrics above use every frame.
 nshow=min(1000,len(y))
 ax.plot(y[:nshow],color='black',linewidth=1,label='Recorded')
 ax.plot(original[:nshow],color='#e07b24',linewidth=.8,label='Original CNN (RGB)')
 ax.plot(candidate[:nshow],color='#2468b4',linewidth=.8,label=candidate_label)
 ax.set(xlabel='Validation-frame offset (fixed first 1,000 rows)',ylabel='Steering angle (deg)')
 ax.legend(ncol=3); ax.grid(alpha=.2)
 fig.savefig(FOLDER/'steering_validation_sequence.png',dpi=240)
 plt.close(fig)

if __name__=='__main__': main()
