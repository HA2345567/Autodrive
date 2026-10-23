"""Manuscript section for actual locally executed steering experiments."""
import json
from pathlib import Path
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon
from reportlab.platypus import Table, TableStyle, Image, KeepTogether, Spacer, CondPageBreak
from PIL import Image as PILImage

ROOT=Path(__file__).resolve().parent.parent
FOLDER=ROOT/'output/steering_audit'

def load_study():
 if not (FOLDER/'comparison_summary.json').exists(): return None
 summary=json.loads((FOLDER/'comparison_summary.json').read_text())
 report=json.loads((FOLDER/'improved_readout_audit.json').read_text())
 return summary,report

def diagram(width):
 d=Drawing(width,75)
 gap=12; bw=(width-3*gap)/4
 labels=[['RGB road frame','Crop and resize'],['Frozen CNN','5 convolutions'],['1,152 features','Training normalization'],['Linear readout','Angle in degrees']]
 for i,lines in enumerate(labels):
  x=i*(bw+gap)
  d.add(Rect(x,18,bw,42,rx=3,ry=3,fillColor=colors.white,strokeColor=colors.black,strokeWidth=.7))
  for j,label in enumerate(lines): d.add(String(x+bw/2,43-j*11,label,fontName='Helvetica',fontSize=7.1,textAnchor='middle',fillColor=colors.black))
  if i<3:
   a=x+bw; b=a+gap
   d.add(Line(a,39,b-3,39,strokeWidth=.7,strokeColor=colors.black))
   d.add(Polygon([b-3,39,b-7,41,b-7,37],fillColor=colors.black,strokeColor=colors.black))
 return d

def study_flowables(p,width):
 summary,report=load_study()
 name=summary['candidate_label']; m=summary['metrics'][name]
 original=summary['metrics']['Original CNN (RGB)']; protocol=report['protocol']
 loss=report.get('selected_loss','squared')
 rows=[]
 def add(s,sty='Body'):
  if sty=='Section': rows.append(CondPageBreak(100))
  elif sty=='Subsection': rows.append(CondPageBreak(70))
  rows.append(p(s,sty))
 add('XVII. Steering Readout Experiment','Section')
 add('A. Diagnosis and target range','Subsection')
 add('A new frame-aligned audit reproduced the original BGR checkpoint result on the same 505-frame sample: MAE 22.9616 degrees, RMSE 33.2350 degrees, and R² -0.8769. The training-folder and runtime checkpoints contain identical weights. Switching the input channel order to RGB alone did not improve that sample. Training-only bias correction reduced MAE but remained weaker than a training-derived constant baseline.')
 add('The recorded training angles range from -152.87 to +501.78 degrees; 219 training labels exceed ±180 degrees. The original output, 2 atan(z), is bounded by ±180 degrees and cannot represent those targets. Every angle in the existing validation partition lies inside this bound, so the range mismatch alone does not explain validation failure. The local checkpoint also contains a 1,164-unit dense layer and 1,595,511 trainable parameters; Table III and Fig. 4 describe the smaller reference configuration rather than that exact implementation.')
 add('B. New training experiment','Subsection')
 add(f'The proposed readout freezes the existing five-convolution backbone and replaces its dense prediction head with a regularized linear output in degrees (Fig. 11). Inputs are converted from BGR to RGB, cropped to the bottom 150 rows, resized to 200×66 pixels, and divided by 255. The flattened backbone contains 1,152 features. Training-set means and scales standardize these features; the new output is unbounded.')
 add(f'The original ordered 36,324/9,082 training/validation partition is retained. Every fourth training row supplies {protocol["training_samples"]:,} training samples. An inner chronological split supplies {protocol["inner_fit_samples"]:,} fitting samples and {protocol["inner_development_samples"]:,} development samples. Five squared-error penalties and four Huber-loss configurations are compared using development MAE. The selected loss is {loss}, with penalty α = {report["selected_alpha"]:g}'+(f' and Huber threshold {report["selected_huber_delta_deg"]:g} degrees' if loss=='huber' else '')+'. The selected readout is refitted on all sampled training rows. Validation labels do not fit the feature standardization, choose the loss or penalty, or fit the final readout.')
 rows.append(KeepTogether([diagram(width),p('Fig. 11. Steering readout experiment. The backbone is frozen; feature normalization and readout parameters are fitted within the original training partition.','Caption')]))
 add('C. Full validation and baselines','Subsection')
 add('Table IX evaluates the original RGB CNN and the selected readout on the same 9,082 validation frames, without smoothing, filtering by steering angle, or discarding high-error frames. Constant predictions use the mean or median of the same sampled training labels. Pearson correlation is undefined for constant predictions. The earlier BGR result in Table VI(c) remains a separate 505-frame audit.')
 table_rows=[['Model','MAE (deg)','RMSE (deg)','Bias (deg)','R²','Pearson r','Within ±5°']]
 for label,v in summary['metrics'].items():
  table_rows.append([label,f'{v["mae_deg"]:.3f}',f'{v["rmse_deg"]:.3f}',f'{v["bias_deg"]:+.3f}',f'{v["r_squared"]:.3f}','Undefined' if v['pearson_r'] is None else f'{v["pearson_r"]:.3f}',f'{v["within_5_deg_pct"]:.2f}%'])
 cells=[[p(v,'SmallHead' if i==0 else 'SmallCell') for v in row] for i,row in enumerate(table_rows)]
 t=Table(cells,colWidths=[width*.27]+[width*.73/6]*6,hAlign='CENTER')
 t.setStyle(TableStyle([('LINEABOVE',(0,0),(-1,0),.7,colors.black),('LINEBELOW',(0,0),(-1,0),.4,colors.black),('LINEBELOW',(0,-1),(-1,-1),.7,colors.black),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),2),('RIGHTPADDING',(0,0),(-1,-1),2),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
 rows.append(KeepTogether([p('Table IX. Newly measured steering results on all 9,082 frames of the existing validation partition.','TableCaption'),t,Spacer(1,7)]))
 reduction=100*(original['mae_deg']-m['mae_deg'])/original['mae_deg']
 add(f'The selected readout obtains MAE {m["mae_deg"]:.4f} degrees, RMSE {m["rmse_deg"]:.4f} degrees, R² {m["r_squared"]:.4f}, and {m["within_5_deg_pct"]:.2f}% of predictions within ±5 degrees. Its MAE changes by {reduction:.1f}% relative to the original RGB CNN on the same frames.')
 if summary['candidate_beats_both_constant_baselines_mae'] and summary['candidate_positive_r_squared']:
  add('The readout improves on both training-derived constant baselines in MAE and achieves positive R² on this partition. This supports an improvement in internal image-conditioned prediction. Generalization to new drives and reliable closed-loop steering remain unestablished.')
 else:
  add('The candidate does not satisfy both checks of lower MAE than the constant baselines and positive R². Reliable image-conditioned steering is therefore not established by this experiment. Lower error relative to the original CNN alone is insufficient evidence of strong steering performance.')
 ci=summary['bootstrap']['intervals'][name]
 paired=summary['bootstrap']['paired_mae_reduction_vs_original_95_ci_deg']
 add(f'A circular block bootstrap uses 100-frame blocks, 1,000 replicates, and seed 20261021 to reflect short-range temporal dependence. The candidate MAE 95% interval is [{ci["mae_95_ci_deg"][0]:.3f}, {ci["mae_95_ci_deg"][1]:.3f}] degrees; its RMSE interval is [{ci["rmse_95_ci_deg"][0]:.3f}, {ci["rmse_95_ci_deg"][1]:.3f}] degrees. The paired MAE reduction relative to the original RGB model has interval [{paired[0]:.3f}, {paired[1]:.3f}] degrees. These intervals depend on the chosen block length and partition.')
 for filename,num,caption in [('steering_full_validation_scatter.png',12,'Recorded versus predicted steering angles for all 9,082 validation frames. The dashed line denotes perfect agreement; both panels use the same axes.'),('steering_validation_sequence.png',13,'A fixed prefix of the first 1,000 validation frames, shown for visual comparison. Table IX and all reported metrics use all 9,082 frames.')]:
  path=FOLDER/filename
  with PILImage.open(path) as im: iw,ih=im.size
  image=Image(str(path),width=width,height=width*ih/iw)
  rows.append(KeepTogether([image,p(f'Fig. {num}. '+caption,'Caption')]))
 add('D. Reproducibility and limits','Subsection')
 add('Experiments run on an Intel Pentium 4415U CPU using Python 3.13.1, TensorFlow 2.20.0, NumPy 2.5.1, SciPy 1.18.0, and OpenCV 5.0.0. The selected Huber fit converges after 96 L-BFGS-B iterations. This local readout experiment is separate from the Colab/T4 perception runs.')
 add('The experiment saves the selected readout, backbone and dataset-index SHA-256 hashes, development scores for every candidate, per-frame predictions, and bootstrap intervals alongside the project manuscript. Scripts are audit_steering_checkpoint.py, train_steering_readout.py, train_robust_steering_readout.py, and summarize_steering_experiment.py. The files document newly executed local experiments; the first-epoch and perception results elsewhere in this paper remain notebook records.')
 add('The validation partition is the existing ordered split and has been inspected previously. It is not an independent test drive. Historical training exposure of the saved backbone and collection provenance are not established. One backbone checkpoint and one readout fitting procedure are evaluated. There is no simulation, vehicle-actuation trial, or closed-loop safety assessment. These limitations apply even if individual offline metrics improve.')
 return rows
