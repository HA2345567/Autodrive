import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {Presentation,PresentationFile} from './artifact_runtime.mjs';

const ROOT='D:/Documents/AutoDrive';
const TMP=ROOT+'/tmp/presentation_rebuild';
const ASSETS=ROOT+'/research_paper/presentation/assets';
const SKILL=process.env.AUTODRIVE_PRESENTATION_SKILL_DIR ?? 'C:/Users/Harsh/.codex/plugins/cache/openai-primary-runtime/presentations/26.1022.11436/skills/presentations';
const PY='C:/Users/Harsh/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const utils=await import(pathToFileURL(SKILL+'/container_tools/artifact_tool_utils.mjs').href);
const data=JSON.parse(await fs.readFile(ASSETS+'/tables.json','utf8'));
const study=JSON.parse(await fs.readFile(ROOT+'/output/steering_audit/comparison_summary.json','utf8'));
const evidence=JSON.parse(await fs.readFile(ROOT+'/research_paper/recorded_comparisons.json','utf8'));
const source='C:/Users/Harsh/Documents/AutoDrive_Research_Paper.pdf';
const reference='D:/Telegram Desktop/Minor_synopsis PPT (2).pdf';
const repo='https://github.com/HA2345567/Autodrive';
const C={black:'#000000',orange:'#B95827',gold:'#EF8B00',blue:'#2878B5',green:'#478B68',purple:'#7857A3',gray:'#555555',line:'#D9D9D9',pale:'#F8EAE2'};
const pres=Presentation.create({slideSize:{width:1280,height:720}});
const tableOwners=[],chartOwners=[],manifest=[];
let objectId=0;
const clean=s=>String(s??'').replace(/\n+/g,' ').replace(/\s+/g,' ').trim();
const tableBy=id=>data.find(t=>t.label===id);

function shape(s,geometry,x,y,w,h,fill='none',lineFill='none',lineWidth=0){
 return s.shapes.add({geometry,name:'object-'+(++objectId),position:{left:x,top:y,width:w,height:h},fill,line:{style:'solid',fill:lineFill,width:lineWidth}});
}
function text(s,value,x,y,w,h,size=28,opt={}){
 const t=shape(s,'textbox',x,y,w,h);
 t.text=value;
 t.text.style={typeface:opt.light?'Calibri Light':'Calibri',fontSize:size,color:opt.color??C.black,bold:opt.bold??false,alignment:opt.align??'left',verticalAlignment:opt.middle?'middle':'top',autoFit:'none',wrap:'square',insets:0};
 if(opt.link)t.text.get(clean(typeof value==='string'?value:'')).link={uri:opt.link,isExternal:true};
 return t;
}
function rule(s,x,y,w,color=C.line){shape(s,'rect',x,y,w,1,color);}
function base(title,subtitle='',appendix=false){
 const s=pres.slides.add();s.background.fill='#FFFFFF';
 shape(s,'rect',0,684,1280,5,C.gold);shape(s,'rect',0,689,1280,31,C.orange);
 text(s,title,72,46,1136,62,45,{light:true});rule(s,72,124,1136);
 if(subtitle)text(s,subtitle,72,139,1136,42,23,{color:C.gray});
 text(s,String(pres.slides.items.length),1170,691,42,24,18,{color:'#FFFFFF',align:'right'});
 manifest.push({slide:pres.slides.items.length,title,appendix});
 return s;
}
function notes(s,speech,citation='AutoDrive research paper',extra=''){
 s.speakerNotes.text=`PRESENTATION NOTES\n${speech}\n\nSOURCE\n${citation}\n${source}\n${extra}`;
}
function points(s,items,x=84,y=202,w=1095,size=28,gap=95){
 items.forEach((v,i)=>{
  text(s,String(i+1).padStart(2,'0'),x,y+i*gap,54,44,size,{bold:true,color:C.orange});
  text(s,v,x+76,y+i*gap,w-76,gap-14,size);
 });
}
async function image(s,file,x,y,w,h,alt,fit='contain'){
 const bytes=await fs.readFile(file);
 return s.images.add({blob:bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength),contentType:file.match(/\.jpe?g$/i)?'image/jpeg':'image/png',alt,fit,position:{left:x,top:y,width:w,height:h}});
}
function table(s,rows,x,y,w,h,widths=null,font=22){
 const n=pres.slides.items.indexOf(s)+1;tableOwners.push(n);
 const cols=rows[0].length;
 const t=s.tables.add({rows:rows.length,columns:cols,left:x,top:y,width:w,height:h,values:rows.map(r=>r.map(clean)),...(widths?{columnWidths:widths.map(v=>v*w/widths.reduce((a,b)=>a+b,0))}:{})});
 t.borders.assign({style:'solid',fill:'#D6C4BA',width:.7});
 for(let r=0;r<rows.length;r++){
  t.rows[r].height=h/rows.length;
  for(let c=0;c<cols;c++){
   const cell=t.getCell(r,c);cell.fill=r===0?C.orange:r%2===1?C.pale:'#FFFFFF';
   cell.text.style={typeface:'Calibri',fontSize:font,bold:r===0,color:r===0?'#FFFFFF':C.black,alignment:'left',verticalAlignment:'middle',autoFit:'none',wrap:'square',insets:7};
  }
 }
 return t;
}
function chart(s,categories,series,x,y,w,h,{max=1,percent=false,title='',legend=false,stacked=false}={}){
 const ch=s.charts.add('bar',{
  position:{left:x,top:y,width:w,height:h},categories,series,
  hasLegend:legend,legend:{position:'bottom',textStyle:{fontSize:20,fill:C.black}},
  barOptions:{direction:'column',grouping:stacked?'stacked':'clustered',gapWidth:85},
  chartFill:'#FFFFFF',plotAreaFill:'#FFFFFF',chartLine:{fill:'none',width:0},plotAreaLine:{fill:'none',width:0},
  xAxis:{textStyle:{fontSize:21,fill:C.black},line:{fill:C.gray,width:1},majorGridlines:null},
  yAxis:{min:0,max,majorUnit:percent?.2:undefined,numberFormatCode:percent?'0%':'0',title:title?{text:title,textStyle:{fontSize:20,fill:C.black}}:undefined,textStyle:{fontSize:19,fill:C.black},majorGridlines:{fill:C.line,width:1},line:{fill:C.gray,width:1}},
  dataLabels:{showValue:true,position:stacked?'center':'outEnd',numberFormatCode:percent?'0.0%':'0.00',textStyle:{fontSize:22,fill:C.black,bold:true}},
 });
 utils.applyPresentationChartFont(ch,{fontFamily:'Calibri'});
 chartOwners.push(pres.slides.items.indexOf(s)+1);return ch;
}
function box(s,label,x,y,w,h,size=25){
 const b=shape(s,'rect',x,y,w,h,'#FFFFFF',C.black,1.5);
 b.text=label;b.text.style={typeface:'Calibri',fontSize:size,color:C.black,alignment:'center',verticalAlignment:'middle',insets:10,autoFit:'none'};
 return b;
}
function connect(s,a,b,from='right',to='left',kind='straight'){
 return s.shapes.connect(a,b,{kind,fromSide:from,toSide:to,line:{style:'solid',fill:C.black,width:2},head:{type:'triangle',width:'med',length:'med'}});
}

// 1. The supplied cover image remains artwork, not evidence of a road trial.
{
 const s=pres.slides.add();
 await image(s,ASSETS+'/cover.jpeg',0,0,1280,720,'Conceptual autonomous vehicle cover image from the supplied synopsis','cover');
 shape(s,'rect',0,0,1280,250,'#000000/65');
 text(s,'AutoDrive',64,45,1100,85,68,{color:'#FFFFFF',light:true});
 text(s,'Autonomous Driving Research Prototype',64,138,1150,67,42,{color:'#FFFFFF',light:true});
 text(s,'Computer vision, lane segmentation and steering prediction',66,215,1100,48,27,{color:'#FFFFFF'});
 shape(s,'rect',0,637,1280,83,C.orange);
 text(s,'AutoDrive Research Team',64,653,560,35,28,{color:'#FFFFFF'});
 text(s,'github.com/HA2345567/Autodrive',658,655,558,33,26,{color:'#FFFFFF',align:'right',link:repo});
 manifest.push({slide:1,title:'AutoDrive',appendix:false});
 notes(s,'Good morning everyone. I am Harsh Bhardwaj, with Shrishti Singh and Rajat Kandpal. AutoDrive explores how a camera frame can support object detection, lane segmentation and steering prediction. Aaj hum architecture, measured results aur limitations explain karenge.','Supplied synopsis, page 1. Research paper, page 1.','Cover image is conceptual artwork from the supplied presentation, not a photograph of an AutoDrive vehicle trial.');
}
// 2
{
 const s=base('Project team and academic guidance');
 text(s,'Department of Electronics and Communication Engineering',72,159,1120,44,29,{bold:true});
 text(s,'Maharaja Surajmal Institute of Technology, New Delhi, India',72,207,1120,40,27);
 const rows=[['Team member','Enrolment number','Email'],['Harsh Bhardwaj','03696302823','bhardwajharsh0312@gmail.com'],['Shrishti Singh','03996302823','shrisht2723@gmail.com'],['Rajat Kandpal','04496302823','rajatkandpal12@gmail.com']];
 table(s,rows,72,281,1136,238,[.24,.23,.53],25);
 text(s,'Mentor: Ms. Jasmine, Assistant Professor',72,552,1136,49,29,{bold:true});
 notes(s,'Our team has three members. Ms. Jasmine is the mentor listed in our synopsis. Department aur college ka naam briefly introduce karo, phir project ki story par move karo.','Supplied synopsis, page 2. Research paper author block, page 1.');
}
// 3
{
 const s=base('Presentation outline');
 await image(s,ASSETS+'/agenda_car.jpeg',72,170,410,440,'Conceptual car image retained from the synopsis');
 points(s,['Problem and project objectives','Architecture, data and training','Detection, lane and steering results','Limitations, applications and next steps'],548,191,640,28,104);
 notes(s,'We will first explain the problem, then the implementation and measured results. Detailed numerical tables are in the appendix for questions. Outline ko sirf 15 seconds mein cover karo.','Presentation structure adapted from supplied synopsis, page 3.');
}
// 4
{
 const s=base('Introduction','One front-camera frame supports three separately evaluated tasks');
 points(s,['Object detection identifies road users and traffic infrastructure.','Lane segmentation estimates the pixels belonging to lane markings.','Steering regression estimates a continuous steering angle.','A visual display combines the outputs for inspection.'],84,207,1110,29,103);
 notes(s,'Driving ke time hum objects dekhte hain, lane samajhte hain, aur steering choose karte hain. Our pipeline explores the same perception tasks. Detection gives boxes and labels, segmentation gives masks, and regression gives an angle. Outputs visual display mein combine hote hain.','Research paper, Sections I and IV.');
}
// 5
{
 const s=base('Project objectives');
 points(s,['Train a compact detector for ten road-relevant COCO categories.','Convert TuSimple lane points into polygons for YOLO segmentation.','Audit the steering checkpoint and evaluate a new frozen-CNN readout.','Document each module’s accuracy, timing and evaluation limits.'],84,195,1110,29,108);
 notes(s,'Our objective is a reproducible engineering prototype. Lane annotation conversion is an important part of the work because sparse points cannot directly supervise instance masks. We also evaluate steering honestly and compare it with simple baselines.','Research paper, Section I, Objectives and Contributions.');
}
// 6
{
 const s=base('Literature review');
 table(s,[['Work','Contribution','Relevance to AutoDrive'],['DAVE-2, Bojarski et al. (2016)','Camera images to steering angle','Reference architecture for steering regression'],['SCNN and LaneNet (2018)','Lane structure and instance segmentation','Context for learning lane markings'],['Ultralytics YOLO11 (2024)','Object detection and instance segmentation','Implemented perception models'],['TuSimple benchmark (2017)','Lane annotations and evaluation protocol','Lane dataset and metric caveats']],72,170,1136,365,[.27,.36,.37],24);
 text(s,'Contribution: integration, annotation conversion and component evaluation',72,567,1136,69,28,{bold:true});
 notes(s,'DAVE-2 inspired our steering branch. Lane research shows why thin structures need careful supervision. We implement YOLO11s and YOLO11s-seg. We do not claim a new neural architecture. Earlier synopsis drawings used YOLOv4 and ConvLSTM, while this deck describes the actual implementation.','Research paper, Section II and references [4], [8], [11], [21], [22].');
}
// 7, editable black diagram
{
 const s=base('Integrated AutoDrive architecture','Parallel branches process the same camera frame');
 const a=box(s,'Front camera\nRGB frame',72,329,210,113,28);
 const det=box(s,'YOLO11s\nObject detection',395,196,260,91,27);
 const lan=box(s,'YOLO11s-seg\nLane segmentation',395,342,260,91,27);
 const ste=box(s,'CNN\nSteering regression',395,488,260,91,27);
 const d=box(s,'Bounding boxes\nand class labels',755,196,240,91,25);
 const l=box(s,'Lane masks\nand geometry cues',755,342,240,91,25);
 const r=box(s,'Estimated\nsteering angle',755,488,240,91,25);
 const out=box(s,'Visual\ndisplay',1082,328,142,115,25);
 [det,lan,ste].forEach(b=>connect(s,a,b,'right','left','elbow'));
 [[det,d],[lan,l],[ste,r]].forEach(([x,y])=>connect(s,x,y));
 [d,l,r].forEach(b=>connect(s,b,out,'right','left','elbow'));
 text(s,'Research prototype. No validated closed-loop vehicle control.',72,623,1136,37,24,{bold:true});
 notes(s,'Diagram ko left to right explain karo. A camera frame feeds three branches. Detection returns boxes and classes, lane segmentation returns masks, and steering returns an angle. The display brings the outputs together. This diagram represents visual integration, not a validated planner or vehicle controller.','Research paper, Fig. 1 and Section IV.');
}
// 8
{
 const s=base('Datasets and preprocessing');
 table(s,[['Module','Training data','Evaluation data','Preprocessing'],['Detection','8,080 COCO-10 images','975 validation + 525 test','640 × 640 input, ten classes'],['Lane segmentation','2,900 TuSimple frames','726 validation frames','Lane points converted to narrow polygons'],['Steering','36,324 ordered training rows','9,082 validation rows','Bottom 150-row crop, 200 × 66 resize']],72,178,1136,264,[.22,.25,.25,.28],24);
 text(s,'Lane annotations: 3,626 valid frames and 2,732 validation lane instances',72,483,1136,75,29,{bold:true});
 text(s,'The existing steering split is not an independent test drive.',72,588,1136,42,26);
 notes(s,'Detection has a held-out test split. Lanes use a random 80/20 split with seed 42, while steering uses the existing ordered 80/20 split. Steering collection provenance is not established. We should not attribute it to a named public dataset without evidence.','Research paper, Section III and Tables II(a), II(b).');
}
// 9
{
 const s=base('Perception models');
 table(s,[['Model','Task','Parameters','Fused compute'],['YOLO11s','10-class object detection','9,416,670','21.4 GFLOPs'],['YOLO11s-seg','1-class lane masks','10,067,203','32.9 GFLOPs']],72,175,1136,204,[.24,.31,.23,.22],28);
 const i=box(s,'640 × 640\nimage',90,459,210,100,28);
 const b=box(s,'Backbone\nfeature extraction',390,459,245,100,26);
 const n=box(s,'Neck\nfeature aggregation',725,459,245,100,26);
 const h=box(s,'Prediction\nhead',1060,459,150,100,25);
 connect(s,i,b);connect(s,b,n);connect(s,n,h);
 text(s,'Detection: boxes and classes. Segmentation: boxes, masks and confidence.',72,605,1136,42,26);
 notes(s,'YOLO models use a backbone to extract features, a neck to aggregate them, and a task-specific prediction head. Both perception modules use 640-pixel inputs. The object model has ten classes, while the lane model has one lane class.','Research paper, Table III and Section IV.');
}
// 10
{
 const s=base('Steering model and new readout');
 const labels=['RGB frame\nCrop and resize','Frozen CNN\n5 convolutions','1,152\nfeatures','Linear readout\nAngle in degrees'];
 const bs=labels.map((v,i)=>box(s,v,72+i*294,202,250,109,26));bs.slice(0,-1).forEach((b,i)=>connect(s,b,bs[i+1]));
 points(s,['The saved CNN contains 1,595,511 parameters, including a 1,164-unit dense layer.','The new experiment freezes the backbone and fits a Huber-loss readout.','The canonical 252,219-parameter DAVE-2 remains a reference configuration.'],84,375,1110,27,89);
 notes(s,'The reference DAVE-2 and the saved checkpoint have different parameter counts. Our local saved CNN includes an extra dense layer. In the new experiment, convolutional weights stay fixed and a regularized linear readout learns the angle in degrees. Explain this distinction clearly if faculty asks about the diagram.','Research paper, Sections IV and XVII, Table III and Fig. 11.');
}
// 11
{
 const s=base('Training setup and requirements');
 table(s,[['Run','Epochs','Batch','Input','Hardware / time'],['Detection','50','16','640 × 640','Colab Tesla T4 / 2.14 h'],['Lane segmentation','60','16','640 × 640','Colab Tesla T4 / 2.31 h'],['Original steering setup','30','100','66 × 200','Historical training time not recorded']],72,181,1136,258,[.27,.12,.12,.16,.33],25);
 text(s,'Software: Python, PyTorch, Ultralytics, TensorFlow, OpenCV and NumPy',72,481,1136,77,29);
 text(s,'New steering readout: local Intel Pentium 4415U CPU',72,584,1136,42,28,{bold:true});
 notes(s,'The perception runs use a T4. The new steering readout experiment runs locally on a CPU, so do not describe it as a GPU training run. We use PyTorch and TensorFlow, not Torch 7. We do not claim use of NVIDIA DRIVE PX hardware.','Research paper, Sections III, V and XVII, Table IV.');
}
// 12
{
 const s=base('Object detection results','Held-out test: 525 images and 3,116 annotated instances');
 chart(s,['Precision','Recall','mAP50','mAP50–95'],[{name:'Test result',values:[.712,.614,.672,.471],fill:C.blue,points:[{idx:0,fill:C.blue},{idx:1,fill:C.orange},{idx:2,fill:C.green},{idx:3,fill:C.purple}]}],72,208,772,369,{percent:true});
 text(s,'71.2%',895,224,300,81,64,{bold:true});text(s,'precision',898,307,290,44,30);
 text(s,'61.4%',895,393,300,81,64,{bold:true});text(s,'recall',898,476,290,44,30);
 text(s,'mAP is a detection metric. It is not classification accuracy.',72,620,1136,38,25,{bold:true});
 notes(s,'On the held-out test split, precision is 71.2 percent and recall is 61.4 percent. mAP50 is 67.2 percent and mAP50-95 is 47.1 percent. Precision tells us how often detections are correct. Recall tells us how many true objects we find. mAP ko accuracy mat bolo.','Research paper, Table VI(a), page 13.');
}
// 13
{
 const s=base('Lane segmentation results','Validation: 726 frames and 2,732 annotated lane instances');
 chart(s,['mAP50','mAP50–95'],[{name:'Boxes',values:[.992,.831],fill:C.blue},{name:'Masks',values:[.890,.389],fill:C.orange}],72,209,780,365,{percent:true,legend:true});
 text(s,'99.2%',900,223,292,83,64,{bold:true});text(s,'box mAP50',900,309,292,45,29);
 text(s,'38.9%',900,398,292,83,64,{bold:true});text(s,'mask mAP50–95',900,484,298,67,28);
 text(s,'Thin lane masks make IoU sensitive to small offsets. These are not official TuSimple leaderboard scores.',72,612,1136,56,24,{bold:true});
 notes(s,'Lane boxes achieve high scores, but thin masks are sensitive to one- or two-pixel offsets. Standalone re-evaluation gives mask mAP50-95 of 0.389, while end-of-training gives 0.363. These scores evaluate synthetic lane ribbons, not the official TuSimple point-wise metric.','Research paper, Table VI(b), Section VIII and Eq. 6.');
}
// 14
{
 const s=base('Detection training progress','50 recorded epochs on the perception training run');
 await image(s,ROOT+'/research_paper/assets/object_training.png',72,196,1136,401,'Recorded detection validation metrics and training losses');
 text(s,'Validation mAP50–95: 0.331 at epoch 1, 0.482 at epoch 50',72,616,1136,44,29,{bold:true});
 notes(s,'The validation curve improves from 0.331 after the first epoch to 0.482 at epoch 50. The losses decrease. These are recorded training outputs. This is a progress comparison and does not isolate the effect of one architecture or augmentation choice.','Research paper, Fig. 5 and Table V.');
}
// 15
{
 const s=base('Lane training progress','60 recorded epochs on the perception training run');
 await image(s,ROOT+'/research_paper/assets/lane_training.png',72,196,1136,401,'Recorded lane validation metrics and training losses');
 text(s,'Mask mAP50–95: 0.173 at epoch 1, 0.363 at epoch 60',72,616,1136,44,29,{bold:true});
 notes(s,'The lane model improves across training. Box metrics converge early, while mask metrics remain lower because lane ribbons are thin. End-of-training mask mAP50-95 reaches 0.363. Re-evaluation of the same checkpoint gives 0.389.','Research paper, Fig. 6 and Table V.');
}
// 16
{
 const s=base('Steering results and baselines','Same 9,082 validation frames for all four models');
 chart(s,['Original RGB','CNN + Huber','Train mean','Train median'],[{name:'MAE in degrees',values:Object.values(study.metrics).map(m=>Number(m.mae_deg.toFixed(4))),fill:C.blue,points:[{idx:0,fill:C.orange},{idx:1,fill:C.blue},{idx:2,fill:C.green},{idx:3,fill:C.gray}]}],72,210,790,354,{max:30,title:'MAE (degrees)'});
 text(s,'36.9%',920,222,283,87,64,{bold:true});text(s,'lower MAE than\nthe original RGB CNN',920,311,288,102,29);
 text(s,'14.6822° MAE\nR² = −0.1336',920,461,288,111,32,{bold:true});
 text(s,'Constant baselines remain stronger. Reliable image-conditioned steering is not established.',72,614,1136,53,25,{bold:true});
 notes(s,'All four models use the same 9,082 frames. The new readout reduces MAE from 23.2570 to 14.6822 degrees, a 36.9 percent reduction. But training-mean and training-median constants still give lower MAE, and R-squared remains negative. Improvement ko strong or reliable steering ka proof mat bolo.','Research paper, Table IX and Section XVII, page 20.');
}
// 17
{
 const s=base('Lane prediction examples');
 await image(s,ROOT+'/research_paper/assets/lane_predictions.jpg',72,160,770,496,'Six lane segmentation prediction examples from the manuscript');
 text(s,'What the image shows',887,192,320,53,30,{bold:true});
 text(s,'Predicted lane boxes and masks on six validation frames.\n\nExamples include shadows, underpasses and a truck occlusion.',887,269,320,260,28);
 text(s,'Qualitative evidence complements the numerical metrics.',887,560,320,85,25,{bold:true});
 notes(s,'Point to the predicted lane masks and explain how they follow painted markings. Some boxes are large because diagonal lanes create large bounding rectangles. These are examples, so do not generalize from six images alone.','Research paper, Fig. 9.');
}
// 18
{
 const s=base('Object prediction examples');
 await image(s,ROOT+'/research_paper/assets/object_predictions.jpg',72,160,770,496,'Six held-out object detection prediction examples');
 text(s,'What the image shows',887,192,320,53,30,{bold:true});
 text(s,'Bounding boxes, class names and confidence scores.\n\nThe detector finds large objects well, but small objects and recall remain challenges.',887,269,320,295,28);
 text(s,'Test images remain separate from training.',887,581,320,65,25,{bold:true});
 notes(s,'Explain boxes, labels and confidence. The test metrics show that recall remains limited even when selected images look good. These examples include six held-out test frames.','Research paper, Fig. 10.');
}
// 19
{
 const s=base('Prototype visualization');
 await image(s,ASSETS+'/prototype_display.jpeg',72,174,1136,406,'Prototype steering display screenshot supplied in the synopsis');
 text(s,'Display demonstration only. Speed and pedal indicators do not establish vehicle control.',72,611,1136,54,26,{bold:true});
 notes(s,'This screenshot from our synopsis illustrates the prototype display and steering wheel visualization. The speed and pedal indicators are interface elements. They are not a measured vehicle speed experiment or evidence of autonomous actuation.','Supplied synopsis, page 11. Research paper, Sections XI and XV.');
}
// 20
{
 const s=base('Latency and model footprint','Separate framework timings on a Tesla T4');
 table(s,[['Timing mode','Detection','Lane','Combined'],['Batch-16 evaluation','11.1 ms','11.8 ms','22.9 ms'],['Single-image prediction','17.0 ms','14.3 ms','31.3 ms'],['Saved weights','19.2 MB','20.5 MB','39.7 MB']],72,192,1136,264,[.37,.21,.21,.21],29);
 text(s,'32–44 FPS',72,503,540,89,62,{bold:true});
 text(s,'Projected throughput for the two YOLO modules only',648,509,560,95,30);
 text(s,'Full pipeline FPS was not measured. Capture, steering, display and control I/O are excluded.',72,623,1136,45,25,{bold:true});
 notes(s,'The combined timings are sums of separately measured modules. They suggest around 32 to 44 FPS for the two perception modules. This is a projection, not an end-to-end benchmark. Steering, frame capture, display and I/O are excluded.','Research paper, Table VIII and Section VIII.');
}
// 21
{
 const s=base('Challenges, applications and next steps');
 text(s,'Current limitations',72,177,535,47,32,{bold:true});
 points(s,['Steering trails simple constant baselines.','Lane scores use synthetic masks.','Independent-drive and closed-loop tests remain outstanding.'],72,250,535,26,98);
 text(s,'Research applications',687,177,521,47,32,{bold:true});
 text(s,'Component benchmarking\nLane annotation experiments\nPerception and visualization teaching',687,254,521,170,28);
 text(s,'Next steps',687,449,521,48,32,{bold:true});
 text(s,'Official lane metrics, new driving data and controlled simulator evaluation',687,517,521,120,28);
 notes(s,'This is useful as an academic research framework. We can compare modules, inspect errors and study annotation conversion. We have not demonstrated lane-keeping control or sensor fusion. Future work includes independent drives, official lane evaluation and closed-loop simulation.','Research paper, Sections IX, XI and XII.');
}
// 22
{
 const s=base('Conclusion and proof of work');
 points(s,['AutoDrive integrates detection, lane segmentation and steering prediction.','Perception results are useful, while steering reliability remains unresolved.','The implementation and experiment artifacts support reproducibility.'],84,191,1110,31,111);
 text(s,'github.com/HA2345567/Autodrive',72,550,1136,59,40,{bold:true,link:repo});
 text(s,'Thank you. Questions are welcome.',72,622,1136,41,29);
 notes(s,'To conclude, we built an integrated research prototype and documented its component results. Perception shows useful capabilities, while steering requires further work. Our repository contains the implementation. Thank you. Questions are welcome.','Research paper, Sections XII and XIV. GitHub repository: '+repo);
}
// 23
{
 const s=base('References');
 const refs=[
  'Bojarski et al. End to End Learning for Self-Driving Cars. arXiv:1604.07316, 2016.',
  'Jocher and Qiu. Ultralytics YOLO11. Ultralytics repository, 2024.',
  'TuSimple. TuSimple Lane Detection Benchmark. Dataset and evaluation code, 2017.',
  'Lin et al. Microsoft COCO: Common Objects in Context. ECCV, 2014.',
  'Neven et al. Towards End-to-End Lane Detection: An Instance Segmentation Approach. IV, 2018.',
  'AutoDrive Research Team. AutoDrive research paper and implementation repository.'
 ];
 refs.forEach((v,i)=>text(s,`[${i+1}]  ${v}`,72,174+i*75,1136,64,26));
 notes(s,'These are the principal references used for the model families, datasets and evaluation. The research paper contains the full bibliography. Appendix slides follow for detailed numerical questions.','Research paper references [4], [8], [11], [22], [27], [36].');
}
// Appendix 24
{
 const s=base('Appendix A  COCO-10 dataset counts','Instances per class. Image split: 8,080 train / 975 validation / 525 test',true);
 table(s,tableBy('II(a)').rows,72,190,1136,452,[.23,.22,.17,.19,.19],23);
 notes(s,'Backup slide. These are instance counts, not image counts. Images can contain multiple annotations. The largest-to-smallest training instance ratio is about 24 to 1.','Research paper, Table II(a).');
}
// Appendix 25
{
 const s=base('Appendix B  Per-class detection results','Held-out test results. Validation AP50–95 is shown for comparison.',true);
 table(s,tableBy('VII').rows,72,189,1136,454,[.20,.12,.12,.14,.16,.16,.12],21);
 notes(s,'Backup slide. Traffic lights and bicycles have low AP, while buses, cats and dogs score higher. Stop sign estimates are unstable because the test set contains only 22 instances.','Research paper, Table VII.');
}
// Appendix 26
{
 const s=base('Appendix C  Full perception metrics','Detection uses a held-out test. Lane results use validation only.',true);
 const det=tableBy('VI(a)').rows;
 table(s,det,72,190,1136,154,[.25,.1,.12,.11,.10,.10,.12,.10],20);
 const lane=tableBy('VI(b)').rows;
 text(s,'Lane segmentation',72,365,1136,36,27,{bold:true});
 table(s,lane,72,409,1136,252,[.23,.25,.26,.26],20);
 notes(s,'Backup slide. This preserves the full reported perception metrics. End-of-training and standalone lane results differ because evaluation conditions vary. These comparisons are not official TuSimple benchmark scores.','Research paper, Tables VI(a) and VI(b).');
}
// Appendix 27
{
 const s=base('Appendix D  Complete steering metrics','All four models below use the same 9,082-frame validation partition.',true);
 const rows=[['Model','MAE °','RMSE °','Bias °','R²','Pearson r','Within ±5°']];
 for(const [label,m] of Object.entries(study.metrics))rows.push([label,m.mae_deg.toFixed(4),m.rmse_deg.toFixed(4),(m.bias_deg>=0?'+':'')+m.bias_deg.toFixed(4),m.r_squared.toFixed(4),m.pearson_r==null?'Undefined':m.pearson_r.toFixed(4),m.within_5_deg_pct.toFixed(2)+'%']);
 table(s,rows,72,192,1136,249,[.29,.13,.13,.13,.10,.11,.14],22);
 text(s,'Legacy BGR audit: 505 frames sampled every 18 rows',72,474,1136,39,28,{bold:true});
 text(s,'MAE 22.9616°    RMSE 33.2350°    Bias +20.9977°\nR² −0.8769    Pearson r −0.0404    Within ±5° 9.11%',72,529,1136,105,27);
 notes(s,'Backup slide. Keep the 505-frame BGR audit separate from the 9,082-frame RGB and readout comparisons. Pearson correlation is undefined for constant predictions. R-squared remains negative for the selected readout.','Research paper, Tables VI(c), IX and Section XVII.');
}
// Appendix 28
{
 const s=base('Appendix E  Recorded experimental comparisons','Saved notebook outputs. These are not controlled ablation results.',true);
 const rows=[['Comparison','Metric','Reference','Observed','Change'],...evidence.comparisons.map(v=>[v.comparison,v.metric,v.reference,v.observed,v.delta])];
 table(s,rows,72,196,1136,396,[.32,.29,.13,.13,.13],23);
 text(s,'Deltas use the precision printed in the source notebooks.',72,620,1136,42,25);
 notes(s,'Backup slide. These are actual saved notebook comparisons. They show training progress, re-evaluation variation and the validation-to-test gap. They do not isolate effects of architecture, mosaic or ribbon width.','Research paper, Table V. research_paper/recorded_comparisons.json');
}
// Appendix 29
{
 const s=base('Appendix F  Training hyperparameters','Original module training configuration from the recorded runs',true);
 const rows=tableBy('IV').rows;
 table(s,rows,72,186,1136,478,[.24,.25,.26,.25],18);
 notes(s,'Backup slide. Detection and lane runs use batch 16 and 640-pixel images. Original steering uses a different training setup. The Huber readout experiment follows its own protocol, described on the next backup slide.','Research paper, Table IV.');
}
// Appendix 30
{
 const s=base('Appendix G  Steering experiment protocol','Development selection stays within the original training partition.',true);
 table(s,[['Item','Recorded value'],['Sampled training rows','9,081, selecting every fourth row'],['Inner chronological split','7,265 fit / 1,816 development'],['Candidate configurations','5 squared-error penalties + 4 Huber configurations'],['Selected readout','Huber threshold 10°, penalty α = 100'],['Final optimization','96 L-BFGS-B iterations'],['Bootstrap','100-frame blocks, 1,000 replicates, seed 20261021'],['MAE 95% interval','11.631° to 18.629°'],['RMSE 95% interval','18.241° to 35.675°'],['Paired MAE reduction interval','6.849° to 10.250° versus original RGB CNN']],72,190,1136,453,[.37,.63],22);
 notes(s,'Backup slide. The mean and scale use training features. Development MAE chooses the loss and penalty. Validation labels do not fit the readout. The existing validation partition was inspected previously and backbone training exposure is unknown, so it is not an independent test drive. Training labels range from minus 152.87 to plus 501.78 degrees, with 219 beyond plus or minus 180. Validation has no labels outside that bound. Runtime: Python 3.13.1, TensorFlow 2.20.0, NumPy 2.5.1, SciPy 1.18.0, OpenCV 5.0.0, local Pentium 4415U CPU.','Research paper, Section XVII. output/steering_audit/comparison_summary.json');
}
// Appendix 31
{
 const s=base('Appendix H  Reference architecture counts','Canonical DAVE-2 reference. The local saved CNN has 1,595,511 parameters.',true);
 const rows=tableBy('III').rows;
 table(s,rows,72,187,1136,474,[.46,.19,.35],19);
 notes(s,'Backup slide. The table reproduces the canonical reference layer counts. It is not the exact saved local graph, which includes an additional 1,164-unit dense layer. This distinction explains the difference between 252,219 reference parameters and 1,595,511 actual checkpoint parameters.','Research paper, Table III and Sections IV, XVII.');
}

await fs.mkdir(ROOT+'/output/presentation',{recursive:true});
await fs.mkdir(TMP+'/renders',{recursive:true});
const candidate=TMP+'/candidate-v2.pptx';
console.log('Built slides:',pres.slides.items.length);
await (await PresentationFile.exportPptx(pres)).save(candidate);
await fs.writeFile(TMP+'/presentation.json',JSON.stringify(pres.toProto()));
await fs.writeFile(TMP+'/slide_manifest.json',JSON.stringify(manifest,null,2));
const requirement={explicitTotalSlideCount:31,requiredNativeTableOwnerSlides:[...new Set(tableOwners)],requiredNativeChartOwnerSlides:[...new Set(chartOwners)],materializeLiteralChartWorkbooks:true};
const hash=crypto.createHash('sha256').update(await fs.readFile(reference)).digest('hex');
const final=ROOT+'/output/presentation/AutoDrive_Final_Presentation.pptx';
const result=await utils.finalizePresentation({...requirement,workspaceDir:ROOT,candidatePath:candidate,finalPath:final,pythonExecutable:PY,integrityValidatorPath:SKILL+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:SKILL+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...[...new Set(tableOwners)].flatMap(n=>['--require-native-table-slide',String(n)])],fontPolicy:{basis:'reference',families:['Calibri','Calibri Light'],referencePath:reference,referenceSha256:hash},verifyArtifactToolImport:true,receiptPath:TMP+'/final_validation.json'});
console.log('Finalization:',JSON.stringify(result));
for(let i=0;i<pres.slides.items.length;i++){
 const s=pres.slides.items[i];
 const png=await pres.export({slide:s,format:'png',scale:1});
 await fs.writeFile(TMP+'/renders/slide-'+String(i+1).padStart(2,'0')+'.png',new Uint8Array(await png.arrayBuffer()));
 console.log('Rendered',i+1);
}
const pdf=await pres.export({format:'pdf'});
await fs.writeFile(ROOT+'/output/presentation/AutoDrive_Final_Presentation.pdf',new Uint8Array(await pdf.arrayBuffer()));
await fs.writeFile(TMP+'/presentation.json',JSON.stringify(pres.toProto()));
console.log('Saved final deck and PDF');
