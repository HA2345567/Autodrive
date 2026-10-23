"""Restyle the completed paper, retaining its prose, figures, tables and results.

Reference: Object_detection.pdf. Fonts: Computer Modern from Matplotlib's
official font distribution. No publication identity or watermark is copied.
"""
from pathlib import Path
from io import BytesIO
from xml.sax.saxutils import escape
from html import unescape
import re, json, urllib.request, sys
import pymupdf as fitz
import pdfplumber
from PIL import Image as PILImage, ImageOps
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import BaseDocTemplate, PageTemplate, Frame, Paragraph, Table, TableStyle, Image, Spacer, KeepTogether, PageBreak, Flowable, CondPageBreak

sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parent
TMP=ROOT.parent/'tmp/pdfs/reference_style'
TMP.mkdir(parents=True,exist_ok=True)
OUT=ROOT.parent/'output/pdf/AutoDrive_Research_Paper_Reference_Style.pdf'
SOURCE=ROOT.parent/'output/pdf/Autodrive_Research_Paper_Completed.pdf'
REPO='https://github.com/HA2345567/Autodrive'
W,H=A4
CW=370.63
BODY_TOP=78
BODY_BOTTOM=56
FOOTER_BASELINE=28
FONTDIR=ROOT/'fonts'
FONTDIR.mkdir(exist_ok=True)
for alias,filename in [('CM','cmr10.ttf'),('CM-Bold','cmb10.ttf'),('CM-Italic','cmti10.ttf')]:
 path=FONTDIR/filename
 if not path.exists():
  urllib.request.urlretrieve('https://raw.githubusercontent.com/matplotlib/matplotlib/main/lib/matplotlib/mpl-data/fonts/ttf/'+filename,path)
 pdfmetrics.registerFont(TTFont(alias,str(path)))
pdfmetrics.registerFont(TTFont('Fallback','C:/Windows/Fonts/times.ttf'))
pdfmetrics.registerFontFamily('CM',normal='CM',bold='CM-Bold',italic='CM-Italic',boldItalic='CM-Bold')

ST={}
def style(name,**kw):
 opts=dict(fontName='CM',fontSize=9.96,leading=11.96,textColor=colors.black,
           alignment=TA_JUSTIFY,spaceAfter=4,allowWidows=0,allowOrphans=0)
 opts.update(kw)
 ST[name]=ParagraphStyle(name,**opts)
style('Body',spaceAfter=3)
style('Section',fontName='CM-Bold',fontSize=14.35,leading=17.2,spaceBefore=14,spaceAfter=8,alignment=TA_LEFT,keepWithNext=False)
style('Subsection',fontName='CM-Bold',fontSize=11.96,leading=14.4,spaceBefore=9,spaceAfter=5,alignment=TA_LEFT,keepWithNext=False)
style('Title',fontSize=17.22,leading=22,alignment=TA_CENTER,spaceAfter=17)
style('Authors',fontSize=11.96,leading=15,alignment=TA_CENTER,spaceAfter=9)
style('Affiliation',fontSize=9.96,leading=12,alignment=TA_CENTER,spaceAfter=10)
style('Email',fontSize=8.97,leading=11.4,alignment=TA_CENTER,spaceAfter=2)
style('Repository',fontSize=8.97,leading=11.2,alignment=TA_CENTER,spaceBefore=7,spaceAfter=11)
style('AbstractLabel',fontName='CM-Bold',fontSize=9.96,leading=12,alignment=TA_CENTER,spaceBefore=4,spaceAfter=5)
style('Abstract',fontSize=8.97,leading=10.75,spaceAfter=0)
style('Keywords',fontSize=8.97,leading=11,spaceBefore=0,spaceAfter=12)
style('Caption',fontSize=7.97,leading=9.6,spaceBefore=5,spaceAfter=9,alignment=TA_LEFT)
style('TableCaption',fontSize=7.97,leading=9.6,spaceBefore=9,spaceAfter=5,alignment=TA_LEFT,keepWithNext=False)
style('Cell',fontSize=7.97,leading=9.4,spaceAfter=0,alignment=TA_LEFT)
style('HeadCell',fontName='CM-Bold',fontSize=7.97,leading=9.4,spaceAfter=0,alignment=TA_LEFT)
style('SmallCell',fontSize=7.0,leading=8.4,spaceAfter=0,alignment=TA_LEFT)
style('SmallHead',fontName='CM-Bold',fontSize=7.0,leading=8.4,spaceAfter=0,alignment=TA_LEFT)
style('Equation',fontSize=9.1,leading=13,spaceBefore=5,spaceAfter=6,alignment=TA_CENTER)
style('Reference',fontSize=9.0,leading=10.9,leftIndent=20,firstLineIndent=-20,spaceAfter=3.5,alignment=TA_LEFT)
style('List',leftIndent=10,firstLineIndent=-10,spaceAfter=4)

CMCHARS=set(pdfmetrics.getFont('CM').face.charToGlyph)
def text(s):
 s=s.replace('\xa0',' ').replace('\u00ad','-').replace('\u2011','-').replace('–','-').replace('—','-')
 s=s.replace('“','"').replace('”','"').replace('‘',"'").replace('’',"'")
 s=s.replace('\u037e',';').replace('\u200b','')
 # Keep mathematics readable when outside the Computer Modern text cmap.
 chunks=[]
 for ch in s:
  # BaKoMa's legacy cmap assigns some Latin-1 codepoints to TeX ligatures.
  # Restrict it to plain text; use Unicode Times for symbols and underscores.
  if (32<=ord(ch)<127 and ch!='_' and ord(ch) in CMCHARS) or ch in '\n\t': chunks.append(escape(ch))
  else: chunks.append('<font name="Fallback">'+escape(ch)+'</font>')
 return ''.join(chunks)

def p(s,sty='Body',raw=False): return Paragraph(s if raw else text(s),ST[sty])

class Article(BaseDocTemplate):
 def __init__(self):
  super().__init__(str(OUT),pagesize=A4,title='AutoDrive: Real-Time Autonomous Driving System Using Computer Vision and Deep Learning',author='Harsh Bhardwaj; Shrishti Singh; Rajat Kandpal',pageCompression=1)
  # Retain the reference's text width and mirrored margins, while using the
  # available A4 page height and placing pagination in the actual bottom margin.
  odd=Frame(100.1,BODY_BOTTOM,CW,H-BODY_TOP-BODY_BOTTOM,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
  even=Frame(124.55,BODY_BOTTOM,CW,H-BODY_TOP-BODY_BOTTOM,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
  first=Frame(100.1,BODY_BOTTOM,CW,H-132-BODY_BOTTOM,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
  self.addPageTemplates([PageTemplate('First',[first],onPage=self.footer,autoNextPageTemplate='Even'),PageTemplate('Even',[even],onPage=self.footer,autoNextPageTemplate='Odd'),PageTemplate('Odd',[odd],onPage=self.footer,autoNextPageTemplate='Even')])
  self.locations=[]
 def footer(self,c,doc):
  c.setFillColor(colors.black); c.setFont('CM',9.96)
  c.drawCentredString(W/2,FOOTER_BASELINE,str(doc.page))
 def afterFlowable(self,f):
  if isinstance(f,Paragraph):
   if f.style.name=='Section':
    key='section'+str(len(self.locations)); self.canv.bookmarkPage(key)
    self.canv.addOutlineEntry(f.getPlainText(),key,0)
   if f.style.name in ['Section','Caption','TableCaption']:
    self.locations.append({'text':f.getPlainText(),'page':self.page})

source=fitz.open(SOURCE)
def original_asset(key):
 candidates=list((ROOT/'assets').glob(key+'.*'))
 path=candidates[0]
 with PILImage.open(path) as im: size=im.size
 return path,size
def img(key,width=CW,maxheight=310):
 path,(iw,ih)=original_asset(key)
 scale=min(width/iw,maxheight/ih)
 return Image(str(path),width=iw*scale,height=ih*scale,hAlign='CENTER')
def pair(keys,ratios=(.5,.5),height=235):
 cells=[img(k,CW*r-4,height) for k,r in zip(keys,ratios)]
 t=Table([cells],colWidths=[CW*r for r in ratios],hAlign='CENTER')
 t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),0),('RIGHTPADDING',(0,0),(-1,-1),0),('TOPPADDING',(0,0),(-1,-1),0),('BOTTOMPADDING',(0,0),(-1,-1),0)]))
 return t

class Diagram(Flowable):
 """Vector drawing derived from the retained Mermaid SVG, in black and white."""
 def __init__(self,key):
  super().__init__()
  self.pdf=fitz.open(ROOT/'diagrams'/(key+'_monochrome.pdf'))
  pg=self.pdf[0]; self.ratio=pg.rect.height/pg.rect.width
  self.width=CW; self.height=CW*self.ratio
  # Preserve vector strokes/text in the final file through a post-build overlay.
  self.key=key
 def draw(self):
  page=self.canv.getPageNumber()
  x,y=self.canv.absolutePosition(0,0)
  overlays.append((page,x,y,self.width,self.height,self.pdf))

overlays=[]
TABLES={3:[('I',[.23,.16,.20,.18,.23])],4:[('II(a)',[.18,.23,.18,.21,.20]),('II(b)',[.44,.56])],6:[('III',[.46,.19,.35])],9:[('IV',[.24,.25,.26,.25])],10:[('V',[.11,.32,.39,.18])],11:[('VI(a)',[.25,.10,.12,.11,.10,.10,.12,.10])],12:[('VI(b)',[.24]+[.095]*8),('VI(c)',[.54,.46]),('VII',[.18,.12,.12,.14,.16,.16,.12])],15:[('VIII',[.20,.27,.08,.11,.08,.14,.12])]}
FIGS={2:[(495,721,1,('diagram','autodrive_workflow'))],5:[(65,264,2,('pair',['lane_labels','object_labels'])),(270,406,3,('image','class_distribution'))],7:[(63,264,4,('diagram','dave2_reference'))],11:[(165,350,5,('image','object_training')),(412,597,6,('image','lane_training'))],12:[(522,719,7,('image','class_ap'))],13:[(136,407,8,('pair',['lane_confusion','object_confusion']))],14:[(77,378,9,('image','lane_predictions')),(383,689,10,('image','object_predictions'))]}
CAPTIONS={}
EVENTS=[]
REFERENCES=[]
EQUATIONS={1:'t<sub>i</sub> = (p<sub>i+1</sub> − p<sub>i−1</sub>) / ‖p<sub>i+1</sub> − p<sub>i−1</sub>‖, &nbsp; n<sub>i</sub> = (−t<sub>i,y</sub>, t<sub>i,x</sub>)',2:'w(y) = w<sub>0</sub> + w<sub>1</sub> · y / H',3:'L<sub>i</sub> = p<sub>i</sub> + w(y<sub>i</sub>) n<sub>i</sub>, &nbsp; R<sub>i</sub> = p<sub>i</sub> − w(y<sub>i</sub>) n<sub>i</sub>',4:'Π = (L<sub>1</sub>, …, L<sub>m</sub>, R<sub>m</sub>, …, R<sub>1</sub>)',5:'L = λ<sub>box</sub> L<sub>CIoU</sub> + λ<sub>cls</sub> L<sub>BCE</sub> + λ<sub>dfl</sub> L<sub>DFL</sub> (+ L<sub>mask</sub>)',6:'IoU(d) = (w − d) / (w + d)',7:'mAP<sub>50-95</sub> = (1/10) Σ<sub>τ = 0.50, 0.55, …, 0.95</sub> AP<sub>τ</sub>',8:'L<sub>steer</sub> = (1/N) Σ<sub>i=1…N</sub> (θ<sub>pred,i</sub> − θ<sub>true,i</sub>)<super>2</super>'}
EQBOX={5:[(623,641,1),(674,690,2),(693,710,3)],6:[(98,114,4)],7:[(556,573,5),(618,631,6),(693,710,7)],8:[(94,114,8)]}

def joinlines(lines):
 return ' '.join(''.join(s['text'] for s in l['spans']).strip() for l in lines).strip()
def rich(lines):
 parts=[]
 for line in lines:
  spans=[]
  maxsize=max((s['size'] for s in line['spans']),default=10)
  baseline=max((s['origin'][1] for s in line['spans'] if s['size']>=maxsize-.2),default=0)
  for s in line['spans']:
   v=text(s['text'])
   font=s['font'].lower()
   if 'bold' in font: v='<b>'+v+'</b>'
   elif 'italic' in font or 'oblique' in font: v='<i>'+v+'</i>'
   if s['size']<maxsize*.85 and s['origin'][1]>baseline+.4: v='<sub>'+v+'</sub>'
   elif s['size']<maxsize*.85 and s['origin'][1]<baseline-.4: v='<super>'+v+'</super>'
   spans.append(v)
  parts.append(''.join(spans))
 return ' '.join(parts)

with pdfplumber.open(SOURCE) as pl:
 for n,page in enumerate(source,1):
  blocks=page.get_text('dict')['blocks']
  tbl=pl.pages[n-1].find_tables() if n in TABLES else []
  if n in TABLES:
   for ti,(label,widths) in enumerate(TABLES[n]):
    found=tbl[ti]; rect=list(found.bbox); rows=found.extract()
    if n==3: rows=rows[:7]
    if n==9:
     values=['Random initialization','30 / none','100','66×200','Adam','MSE','Fixed 1e-4','L2 0.001','-','None','-','-','-','Keep probability 0.5','Not set','No','Not recorded']
     rows=[r[:4] for r in rows]
     for ri,v in enumerate(values,1): rows[ri][3]=v
    if n==12 and ti==0:
     # Turn the very wide nine-column table into a readable metric comparison.
     old=rows
     rows=[['Metric','Epoch-1 reference','End of training (best.pt)','Standalone re-evaluation']]
     rows.extend([[re.sub(r'\s+',' ',old[0][j]),old[1][j],old[2][j],old[3][j]] for j in range(1,9)])
     widths=[.28,.24,.24,.24]
    if n==12 and ti==1:
     rows=[['Metric','Result'],['Model','DAVE-2'],['MAE','22.9616°'],['RMSE','33.2350°'],['Mean bias','+20.9977°'],['R²','−0.8769'],['Pearson correlation','−0.0404'],['Within ±5°','9.11%']]
     cap='Table VI(c). Steering checkpoint audit (n = 505); closed-loop autonomy was not evaluated.'
    else:
     candidates=[]
     for b in blocks:
      if b['type']!=0: continue
      v=joinlines(b['lines'])
      if rect[1]-32<b['bbox'][1]<rect[1] and (v.startswith('Table ') or (candidates and b['bbox'][0]>100)):
       candidates.append(v)
     cap=' '.join(candidates)
    # Record the full table/caption region to avoid duplicate text extraction.
    top=rect[1]-13
    if n==6: top=405
    if n==12 and ti==1: top=253
    EVENTS.append((n,top,'table',(label,cap,rows,widths)))
    CAPTIONS[(n,'table',ti)]=(top,rect[3]+1)
  for top,bottom,num,kind in FIGS.get(n,[]):
   caps=[]
   for b in sorted(blocks,key=lambda b:b['bbox'][1]):
    if b['type']==0 and top<=b['bbox'][1]<bottom:
     v=joinlines(b['lines'])
     if v.startswith('Fig.') or (caps and b['bbox'][1]>bottom-35): caps.append(v)
   # Explicitly preserve captions split across PDF text blocks.
   if num==4: cap='Fig. 4. DAVE-2 reference architecture [4]. Arrows show the forward pass; convolutional labels give filters, kernel/stride, and output dimensions (height × width × channels). Dense layers contain 100, 50, 10, and 1 unit.'
   else: cap=' '.join(caps)
   EVENTS.append((n,top,'figure',(num,kind,cap)))
  for top,bottom,num in EQBOX.get(n,[]): EVENTS.append((n,top,'equation',num))
  for b in blocks:
   if b['type']!=0: continue
   bbox=b['bbox']; raw=joinlines(b['lines'])
   if bbox[1]>740: continue
   if n==1 and bbox[1]<359: continue
   if any(a<=bbox[1]<z for a,z,*_ in FIGS.get(n,[])): continue
   if any(a<=bbox[1]<z for (pn,typ,ti),(a,z) in CAPTIONS.items() if pn==n): continue
   if any(a<=bbox[1]<z for a,z,_ in EQBOX.get(n,[])): continue
   if n==7 and bbox[0]>300 and 'radians internally' in raw: continue
   if n==8 and raw=='Adam, lr=1e-4': continue
   # References are rebuilt as complete entries, independent of source-page wraps.
   if (n==17 and bbox[1]>=546) or n==18:
    REFERENCES.append((n,bbox[1],raw)); continue
   lines=b['lines']; groups=[]; current=[]
   for line in lines:
    value=joinlines([line]); sizes=[s['size'] for s in line['spans']]
    section=any(s['size']>10.7 and 'Bold' in s['font'] for s in line['spans'])
    sub=all('Bold' in s['font'] and 9.2<s['size']<9.8 for s in line['spans'])
    if section or sub:
     if current: groups.append(('Body',current)); current=[]
     groups.append(('Section' if section else 'Subsection',[line]))
    else: current.append(line)
   if current: groups.append(('Body',current))
   for sty,ls in groups:
    v=joinlines(ls); html=rich(ls)
    if n==7 and v.endswith('Autodrive steering data is'):
     html+=' '+text('radians internally; reported in degrees.')
    if n==8 and v.endswith('usual choice for DAVE-2 reimplementations;'):
     html+=' '+text('Adam, lr=1e-4.')
    if sty=='Section':
     # Retain Roman identifiers so every existing section cross-reference remains valid.
     prefix=re.match(r'^([IVX]+\.)\s+(.*)$',v)
     if prefix: html=text(prefix[1]+' '+prefix[2].title())
     else: html=text(v.title())
    if re.match(r'^(•|[1-4]\))',v): sty='List'
    if n==1 and v.startswith('Keywords'): sty='Keywords'
    EVENTS.append((n,ls[0]['bbox'][1],sty,html))

# Replace the proposed experiment matrix with measurements traceable to notebook
# outputs. Keep the distinction between empirical comparisons and causal ablation.
from extract_recorded_comparisons import extract as extract_comparisons
from steering_experiment_section import load_study, study_flowables
evidence=extract_comparisons()
steering_study=load_study()
revised=[]
for n,y,kind,data in EVENTS:
 plain=unescape(re.sub(r'<[^>]+>','',data)) if isinstance(data,str) else ''
 if kind=='table' and data[0]=='V':
  rows=[['Comparison','Metric','Reference','Observed','Change']]
  rows.extend([[c['comparison'],c['metric'],c['reference'],c['observed'],c['delta']] for c in evidence['comparisons']])
  data=('V','Table V. Recorded training and evaluation comparisons from the project notebooks [36]. Change = observed - reference, at printed log precision.',rows,[.32,.27,.14,.14,.13])
 elif n==10 and kind=='Subsection' and plain=='Ablation Study':
  data=text('Recorded Experimental Comparisons')
 elif n==10 and kind=='Body' and plain.startswith('No ablation experiments have been run yet.'):
  data=text('Table V reports measurements recovered from the saved training and evaluation outputs in the project notebooks [36]. On validation data, detection mAP50-95 rises from 0.331 at epoch 1 to 0.482 at epoch 50; lane box and mask mAP50-95 rise by 0.159 and 0.190 over 60 epochs. Re-evaluating the lane checkpoint changes mask mAP50-95 from 0.363 to 0.389. Detection mAP50-95 is 0.0144 lower on the held-out test split than on validation. These comparisons quantify training progress, evaluation variation, and the validation-to-test gap. The effects of individual architectural or augmentation choices remain unisolated.')
 elif n==15 and kind=='Subsection' and plain=='Ablation/Error Analysis':
  data=text('Empirical and Error Analysis')
 elif n==15 and kind=='Body' and plain.startswith('Without ablations we cannot attribute performance'):
  data=text('The recorded comparisons in Table V show improvements over training and variation across evaluation conditions. They do not isolate the effects of model scale, mosaic, mask resolution, or ribbon width; those choices require matched experiments that vary one factor at a time. The documented failure modes are truck/car confusion, missed small objects at 640-px input, unstable per-class AP on small test counts, and thin-structure IoU sensitivity.')
 elif n==16 and kind=='List' and 'Single runs.' in plain:
  data=text('• ')+'<b>Single runs.</b> '+text('Each training configuration uses one seed. Robustness across seeds and the isolated effects of individual components remain unquantified.')
 elif n==16 and kind=='List' and 'Run the ablation matrix of Table V' in plain:
  data=text('• Extend the recorded comparisons in Table V with controlled experiments on model scale, pretraining, mosaic, ribbon width, mask resolution, and input size, using multiple seeds and bootstrap confidence intervals.')
 if steering_study:
  if kind=='table' and data[0]=='III':
   label,caption,rows,widths=data
   data=(label,caption.replace('DAVE-2 steering network','DAVE-2 reference network'),rows,widths)
  elif kind=='table' and data[0]=='VI(c)':
   label,caption,rows,widths=data
   data=(label,caption.replace('Steering checkpoint audit','Legacy BGR steering checkpoint audit'),rows,widths)
  elif n==6 and kind=='Body' and plain.startswith('Autodrive runs three networks on each camera frame'):
   data=text('Autodrive runs three networks on each camera frame (Fig. 1). The detector and lane network are YOLO11s variants with 9.42 M and 10.07 M parameters (21.4 and 32.9 GFLOPs after layer fusion). The implemented steering checkpoint contains 1,595,511 parameters, including an additional 1,164-unit dense layer. Table III and Fig. 4 describe the smaller DAVE-2 reference configuration. Section XVII reports a newly executed readout experiment using the frozen convolutional backbone.')
  elif n==4 and kind=='Body' and plain.startswith('The workflow is:'):
   data=data.replace('All training and evaluation ran on a Google', 'Perception training and evaluation ran on a Google')
   data+=' '+text('The newly executed steering experiments in Section XVII use a local Intel Pentium 4415U CPU with Python 3.13.1, TensorFlow 2.20.0, and SciPy 1.18.0.')
  elif n==7 and kind=='Body' and plain.startswith('DAVE-2. The input is a'):
   data=text('DAVE-2 and the local checkpoint. The reference pipeline uses a 66×200 RGB-to-YUV image and fixed normalization [4]. The local loader instead supplies a bottom-150-row crop resized to 200×66, with BGR channels divided by 255; it performs no YUV conversion. The five convolutions have 24, 36, 48, 64, and 64 filters with the shapes in Table III. Their output contains 1×18×64 = 1,152 features. The local checkpoint adds dense layers of 1,164, 100, 50, 10, and 1 units and a 2 atan output. Section XVII separately evaluates RGB inputs and a new linear degree readout.')
  elif n==17 and kind=='Body' and plain.startswith('Autodrive combines a YOLO11s road-object detector'):
   summary,study_report=steering_study
   candidate_metrics=summary['metrics'][summary['candidate_label']]
   data+=' '+text(f'A newly trained frozen-backbone readout (Section XVII) records {candidate_metrics["mae_deg"]:.4f} degrees MAE on all 9,082 frames of the existing validation partition, compared with 23.2570 degrees for the original RGB head. Constant baselines remain stronger and R² remains negative, so this measured improvement does not establish reliable image-conditioned steering.')
 revised.append((n,y,kind,data))
EVENTS=revised

def table_flow(label,caption,rows,widths):
 # Dense result tables use 7 pt type; descriptive tables retain 8 pt type.
 small=len(rows[0])>=7
 cs,hs=('SmallCell','SmallHead') if small else ('Cell','HeadCell')
 rows=[[p(re.sub(r'\s+',' ',str(v or '')).strip(),hs if ri==0 else cs) for v in row] for ri,row in enumerate(rows)]
 t=Table(rows,colWidths=[CW*v/sum(widths) for v in widths],repeatRows=1,hAlign='CENTER')
 t.setStyle(TableStyle([('LINEABOVE',(0,0),(-1,0),.7,colors.black),('LINEBELOW',(0,0),(-1,0),.4,colors.black),('LINEBELOW',(0,-1),(-1,-1),.7,colors.black),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),3),('RIGHTPADDING',(0,0),(-1,-1),3),('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3)]))
 caption=caption or 'Table '+label+'.'
 caption=re.sub(r'^(Table\s+[^.]+\.)',r'<b>\1</b>',text(caption),count=1)
 caption_para=p(caption,'TableCaption',True)
 t.wrap(CW,10000)
 minimum=caption_para.wrap(CW,10000)[1]+sum(t._rowHeights[:3])+20
 return [CondPageBreak(minimum),caption_para,t,Spacer(1,7)]

S=[]
S.append(p('AutoDrive: Real-Time Autonomous Driving System Using Computer Vision and Deep Learning','Title'))
S.append(p('Harsh Bhardwaj, Shrishti Singh and Rajat Kandpal','Authors'))
S.append(p('Department of Electronics and Communication Engineering, Maharaja Surajmal Institute of Technology, New Delhi, India.','Affiliation'))
for email in ['bhardwajharsh0312@gmail.com','shrisht2723@gmail.com','rajatkandpal12@gmail.com']:
 S.append(p(f'<link href="mailto:{email}" color="black">{email}</link>','Email',True))
S.append(p('<b>AutoDrive Research Team | Code &amp; proof of work</b><br/><link href="'+REPO+'" color="black">github.com/HA2345567/Autodrive</link>','Repository',True))
abstract=next(joinlines(b['lines']) for b in source[0].get_text('dict')['blocks'] if b['type']==0 and 175<b['bbox'][1]<178)
if steering_study:
 summary,study_report=steering_study
 candidate_metrics=summary['metrics'][summary['candidate_label']]
 replacement=f'A legacy 505-frame steering audit records 22.9616 deg MAE. A newly trained frozen-CNN readout achieves {candidate_metrics["mae_deg"]:.4f} deg MAE and R² {candidate_metrics["r_squared"]:.4f} on all 9,082 frames of the existing validation partition, versus 23.2570 deg MAE for the original RGB head. Constant baselines remain stronger; independent-drive and closed-loop validation remain outstanding.'
 abstract=re.sub(r'On a systematic sample of 505 steering-validation frames,.*?closed-loop behavior has not been evaluated\.',replacement,abstract)
abstract_table=Table([[p('Abstract','AbstractLabel')],[p(abstract,'Abstract')]],colWidths=[CW-48],hAlign='CENTER')
abstract_table.setStyle(TableStyle([('LEFTPADDING',(0,0),(-1,-1),0),('RIGHTPADDING',(0,0),(-1,-1),0),('TOPPADDING',(0,0),(-1,-1),0),('BOTTOMPADDING',(0,0),(-1,-1),0)]))
S.extend([abstract_table,Spacer(1,12)])

ordered=sorted(EVENTS,key=lambda e:(e[0],e[1]))
pending_heading=None
for index,(n,y,kind,data) in enumerate(ordered):
 if kind=='table':
  contents=table_flow(*data)
  if pending_heading:
   required=contents[0].height+pending_heading.wrap(CW,10000)[1]+20
   contents=[CondPageBreak(required),pending_heading]+contents[1:]
  pending_heading=None
  S.extend(contents)
 elif kind=='figure':
  num,(typ,key),cap=data
  graphic=Diagram(key) if typ=='diagram' else pair(key) if typ=='pair' else img(key,maxheight=285 if num in [9,10] else 240)
  caption=re.sub(r'^(Fig\.\s*\d+\.)',r'<b>\1</b>',text(cap),count=1)
  contents=([pending_heading] if pending_heading else [])+[Spacer(1,6),graphic,p(caption,'Caption',True)]
  pending_heading=None
  S.append(KeepTogether(contents))
 elif kind=='equation':
  # All equation glyphs use Times' Unicode coverage; mathematical content is retained.
  formula='<font name="Fallback">'+EQUATIONS[data]+'</font>'
  t=Table([[p(formula,'Equation',True),p('('+str(data)+')','Equation')]],colWidths=[CW-25,25])
  t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),0),('RIGHTPADDING',(0,0),(-1,-1),0),('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
  S.append(t)
 else:
  if kind=='Section' and unescape(re.sub(r'<[^>]+>','',data))=='References' and steering_study:
   S.extend(study_flowables(p,CW))
  if kind=='Subsection' and index+1<len(ordered) and ordered[index+1][2] in ['figure','table']:
   pending_heading=p(data,kind,True)
  else:
   if kind=='Section': S.append(CondPageBreak(100))
   elif kind=='Subsection': S.append(CondPageBreak(70))
   S.append(p(data,kind,True))

refs=' '.join(v for _,_,v in sorted(REFERENCES))
refs=re.sub(r'^References are numbered in order of first citation \(IEEE style\)\.\s*','',refs)
for entry in re.split(r'(?=\[\d+\])',refs):
 if entry.strip(): S.append(p(entry.strip(),'Reference'))
doc=Article(); doc.build(S)

# Embed the monochrome Mermaid diagrams as vector PDF objects after pagination.
result=fitz.open(OUT)
for page,x,y,w,h,diagram in overlays:
 result[page-1].show_pdf_page(fitz.Rect(x,H-y-h,x+w,H-y),diagram,0)
result.save(TMP/'assembled.pdf',garbage=4,deflate=True)
result.close()
OUT.write_bytes((TMP/'assembled.pdf').read_bytes())
(ROOT/'reference_style_manifest.json').write_text(json.dumps({'input':str(SOURCE),'output':str(OUT),'reference':'D:/Documents/Object_detection.pdf','pages':len(fitz.open(OUT)),'layout':{'text_width_pt':CW,'odd_left_pt':100.1,'even_left_pt':124.55,'top_pt':BODY_TOP,'bottom_margin_pt':BODY_BOTTOM,'footer_baseline_from_bottom_pt':FOOTER_BASELINE,'body_font_pt':9.96,'leading_pt':11.96,'color':'black text and diagrams; original color images and graphs'},'locations':doc.locations,'source_event_count':len(EVENTS),'table_v_evidence':'research_paper/recorded_comparisons.json'},indent=2,ensure_ascii=False),encoding='utf-8')
print(str(OUT)); print('Pages:',len(fitz.open(OUT)),'Events:',len(EVENTS),'Diagrams:',len(overlays))
