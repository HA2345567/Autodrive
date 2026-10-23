"""Build the submission manuscript with figures placed in their relevant sections.

Run with the Codex bundled Python. The original figure assets and aggregate audit
data are retained alongside this script so the manuscript remains editable.
"""
import json
from io import BytesIO
import math
import re
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from pypdf import PdfWriter
from pypdf.generic import NameObject
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon, Image as GraphicImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, StyleSheet1
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table,
    TableStyle, Image, KeepTogether, CondPageBreak, PageBreak,
)

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / 'manuscript_data.json').read_text(encoding='utf-8'))
OUT = ROOT.parent / 'output/pdf/AutoDrive_Research_Paper_Final.pdf'
OUT.parent.mkdir(parents=True, exist_ok=True)
TITLE = 'AutoDrive : Real-Time Autonomous Driving System Using Computer Vision and Deep Learning'
NAVY = colors.HexColor('#17324F')
TEAL = colors.HexColor('#087F8C')
BLUE = colors.HexColor('#286FBD')
INK = colors.HexColor('#172B40')
SLATE = colors.HexColor('#50657A')
PALE = colors.HexColor('#EDF4F8')
RULE = colors.HexColor('#CAD8E3')
AMBER = colors.HexColor('#A66017')

for name, filename in [('Body','times.ttf'),('Body-Bold','timesbd.ttf'),
                       ('Body-Italic','timesi.ttf'),('Body-BoldItalic','timesbi.ttf'),
                       ('Sans','arial.ttf'),('Sans-Bold','arialbd.ttf'),
                       ('Sans-Italic','ariali.ttf'),('Sans-BoldItalic','arialbi.ttf')]:
    pdfmetrics.registerFont(TTFont(name, str(Path('C:/Windows/Fonts') / filename)))
pdfmetrics.registerFontFamily('Body', normal='Body', bold='Body-Bold', italic='Body-Italic', boldItalic='Body-BoldItalic')
pdfmetrics.registerFontFamily('Sans', normal='Sans', bold='Sans-Bold', italic='Sans-Italic', boldItalic='Sans-BoldItalic')

W, H = A4
M = 18 * mm
CW = W - 2*M
styles = StyleSheet1()
style_args = {
    'Body': dict(fontName='Body', fontSize=10.2, leading=13.8, alignment=TA_JUSTIFY, spaceAfter=6.5),
    'Title': dict(fontName='Sans-Bold', fontSize=22.5, leading=27, textColor=NAVY, spaceAfter=12),
    'Eyebrow': dict(fontName='Sans-Bold', fontSize=8.0, leading=10.5, textColor=TEAL, spaceAfter=8),
    'H1': dict(fontName='Sans-Bold', fontSize=13, leading=17, textColor=NAVY, spaceBefore=12, spaceAfter=7, keepWithNext=True),
    'H2': dict(fontName='Sans-Bold', fontSize=10.2, leading=14, textColor=TEAL, spaceBefore=9, spaceAfter=5, keepWithNext=True),
    'Abstract': dict(fontName='Body', fontSize=9.6, leading=12.8, alignment=TA_JUSTIFY, spaceAfter=5),
    'AbstractLabel': dict(fontName='Sans-Bold', fontSize=9, leading=12, textColor=TEAL, spaceAfter=6),
    'Small': dict(fontName='Sans', fontSize=8, leading=10.5, textColor=SLATE, spaceAfter=6),
    'FigureCaption': dict(fontName='Body-Italic', fontSize=8.8, leading=11.4, alignment=TA_LEFT, spaceBefore=6, spaceAfter=10),
    'TableCaption': dict(fontName='Body', fontSize=9, leading=11.5, textColor=INK, spaceBefore=5, spaceAfter=5, keepWithNext=True),
    'Cell': dict(fontName='Sans', fontSize=8.0, leading=10.6, alignment=TA_LEFT, textColor=INK),
    'CellHead': dict(fontName='Sans-Bold', fontSize=8.0, leading=10.6, alignment=TA_LEFT, textColor=NAVY),
    'TinyCell': dict(fontName='Sans', fontSize=7.3, leading=9.4, alignment=TA_CENTER, textColor=INK),
    'TinyHead': dict(fontName='Sans-Bold', fontSize=7.3, leading=9.4, alignment=TA_CENTER, textColor=NAVY),
    'Equation': dict(fontName='Body', fontSize=10.4, leading=16, alignment=TA_CENTER, spaceBefore=5, spaceAfter=8),
    'Author': dict(fontName='Sans-Bold', fontSize=9.3, leading=12, alignment=TA_CENTER, textColor=NAVY),
    'Email': dict(fontName='Sans', fontSize=7.1, leading=10, alignment=TA_CENTER, textColor=TEAL),
    'Team': dict(fontName='Sans', fontSize=8.5, leading=12, alignment=TA_CENTER, textColor=SLATE, spaceAfter=10),
    'Reference': dict(fontName='Body', fontSize=8.5, leading=10.7, leftIndent=19, firstLineIndent=-19, spaceAfter=4.5),
}
for name, kw in style_args.items():
    kw.setdefault('textColor', INK)
    kw.setdefault('allowWidows', 0)
    kw.setdefault('allowOrphans', 0)
    styles.add(ParagraphStyle(name, **kw))

used_refs = set()
old_ids = [n for n in range(1,37) if n != 29]
ref_map = {old: i+1 for i,old in enumerate(old_ids)}

def C(*ids):
    used_refs.update(ids)
    return ', '.join(f'[{ref_map[i]}]' for i in ids)

def p(text, style='Body'):
    return Paragraph(text, styles[style])

class Manuscript(BaseDocTemplate):
    def __init__(self, path):
        super().__init__(str(path), pagesize=A4, leftMargin=M, rightMargin=M,
                         topMargin=17*mm, bottomMargin=18*mm, title=TITLE,
                         author='Harsh Bhardwaj; Shrishti Singh; Rajat Kandpal',
                         subject='AutoDrive research manuscript with integrated figures',
                         pageCompression=1)
        self.addPageTemplates(PageTemplate('Article', [Frame(M,18*mm,CW,H-35*mm,
                                                            leftPadding=0,rightPadding=0,
                                                            topPadding=0,bottomPadding=0)], onPage=self.decorate))
        self.figure_locations = []
        self.current_section = ''
    def decorate(self, c, doc):
        c.saveState()
        c.setStrokeColor(RULE)
        c.setLineWidth(.5)
        if doc.page > 1:
            c.setFont('Sans-Bold',7.2)
            c.setFillColor(NAVY)
            c.drawString(M,H-10.5*mm,'AutoDrive')
            c.setFont('Sans',7.0)
            c.setFillColor(SLATE)
            c.drawRightString(W-M,H-10.5*mm,'Computer Vision and Deep Learning')
            c.line(M,H-13*mm,W-M,H-13*mm)
        c.line(M,13*mm,W-M,13*mm)
        c.setFont('Sans',7.1)
        c.setFillColor(SLATE)
        c.drawString(M,9*mm,'Harsh Bhardwaj | Shrishti Singh | Rajat Kandpal')
        c.drawRightString(W-M,9*mm,str(doc.page))
        c.restoreState()
    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph):
            text = flowable.getPlainText()
            if flowable.style.name in ('H1','H2'):
                self.current_section = text
                if flowable.style.name == 'H1':
                    key = 's' + str(self.seq.nextf('section'))
                    self.canv.bookmarkPage(key)
                    self.canv.addOutlineEntry(text,key,level=0)
            if flowable.style.name == 'FigureCaption':
                self.figure_locations.append({'caption':text,'page':self.page,'section':self.current_section})

def asset(key):
    return ROOT / DATA['assets'][key]['file']

def img(key, width=CW, maxheight=350):
    with PILImage.open(asset(key)) as im:
        iw, ih = im.size
    s = min(width/iw, maxheight/ih)
    return Image(str(asset(key)),width=iw*s,height=ih*s,hAlign='CENTER')

def pair(keys, ratios=(.49,.49), maxheight=235, labels=None):
    images = [img(k,CW*r,maxheight) for k,r in zip(keys,ratios)]
    rows = [images]
    if labels:
        rows.append([p(escape(x),'Small') for x in labels])
    t = Table(rows,colWidths=[CW*r for r in ratios],hAlign='CENTER')
    t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),1),
                          ('RIGHTPADDING',(0,0),(-1,-1),1),('TOPPADDING',(0,0),(-1,-1),1),
                          ('BOTTOMPADDING',(0,0),(-1,-1),1)]))
    return t

def figure(number, graphic, caption):
    return KeepTogether([Spacer(1,5),graphic,p(f'<b>Figure {number}.</b> {caption}','FigureCaption')])

def table(number, caption, rows, widths, tiny=False):
    st = 'TinyCell' if tiny else 'Cell'
    hd = 'TinyHead' if tiny else 'CellHead'
    cells=[[p(str(v),hd if ri==0 else st) for v in row] for ri,row in enumerate(rows)]
    t=Table(cells,colWidths=[CW*w/sum(widths) for w in widths],repeatRows=1,hAlign='CENTER')
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),PALE),('LINEABOVE',(0,0),(-1,0),.8,TEAL),
        ('LINEBELOW',(0,0),(-1,0),.5,RULE),('LINEBELOW',(0,-1),(-1,-1),.5,RULE),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),5),
        ('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),
        ('BOTTOMPADDING',(0,0),(-1,-1),5),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F8FAFC')]),
    ]))
    return KeepTogether([p(f'<b>Table {number}.</b> {caption}','TableCaption'),t,Spacer(1,7)])

def arrow(d,x1,y1,x2,y2,color=SLATE,dashed=False):
    d.add(Line(x1,y1,x2,y2,strokeColor=color,strokeWidth=1.0,
               strokeDashArray=[4,3] if dashed else None))
    a=math.atan2(y2-y1,x2-x1)
    size=5
    d.add(Polygon([x2,y2,x2-size*math.cos(a-.45),y2-size*math.sin(a-.45),
                   x2-size*math.cos(a+.45),y2-size*math.sin(a+.45)],fillColor=color,strokeColor=color))

def box(d,x,y,w,h,title,lines=(),fill=PALE,fs=7.6):
    d.add(Rect(x,y,w,h,rx=4,ry=4,fillColor=fill,strokeColor=RULE,strokeWidth=.8))
    text=[title]+list(lines)
    yy=y+h/2+(len(text)-1)*5.2
    for j,line in enumerate(text):
        d.add(String(x+w/2,yy-j*10.4,line,fontName='Sans-Bold' if j==0 else 'Sans',
                     fontSize=fs,textAnchor='middle',fillColor=NAVY))

def workflow():
    d=Drawing(CW,224)
    box(d,2,100,78,58,'Frame source',['road image','per-branch resize'])
    specs=[(166,'Object worker',['YOLO11s','boxes + classes']),
           (97,'Lane worker',['YOLO11s-seg','ribbon masks']),
           (28,'Steering worker',['66 x 200 BGR','DAVE-2 regression'])]
    for yy,title,lines in specs:
        box(d,120,yy,143,52,title,lines)
        arrow(d,80,129,120,yy+26)
        arrow(d,263,yy+26,321,129)
    box(d,321,100,86,58,'Visual HUD',['overlays +','angle display'],fill=colors.HexColor('#E6F3F1'))
    box(d,319,5,169,47,'Proposed extension',['time alignment / safety gate','planner / simulated control'],
        fill=colors.HexColor('#FCF3E9'),fs=7.2)
    arrow(d,364,100,364,52,AMBER,True)
    d.add(String(3,8,'Solid arrows: implemented inference/display path.',fontName='Sans',fontSize=7,fillColor=SLATE))
    return d

def steering_graph():
    d=Drawing(CW,158)
    xs=[2,102,202,302,402]
    upper=[('Input',['66 x 200 x 3','BGR / 255']),('Conv 1',['24 x 5 x 5, s=2','31 x 98 x 24']),
           ('Conv 2',['36 x 5 x 5, s=2','14 x 47 x 36']),('Conv 3',['48 x 5 x 5, s=2','5 x 22 x 48']),
           ('Conv 4 + 5',['64 x 3 x 3, s=1','3 x 20 -> 1 x 18'])]
    for x,(title,lines) in zip(xs,upper):
        box(d,x,101,89,48,title,lines,fs=7.2)
    for a,b in zip(xs[:-1],xs[1:]): arrow(d,a+89,125,b,125)
    lower=[('Flatten',['1,152 features']),('Dense 1',['1,164 + ReLU','dropout']),
           ('Dense 2 + 3',['100 -> 50','ReLU']),('Dense 4',['10 + ReLU','dropout']),
           ('Dense output',['1 unit','2 atan(z) radians'])]
    for x,(title,lines) in zip(xs,lower):
        box(d,x,18,89,48,title,lines,fill=colors.HexColor('#E6F3F1'),fs=7.2)
    arrow(d,446,101,446,83)
    arrow(d,446,83,46,83)
    arrow(d,46,83,46,66)
    for a,b in zip(xs[:-1],xs[1:]): arrow(d,a+89,42,b,42)
    d.add(String(CW/2,1,'Five convolutional layers and five dense layers, including the output layer.',
                 fontName='Sans',fontSize=7.6,textAnchor='middle',fillColor=SLATE))
    return d

def lane_curve():
    # Repair only the truncated source legend; the curves and axes are unchanged.
    h=CW*616/1628
    d=Drawing(CW,h)
    d.add(GraphicImage(0,0,CW,h,str(asset('lane_training'))))
    sx=CW/1628
    x,y=1425*sx,(616-225)*sx
    d.add(Rect(x,y,203*sx,220*sx,fillColor=colors.white,strokeColor=RULE,strokeWidth=.4))
    for j,(label,col) in enumerate([('box loss','#1F4E79'),('seg loss','#C55A11'),
                                   ('cls loss','#2E7D32'),('dfl loss','#7030A0')]):
        yy=y+(190-j*45)*sx
        d.add(Line(x+10*sx,yy,x+50*sx,yy,strokeColor=colors.HexColor(col),strokeWidth=1))
        d.add(String(x+60*sx,yy-7*sx,label,fontName='Body',fontSize=24*sx,fillColor=INK))
    return d

def range_plot():
    d=Drawing(CW,155)
    left,right=81,CW-18
    lo,hi=-170,80
    def xx(v): return left+(v-lo)/(hi-lo)*(right-left)
    for v in [-150,-100,-50,0,50]:
        d.add(Line(xx(v),35,xx(v),126,strokeColor=RULE,strokeWidth=.4))
        d.add(String(xx(v),21,str(v),fontName='Sans',fontSize=7.5,textAnchor='middle',fillColor=SLATE))
    for label,vals,y,col in [('Recorded',[-158.12,68.17],105,BLUE),('Predicted',[5.648,43.119],60,TEAL)]:
        d.add(String(2,y-3,label,fontName='Sans-Bold',fontSize=9,fillColor=col))
        d.add(Line(xx(vals[0]),y,xx(vals[1]),y,strokeColor=col,strokeWidth=7))
        for val in vals:
            d.add(Line(xx(val),y-8,xx(val),y+8,strokeColor=col,strokeWidth=1.5))
            d.add(String(xx(val),y+14,f'{val:+.2f}°',fontName='Sans',fontSize=8,textAnchor='middle',fillColor=col))
    d.add(String(CW/2,2,'Steering angle (degrees); systematic validation sample, n = 505',
                 fontName='Sans',fontSize=8,textAnchor='middle',fillColor=SLATE))
    return d

S=[]
def add(text,style='Body'): S.append(p(text,style))
def h1(n,title): add(f'{n}. {title}','H1')
def h2(n,title): add(f'{n} {title}','H2')

add('RESEARCH ARTICLE  /  COMPUTER VISION','Eyebrow')
add(TITLE,'Title')
authors=[('Harsh Bhardwaj','bhardwajharsh0312@gmail.com'),
         ('Shrishti Singh','shrisht2723@gmail.com'),('Rajat Kandpal','rajatkandpal12@gmail.com')]
at=Table([[p(n,'Author') for n,e in authors],[p(e,'Email') for n,e in authors]],colWidths=[CW/3]*3)
at.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),PALE),('BOX',(0,0),(-1,-1),.5,RULE),
                       ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),
                       ('LEFTPADDING',(0,0),(-1,-1),2),('RIGHTPADDING',(0,0),(-1,-1),2)]))
S += [at,Spacer(1,5)]
add('AutoDrive Team','Team')
abstract=('AutoDrive integrates road-object detection, lane segmentation, and camera-based steering-angle regression '
          'in a modular visual replay system. Its perception branches use a ten-class YOLO11s detector and a '
          'YOLO11s-seg model supervised with lane ribbons derived from TuSimple polylines; steering is estimated '
          'by a TensorFlow DAVE-2-style convolutional regressor. Project-reported object detection on 525 held-out '
          'COCO-10 images reaches 0.672 AP50 and 0.471 AP50-95. Lane validation on 726 frames yields 0.877-0.890 '
          'mask AP50 and 0.363-0.389 mask AP50-95 against the converted ribbon labels. A frame-aligned audit of '
          'the saved steering checkpoint on a systematic sample of 505 validation frames measures 22.96° MAE, '
          '33.24° RMSE, and R² = -0.8769. Its +21.00° mean error and near-zero correlation identify substantial '
          'checkpoint limitations. The architecture, data conversions, component results, and validation limits are '
          'documented together to support reproducible analysis. The implemented system provides offline inference '
          'and visualization; closed-loop control and vehicle actuation have not been evaluated.')
abt=Table([[p('ABSTRACT','AbstractLabel')],[p(abstract,'Abstract')]],colWidths=[CW])
abt.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#F2F7FA')),
                        ('LINEBEFORE',(0,0),(0,-1),2.5,TEAL),('LEFTPADDING',(0,0),(-1,-1),10),
                        ('RIGHTPADDING',(0,0),(-1,-1),10),('TOPPADDING',(0,0),(-1,-1),7),
                        ('BOTTOMPADDING',(0,0),(-1,-1),3)]))
S += [abt,Spacer(1,7)]
add('<b>Keywords:</b> autonomous-driving research; behavior cloning; steering-angle regression; YOLO11; lane segmentation; validation.','Small')

h1(1,'Introduction')
add(f'Automated-driving systems combine scene understanding with decisions about vehicle motion. Modular methods expose '
    f'intermediate representations such as object boxes and lane geometry, while learned steering models map camera '
    f'observations directly to a driving target {C(1,5)}. Early demonstration-learning systems established the feasibility '
    f'of steering prediction from visual input {C(2,3,4)}, but accurate open-loop predictions do not by themselves establish safe closed-loop behavior.')
add(f'This study examines AutoDrive, a three-branch prototype combining object detection, lane segmentation, and '
    f'steering regression. Its runtime evaluates the branches independently and displays their outputs in an OpenCV '
    f'heads-up display (HUD). The study addresses component performance and the gap between a visually integrated '
    f'demonstration and reliable steering. This distinction matters because behavior-cloning models can fail under '
    f'covariate shift and provide limited explanations for individual predictions {C(10,14,15)}.')
add('The contributions are: (i) a code-aligned description of the implemented replay architecture; (ii) documentation '
    'of COCO category selection and TuSimple centerline-to-ribbon conversion; (iii) component results and qualitative '
    'examples from the project records; and (iv) a frame-aligned audit of the steering checkpoint on a deterministic '
    'validation sample. The system is evaluated as an engineering research baseline.')

h1(2,'Related Work')
add(f'Learning-based steering evolved from ALVINN and DAVE to convolutional end-to-end models such as DAVE-2 '
    f'{C(2,3,4)}. Conditional imitation and video-based driving models extend the direct-regression formulation '
    f'{C(12,13)}. Dataset shift and compounding errors remain central evaluation concerns {C(14,15)}.')
add(f'YOLO-family detectors combine efficient one-stage inference with a common training interface {C(6,7,8,9)}. '
    f'Related perception methods include two-stage detection and instance segmentation {C(16,17,18)}. '
    f'Modern detectors use geometric and distribution-aware localization losses {C(19,20)}. '
    f'Lane methods use spatial context, instance segmentation, attention distillation, row classification, '
    f'and anchor attention {C(21,22,23,24,25)}.')
add(f'TuSimple evaluates lane points at prescribed image rows {C(11)}. The present lane experiment instead converts '
    'polylines into masks and evaluates those masks with IoU-based metrics. Its mask scores therefore measure '
    'agreement with the generated ribbons and cannot be interpreted as official TuSimple leaderboard scores.')

h1(3,'System Architecture')
add(f'Figure 1 places the three inference branches within the implemented runtime. A road frame is resized and '
    f'submitted to asynchronous object and lane workers and a TensorFlow steering worker. Cached results are drawn '
    f'into the visual HUD. TensorFlow and OpenCV support the steering graph and visualization {C(28,30)}; the '
    f'project source is publicly accessible {C(36)}.')
S.append(figure(1,workflow(),'Implemented inference and visualization path. Object boxes, lane masks, and the steering estimate '
               'are displayed together. The dashed branch denotes proposed timestamp alignment, safety supervision, and '
               'simulated control, which are not part of the evaluated implementation.'))
add('The current code does not use object or lane outputs to correct the steering estimate, and it provides no '
    'validated planner or vehicle-actuation interface. Independent workers keep model calls off the display loop, '
    'but cached detections are not associated with source-frame timestamps. Result age and inter-branch alignment '
    'therefore remain architectural limitations.')

h1(4,'Data and Preprocessing')
add('Object detection and lane segmentation use separate public datasets with separate partitions. Steering uses '
    'the local indexed driving dataset. Table 1 distinguishes the reported perception splits from the locally '
    'checked steering index; no common multimodal test split is available across all three tasks.')
S.append(table(1,'Datasets, partitions, and annotation targets.',[
    ['Task','Partitions','Targets'],
    ['Object detection','COCO-10: 8,080 training; 975 validation; 525 test images',
     '10 classes; 56,267 / 5,604 / 3,116 instances'],
    ['Lane segmentation','TuSimple: 2,900 training; 726 validation frames',
     'One lane class; 2,732 validation lane instances'],
    ['Steering regression','45,406 indexed frames: first 36,324 training; final 9,082 validation',
     'Degree-valued angles; ordered, unshuffled partition'],
], [1.0,2.6,2.3]))
h2('4.1','Object labels and class selection')
add(f'COCO-10 selects person, bicycle, car, motorcycle, bus, truck, traffic light, stop sign, dog, and cat '
    f'from COCO {C(27)}. Project records describe round-robin sampling across the selected categories, 8,000 '
    'sampled training images plus 80 background images, and disjoint validation and test subsets from val2017. '
    'Category identifiers are remapped to contiguous model classes. Bounding boxes are clipped to image bounds '
    'and normalized to YOLO coordinates. Figure 2 shows converted labels on sample images.')
h2('4.2','Lane centerline-to-ribbon conversion')
add(f'TuSimple represents each lane as sparse centerline points at fixed image rows {C(11)}. For each visible '
    'lane, the conversion removes near-duplicate points, estimates tangents and unit normals, and offsets the '
    'centerline to construct a closed ribbon. The half-width increases with image row to reflect perspective:')
add('w(y) = w<sub>0</sub> + w<sub>1</sub> y/H; &nbsp; w<sub>0</sub> = 4 px, &nbsp; w<sub>1</sub> = 8 px, &nbsp; H = 720 px.','Equation')
add('Coordinates are normalized and clipped to [0, 1]. The project reports a 2,900/726 training/validation split '
    'with seed 42 and no independent lane test partition. The width choice directly affects mask-IoU scores.')
S.append(figure(2,pair(['lane_labels','object_labels'],(.52,.46),170),
    'Label verification examples. Left: TuSimple ribbon polygons over a validation frame (green fill and red outlines). '
    'Right: COCO-10 boxes drawn from converted YOLO labels. The examples illustrate the two perception targets.'))
add('Class balancing at image-sampling time does not produce equal object-instance counts because individual '
    'images contain different numbers of objects. Figure 3 shows the resulting training-instance imbalance, '
    'which must be considered when interpreting per-class performance.')
S.append(figure(3,img('class_distribution',CW*.87,185),
    'COCO-10 training-instance distribution on a logarithmic scale. The most and least frequent classes differ '
    'by approximately 24:1 in the project records.'))
h2('4.3','Steering samples and preprocessing')
add('The local steering index contains 45,406 frame-path/angle pairs. Training labels are converted from degrees '
    'to radians. OpenCV loads BGR images; the bottom 150 image rows are retained, resized to 200 × 66 pixels, '
    'and divided by 255. The first 80% of index rows form the training subset and the final 20% form validation; '
    'the loader does not shuffle the index. The split can therefore reflect temporal or distributional ordering.')
add('The original collection protocol, dataset source, and redistribution license for the steering images could '
    'not be established from the available project records. This limits data provenance and independent replication.')

h1(5,'Models and Training')
h2('5.1','Object detection and lane segmentation')
add(f'The object branch uses YOLO11s with ten output classes; the lane branch uses YOLO11s-seg with a single '
    f'lane class {C(8,9)}. The reported model sizes are 9.42 million and 10.07 million parameters. Both start from '
    'vendor-pretrained checkpoints. Object detection is trained for 50 epochs at a 640-pixel input size and '
    'batch size 16; lane segmentation is trained for 60 epochs at the same input and batch sizes. The lane '
    'configuration uses mosaic = 0 and mask_ratio = 2 to preserve thin structures.')
h2('5.2','Implemented steering regressor')
add(f'The steering graph follows the DAVE-2 convolutional-regression pattern {C(4)} with five valid-padding '
    'convolutional layers and five dense layers including its scalar output (Figure 4). ReLU is used in the '
    'convolutional layers and four hidden dense layers. Dropout is applied after the 1,164-unit and 10-unit '
    'hidden layers. The 66 × 200 × 3 input uses BGR values scaled to [0, 1]. The graph does not perform '
    'an RGB-to-YUV conversion or a separate [-1, 1] normalization.')
S.append(figure(4,steering_graph(),'Code-aligned steering network. The flattened tensor contains 1,152 features; '
    'the first dense layer contains 1,164 units. A scalar affine output z is mapped to 2 atan(z) radians and '
    'converted to degrees at inference. Dropout follows the 1,164-unit and 10-unit hidden layers.'))
S.append(table(2,'Implemented steering graph dimensions and trainable parameters, including biases.',[
    ['Layer','Configuration and output shape','Parameters'],
    ['Input','66 × 200 × 3 BGR','-'],
    ['Conv 1','24 filters, 5 × 5, stride 2; 31 × 98 × 24','1,824'],
    ['Conv 2','36 filters, 5 × 5, stride 2; 14 × 47 × 36','21,636'],
    ['Conv 3','48 filters, 5 × 5, stride 2; 5 × 22 × 48','43,248'],
    ['Conv 4','64 filters, 3 × 3, stride 1; 3 × 20 × 64','27,712'],
    ['Conv 5','64 filters, 3 × 3, stride 1; 1 × 18 × 64','36,928'],
    ['Dense layers','1,152 → 1,164 → 100 → 50 → 10 → 1','1,464,163'],
    ['Total','All trainable weights and biases','1,595,511'],
], [1,3.6,1.1]))
h2('5.3','Steering optimization')
add(f'The training entry point uses Adam {C(31)} with a learning rate of 10<super>-4</super>, batch size 100, '
    '30 nominal epoch loops, and an MSE objective with L2 regularization coefficient 0.001. Dropout keep '
    'probability is 0.5 during training and 1.0 during inference. The inspected loader applies no image augmentation.')
add('An implementation detail affects epoch interpretation: the training loop computes batch counts from the '
    'full index, while the batch loader cycles only through the first 80% training subset. A nominal loop '
    'therefore does not equal one unique pass through the training frames. A validation loader is defined but '
    'unused by the training entry point, so validation-based checkpoint selection is not demonstrated.')

h1(6,'Experimental Design and Metrics')
add('Object detection is evaluated with precision, recall, AP50, AP75, and AP50-95. AP50-95 averages precision '
    'over IoU thresholds from 0.50 to 0.95. Lane results include box and mask metrics against the converted '
    'ribbons. They do not measure the official TuSimple point-based accuracy or false-positive/false-negative rates.')
add('For steering, let a<sub>i</sub> be the recorded angle and p<sub>i</sub> the model prediction in degrees, '
    'with error e<sub>i</sub> = p<sub>i</sub> - a<sub>i</sub>. The audit reports:')
add('MAE = (1/n) Σ |e<sub>i</sub>|; &nbsp; RMSE = √[(1/n) Σ e<sub>i</sub><super>2</super>]; &nbsp; bias = (1/n) Σ e<sub>i</sub>.','Equation')
add('R<super>2</super> = 1 - Σ e<sub>i</sub><super>2</super> / Σ (a<sub>i</sub> - mean(a))<super>2</super>.','Equation')
add('Pearson correlation and the proportion satisfying |e<sub>i</sub>| ≤ 5° complete the summary. The audit '
    'selects every 18th row of the final 9,082-row validation partition, yielding 505 frames. Predictions and '
    'labels are compared on the same frame with the preprocessing in Section 4.3, and radian outputs are '
    'converted to degrees. This is a systematic validation sample rather than a full held-out test evaluation.')
add('The detection and lane tables and learning curves below are retained from the project experiment records. '
    'Original evaluator logs were unavailable for independent re-evaluation. The steering audit '
    'is reported separately so that these evidence sources are not treated as one common end-to-end benchmark.')

h1(7,'Results')
h2('7.1','Object detection')
add('The detection learning curves in Figure 5 show rising validation AP and declining losses across 50 '
    'epochs. The recorded AP50-95 increases from 0.331 after the first epoch to approximately 0.482 at the '
    'end of training, with a reported peak near 0.484. Final-checkpoint curves and best-checkpoint evaluation '
    'may differ; Table 3 records the separately reported evaluation values.')
S.append(figure(5,img('object_training',CW,220),'COCO-10 detector learning curves over 50 epochs. '
    'Left: validation AP50-95, AP50, precision, and recall. Right: box, classification, and distribution-focal losses. '
    'Curves are preserved from the original project records.'))
S.append(table(3,'Project-reported object-detection results. The test partition is disjoint from validation.',[
    ['Split','Images','Instances','P','R','AP50','AP50-95','AP75'],
    ['Validation','975','5,604','0.724','0.621','0.688','0.486','0.518'],
    ['Held-out test','525','3,116','0.712','0.614','0.672','0.471','0.511'],
],[1.5,.7,.85,.6,.6,.7,.9,.7],True))
add('On the held-out test split, precision (0.712) exceeds recall (0.614), and AP50-95 is 0.471. The '
    'validation score is 0.015 higher. Class-level differences are substantial (Table 4 and Figure 6), '
    'especially for small categories and classes with few test instances.')
classes=[
    ['Person','.812','.613','.717','.457','1,567'],['Bicycle','.686','.399','.498','.270','104'],
    ['Car','.763','.607','.680','.456','642'],['Motorcycle','.802','.633','.771','.472','147'],
    ['Bus','.842','.711','.791','.662','114'],['Truck','.561','.463','.521','.331','147'],
    ['Traffic light','.636','.476','.494','.265','231'],['Stop sign','.488','.607','.619','.556','22'],
    ['Dog','.651','.806','.771','.596','72'],['Cat','.882','.829','.861','.645','70'],
]
classes=[[r[0]]+[('0'+x if x.startswith('.') else x) for x in r[1:]] for r in classes]
S.append(table(4,'Per-class held-out test results. N denotes the number of annotated test instances.',
               [['Class','P','R','AP50','AP50-95','N']]+classes,[1.8,1,1,1,1.3,1],True))
S.append(figure(6,img('class_ap',CW,220),'Per-class AP50-95 on validation (975 images) and held-out '
    'test data (525 images). The stop-sign test estimate is based on only 22 instances, so split differences '
    'should be interpreted cautiously.'))
add('The strongest reported test classes are bus (AP50-95 = 0.662), cat (0.645), and dog (0.596); '
    'traffic light (0.265) and bicycle (0.270) are weaker. Qualitative detections are shown in Figure 7. '
    'The examples illustrate model behavior and do not replace the aggregate evaluation.')
S.append(figure(7,img('object_predictions',CW*.88,375),'Object predictions on six held-out COCO-10 '
    'images at confidence ≥ 0.25 and NMS IoU = 0.7. The original examples include large vehicles, '
    'partially occluded people, smaller road users, and an animal category.'))

h2('7.2','Lane segmentation')
add('Lane training improves more rapidly in box terms than in mask terms (Figure 8). The project records '
    'report box AP50-95 increasing from 0.672 to 0.831 over 60 epochs, while mask AP50-95 rises from '
    '0.173 to 0.363. The transient early dip recovers within the initial epochs. The smaller mask score '
    'reflects the sensitivity of thin ribbons to lateral displacement and to the chosen width rule.')
S.append(figure(8,lane_curve(),'TuSimple ribbon-segmentation learning curves over 60 epochs. Left: box '
    'and mask AP50 and AP50-95. Right: box, segmentation, classification, and distribution-focal losses. '
    'Loss terms cover box regression, segmentation, classification, and distribution-focal localization.'))
S.append(table(5,'Project-reported lane validation on 726 frames and 2,732 lane instances.',[
    ['Evaluation','Box P','Box R','Box AP50','Box AP50-95','Mask P','Mask R','Mask AP50','Mask AP50-95'],
    ['End of training','0.976','0.974','0.992','0.831','0.918','0.916','0.877','0.363'],
    ['Standalone re-evaluation','0.976','0.974','0.992','0.831','0.925','0.923','0.890','0.389'],
],[1.6,.68,.68,.85,1,.68,.68,.9,1],True))
add('Standalone re-evaluation reports mask AP50 = 0.890 and AP50-95 = 0.389, compared with 0.877 and '
    '0.363 in the training-run validator. Both results are retained because the 0.026 AP50-95 difference '
    'indicates evaluation-configuration sensitivity. No independent lane test split is reported. Figure 9 '
    'shows the original validation examples under different traffic and shadow conditions.')
S.append(figure(9,img('lane_predictions',CW*.88,365),'Lane predictions on six TuSimple validation '
    'frames at confidence ≥ 0.25. Three or four lanes are detected per frame in these examples, with '
    'reported confidences approximately 0.73-0.93. Large overlapping boxes arise from diagonal, thin lane masks.'))

h2('7.3','Confusion diagnostics')
add('Figure 10 places the validation confusion matrices beside the interpretation they support. The lane '
    'record contains 2,706 matched lane instances, 119 false-positive lane predictions, and 26 missed lanes. '
    'The detector matrix identifies semantic confusion and missed objects at the evaluated confidence '
    'threshold; these validation diagnostics use a different split from the held-out detection table.')
S.append(figure(10,pair(['lane_confusion','object_confusion'],(.38,.60),275),
    'Validation confusion matrices. Left: lane-model raw counts (columns: true class; rows: predicted class). '
    'Right: detection matrix normalized by true class. Its bottom row represents missed objects and its '
    'rightmost column represents predictions on background.'))
add('The detector confuses truck with car for approximately 0.18 of true truck instances in the recorded '
    'matrix. Missed-object proportions are about 0.31 for person, 0.46 for bicycle, 0.47 for traffic light, '
    'and 0.37 for car. These patterns are consistent with lower recall for small or ambiguous categories.')

h2('7.4','Steering validation audit')
add('The checkpoint audit compares synchronous model outputs with labels on 505 selected validation '
    'frames. Table 6 reports degree-valued errors after output conversion. Negative R² and near-zero '
    'correlation indicate that this checkpoint does not provide useful steering estimates on the sampled partition.')
S.append(table(6,'Steering checkpoint audit on the systematic validation sample (n = 505).',[
    ['Metric','Result','Interpretation'],
    ['MAE','22.9616°','Mean absolute angular error'],['RMSE','33.2350°','Root mean squared angular error'],
    ['Mean signed error','+20.9977°','Strong positive prediction bias'],['R²','-0.8769','Worse than predicting the sample mean'],
    ['Pearson correlation','-0.0404','Near-zero linear association'],['Within ±5°','9.11%','Share of predictions meeting the tolerance'],
],[1.5,1,2.8]))
S.append(figure(11,range_plot(),'Recorded and predicted steering-angle ranges in the same 505-frame '
    'validation sample. Recorded angles span -158.12° to +68.17°, whereas predictions remain between '
    '+5.65° and +43.12°. The range summary illustrates the observed positive bias.'))
add('For example, a recorded -54.86° target is paired with a +16.85° prediction, and a +51.23° target '
    'with a +9.21° prediction. The aggregate results and range mismatch support a diagnostic conclusion '
    'about the current checkpoint rather than a claim of successful autonomous steering.')

h2('7.5','Latency and footprint')
add('The project manuscript records Tesla T4 per-image totals of 11.1 ms for detection and 11.8 ms for '
    'lane segmentation under batch-16 timing. Single-image totals are reported as 17.0 ms and 14.3 ms. '
    'Adding the two component times gives 22.9-31.3 ms, corresponding to a component-only sequential '
    'projection of approximately 32-44 frames/s. This projection excludes frame capture, steering, '
    'synchronization, display, and control I/O. It is not an independently reproduced end-to-end runtime benchmark.')

h1(8,'Discussion and Limitations')
add('The object and lane experiments provide component baselines, while the steering audit reveals a '
    'substantial gap between visual demonstration and reliable angle prediction. The sample has negative '
    'R², effectively zero correlation, and a strong positive bias. The implemented graph contains '
    '1,595,511 trainable parameters with a '
    '1,164-unit first dense layer. Figures 1 and 4 and Table 2 describe that implementation consistently.')
add('Several evaluation limits constrain generalization. The steering sample comes from an ordered '
    'validation partition, rather than an independently collected or session-grouped test set. Its data '
    'provenance and license are undocumented. Perception results and learning curves are project-reported '
    'records whose original evaluator logs were unavailable for independent rerun. Experiments are '
    'single-run, with no confidence intervals or systematic ablation study. The lane mask target depends '
    'on an engineered ribbon width and is distinct from the official TuSimple target.')
add('No closed-loop simulation or public-road evaluation is reported. Perception outputs are displayed '
    'without temporal alignment and do not supervise steering. Consequently, component accuracy and '
    'latency figures cannot establish a safe autonomous-driving capability or a complete end-to-end frame rate.')

h1(9,'Proposed Architecture and Further Evaluation')
add('A stronger next iteration should first rebuild the steering partition by driving session, verify '
    'angle sign and degree/radian conventions, introduce a separate test set, and select checkpoints '
    'using validation loss. The runtime should attach frame identifiers and timestamps to every result, '
    'reject stale outputs, and evaluate a confidence-aware supervisory layer before planning is introduced. '
    'The dashed extension in Figure 1 denotes this proposed work rather than an evaluated feature.')
add(f'Closed-loop evaluation can then be conducted in a simulator such as CARLA {C(26)}, with lane '
    'departure, collision, and intervention measures in addition to open-loop angular errors. '
    f'Grad-CAM-style analysis may help inspect steering-network attention {C(32)}, while datasets '
    f'such as BDD100K, KITTI, and nuScenes can support broader scene and domain coverage {C(33,34,35)}. '
    'These are experimental directions; no such additional results are claimed here.')

h1(10,'Conclusion')
add('AutoDrive brings object detection, lane segmentation, and steering-angle regression into a modular '
    'visual replay system. The reported detection and lane results demonstrate component-level '
    'perception performance within their dataset-specific protocols. The steering audit measures '
    '22.96° MAE, 33.24° RMSE, and R² = -0.8769 on a systematic sample of 505 validation frames, '
    'identifying a checkpoint that requires further development. The code-aligned architecture '
    'and explicit evaluation boundaries support a reproducible research account '
    'and define the data, training, and closed-loop validation work needed next.')

add('Declarations','H1')
add('<b>Author contributions.</b> Harsh Bhardwaj, Shrishti Singh, and Rajat Kandpal contributed to '
    'conceptualization, implementation, validation, analysis, and manuscript preparation. All three '
    'authors reviewed and approved the final manuscript.')
add('<b>Funding.</b> This research received no external funding.')
add('<b>Competing interests.</b> The authors declare no competing interests.')
add('<b>Data availability.</b> COCO and TuSimple are publicly documented datasets, subject to their '
    'respective access and licensing terms. The steering dataset source, collection protocol, and '
    'redistribution terms were not established from the available project records; the steering '
    'images are not redistributed with this manuscript.')
add('<b>Code availability.</b> The AutoDrive Team project repository is available at '
    '<link href="https://github.com/HA2345567/Autodrive" color="#087F8C">'
    'https://github.com/HA2345567/Autodrive</link>. Source files describe training, inference, '
    'and visualization; large datasets and model artifacts require separate access. Repository '
    'access date: 21 October 2026.')
add('<b>Scope and safety.</b> The reported system is an offline research prototype and has not '
    'been validated for vehicle control or public-road operation.')

add('References','H1')
assert used_refs == set(old_ids), (set(old_ids)-used_refs,used_refs-set(old_ids))
for old in old_ids:
    text=DATA['references'][str(old)]
    if old==36:
        text='AutoDrive Team, "AutoDrive: Real-Time Autonomous Driving System Using Computer Vision and Deep Learning," GitHub repository. Available: https://github.com/HA2345567/Autodrive. Accessed: Oct. 21, 2026.'
    add(f'[{ref_map[old]}] '+escape(text),'Reference')

doc=Manuscript(OUT)
doc.build(S)
# ReportLab declares default base fonts for empty initial text states. All actual
# text uses embedded TrueType fonts. Point those unused default resources to the
# embedded equivalents too, so submission font checks see no unembedded fonts.
writer = PdfWriter(clone_from=BytesIO(OUT.read_bytes()))
for page in writer.pages:
    fonts = page['/Resources']['/Font'].get_object()
    embedded = {str(ref.get_object().get('/BaseFont')): ref for ref in fonts.values()}
    sans = next(ref for name,ref in embedded.items() if name.endswith('+ArialMT'))
    serif = next((ref for name,ref in embedded.items() if name.endswith('+TimesNewRomanPSMT')),sans)
    for name,ref in list(fonts.items()):
        base = str(ref.get_object().get('/BaseFont'))
        if base == '/Helvetica':
            fonts[NameObject(name)] = sans
        elif base == '/Times-Roman':
            fonts[NameObject(name)] = serif
pdf_buffer = BytesIO()
writer.write(pdf_buffer)
OUT.write_bytes(pdf_buffer.getvalue())
(ROOT/'layout_manifest.json').write_text(json.dumps({'title':TITLE,'pages':doc.page,
    'figures':doc.figure_locations,'reference_count':len(old_ids),'funding':'No external funding',
    'competing_interests':'None declared','authors':[n for n,e in authors]},indent=2),encoding='utf-8')
print(f'Built {OUT}: {doc.page} pages; {len(doc.figure_locations)} integrated figures; {len(old_ids)} references.')
for f in doc.figure_locations:
    print(f'Page {f["page"]}: {f["caption"].split(".")[0]} / {f["section"]}')
