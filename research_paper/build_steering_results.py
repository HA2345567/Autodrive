"""Create a one-page, steering-only summary from the retained audit results."""
import json
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer

ROOT=Path(__file__).resolve().parent
audit=json.loads((ROOT/'manuscript_data.json').read_text(encoding='utf-8'))['audit']
OUT=ROOT.parent/'output/pdf/AutoDrive_Steering_Prediction_Results.pdf'
NAVY=colors.HexColor('#17324F')
TEAL=colors.HexColor('#087F8C')
SLATE=colors.HexColor('#50657A')
RULE=colors.HexColor('#CAD8E3')
for name,file in [('ResultsSans','arial.ttf'),('ResultsSans-Bold','arialbd.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(Path('C:/Windows/Fonts')/file)))
pdfmetrics.registerFontFamily('ResultsSans',normal='ResultsSans',bold='ResultsSans-Bold')
body=ParagraphStyle('Body',fontName='ResultsSans',fontSize=9.5,leading=14,textColor=NAVY,spaceAfter=10)
small=ParagraphStyle('Small',parent=body,fontSize=8.4,leading=12,textColor=SLATE)
heading=ParagraphStyle('Heading',parent=body,fontName='ResultsSans-Bold',fontSize=12,leading=16,textColor=TEAL,spaceBefore=10)
title=ParagraphStyle('Title',parent=body,fontName='ResultsSans-Bold',fontSize=23,leading=28,spaceAfter=8)
doc=SimpleDocTemplate(str(OUT),pagesize=A4,leftMargin=48,rightMargin=48,topMargin=46,bottomMargin=48,
    title='AutoDrive - Steering-Angle Prediction Results',
    author='Harsh Bhardwaj; Shrishti Singh; Rajat Kandpal')
P=lambda t,st=body:Paragraph(t,st)

def result_table(rows,widths):
    t=Table([[P(str(x)) for x in row] for row in rows],colWidths=widths,hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#EDF4F8')),
        ('LINEABOVE',(0,0),(-1,0),1,TEAL),('LINEBELOW',(0,-1),(-1,-1),.6,RULE),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F8FAFC')]),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),7),
        ('BOTTOMPADDING',(0,0),(-1,-1),7),('LEFTPADDING',(0,0),(-1,-1),9),
        ('RIGHTPADDING',(0,0),(-1,-1),9),
    ]))
    return t

story=[P('AutoDrive',heading),P('Steering-Angle Prediction Results',title),
       P('AutoDrive Team | Harsh Bhardwaj, Shrishti Singh, and Rajat Kandpal',small),
       Spacer(1,7),P('Validation sample',heading),
       P(f'{audit["n"]} frame-aligned predictions, selected every {audit["stride"]}th row of the '
         f'{audit["validation_partition"]:,}-frame validation partition. Angular errors are reported in degrees; '
         'R² and Pearson correlation are dimensionless.'),
       result_table([
           ['<b>Metric</b>','<b>Result</b>'],
           ['Mean absolute error (MAE)',f'{audit["mae_deg"]:.4f}°'],
           ['Root mean squared error (RMSE)',f'{audit["rmse_deg"]:.4f}°'],
           ['Mean signed error / bias',f'{audit["bias_deg"]:+.4f}°'],
           ['Coefficient of determination (R²)',f'{audit["r_squared"]:.4f}'],
           ['Pearson correlation',f'{audit["pearson_r"]:.4f}'],
           ['Predictions within ±5°',f'{audit["within_5_deg_pct"]:.2f}%'],
       ],[doc.width*.70,doc.width*.30]),
       Spacer(1,9),P('Angle ranges in the same sample',heading),
       result_table([
           ['<b>Quantity</b>','<b>Range</b>'],
           ['Recorded steering angle','-158.12° to +68.17°'],
           ['Predicted steering angle','+5.648° to +43.119°'],
       ],[doc.width*.53,doc.width*.47]),Spacer(1,11),
       P('<b>Interpretation.</b> The checkpoint shows poor performance on this sample, with strong positive bias '
         'and near-zero correlation. Its negative R² indicates performance below the sample-mean prediction baseline.'),
       P('Reporting scope: systematic validation sample only. These metrics do not describe a complete independent '
         'test set or closed-loop vehicle performance.',small)]

def footer(c,d):
    c.saveState()
    c.setStrokeColor(RULE)
    c.line(48,35,A4[0]-48,35)
    c.setFont('ResultsSans',7.5)
    c.setFillColor(SLATE)
    c.drawString(48,23,'AutoDrive | Steering prediction results')
    c.drawRightString(A4[0]-48,23,'1')
    c.restoreState()

def make_canvas(*a,**kw):
    kw['initialFontName']='ResultsSans'
    return canvas.Canvas(*a,**kw)

doc.build(story,onFirstPage=footer,onLaterPages=footer,canvasmaker=make_canvas)
print(OUT)
