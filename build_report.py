"""Generate the public PDF and resolved Markdown from audited experiment outputs."""
from pathlib import Path
import hashlib
import html
import json
import re
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer,
                               Image, Table, TableStyle, PageBreak, Preformatted, KeepTogether)
from reportlab.platypus.tableofcontents import TableOfContents
from project_config import ROOT, OUTPUT


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                       "| " + " | ".join(["---"]*len(headers)) + " |"] +
                      ["| " + " | ".join(map(str,row)) + " |" for row in rows])


def fmt(value, digits=5):
    if not np.isfinite(value):
        return "undefined"
    return f"{value:.{digits}f}" if abs(value)>=1e-4 or value==0 else f"{value:.2e}"


def resolved_markdown():
    original=pd.read_csv(OUTPUT/"original_metrics.csv")
    corrected=pd.read_csv(OUTPUT/"corrected_metrics.csv")
    sensitivity=pd.read_csv(OUTPUT/"sensitivity70_metrics.csv")
    screened=pd.read_csv(OUTPUT/"coverage30_metrics.csv")
    audit=pd.read_csv(OUTPUT/"data_audit.csv")
    gaps=pd.read_csv(OUTPUT/"news_gap_audit.csv")
    fits=json.loads((OUTPUT/"corrected_fits.json").read_text())
    q=corrected.query("loss=='QLIKE'")
    orig=original.query("loss=='QLIKE'")
    subs={}
    subs['DATA_AUDIT_TABLE']=table(['Ticker','Articles','Duplicates','First article','Last article','Zero lag %'],
        [[r.ticker,f'{r.articles:,}',r.duplicate_date_headline,r.news_first[:10],r.news_last[:10],f'{r.zero_lag_pct:.1f}'] for r in audit.itertuples()])
    subs['GAP_TABLE']=table(['Ticker','Gap days','From news date','To news date'],
        [[r.ticker,r.max_news_gap_days,r.gap_start,r.gap_end] for r in gaps.itertuples()])
    subs['ORIGINAL_RESULTS_TABLE']=table(['Ticker','GARCH','GARCH X','G minus GX','Original p'],
        [[r.ticker,fmt(r.mean_garch),fmt(r.mean_garch_x),fmt(r.difference),fmt(r.p_original)] for r in orig.itertuples()])
    subs['CORRECTED_RESULTS_TABLE']=table(['Ticker','GARCH','GARCH X','EWMA','G minus GX','HAC p','Holm p'],
        [[r.ticker,fmt(r.mean_garch),fmt(r.mean_garch_x),fmt(r.mean_ewma),fmt(r.difference_g_minus_gx),fmt(r.p_exploratory),fmt(r.p_holm_exploratory)] for r in q.itertuples()])
    subs['MSE_RESULTS_TABLE']=table(['Ticker','GARCH','GARCH X','EWMA','Constant'],
        [[r.ticker,f'{r.mean_garch:.3f}',f'{r.mean_garch_x:.3f}',f'{r.mean_ewma:.3f}',f'{r.mean_constant:.3f}'] for r in corrected.query("loss=='MSE'").itertuples()])
    s=sensitivity.query("loss=='QLIKE'").set_index('ticker')
    c=screened.query("loss=='QLIKE'").set_index('ticker')
    subs['SENSITIVITY_TABLE']=table(['Ticker','Main diff','70 percent diff','Coverage diff','Coverage test N'],
        [[r.ticker,fmt(r.difference_g_minus_gx),fmt(s.loc[r.ticker,'difference_g_minus_gx']),fmt(c.loc[r.ticker,'difference_g_minus_gx']),int(c.loc[r.ticker,'n_test'])] for r in q.itertuples()])
    subs['PARAMETER_TABLE']=table(['Ticker','mu','omega','alpha','beta','gamma','a plus b','Boundary'],
        [[ticker]+[f'{v:.5f}' for v in record['garch_x']['params']]+[f"{sum(record['garch_x']['params'][2:4]):.5f}",str(record['garch_x']['boundary'])] for ticker,record in fits.items()])
    subs['DIAGNOSTICS_TABLE']=table(['Ticker','LB z p','LB z squared p','ARCH LM p','Minimum h'],
        [[ticker]+[fmt(record['garch_x'][k]) for k in ['lb_z_p10','lb_z2_p10','arch_lm_p10','min_variance']] for ticker,record in fits.items()])
    for name,data in [('ORIGINAL',orig),('CORRECTED',q)]:
        winners=data.loc[data.mean_garch_x<data.mean_garch,'ticker'].tolist()
        subs[name+'_WINNERS_NOTE']=f"Verified QLIKE count: {len(winners)} of {len(data)} stocks favor GARCH-X on average. The stocks are {', '.join(winners)}. This count describes average loss, not a count of statistically validated discoveries."
    content=(ROOT/'docs/report_template.md').read_text(encoding='utf-8')
    for name,value in subs.items():
        content=content.replace('{{'+name+'}}',value)
    if '{{' in content:
        raise ValueError('Unresolved report placeholder')
    (OUTPUT/'volatility_project_handout.md').write_text(content,encoding='utf-8')
    return content


def figures():
    directory=OUTPUT/'figures'; directory.mkdir(exist_ok=True)
    main=pd.read_csv(OUTPUT/'corrected_metrics.csv').query("loss=='QLIKE'").set_index('ticker')
    alternate=pd.read_csv(OUTPUT/'sensitivity70_metrics.csv').query("loss=='QLIKE'").set_index('ticker')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(8.1,4.2),layout='constrained')
    x=np.arange(len(main)); width=.36
    ax.bar(x-width/2,main.difference_g_minus_gx,width,label='80% training',color='#24465b')
    ax.bar(x+width/2,alternate.loc[main.index,'difference_g_minus_gx'],width,label='70% training',color='#9eb6c1')
    ax.axhline(0,color='black',linewidth=.8)
    ax.set(xticks=x,xticklabels=main.index,ylabel='Mean QLIKE loss of GARCH minus GARCH X')
    ax.legend(frameon=False,loc='lower right'); ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
    fig.savefig(directory/'qlike_differences.png',dpi=210); plt.close(fig)
    data=pd.read_csv(OUTPUT/'corrected_NVDA_forecasts.csv',index_col=0,parse_dates=True)
    fig,(ax,proxy)=plt.subplots(2,1,figsize=(8.1,4.7),layout='constrained',sharex=True,
                               gridspec_kw={'height_ratios':[2,1]})
    ax.plot(data.index,data.garch,color='#1b384a',lw=1.7,label='GARCH')
    ax.plot(data.index,data.garch_x,color='#b25e29',lw=1.3,label='GARCH X')
    ax.plot(data.index,data.ewma,color='#8ca6af',lw=1,label='EWMA')
    ax.set(ylabel='Forecast variance')
    ax.legend(frameon=False,loc='upper right',ncol=3); ax.grid(axis='y',alpha=.2)
    proxy.plot(data.index,data.realized_proxy,color='#84949b',lw=.8)
    proxy.set(ylabel='Squared residual',xlabel='Target return date')
    proxy.grid(axis='y',alpha=.2)
    fig.savefig(directory/'nvda_forecasts.png',dpi=210); plt.close(fig)


class ReportDoc(BaseDocTemplate):
    def afterFlowable(self,flowable):
        if isinstance(flowable,Paragraph) and flowable.style.name=='Chapter':
            key='section_'+str(self.seq.nextf('chapter'))
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(flowable.getPlainText(),key,level=0,closed=False)
            self.notify('TOCEntry',(0,flowable.getPlainText(),self.page,key))


def register_fonts():
    """Use Calibri when available (Windows); otherwise alias ReportLab's built-in Helvetica."""
    font_dir=Path('C:/Windows/Fonts')
    files={'Body':'calibri.ttf','BodyBold':'calibrib.ttf','BodyItalic':'calibrii.ttf','BodyBoldItalic':'calibriz.ttf'}
    if all((font_dir/f).exists() for f in files.values()):
        for name,file in files.items():
            pdfmetrics.registerFont(TTFont(name,str(font_dir/file)))
        pdfmetrics.registerFontFamily('Body',normal='Body',bold='BodyBold',italic='BodyItalic',boldItalic='BodyBoldItalic')
        return
    from reportlab.lib.fonts import addMapping
    builtin={'Body':'Helvetica','BodyBold':'Helvetica-Bold','BodyItalic':'Helvetica-Oblique','BodyBoldItalic':'Helvetica-BoldOblique'}
    for alias,base in builtin.items():
        pdfmetrics.registerFont(pdfmetrics.Font(alias,base,'WinAnsiEncoding'))
    for bold,italic,alias in [(0,0,'Body'),(1,0,'BodyBold'),(0,1,'BodyItalic'),(1,1,'BodyBoldItalic')]:
        addMapping('Body',bold,italic,alias)


def pdf(content):
    register_fonts()
    styles={
        'body':ParagraphStyle('Body',fontName='Body',fontSize=10.5,leading=14.1,spaceAfter=7.5),
        'title':ParagraphStyle('Title',fontName='BodyBold',fontSize=27,leading=31,spaceAfter=18,textColor=colors.black),
        'subtitle':ParagraphStyle('Subtitle',fontName='Body',fontSize=17,leading=22,spaceAfter=18,textColor=colors.black),
        'chapter':ParagraphStyle('Chapter',fontName='BodyBold',fontSize=17,leading=21,spaceBefore=19,spaceAfter=10,keepWithNext=True,textColor=colors.black),
        'sub':ParagraphStyle('Sub',fontName='BodyBold',fontSize=12.3,leading=16,spaceBefore=13,spaceAfter=7,keepWithNext=True,textColor=colors.black),
        'caption':ParagraphStyle('Caption',fontName='BodyItalic',fontSize=9,leading=12,spaceBefore=5,spaceAfter=12),
        'cell':ParagraphStyle('Cell',fontName='Body',fontSize=8.6,leading=11,alignment=TA_CENTER),
        'cellhead':ParagraphStyle('Cellhead',fontName='BodyBold',fontSize=8.5,leading=11,alignment=TA_CENTER,textColor=colors.white),
        'code':ParagraphStyle('Code',fontName='Courier',fontSize=8.0,leading=11,spaceBefore=5,spaceAfter=10),
    }
    page_w,page_h=A4; width=page_w-104
    path=OUTPUT/'volatility_project_report.pdf'
    doc=ReportDoc(str(path),pagesize=A4,leftMargin=52,rightMargin=52,topMargin=52,bottomMargin=48,
                  title='News sentiment and daily equity volatility forecasting',author='Dhruv Natu',
                  subject='Project theory implementation audit and empirical results')
    def page(canvas,document):
        canvas.saveState(); canvas.setFillColor(colors.black)
        if document.page>1:
            canvas.setFont('Body',8)
            canvas.drawString(52,page_h-29,'News sentiment and daily equity volatility forecasting')
        canvas.setFont('Body',8)
        canvas.drawString(52,27,'21 October 2026')
        canvas.drawRightString(page_w-52,27,str(document.page))
        canvas.restoreState()
    doc.addPageTemplates(PageTemplate(id='main',frames=[Frame(52,48,width,page_h-100,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],onPage=page))
    def markup(text):
        text=html.escape(text)
        # Auto-link references without leaking local workspace paths.
        text=re.sub(r'(https?://[^\s]+)',lambda m:f'<link href="{m.group(1)}" color="#24465b">{m.group(1)}</link>',text)
        return text
    story=[]; lines=content.splitlines(); i=0; toc_added=False
    eqdir=ROOT/'tmp/report_equations'; eqdir.mkdir(parents=True,exist_ok=True)
    while i<len(lines):
        line=lines[i].strip()
        if not line:
            i+=1; continue
        if line.startswith('# '):
            story += [Spacer(1,18),Paragraph(markup(line[2:]),styles['title'])]; i+=1;continue
        if line=='## A mathematical and implementation handout':
            story.append(Paragraph('A mathematical and implementation handout',styles['subtitle']));i+=1;continue
        if line=='## How to use this handout' and not toc_added:
            story.append(PageBreak()); story.append(Paragraph('Contents',styles['subtitle']))
            toc=TableOfContents()
            toc.levelStyles=[ParagraphStyle('TOC',fontName='Body',fontSize=9.5,leading=12,spaceBefore=1,spaceAfter=1,leftIndent=0,firstLineIndent=0)]
            story.append(toc);story.append(PageBreak());toc_added=True
        if line.startswith('## '):
            story.append(Paragraph(markup(line[3:]),styles['chapter']));i+=1;continue
        if line.startswith('### '):
            story.append(Paragraph(markup(line[4:]),styles['sub']));i+=1;continue
        if line.startswith('$$'):
            expression=line[2:-2].strip()
            digest=hashlib.sha256(expression.encode()).hexdigest()[:16]
            equation=eqdir/f'{digest}.png'
            if not equation.exists():
                fig=plt.figure(figsize=(.1,.1)); fig.text(0,0,'$'+expression+'$',fontsize=13,color='black')
                fig.savefig(equation,dpi=210,bbox_inches='tight',pad_inches=.08,transparent=False,facecolor='white');plt.close(fig)
            from PIL import Image as PILImage
            with PILImage.open(equation) as im: w,h=im.size
            pw=w/210*72; ph=h/210*72
            factor=min(1,width/pw)
            image=Image(str(equation),pw*factor,ph*factor)
            image.hAlign='CENTER';story += [Spacer(1,3),image,Spacer(1,10)];i+=1;continue
        if line.startswith('```'):
            i+=1; code=[]
            while i<len(lines) and not lines[i].startswith('```'):
                code.extend(textwrap.wrap(lines[i],width=91,replace_whitespace=False,drop_whitespace=False) or [''])
                i+=1
            story.append(Preformatted('\n'.join(code),styles['code']));i+=1;continue
        if line.startswith('!['):
            match=re.match(r'!\[(.*?)\]\((.*?)\)',line)
            source=OUTPUT/match.group(2)
            from PIL import Image as PILImage
            with PILImage.open(source) as im:w,h=im.size
            story.append(Image(str(source),width,width*h/w));i+=1;continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                values=[v.strip() for v in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[-: ]+',v) for v in values): rows.append(values)
                i+=1
            count=len(rows[0]); colwidths=[width/count]*count
            if count>=5:
                colwidths=[43]+[(width-43)/(count-1)]*(count-1)
            if rows[0][0]=='Symbol':colwidths=[70,235,width-305]
            cells=[[Paragraph(markup(v),styles['cellhead' if j==0 else 'cell']) for v in row] for j,row in enumerate(rows)]
            tab=Table(cells,colWidths=colwidths,repeatRows=1,hAlign='LEFT')
            tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#24465b')),
                                      ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f2f5f7')]),
                                      ('GRID',(0,0),(-1,-1),.4,colors.HexColor('#D9D9D9')),
                                      ('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),5),
                                      ('BOTTOMPADDING',(0,0),(-1,-1),5),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5)]))
            story += [Spacer(1,5),tab,Spacer(1,5)];continue
        paragraph=[line];i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','$$','|','```','![')):
            paragraph.append(lines[i].strip());i+=1
        text=' '.join(paragraph)
        style=styles['caption'] if re.match(r'^(Table|Figure) \d+\.',text) else styles['body']
        story.append(Paragraph(markup(text),style))
    doc.multiBuild(story)
    print(path,flush=True)


def main():
    figures()
    content=resolved_markdown()
    pdf(content)


if __name__=='__main__':
    main()
