"""Verify inputs, output invariants, and PDF page bounds; render every page."""
from pathlib import Path
import hashlib
import json
import unittest
import numpy as np
import pandas as pd
import fitz
from PIL import Image, ImageDraw
from project_config import ROOT, OUTPUT


def main():
    manifest=json.loads((OUTPUT/'input_data_sha256.json').read_text())
    tests=unittest.TextTestRunner(verbosity=0).run(unittest.defaultTestLoader.loadTestsFromName('test_project'))
    assert tests.wasSuccessful(), 'behavioral tests failed'
    changed=[name for name,sha in manifest.items() if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha]
    assert not changed, changed
    checks=[]
    for label in ['corrected','sensitivity70','coverage30']:
        metrics=pd.read_csv(OUTPUT/f'{label}_metrics.csv')
        fits=json.loads((OUTPUT/f'{label}_fits.json').read_text())
        assert len(metrics)==18 and len(fits)==9
        for ticker,record in fits.items():
            g,gx=record['garch'],record['garch_x']
            assert gx['nll'] <= g['nll'] + 1e-5, (label,ticker,'nesting objective')
            for name,fit in record.items():
                p=fit['params']
                assert p[2]>=-1e-9 and p[3]>=-1e-9 and p[2]+p[3]<=.999+1e-8
                if name=='garch_x':assert p[1]-abs(p[4])>=1e-6-1e-9
            forecasts=pd.read_csv(OUTPUT/f'{label}_{ticker}_forecasts.csv',index_col=0,parse_dates=True)
            assert np.isfinite(forecasts[['realized_proxy','garch','garch_x','ewma','constant']]).all().all()
            assert (forecasts[['garch','garch_x','ewma','constant']]>0).all().all()
            assert forecasts.index.is_monotonic_increasing and not forecasts.index.has_duplicates
            origins=pd.to_datetime(forecasts.forecast_origin,utc=True)
            assert (origins.dt.tz_localize(None).dt.normalize()<forecasts.index).all()
            for row in metrics.loc[metrics.ticker.eq(ticker)].itertuples():
                assert row.n_test==len(forecasts)
        checks.append({'label':label,'stocks':len(fits),'comparisons':len(metrics)})
    source=(OUTPUT/'volatility_project_handout.md').read_text(encoding='utf-8')
    assert '{{' not in source
    document=fitz.open(OUTPUT/'volatility_project_report.pdf')
    directory=ROOT/'tmp/report_qa'; directory.mkdir(parents=True,exist_ok=True)
    bounds=[]
    page_words=[]
    for i,page in enumerate(document):
        page.get_pixmap(matrix=fitz.Matrix(1.35,1.35),alpha=False).save(directory/f'page-{i+1:02d}.png')
        page_words.append(len(page.get_text().split()))
        for block in page.get_text('dict')['blocks']:
            x0,y0,x1,y1=block['bbox']
            if x0<45 or x1>page.rect.width-45 or y0<15 or y1>page.rect.height-15:
                bounds.append({'page':i+1,'bbox':block['bbox']})
    assert not bounds,bounds
    for start in range(0,len(document),4):
        canvas=Image.new('RGB',(1200,1740),'#d5d5d5'); draw=ImageDraw.Draw(canvas)
        for j in range(start,min(start+4,len(document))):
            im=Image.open(directory/f'page-{j+1:02d}.png'); im.thumbnail((580,830))
            x=(j-start)%2*600+10; y=(j-start)//2*870+25
            canvas.paste(im,(x,y));draw.text((x,y-18),f'Page {j+1}',fill='black')
        canvas.save(directory/f'final-contact-{start//4+1:02d}.png')
    result={'pdf_pages':len(document),'pdf_text_words':sum(page_words),'pdf_text_bounds_issues':bounds,
            'page_text_word_counts':page_words,'original_csv_files_checked':len(manifest),
            'changed_original_csvs':changed,'experiment_invariant_checks':checks,
            'behavioral_tests':{'count':tests.testsRun,'result':'passed'},
            'page_renders':str(directory.relative_to(ROOT))}
    (OUTPUT/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
