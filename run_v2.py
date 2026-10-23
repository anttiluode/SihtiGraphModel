"""SihtiGraphModel v2 — cached and held-out black-box graph distillation.

Examples:
    python run_v2.py --mode synthetic
    python run_v2.py --mode legacy
    python run_v2.py --mode cifar --device cuda
No mode trains or modifies the v1 CIFAR checkpoint.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
from html import escape

import numpy as np

from sihtigraph.v2 import (choose_graph, degree12_moments, degree3_moments,
                           fit_graph, graph_predict, local_allowed,
                           ranked_candidates, summary, synthetic_40)
from sihtigraph.walsh import km_search, metrics, random_masks


STAGES = {
    'selection': (101, .5), 'fit': (202, .5), 'validation': (303, .5),
    'test_p50': (404, .5), 'test_p75': (505, .75), 'test_p90': (606, .9),
}


def native(obj):
    if isinstance(obj, np.integer):return int(obj)
    if isinstance(obj, np.floating):return float(obj)
    if isinstance(obj, np.bool_):return bool(obj)
    if isinstance(obj, np.ndarray):return obj.tolist()
    raise TypeError(f'Unexpected JSON type {type(obj).__name__}')


def write_json(path: Path, data: dict):
    payload=json.dumps(data,indent=2,allow_nan=False,default=native)+'\n'
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(payload,encoding='utf-8')
    os.replace(tmp,path)


def file_hash(path: Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def stage_masks(seed: int, n: int, count: int, stage: str) -> np.ndarray:
    offset, keep = STAGES[stage]
    return random_masks(np.random.default_rng(seed+offset),count,n,keep)


class StageCache:
    """Persist teacher queries per split. Content-keyed by checkpoint and controls."""
    def __init__(self, out: Path, signature: dict):
        self.signature=signature
        key=hashlib.sha256(json.dumps(signature,sort_keys=True).encode('utf-8')).hexdigest()[:20]
        self.dir=out/'cache'/key
        self.dir.mkdir(parents=True,exist_ok=True)
        self.loaded=0
        self.queried=0
    def read(self,stage: str,nrows: int,seed:int,oracle):
        path=self.dir/f'{stage}.npz'
        z=stage_masks(seed,64,nrows,stage)
        if path.exists():
            with np.load(path,allow_pickle=False) as f:
                cached_z=f['z']
                y=f['y']
                clipped=int(f['clipped_pixels']) if 'clipped_pixels' in f.files else None
                total=int(f['total_pixels']) if 'total_pixels' in f.files else None
            if cached_z.shape != z.shape or not np.array_equal(cached_z,z) or y.shape!=(nrows,) or not np.isfinite(y).all():
                raise ValueError(f'{path}: cached responses disagree with current masks; use a distinct --out')
            if hasattr(oracle,'clipped_pixels'):
                if clipped is None or total is None:
                    raise ValueError(f'{path}: cache lacks clipping counters; use a fresh --out')
                oracle.clipped_pixels+=clipped
                oracle.total_pixels+=total
            self.loaded+=nrows
            print(f'  {stage}: loaded {nrows:,} cached teacher responses',flush=True)
            return z,np.asarray(y,dtype=np.float64)
        before_clip=int(getattr(oracle,'clipped_pixels',0))
        before_total=int(getattr(oracle,'total_pixels',0))
        y=np.asarray(oracle(z),dtype=np.float64).reshape(-1)
        if y.shape != (nrows,) or not np.isfinite(y).all():
            raise ValueError(f'{stage}: nonfinite or malformed teacher responses')
        delta_clip=int(getattr(oracle,'clipped_pixels',0))-before_clip
        delta_total=int(getattr(oracle,'total_pixels',0))-before_total
        tmp=path.with_name(path.name+'.tmp')
        with tmp.open('wb') as fp:
            np.savez_compressed(fp,z=z,y=y,clipped_pixels=delta_clip,total_pixels=delta_total)
        os.replace(tmp,path)
        self.queried+=nrows
        print(f'  {stage}: queried and cached {nrows:,} teacher responses',flush=True)
        return z,y


def pretty(t):
    return {'indices':[int(i) for i in t['indices']],
            'names':[f'B{i//16}:r{i%16//4}c{i%4}' for i in t['indices']],
            'coefficient':float(t['coefficient'])}


def html_report(data:dict,out:Path):
    panels=[]
    for replacement,result in data['distillation'].items():
        measurements=result['tests']
        def val(x):
            return 'n/a' if x is None else f'{x:.4f}'
        rows=''.join('<tr><th>'+escape(k)+'</th>'+
            ''.join('<td>'+val(measurements[k][method]['r2'])+'</td>'
                    for method in ('global50','local50','unary64','random50','constant'))+'</tr>'
            for k in ('p=0.5','p=0.75','p=0.9'))
        branches=''.join('<tr><td>'+str(i+1)+'</td><td>'+
            escape(' + '.join(t['names']))+'</td><td>'+val(t['coefficient'])+
            '</td></tr>' for i,t in enumerate(result['graphs']['global50']['terms'][:50]))
        weights=[0.]*64
        for t in result['graphs']['global50']['terms']:
            for idx in t['indices']:weights[idx]+=abs(t['coefficient'])
        maximum=max(weights) or 1.
        rects=[]
        for b in range(4):
            for i in range(16):
                v=weights[b*16+i]/maximum
                x=12+b*138+(i%4)*29
                y=20+(i//4)*29
                rects.append(f'<rect x="{x}" y="{y}" width="26" height="26" rx="3" fill="#089785" opacity="{.08+.86*v:.3f}"/>')
            rects.append(f'<text x="{12+b*138}" y="154" font-size="12">band {b}</text>')
        panels.append(f'''<section><h2>{escape(replacement)} replacement</h2>
        <p>Teacher queries this invocation: {result['cache']['fresh']:,}; responses loaded from cache: {result['cache']['loaded']:,}; clipped pixel fraction {result['teacher_oracle']['fraction_pixels_clipped']:.3%}.</p>
        <p>Global selected {result['selection']['global50']['count']} branches; local selected {result['selection']['local50']['count']}; no geometry restrictions on global candidates.</p>
        <table><tr><th>Retention</th><th>Discovered global</th><th>Given local topology</th><th>64 singletons</th><th>Random control</th><th>Constant</th></tr>{rows}</table>
        <h3>Graph connections by octave and tile</h3><svg viewBox="0 0 576 171" style="max-width:100%;height:auto">{''.join(rects)}</svg>
        <h3>Selected global graph terms</h3><table><tr><th>#</th><th>Band-patches in branch</th><th>Coefficient</th></tr>{branches}</table>
        </section>''')
    html=f'''<!DOCTYPE html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>SihtiGraphModel v2</title>
    <style>body{{max-width:1000px;margin:2rem auto;padding:0 1rem;font:15px system-ui;line-height:1.5}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border-bottom:1px solid #7776;text-align:left;padding:7px}}section{{border-top:1px solid #999;margin-top:2rem}}small{{color:#666}}</style>
    <h1>SihtiGraphModel v2 — direct spectral reader</h1>
    <p><strong>Held-out local behavioral graph on a single CIFAR image.</strong> CNN weights frozen, v1 results unchanged. This is not a recovered internal neural circuit or a general-image graph.</p>
    <p>Test image #{data['image_index']}, class {escape(data['class_name'])}. Readout: {escape(data['readout'])}. Validation accuracy of unchanged teacher: {data['checkpoint']['validation_accuracy']:.2%}.</p>
    <p>Independent samples: discovery, joint coefficient fit, validation for choosing branch count, and three untouched tests. Graph candidates include <strong>all 64 unaries and 2016 pairs</strong>; triples only if explicitly enabled. Same budget comparison to given local topology. Readout changes and all-piece retention shifts must be considered.</p>
    {''.join(panels)}<p><small>Open results.json and PREREG_V2.md for numerical gates, query accounting and limitations. No test metrics were used to choose the graph.</small></p></html>'''
    (out/'report.html').write_text(html,encoding='utf-8')


def cifar(args):
    import torch
    from sihtigraph.teacher import datasets, load_teacher, normalize, ImageOracle, CIFAR_LABELS
    device=args.device
    if device=='auto':device='cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cuda' and not torch.cuda.is_available():raise SystemExit('CUDA requested but unavailable')
    ck=Path(args.checkpoint)
    if not ck.is_file():raise SystemExit(f'Missing trained checkpoint: {ck}. Do not retrain; copy/use the saved v1 checkpoint.')
    if args.image_index<0:raise SystemExit('v2 needs a fixed image index (default 65)')
    model,checkpoint=load_teacher(str(ck),device)
    _,_,test=datasets(args.data,args.download)
    image,label=test[args.image_index]
    with torch.inference_mode():
        prediction=int(model(normalize(image.unsqueeze(0).to(device))).argmax(1).item())
    if label != args.class_id or prediction!=label:
        raise SystemExit(f'Original teacher image/class changed: label={label}, predicted={prediction}; expected {args.class_id}')
    if args.mode!='cifar':raise AssertionError('invalid mode')
    out=Path(args.out)
    if out.resolve()==Path('runs/v1_bird_seed41').resolve() or 'v1_bird_seed41' in out.parts:
        raise SystemExit('Refusing to overwrite the v1 run: choose a v2 output directory')
    out.mkdir(parents=True,exist_ok=True)
    signature={'version':'v2_direct_1','checkpoint_sha256':file_hash(ck),
               'image_sha256':hashlib.sha256(image.numpy().tobytes()).hexdigest(),
               'image_index':args.image_index, 'class_id':label,'seed':args.seed,
               'readout':args.readout,'fixed_competitor':args.fixed_competitor,
               'replacement':args.replacement,'selection':args.selection_masks,
               'fit':args.fit_masks,'validation':args.validation_masks,
               'test':args.test_masks,'max_terms':args.max_terms,
               'scan_triples':args.scan_triples,'ridge':args.ridge}
    manifest_path=out/'manifest.json'
    if manifest_path.exists():
        saved=json.loads(manifest_path.read_text(encoding='utf-8'))
        if saved != signature:
            raise SystemExit('Existing output has a different configuration/checkpoint. Use a different --out path.')
    else:
        if (out/'results.json').exists():
            raise SystemExit('Existing results without v2 manifest; choose a new --out path.')
        write_json(manifest_path,signature)
    result_path=out/'results.json'
    if result_path.exists():
        result=json.loads(result_path.read_text(encoding='utf8'))
        if result.get('complete'):
            raise SystemExit(f'Already completed: {result_path}. Do not overwrite; select another --out.')
    else:
        result={'protocol':'v2_direct_1','complete':False,'image_index':args.image_index,
                'class_name':CIFAR_LABELS[label], 'class_id':label,'readout':args.readout,
                'checkpoint':checkpoint,'teacher_correct':True,
                'distillation':{},'config':vars(args)}
    for replacement in (('zero','mean') if args.replacement=='both' else (args.replacement,)):
        if replacement in result['distillation']:
            print(f'[{replacement}]: complete in saved interim receipt, skipping',flush=True)
            continue
        print(f'[{replacement}]: direct all-unary/all-pair discovery...',flush=True)
        oracle=ImageOracle(model,image.unsqueeze(0),label,replacement,device,batch=args.batch,
                           readout=args.readout,fixed_competitor=args.fixed_competitor)
        sign=dict(signature,replacement=replacement)
        cache=StageCache(out,sign)
        samples={}
        for stage,rows in [('selection',args.selection_masks),('fit',args.fit_masks),
                           ('validation',args.validation_masks),
                           ('test_p50',args.test_masks),('test_p75',args.test_masks),('test_p90',args.test_masks)]:
            samples[stage]=cache.read(stage,rows,args.seed,oracle)
        zsel,ysel=samples['selection']
        zfit,yfit=samples['fit']
        zv,yv=samples['validation']
        moments=degree12_moments(zsel,ysel)
        if args.scan_triples:
            print(f'  scanning {64*63*62//6:,} independent triples...',flush=True)
            moments+=degree3_moments(zsel,ysel,chunk=args.triple_chunk)
        globals50=ranked_candidates(moments,args.max_terms)
        loc50=ranked_candidates(moments,args.max_terms,allowed=local_allowed())
        graph,graph_selection=choose_graph(globals50,zfit,yfit,zv,yv)
        local,local_selection=choose_graph(loc50,zfit,yfit,zv,yv)
        unaries=[t for t in moments if len(t['indices'])==1]
        unary=fit_graph(zfit,yfit,unaries)
        random_rng=np.random.default_rng(args.seed+2001)
        chosen=graph["terms"]
        candidate_by_degree={}
        chosen_ids={tuple(t['indices']) for t in chosen}
        for entry in moments:
            if tuple(entry['indices']) in chosen_ids: continue
            candidate_by_degree.setdefault(len(entry['indices']),[]).append(entry)
        random_choices=[]
        for item in chosen:
            choices=candidate_by_degree[len(item['indices'])]
            if not choices:raise RuntimeError('Insufficient matched-order random candidates')
            index=int(random_rng.integers(len(choices)))
            random_choices.append(choices.pop(index))
        random_graph=fit_graph(zfit,yfit,random_choices)
        constant=fit_graph(zfit,yfit,[])
        graphs={'global50':graph,'local50':local,'unary64':unary,
                'random50':random_graph,'constant':constant}
        heldout={}
        for stage,name in [('test_p50','p=0.5'),('test_p75','p=0.75'),('test_p90','p=0.9')]:
            z,y=samples[stage]
            heldout[name]={key:summary(z,y,model) for key,model in graphs.items()}
        mean=heldout['p=0.5']; rglobal=mean['global50']['r2'];rlocal=mean['local50']['r2']
        gates={'G0_teacher_reconstruction':oracle.reconstruction_error <= 1e-5,
               'G0_margin_has_variance':mean['global50']['teacher_variance']>1e-4,
               'G1_p50_r2_at_least_050':rglobal is not None and rglobal>=.50,
               'G1_near_given_local_graph':rglobal is not None and rlocal is not None and rglobal >= rlocal-.03,
               'G2_p75_r2_at_least_020':heldout['p=0.75']['global50']['r2'] is not None and heldout['p=0.75']['global50']['r2']>=.20,
               'G2_p90_nonnegative_r2':heldout['p=0.9']['global50']['r2'] is not None and heldout['p=0.9']['global50']['r2']>=0}
        result['distillation'][replacement]={
            'graphs':{k:{'intercept':float(g['intercept']),
                         'terms':[pretty(t) for t in g['terms']]} for k,g in graphs.items()},
            'selection':{'global50':graph_selection,'local50':local_selection},
            'discovery':{'degree12_candidates':64+2016,'triple_candidates':41664 if args.scan_triples else 0,
                         'top_discovery':[{'indices':t['indices'],'score':t['score']} for t in globals50]},
            'tests':heldout,'teacher_oracle':oracle.summary(),
            'cache':{'fresh':cache.queried,'loaded':cache.loaded,'needed':cache.loaded+cache.queried},
            'gates':gates}
        write_json(result_path,result)
        print(f'  test p=.5 R² global={rglobal:.4f}, given-local={rlocal:.4f}; '
              f'graph branches={len(graph["terms"])}',flush=True)
    result['complete']=True
    write_json(result_path,result)
    html_report(result,out)
    print(f'Complete: {result_path} and {out/"report.html"}',flush=True)


def legacy_diagnostic(seed:int):
    print('Legacy KM diagnostic (not a v2 fitting baseline).')
    for start in (0,24):
        coords=list(range(start,start+40))
        def teacher(z):
            return np.sum(z[:,coords],axis=1)/np.sqrt(40.)
        km=km_search(teacher,n=64,seed=seed,tau=.16,q=1024,max_live=64,
                     calibration_masks=2048,coefficient_masks=8192,max_terms=50)
        rng=np.random.default_rng(seed+900)
        z=random_masks(rng,4096,64)
        pred=np.full(len(z),km['intercept'])
        for term in km['terms']:
            pred+=term['coefficient']*np.prod(z[:,term['indices']],axis=1)
        r=metrics(teacher(z),pred)
        print(json.dumps({'start':start,'found':len(km['terms']),
                          'selected':km['terms'],'test_r2':r['r2'],
                          'candidate_cap_hit':any(l['capped'] for l in km['levels'])},indent=2))


def main():
    ap=argparse.ArgumentParser(description='SihtiGraphModel v2 direct reader; reuses v1 checkpoint')
    ap.add_argument('--mode',choices=('synthetic','legacy','cifar'),default='synthetic')
    ap.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    ap.add_argument('--checkpoint',default='runs/checkpoints/cifar_tiny_cnn.pt')
    ap.add_argument('--data',default='data')
    ap.add_argument('--download',action='store_true')
    ap.add_argument('--out',default='runs/v2_bird_seed41')
    ap.add_argument('--image-index',type=int,default=65)
    ap.add_argument('--class-id',type=int,default=2)
    ap.add_argument('--seed',type=int,default=41)
    ap.add_argument('--replacement',choices=('zero','mean','both'),default='both')
    ap.add_argument('--readout',choices=('margin','raw','fixed'),default='margin')
    ap.add_argument('--fixed-competitor',type=int,default=3)
    ap.add_argument('--batch',type=int,default=256)
    ap.add_argument('--selection-masks',type=int,default=20000)
    ap.add_argument('--fit-masks',type=int,default=8192)
    ap.add_argument('--validation-masks',type=int,default=4096)
    ap.add_argument('--test-masks',type=int,default=4096)
    ap.add_argument('--max-terms',type=int,default=50)
    ap.add_argument('--ridge',type=float,default=1e-4)
    ap.add_argument('--scan-triples',action='store_true',
                    help='Full 41,664-triple CPU scan; optional and costly')
    ap.add_argument('--triple-chunk',type=int,default=64)
    args=ap.parse_args()
    if args.mode=='synthetic':
        cases=[synthetic_40(args.seed,0),synthetic_40(args.seed,24)]
        print(json.dumps({'synthetic_40':cases},indent=2))
        if not all(c['truth_found_in_top40']==40 and c['test_r2']>.999 for c in cases):
            raise SystemExit('FAIL: distributed 40-term synthetic gate')
        print('PASS: all 40 terms recovered at both coordinate positions')
        return
    if args.mode=='legacy':
        legacy_diagnostic(args.seed)
        return
    if not 1<=args.max_terms<=50:raise SystemExit('--max-terms must be 1..50')
    if min(args.selection_masks,args.fit_masks,args.validation_masks,args.test_masks)<100:
        raise SystemExit('At least 100 independent masks per stage required')
    cifar(args)


if __name__=='__main__':main()
