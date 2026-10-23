"""SihtiGraphModel v3 — retention-mixture graph and cross-bird replication.

Reuses the v1 frozen CNN; never trains. Reads frozen v2 graph for matched
evaluation on NEW masks; preserves runs/v1_bird_seed41 and runs/v2_bird_seed41.

  python run_v3.py --mode synthetic
  python run_v3.py --mode cifar --device cuda
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from html import escape
from pathlib import Path
import numpy as np

from sihtigraph.v2 import fit_graph, graph_predict, local_allowed
from sihtigraph.v3 import (BUDGETS, RETENTIONS, choose_for_strata,
    compare_same_tile, concat_strata, mixed_candidate_scores,
    same_tile_cross_octave, synthetic_biased)
from sihtigraph.walsh import metrics,random_masks
from run_v2 import file_hash,pretty,write_json

STAGE_OFFSETS={'discover':101,'fit':202,'validate':303,'test':404,'all_on':505}

def masks_for(seed:int,image_idx:int,stage:str,p:float,nrows:int):
    if stage not in STAGE_OFFSETS:raise ValueError('invalid stage')
    rng=np.random.default_rng(np.random.SeedSequence(
        [seed,image_idx,STAGE_OFFSETS[stage],int(round(p*1000))]))
    if stage=='all_on':return np.ones((nrows,64),dtype=np.int8)
    return random_masks(rng,nrows,64,p)

class Cache:
    """Replay masked teacher scores; retain clipping numerator and denominator."""
    def __init__(self,out:Path,signature:dict):
        key=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()[:20]
        self.path=out/'cache'/key
        self.path.mkdir(parents=True,exist_ok=True)
        self.fresh=0;self.loaded=0
    def read(self,seed:int,index:int,stage:str,p:float,n:int,oracle):
        name=f'{stage}_p{int(round(100*p)):03d}'
        path=self.path/(name+'.npz')
        z=masks_for(seed,index,stage,p,n)
        if path.exists():
            with np.load(path,allow_pickle=False) as d:
                oz=d['z'];y=d['y'];clipped=int(d['clipped']);total=int(d['total'])
            if not np.array_equal(z,oz) or y.shape!=(n,) or not np.isfinite(y).all():
                raise ValueError(f'{path}: cache mismatch. Use a distinct --out.')
            if hasattr(oracle,'clipped_pixels'):
                oracle.clipped_pixels+=clipped;oracle.total_pixels+=total
            self.loaded+=n
            print(f'   {name}: loaded {n:,}',flush=True)
            return z,np.asarray(y,dtype=np.float64)
        before_clip=int(getattr(oracle,'clipped_pixels',0))
        before_total=int(getattr(oracle,'total_pixels',0))
        y=np.asarray(oracle(z),dtype=np.float64).reshape(-1)
        if y.shape!=(n,) or not np.isfinite(y).all():
            raise ValueError('nonfinite or malformed teacher response')
        clipped=int(getattr(oracle,'clipped_pixels',0))-before_clip
        total=int(getattr(oracle,'total_pixels',0))-before_total
        tmp=path.with_name(path.name+'.tmp')
        with tmp.open('wb') as f:np.savez_compressed(f,z=z,y=y,clipped=clipped,total=total)
        os.replace(tmp,path)
        self.fresh+=n
        print(f'   {name}: queried and cached {n:,}',flush=True)
        return z,y

def load_v2_frozen(path:Path,replacement:str):
    if not path.is_file():return None
    d=json.loads(path.read_text(encoding='utf8'))
    if d.get('protocol')!='v2_direct_1' or d.get('image_index')!=65:
        raise ValueError('v2 reference is not the fixed source-image-65 run')
    if d.get('readout')!='margin':return None
    return d['distillation'][replacement]['graphs']['global50']

def select_second_bird(model,test,device:str,after:int=65,class_id:int=2)->int:
    """First subsequent intact image recognized as bird; never consult masks."""
    import torch
    from sihtigraph.teacher import normalize
    with torch.inference_mode():
        for index in range(after+1,len(test)):
            img,label=test[index]
            if label!=class_id:continue
            guess=int(model(normalize(img.unsqueeze(0).to(device))).argmax(1).item())
            if guess==class_id:return index
    raise RuntimeError('No second correctly classified bird after original image')

def evaluate(samples,graphs):
    return {str(p):{name:metrics(y,graph_predict(z,g)) for name,g in graphs.items()}
            for p,(z,y) in samples.items()}

def result_for_image(args,model,test,device,index,replacement,out,signature,v2_frozen):
    import torch
    from sihtigraph.teacher import ImageOracle,normalize
    image,label=test[index]
    with torch.inference_mode():
        correct=int(model(normalize(image.unsqueeze(0).to(device))).argmax(1).item())==label
    if label!=args.class_id or not correct:
        raise RuntimeError(f'Image {index} not a correctly recognized target-class bird')
    oracle=ImageOracle(model,image.unsqueeze(0),label,replacement,device,
        batch=args.batch,readout=args.readout,fixed_competitor=args.fixed_competitor)
    cache=Cache(out,dict(signature,image_index=index,replacement=replacement,
       image_hash=hashlib.sha256(image.numpy().tobytes()).hexdigest()))
    phases={}
    for stage,n in (('discover',args.discovery_per_p),('fit',args.fit_per_p),
                    ('validate',args.validation_per_p),('test',args.test_per_p)):
        phases[stage]={p:cache.read(args.seed,index,stage,p,n,oracle) for p in RETENTIONS}
    zfull,yfull=cache.read(args.seed,index,'all_on',1.,1,oracle)
    moments=mixed_candidate_scores(phases['discover'])
    global50=moments[:args.max_terms]
    local50=[t for t in moments if tuple(t['indices']) in local_allowed()][:args.max_terms]
    graph,selected=choose_for_strata(global50,phases['fit'],phases['validate'],
                                      budgets=BUDGETS,ridge=args.ridge)
    local,local_selected=choose_for_strata(local50,phases['fit'],phases['validate'],
                                           budgets=BUDGETS,ridge=args.ridge)
    # Pure fixed-p=.9 comparison reuses that stratum's independent splits.
    p90_moments=mixed_candidate_scores({.9:phases['discover'][.9]})
    biased90,biased90_sel=choose_for_strata(p90_moments[:args.max_terms],
        {.9:phases['fit'][.9]},{.9:phases['validate'][.9]},
        budgets=BUDGETS,ridge=args.ridge,focus=.9)
    zfit,yfit=concat_strata(phases['fit'])
    unary=fit_graph(zfit,yfit,[t for t in moments if len(t['indices'])==1],ridge=args.ridge)
    constant=fit_graph(zfit,yfit,[],ridge=args.ridge)
    graphs={'mixture_global':graph,'mixture_local':local,'biased90_global':biased90,
            'mixture_unary64':unary,'mixture_constant':constant}
    if v2_frozen is not None:graphs['v2_frozen_global50']=v2_frozen
    heldout=evaluate(phases['test'],graphs)
    full={name:{'prediction':float(graph_predict(zfull,g)[0]),
                'abs_margin_error':float(abs(graph_predict(zfull,g)[0]-yfull[0]))}
          for name,g in graphs.items()}
    p50=heldout['0.5']['mixture_global']['r2']
    p90=heldout['0.9']['mixture_global']['r2']
    p95=heldout['0.95']['mixture_global']['r2']
    v2p90=heldout['0.9'].get('v2_frozen_global50',{}).get('r2')
    return {
      'image_index':index,'class_id':label,'replacement':replacement,
      'teacher_correct_intact':correct,
      'source':'first subsequent correctly predicted bird' if index!=65 else 'fixed v1 bird index 65',
      'graphs':{key:{'intercept':float(g['intercept']),
                    'terms':[pretty(t) for t in g['terms']]}
                for key,g in graphs.items()},
      'selection':{'mixture_global':selected,'mixture_local':local_selected,
                   'biased90_global':biased90_sel},
      'discovery':{'candidate_count':len(moments),'top_scores':moments[:50],
                   'p_biased_per_stratum':True,'mixture_orthogonality_claimed':False},
      'tests':heldout,
      'full_image':{'teacher_margin':float(yfull[0]),'models':full},
      'same_tile_B0_B3':same_tile_cross_octave(graph),
      'gates':{'G0_reconstruction':oracle.reconstruction_error<=1e-5,
               'G0_correct_original':bool(correct),
               'G1_p90_r2_nonnegative':p90 is not None and p90>=0,
               'G2_p95_r2_nonnegative':p95 is not None and p95>=0,
               'G3_p50_r2_ge_035':p50 is not None and p50>=.35,
               'G4_p90_gain_ge_050_vs_frozen_v2':None if v2p90 is None else
                    p90 is not None and p90>=v2p90+.5},
      'oracle':oracle.summary(),'cache':{'fresh':cache.fresh,'loaded':cache.loaded}
    }

def render_html(data,path:Path):
    lines=['<!doctype html><html lang="en"><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width"><title>SihtiGraphModel v3</title>',
        '<style>body{font:15px system-ui;max-width:1040px;margin:2rem auto;padding:0 1rem;line-height:1.5}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid #7775;padding:7px;text-align:left}section{border-top:1px solid #999;margin:2rem 0;padding-top:1rem}</style>',
        '<h1>SihtiGraphModel v3 — retention-aware graph reader</h1>',
        '<p>Frozen CNN; separate discovery/fit/validation/test masks within p=.5/.75/.9/.95. Per-p biased Walsh moments rank terms; the mixture fit is NOT itself an orthogonal Fourier spectrum. Frozen v2 is evaluated on new paired masks.</p>']
    for key,r in data['images'].items():
        lines.append(f'<section><h2>Bird #{r["image_index"]} · {escape(r["replacement"])}</h2>')
        lines.append('<table><tr><th>Retention p</th><th>Mixture global R²</th><th>Mixture local R²</th><th>Fixed-p=.90 R²</th><th>Unaries R²</th><th>Frozen v2 R²</th></tr>')
        for p,tests in r['tests'].items():
            def rv(name):
                a=tests.get(name,{}).get('r2')
                return 'n/a' if a is None else f'{a:.4f}'
            lines.append('<tr><td>'+p+'</td>'+''.join('<td>'+rv(name)+'</td>' for name in
                ('mixture_global','mixture_local','biased90_global','mixture_unary64',
                 'v2_frozen_global50'))+'</tr>')
        lines.append('</table>')
        lines.append(f'<p>Unmasked teacher margin {r["full_image"]["teacher_margin"]:.4f}; '+'; '.join(
          f'{name} error={a["abs_margin_error"]:.3f}' for name,a in
          r['full_image']['models'].items())+'</p>')
        lines.append('<p>B0↔B3 same-tile links: '+escape(', '.join(
          f'tile {t["tile"]}: {t["coefficient"]:+.3f}' for t in r['same_tile_B0_B3']) or 'none')+'</p>')
        lines.append(f'<p>Selected {r["selection"]["mixture_global"]["chosen_count"]} terms; '
                     f'clipped pixels {r["oracle"]["fraction_pixels_clipped"]:.2%}</p></section>')
    if data.get('cross_bird'):
        b=data['cross_bird']
        lines.append('<h2>Cross-bird structural recurrence</h2><p>Shared same-tile B0↔B3 links: '+
          escape(str(b['shared_tiles']))+'; same sign: '+escape(str(b['same_sign_shared']))+
          '. Independently fitted graphs, not transferred coefficients.</p>')
    lines.append('<p>Read results.json for complete gates, coefficients and controls. v1/v2 historical results are unchanged.</p></html>')
    path.write_text('\n'.join(lines),encoding='utf8')

def cifar(args):
    import torch
    from sihtigraph.teacher import datasets,load_teacher
    device=args.device if args.device!='auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
    if device=='cuda' and not torch.cuda.is_available():raise SystemExit('CUDA unavailable')
    ck=Path(args.checkpoint)
    if not ck.is_file():raise SystemExit('Saved v1 teacher checkpoint missing. This runner never trains.')
    model,meta=load_teacher(str(ck),device)
    _,_,test=datasets(args.data,args.download)
    second=select_second_bird(model,test,device,65,args.class_id)
    print(f'Frozen checkpoint epoch {meta["best_epoch"]}; second prespecified bird index {second}',flush=True)
    out=Path(args.out)
    if out.resolve() in (Path('runs/v1_bird_seed41').resolve(),Path('runs/v2_bird_seed41').resolve()) or any(
        q in ('v1_bird_seed41','v2_bird_seed41') for q in out.parts):
        raise SystemExit('Refusing to write into preserved v1/v2 directories')
    out.mkdir(parents=True,exist_ok=True)
    v2_path=Path(args.v2_results)
    sign={'protocol':'v3_retention_mixture_1','checkpoint_sha256':file_hash(ck),
          'images':[65,second],
          'image_sha256':{str(i):hashlib.sha256(test[i][0].numpy().tobytes()).hexdigest()
                          for i in (65,second)},
          'v2_results_sha256':file_hash(v2_path) if v2_path.is_file() else None,
          'seed':args.seed,'discovery_per_p':args.discovery_per_p,
          'fit_per_p':args.fit_per_p,'validation_per_p':args.validation_per_p,
          'test_per_p':args.test_per_p,'retentions':list(RETENTIONS),
          'replacement':args.replacement,'readout':args.readout,
          'fixed_competitor':args.fixed_competitor,'ridge':args.ridge,
          'max_terms':args.max_terms}
    manifest=out/'manifest.json'
    if manifest.is_file():
        if json.loads(manifest.read_text())!=sign:raise SystemExit('v3 manifest differs; choose a different --out')
    else:
        if (out/'results.json').exists():raise SystemExit('Old results without manifest; use a new --out')
        write_json(manifest,sign)
    result_path=out/'results.json'
    if result_path.is_file():
        data=json.loads(result_path.read_text())
        if data.get('complete'):raise SystemExit('v3 run already completed. Use a new --out.')
    else:
        data={'protocol':sign['protocol'],'complete':False,'checkpoint':meta,
              'retentions':list(RETENTIONS),'images':{},'config':vars(args),
              'index_second_bird':second}
    replacements=('mean','zero') if args.replacement=='both' else (args.replacement,)
    for idx in (65,second):
        for repl in replacements:
            key=f'{idx}_{repl}'
            if key in data['images']:
                print(f'[{key}]: saved result found, skipping',flush=True)
                continue
            print(f'[{key}]: mixture discovery, joint fit, validation, held-out tests',flush=True)
            frozen=(load_v2_frozen(v2_path,repl) if idx==65 and args.readout=='margin' else None)
            data['images'][key]=result_for_image(args,model,test,device,idx,repl,out,sign,frozen)
            write_json(result_path,data)
            print(f'[{key}] held-out 90% R²={data["images"][key]["tests"]["0.9"]["mixture_global"]["r2"]:.4f}',flush=True)
    comparisons=[]
    for repl in replacements:
        a=data['images'][f'65_{repl}']['graphs']['mixture_global']
        b=data['images'][f'{second}_{repl}']['graphs']['mixture_global']
        c=compare_same_tile(a,b);c['replacement']=repl;comparisons.append(c)
    data['cross_bird_by_replacement']=comparisons
    data['cross_bird']=next((c for c in comparisons if c['replacement']=='mean'),comparisons[0])
    data['complete']=True
    write_json(result_path,data)
    render_html(data,out/'report.html')
    print(f'Complete: {result_path} and {out/"report.html"}',flush=True)

def main():
    ap=argparse.ArgumentParser(description='Sihti v3 retention mixture. Never trains CNN.')
    ap.add_argument('--mode',choices=('synthetic','cifar'),default='synthetic')
    ap.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    ap.add_argument('--data',default='data');ap.add_argument('--download',action='store_true')
    ap.add_argument('--checkpoint',default='runs/checkpoints/cifar_tiny_cnn.pt')
    ap.add_argument('--v2-results',default='runs/v2_bird_seed41/results.json')
    ap.add_argument('--out',default='runs/v3_birds_mixture_seed41')
    ap.add_argument('--class-id',type=int,default=2);ap.add_argument('--seed',type=int,default=41)
    ap.add_argument('--replacement',choices=('mean','zero','both'),default='mean')
    ap.add_argument('--readout',choices=('margin','raw','fixed'),default='margin')
    ap.add_argument('--fixed-competitor',type=int,default=3)
    ap.add_argument('--discovery-per-p',type=int,default=6000)
    ap.add_argument('--fit-per-p',type=int,default=4000)
    ap.add_argument('--validation-per-p',type=int,default=1500)
    ap.add_argument('--test-per-p',type=int,default=2048)
    ap.add_argument('--max-terms',type=int,default=50)
    ap.add_argument('--ridge',type=float,default=1e-4)
    ap.add_argument('--batch',type=int,default=256)
    args=ap.parse_args()
    if args.mode=='synthetic':
        r=synthetic_biased(args.seed)
        print(json.dumps(r,indent=2))
        if abs(r['unary_biased_2']-1.5)>.065 or abs(r['pair_biased_1_6']+.8)>.065 or r['heldout_r2']<.999:
            raise SystemExit('FAIL: p-biased synthetic oracle')
        print('PASS: fixed-p biased moments and full raw-basis joint refit')
        return
    if args.class_id!=2 or args.max_terms!=50:raise SystemExit('Locked v3 class=2, max_terms=50')
    if min(args.discovery_per_p,args.fit_per_p,args.validation_per_p,args.test_per_p)<100:
        raise SystemExit('Need at least 100 examples per stage and p')
    cifar(args)

if __name__=='__main__':main()
