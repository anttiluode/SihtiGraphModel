"""Preregistered SihtiGraphModel v1. CPU smoke: python run.py --mode smoke."""
from __future__ import annotations
import argparse
import json
from html import escape
from pathlib import Path
from collections import Counter
import numpy as np
from sihtigraph.walsh import (fit_fourier, km_search, local_pairs, metrics,
                              parity, predict, random_masks, r2)


def make_report(payload: dict, path: Path):
    def num(x): return 'n/a' if x is None else f'{x:.3f}'
    def table(row):
        return ''.join(f'<tr><th>{escape(repl)}</th>' + ''.join(
            f'<td>{num(row[repl][key]["r2"])}</td>' for key in ('graph','linear','local_pair50','random_matched','intercept')) + '</tr>'
            for repl in row)
    panels = []
    for baseline, entry in payload['distillation'].items():
        terms=entry['terms']
        heat=[0]*64
        for t in terms:
            for j in t['indices']:heat[j]+=abs(t['coefficient'])
        ceiling=max(heat or [1]) or 1
        squares=''
        for b in range(4):
            for row in range(4):
                for col in range(4):
                    v=heat[b*16+row*4+col]
                    opa=.08+.86*v/ceiling
                    squares+=f'<rect x="{20+b*138+col*29}" y="{25+row*29}" width="26" height="26" rx="3" fill="#0d9488" opacity="{opa:.3f}" />'
            squares+=f'<text x="{20+b*138}" y="160" font-size="12" fill="currentColor">band {b}</text>'
        edges=''.join(f'<tr><td>{i+1}</td><td>{escape(", ".join(t["names"]))}</td><td>{num(t["coefficient"])}</td><td>{len(t["indices"])}</td></tr>' for i,t in enumerate(terms[:30]))
        stats=entry['evaluation']
        panels.append(f'''<section><h2>{escape(baseline)} replacement</h2>
        <p>KM queries {entry["km_query_count"]:,}; all teacher forwards {entry["teacher_forwards"]:,}; graph branches {len(terms)}; calibration SD {num(entry["teacher_calibration_sd"])}.</p>
        <table><tr><th>Mask retention</th><th>Graph R²</th><th>Linear R²</th><th>Pairwise R²</th><th>Random R²</th><th>Constant R²</th></tr>{table(stats)}</table>
        <p>High-order mixed-difference control: {escape(json.dumps(entry['mixed_difference'],ensure_ascii=False))}</p>
        <h3>Influence map (sum of absolute coefficients touching each band/patch)</h3>
        <svg viewBox="0 0 578 178" style="max-width:100%;height:auto">{squares}</svg>
        <h3>Top retained branches</h3><table><tr><th>#</th><th>Input pieces</th><th>θ (logit-margin units)</th><th>Order</th></tr>{edges}</table></section>''')
    html=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>SihtiGraphModel v1</title>
    <style>body{{font:15px system-ui;max-width:1000px;margin:2rem auto;padding:0 1rem;line-height:1.5}} table{{border-collapse:collapse;width:100%;font-size:13px}} td,th{{padding:7px;border-bottom:1px solid #9997;text-align:left}} th{{font-weight:600}} section{{border-top:2px solid #7775;margin-top:2rem}} code{{white-space:pre-wrap}}</style>
    <h1>SihtiGraphModel v1 — black-box-to-band/patch graph</h1>
    <p><strong>Local explanation for one fixed CIFAR image.</strong> Teacher weights are frozen. The graph describes its response to controlled ablations of this image, not the model's full mechanism or a shared universal image classifier.</p>
    <p>Image index {payload['image_index']}; label {escape(payload['class_name'])}; teacher full-image correct {payload['teacher_correct']}; checkpoint validation accuracy {num(payload['checkpoint'].get('validation_accuracy'))}.</p>
    <p><strong>Locked gate:</strong> R² ≥ .80 with ≤50 branches; meaningful extra value requires graph R² ≥ linear R² + .05 on uniform held-out masks. R² and sign agreement are measured independently; neither is a full-spectrum Parseval certificate.</p>
    {''.join(panels)}<p>Raw data and complete gate outcomes in results.json. All zeroed/mean-replaced images are distribution shifts relative to normal CIFAR, regardless of the lossless full-image reconstruction.</p></html>'''
    path.write_text(html,encoding='utf8')


def audit_mixed_difference(oracle, graph_terms, intercept, rng, sigma: float, contexts: int = 16):
    # A finite mixed derivative is a valid matched *input* intervention; it is
    # NOT deletion of an edge from the original trained network.
    high = next((t for t in graph_terms if len(t['indices'])>=3),None)
    if high is None: return {'status':'no recovered degree >=3 branch; cannot claim high-order causal check'}
    s=high['indices'];k=len(s)
    if k>6: return {'status':'order too large for fixed factorial intervention'}
    probes=[]; weights=[]
    for _ in range(contexts):
        base=random_masks(rng,1,64)[0]
        for mask in range(2**k):
            z=base.copy(); w=1
            for j,idx in enumerate(s):
                z[idx]=1 if (mask>>j)&1 else -1
                w*=int(z[idx])
            probes.append(z);weights.append(w)
    z=np.asarray(probes,dtype=np.int8)
    wt=np.asarray(weights,dtype=np.int8).reshape(contexts,-1)
    true=(oracle(z).reshape(contexts,-1)*wt).mean(1)
    approx=(predict(z,graph_terms,intercept).reshape(contexts,-1)*wt).mean(1)
    return {'status':'measured','top_branch':s,'coefficient':high['coefficient'],
            'contexts':contexts,'mean_abs_mixed_derivative_error':float(np.abs(true-approx).mean()),
            'error_over_calibration_sd':float(np.abs(true-approx).mean()/max(sigma,1e-8)),
            'teacher_mean_abs_mixed_derivative':float(np.abs(true).mean()),
            'graph_mean_abs_mixed_derivative':float(np.abs(approx).mean())}


def smoke():
    from sihtigraph.walsh import fwht
    rng=np.random.default_rng(57)
    n=12
    def oracle(z):return .25+1.3*z[:,0]-1.1*z[:,1]*z[:,4]+.8*z[:,2]*z[:,7]*z[:,10]
    masks=random_masks(rng,2048,n)
    got=km_search(oracle,n=n,seed=41,tau=.20,q=1500,max_live=32,coefficient_masks=8192,calibration_masks=4096,max_terms=12)
    found={tuple(t['indices']) for t in got['terms']}
    expected={(0,),(1,4),(2,7,10)}
    print(json.dumps({'oracle':'three-term order 1,2,3 synthetic function','expected':list(map(list,expected)),
                     'recovered':got['terms'],'query_count':got['query_count'],
                     'r2_on_fresh_masks':r2(oracle(masks),predict(masks,got['terms'],got['intercept']))},indent=2))
    if not expected.issubset(found):raise SystemExit('SMOKE FAIL: missing synthetic parity branch')
    if r2(oracle(masks),predict(masks,got['terms'],got['intercept'])) < .95:raise SystemExit('SMOKE FAIL: low R2')
    # Optional visual sieve check without downloading anything.
    try:
        import torch
        from sihtigraph.sieve import pieces,render_mask
        img=torch.rand(1,3,32,32)
        for method in ('zero','mean'):
            base,parts=pieces(img,method)
            z=torch.ones(1,64)
            assert torch.max(torch.abs(render_mask(z,base,parts)-img))<1e-5
            assert torch.max(torch.abs(render_mask(-z,base,parts)-base))<1e-5
        print('SMOKE PASS: exact sieve toggles')
    except ImportError:print('Sieve not checked (torch not installed)')
    print('SMOKE PASS: synthetic Walsh recovery')


def run_distillation(args, out: Path):
    import torch
    from sihtigraph.teacher import datasets, load_teacher, ImageOracle, CIFAR_LABELS, normalize
    from sihtigraph.walsh import fit_fourier
    set_device=args.device if args.device!='auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
    if set_device=='cuda' and not torch.cuda.is_available():raise SystemExit('CUDA requested but not detected')
    model,checkpoint=load_teacher(args.checkpoint,set_device)
    _,_,test=datasets(args.data,args.download)
    selection = 'explicit fixed test index' if args.image_index >= 0 else 'first correctly predicted target-class CIFAR test item'
    selected = args.image_index
    if selected < 0:
        selected = None
        with torch.inference_mode():
            for idx in range(len(test)):
                candidate, target = test[idx]
                if target != args.class_id: continue
                pred = int(model(normalize(candidate.unsqueeze(0).to(set_device))).argmax(1))
                if pred == target:
                    selected = idx
                    break
        if selected is None:
            raise SystemExit('No correctly predicted target-class test image; teacher quality insufficient')
    img,label=test[selected]
    image=img.unsqueeze(0)
    with torch.inference_mode():
        logits=model(normalize(image.to(set_device)))
        prediction=int(logits.argmax(1).item())
    print(f'Teacher original image index={selected}: ground-truth={CIFAR_LABELS[label]} predicted={CIFAR_LABELS[prediction]}',flush=True)
    payload={'protocol':'v1-2026-10-23','image_index':selected,'selection_policy':selection,'class_id':label,
             'class_name':CIFAR_LABELS[label],'teacher_correct':prediction==label,
             'teacher_predicted_label':CIFAR_LABELS[prediction], 'checkpoint':checkpoint,
             'flags':vars(args),'distillation':{}}
    for replacement in (('zero','mean') if args.replacement=='both' else (args.replacement,)):
        print(f'[{replacement}] KM search starts...',flush=True)
        oracle=ImageOracle(model,image,label,replacement,set_device,batch=args.batch)
        km=km_search(oracle,64,seed=args.seed,tau=args.tau,q=args.km_q,max_live=args.max_live,
                     coefficient_masks=args.coefficient_masks,calibration_masks=args.calibration_masks,max_terms=args.max_terms)
        # All comparisons use the same fresh feature/coeff estimation probes.
        rng=np.random.default_rng(args.seed+17)
        zfit=random_masks(rng,args.refit_masks,64)
        yfit=oracle(zfit)
        intercept=km['intercept']
        subsets=[tuple(t['indices']) for t in km['terms']]
        graph=fit_fourier(zfit,yfit,subsets,args.max_terms,intercept)
        linear=fit_fourier(zfit,yfit,[(i,) for i in range(64)],64,intercept)
        local_pair=fit_fourier(zfit,yfit,[(i,) for i in range(64)]+local_pairs(),50,intercept)
        rand_subsets=[];existing=set(subsets)
        for t in graph:
            k=len(t['indices'])
            if k==0:continue
            for _ in range(200):
                s=tuple(sorted(rng.choice(64,size=k,replace=False)))
                if s not in existing and s not in rand_subsets:break
            rand_subsets.append(s)
        random_graph=fit_fourier(zfit,yfit,rand_subsets,len(graph),intercept)
        evaluation={};keep_fracs=[.5,.75,.9]
        for keep in keep_fracs:
            z=random_masks(rng,args.heldout_masks,64,keep)
            y=oracle(z)
            entry={'graph':metrics(y,predict(z,graph,intercept)),
                   'linear':metrics(y,predict(z,linear,intercept)),
                   'local_pair50':metrics(y,predict(z,local_pair,intercept)),
                   'random_matched':metrics(y,predict(z,random_graph,intercept)),
                   'intercept':metrics(y,np.full(len(y),intercept))}
            evaluation[f'p={keep}']=entry
        knockout=audit_mixed_difference(oracle,graph,intercept,rng,km['scale'])
        baseline= evaluation['p=0.5']
        graph_r2=baseline['graph']['r2'];linear_r2=baseline['linear']['r2']
        gate = {'G0_reconstruction':oracle.reconstruction_error <= 1e-5,
                'G0_teacher_original_correct':prediction==label,
                'G0_nonconstant_teacher':baseline['graph']['teacher_variance'] > 1e-4,
                'G0_nontrivial_sign_balance':.1<=baseline['graph']['teacher_positive_fraction']<=.9,
                'G1_heldout_r2_ge_080':graph_r2 is not None and graph_r2>=.80 and len(graph)<=50,
                'G1_plus_005_over_linear':graph_r2 is not None and linear_r2 is not None and graph_r2>=.8 and graph_r2>=linear_r2+.05,
                'G2_shift_p075_r2_ge_060':evaluation['p=0.75']['graph']['r2'] is not None and evaluation['p=0.75']['graph']['r2']>=.6}
        names=lambda t: [f'B{i//16}:r{(i%16)//4}c{i%4}' for i in t['indices']]
        entry={'terms':[dict(t,names=names(t)) for t in graph], 'linear_terms':linear,'pairwise_terms':local_pair,
               'random_terms':random_graph, 'evaluation':evaluation,'mixed_difference':knockout,
               'km_candidates':km['candidates'],'km_levels':km['levels'],'km_status':km['status'],
               'km_raw_final_coefficients':km['terms'], 'km_query_count':km['query_count'],
               'teacher_forwards':oracle.calls,'teacher_calibration_sd':km['scale'],
               'teacher_calibration_mean':intercept,'teacher_oracle':oracle.summary(),'gates':gate}
        payload['distillation'][replacement]=entry
        out.mkdir(parents=True,exist_ok=True)
        (out/'results.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        make_report(payload,out/'report.html')
        print(f'[{replacement}] R2(graph/linear/pair)={graph_r2}/{linear_r2}/{baseline["local_pair50"]["r2"]}; {len(graph)} branches, {oracle.calls:,} model forwards',flush=True)
    return payload


def main():
    ap=argparse.ArgumentParser(description='SihtiGraphModel v1: local Fourier graph of a frozen image classifier')
    ap.add_argument('--mode',choices=('smoke','train','distill','all'),default='smoke')
    ap.add_argument('--device',default='auto',choices=('cpu','cuda','auto'))
    ap.add_argument('--data',default='data')
    ap.add_argument('--download',action='store_true')
    ap.add_argument('--checkpoint',default='runs/checkpoints/cifar_tiny_cnn.pt')
    ap.add_argument('--out',default='runs/v1_bird_seed41')
    ap.add_argument('--overwrite',action='store_true')
    ap.add_argument('--seed',type=int,default=41)
    ap.add_argument('--epochs',type=int,default=8)
    ap.add_argument('--train-limit',type=int,default=0,help='0 means all 45k train examples; exploratory if changed')
    ap.add_argument('--image-index',type=int,default=-1,help='-1: first correctly predicted bird; selection on original image only')
    ap.add_argument('--class-id',type=int,default=2,help='class to use in original-correct selection; 2 is bird')
    ap.add_argument('--replacement',choices=('zero','mean','both'),default='both')
    ap.add_argument('--batch',type=int,default=256)
    ap.add_argument('--km-q',type=int,default=1024)
    ap.add_argument('--tau',type=float,default=.16)
    ap.add_argument('--max-live',type=int,default=64)
    ap.add_argument('--calibration-masks',type=int,default=2048)
    ap.add_argument('--coefficient-masks',type=int,default=8192)
    ap.add_argument('--refit-masks',type=int,default=8192)
    ap.add_argument('--heldout-masks',type=int,default=2048)
    ap.add_argument('--max-terms',type=int,default=50)
    args=ap.parse_args()
    if args.mode=='smoke':smoke();return
    if args.max_terms>50:raise SystemExit('v1 G1 requires max_terms <= 50')
    if args.mode in ('train','all'):
        from sihtigraph.teacher import train_teacher
        import torch
        dev=args.device if args.device!='auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        if Path(args.checkpoint).exists() and not args.overwrite:
            print('Checkpoint exists; not overwriting. Use --overwrite to retrain.',flush=True)
        else:
            train_result=train_teacher(args.data,args.checkpoint,dev,args.seed,args.epochs,args.download,limit=args.train_limit)
            log=Path(args.checkpoint).with_suffix('.train.json')
            log.write_text(json.dumps(train_result,indent=2)+'\n',encoding='utf8')
    if args.mode in ('distill','all'):
        out=Path(args.out)
        if (out/'results.json').exists() and not args.overwrite:
            raise SystemExit(f'{out}/results.json exists. Use --out different/path or --overwrite.')
        out.mkdir(parents=True,exist_ok=True)
        (out/'config.json').write_text(json.dumps(vars(args),indent=2)+'\n',encoding='utf8')
        payload=run_distillation(args,out)
        print('Complete. Read '+str(out/'report.html')+' and '+str(out/'results.json'))


if __name__=='__main__':main()
