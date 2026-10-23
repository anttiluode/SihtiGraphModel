"""Retention-aware Sihti graph reader.

At fixed p, phi_i=(z_i-(2p-1))/sqrt(1-(2p-1)^2) gives p-biased
orthonormal Walsh products. Across a mixture of p, products are NOT
orthogonal. Estimate within-stratum moments for branch selection, then
jointly refit RAW +/-1 branch features on independent mixture masks.
"""
from __future__ import annotations
import numpy as np
from .v2 import fit_graph,graph_predict
from .walsh import metrics

RETENTIONS=(.5,.75,.9,.95)
BUDGETS=(0,10,20,35,50)

def biased_coordinates(z:np.ndarray,p:float)->np.ndarray:
    if not 0<p<1:raise ValueError('p strictly between 0 and 1')
    mu=2*p-1
    return (np.asarray(z,dtype=np.float64)-mu)/np.sqrt(1-mu*mu)

def biased_degree12(z:np.ndarray,y:np.ndarray,p:float):
    z=np.asarray(z);y=np.asarray(y,dtype=np.float64)
    if z.ndim!=2 or y.shape!=(len(z),) or not len(z):raise ValueError('invalid shapes')
    x=biased_coordinates(z,p);v=y-y.mean()
    return (x.T@v)/len(z), (x.T@(x*v[:,None]))/len(z)

def mixed_candidate_scores(samples:dict,weights:dict|None=None)->list[dict]:
    """Rank all 64 singles and 2016 pairs by mean absolute p-biased moment.

    Per-stratum moments are scaled by that stratum's teacher response SD.
    This is a screening score, NOT a mixture spectrum or guaranteed energy.
    """
    if not samples:raise ValueError('no sample strata')
    n=next(iter(samples.values()))[0].shape[1]
    pair=np.zeros((n,n));unary=np.zeros(n)
    if weights is None:weights={p:1/len(samples) for p in samples}
    for p,(z,y) in samples.items():
        if z.shape[1]!=n:raise ValueError('inconsistent widths')
        a,b=biased_degree12(z,y,p)
        norm=max(float(np.std(y)),1e-12)
        unary+=float(weights[p])*np.abs(a)/norm
        pair+=float(weights[p])*np.abs(b)/norm
    out=[{'indices':[i],'score':float(unary[i])} for i in range(n)]
    out.extend({'indices':[i,j],'score':float(pair[i,j])}
               for i in range(n) for j in range(i+1,n))
    return sorted(out,key=lambda t:(-t['score'],tuple(t['indices'])))

def concat_strata(samples:dict):
    return (np.concatenate([x for x,_ in samples.values()]),
            np.concatenate([y for _,y in samples.values()]))

def choose_for_strata(candidates:list[dict],fit_samples:dict,val_samples:dict,
                      budgets=BUDGETS,ridge:float=1e-4,focus:float|None=None):
    """Choose count using validation mean per-stratum R², not test results."""
    zfit,yfit=concat_strata(fit_samples)
    options=sorted({min(int(n),len(candidates)) for n in budgets if n>=0})
    report={};winner=None;best=-float('inf')
    for n in options:
        g=fit_graph(zfit,yfit,candidates[:n],ridge=ridge)
        scores={str(p):metrics(y,graph_predict(z,g))['r2']
                for p,(z,y) in val_samples.items()}
        ps=list(val_samples) if focus is None else [focus]
        vals=[scores[str(p)] for p in ps]
        criterion=float(np.mean(vals)) if all(v is not None for v in vals) else -float('inf')
        report[str(n)]={'score':criterion,'r2':scores}
        if criterion>best+1e-12:
            winner=g;best=criterion
    return winner,{'chosen_count':len(winner['terms']),'criterion':best,
                   'validation_by_count':report,
                   'criterion_type':'mean_per_stratum_r2' if focus is None else f'r2_at_p{focus}'}

def same_tile_cross_octave(graph:dict,first:int=0,last:int=3):
    out=[]
    for t in graph['terms']:
        if len(t['indices'])!=2:continue
        i,j=sorted(t['indices'])
        if i//16==first and j//16==last and i%16==j%16:
            out.append({'tile':int(i%16),'indices':[int(i),int(j)],
                        'coefficient':float(t['coefficient'])})
    return out

def compare_same_tile(a:dict,b:dict)->dict:
    va={x['tile']:x['coefficient'] for x in same_tile_cross_octave(a)}
    vb={x['tile']:x['coefficient'] for x in same_tile_cross_octave(b)}
    overlap=sorted(set(va)&set(vb))
    signs=[i for i in overlap if np.sign(va[i])==np.sign(vb[i])]
    return {'tiles_a':sorted(va),'tiles_b':sorted(vb),'shared_tiles':overlap,
            'same_sign_shared':signs,'count_shared':len(overlap),'count_same_sign':len(signs)}

def synthetic_biased(seed:int=41):
    """Check correct fixed-p Fourier moments and raw-basis joint refit."""
    rng=np.random.default_rng(seed);p=.9
    masks=lambda n:np.where(rng.random((n,8))<p,1,-1).astype(np.int8)
    def teacher(z):
        x=biased_coordinates(z,p)
        return 1.2+1.5*x[:,2]-.8*x[:,1]*x[:,6]
    z=masks(75000);a,b=biased_degree12(z,teacher(z),p)
    fit=masks(12000);test=masks(4000)
    # Biased pair products expand into raw pair PLUS its associated unaries.
    # Leaving those unaries out gives misleadingly low fidelity.
    terms=[{'indices':[2]},{'indices':[1]},{'indices':[6]},{'indices':[1,6]}]
    graph=fit_graph(fit,teacher(fit),terms)
    return {'unary_biased_2':float(a[2]),'pair_biased_1_6':float(b[1,6]),
            'heldout_r2':metrics(teacher(test),graph_predict(test,graph))['r2']}
