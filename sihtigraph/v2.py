"""v2 graph discovery: direct orthogonal Fourier moments, *without* KM pruning.

All discovery uses independent uniform masks. Selection, fitting, validation and
test are separate datasets. This is a behavioral surrogate of one masked image.
"""
from __future__ import annotations

from itertools import combinations
import numpy as np
from .walsh import local_pairs, metrics, random_masks


def degree12_moments(z: np.ndarray, y: np.ndarray) -> list[dict]:
    """Measure all 64 unary and 2,016 pair moments in one matrix multiplication.

    Uniform independent ±1 masks make Walsh products orthogonal in expectation.
    Values are estimates, not orthogonal on a finite sample, and selecting the
    largest estimates creates winner's-curse bias. Refit on independent masks.
    """
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if z.ndim != 2 or len(y) != len(z) or not len(y):
        raise ValueError("z must be a non-empty 2D matrix and y its response")
    n = z.shape[1]
    centered = y - y.mean()
    unary = (z.T @ centered) / len(y)
    weighted = z * centered[:, None]
    pair = (z.T @ weighted) / len(y)
    terms = [{"indices":[i], "score":float(unary[i])} for i in range(n)]
    terms += [{"indices":[i,j], "score":float(pair[i,j])} for i in range(n) for j in range(i+1,n)]
    return terms


def degree3_moments(z: np.ndarray, y: np.ndarray, chunk: int = 64) -> list[dict]:
    """Independent exhaustive triple scan: *never* requires a selected pair.

    O(N * choose(n,3)) CPU operations; explicitly opt-in on real 64-bit images.
    Computes exact sample correlations via chunked vectorized products.
    """
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if z.ndim != 2 or len(y)!=len(z) or not len(y) or chunk < 1:
        raise ValueError("invalid discovery data or chunk")
    centered = y - y.mean()
    combos = list(combinations(range(z.shape[1]),3))
    answer = []
    for start in range(0,len(combos),chunk):
        batch=combos[start:start+chunk]
        idx = np.asarray(batch,dtype=np.int64)
        prod = z[:,idx[:,0]] * z[:,idx[:,1]] * z[:,idx[:,2]]
        scores = (centered @ prod)/len(z)
        answer.extend({"indices":[int(i) for i in trip],"score":float(score)}
                      for trip,score in zip(batch,scores))
    return answer


def ranked_candidates(moments: list[dict], size: int = 50, allowed: set[tuple[int,...]] | None = None) -> list[dict]:
    if allowed is not None:
        moments=[t for t in moments if tuple(t["indices"]) in allowed]
    return sorted(moments,key=lambda t:(-abs(t["score"]),tuple(t["indices"])))[:size]


def design(z: np.ndarray, terms: list[dict]) -> np.ndarray:
    z=np.asarray(z,dtype=np.float64)
    x=np.empty((len(z),len(terms)+1),dtype=np.float64)
    x[:,0]=1.
    for j,t in enumerate(terms):
        x[:,j+1] = np.prod(z[:,t["indices"]],axis=1)
    return x


def fit_graph(z: np.ndarray, y: np.ndarray, terms: list[dict], ridge: float=1e-4) -> dict:
    """Independent joint ridge refit; constant is unpenalized."""
    if ridge<0: raise ValueError("ridge must be nonnegative")
    x=design(z,terms)
    y=np.asarray(y,dtype=np.float64).reshape(-1)
    xtx=x.T@x
    penalty=np.eye(x.shape[1])*float(ridge)
    penalty[0,0]=0
    theta=np.linalg.solve(xtx+penalty, x.T@y)
    return {"intercept":float(theta[0]),
            "terms":[{"indices":list(t["indices"]),"coefficient":float(c)} for t,c in zip(terms,theta[1:])]}


def graph_predict(z: np.ndarray, graph: dict) -> np.ndarray:
    return design(z,graph["terms"]) @ np.asarray([graph["intercept"]]+
        [t["coefficient"] for t in graph["terms"]],dtype=np.float64)


def choose_graph(ranked: list[dict], zfit: np.ndarray, yfit: np.ndarray,
                 zval: np.ndarray, yval: np.ndarray, budgets=(0,5,10,20,35,50)) -> tuple[dict,dict]:
    """Tune branch *count* using validation, never held-out testing."""
    candidates=tuple(sorted({min(len(ranked),int(x)) for x in budgets if x>=0} | {min(len(ranked),len(ranked))}))
    chosen=None;score=-np.inf;report={}
    for count in candidates:
        fitted=fit_graph(zfit,yfit,ranked[:count])
        r=metrics(yval,graph_predict(zval,fitted))["r2"]
        report[str(count)]=r
        if r is not None and r>score+1e-12:
            chosen=fitted;score=r
    return chosen,{"count":len(chosen["terms"]),"validation_r2_by_count":report,
                   "selected_validation_r2":float(score)}


def local_allowed(nbands=4,tiles=4)->set[tuple[int,...]]:
    return {(i,) for i in range(nbands*tiles*tiles)} | set(local_pairs(nbands,tiles))


def summary(z:np.ndarray,y:np.ndarray,graph:dict)->dict:
    return metrics(y,graph_predict(z,graph))


def synthetic_40(seed: int = 41, start: int = 0, n: int = 64) -> dict:
    """40 equal small coefficients, indices early or late; no CNN required."""
    rng=np.random.default_rng(seed)
    truth=set(range(start,start+40))
    if max(truth)>=n:raise ValueError("40 indices exceed width")
    oracle=lambda z:np.sum(z[:,sorted(truth)],axis=1)/np.sqrt(40.)
    selection=random_masks(rng,18000,n)
    fitting=random_masks(rng,7000,n)
    validation=random_masks(rng,4000,n)
    testing=random_masks(rng,4000,n)
    moments=degree12_moments(selection,oracle(selection))
    best=ranked_candidates(moments,50)
    recovered={tuple(t["indices"]) for t in best[:40]}
    model=fit_graph(fitting,oracle(fitting),best[:40])
    val=summary(validation,oracle(validation),model)
    test=summary(testing,oracle(testing),model)
    return {"start":start,"truth_found_in_top40":len(recovered & {(i,) for i in truth}),
            "total_selected":len(model["terms"]),"validation_r2":val["r2"],
            "test_r2":test["r2"],"max_pair_in_top40":sum(len(t["indices"])==2 for t in best[:40])}
