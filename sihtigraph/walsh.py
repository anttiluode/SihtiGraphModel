"""Pure NumPy Walsh Fourier tools for a black-box oracle with ±1 inputs.

The search is a Monte Carlo *KM-style* bucket descent. Its finite-sample
threshold has no guaranteed recall; the returned list is audited on fresh masks.
"""
from __future__ import annotations

import numpy as np


def random_masks(rng, nrows: int, n: int, keep: float = .5) -> np.ndarray:
    return np.where(rng.random((nrows, n)) < keep, 1, -1).astype(np.int8)


def parity(z: np.ndarray, subset: tuple[int, ...]) -> np.ndarray:
    return np.prod(z[:, list(subset)], axis=1) if subset else np.ones(len(z))


def predict(z: np.ndarray, terms: list[dict], intercept: float = 0.) -> np.ndarray:
    out = np.full(len(z), intercept, dtype=np.float64)
    for t in terms:
        out += float(t['coefficient']) * parity(z, tuple(t['indices']))
    return out


def fit_fourier(z: np.ndarray, scores: np.ndarray, subsets: list[tuple[int, ...]], top: int = 50,
                intercept: float | None = None) -> list[dict]:
    """MC moments on uniform masks. Does not claim they are exact coefficients."""
    if intercept is None:
        intercept = float(np.mean(scores))
    centered = np.asarray(scores, dtype=np.float64) - intercept
    terms = []
    for s in sorted(set(tuple(sorted(s)) for s in subsets)):
        if s:
            terms.append({'indices': [int(i) for i in s], 'coefficient': float(np.mean(centered * parity(z, s)))})
    return sorted(terms, key=lambda t: -abs(t['coefficient']))[:top]


def local_pairs(nbands: int = 4, tiles: int = 4) -> list[tuple[int, ...]]:
    """Across adjacent octave bands and within-band four-neighbor patch pairs."""
    n = tiles * tiles
    pairs = []
    for band in range(nbands):
        for row in range(tiles):
            for col in range(tiles):
                k = band * n + row * tiles + col
                if row + 1 < tiles: pairs.append((k, k + tiles))
                if col + 1 < tiles: pairs.append((k, k + 1))
                if band + 1 < nbands: pairs.append((k, k + n))
    return pairs


def km_search(oracle, n: int, seed: int = 41, tau: float = .16,
              q: int = 256, max_live: int = 64, coefficient_masks: int = 8192,
              calibration_masks: int = 2048, max_terms: int = 50) -> dict:
    """Query-only KM bucket-mass estimator; never reads hidden teacher weights.

    f is centered and normalized from independent uniform-mask calibration.
    A bucket (prefix subset a, length k) has Fourier mass
       E[f(x) chi_a(x) f(y) chi_a(y)],
    x,y uniform, sharing coordinates >= k. Negative estimates are possible.
    All requested forward evaluations count against query_count, including
    calibration and final moment estimates. Capping live buckets can miss terms.
    """
    rng = np.random.default_rng(seed)
    calls = 0
    zcal = random_masks(rng, calibration_masks, n)
    base_scores = np.asarray(oracle(zcal), dtype=np.float64).reshape(-1)
    calls += len(zcal)
    center = float(base_scores.mean())
    scale = float(base_scores.std())
    if scale < 1e-8:
        return {'terms': [], 'candidates': [], 'intercept': center, 'scale': scale,
                'query_count': calls, 'levels': [], 'status': 'degenerate teacher response'}

    def f(z):
        nonlocal calls
        calls += len(z)
        return (np.asarray(oracle(z), dtype=np.float64).reshape(-1) - center) / scale

    live = [()]
    levels = []
    for k in range(1, n + 1):
        # Reuse the same oracle pair evaluations for *all* candidate prefixes
        # at this depth. Only parity arithmetic scales with live buckets.
        # This is a shared-sample variant of KM-style bucket descent.
        x = random_masks(rng, q, n)
        y = x.copy()
        y[:, :k] = random_masks(rng, q, k)
        fx = f(x)
        fy = f(y)
        pair_scores = fx * fy
        options = []
        for a in live:
            for b in (a, a + (k - 1,)):
                mass = float(np.mean(pair_scores * parity(x, b) * parity(y, b)))
                if mass > tau * tau / 2:
                    options.append((mass, b))
        # A cap is a conscious compute budget, NOT a recall guarantee.
        options.sort(reverse=True, key=lambda a: a[0])
        live = [b for _, b in options[:max_live]]
        levels.append({'level': k, 'surviving': len(live),
                       'candidates_above_threshold': len(options),
                       'capped': len(options) > max_live})
        if not live: break

    zm = random_masks(rng, coefficient_masks, n)
    normalized = f(zm)
    estimates = fit_fourier(zm, normalized, live, top=max_terms, intercept=0.)
    kept = [dict(indices=t['indices'], coefficient=float(t['coefficient'] * scale))
            for t in estimates if abs(t['coefficient']) >= tau / 2]
    # Intercept is calibrated independently of the selection masks.
    return {'terms': kept, 'candidates': [list(s) for s in live], 'intercept': center,
            'scale': scale, 'query_count': calls, 'levels': levels,
            'status': 'ok', 'coefficient_masks': coefficient_masks,
            'calibration_masks': calibration_masks}


def r2(y: np.ndarray, yhat: np.ndarray) -> float | None:
    y, yhat = np.asarray(y, dtype=np.float64), np.asarray(yhat, dtype=np.float64)
    var = float(np.mean((y - y.mean()) ** 2))
    return 1 - float(np.mean((y - yhat) ** 2)) / var if var > 1e-12 else None


def metrics(y: np.ndarray, yhat: np.ndarray) -> dict:
    y, yhat = np.asarray(y), np.asarray(yhat)
    positive = float(np.mean(y >= 0))
    return {'r2': r2(y, yhat), 'mae': float(np.mean(np.abs(y - yhat))),
            'rmse': float(np.sqrt(np.mean((y - yhat) ** 2))),
            'sign_agreement': float(np.mean((y >= 0) == (yhat >= 0))),
            'teacher_positive_fraction': positive, 'teacher_variance': float(y.var())}


def fwht(a: np.ndarray) -> np.ndarray:
    """Exact unnormalized matrix-free Walsh transform in bitmask order."""
    out = np.asarray(a, dtype=np.float64).copy()
    if len(out) & (len(out) - 1): raise ValueError('input length must be a power of two')
    h = 1
    while h < len(out):
        t = out.reshape(-1, 2*h)
        out = np.concatenate((t[:, :h] + t[:, h:], t[:, :h] - t[:, h:]), 1).ravel()
        h *= 2
    return out / len(out)
