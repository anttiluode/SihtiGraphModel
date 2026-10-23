# SihtiGraphModel v1 preregistration — 2026-10-23

**Locked before any real CIFAR/GPU experiment in this repository.** These settings and results are not to be retroactively reinterpreted as passing new thresholds. Pure NumPy oracle and PyTorch sieve smoke tests are development checks, not CIFAR evidence.

## Target, teacher and source image

- CIFAR-10; PyTorch convolutional image classifier trained on a fixed seeded permutation of 45,000 training examples, 5,000 validation examples, seed 41, AdamW LR 0.001, up to 8 epochs, best validation-accuracy checkpoint, no use of the CIFAR test set to train or choose the epoch.
- The target is the **bird** class (class index 2). Select the **first test-set bird correctly classified on the *fully visible image***; this conditioning is preregistered and selection must not look at mask results. Report chosen index and checkpoint validation accuracy; no claim about average CIFAR test images.
- Freeze all teacher weights; query the bird logit minus the largest other-class logit. This is a real continuous score, not a Boolean sign function.
- A 32×32 RGB original is decomposed into four telescoping Laplacian-like octave residuals (pooled/coarse interpolations at factors 2, 4, 8) and 16 disjoint 8×8 spatial tiles in each band, total `n=64` signs.
- Two independent masking functions: **zero** and **constant RGB CIFAR training mean** replacement. All 64 pieces enabled must reproduce the input to maximum absolute error <= 1e-5. Masks are rendered additively then clamped to [0,1]; record clipping frequency.

## Spectral protocol, fixed budgets

- Uniform random masks (each sign ±1 with equal probability) for coefficient calibration and graph discovery, using RNG seed 41. Independent refit/probe-mask stream seeded 58.
- KM-style tree of shared-sample pairwise bucket-mass estimates with `q=1024` pairs per depth, threshold `tau^2/2`, `tau=.16`, at most 64 active prefixes. Reuse a pair of teacher-response batches for every candidate at each level. Log candidate counts and whenever the active cap is reached; finite sampling and caps mean **no completeness guarantee**.
- Independent initial calibration: 2,048 masks; coefficient screening: 8,192; refit/equal-control probes: 8,192. Keep only estimated normalized coefficients with |theta_normalized| >= tau/2, at most 50 branches, then reestimate coefficients from the common refit probes.
- Comparators: intercept only, **all** 64 unary Fourier terms, top 50 unary+local-neighbor pair terms, plus order-matched random terms. All competitor coefficients use the same refit-probe observations as the extracted graph. This matches coefficient *fit data* but not the greater discovery-query cost of the graph; account for total forwards explicitly.
- Independent test masks: **2,048 each** at sign-on probabilities p=.5, .75, .9. Identical seeds and masks can be reused *between* models and replacement conditions for paired comparisons, but never for fitting.
- If any recovered branch has order >=3, choose the highest-ranked such branch. On 16 new p=.5 contexts, enumerate all 2^k assignments of that branch's k input toggles (skip order >6). Compare the exact finite mixed difference of original teacher and extracted graph. This is a matched input intervention; NOT a direct intervention on the trained CNN's weights or neurons.

## Interpretation gates (do not move)

- **G0** instrumentation: exact reconstruction error <=1e-5; original correctly classified; held-out teacher margin variance >1e-4 and both positive and negative teacher decisions occur at least 10% of the time under p=.5 masks. A degenerate teacher response cannot establish successful class binding even if R² is high.
- **G1** local approximation: held-out p=.5 continuous-score R² >= **.80**, <= **50** nonconstant branches, and a margin >= **+.05 R² above the complete 64-unary baseline**. If the approximation passes but simple unary terms tie, report an accurate sparse *functional* surrogate, **not** evidence that higher-order binding earns its complexity.
- **G2** retention shift: held-out p=.75 R² >= **.60**. p=.9 is reported, with no retroactive threshold.
- **G3** high-order intervention: when there is a degree>=3 retained branch, report its raw teacher mixed derivative amplitude and teacher-vs-surrogate mean absolute derivative error, normalized by calibration SD. No numeric PASS threshold is set for this exploratory diagnostic; no graph-imposed teacher neuron knockout claim.
- **G4** accounting: record search queries, *all* teacher forwards, mean/zero replacement effects, control comparison, sign balance and clipping. A method with high R² but catastrophic input/replacement sensitivity is **not** a validated general image explanation.

These are **per-image, per-replacement** results. Passing zero replacement does not transfer a PASS to mean replacement, and vice versa. The combined experimental claim is no stronger than the weaker demonstrated control condition.

## Missing and negative claims

This v1 does **not** recover internal teacher causal structure, contain trained Sihti recurrence/iterative image purification, produce a universal graph for unseen images, prove efficient high-order search at n=64, or guarantee low-degree concentration. If the search returns no graph, a zero-only intercept, or a graph worse than unary controls, that is the result. If all masked inputs force one class, stop claims about semantics. The next scientific test would require shared coefficients across independently held-out images and a separate mechanistically aligned intervention protocol.
