# SihtiGraphModel v2 — protocol locked before CIFAR rerun

This is separate from PREREG.md and the measured v1 results. v1 failed; none of its gates are redefined. This v2 measures an improved reader using exactly the old frozen teacher checkpoint, seed 41, CIFAR test bird image #65 and the four additive octave bands × sixteen tiles (64 ±1 inputs).

## Independent observations

For each of zero and CIFAR RGB-mean replacement, use:
- 20,000 uniformly masked (p=.5) **discovery** queries for every single and pair coefficient (64 + 2,016 terms). No recursive pruning, no hard-coded geometry.
- 8,192 independent p=.5 **fitting** masks for jointly ridge-fitting coefficients of selected branches, with an unpenalized intercept.
- 4,096 independent p=.5 **validation** masks to choose term count among 0, 5, 10, 20, 35, 50 for the unrestricted graph and predefined-local-geometry graph separately. The validation set never enters the fit.
- Three independent untouched **test** sets: 4,096 each at p=.5, .75, .9. Use identical test masks across compared models and across replacement conditions. No test labels are used for model selection.
- Controls: full 64 singletons; top candidate local singletons and local pairs with matched selection/fitting/validation budget; random matched-degree terms; constant only.
- Default teacher score bird logit minus largest competing logit, exactly as v1. Optional distinct runs with raw bird logit or difference to a **fixed** competing class to isolate changes in the winning competitor.
- Cache each split by checkpoint hash, original-image hash, replacement, readout and settings. Preserve all masks, scores, fitted graphs, held-out metrics and query counts.

## New preregistered gates (not v1 gates)

**S0:** Synthetic 40-term purely additive functions supported on coordinates 0–39 and 24–63 both yield exactly 40 recovered true singletons in the top 40 and held-out R² > .999. Independently selected three-way function with zero lower-order coefficients must be found by an optional full triple scan.

**G0:** full-piece reconstruction max error ≤1e-5; original bird recognized by the fixed CNN; masked teacher response variance >1e-4. If one class dominates masks, nominal classification accuracy alone is not evidence.

**G1:** unrestricted discovered graph of ≤50 terms at p=.5 must score R² ≥ .50 **and** be within .03 R² of the given-geometry local graph. Report both separate booleans.

**G2:** same graph R² at p=.75 ≥ .20 and at p=.90 ≥ 0. Both evaluated regardless of G1.

**G3:** Record sample separation, teacher forward queries (including cache hits), clipping, every graph's terms, validation selection and all p-shift metrics. Results are specific to a single masked input, not an across-images AI model or a faithful causal reconstruction of hidden CNN layers.

## Important limits

Sample Walsh moments are unbiased for pre-specified coefficients on uniform signs; selecting large sample moments creates upward selection bias. The independent fit and held-out test assess usefulness. Full pair enumeration can miss pure triples; optional independent exhaustive 41,664-triple scan does not require successful lower-degree ancestors, but involves many comparisons and is an **exploratory follow-up**, not a silent modification of default v2.

Both zero- and mean-substitution masks remain out-of-distribution for CIFAR. Zero clipping in mean-replacement is not proof of natural-feature identification. Changing competitor class inside the margin can itself create apparent high-order Fourier terms. Retention-shift failure is a genuine failure even if p=.5 passes. No claims of exact degree spectrum or guaranteed sparse graph recovery.

## Commands

Run in the existing clone with the checkpoint and dataset already present:

~~~powershell
git pull
python3.13 -m unittest discover -s tests -v
python3.13 run_v2.py --mode synthetic
python3.13 run_v2.py --mode cifar --device cuda
~~~

The final command does **not train** the CNN and writes only to runs/v2_bird_seed41/. If interrupted, reissue the same command: split responses and completed replacement conditions are reused. Use a different --out path for changed parameters or readout (e.g. --readout raw). The existing runs/v1_bird_seed41/ folder is never changed. Enable --scan-triples only as a separate, explicitly exploratory run because of the larger CPU cost.
