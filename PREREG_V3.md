# SihtiGraphModel v3 — retention-mixture protocol (locked before real v3 CNN run)

This protocol is new. Do not modify v1's [PREREG.md](PREREG.md), v2's [PREREG_V2.md](PREREG_V2.md), or their recorded runs. **v1 target R²≥0.80 and v2 target R²≥0.50 were different preregistered thresholds; neither is silently changed here.** v2 passed its p=.5 graph-discovery gates but failed the high-retention transfer tests.

## Fixed teacher and image selection

- Exactly the original trained CNN (v1 checkpoint epoch 7; 77.46% validation), target CIFAR bird class 2.
- Image A is the original CIFAR test index **65**.
- Image B is selected *before looking at any masks*: the **first later test-set bird the same frozen CNN classifies correctly on the completely intact image**. Record its index.
- Primary replacement is fixed RGB **CIFAR mean**, no training of the classifier. Mean fill avoids clipping in v2, but still makes unnatural masked images. Zero replacement is optional and **separate**.
- Primary score remains bird logit minus largest competing class logit, for comparison with v2. The source/class can switch competitors as masks change. Raw bird logit and fixed competitor are optional separate investigations, never silently substituted for this protocol.

## Critical basis distinction

For each fixed mask-retention probability p, standardized input phi_i=(z_i-(2p−1))/sqrt(1−(2p−1)²), z_i∈{−1,+1}, yields a **p-biased Walsh basis**. Estimate all 64 singleton and 2,016 pair moments *separately within each p stratum*, normalize their magnitudes by that stratum's teacher-score SD, and rank candidates using equal-weight mean absolute normalized moment across strata.

The mixture of different p values does **not** preserve Walsh orthogonality. The top 50 terms are refit jointly as raw sign/product features using independent mixed training data. Treat the scores as candidate rankings, not the exact degree-energy spectrum.

A separate fixed-p=.90 model ranks on the p=.90 stratum, fits on independent p=.90 data and selects term count on p=.90 validation. It consumes the same cached teacher observations, but a *different number and distribution of fitting examples* from the mixture model; do not interpret its comparison as equal-compute superiority.

## Independent samples (per image and replacement)

Four p strata **0.50, 0.75, 0.90, 0.95**, equal weighting in training/validation:

- **Discovery**: 6,000 unique masks *per p* (24,000 per image).
- **Fit**: 4,000 independent masks *per p* (16,000 per image), joint ridge fit with intercept unpenalized, ridge 1e−4.
- **Validation**: 1,500 independent masks *per p* (6,000 per image), choose branch budget among 0, 10, 20, 35, 50 by equal-weight average per-stratum validation R². No held-out test selection.
- **Test**: 2,048 independent, fresh masks *per p* (8,192 per image). Same masks for all graph comparators at that image and replacement, including the frozen v2 graph.
- **Full image**: one all-on mask (p=1); report true bird-class margin, predicted margin and *absolute error*. R² is not defined on one intact image.
- For each image, compare unconstrained discovered mixture graph (≤50), predefined-local candidate graph (≤50), p=.90-only discovered graph (≤50), all 64 singletons jointly fitted on mixture, fitted constant. **Only for original image A**, also evaluate unchanged v2 learned graph coefficients on the v3 *new* held-out masks. v2 is a frozen comparator, not retrained and not selected using these tests.
- Both birds receive **independently fitted graphs**. Compare which same-tile B0↔B3 links occur (out of 16 possibilities), which exact tile indices overlap, and whether the sign matches. Repeatability of two local explanations is not cross-image transfer of fitted coefficients.

## Precommitted interpretation gates

- **S0 (instrument):** fixed-p=.90 synthetic with known unary and pure pair biased Fourier coefficients recovered within 0.065 and raw-feature joint refit on fresh masks gives R²>.999. Separately, uniform and biased raw-unary toy must not create a large phantom pair coefficient.
- **G0 (source):** intact teacher identifies both birds correctly; exactly reconstructed full sieve, max pixel error ≤1e−5; report clipping and masked-score sign imbalance.
- **G1 (near-intact):** mixture graph p=.90 held-out R² **≥0** on image #65, compared against frozen v2 on identical new masks; also report whether improvement in R² is ≥+.50. Report both, not an either/or success.
- **G2 (very-near-intact):** mixture graph p=.95 held-out R² **≥0** on image #65; and the full p=1 teacher margin error is explicitly reported (no invented p=1 R²).
- **G3 (no loss of original regime):** mixture graph p=.50 held-out R² **≥0.35** on image #65.
- **G4 (second image):** separately repeat p=.90 and p=.95 R² checks on prespecified second bird. Report same-tile B0↔B3 support and signed overlap as exploratory structural replication; **no pass threshold** set for the number of matching B0↔B3 links. Two pictures cannot establish a stable classifier-wide rule.
- **G5 (accountability):** separate sample generation, manifest tied to checkpoint/file hash and image, cache replay without altering clipping totals, unchanged historical v1/v2 outputs.

## Scope of evidence

A graph explaining heavily modified masks is not necessarily explaining natural bird recognition. Even p=.95 means around 3 of the 64 switched pieces are missing on average; p=1 tests only one exact image and gives absolute margin error. Learned graph stability on a second image is only a second local explanation. Moving competing logits can produce interactions independently of the CNN's internal binding. No full CNN causal circuit, universal image representation, general image completion, or accurate probability calibration is claimed.

## Local run

~~~powershell
git pull
python3.13 -m unittest discover -s tests -v
python3.13 run_v3.py --mode synthetic
python3.13 run_v3.py --mode cifar --device cuda
~~~

The last command reuses the existing trained checkpoint and downloaded CIFAR; **it never retrains**. Default output: runs/v3_birds_mixture_seed41/results.json and report.html. Interruption: rerun the exact command to reuse cached teacher responses; the manifest rejects changed parameters/checkpoints/images. Use a new --out directory for exploratory raw-logit or zero replacement runs. The v1/v2 directories are never written.
