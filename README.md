# SihtiGraphModel — image AI → explicit Sihti piece graph

**v1: an executable, falsifiable beginning.** This is a local black-box *behavioral* distillation experiment. A small CNN is trained on CIFAR-10; a fixed image is decomposed into **4 additive octave residues × 4 × 4 disjoint spatial tiles = 64 binary switches**. A query-only KM-style Walsh search proposes a sparse graph of interactions in that input basis. A local offline HTML report shows the recovered branches, their band/patch locations, held-out fidelity and controls.

**No CIFAR model training or full GPU distillation is claimed as completed.** The CPU synthetic spectral smoke and the local unit tests were run before this first code push. See [PREREG.md](PREREG.md) for the untouched numerical gates.

## v2 repair — direct graph reading (new, separate protocol)

**The completed v1 run failed**, and those results remain unchanged in [runs/v1_bird_seed41](runs/v1_bird_seed41). A pure-additive 40-term diagnostic showed that its noisy recursive KM search can discard a function that fits in the 50-branch budget. v2 fixes the **reader**, without changing the original image, CNN checkpoint or v1 measurements.

v2 directly estimates **all 64 single-piece and 2,016 pairwise interactions** on uniform masks, chooses a candidate graph without supplied spatial geometry, jointly refits its coefficients and compares it with the 50-term local-geometry baseline. It uses distinct *discovery → fitting → validation → held-out tests*, preserves p=.5/.75/.9 retention shifts, and stores teacher queries for safe restart. There is an optional independent full triple scan, pure-triple regression test and legacy coordinate-order diagnostic.

**No v2 CIFAR results are claimed until you run them on the trained CNN.** [PREREG_V2.md](PREREG_V2.md) locks the new test gates separately from v1. v2 is still a functional surrogate for **one selected image**, not a reconstruction of the CNN's hidden mechanism.

### Run v2 on your already-trained CNN (no re-download or retraining)

~~~powershell
git pull
python3.13 -m unittest discover -s tests -v
python3.13 run_v2.py --mode synthetic
python3.13 run_v2.py --mode cifar --device cuda
~~~

The default output is runs/v2_bird_seed41/results.json and runs/v2_bird_seed41/report.html. It reads the saved checkpoint at runs/checkpoints/cifar_tiny_cnn.pt and uses the already downloaded CIFAR data. If interrupted, run the same command: cached split responses and completed zero/mean conditions are reused. No overwrite flag or second training command is necessary.

Other exploratory probes (distinct output folders):

~~~powershell
python3.13 run_v2.py --mode legacy
python3.13 run_v2.py --mode cifar --device cuda --readout raw --out runs/v2_raw_bird_seed41
python3.13 run_v2.py --mode cifar --device cuda --scan-triples --out runs/v2_triples_bird_seed41
~~~

The last option scans 41,664 triples and may be CPU-intensive; it does not require surviving pairs. The raw readout isolates the bird logit from moving-class competition; its results are separate.

---

## How to run at home

On Windows, install a compatible CUDA-enabled PyTorch and torchvision pair using the command recommended by [PyTorch](https://pytorch.org/get-started/locally/) for your machine. Python 3.10+ is recommended. Then:

~~~powershell
git clone https://github.com/anttiluode/SihtiGraphModel.git
cd SihtiGraphModel
python -m pip install numpy torchvision
python -m unittest discover -s tests -v
python run.py --mode smoke
python run.py --mode all --download --device cuda
~~~

The last command trains a small CNN (8 epochs by default; 45k train and 5k fixed validation examples), saves the best checkpoint by validation accuracy and then performs the locked distillation. CIFAR-10 downloads on first run. It uses **the first correctly classified CIFAR test bird** under a fixed, documented selection rule, not a cherry-picked mask result; it conditions the experiment on the original being correctly recognized. The single image is *not* a benchmark of transfer across photographs.

When the experiment finishes, open:

- `runs/v1_bird_seed41/report.html`: no server, visual band/patch influence and comparison tables.
- `runs/v1_bird_seed41/results.json`: full metrics, candidate branch lists, query budgets and gates.
- `runs/v1_bird_seed41/config.json`: the invocation settings.
- `runs/checkpoints/cifar_tiny_cnn.pt` and adjacent `.train.json`: selected checkpoint and training history.

These generated files are ignored by Git. The default run evaluates **both** completely zeroed and mean-image replacement baselines; each produces its own learned graph because it defines a different query function. Prefer leaving all default knobs untouched for the first gate. If GPU memory becomes an issue, reducing `--batch 256` changes execution batching but not the planned mask distribution. Later reruns should use a different `--out` directory so the first receipt remains intact. Changing seeds, sample budgets, number of epochs or the source image is exploratory unless separately preregistered.

For a short preflight without CIFAR or GPU:

~~~powershell
python run.py --mode smoke
~~~

This checks exact known one-, two- and three-way Fourier recovery in a synthetic oracle, and exact multiscale image reassembly when torch is available.

## Recover after the NumPy `int64` JSON crash (October 2026)

If training completed but `run.py --mode all` failed with
`TypeError: Object of type int64 is not JSON serializable`, **do not train again**.
The completed teacher checkpoint remains at
`runs/checkpoints/cifar_tiny_cnn.pt` (the best validation epoch was already
saved before distillation began). The crash occurred *after* the zero-replacement
KM search had reached result serialization, so that search must be run again.

From the existing cloned repository:

~~~powershell
git pull
python3.13 -m unittest discover -s tests -v
python3.13 run.py --mode distill --device cuda
~~~

`--mode distill` loads the saved teacher checkpoint and the already-downloaded
CIFAR-10 data. It does **not** retrain the CNN or download the dataset again.
The failed run called `json.dumps` **before** writing `results.json`, so
the default output directory should still be reusable.

If `runs/v1_bird_seed41/results.json` already exists from a separate
completed run, keep it intact and choose a fresh folder instead:
`python3.13 run.py --mode distill --device cuda --out runs/v1_bird_seed41_resume`.
Do **not** add `--overwrite` merely to work around the crash; retaining the
original trained checkpoint and the unmodified sample budgets matters for
comparability.

This is a serialization repair, not a new distillation result. Report the
held-out graph metrics and both zero/mean replacement outcomes after rerunning
the inference phase.

## What the pieces really mean

An image `I` has four Laplacian-like octave components `R0..R3` with `sum(R)=I` to floating tolerance. Each component is partitioned into 16 **non-overlapping 8×8 tiles**, including all three RGB channels. If all 64 switches are on, the original 32×32 image is reconstructed **exactly**. With all switches off the replacement image is either black or a constant RGB CIFAR mean. Partial images are summed then clamped to valid RGB before entering the frozen classifier; clipping frequency is reported.

These are a valid *additive input coordinate system*, **not** naturally occluded photographs or interventions limited to one internal neuron. The pyramid is built by pooled/blended resampling and telescoping differences, **not** an exact disjoint Fourier bandpass and not an imported SighImageSuper operator. The claim is exact *reconstruction*, not unique semantic segmentation, input-distribution validity or exact image-frequency isolation.

For each gate vector `z_i∈{−1,+1}`, query the class-vs-best-other **logit margin** `f_I(z)`. The Fourier basis `chi_S(z)=product(z_i for i in S)` is complete for the **fixed image's masked-cube function**. It does not guarantee a compact expansion. The source network's hidden causal circuit is not directly recovered.

## Run design and baselines

1. Train a small CNN on the CIFAR training split, choosing checkpoint by validation only. The test set supplies **one correctly identified bird** by a deterministic policy. Freeze the CNN.
2. For each replacement condition, calibrate the distribution of logit margins over uniform 64-bit masks, run **sample-reusing KM-style prefix-mass search** (default 1,024 shared query pairs per level) and retain branches with normalized magnitude ≥ `tau/2`, capped at 50. If search caps buckets or loses candidates, those details remain in JSON.
3. Use **fresh uniform masks** for all estimated coefficients: graph branch list, all 64 singleton controls, top 50 local spatial/cross-band singleton-or-pair controls, and random matched-order edge controls.
4. Measure actual **held-out** teacher vs graph continuous-score R², RMSE, sign agreement and teacher positive fraction on separate masks retained with probability `0.5`, `0.75` and `0.9`.
5. If a degree≥3 branch survives, compute a true **factorial mixed difference** by changing every combination of its input pieces under 16 fresh random backgrounds; compare teacher vs graph mixed differences. This is a *matched input intervention*, **not** ablation of an actual CNN circuit or deleting an internal teacher connection.

The graph shown is **local to this single CIFAR image** and its replacement scheme. No sharing learned `theta` across different images is implemented. Query counts distinguish KM search from **total teacher forward examples** including coefficient fitting, held-out testing and factorial intervention.

## The crucial metric correction

For real-valued logit margins, `heldout_R2 = 1 − MSE(f,g) / Var(f)` using the held-out teacher variance. We do **not** call `1−MSE` a Parseval certificate or claim `classification_disagreement ≤ 1−R2`. A logit can change sign after an arbitrarily small perturbation near zero. Report decision agreement separately, including the teacher's positive fraction; an almost-always-negative masked teacher would make accuracy uninformative.

A small high-order Walsh coefficient is a joint input dependence in the *chosen* basis, not automatically evidence of a causal shortcut in the teacher's learned hidden mechanism.

## Existing lineage and next step

[SighImageSuper](https://github.com/anttiluode/SighImageSuper) → [Sihti](https://github.com/anttiluode/Sihti) / [SihtiImageModel](https://github.com/anttiluode/SihtiImageModel) → [BindingClaude](https://github.com/anttiluode/BindingClaude) (high-order toy constraints and masked settling) → [WhiteBox](https://github.com/anttiluode/WhiteBox) (exact/sampled Walsh extraction) → **this image-specific Sihti query instrument**.

The next stage, conditional on the first run, is a **shared graph tested on different held-out photographs**, followed by a separate recurrent completion experiment. Current v1 does *not* build BindingClaude's settling layer into an image classifier, and exact pyramid reconstruction does not demonstrate a new generative vision model.
