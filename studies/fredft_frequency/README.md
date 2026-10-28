# FreDFT frequency-core adaptation: approved F/S comparison

User approved 2026-10-28: smoke first, then two seed1234/20-epoch runs on the
current RTX5070Ti. Reuse existing local BF16 baseline and NAF+original-decoder
results. Expand only if F exceeds both baseline and S.

Completed: all eight new20-epoch runs, including the triggered six-run
confirmation. Three-seed mean IoU baseline0.867876, F0.872085, S0.870004.
F gains0.4208pp over baseline and0.2081pp over S, with2/3 wins in both
comparisons. See [RESULTS.md](RESULTS.md) for complete results and boundaries;
[RESULTS_SEED1234.md](RESULTS_SEED1234.md) retains the initial comparison.

## Scientific contrast

Both variants keep the original Hiera/adapters, RFBs, three UpBlocks, three heads,
and NAF cascade/projections/GAP+RMS gate/zero output projection. Replace only the
two-stream selection/filter core by MFDA followed by shared FDFFL. Discard the
old selector, discrete sampling, 90% budget and score regularizer. No LFEM/CGMM,
new cross-scale skips, threshold adjustment, or extra loss.

- F (`fredft_frequency`): un-conjugated rFFT(Q)*rFFT(K) within each stream,
  inverse FFT, channel LayerNorm, multiply the other stream's V, projection and
  residual; then FDFFL with spectral channel mixing.
- S (`fredft_spatial`): replace spectral Q/K response by Q*K at each location,
  and execute FDFFL channel mixing directly in space. Identical parameter keys,
  dimensions, initial tensors and outside RNG. This is our matched control,
  not the paper's MSDA.

Follow official `models/common.py` at commit
`d275f96eca0090e2267c6b24ae01361bc98850c0`: Q/K/V width 2C, FFN expansion 3,
shared stream weights, channel LayerNorm eps1e-5, default FFT normalization,
3/5/7 depthwise branches, ReLU. Paper v2 equations (1)-(4), pages5-6.
FDFFL's channel permutation commutes with FFT; its space implementation is
mathematically equivalent apart from floating-point roundoff. Verify outputs
and task gradients. No claim of equivalence between spectral Q/K multiplication
and spatial pointwise multiplication.

## Execution

Use the unchanged audited loop from `tools/local_decoder_study.py` with explicit
process-local loss/gradient/context hooks installed by `tools/fredft_study.py`.
Snapshots include both drivers, all expanded baseline source, and the core.
No existing experiment or training driver is edited.

352, batch12/valid6, BF16, workers2, CPU4, strict CUDA, paired_aug_v2,
880/120 split, Structure main+D2+D3 all weights1, AdamW lr0.001 both groups,
weight_decay0.0005, 20-epoch CosineAnnealing to0. Original mask>0 and probability
FP32 sigmoid>0.5. Fresh official Hiera initialization for each formal run;
smoke updates never seed training. Three full checkpoints kept per run.

From workspace parent:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 NO_ALBUMENTATIONS_UPDATE=1 work/local_train_env/bin/python -u MSLAU_OPT/tools/fredft_study.py --mode smoke --variant fredft_frequency
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 NO_ALBUMENTATIONS_UPDATE=1 work/local_train_env/bin/python -u MSLAU_OPT/tools/fredft_study.py --mode smoke --variant fredft_spatial
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 NO_ALBUMENTATIONS_UPDATE=1 work/local_train_env/bin/python -u MSLAU_OPT/tools/fredft_study.py --mode train --variant fredft_frequency --seed 1234 --epochs 20
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 NO_ALBUMENTATIONS_UPDATE=1 work/local_train_env/bin/python -u MSLAU_OPT/tools/fredft_study.py --mode train --variant fredft_spatial --seed 1234 --epochs 20
```

Shared initialization/RNG and all40 input fingerprints must match cached local
seed1234 results. New trainable parameters:4,678,755 in both arms (baseline
4,269,747; old NAF+decoder4,325,865). F/S are an architecture experiment;
comparison to old NAF also removes its selector regularizer.

## Decision

Primary comparator: cached local baseline IoU0.8655597229177753. Secondary
reference: cached local NAF+decoder0.8646848477112751. Compare independently
reloaded best weights, all epochs, last-five metrics, per-image errors,
precision/recall, 4px boundary errors, time and memory. This120-image set was
used for checkpoint selection; it is not an independent test set.

If F>baseline and F>S, run baseline/F/S at seeds42 and2026 in this same local
environment. Do not reuse3090 results for those controls. Otherwise report the
observed result without automatically expanding training or tuning thresholds.
No significant/stable improvement claim from one seed.
