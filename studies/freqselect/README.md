# FreqSelect: authorized H/G/FS controlled study

User approved 2026-10-28: implement and smoke, then H/G/FS each at seeds
1234,42,2026 for20epochs on this5070Ti. Reuse the nine existing B/F/S results;
do not select seeds or expand the experiment based on an early result.

All new arms keep the NAF three-stage shell, local/context projections,
GAP+centered-RMS channel gate, zero output projection/residual and original
RFB/three UpBlocks/three heads. Remove selector, Gumbel sampling and selector
regularizer. No AdaKern, AdaDR, MFDA, extra skips or new losses.

- H (`freqselect_h`): identity inside the shell; no frequency/gate parameters.
- G (`freqselect_g`): four branches X/4; three learned spatial gates, lowest
  branch unchanged. Same gate parameters and complete initialization as FS.
- FS (`freqselect_fs`): four Fourier bands ordered high to low, split at
  normalized max(abs(fy),abs(fx))=1/4,1/8,1/16; three learned gates, lowest
  band unchanged. Each gate is Conv64->1,k3,p1 and2*sigmoid, zero initialized.
  Gate weights are shared between local/context streams at a stage, separate
  across stages. Additional gate parameters5193 relative to H.

Source: provided arXiv2403.05369v7, equations8-10 and official
Linwei-Chen/FADC commit923102e6344d9b90ec427cb15201efd514bf316e,
FADC_only/conv_custom.py. Explicit adaptation: use symmetric fftfreq masks
instead of rounded fftshift slices; normalized orthonormal rFFT/irFFT inFP32.
This is a two-stream FreqSelect adaptation, not full FADC reproduction.
Spatial gating creates a space-varying operator; FS is not a fixed global
spectral multiplier. G tests generic spatial attention and has three
redundant branches intentionally matching FS's parameterization.

352/b12/valid6/BF16/nw2/CPU4, paired_aug_v2 and strictCUDA, mask>0,
FP32 sigmoid>0.5, fresh official Hiera initialization, Structure main+D2+D3
weights1, AdamW groups both lr.001/wd.0005, Cosine20/eta_min0 unchanged.
Original training loop retained; phase state/RNG hashing is observation-only.
Three full best checkpoints retained. Smoke updates are discarded.

Primary: independently reloaded best validation per-image IoU, plus Dice.
Secondary: last5/10 means, per-image small-GT<5% group,4px boundary IoU,
farFP/FN and the two already fixed seed42 failure images. Report paired
seed differences and sampleSD, time/memory and all inputs/initial hashes.
These120validation images also select checkpoints; no independent test claim.
No additional training or hyperparameter search automatically follows.
