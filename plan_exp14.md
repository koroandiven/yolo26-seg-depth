# YOLO26-Seg-Depth exp14 Improvement Plan

This document records the rationale and concrete code changes that go into
`exp14`. It supersedes the proto-only baseline in `plan.md` after observing two
remaining issues at inference time:

1. Persistent left/right depth bias (left side is consistently predicted deeper).
2. Soft / blurry depth boundaries; the segmentation guidance does not appear to
   reach the depth output.

The changes below address the three highest-priority items identified during
the exp7/exp8/exp12 audit.

---

## 1. Diagnosis Summary

| Symptom | Root cause(s) |
|---|---|
| Left/right depth bias | (a) `SpatialBiasCorrection` learns an unconstrained 80×80 bias map that is added at every forward pass without any regularisation, (b) horizontal flip is disabled in `DepthSegmentDataset.build_transforms`, so the model never sees mirrored NYU samples, (c) `TaskDecouplingAttention` uses global average pooling, which feeds whole-image statistics back into the depth branch and encourages position-dependent priors. |
| Blurred boundaries | (a) `Proto26` features are low-frequency (3×3 conv + multi-scale add); plain concatenation lets `fusion_in` suppress the guidance channels, (b) guidance is injected only at P3 and then bilinearly upsampled ×8 to image resolution, (c) `MaskConsistencyDepthLoss` (variance + smoothness inside masks) actively flattens depth inside instances, while `MaskEdgeDepthLoss` requires GT masks and cannot help at inference. |
| exp12 `val/depth_loss == 0` | Stale EMA criterion / `_cached_batch` not propagated. Not addressed in exp14, treated as an evaluation pipeline issue. |

---

## 2. exp14 Scope

The three modifications applied for exp14 are:

1. **Remove `SpatialBiasCorrection` and enable depth-aware horizontal flip.**
2. **Replace proto guidance with a soft predicted mask union + edge map and
   inject it via a FiLM modulator.**
3. **Rebalance the depth loss (`edge=0.6`, `consistency=0.05`) and add an
   explicit image-edge alignment term that does not depend on GT masks.**

Items below describe each change, the principle, and the concrete files /
locations that were modified.

---

## 3. Change 1 — Remove SpatialBiasCorrection + Horizontal Flip with Depth

### Principle

`SpatialBiasCorrection` is an additive 80×80 learnable map placed at the very
end of the depth decoder. The original intent was: any residual dataset bias
(top-left always farther, etc.) should be absorbed into a single map that can
be turned off at inference. In practice no L1/L2 regularisation was attached,
so the network freely uses the bias map to memorise the NYU left-right asymmetry
that comes from Kinect's fixed pose, and that bias is then applied unconditionally
to every test image.

Horizontal flip is the textbook fix for left/right asymmetry. The reason it
was disabled in `DepthSegmentDataset` is that depth is a dense per-pixel target
and Ultralytics' standard `RandomFlip` only mirrors the image and instance
labels, not the depth array. Adding a small wrapper that flips the depth map
together with the image makes the augmentation safe for depth.

### Code changes

- `ultralytics/ultralytics/nn/modules/head.py`
  - `MaskGuidedDepthDecoder.__init__`: drop the `SpatialBiasCorrection` module.
  - `MaskGuidedDepthDecoder.forward`: drop the `bias_correction` call.
  - Old checkpoints may still contain `bias_correction.bias_map`. We accept the
    "missing key" warning instead of silently restoring the bias.
- `ultralytics/ultralytics/data/depth_dataset.py`
  - New `DepthRandomFlip` transform that mirrors `img`, `instances` and the
    pre-letterboxed `depth` array together when `random.random() < p`.
  - `build_transforms` inserts `DepthRandomFlip(p=hyp.fliplr)` after `LetterBox`
    so depth, instances and image are always mirrored consistently.

---

## 4. Change 2 — Soft Mask Union+Edge Guidance with FiLM

### Principle

Proto features are semantic-level, low-frequency, and identical for every
instance class. They do not give the depth decoder explicit instance boundaries,
so the convolutional fusion can easily ignore them.

A *soft predicted mask* is mathematically `sigmoid(MC @ proto)` where `MC` are
the predicted mask coefficients of the top-K detections. Combined with a Sobel
magnitude operator we get two channels that explicitly encode (a) where objects
are, (b) where their boundaries are. Both are produced by the model itself and
require no NMS.

To make sure the guidance is actually used (and not absorbed into noise by the
fusion conv), we modulate the depth feature with a FiLM
(`feature = feature * (1 + gamma) + beta`) layer where `gamma` and `beta` are
spatial maps produced from the guidance encoder. This is a multiplicative
injection that the network cannot trivially zero out.

The guidance is detached so depth gradients do not flow back into the (frozen)
segmentation path.

### Code changes

- `ultralytics/ultralytics/nn/modules/head.py`
  - `MaskGuidedDepthDecoder`: replace `mask_encoder` (2-ch, weak) with a
    `guidance_encoder` that processes a 2-channel `[soft_union, soft_edge]`
    map and outputs `2 * c_depth` channels that are split into `(gamma, beta)`.
    `gamma` initialises to 0 (identity FiLM) so the decoder can be loaded from
    earlier checkpoints without divergence.
  - The encoder now adds a FiLM modulator after `fusion_in`: depth_feat is
    refined inside the residual blocks under FiLM-modulated activations.
  - `default_mask_feat` becomes a learnable zero `(gamma, beta)` baseline so
    inference works when no mask is produced (e.g. empty scene).
- `DepthSegment26._extract_seg_guidance`: switched from extracting a `proto`
  tensor to building soft mask logits from predicted mask coefficients
  (`one2one` preferred for stability, `one2many` fallback). Top-K instances by
  cls score are selected, multiplied by proto, sigmoided and unioned. A Sobel
  magnitude is computed for the edge channel.
- The guidance is detached and resized to the depth feature resolution inside
  the decoder.

### Inference-time consistency

`Segment26.forward` already returns proto+predictions in both train and eval
modes; the helper now only requires the mask coefficient tensor that is also
available in both modes.

---

## 5. Change 3 — Loss Rebalance + Image-Edge Alignment

### Principle

The previous configuration was:
```
SILog + 0.5 * BerHu  (multi-scale)
EdgeAware smoothness   weight=0.1
MaskEdge edge_align    edge_weight=0.2
MaskConsistency        weight=0.15  (variance + in-mask smoothness)
```

The mask consistency term suppresses depth variance inside an instance. For
flat-ish objects this is fine, but for many NYU scenes (beds, sofas, complex
furniture) this aggressively flattens valid intra-object depth variation, which
both hurts validation RMSE and softens the perceived 3D structure. Combined
with a low edge weight (0.2), the gradient towards sharper transitions is weak.

We therefore:
- Raise `edge_weight` to 0.6 and lower `consistency_weight` to 0.05.
- Add `ImageEdgeAlignmentLoss`, an unsupervised term that encourages large
  depth gradients where the image has strong edges. Unlike `MaskEdgeDepthLoss`
  this needs no GT masks, so it is active at validation as well as training.

### Code changes

- `ultralytics/ultralytics/utils/loss.py`
  - New `ImageEdgeAlignmentLoss` class. It computes Sobel magnitude on RGB
    luminance (with a soft threshold) and Sobel magnitude on depth, then
    minimises `(image_edge * exp(-k * depth_grad)).mean()`. This is the
    complement of the existing smoothness term.
  - `MultiScaleDepthLoss.__init__` accepts `edge_weight=0.6`,
    `consistency_weight=0.05`, `image_edge_weight=0.2`.
  - `MultiScaleDepthLoss.forward` adds the new image-edge alignment term.
- `DepthSegmentationLoss.__init__` updated to construct the loss with the new
  weights.

---

## 6. Training Command (exp14)

```
python yolo26_train_depth.py \
  --model ultralytics/ultralytics/cfg/models/26/yolo26-seg-depth.yaml \
  --data nyu_yolo/nyu_depth_seg.yaml \
  --epochs 150 \
  --batch 8 \
  --pretrained yolo26s-seg.pt \
  --device 0 \
  --project runs/train_depth \
  --name yolo26-seg-depth-exp14 \
  --depth-weight 0.5 \
  --freeze-depth-epochs 50
```

Diagnostic checks after a few epochs:
1. `train/depth_loss` should decrease but be slightly higher than exp8
   (the mask consistency shortcut is gone).
2. `val/depth_loss` should be at least on par with exp8 (~0.76) and ideally
   below exp7's best (~0.736).
3. Visually: left/right disparity on the same scene flipped horizontally
   (`abs(D(x) - flip(D(flip(x)))).mean()`) should drop by an order of
   magnitude versus exp8.
4. Object boundaries should coincide more visibly with edges in the RGB
   visualisation when overlayed.

---

## 7. Out of Scope (deferred)

- EMA `criterion` re-init for `val/depth_loss==0` (exp12). To be handled
  separately as it is unrelated to model quality.
- High-resolution skip path (concat with RGB-derived high-frequency features).
  This is the next experiment if exp14 still shows residual blur.
- Switching `TaskDecouplingAttention` from GAP-channel attention to a spatial
  decoupling. Not done in exp14 to keep one variable per experiment isolated.
