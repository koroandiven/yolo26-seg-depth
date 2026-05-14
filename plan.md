# YOLO26-Seg-Depth Predicted Segmentation Guidance Plan

## 1. Goal

The current depth branch can learn from shared backbone/neck features, but prior experiments show two practical issues during inference:

- Depth edges are not sharp enough around object boundaries.
- Depth inside the same object can be spatially inconsistent, for example left/right depth drift.

GT mask guidance helped training loss, but it introduced a train/inference mismatch:

- Training used ground-truth masks from `batch["masks"]`.
- Validation and inference did not have GT masks and used `mask_guidance=None`.

The goal of this plan is to replace GT mask guidance with segmentation-head-derived guidance that is available in both training and inference.

Target behavior:

```text
image
  -> frozen COCO-pretrained backbone/neck/segmentation path
  -> predicted segmentation representation
  -> detached segmentation guidance
  -> depth decoder
  -> depth map
```

This keeps the segmentation model protected while allowing the depth head to use object boundary and region information during both training and inference.

## 2. Current Status And Latest Findings

### 2.1 Implemented Fixes Before Proto Guidance

The current codebase already includes several important stability fixes that should remain enabled for all following experiments:

- Depth target is now LetterBox-aligned with the input image instead of being stretched to square shape.
- Unsupported spatial augmentations for depth training are disabled; only safe color jitter is kept.
- Depth cache hash includes depth files to avoid stale cache after depth map updates.
- `task_attention.seg_branch` is frozen more strictly; only depth-related modules and `task_attention.depth_branch` are trainable.
- Depth loss and depth metrics use FP32-safe log/clamp operations for AMP stability.

These fixes are prerequisites for reliable depth validation. They address geometry and validation correctness rather than model capacity.

### 2.2 exp7 Training Result Summary

Latest exp7 CSV:

```text
runs/segment/runs/train_depth/yolo26-seg-depth-exp7/results.csv
```

exp7 currently reached epoch 127.

Key metrics:

```text
train/depth_loss: 4.00044 -> 0.47862, best 0.47862 @ epoch 127
val/depth_loss:   0.99169 -> 0.74608, best 0.73588 @ epoch 116
```

Interpretation:

- Training is technically correct: depth loss is computed, validation depth loss is non-zero, and the model learns.
- Segmentation-related losses are stable, so the freezing strategy is mostly protecting the segmentation path.
- exp7 strongly fits the training set, but validation improvement is limited.
- The late-stage train/val depth gap is large, about `0.27`.

This indicates that exp7 is not a clear improvement over exp5. exp7 likely still suffers from GT mask guidance train/eval mismatch:

```text
train: uses GT mask guidance
eval/inference: no GT mask guidance
```

Therefore exp7 should be treated as evidence that GT mask guidance can reduce training loss but does not reliably improve validation or inference behavior.

### 2.3 Current Code Change For exp8

The recommended proto-guidance implementation has now been added to:

```text
ultralytics/ultralytics/nn/modules/head.py
```

Current behavior:

```text
train: uses segmentation proto guidance from Segment26 output
eval:  uses segmentation proto guidance from Segment26 output
GT mask guidance: disabled in DepthSegment26.forward()
guidance detach: enabled inside decoder
```

Validation already performed:

```text
py_compile: passed
model build: passed
train forward: returns dict with depth
eval forward: returns tuple and injects preds_dict["depth"]
trainable check: depth_guidance_encoder is trainable, segmentation proto is frozen
```

This makes exp8 the next required experiment.

## 3. Design Principles

1. Do not use GT mask guidance as the main training signal.

GT masks are not available at inference time. Using them during training makes `train/depth_loss` overly optimistic and can increase the train/val gap.

2. Use segmentation information that exists during inference.

The depth branch should consume only information produced by the model itself, such as segmentation proto features, mask features, or predicted mask maps.

3. Detach segmentation guidance.

Depth loss must not update the segmentation path.

```python
seg_guidance = seg_guidance.detach()
```

This protects COCO segmentation performance and respects the project constraint that segmentation-related parameters remain frozen.

4. Prefer soft guidance over hard masks.

Soft proto/mask features preserve uncertainty and avoid brittle threshold/NMS behavior.

5. Avoid forcing constant depth inside objects.

Segmentation guidance should encourage local consistency and boundary awareness, not force each object instance to have one constant depth.

## 4. Recommended Architecture

### 4.1 Phase 1: Segmentation Proto Feature Guidance

This is the recommended first implementation because it is the simplest and most stable.

Instead of generating final predicted masks through NMS, use the segmentation head's internal proto feature as guidance.

Expected flow:

```text
P3/P4/P5 features
  -> DepthSegment26 task decoupling
  -> x_seg -> Segment26.forward()
      -> segmentation outputs + proto feature
  -> proto.detach()
  -> guidance encoder
  -> depth decoder fusion
```

Advantages:

- Single forward pass.
- No NMS inside training forward.
- No dependency on confidence thresholds.
- Works even when no object is detected.
- Uses segmentation structure learned from COCO pretraining.
- Training and inference can use the same guidance source.

### 4.2 Depth Decoder Change

Current decoder expects a 2-channel mask guidance tensor:

```text
[occupancy, edge]
```

New decoder should support segmentation feature guidance, for example 32-channel proto features:

```python
class MaskGuidedDepthDecoder(nn.Module):
    def __init__(self, ch, c_depth=128, seg_guidance_ch=32):
        self.depth_guidance_encoder = nn.Sequential(
            DepthConv(seg_guidance_ch, c_depth // 4, 3),
            DepthConv(c_depth // 4, c_depth // 2, 3),
        )

        self.default_guidance_feat = nn.Parameter(torch.zeros(1, c_depth // 2, 1, 1))
        self.fusion_in = DepthConv(c_depth * 3 + c_depth // 2, c_depth, k=3)
```

Forward behavior:

```python
if seg_guidance is not None:
    seg_guidance = F.interpolate(seg_guidance, size=d_p3.shape[-2:], mode="bilinear", align_corners=False)
    guidance_feat = self.depth_guidance_encoder(seg_guidance.detach())
else:
    guidance_feat = self.default_guidance_feat.expand(B, -1, H, W)
```

The default guidance path remains useful for ablation and backward compatibility.

## 5. Implementation Plan

Implementation status: Phase 1 proto guidance is implemented. The notes below document the intended behavior and serve as checklist for future refactors.

### Step 1: Inspect Segment26 Output Structure

File:

- `ultralytics/ultralytics/nn/modules/head.py`

Tasks:

- Confirm the exact return format of `Segment26.forward()` in training mode.
- Confirm the exact return format of `Segment26.forward()` in eval mode.
- Identify where proto features are available.

Expected possibilities:

```python
# training
outputs = {"one2many": ..., "one2one": ..., "proto": ...}

# eval
outputs = ((y, proto), preds_dict)
```

Actual implementation must follow the current code, not assumptions.

### Step 2: Add Proto Extraction Helper

Add a helper inside `DepthSegment26`:

```python
def _extract_seg_guidance(self, outputs):
    """Extract segmentation proto/features from Segment26 outputs."""
    proto = None

    if isinstance(outputs, dict):
        proto = outputs.get("proto")
        if proto is None and "one2many" in outputs:
            # inspect nested structure if needed
            pass

    elif isinstance(outputs, tuple):
        # eval path, likely ((y, proto), preds_dict)
        if len(outputs) > 0 and isinstance(outputs[0], tuple) and len(outputs[0]) > 1:
            proto = outputs[0][1]

    return proto.detach() if proto is not None else None
```

Requirements:

- Do not throw if proto is missing.
- Return `None` for unsupported output shapes.
- Add logging or debug assertions only during development, not noisy training logs.

Current implementation detail:

- Training end2end path checks `outputs["one2many"]["proto"]` and `outputs["one2one"]["proto"]`.
- Non-end2end path checks `outputs["proto"]`.
- Eval path checks `((y, proto), preds_dict)`.
- If `Proto26` returns `(proto, semseg)`, only the first tensor is used.

### Step 3: Replace GT Mask Guidance With Segmentation Guidance

File:

- `ultralytics/ultralytics/nn/modules/head.py`

Current behavior to remove or disable:

```python
if self.training and self._cached_batch is not None:
    masks = self._cached_batch.get("masks")
    ...
    mask_guidance = self._generate_mask_guidance(...)
```

New behavior:

```python
outputs = Segment26.forward(self, x_seg)
seg_guidance = self._extract_seg_guidance(outputs)
depth = self.mask_guided_depth_decoder(x_depth, seg_guidance)
```

Important:

- `seg_guidance` must come from segmentation head output, not GT labels.
- `seg_guidance` must be detached.
- Training and eval should use the same extraction logic.
- Keep `_cached_batch` only if still needed elsewhere, but depth guidance should not depend on it.

### Step 4: Modify MaskGuidedDepthDecoder

File:

- `ultralytics/ultralytics/nn/modules/head.py`

Tasks:

- Rename or generalize `mask_encoder` to `guidance_encoder`.
- Support multi-channel proto guidance.
- Keep compatibility with 2-channel mask guidance only if needed for ablation.
- Use `bilinear` interpolation for feature guidance.
- Keep no-BN depth convs to avoid spatial-statistics memorization.

Suggested behavior:

```python
if guidance is not None:
    if guidance.shape[-2:] != d_p3.shape[-2:]:
        guidance = F.interpolate(guidance, size=d_p3.shape[-2:], mode="bilinear", align_corners=False)
    guidance_feat = self.guidance_encoder(guidance.detach())
else:
    guidance_feat = self.default_guidance_feat.expand(d_p3.size(0), -1, d_p3.size(2), d_p3.size(3))
```

Current implementation detail:

- `depth_guidance_encoder` encodes `nm`-channel proto features.
- 2-channel mask guidance remains supported for ablation, but it is no longer used by `DepthSegment26.forward()`.
- The new module name includes `depth` so existing freeze filters keep it trainable.

### Step 5: Keep Segmentation Parameters Frozen

Files:

- `yolo26_train_depth.py`
- `ultralytics/ultralytics/models/yolo/segment/train.py`

Rules:

Only these parameters should be trainable:

```text
mask_guided_depth_decoder.*
task_attention.depth_branch.*
task_attention_p4.depth_branch.* if enabled
task_attention_p5.depth_branch.* if enabled
any explicitly depth-named modules
```

These must stay frozen:

```text
backbone
neck
Segment26 cv/proto/detect branches
task_attention.seg_branch
```

Verification command idea:

```python
for name, p in model.named_parameters():
    if p.requires_grad:
        print(name)
```

Expected trainable names should only contain `depth` or `task_attention.depth_branch`.

### Step 6: Preserve Eval Output Depth Injection

Keep the existing eval fix:

```python
self._last_depth = depth
if isinstance(outputs, tuple) and len(outputs) == 2 and isinstance(outputs[1], dict):
    outputs[1]["depth"] = depth
```

This is necessary for validation loss and metrics.

## 6. Experiment Plan

### exp8: Proto Guidance Baseline

Purpose:

- Replace GT mask guidance with predicted segmentation proto guidance.
- Keep training and inference guidance source consistent.

Settings:

```text
backbone frozen: yes
segmentation head frozen: yes
task_attention.seg_branch frozen: yes
GT mask guidance: off
seg proto guidance: on
decouple_p4p5: False
depth_scale: 20.0
```

Expected metrics:

- `train/depth_loss` may be higher than exp7 because GT mask shortcut is removed.
- `val/depth_loss` should be comparable or better than exp7.
- train/val gap should shrink.
- Visual depth edges should be sharper than no-guidance baseline.

Additional expectation after exp7:

- Do not judge exp8 only by lower `train/depth_loss`. exp7 achieved very low training loss but had poor train/val gap.
- exp8 is successful if validation is more stable and visual inference improves, even if training loss is higher than exp7.

Success criteria:

```text
val/depth_loss <= exp7 best (~0.7359)
train/val gap < exp7 gap (~0.26)
visual edges improve over no-guidance baseline
COCO segmentation confidence remains preserved
```

Recommended early stopping / checkpoint selection:

```text
Prefer best.pt by val/depth_loss, not last.pt.
If val/depth_loss plateaus for 30 epochs and visual quality does not improve, stop and move to exp9.
```

### exp9: Proto Guidance With Simpler Depth Loss

Purpose:

- Check whether mask consistency / mask edge losses are over-regularizing.

Settings:

```text
seg proto guidance: on
MaskConsistencyDepthLoss: off or very low
MaskEdgeDepthLoss: reduced
main loss: SILog + BerHu
```

Success criteria:

- Better validation loss or better visual geometry than exp8.
- Less over-smoothing inside large objects or planar surfaces.

Trigger condition:

- Run exp9 if exp8 still shows large train/val gap or visibly over-smoothed depth.
- Also run exp9 if exp8 improves edges but worsens planar geometry.

### exp10: Predicted Soft Mask Edge Guidance

Only attempt after exp8 validates proto guidance.

Purpose:

- Use explicit predicted mask/edge maps rather than proto features.

Approach:

- Avoid full NMS in training if possible.
- Generate soft union/edge maps from mask logits or proto-derived features.
- Detach guidance.

Success criteria:

- Sharper object boundaries than proto guidance.
- No large train/val gap increase.
- No major inference slowdown unless acceptable.

Trigger condition:

- Run exp10 only if exp8/exp9 prove that proto guidance is stable but insufficient for boundary sharpness.
- Do not start exp10 before establishing a stable proto-guidance baseline.

## 7. Validation Plan

### 7.1 CSV Metrics

Track:

```text
train/depth_loss
val/depth_loss
train/val depth gap
val/box_loss
val/seg_loss
val/cls_loss
```

Do not use NYU mAP as proof of COCO segmentation quality.

Recommended numeric comparisons:

```text
exp5 best val/depth_loss: ~0.70791
exp7 best val/depth_loss: ~0.73588
exp7 late train/val gap:  ~0.27
```

exp8 should primarily aim to reduce the gap and improve inference consistency. A validation loss below exp7 is required; a validation loss below exp5 would be a strong result.

### 7.2 Visual Evaluation

Use the same fixed validation images for all experiments.

Compare:

```text
exp5 best
exp7 best
exp8 proto guidance
exp9 simplified loss
```

Use fixed images and identical color mapping for all comparisons. Dynamic min/max normalization can hide scale errors, so save both normalized visualization and raw metric summaries when possible.

Look for:

- Object boundary sharpness.
- Same-object depth consistency.
- Wall/floor/cabinet geometry.
- Left/right spatial bias.
- Over-smoothing.
- Depth bleeding across masks.

### 7.3 COCO Segmentation Preservation

Run fixed COCO-style images through:

```text
yolo26s-seg.pt
latest depth checkpoint
```

Check:

- Detection confidence.
- Mask quality.
- Class predictions.
- No obvious degradation of segmentation output.

Expected behavior:

- Depth training should not update `Segment26` proto/cv/detect branches.
- Segmentation confidence on COCO-like images should remain close to the original `yolo26s-seg.pt` baseline.

## 8. Risks And Mitigations

### Risk 1: Proto guidance is too weak

Mitigation:

- Add a lightweight edge extraction branch from proto features.
- Add predicted soft mask edge guidance in exp10.

### Risk 2: Guidance dominates RGB depth learning

Mitigation:

- Keep guidance feature weight small, for example multiply encoded guidance by `0.3` initially.
- Add guidance dropout, but apply it equally in train/eval ablation logic, not as GT-only behavior.
- Monitor no-guidance inference as an ablation.

### Risk 3: Depth loss contaminates segmentation path

Mitigation:

- Always detach guidance.
- Keep segmentation parameters frozen.
- Verify trainable parameter names before training.

### Risk 4: Predicted masks are wrong or sparse

Mitigation:

- Start with proto features instead of hard predicted masks.
- Avoid confidence threshold dependency in the first implementation.

### Risk 5: Large train/val gap remains

Mitigation:

- Simplify depth loss.
- Disable or reduce mask consistency loss.
- Ensure depth target geometry remains aligned.
- Verify train/eval guidance paths are identical.

### Risk 6: Proto guidance improves loss but not edge quality

Mitigation:

- Add an explicit edge extractor on top of proto features.
- Use a shallow `depth_edge_guidance_encoder` with Sobel-like learned filters.
- Move to exp10 predicted soft mask edge guidance only after proto baseline is stable.

### Risk 7: Proto guidance is too semantic and not instance-specific enough

Mitigation:

- Use mask coefficients or raw mask logits to create soft instance union maps.
- Avoid NMS in training unless necessary.
- Detach any predicted mask guidance before feeding the depth decoder.

### Risk 8: Existing inference scripts patch DepthSegment26.forward()

Mitigation:

- Re-check `infer_depth_fixed.py` and `yolo26_inference.py` before using them with proto-guidance checkpoints.
- Prefer the native `DepthSegment26.forward()` path for exp8 evaluation.
- Remove or update monkey patches that manually bypass the current forward logic.

## 9. Immediate Next Steps

Run exp8 with the current code:

```text
GT mask guidance off
segmentation proto feature guidance on
seg guidance detached
segmentation path frozen
training and eval use same guidance extraction
```

Recommended command template:

```bash
python yolo26_train_depth.py \
  --model ultralytics/ultralytics/cfg/models/26/yolo26-seg-depth.yaml \
  --data nyu_yolo/nyu_depth_seg.yaml \
  --epochs 150 \
  --batch 8 \
  --pretrained yolo26s-seg.pt \
  --device 0 \
  --project runs/train_depth \
  --name yolo26-seg-depth-exp8-proto-guidance \
  --depth-weight 0.5 \
  --freeze-depth-epochs 50
```

Evaluation priority:

1. Check first 10 epochs to ensure `val/depth_loss` is non-zero and stable.
2. Check trainable parameter names and confirm segmentation proto is frozen.
3. Compare `train/depth_loss` and `val/depth_loss` gap after epoch 50.
4. Save fixed-image visualizations at best checkpoint.
5. Compare exp8 best against exp5 best and exp7 best.

Decision rule:

```text
If exp8 val/depth_loss <= 0.7359 and train/val gap is clearly smaller than exp7, continue proto guidance.
If exp8 val/depth_loss is worse but visuals are better, proceed to exp9 with simplified losses.
If exp8 neither improves metrics nor visuals, inspect proto extraction and consider explicit predicted mask edge guidance.
```
