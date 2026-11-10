# Implementation record

This is a reconstruction from the published description, not recovered original
source and not a claim of exact numerical reproduction. No training results are
included in the initial package.

Paper: https://doi.org/10.1016/j.jrras.2025.101534
Upstream: https://github.com/WongKinYiu/yolov9
Pinned source commit: 5b1ea9a8b3f0ffe4fe0e203ec6232d788bb3fcff (GPL-3.0).

## Dataset and experiment differences

Use Roboflow taco-ihjgk/yolov8-trash-detections-kgnug version 11, generated
2024-10-09: 12070 exported images, 11076 train / 891 valid / 103 test.
https://universe.roboflow.com/taco-ihjgk/yolov8-trash-detections-kgnug/dataset/11
All 12 category names match the paper semantically. Count/date/classes support
this as a strong candidate; identity with the original experimental dataset
is not proven. The paper describes conflicting validation/test percentages;
neither equals this published export split. Preserve the export split and report
it explicitly. Do not randomly split augmented copies across train/test.
Training source images were augmented threefold in the export. Audit exact RGB
duplicates and Roboflow filename families across splits before training. This
does not establish absence of near duplicates or shared scene leakage.
Missing labels, malformed boxes, changed counts or overlaps stop training.

## Architecture choices

The official small config uses AConv, whereas the paper describes ADown.
Baseline remains official YOLOv9-s (12 outputs) with dual PGI training heads.
DynConv replaces backbone layers 0 and 1, preserving channels/stride. Four
expert kernels, global pooling + MLP with hidden max(8,C/4), softmax temperature
1, mixed kernels, BN and SiLU: explicit reconstruction choices because complete
original hyperparameters and placement are not available.
SDConvADown replaces main neck downsample layers 16 and 19. It follows ADown:
average-pool then split; first branch space-to-depth + stride-one 3x3 conv,
second branch max-pool + 1x1 conv. Replicate-padding makes the first branch even
after ADown pooling. Auxiliary heads and remaining backbone blocks are unchanged.
MPDIoU = IoU - (top-left squared distance + bottom-right squared distance) /
(input width squared + input height squared). Loss is 1 - MPDIoU; distances
are computed in pixels after restoring per-anchor stride, for BOTH branches.
DFL, task-aligned assignment and original box/class/DFL gains remain unchanged.

## Training protocol

Scratch initialization (empty weights), SGD, 200 maximum epochs, batch 8,
640 input, initial LR .01, momentum .931, weight decay .0005, patience 50,
mosaic 1.0 with close-mosaic last 15 planned epochs, seed 0, four workers.
Remaining scheduler/HSV/scale defaults are documented in each hyp YAML;
mixup/copy-paste disabled. Early stopping can occur before the planned
close-mosaic period. Checkpoint selection uses upstream validation fitness
(0.1*mAP50 + 0.9*mAP50–95), not the test set. Warn-only determinism permits
nondeterministic CUDA operations. One seed is not statistical significance.

Paper-style cumulative study: baseline, dyn, dyn-sd, full. Additional independent
sd-only and mpdiou-only configs are optional and must be declared before test
inspection. GPU queue runs sequentially, requires 12GB free and respects a
project lock; it does not kill or change other users' jobs. A completed marker
plus checkpoint hash permits skip; partial experiments are not auto-resumed.

## Evaluation and publication

Freeze configs/checkpoints before evaluating test. Use confidence .001,
NMS IoU .7, FP32 and image 640 for every variant. Report per-class AP and
test image/object counts, not just aggregate metrics. Timing depends on GPU,
batch and preprocessing. Upstream per-class maps assign aggregate values to
absent classes: inspect test class counts before quoting a class's AP.
Upstream validation-loss CSV columns are zero when no loss is computed; do
not describe these as measured losses. Bounding-box visualizations are detector
predictions; feature activation maps alone do not prove localization quality.

GitHub export excludes raw dataset images, labels, secrets, environments and
checkpoints by explicit allowlist. It contains code, configs, aggregate results
and plots. Weights remain local unless separately requested for publication.
Retain upstream license and cite the dataset (Roboflow lists CC BY 4.0).
