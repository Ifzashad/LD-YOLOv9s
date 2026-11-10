from pathlib import Path
import csv
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

root = Path(__file__).resolve().parent
run = root / "runs/study-20261107-173051"
out = root / "results"
out.mkdir(exist_ok=True)
models = ("baseline", "dyn", "dyn-sd", "full")

# Use the audit of the actual training data.
audit = json.loads((out / "dataset-audit.json").read_text())
counts = audit["counts"]
splits = ("train", "valid", "test")
classes = sorted(set().union(*(counts[s]["objects"] for s in splits)))

fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
for ax, split, color in zip(axes, splits, ("#147d82", "#7053a0", "#b45a32")):
    values = [counts[split]["objects"].get(c, 0) for c in classes]
    bars = ax.barh(classes, values, color=color)
    ax.bar_label(bars, padding=3, fontsize=8)
    ax.set_xlim(0, max(values) * 1.22 + 1)
    ax.set_title(f"{split.title()} · {counts[split]['images']:,} images")
    ax.set_xlabel("Annotated objects")
    ax.grid(axis="x", alpha=.15)
axes[0].invert_yaxis()
fig.suptitle("Litter annotations by class and dataset split")
fig.tight_layout()
fig.savefig(out / "dataset-distribution.png", dpi=170, bbox_inches="tight")
plt.close(fig)

# Keep measured scores together with each model's native plots.
for model in models:
    metric = run / f"{model}-test.json"
    if not metric.is_file():
        raise SystemExit(f"Missing test result: {metric}")
    train_dir = run / model
    test_dir = run / f"{model}-test"
    for name in ("results.png", "labels.jpg", "labels_correlogram.jpg"):
        src = train_dir / name
        if src.is_file():
            shutil.copy2(src, out / f"{model}-{name}")
    for name in ("confusion_matrix.png", "PR_curve.png", "F1_curve.png",
                 "P_curve.png", "R_curve.png"):
        src = test_dir / name
        if src.is_file():
            shutil.copy2(src, out / f"{model}-{name}")

# Same test images on both sides: dataset annotations vs model predictions.
test_dir = run / "full-test"
pairs = []
for labels in sorted(test_dir.glob("val_batch*_labels.jpg")):
    pred = labels.with_name(labels.name.replace("_labels.jpg", "_pred.jpg"))
    if pred.is_file():
        pairs.append((labels, pred))
if not pairs:
    raise SystemExit(f"No matching test preview images found in {test_dir}. "
                     "Dataset distribution is saved. Check the full-test log.")

for i, (labels, pred) in enumerate(pairs, 1):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, src, title in zip(axes, (labels, pred),
                              ("Ground-truth boxes", "Full LD-YOLOv9s predictions")):
        with Image.open(src) as image:
            ax.imshow(np.asarray(image.convert("RGB")))
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / f"qualitative-test-{i}.jpg", dpi=150, bbox_inches="tight")
    plt.close(fig)

# Add a gallery to the top-level README using repository-relative links.
marker = "<!-- ld-test-gallery -->"
readme = root / "README.md"
body = readme.read_text().split(marker)[0].rstrip()
gallery = [
    marker, "## Test results", "",
    "Evaluation on 103 held-out images. All values are percentages.", "",
    "| Model | Precision | Recall | mAP50 | mAP50–95 |",
    "|---|---:|---:|---:|---:|",
]
with (out / "comparison.csv").open() as f:
    for row in csv.DictReader(f):
        scores = [f"{100 * float(row[k]):.2f}" for k in
                  ("precision", "recall", "map50", "map50_95")]
        gallery.append("| " + " | ".join([row["variant"], *scores]) + " |")

gallery += [
    "", "### Qualitative test examples", "",
    "Each pair shows the same test images with annotated boxes on the left "
    "and predictions on the right. Examples follow test-loader order.", "",
    "![Ground truth and model predictions](results/qualitative-test-1.jpg)",
]
for i in range(2, len(pairs) + 1):
    gallery += ["", f"[View test batch {i}](results/qualitative-test-{i}.jpg)"]

gallery += [
    "", "### Dataset distribution", "",
    "Bars count annotated objects. Each panel uses its own horizontal scale.", "",
    "![Class distribution by split](results/dataset-distribution.png)",
]
for name, title in (
    ("full-PR_curve.png", "Full model precision–recall curve"),
    ("full-F1_curve.png", "F1 across confidence thresholds"),
    ("full-results.png", "Full model training and validation history"),
    ("full-labels.jpg", "Training labels and bounding-box distribution"),
    ("full-labels_correlogram.jpg", "Bounding-box coordinate correlations"),
):
    if (out / name).is_file():
        gallery += ["", f"[{title}](results/{name})"]
readme.write_text(body + "\n\n" + "\n".join(gallery) + "\n")
print(f"FIGURES READY: {len(pairs)} qualitative panels, dataset distribution, README updated.")
