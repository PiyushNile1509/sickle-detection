"""Compare field-classifier inference strategies on the held-out test split.

The trained checkpoints are unchanged; this script asks whether cheap
test-time changes (ensembling, horizontal-flip TTA, tiled inference on
high-resolution fields) actually improve held-out accuracy.

Usage (run from the project root):

    python scripts/evaluate_classifier.py
    python scripts/evaluate_classifier.py --device cuda --tile-size 512

Writes results/classifier_eval/metrics.json and per_image_predictions.csv.
Research prototype only.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import (  # noqa: E402
    build_inference_model,
    build_transform,
    load_checkpoint,
    resolve_device,
)

DEFAULT_CHECKPOINTS = [
    PROJECT_ROOT / "models" / "mobilenet_v2_sickle.pt",
    PROJECT_ROOT / "models" / "resnet18_sickle.pt",
]
DEFAULT_SPLITS = PROJECT_ROOT / "results" / "splits" / "primary_dataset_splits.csv"


def tile_starts(length, tile_size, stride):
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def load_models(paths, device):
    loaded = []
    for path in paths:
        checkpoint = load_checkpoint(str(path), device)
        model = build_inference_model(checkpoint, device)
        transform, image_size = build_transform(checkpoint)
        positive_idx = checkpoint["class_names"].index("positive")
        loaded.append({
            "name": checkpoint["model_name"],
            "path": str(path),
            "model": model,
            "transform": transform,
            "image_size": image_size,
            "positive_idx": positive_idx,
        })
    return loaded


def image_probabilities(models, image, device, tta=False):
    """Mean positive-class probability across models (and optional hflip)."""
    probabilities = []
    for entry in models:
        tensor = entry["transform"](image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = entry["model"](tensor)
            probabilities.append(
                torch.softmax(logits, dim=1)[0, entry["positive_idx"]].item()
            )
            if tta:
                flipped = torch.flip(tensor, dims=[3])
                logits_flip = entry["model"](flipped)
                probabilities.append(
                    torch.softmax(
                        logits_flip, dim=1
                    )[0, entry["positive_idx"]].item()
                )
    return float(np.mean(probabilities))


def tiled_probabilities(models, image, device, tile_size, stride):
    """Positive probability for each tile, averaged across models."""
    width, height = image.size
    xs = tile_starts(width, tile_size, stride)
    ys = tile_starts(height, tile_size, stride)
    tiles = [
        image.crop((x, y, min(x + tile_size, width), min(y + tile_size, height)))
        for y in ys for x in xs
    ]
    per_model = []
    for entry in models:
        tensors = torch.stack([entry["transform"](t) for t in tiles]).to(device)
        with torch.no_grad():
            logits = entry["model"](tensors)
            probs = torch.softmax(logits, dim=1)[:, entry["positive_idx"]]
        per_model.append(probs.cpu().numpy())
    return np.mean(per_model, axis=0)


def top_fraction_mean(values, fraction):
    values = np.sort(np.asarray(values))[::-1]
    k = max(1, int(np.ceil(len(values) * fraction)))
    return float(values[:k].mean())


def compute_metrics(labels, scores, threshold=0.5):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "sensitivity": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "threshold": threshold,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate classifier inference strategies on the held-out "
                    "test split. Research prototype only."
    )
    parser.add_argument(
        "--checkpoints", nargs="+", default=[str(p) for p in DEFAULT_CHECKPOINTS],
        help="One or more classifier checkpoints",
    )
    parser.add_argument("--splits", default=str(DEFAULT_SPLITS))
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--tile-stride", type=int, default=384)
    parser.add_argument(
        "--outdir",
        default=str(PROJECT_ROOT / "results" / "classifier_eval"),
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    splits = pd.read_csv(args.splits)
    test_df = splits[splits["final_split"] == args.split].reset_index(drop=True)
    if test_df.empty:
        raise SystemExit(f"No rows with final_split == {args.split!r}")

    models = load_models([Path(p) for p in args.checkpoints], device)
    names = [entry["name"] for entry in models]
    print(f"Device: {device}")
    print(f"Models: {names}")
    print(f"Test images: {len(test_df)}")

    rows = []
    for position, row in test_df.iterrows():
        image_path = PROJECT_ROOT / row["path"]
        image = Image.open(image_path).convert("RGB")
        label = int(row["label"])

        scores = {}
        scores["ensemble"] = image_probabilities(models, image, device)
        scores["ensemble_tta"] = image_probabilities(models, image, device, tta=True)
        for entry in models:
            single = [entry]
            scores[entry["name"]] = image_probabilities(single, image, device)
            scores[f"{entry['name']}_tta"] = image_probabilities(
                single, image, device, tta=True
            )

        tile_scores = tiled_probabilities(
            models, image, device, args.tile_size, args.tile_stride
        )
        scores["ensemble_tiled_mean"] = float(tile_scores.mean())
        scores["ensemble_tiled_max"] = float(tile_scores.max())
        scores["ensemble_tiled_top25"] = top_fraction_mean(tile_scores, 0.25)

        rows.append({
            "path": str(row["path"]),
            "label": label,
            "n_tiles": int(len(tile_scores)),
            **{key: float(value) for key, value in scores.items()},
        })
        if (position + 1) % 10 == 0 or position + 1 == len(test_df):
            print(f"  processed {position + 1}/{len(test_df)}")

    predictions = pd.DataFrame(rows)
    labels = predictions["label"].to_numpy()
    strategies = [
        column for column in predictions.columns
        if column not in {"path", "label", "n_tiles"}
    ]
    metrics = {
        strategy: compute_metrics(labels, predictions[strategy].to_numpy())
        for strategy in strategies
    }
    metrics["_meta"] = {
        "split": args.split,
        "n_images": len(predictions),
        "n_positive": int(labels.sum()),
        "n_negative": int((labels == 0).sum()),
        "checkpoints": [str(p) for p in args.checkpoints],
        "tile_size": args.tile_size,
        "tile_stride": args.tile_stride,
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(outdir / "per_image_predictions.csv", index=False)
    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    summary = pd.DataFrame([
        {"strategy": strategy, **values}
        for strategy, values in metrics.items()
        if strategy != "_meta"
    ]).sort_values("accuracy", ascending=False)
    print()
    print(summary.to_string(index=False))
    print()
    print(f"Saved to {outdir}")


if __name__ == "__main__":
    main()
