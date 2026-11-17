"""Evaluate a YOLOv8 detector on a labelled split.

Supports both the SKDrepaData 3-class detector and the Roboflow 5-class detector.

Usage (run from the project root):

    python scripts/evaluate_detector.py
    python scripts/evaluate_detector.py --split test
    python scripts/evaluate_detector.py --detector roboflow_5class --split test
    python scripts/evaluate_detector.py --weights models/skdrepadata_yolov8n.pt --split test
    python scripts/evaluate_detector.py --detector roboflow_5class --weights results/sickle_5class_runs/sickle5k_detector/weights/best.pt --split test

Writes overall metrics, a per-class table, and the confusion matrix to
results/skdrepadata-evaluation/ or results/sickle_5class_runs-sickle5k_detector-evaluation/.
Research prototype only.
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Defaults for the two supported detectors (used when --detector is specified
# without explicit --weights/--data)
DETECTOR_PRESETS = {
    "skdrepadata": {
        "weights": PROJECT_ROOT / "models" / "skdrepadata_yolov8n.pt",
        "data": PROJECT_ROOT / "data" / "external" / "skdrepadata" / "skdrepadata.yaml",
        "name": "SKDrepaData-v1 3-class detector",
    },
    "roboflow_5class": {
        "weights": PROJECT_ROOT / "models" / "sickle5k_detector" / "weights" / "best.pt",
        "data": PROJECT_ROOT / "scd_checked" / "data.yaml",
        "name": "Roboflow 5-class detector",
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a YOLOv8 detector on a labelled split. "
        "Supports SKDrepaData 3-class and Roboflow 5-class detectors."
    )
    parser.add_argument(
        "--detector", choices=list(DETECTOR_PRESETS.keys()), default="skdrepadata",
        help="Which detector to evaluate (default: skdrepadata)",
    )
    parser.add_argument("--weights", default=None,
        help="Path to weights file (.pt). If not specified, uses the preset for --detector.")
    parser.add_argument("--data", default=None,
        help="Path to data.yaml config file. If not specified, uses the preset for --detector.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
        help="Which data split to evaluate (default: test)")
    parser.add_argument("--imgsz", type=int, default=640,
        help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.001,
        help="Low confidence floor used for mAP computation")
    parser.add_argument("--iou", type=float, default=0.70,
        help="NMS IoU threshold during validation")
    parser.add_argument("--device", default="0",
        help="Device to run evaluation on (e.g. 0, cpu, auto)")
    parser.add_argument("--batch", type=int, default=16,
        help="Batch size for evaluation (default: 16, reduces OOM risk)")
    parser.add_argument(
        "--outdir",
        default=str(PROJECT_ROOT / "results"),
        help="Output directory for metrics",
    )
    args = parser.parse_args()

    # Determine which preset to use and get weights/data
    if args.detector in DETECTOR_PRESETS:
        preset = DETECTOR_PRESETS[args.detector]
        # Use preset weights/data if not explicitly overridden
        weights = Path(args.weights) if args.weights else Path(preset["weights"])
        data = Path(args.data) if args.data else Path(preset["data"])
        detector_name = preset["name"]
    else:
        # Custom paths - user specified their own weights and data
        weights = Path(args.weights)
        data = Path(args.data)
        detector_name = args.detector or "custom_detector"

    if not weights.exists():
        parser.error(f"Weights not found: {weights}")
    if not data.exists():
        parser.error(f"Data config not found: {data}")

    from ultralytics import YOLO

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data),
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        batch=args.batch,
        project=str(Path(args.outdir) / detector_name.replace(" ", "_").replace("-", "_")),
        name=f"{args.split}_val",
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    box = metrics.box
    names = metrics.names
    per_class = []
    for position, class_id in enumerate(box.ap_class_index):
        class_id = int(class_id)
        per_class.append({
            "class_id": class_id,
            "class_name": names.get(class_id, str(class_id)),
            "precision": float(box.p[position]),
            "recall": float(box.r[position]),
            "mAP50": float(box.ap50[position]),
            "mAP50-95": float(box.ap[position]),
        })

    overall = {
        "split": args.split,
        "weights": str(weights),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50-95": float(box.map),
        "per_class": per_class,
        "results_dict": {
            key: float(value)
            for key, value in metrics.results_dict.items()
        },
    }

    outdir = Path(args.outdir) / detector_name.replace(" ", "_").replace("-", "_")
    outdir.mkdir(parents=True, exist_ok=True)

    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)

    import pandas as pd
    pd.DataFrame(per_class).to_csv(outdir / "per_class_metrics.csv", index=False)

    try:
        matrix = metrics.confusion_matrix.matrix
        labels = [names[i] for i in sorted(names)]
        if matrix.shape[0] == len(labels) + 1:
            labels = labels + ["background"]
        pd.DataFrame(
            matrix, index=labels, columns=labels
        ).to_csv(outdir / "confusion_matrix.csv")
    except Exception as exc:
        print(f"Could not save confusion matrix: {exc}")

    print()
    print(f"Detector: {detector_name}")
    print(f"Split: {args.split} | weights: {weights}")
    print(f"Overall precision: {overall['precision']:.4f}")
    print(f"Overall recall:    {overall['recall']:.4f}")
    print(f"mAP50:             {overall['mAP50']:.4f}")
    print(f"mAP50-95:          {overall['mAP50-95']:.4f}")
    print()
    print(pd.DataFrame(per_class).to_string(index=False))
    print()
    print(f"Saved metrics to: {outdir}")


if __name__ == "__main__":
    main()