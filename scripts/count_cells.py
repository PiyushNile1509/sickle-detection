"""Count sickle / normal / other cells with the trained SKDrepaData detector.

This is a research/portfolio prototype, NOT a medical diagnostic tool. Its
output must never be used as a diagnosis or as a substitute for laboratory
testing or a qualified medical professional.

Usage examples (run from the project root):

    python scripts/count_cells.py path/to/smear.jpg
    python scripts/count_cells.py path/to/folder --conf 0.30
    python scripts/count_cells.py path/to/smear.jpg --weights models/skdrepadata_yolov8n.pt

Outputs:
    - per-class counts printed to the console
    - one JSON per image and a colour-coded overlay in results/detections/
    - if the input is a folder, results/detections/detections.csv
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.detector_counting import (  # noqa: E402
    DEFAULT_WEIGHTS,
    analyze_field,
    load_detector,
)
from src.inference import DISCLAIMER, collect_images, resolve_device  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Count sickle, normal, and other abnormal cells with the "
                    "SKDrepaData-trained detector. Research prototype only - "
                    "not a medical diagnosis."
    )
    parser.add_argument("input", help="Image file or folder of images")
    parser.add_argument(
        "--weights",
        default=str(DEFAULT_WEIGHTS),
        help="Detector weights (default: models/skdrepadata_yolov8n.pt)",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--iou", type=float, default=0.50,
        help="NMS IoU threshold (default: 0.50)",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Inference size in pixels (default: 640, the training size)",
    )
    parser.add_argument(
        "--max-det", type=int, default=300,
        help="Maximum detections per image (default: 300)",
    )
    parser.add_argument(
        "--no-tile", action="store_true",
        help="Disable tiled detection on large images (tiling is on by "
             "default and recommended for high-resolution fields)",
    )
    parser.add_argument(
        "--outdir", default=str(PROJECT_ROOT / "results" / "detections"),
        help="Directory for JSON/CSV/overlay outputs",
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "cpu"],
        help="Inference device (default: auto)",
    )
    args = parser.parse_args()

    if not Path(args.weights).exists():
        parser.error(
            f"Detector weights not found: {args.weights}. Train the detector "
            "first (see docs/skdrepadata.md)."
        )

    device = resolve_device(args.device)
    images = collect_images(args.input)
    if not images:
        raise FileNotFoundError(f"No supported images found in {args.input}")

    model = load_detector(args.weights)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(DISCLAIMER)
    print(f"Detector: {args.weights}")
    print(f"Device: {device} | conf: {args.conf} | IoU: {args.iou} | "
          f"imgsz: {args.imgsz}")
    print(f"Images: {len(images)}")
    print()

    rows = []
    for image_path in images:
        image = Image.open(image_path).convert("RGB")
        result = analyze_field(
            model,
            image,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            max_det=args.max_det,
            device=device,
            tile_large=not args.no_tile,
        )
        summary = result["summary"]

        print(f"Image: {image_path}")
        if summary.get("tiled"):
            print(f"  Tiled detection: {summary['tiles']} tiles at 1280 px")
        print(f"  Total detected: {summary['total']}")
        print(f"  Sickle:         {summary['sickle']}")
        print(f"  Normal:         {summary['normal']}")
        print(f"  Other abnormal: {summary['other']}")
        print(f"  Sickle fraction (sickle + normal): "
              f"{summary['sickle_fraction']:.1%}")

        overlay_path = outdir / f"{image_path.stem}_detections.png"
        Image.fromarray(result["overlay"]).save(overlay_path)
        print(f"  Overlay: {overlay_path}")

        record = {
            "image": str(image_path),
            "weights": str(args.weights),
            "conf": args.conf,
            "iou": args.iou,
            "imgsz": args.imgsz,
            "tiled": bool(summary.get("tiled")),
            "tiles": int(summary.get("tiles", 1)),
            "counts": {
                "total": summary["total"],
                "sickle": summary["sickle"],
                "normal": summary["normal"],
                "other": summary["other"],
            },
            "sickle_fraction": summary["sickle_fraction"],
            "other_fraction": summary["other_fraction"],
            "cells": [cell.as_row() for cell in result["cells"]],
            "overlay": str(overlay_path),
            "disclaimer": DISCLAIMER,
        }
        json_path = outdir / f"{image_path.stem}_detections.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        print(f"  Saved: {json_path}")
        print()

        rows.append({
            "image": str(image_path),
            "total": summary["total"],
            "sickle": summary["sickle"],
            "normal": summary["normal"],
            "other": summary["other"],
            "sickle_fraction": summary["sickle_fraction"],
        })

    if len(images) > 1:
        import pandas as pd
        csv_path = outdir / "detections.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"Saved batch summary: {csv_path}")


if __name__ == "__main__":
    main()
