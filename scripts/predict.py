"""Run inference on a blood-smear image (or a folder of images).

This is a research/portfolio prototype, NOT a medical diagnostic tool. Its
output must never be used as a diagnosis or as a substitute for laboratory
testing or a qualified medical professional.

Usage examples (run from the project root):

    python scripts/predict.py path/to/smear.jpg
    python scripts/predict.py path/to/folder --gradcam
    python scripts/predict.py path/to/smear.jpg --checkpoint models/resnet18_sickle.pt

Outputs:
    - prediction + class probabilities printed to the console
    - one JSON per image in results/predictions/
    - optional Grad-CAM overlay in results/predictions/
    - if the input is a folder, results/predictions/predictions.csv
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import (  # noqa: E402
    DISCLAIMER,
    build_inference_model,
    build_transform,
    collect_images,
    load_checkpoint,
    make_gradcam,
    predict_image_path,
    resolve_device,
)


def main():
    parser = argparse.ArgumentParser(
        description="Predict sickle-cell vs normal on blood-smear image(s). "
                    "Research prototype only - not a medical diagnosis."
    )
    parser.add_argument("input", help="Image file or folder of images")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a checkpoint saved by the final training run "
             "(default: MobileNetV2 if present, otherwise ResNet18)",
    )
    parser.add_argument(
        "--gradcam", action="store_true",
        help="Also save a Grad-CAM overlay showing which regions drove the prediction",
    )
    parser.add_argument(
        "--outdir", default=str(PROJECT_ROOT / "results" / "predictions"),
        help="Directory for JSON/CSV/Grad-CAM outputs",
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "cpu"],
        help="Inference device (default: auto)",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)

    images = collect_images(args.input)
    if not images:
        raise FileNotFoundError(f"No supported images found in {args.input}")

    if args.checkpoint is None:
        candidates = [
            PROJECT_ROOT / "models" / "mobilenet_v2_sickle.pt",
            PROJECT_ROOT / "models" / "resnet18_sickle.pt",
        ]
        existing = [p for p in candidates if p.exists()]
        if not existing:
            parser.error(
                "No checkpoint found. Train the final model first (see "
                "docs/inference.md) or pass --checkpoint explicitly."
            )
        args.checkpoint = str(existing[0])

    if not Path(args.checkpoint).exists():
        parser.error(f"Checkpoint not found: {args.checkpoint}")

    checkpoint = load_checkpoint(args.checkpoint, device)
    model = build_inference_model(checkpoint, device)
    transform, _ = build_transform(checkpoint)
    class_names = checkpoint["class_names"]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(DISCLAIMER)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model: {checkpoint['model_name']} | device: {device}")
    print(f"Images: {len(images)}")
    print()

    rows = []
    for image_path in images:
        predicted_idx, probs = predict_image_path(
            model, transform, image_path, device
        )
        predicted_name = class_names[predicted_idx]
        confidence = probs[predicted_idx]

        print(f"Image: {image_path}")
        print(f"  Prediction: {predicted_name.upper()}")
        print(f"  Confidence: {confidence:.1%}")
        for i, name in enumerate(class_names):
            print(f"    P({name}) = {probs[i]:.4f}")

        cam_path = None
        if args.gradcam:
            image = Image.open(image_path).convert("RGB")
            cam_rgb = make_gradcam(
                model, checkpoint, image, device, predicted_idx
            )
            cam_path = outdir / f"{image_path.stem}_gradcam.png"
            Image.fromarray(cam_rgb).save(cam_path)
            print(f"  Grad-CAM: {cam_path}")

        record = {
            "image": str(image_path),
            "model": checkpoint["model_name"],
            "checkpoint": str(args.checkpoint),
            "prediction": predicted_name,
            "predicted_label": predicted_idx,
            "confidence": confidence,
            "probabilities": {
                name: probs[i] for i, name in enumerate(class_names)
            },
            "gradcam": str(cam_path) if cam_path else None,
            "disclaimer": DISCLAIMER,
        }
        json_path = outdir / f"{image_path.stem}_prediction.json"
        with open(json_path, "w") as f:
            json.dump(record, f, indent=2)
        print(f"  Saved: {json_path}")
        print()
        rows.append(record)

    if len(images) > 1:
        import pandas as pd
        csv_path = outdir / "predictions.csv"
        pd.DataFrame([
            {
                "image": r["image"],
                "prediction": r["prediction"],
                "confidence": r["confidence"],
                "prob_negative": r["probabilities"]["negative"],
                "prob_positive": r["probabilities"]["positive"],
            }
            for r in rows
        ]).to_csv(csv_path, index=False)
        print(f"Saved batch summary: {csv_path}")


if __name__ == "__main__":
    main()
