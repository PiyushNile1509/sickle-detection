"""Shared inference helpers for the trained whole-field classifier.

The classifier predicts whether a microscope field is negative or positive
for sickle cells. It does **not** detect or count individual cells - see
``src/counting.py`` for the separate heuristic counting pipeline.

Research prototype only. Not a medical diagnostic device.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.training import build_model

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DISCLAIMER = (
    "Research prototype only - NOT a medical diagnosis. "
    "Do not use this output for clinical decisions."
)


def resolve_device(device="auto"):
    """Resolve 'auto'/'cuda'/'cpu' to a torch.device, falling back to CPU
    when CUDA was requested but is unavailable.
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    required = {"model_name", "state_dict", "class_names"}
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint {path} is missing keys: {sorted(missing)}")
    return checkpoint


def build_inference_model(checkpoint, device):
    model = build_model(checkpoint["model_name"], pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def build_transform(checkpoint):
    image_size = checkpoint.get("image_size", 224)
    mean = checkpoint.get("imagenet_mean", [0.485, 0.456, 0.406])
    std = checkpoint.get("imagenet_std", [0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]), image_size


def predict_pil_image(model, transform, image, device):
    """Return (predicted_index, class_probabilities) for a PIL image."""
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().tolist()
    return int(np.argmax(probs)), probs


def predict_image_path(model, transform, image_path, device):
    image = Image.open(image_path).convert("RGB")
    return predict_pil_image(model, transform, image, device)


def _gradcam_tensor(rgb_img, checkpoint):
    mean = checkpoint.get("imagenet_mean", [0.485, 0.456, 0.406])
    std = checkpoint.get("imagenet_std", [0.229, 0.224, 0.225])
    tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).float()
    mean_t = torch.tensor(mean, dtype=tensor.dtype).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean_t) / std_t


def make_gradcam(model, checkpoint, image, device, target_index):
    """Return an RGB uint8 Grad-CAM overlay for a PIL image.

    ``target_index`` selects which class the CAM explains (0 = negative,
    1 = positive). The overlay is at the model's input resolution (usually
    224x224), not the original image resolution.
    """
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    if checkpoint["model_name"] == "resnet18":
        target_layers = [model.layer4[-1]]
    elif checkpoint["model_name"] == "mobilenet_v2":
        target_layers = [model.features[-1]]
    else:
        raise ValueError(f"Unsupported model: {checkpoint['model_name']}")

    image_size = checkpoint.get("image_size", 224)
    rgb_img = (
        np.asarray(
            image.convert("RGB").resize((image_size, image_size)),
            dtype=np.float32,
        )
        / 255.0
    ).astype(np.float32)
    tensor = _gradcam_tensor(rgb_img, checkpoint).unsqueeze(0).to(device)

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(
            input_tensor=tensor,
            targets=[ClassifierOutputTarget(target_index)],
        )[0]
    return show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)


def collect_images(input_path):
    """Return a list of image files from a file or directory path."""
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(
            p for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
    raise FileNotFoundError(f"Input not found: {input_path}")
