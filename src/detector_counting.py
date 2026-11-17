"""Trained per-cell detection and counting for blood-smear fields.

The field classifier in ``src/inference.py`` answers "does this field look
positive?" and the Cellpose pipeline in ``src/counting.py`` estimates counts
with a hand-written shape rule. This module is the third, strongest option:
a YOLOv8 detector trained on **SKDrepaData-v1** bounding boxes.

It detects individual red blood cells and assigns each one to one of three
classes:

- ``other``  - other abnormal shapes (e.g. codocytes, ovalocytes)
- ``sickle`` - sickle-shaped cells
- ``normal`` - normal round cells

Unlike the Cellpose heuristic, the class decision here comes from a model
trained on expert-annotated cells, so the counts have a real per-cell
ground-truth basis. They are still estimates on out-of-distribution images
(different stain, magnification, focus, or camera), and this remains a
research prototype - not a medical diagnostic device.

Weights are produced by ``scripts/train_detector.py`` (or the equivalent
``yolo detect train`` command) and copied to
``models/skdrepadata_yolov8n.pt``. See ``docs/skdrepadata.md``.
"""
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "skdrepadata_yolov8n.pt"

# Roboflow 5-class detector weights (trained on sickle-cell-detector v2)
ROBOFLOW_WEIGHTS = PROJECT_ROOT / "models" / "sickle5k_detector" / "weights" / "best.pt"

# SKDrepaData-v1 class order (see data/external/skdrepadata/skdrepadata.yaml).
CLASS_ORDER = ("other", "sickle", "normal")

# RGB colours used by render_overlay. "other" is orange so it is not confused
# with the red sickle boxes or green normal boxes.
CLASS_COLORS = {
    "other": (255, 165, 0),
    "sickle": (255, 70, 70),
    "normal": (40, 200, 90),
}

# Roboflow 5-class dataset class order
CLASS_ORDER_5 = ("Platelets", "RBC", "Spherocyte", "WBC", "sickle cell")

# Summary buckets used by the app (sickle / normal / other). Detector
# vocabularies differ, so each bucket lists the class names it includes.
SUMMARY_GROUPS = {
    "sickle": ("sickle",),
    "normal": ("normal",),
    "other": ("other",),
}
SUMMARY_GROUPS_5 = {
    "sickle": ("sickle cell",),
    "normal": ("RBC",),
    "other": ("Platelets", "Spherocyte", "WBC"),
}

# RGB colours for 5-class overlay.
# Using distinct colors per class from the existing palette plus new ones.
CLASS_COLORS_5 = {
    "Platelets": (255, 180, 180),
    "RBC": (66, 135, 245),
    "Spherocyte": (255, 215, 0),
    "WBC": (148, 103, 189),
    "sickle cell": (255, 70, 70),
}


@dataclass
class DetectedCell:
    index: int
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    width: float
    height: float
    center_x: float
    center_y: float
    area: float

    def as_row(self):
        return asdict(self)


def load_detector(weights_path=DEFAULT_WEIGHTS):
    """Load the trained YOLOv8 detector.

    Imported lazily so the classifier, Cellpose pipeline, and CLI still work
    if ``ultralytics`` is not installed.
    """
    from ultralytics import YOLO

    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Detector weights not found: {weights_path}. Train the detector "
            "first (see docs/skdrepadata.md) or pass an explicit path."
        )
    return YOLO(str(weights_path))


def _to_rgb_array(image):
    """Convert a PIL image or array to a contiguous RGB uint8 array."""
    if isinstance(image, Image.Image):
        return np.ascontiguousarray(np.asarray(image.convert("RGB")))
    array = np.asarray(image)
    if array.ndim == 2:
        array = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
    elif array.ndim == 3 and array.shape[2] == 4:
        array = cv2.cvtColor(array, cv2.COLOR_RGBA2RGB)
    elif array.ndim == 3 and array.shape[2] == 3:
        array = array
    else:
        raise ValueError(f"Unsupported image shape: {array.shape}")
    return np.ascontiguousarray(array.astype(np.uint8))


def _ultralytics_device(device):
    """Map 'auto'/'cuda'/'cpu'/torch.device to an Ultralytics device value."""
    if hasattr(device, "type"):  # torch.device
        if device.type == "cuda":
            return str(device.index) if device.index is not None else "0"
        return "cpu"
    if device == "auto":
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        return "0"
    return device


def detect_cells(
    model,
    image,
    conf=0.25,
    iou=0.5,
    imgsz=640,
    max_det=300,
    device="auto",
):
    """Detect and classify cells in one field.

    Returns a list of :class:`DetectedCell` in the original image coordinate
    system. Ultralytics expects BGR numpy arrays, so the RGB input is
    converted before prediction and the boxes are mapped back onto the RGB
    image by the caller.
    """
    rgb = _to_rgb_array(image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    results = model.predict(
        source=bgr,
        conf=float(conf),
        iou=float(iou),
        imgsz=int(imgsz),
        max_det=int(max_det),
        device=_ultralytics_device(device),
        verbose=False,
    )
    result = results[0]
    names = result.names
    boxes = result.boxes

    cells = []
    if boxes is None or len(boxes) == 0:
        return cells

    xyxy = boxes.xyxy.detach().cpu().numpy()
    confidences = boxes.conf.detach().cpu().numpy()
    class_ids = boxes.cls.detach().cpu().numpy().astype(int)

    for index, (box, score, class_id) in enumerate(
        zip(xyxy, confidences, class_ids), start=1
    ):
        x1, y1, x2, y2 = (float(v) for v in box)
        class_name = str(names.get(int(class_id), class_id))
        cells.append(DetectedCell(
            index=index,
            class_id=int(class_id),
            class_name=class_name,
            confidence=float(score),
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            width=max(x2 - x1, 0.0),
            height=max(y2 - y1, 0.0),
            center_x=(x1 + x2) / 2.0,
            center_y=(y1 + y2) / 2.0,
            area=max(x2 - x1, 0.0) * max(y2 - y1, 0.0),
        ))
    return cells


def _tile_starts(length, tile_size, stride):
    """Start coordinates for a sliding window, including the final edge."""
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def _shift_cell(cell, dx, dy):
    """Return a copy of ``cell`` shifted by (dx, dy) in image coordinates."""
    x1 = cell.x1 + dx
    y1 = cell.y1 + dy
    x2 = cell.x2 + dx
    y2 = cell.y2 + dy
    return DetectedCell(
        index=cell.index,
        class_id=cell.class_id,
        class_name=cell.class_name,
        confidence=cell.confidence,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        width=max(x2 - x1, 0.0),
        height=max(y2 - y1, 0.0),
        center_x=(x1 + x2) / 2.0,
        center_y=(y1 + y2) / 2.0,
        area=max(x2 - x1, 0.0) * max(y2 - y1, 0.0),
    )


def detect_cells_tiled(
    model,
    image,
    tile_size=640,
    overlap=0.2,
    tile_imgsz=1280,
    conf=0.25,
    iou=0.5,
    max_det=300,
    device="auto",
    edge_margin_ratio=0.1,
    merge_distance_ratio=0.5,
):
    """Detect cells on overlapping tiles of a high-resolution field.

    A whole 2800x2800 field resized to the detector's 640 px input makes
    cells far smaller than the close-up cells the detector was trained on.
    Tiling keeps native pixel scale, and ``tile_imgsz`` (default 1280)
    upscales each tile so cells are closer to the training scale.

    Each cell is assigned to the tile where its center is not near a tile
    edge (except at the image border), which removes most duplicate boxes.
    A final center-distance merge removes the rest. Returns a list of
    :class:`DetectedCell` in full-image coordinates.
    """
    rgb = _to_rgb_array(image)
    height, width = rgb.shape[:2]
    stride = max(1, int(round(tile_size * (1.0 - overlap))))
    xs = _tile_starts(width, tile_size, stride)
    ys = _tile_starts(height, tile_size, stride)
    margin = int(round(tile_size * edge_margin_ratio))

    collected = []
    for y in ys:
        for x in xs:
            tile = rgb[y:y + tile_size, x:x + tile_size]
            tile_cells = detect_cells(
                model,
                tile,
                conf=conf,
                iou=iou,
                imgsz=tile_imgsz,
                max_det=max_det,
                device=device,
            )
            for cell in tile_cells:
                global_cx = cell.center_x + x
                global_cy = cell.center_y + y
                if x > 0 and global_cx < x + margin:
                    continue
                if x + tile_size < width and global_cx > x + tile_size - margin:
                    continue
                if y > 0 and global_cy < y + margin:
                    continue
                if y + tile_size < height and global_cy > y + tile_size - margin:
                    continue
                collected.append(_shift_cell(cell, x, y))

    # Greedy center-distance merge. This is safer than box NMS for dense
    # fields because neighbouring cells can overlap without being duplicates.
    collected.sort(key=lambda cell: cell.confidence, reverse=True)
    kept = []
    for cell in collected:
        duplicate = False
        for other in kept:
            dx = cell.center_x - other.center_x
            dy = cell.center_y - other.center_y
            distance = (dx * dx + dy * dy) ** 0.5
            scale = min(cell.width, cell.height, other.width, other.height)
            if scale > 0 and distance < merge_distance_ratio * scale:
                duplicate = True
                break
        if not duplicate:
            kept.append(cell)

    for index, cell in enumerate(kept, start=1):
        cell.index = index
    return kept


def summarize_detections(cells, class_order=None, summary_groups=None):
    """Count detections per class and compute sickle fractions.

    ``class_order`` determines the expected class names for summarization.
    If ``None``, uses the default ``CLASS_ORDER`` (``("other", "sickle", "normal")``).
    ``summary_groups`` maps the reported ``sickle`` / ``normal`` / ``other``
    buckets to the detector's class names. It defaults to the vocabulary for
    the given ``class_order``, so the Roboflow 5-class model groups
    ``sickle cell`` as sickle and ``RBC`` as normal.

    ``sickle_fraction`` is the sickle share of cells that are either sickle or
    normal (the two classes the field classifier is trained to distinguish).
    ``other_fraction`` is the share of all detections that are other abnormal
    shapes. Both are 0.0 when the denominator is 0.
    """
    if class_order is None:
        class_order = CLASS_ORDER
    if summary_groups is None:
        summary_groups = (
            SUMMARY_GROUPS_5
            if tuple(class_order) == CLASS_ORDER_5
            else SUMMARY_GROUPS
        )
    counts = {name: 0 for name in class_order}
    for cell in cells:
        counts[cell.class_name] = counts.get(cell.class_name, 0) + 1

    def bucket(name):
        return sum(
            counts.get(class_name, 0)
            for class_name in summary_groups.get(name, ())
        )

    total = len(cells)
    sickle = bucket("sickle")
    normal = bucket("normal")
    other = bucket("other")
    typed = sickle + normal
    return {
        "total": total,
        "sickle": sickle,
        "normal": normal,
        "other": other,
        "sickle_fraction": (sickle / typed if typed else 0.0),
        "other_fraction": (other / total if total else 0.0),
        "class_counts": counts,
    }


def render_overlay(image, cells, line_thickness=2, font_scale=0.45, class_colors=None):
    """Draw colour-coded boxes with class names and confidences.

    Red = sickle, green = normal, orange = other abnormal. Boxes are drawn on
    a copy of the RGB image, so the caller's array is not modified.

    If ``class_colors`` is provided, it overrides the default ``CLASS_COLORS``
    for models with different class vocabularies.
    """
    if class_colors is None:
        class_colors = CLASS_COLORS
    vis = _to_rgb_array(image).copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_thickness = 1
    for cell in cells:
        color = class_colors.get(cell.class_name, (255, 255, 255))
        p1 = (int(round(cell.x1)), int(round(cell.y1)))
        p2 = (int(round(cell.x2)), int(round(cell.y2)))
        cv2.rectangle(vis, p1, p2, color, line_thickness)

        label = f"{cell.class_name} {cell.confidence:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label, font, font_scale, text_thickness
        )
        text_x = max(0, p1[0])
        text_y = max(text_h + 2, p1[1] - 3)
        cv2.rectangle(
            vis,
            (text_x, text_y - text_h - 2),
            (text_x + text_w + 2, text_y + baseline),
            color,
            thickness=-1,
        )
        cv2.putText(
            vis,
            label,
            (text_x + 1, text_y),
            font,
            font_scale,
            (20, 20, 20),
            text_thickness,
            lineType=cv2.LINE_AA,
        )
    return vis


def analyze_field(
    model,
    image,
    conf=0.25,
    iou=0.5,
    imgsz=640,
    max_det=300,
    device="auto",
    tile_large=False,
    tile_size=640,
    tile_imgsz=1280,
    class_order=None,
    class_colors=None,
    summary_groups=None,
):
    """End-to-end detection analysis for one field.

    When ``tile_large`` is true and the image is substantially larger than
    ``tile_size``, detection runs on overlapping tiles instead of shrinking
    the whole field to ``imgsz``. This preserves small cells on
    high-resolution fields.

    Optional ``class_order`` and ``class_colors`` override the default
    ``CLASS_ORDER`` / ``CLASS_COLORS`` for models with different class
    vocabularies (e.g. the 5-class Roboflow detector). ``summary_groups``
    controls how per-class detections are rolled up into the reported
    sickle / normal / other counts.

    Returns a dict with the RGB image, detected cells, summary counts, tiling
    metadata, and a colour-coded overlay.
    """
    if class_order is None:
        class_order = CLASS_ORDER
    if class_colors is None:
        class_colors = CLASS_COLORS
    rgb = _to_rgb_array(image)
    height, width = rgb.shape[:2]
    use_tiles = bool(
        tile_large and max(height, width) > tile_size * 1.5
    )

    if use_tiles:
        cells = detect_cells_tiled(
            model,
            rgb,
            tile_size=tile_size,
            tile_imgsz=tile_imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            device=device,
        )
        stride = max(1, int(round(tile_size * 0.8)))
        tiles = (
            len(_tile_starts(width, tile_size, stride))
            * len(_tile_starts(height, tile_size, stride))
        )
    else:
        cells = detect_cells(
            model,
            rgb,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            max_det=max_det,
            device=device,
        )
        tiles = 1

    summary = summarize_detections(
        cells, class_order=class_order, summary_groups=summary_groups
    )
    summary["tiled"] = use_tiles
    summary["tiles"] = tiles
    return {
        "image": rgb,
        "cells": cells,
        "summary": summary,
        "overlay": render_overlay(rgb, cells, class_colors=class_colors),
        "class_names": list(class_order),
    }
