"""Heuristic cell segmentation and counting for blood-smear fields.

The trained classifier in ``src/inference.py`` classifies whole microscope
fields. It does **not** detect or count individual cells. This module adds a
separate, clearly-labelled estimation pipeline:

1. Cellpose (a general-purpose pretrained cell segmentation model) proposes
   individual cell masks.
2. Simple shape measurements (eccentricity, circularity, solidity) decide
   whether each mask looks more "sickle-like" (elongated/curved) or
   "normal-like" (round).

There are no ground-truth per-cell annotations in this project, so these
counts are **estimates**, not validated measurements. Overlapping, faint, or
partly cut-off cells may be missed, and the shape rule is a heuristic rather
than a trained per-cell classifier.
"""
from dataclasses import asdict, dataclass

import cv2
import numpy as np
from PIL import Image
from skimage.measure import regionprops

# RGB colours used by render_overlay. The source image is RGB, so these are
# passed to cv2.drawContours directly (no BGR conversion).
SICKLE_COLOR = (255, 70, 70)
NORMAL_COLOR = (40, 200, 90)


@dataclass
class CellMeasurement:
    label: int
    area: float
    eccentricity: float
    solidity: float
    circularity: float
    aspect_ratio: float
    centroid_x: float
    centroid_y: float
    sickle_like: bool

    def as_row(self):
        return asdict(self)


def prepare_image(image, max_dim=1024):
    """Convert a PIL image or array to an RGB uint8 array, resized so the
    longest side is at most ``max_dim``. Returns (array, scale) where
    ``scale`` is the factor applied to the original image.
    """
    if isinstance(image, Image.Image):
        image = np.asarray(image.convert("RGB"))
    else:
        image = np.asarray(image)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    h, w = image.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1:
        image = cv2.resize(
            image,
            (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        scale = 1.0
    return np.ascontiguousarray(image), scale


def load_cellpose_model(use_gpu=True):
    """Load the pretrained Cellpose model (downloads weights on first use).

    Imported lazily so the classifier/CLI still work if Cellpose is not
    installed.
    """
    from cellpose import models

    return models.CellposeModel(gpu=use_gpu)


def segment_cells(
    model,
    image,
    diameter=20,
    cellprob_threshold=-2.0,
    flow_threshold=0.9,
    min_size=5,
):
    """Run Cellpose on an RGB uint8 image and return an integer label mask.

    The grayscale version of the image is used because it gave more complete
    detections than RGB on these faint brightfield fields during tuning.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    result = model.eval(
        gray_rgb,
        channels=[0, 0],
        diameter=diameter,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
        min_size=min_size,
    )
    return result[0]


def measure_cells(
    masks,
    min_area=80,
    max_area=None,
    ecc_threshold=0.75,
    circ_threshold=0.80,
):
    """Measure each segmented object and flag likely sickle-shaped cells.

    The shape rule is intentionally simple and transparent:
    ``eccentricity >= ecc_threshold`` and ``circularity <= circ_threshold``.
    A round red blood cell has low eccentricity and high circularity; an
    elongated/crescent cell tends to have high eccentricity and lower
    circularity.
    """
    cells = []
    for prop in regionprops(masks):
        if prop.area < min_area:
            continue
        if max_area is not None and prop.area > max_area:
            continue
        perimeter = prop.perimeter if prop.perimeter > 0 else 1e-6
        circularity = 4 * np.pi * prop.area / (perimeter ** 2)
        aspect_ratio = prop.axis_major_length / max(prop.axis_minor_length, 1e-6)
        sickle_like = bool(
            prop.eccentricity >= ecc_threshold
            and circularity <= circ_threshold
        )
        cells.append(CellMeasurement(
            label=int(prop.label),
            area=float(prop.area),
            eccentricity=float(prop.eccentricity),
            solidity=float(prop.solidity),
            circularity=float(circularity),
            aspect_ratio=float(aspect_ratio),
            centroid_x=float(prop.centroid[1]),
            centroid_y=float(prop.centroid[0]),
            sickle_like=sickle_like,
        ))
    return cells


def summarize_cells(cells):
    total = len(cells)
    sickle = sum(1 for c in cells if c.sickle_like)
    normal = total - sickle
    return {
        "total": total,
        "sickle": sickle,
        "normal": normal,
        "sickle_fraction": (sickle / total) if total else 0.0,
    }


def render_overlay(
    image,
    masks,
    cells,
    sickle_color=SICKLE_COLOR,
    normal_color=NORMAL_COLOR,
    line_thickness=2,
):
    """Draw colour-coded contours: red = sickle-like, green = normal-like."""
    vis = image.copy()
    for cell in cells:
        mask = (masks == cell.label).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        color = sickle_color if cell.sickle_like else normal_color
        cv2.drawContours(vis, contours, -1, color, line_thickness)
    return vis


def analyze_field(
    model,
    image,
    max_dim=1024,
    diameter=20,
    cellprob_threshold=-2.0,
    flow_threshold=0.9,
    min_size=5,
    min_area=80,
    max_area=None,
    ecc_threshold=0.75,
    circ_threshold=0.80,
):
    """End-to-end heuristic analysis for one field.

    Returns a dict with the prepared image, masks, cell measurements,
    summary counts, and colour-coded overlay.
    """
    prepared, scale = prepare_image(image, max_dim=max_dim)
    masks = segment_cells(
        model,
        prepared,
        diameter=diameter,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
        min_size=min_size,
    )
    cells = measure_cells(
        masks,
        min_area=min_area,
        max_area=max_area,
        ecc_threshold=ecc_threshold,
        circ_threshold=circ_threshold,
    )
    summary = summarize_cells(cells)
    overlay = render_overlay(prepared, masks, cells)
    return {
        "prepared": prepared,
        "scale": scale,
        "masks": masks,
        "cells": cells,
        "summary": summary,
        "overlay": overlay,
    }
