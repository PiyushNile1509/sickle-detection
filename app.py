"""Streamlit interface for the sickle-cell research prototype.

Run from the project root:

    streamlit run app.py

The app has two independent parts:

1. **Field classification** - the trained CNN (MobileNetV2 or ResNet18)
   predicts whether the whole microscope field is negative or positive for
   sickle cells, with an optional Grad-CAM explanation.
2. **Cell counting** - two selectable pipelines:
   - a **YOLOv8 detector trained on SKDrepaData-v1** expert-annotated cells
     (preferred when the weights are present), which detects and classifies
     individual sickle / normal / other-abnormal cells;
   - the older **Cellpose + shape-rule heuristic**, which segments cells and
     labels them by elongation/circularity without per-cell training.

Both counting modes are separate from the field classifier. They are
estimates, not validated measurements.

Research prototype only. Not a medical diagnostic device.
"""
import io
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from src.counting import analyze_field, load_cellpose_model
from src.detector_counting import (
    DEFAULT_WEIGHTS as DETECTOR_WEIGHTS,
    ROBOFLOW_WEIGHTS,
    analyze_field as analyze_field_detector,
    load_detector,
)
from src.inference import (
    DISCLAIMER,
    build_inference_model,
    build_transform,
    load_checkpoint,
    make_gradcam,
    predict_pil_image,
    resolve_device,
)

PROJECT_ROOT = Path(__file__).resolve().parent

CHECKPOINTS = {
    "MobileNetV2 (smaller, faster)": PROJECT_ROOT / "models" / "mobilenet_v2_sickle.pt",
    "ResNet18 (alternative)": PROJECT_ROOT / "models" / "resnet18_sickle.pt",
}

SAMPLE_PATHS = [
    (
        "Positive sample 1",
        PROJECT_ROOT / "data" / "train_source" / "Positive" / "Unlabelled" / "1.jpg",
    ),
    (
        "Positive sample 115",
        PROJECT_ROOT / "data" / "train_source" / "Positive" / "Unlabelled" / "115.jpg",
    ),
    (
        "Negative sample 1",
        PROJECT_ROOT / "data" / "train_source" / "Negative" / "Clear" / "1.jpg",
    ),
    (
        "Negative sample 107",
        PROJECT_ROOT / "data" / "train_source" / "Negative" / "Clear" / "107.jpg",
    ),
]

DISCLAIMER_LONG = (
    "**Research prototype only - not a medical diagnostic device.** "
    "This app demonstrates a machine-learning workflow. It does not diagnose "
    "sickle cell disease and must never replace laboratory testing (e.g. "
    "hemoglobin electrophoresis) or a qualified medical professional."
)


# ---------------------------------------------------------------------------
# Cached model / analysis helpers
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_classifier(checkpoint_path: str, device_choice: str):
    device = resolve_device(device_choice)
    checkpoint = load_checkpoint(checkpoint_path, device)
    model = build_inference_model(checkpoint, device)
    transform, image_size = build_transform(checkpoint)
    return model, checkpoint, transform, image_size, device


@st.cache_resource(show_spinner=False)
def get_cellpose(use_gpu: bool):
    return load_cellpose_model(use_gpu=use_gpu)


@st.cache_resource(show_spinner=False)
def get_detector(weights_path: str, weights_mtime: float):
    # weights_mtime is part of the cache key so replacing the checkpoint
    # invalidates the cached model.
    return load_detector(weights_path)


def _png_bytes(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def _file_mtime(path):
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


@st.cache_data(show_spinner=False, max_entries=8)
def cached_gradcam(image_bytes, checkpoint_path, target_index, device_choice):
    model, checkpoint, _, _, device = get_classifier(
        checkpoint_path, device_choice
    )
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    cam = make_gradcam(model, checkpoint, image, device, target_index)
    return _png_bytes(cam)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_analysis(
    image_bytes,
    max_dim,
    diameter,
    cellprob_threshold,
    flow_threshold,
    min_size,
    min_area,
    ecc_threshold,
    circ_threshold,
    use_gpu,
):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    model = get_cellpose(use_gpu)
    result = analyze_field(
        model,
        image,
        max_dim=max_dim,
        diameter=diameter,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
        min_size=min_size,
        min_area=min_area,
        ecc_threshold=ecc_threshold,
        circ_threshold=circ_threshold,
    )
    cells_df = pd.DataFrame([c.as_row() for c in result["cells"]])
    return result["summary"], cells_df, _png_bytes(result["overlay"])


@st.cache_data(show_spinner=False, max_entries=8)
def cached_detector_analysis(
    image_bytes,
    conf_threshold,
    iou_threshold,
    imgsz,
    device_choice,
    weights_mtime,
    tile_large,
    detector_type,  # "skdrepadata" or "roboflow_5class"
):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if detector_type == "skdrepadata":
        model = get_detector(str(DETECTOR_WEIGHTS), weights_mtime)
        class_order = ("other", "sickle", "normal")
        class_colors = {
            "other": (255, 165, 0),
            "sickle": (255, 70, 70),
            "normal": (40, 200, 90),
        }
    elif detector_type == "roboflow_5class":
        model = get_detector(str(ROBOFLOW_WEIGHTS), weights_mtime)
        class_order = ("Platelets", "RBC", "Spherocyte", "WBC", "sickle cell")
        class_colors = {
            "Platelets": (255, 180, 180),
            "RBC": (66, 135, 245),
            "Spherocyte": (255, 215, 0),
            "WBC": (148, 103, 189),
            "sickle cell": (255, 70, 70),
        }
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")
    result = analyze_field_detector(
        model,
        image,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=imgsz,
        device=device_choice,
        tile_large=tile_large,
        class_order=class_order,
        class_colors=class_colors,
    )
    cells_df = pd.DataFrame([c.as_row() for c in result["cells"]])
    return result["summary"], cells_df, _png_bytes(result["overlay"])


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Sickle Cell Detection - Research Prototype",
    page_icon="🩸",
    layout="wide",
)

st.title("🩸 Sickle Cell Detection - Research Prototype")
st.markdown(
    "Upload a microscope field of a blood smear to get a **field-level "
    "prediction** from the trained CNN and an **estimated cell count** from "
    "a separate trained detector or segmentation pipeline."
)
st.warning(DISCLAIMER_LONG)

available_checkpoints = {
    label: path for label, path in CHECKPOINTS.items() if path.exists()
}
if not available_checkpoints:
    st.error(
        "No trained checkpoint found in `models/`. Train the final model first "
        "(see `docs/inference.md`) or place `mobilenet_v2_sickle.pt` / "
        "`resnet18_sickle.pt` in `models/`."
    )
    st.stop()

detector_available_skdrepadata = DETECTOR_WEIGHTS.exists()
detector_available_roboflow = ROBOFLOW_WEIGHTS.exists()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    model_label = st.radio(
        "Field classifier",
        list(available_checkpoints.keys()),
        index=0,
        key="field_classifier",
        help="Both models scored similarly in 5-fold cross-validation. "
             "MobileNetV2 is smaller and faster.",
    )
    checkpoint_path = available_checkpoints[model_label]

    device_choice = st.selectbox(
        "Compute device",
        ["auto", "cuda", "cpu"],
        index=0,
        key="device",
        help="'auto' uses CUDA when available, otherwise CPU.",
    )
    actual_device = resolve_device(device_choice)
    st.caption(f"Using: `{actual_device.type}`")
    if actual_device.type == "cpu":
        st.caption(
            "Cellpose segmentation can be slow on CPU; the trained detector "
            "is much faster. Choose 'Off' below if you only need the field "
            "prediction."
        )

    st.divider()
    counting_options = []
    if detector_available_skdrepadata:
        counting_options.append("Trained detector (SKDrepaData)")
    if detector_available_roboflow:
        counting_options.append("Trained detector (Roboflow 5-class)")
    counting_options.extend(["Cellpose shape heuristic", "Off"])
    counting_mode = st.radio(
        "Cell counting",
        counting_options,
        index=0,
        key="counting_mode",
        help="The trained detector counts individual cells. "
             "SKDrepaData: sickle/normal/other. Roboflow 5-class: Platelets/RBC/Spherocyte/WBC/sickle cell. "
             "The Cellpose heuristic segments cells and labels them by shape without per-cell training.",
    )
    estimate_counts = counting_mode != "Off"
    use_detector = counting_mode.startswith("Trained detector")
    if counting_mode == "Trained detector (SKDrepaData)":
        detector_type = "skdrepadata"
        detector_weights_path = DETECTOR_WEIGHTS
    elif counting_mode == "Trained detector (Roboflow 5-class)":
        detector_type = "roboflow_5class"
        detector_weights_path = ROBOFLOW_WEIGHTS
    else:
        detector_type = "skdrepadata"  # default, won't be used
        detector_weights_path = DETECTOR_WEIGHTS

    if not detector_available_skdrepadata and not detector_available_roboflow:
        st.caption(
            "Detector weights not found at "
            "`models/skdrepadata_yolov8n.pt` or "
            "`models/sickle5k_detector/weights/best.pt`; using the Cellpose "
            "heuristic. See `docs/skdrepadata.md` and "
            "`docs/supplementary_datasets.md` to train detectors."
        )

    # Defaults so the methodology section always has values to show.
    max_dim, diameter, min_area = 1024, 20, 80
    ecc_threshold, circ_threshold = 0.75, 0.80
    cellprob_threshold, flow_threshold, min_size = -2.0, 0.9, 5
    det_conf, det_iou, det_imgsz = 0.25, 0.50, 640
    det_tile = True

    with st.expander("Counting settings", expanded=False):
        if use_detector:
            det_conf = st.slider(
                "Detector confidence threshold",
                min_value=0.05, max_value=0.90, value=0.25, step=0.05,
                help="Lower values detect more faint cells but add false "
                     "positives.",
            )
            det_iou = st.slider(
                "Detector NMS IoU threshold",
                min_value=0.30, max_value=0.90, value=0.50, step=0.05,
                help="Lower values suppress overlapping boxes more "
                     "aggressively.",
            )
            det_imgsz = st.select_slider(
                "Detector inference size (px)",
                options=[480, 640, 800, 1024],
                value=640,
                help="The detector was trained at 640 px. Larger values can "
                     "help small cells but are slower and differ from the "
                     "training distribution.",
            )
            det_tile = st.checkbox(
                "Tile large images (high-resolution fields)",
                value=True,
                help="Runs the detector on overlapping 640 px tiles, each "
                     "upscaled to 1280 px, instead of shrinking the whole "
                     "field to 640 px. Recommended for images larger than "
                     "about 1000 px; slower on CPU.",
            )
            st.caption(
                "Defaults match the training configuration. The detector "
                "outputs **sickle**, **normal**, and **other abnormal** "
                "cells. Tiling keeps small cells visible on high-resolution "
                "fields; untick it for close-up images similar to "
                "SKDrepaData."
            )
        else:
            max_dim = st.select_slider(
                "Analysis resolution (longest side, px)",
                options=[512, 768, 1024, 1280, 1536],
                value=1024,
                help="Higher values preserve small cells but take longer.",
            )
            diameter = st.slider(
                "Approximate cell diameter (px at analysis resolution)",
                min_value=10, max_value=40, value=20, step=1,
                help="Calibrated at 1024 px. If you raise the analysis "
                     "resolution, increase this roughly proportionally.",
            )
            min_area = st.slider(
                "Minimum cell area (px²)",
                min_value=20, max_value=400, value=80, step=10,
                help="Objects smaller than this are treated as debris/noise.",
            )
            ecc_threshold = st.slider(
                "Sickle shape: minimum eccentricity",
                min_value=0.50, max_value=0.95, value=0.75, step=0.01,
                help="0 = circle, 1 = line. Higher is stricter.",
            )
            circ_threshold = st.slider(
                "Sickle shape: maximum circularity",
                min_value=0.50, max_value=1.00, value=0.80, step=0.01,
                help="1 = perfect circle. Lower is stricter.",
            )
            st.caption(
                "A cell is counted as sickle-like when **eccentricity ≥ "
                f"{ecc_threshold:.2f}** and **circularity ≤ {circ_threshold:.2f}**. "
                "The default values were sanity-checked on a few fields, not "
                "validated against per-cell ground truth."
            )
            cellprob_threshold = st.slider(
                "Cellpose cell-probability threshold",
                min_value=-3.0, max_value=1.0, value=-2.0, step=0.1,
                help="Lower values detect more (including fainter) objects.",
            )
            flow_threshold = st.slider(
                "Cellpose flow threshold",
                min_value=0.0, max_value=1.2, value=0.9, step=0.05,
                help="Higher values keep more masks but may include poorer "
                     "ones.",
            )
            min_size = st.slider(
                "Cellpose minimum mask size (px)",
                min_value=0, max_value=50, value=5, step=1,
            )

    st.divider()
    if st.button("Clear cached models/results"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

# ---------------------------------------------------------------------------
# Image input
# ---------------------------------------------------------------------------
uploaded = st.file_uploader(
    "Upload a blood-smear image",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
    help="A well-focused microscope field similar to the training images "
         "works best. Whole-slide scans or phone photos of a slide are "
         "out-of-distribution and may give unreliable results.",
)

available_samples = [
    (label, path) for label, path in SAMPLE_PATHS if path.exists()
]
sample_label = "None"
if available_samples:
    sample_label = st.selectbox(
        "Or try a local sample image from the dataset",
        ["None"] + [label for label, _ in available_samples],
        key="sample_image",
        help="These files are read from your local data/ directory; they are "
             "not bundled with the repository.",
    )

image_bytes = None
source_name = None
if uploaded is not None:
    image_bytes = uploaded.getvalue()
    source_name = uploaded.name
elif sample_label != "None":
    sample_path = dict(available_samples)[sample_label]
    image_bytes = sample_path.read_bytes()
    source_name = sample_path.relative_to(PROJECT_ROOT).as_posix()

if image_bytes is None:
    st.info(
        "Upload an image (or pick a local sample above) to start. "
        "The app expects a microscope field, not a single cropped cell."
    )
    st.stop()

try:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
except Exception as exc:
    st.error(f"Could not read the image: {exc}")
    st.stop()

st.caption(
    f"Input: `{source_name}` | {image.width} × {image.height} px | "
    f"analysis resolution ≤ {max_dim} px"
)

# ---------------------------------------------------------------------------
# Field-level classifier + Grad-CAM
# ---------------------------------------------------------------------------
try:
    model, checkpoint, transform, _, device = get_classifier(
        str(checkpoint_path), device_choice
    )
except Exception as exc:
    st.error(f"Could not load the classifier checkpoint: {exc}")
    st.stop()

pred_idx, probs = predict_pil_image(model, transform, image, device)
class_names = checkpoint["class_names"]
positive_idx = class_names.index("positive") if "positive" in class_names else 1
pred_name = class_names[pred_idx]
confidence = probs[pred_idx]
positive_prob = probs[positive_idx]

st.subheader("Field-level prediction")
st.caption(
    f"Model: `{checkpoint['model_name']}` | checkpoint: "
    f"`{checkpoint_path.name}`"
)

c1, c2, c3 = st.columns(3)
c1.metric("Prediction", pred_name.upper())
c2.metric("Confidence", f"{confidence:.1%}")
c3.metric("P(positive / sickle)", f"{positive_prob:.1%}")
st.progress(
    min(max(float(positive_prob), 0.0), 1.0),
    text=f"Positive-class probability: {positive_prob:.1%}",
)

img_col, cam_col = st.columns(2)
with img_col:
    st.image(image, caption="Input field", width="stretch")
with cam_col:
    try:
        with st.spinner("Generating Grad-CAM..."):
            cam_png = cached_gradcam(
                image_bytes, str(checkpoint_path), pred_idx, device_choice
            )
        st.image(
            cam_png,
            caption=f"Grad-CAM for the predicted class ({pred_name.upper()})",
            width="stretch",
        )
    except Exception as exc:
        st.warning(f"Grad-CAM could not be generated: {exc}")

st.caption(
    "The prediction is a whole-field classification. It is a model score, "
    "not a calibrated clinical probability."
)

# ---------------------------------------------------------------------------
# Cell counting (trained detector or Cellpose heuristic)
# ---------------------------------------------------------------------------
if estimate_counts and use_detector:
    st.subheader("Estimated cell counts")
    # Dynamic caption based on detector type
    if detector_type == "roboflow_5class":
        detector_label = "Roboflow 5-class detector"
        normal_label = "Normal RBCs"
        other_label = "Other cells"
        overlay_legend = (
            "Detections - red: sickle cell, blue: RBC, yellow: spherocyte, "
            "purple: WBC, pink: platelets"
        )
    else:
        detector_label = "SKDrepaData detector"
        normal_label = "Normal cells"
        other_label = "Other abnormal"
        overlay_legend = (
            "Detections - red: sickle, green: normal, orange: other abnormal"
        )
    st.caption(
        f"This section uses a **{detector_label}** trained on "
        "expert-annotated cells. It is a separate model from the field "
        "classifier above and counts individual cells."
    )

    try:
        with st.spinner("Detecting cells with the detector..."):
            summary, cells_df, overlay_png = cached_detector_analysis(
                image_bytes,
                det_conf,
                det_iou,
                det_imgsz,
                device_choice,
                _file_mtime(detector_weights_path),
                det_tile,
                detector_type,
            )
    except Exception as exc:
        st.error(f"Cell detection failed: {exc}")
        summary = None

    if summary is not None:
        if summary["total"] == 0:
            st.warning(
                "No cells were detected. Try lowering the detector confidence "
                "threshold or raising the inference size."
            )
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total detected cells", summary["total"])
            m2.metric("Sickle cells", summary["sickle"])
            m3.metric(normal_label, summary["normal"])
            m4.metric(other_label, summary["other"])

            typed_total = summary["sickle"] + summary["normal"]
            if typed_total:
                st.progress(
                    min(max(summary["sickle_fraction"], 0.0), 1.0),
                    text=(
                        "Sickle share of sickle + normal cells: "
                        f"{summary['sickle_fraction']:.1%}"
                    ),
                )

            st.image(
                overlay_png,
                caption=overlay_legend,
                width="stretch",
            )
            if summary.get("tiled"):
                st.caption(
                    f"Tiled detection: {summary['tiles']} overlapping "
                    "640 px tiles, each run at 1280 px."
                )

            if pred_name == "positive" and summary["sickle_fraction"] < 0.10:
                st.info(
                    "The field classifier says POSITIVE, but the detector "
                    "found few sickle cells. This can happen when the field "
                    "is borderline, when the detector misses faint or "
                    "overlapping cells, or when the classifier is relying on "
                    "texture rather than obvious cell shapes."
                )
            elif pred_name == "negative" and summary["sickle_fraction"] > 0.25:
                st.info(
                    "The field classifier says NEGATIVE, but the detector "
                    "found many sickle cells. Treat the count with caution - "
                    "different staining, focus, or magnification can shift "
                    "the detector's behavior."
                )

            with st.expander("Per-cell detections and CSV download"):
                st.dataframe(cells_df, width="stretch", hide_index=True)
                st.download_button(
                    "Download cell detections (CSV)",
                    data=cells_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{Path(source_name).stem}_detections.csv",
                    mime="text/csv",
                )

            st.caption(
                "Counts come from a detector trained on one dataset. Faint, "
                "overlapping, or partly cut-off cells may be missed, and "
                "images with a different stain, focus, or magnification are "
                "out-of-distribution."
            )

elif estimate_counts:
    st.subheader("Estimated cell counts")
    st.caption(
        "This section is a **separate heuristic pipeline**, not an output of "
        "the trained CNN. Cellpose segments cell-sized objects, then a shape "
        "rule labels each object as sickle-like or normal-like."
    )

    use_gpu = device.type == "cuda"
    try:
        with st.spinner(
            "Segmenting cells with Cellpose... (the first run may take a "
            "minute while the model loads)"
        ):
            summary, cells_df, overlay_png = cached_analysis(
                image_bytes,
                max_dim,
                diameter,
                cellprob_threshold,
                flow_threshold,
                min_size,
                min_area,
                ecc_threshold,
                circ_threshold,
                use_gpu,
            )
    except Exception as exc:
        st.error(f"Cell segmentation failed: {exc}")
        summary = None

    if summary is not None:
        if summary["total"] == 0:
            st.warning(
                "No cells were detected. Try a lower approximate cell "
                "diameter, a lower cell-probability threshold, or a lower "
                "minimum area."
            )
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total detected cells", summary["total"])
            m2.metric("Sickle-shaped (est.)", summary["sickle"])
            m3.metric("Normal-shaped (est.)", summary["normal"])
            m4.metric("Sickle fraction (est.)", f"{summary['sickle_fraction']:.1%}")

            st.image(
                overlay_png,
                caption=(
                    "Segmentation overlay - red: sickle-like shape, "
                    "green: normal-like shape"
                ),
                width="stretch",
            )

            if pred_name == "positive" and summary["sickle_fraction"] < 0.10:
                st.info(
                    "The field classifier says POSITIVE, but the shape "
                    "heuristic found few sickle-shaped cells. This can happen "
                    "when sickle cells are missed by segmentation, when the "
                    "field is borderline, or when the classifier is relying on "
                    "texture rather than obvious cell shapes."
                )
            elif pred_name == "negative" and summary["sickle_fraction"] > 0.25:
                st.info(
                    "The field classifier says NEGATIVE, but the shape "
                    "heuristic found many elongated cells. Treat the count "
                    "with caution - overlapping cells, staining artifacts, "
                    "and partial cells at the field edge can look elongated."
                )

            with st.expander("Per-cell measurements and CSV download"):
                st.dataframe(cells_df, width="stretch", hide_index=True)
                st.download_button(
                    "Download cell measurements (CSV)",
                    data=cells_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{Path(source_name).stem}_cells.csv",
                    mime="text/csv",
                )

            st.caption(
                "Counts are estimates. Overlapping, faint, or partly cut-off "
                "cells may be missed; the shape rule is not a trained "
                "per-cell classifier."
            )

# ---------------------------------------------------------------------------
# Methodology / limitations
# ---------------------------------------------------------------------------
if not estimate_counts:
    counting_md = (
        "**2. Cell counts are turned off.** Only the whole-field prediction "
        "and Grad-CAM are shown."
    )
elif use_detector:
    if detector_type == "roboflow_5class":
        counting_md = f"""**2. Estimated cell counts (trained detector).**
- The field is passed to a YOLOv8 detector trained on the **Roboflow
  5-class sickle-cell dataset** with expert bounding boxes.
- The detector predicts boxes and class scores for five classes:
  **Platelets**, **RBC**, **Spherocyte**, **WBC**, and **sickle cell**.
  The app reports **sickle cells**, **normal RBCs**, and the remaining
  classes together as **other cells**.
- Only detections above a confidence of {det_conf:.2f} are kept, and
  overlapping boxes are suppressed with an NMS IoU threshold of
  {det_iou:.2f}. Inference runs at {det_imgsz} px.
- The displayed **sickle fraction** is
  `sickle cell / (sickle cell + RBC)`."""
    else:
        counting_md = f"""**2. Estimated cell counts (trained detector).**
- The field is passed to a YOLOv8n detector trained on **SKDrepaData-v1**
  expert-annotated bounding boxes.
- The detector predicts boxes and class scores for three classes:
  **sickle**, **normal**, and **other abnormal** (e.g. codocytes,
  ovalocytes).
- Only detections above a confidence of {det_conf:.2f} are kept, and
  overlapping boxes are suppressed with an NMS IoU threshold of
  {det_iou:.2f}. Inference runs at {det_imgsz} px.
- The displayed **sickle fraction** is `sickle / (sickle + normal)`.
  Other abnormal cells are reported separately and excluded from that
  fraction."""
else:
    counting_md = f"""**2. Estimated cell counts (separate heuristic).**
- The input is resized so its longest side is at most {max_dim} px.
- Cellpose, a general-purpose pretrained segmentation model, proposes
  individual cell masks.
- Each mask is measured (area, eccentricity, circularity, solidity).
- A mask is counted as **sickle-like** when eccentricity ≥
  {ecc_threshold:.2f} and circularity ≤ {circ_threshold:.2f}; otherwise it is
  counted as **normal-like**."""

if use_detector:
    detector_dataset = (
        "the Roboflow 5-class dataset"
        if detector_type == "roboflow_5class"
        else "SKDrepaData"
    )
    meaning_md = f"""- The count is a model estimate, not a validated measurement.
- The detector was trained on one dataset ({detector_dataset}). Different
  stains, lighting, focus, magnification, or camera setups are
  out-of-distribution and can change both detection and class assignment.
- The field classifier and the detector can disagree. That is useful
  information, not a bug: one looks at the whole field's appearance, the
  other at individual annotated cells."""
else:
    meaning_md = """- The count is an image-processing estimate, not a validated measurement.
- There are no per-cell ground-truth annotations in the primary dataset, so
  the shape thresholds were only sanity-checked on a few fields.
- The field classifier and the shape heuristic can disagree. That is useful
  information, not a bug: one looks at the whole field's appearance, the
  other at segmented object shapes.
- Different stains, lighting, focus, magnification, or camera setups can
  change both the prediction and the segmentation."""

with st.expander("How the prediction and counts are produced"):
    st.markdown(
        f"""
**1. Field prediction (the trained model).**
The checkpoint (`{checkpoint['model_name']}`) is a CNN trained on 224×224
whole-field images. It outputs a negative/positive score for the entire
field. Grad-CAM highlights the regions that most influenced the selected
class. This is the only part of the app that uses the trained classifier.

{counting_md}

**What this means.**
{meaning_md}

**Not for clinical use.**
{DISCLAIMER}
"""
    )

st.divider()
st.caption(
    "Research/portfolio prototype. Counts are model estimates, not validated "
    "measurements. Do not use for diagnosis or clinical decision-making."
)
