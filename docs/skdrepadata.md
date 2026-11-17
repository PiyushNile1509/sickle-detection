# SKDrepaData: Trained Per-Cell Detection and Counting

> **Research prototype only.** The detector counts cells; it does not diagnose
> sickle cell disease. Do not use its output for clinical decisions.

## Why this dataset

The project's original classifier is a **whole-field binary classifier**: one
score per microscope field. Its Cellpose-based counting mode segments cells
and applies a hand-written shape rule, with no per-cell training data.

SKDrepaData adds what the project was missing: **expert-annotated individual
red blood cells**. Training a detector on those boxes turns counting from a
shape heuristic into a supervised per-cell model that distinguishes:

- **sickle** cells,
- **normal** round cells,
- **other abnormal** cells (e.g. codocytes, ovalocytes, poikilocytes).

## Data source and license

The paper's Zenodo record (21365925) is restricted. The authors also published
an **open** three-part export:

- **Zenodo record:** https://zenodo.org/records/20719504
- **DOI:** 10.5281/zenodo.20719504
- **License:** CC BY 4.0
- **Total download:** ~275 MB (v1 ~40 MB, v2 ~235 MB, v3 ~18 KB)

| Subset | Contents | Used here |
|---|---|---|
| **v1** | 1,487 full-field 640×640 images with YOLO boxes | **Yes - training and evaluation** |
| **v2** | 33,832 class-organised 224×224 cell patches | No (no source-image IDs, so leakage-safe grouping is impossible) |
| **v3** | 33,835 rows of morphological descriptors | No (no class labels) |

### Getting the data

The files are public and small enough for a laptop (~275 MB total):

```powershell
$dir = "data/external/skdrepadata"
New-Item -ItemType Directory -Force -Path $dir | Out-Null
curl.exe -L -o "$dir/SKDrepaData-v1.zip" "https://zenodo.org/api/records/20719504/files/SKDrepaData-v1.zip/content"
curl.exe -L -o "$dir/SKDrepaData-v2.zip" "https://zenodo.org/api/records/20719504/files/SKDrepaData-v2.zip/content"
curl.exe -L -o "$dir/SKDrepaData-v3.zip" "https://zenodo.org/api/records/20719504/files/SKDrepaData-v3.zip/content"
Expand-Archive "$dir/SKDrepaData-v1.zip" -DestinationPath "$dir/SKDrepaData-v1"
Expand-Archive "$dir/SKDrepaData-v2.zip" -DestinationPath "$dir/SKDrepaData-v2"
Expand-Archive "$dir/SKDrepaData-v3.zip" -DestinationPath "$dir/SKDrepaData-v3"
```

Only v1 is needed for detector training. The extracted v1 path used by
`skdrepadata.yaml` is
`data/external/skdrepadata/SKDrepaData-v1/SKDrepaData-v1/`.

### v1 verification

- 1,487 images, 640×640, Roboflow export (CC BY 4.0 per `data.yaml`).
- Split by the export: **1,302 train / 123 valid / 62 test**.
- 973 source-image prefixes; **0 prefixes span splits**, so augmented versions
  of the same source field do not leak across train/valid/test.
- **46,847 valid YOLO boxes**, 0 malformed lines:

| Class ID | Name (French export) | Name (project) | Boxes |
|---:|---|---|---:|
| 0 | autres | other | 17,813 |
| 1 | falciformes | sickle | 10,134 |
| 2 | normales | normal | 18,900 |

Boxes per image: min 3, mean 31.5, max 63. Every image has a label file.

Class order is defined in
`data/external/skdrepadata/skdrepadata.yaml`.

## Training

The detector is **YOLOv8n** (Ultralytics), chosen to fit a 4 GB laptop GPU.

| Setting | Value |
|---|---|
| Base weights | `yolov8n.pt` (COCO pretrained) |
| Epochs | 80 (early-stopping patience 20) |
| Image size | 640 |
| Batch size | 16 |
| Optimizer | AdamW, automatic LR (Ultralytics `optimizer=auto`) |
| Seed | 42, deterministic |
| Augmentation | degrees 10, translate 0.1, scale 0.3, horizontal/vertical flip 0.5, HSV-V 0.2, mosaic 0.5 (off for last 10 epochs) |
| Device | NVIDIA RTX 3050 Laptop, 4.29 GB VRAM |

On Windows the first run used `workers=4` and crashed during validation with
`OSError: [WinError 1455] The paging file is too small` while spawning
dataloader workers. The successful run uses **`workers=0`** (in-process data
loading), which avoids loading CUDA DLLs in several child processes.

Reproduce with:

```bash
yolo detect train \
  data=data/external/skdrepadata/skdrepadata.yaml \
  model=yolov8n.pt epochs=80 imgsz=640 batch=16 device=0 workers=0 \
  patience=20 seed=42 deterministic=True optimizer=auto \
  degrees=10.0 translate=0.1 scale=0.3 fliplr=0.5 flipud=0.5 \
  hsv_h=0.0 hsv_s=0.0 hsv_v=0.2 mosaic=0.5 close_mosaic=10 \
  project=results/skdrepadata/runs name=detector_yolov8n
```

Artifacts: `results/skdrepadata/runs/detector_yolov8n/` (best/last weights,
curves, confusion matrix, validation plots).

## Evaluation

Run the held-out **v1 test split** evaluation with:

```bash
python scripts/evaluate_detector.py --split test
```

This writes `results/skdrepadata/evaluation/metrics.json`,
`per_class_metrics.csv`, and `confusion_matrix.csv`.

### Final metrics

_To be filled from `scripts/evaluate_detector.py` after the 80-epoch run._

For context, the one-epoch smoke test reached mAP50 **0.292** overall
(other 0.281, sickle 0.130, normal 0.464), confirming the pipeline learned
something before full training. Those numbers are not a result - they are a
sanity check.

## Using the detector

### Streamlit app

`app.py` exposes a **Cell counting** selector:

- **Trained detector (SKDrepaData)** - preferred when
  `models/skdrepadata_yolov8n.pt` exists.
- **Cellpose shape heuristic** - the older untrained-shape pipeline.
- **Off** - field prediction and Grad-CAM only.

Detector settings: confidence threshold, NMS IoU threshold, and inference
size. Defaults match training (conf 0.25, IoU 0.50, 640 px).

The app shows total, sickle, normal, and other-abnormal counts, the sickle
share of sickle + normal cells, and a colour-coded overlay (red sickle, green
normal, orange other). Per-cell boxes and confidences can be downloaded as
CSV.

### Command line

```bash
python scripts/count_cells.py path/to/smear.jpg
python scripts/count_cells.py path/to/folder --conf 0.30
```

Outputs JSON, PNG overlays, and (for folders) a CSV summary under
`results/detections/`.

### Python

```python
from src.detector_counting import load_detector, analyze_field

model = load_detector("models/skdrepadata_yolov8n.pt")
result = analyze_field(model, image, conf=0.25, iou=0.5, imgsz=640)
print(result["summary"])
```

## Important limitations

- **Single training dataset and magnification.** All boxes come from
  SKDrepaData-v1, whose 640×640 fields contain relatively large, close-up
  cells. A different stain, microscope, magnification, focus, or camera is
  out-of-distribution; counts may shift without warning. In particular,
  high-resolution fields with many small cells (like the primary Tushabe
  images) are undercounted at the training inference size - try a larger
  inference size, or use the Cellpose mode for those images.
- **Source-grouped, not patient-grouped, split.** The export does not include
  patient IDs. The split prevents augmented copies of the same source field
  from crossing splits, but it cannot prove patient-level separation.
- **Class imbalance.** Sickle cells are the minority class (10,134 of 46,847
  boxes, 21.6%), so sickle precision/recall are usually weaker than normal.
- **"Other" is not "normal".** Codocytes, ovalocytes, and other abnormal
  shapes are a separate class. The displayed sickle fraction excludes them;
  inspect the other-abnormal count as well.
- **Annotation and export limits.** Roboflow resized images to 640×640 with
  stretch, which can distort cells relative to the original fields.
- **Counting is not a clinical measurement.** A cell count from one field is
  not a patient's sickle-cell burden, and the model is not validated for
  diagnosis or treatment decisions.

## Citation

Baldé, A., Sall, O., Ossami, D.-D. O., Soumboundou, M., Sall, C., Faye, Y.,
M'Kacher, R., Bassene, A., Faty, L., Wane, P. S., Dione, M., & Niass, O.
(2026). *SKDrepaData: A Multisite West African Blood Smear Dataset for
Automated Sickle Cell Disease Detection and Classification.* MICCAI 2026
Satellite Events. Zenodo. https://doi.org/10.5281/zenodo.21365925

Open dataset export used here:

Baldé, A., Sall, C., Sall, O., Faye, Y., Soumboundou, M., M'Kacher, R.,
Ossami, D.-D. O., Bassene, A., & Faty, L. (2026). *SKDrepaData: Dataset pour
la détection et classification de cellules falciformes (Drépanocytose).*
Zenodo. https://doi.org/10.5281/zenodo.20719504 (CC BY 4.0)

See also `docs/citations.md`.
