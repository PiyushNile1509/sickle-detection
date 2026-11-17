# Running Inference on a Blood-Smear Image

> **Research prototype only.** This model is not a medical diagnostic device.
> Never use its output to make clinical decisions or as a substitute for
> laboratory testing or a qualified medical professional.

## What you need

A trained checkpoint. Two are produced by the final-training step:

- `models/resnet18_sickle.pt`
- `models/mobilenet_v2_sickle.pt`

Both were trained on the full development set (390 train / 65 validation
images) with the same staged transfer-learning protocol used in
cross-validation. The held-out test set was not used.

## Single image

From the project root:

```bash
python scripts/predict.py path/to/blood_smear.jpg
```

Example output:

```
Research prototype only - NOT a medical diagnosis...
Model: mobilenet_v2 | device: cuda
Image: path/to/blood_smear.jpg
  Prediction: POSITIVE
  Confidence: 93.4%
    P(negative) = 0.0660
    P(positive) = 0.9340
```

The script writes a JSON record to `results/predictions/`.

## With Grad-CAM

Grad-CAM highlights the image regions that most influenced the positive-class
score. Useful for sanity-checking whether the model is looking at cells rather
than background/artifacts:

```bash
python scripts/predict.py path/to/blood_smear.jpg --gradcam
```

The overlay is saved as `results/predictions/<name>_gradcam.png`.

## A folder of images

```bash
python scripts/predict.py path/to/folder --gradcam
```

This writes one JSON per image plus a batch summary at
`results/predictions/predictions.csv`.

## Choosing a checkpoint

```bash
python scripts/predict.py image.jpg --checkpoint models/resnet18_sickle.pt
```

Default is `models/mobilenet_v2_sickle.pt`. Both models scored essentially the
same in 5-fold CV (92.5% vs 92.8% mean validation accuracy), so either is a
reasonable demo choice; MobileNetV2 is smaller/faster.

## Streamlit interface (upload + prediction + cell counts)

The repo includes a browser interface at `app.py`:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually <http://localhost:8501>).

What it shows:

1. **Field-level prediction** from the selected checkpoint (MobileNetV2 or
   ResNet18), with confidence and a Grad-CAM overlay.
2. **Cell counts** from one of two selectable pipelines:
   - **Trained detector (SKDrepaData)** - preferred when
     `models/skdrepadata_yolov8n.pt` exists. A YOLOv8n detector trained on
     expert-annotated cells detects and classifies individual sickle / normal
     / other-abnormal cells. The overlay is red (sickle), green (normal), and
     orange (other abnormal); per-cell boxes and confidences can be
     downloaded as CSV.
   - **Cellpose shape heuristic** - segments cell-sized objects, then a
     transparent shape rule labels each object as sickle-like or
     normal-like. Red = sickle-like, green = normal-like.

Both counting modes are **separate from the field classifier**. The detector
has a real per-cell training basis; the Cellpose heuristic does not. Neither
is a validated measurement. See [`skdrepadata.md`](skdrepadata.md) and
[`counting.md`](counting.md) for methods, defaults, and limitations.

The first time the Cellpose mode runs, it downloads about 1.1 GB of pretrained
weights. The field prediction and Grad-CAM work without either counting mode
if you select "Off".

## What the prediction means

- `POSITIVE` = the model thinks the image resembles the sickle-cell-positive
  class it was trained on.
- `NEGATIVE` = the model thinks the image resembles the normal/negative class.
- `Confidence` is the softmax probability for the predicted class. It is a
  model score, not a calibrated clinical probability.

## Important limitations

- **Whole-field classification, not single-cell detection.** The training
  images are microscope fields containing many red blood cells. The model
  classifies the whole field. It does not segment or count individual cells.
  The Streamlit app's counts come from a separate trained detector or Cellpose
  heuristic, not from the field classifier.
- **Counting is only as good as its data.** The detector was trained on one
  external dataset (SKDrepaData-v1) and the heuristic has no per-cell
  training data at all. Both can be wrong on images with a different stain,
  focus, magnification, or camera setup.
- **Framing matters.** The dataset images are circular microscope fields with a
  dark border. A phone photo of a slide, a whole-slide scan, or a tightly
  cropped single cell is a different kind of input and may not work well.
- **Stain/scanner shift.** The dataset comes from one source with a particular
  staining and imaging setup. Different stains, lighting, cameras, or
  microscopes can change predictions.
- **Small dataset.** 569 images total, with only 147 negatives. The model has
  not been externally validated.
- **No clinical validation.** Sensitivity/specificity on a real clinical
  population are unknown. This is a portfolio/research demonstration.

## If a prediction looks wrong

1. Check that the image is a well-focused microscope field, not a whole-slide
   scan or a single isolated cell.
2. Try the other checkpoint (`--checkpoint models/resnet18_sickle.pt`).
3. Run with `--gradcam` and check whether the highlighted region is actually
   cells, or an artifact/border.
4. Remember that a wrong prediction on an out-of-distribution image is
   expected behavior for a small research model, not necessarily a bug.
