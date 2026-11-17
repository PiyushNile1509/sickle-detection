# Estimated Cell Counting (Heuristic)

> **A trained per-cell detector is now the preferred counting mode.** When
> `models/skdrepadata_yolov8n.pt` is present, the Streamlit app offers
> **Trained detector (SKDrepaData)**, which detects and classifies individual
> sickle / normal / other-abnormal cells using expert-annotated training
> boxes. See [`skdrepadata.md`](skdrepadata.md). This document describes the
> older Cellpose + shape-rule heuristic, kept as a fallback and comparison.
>
> **Research prototype only.** These counts are image-processing estimates,
> not a diagnosis and not a validated measurement. Do not use them for
> clinical decisions.

## Why counting is a separate pipeline

The trained CNN in `src/inference.py` is a **whole-field binary classifier**.
It was trained on 224×224 microscope fields and outputs one score per field
(negative vs positive). It has never been trained to detect, segment, or count
individual cells.

To estimate *how many* sickle cells are present, `src/counting.py` adds a
separate pipeline:

1. **Segment cell-sized objects.** Cellpose (a general-purpose pretrained
   segmentation model) proposes one mask per object.
2. **Measure each mask.** Area, eccentricity, solidity, circularity, and
   aspect ratio are computed with `skimage.measure.regionprops`.
3. **Apply a transparent shape rule.** A mask is counted as **sickle-like**
   when:

   ```
   eccentricity >= 0.75   and   circularity <= 0.80
   ```

   Otherwise it is counted as **normal-like**. Eccentricity is 0 for a circle
   and approaches 1 for a line; circularity is 1 for a perfect circle and
   decreases for irregular/elongated shapes.

The app shows the colour-coded overlay (red = sickle-like, green =
normal-like) so the result can be inspected visually rather than trusted as a
number.

## Defaults

| Setting | Default | Notes |
|---|---:|---|
| Analysis resolution (longest side) | 1024 px | Images are downscaled before segmentation. |
| Approximate cell diameter | 20 px | At the analysis resolution. Tune if cells look much larger/smaller. |
| Minimum cell area | 80 px² | Smaller objects are treated as debris. |
| Sickle eccentricity threshold | 0.75 | Higher = stricter. |
| Sickle circularity threshold | 0.80 | Lower = stricter. |
| Cellpose cell-probability threshold | -2.0 | Lower detects more faint objects. |
| Cellpose flow threshold | 0.9 | Higher keeps more masks. |
| Cellpose minimum mask size | 5 px | Passed to Cellpose. |

The segmentation uses the **grayscale** version of the field. During tuning
this detected substantially more faint brightfield cells than RGB input.

## Sanity check (not validation)

On five dataset fields used while developing the app:

| Field | Total detected | Sickle-like | Fraction |
|---|---:|---:|---:|
| Positive 1 | 183 | 59 | 32.2% |
| Positive 115 | 200 | 68 | 34.0% |
| Positive 50 | 227 | 28 | 12.3% |
| Negative 1 | 233 | 22 | 9.4% |
| Negative 107 | 323 | 21 | 6.5% |

Strongly positive fields had roughly 32–34% sickle-like objects, while
negative fields had roughly 6–9%. Positive 50 is a milder example. These
numbers are a **sanity check only**: there are no exhaustive per-cell labels,
so sensitivity/specificity of the counting heuristic are unknown.

## Important limitations

- **No per-cell ground truth.** The dataset's `Positive/Labelled` images
  contain only a few sparse bounding boxes, not exhaustive annotations. The
  shape thresholds were chosen by inspection, not fitted to labelled cells.
- **Segmentation misses cells.** Faint, out-of-focus, overlapping, or
  partially cut-off cells may not be segmented. The total count is therefore
  usually a lower bound on the visible cells.
- **Shape is not diagnosis.** Some normal cells can look elongated
  (especially at the field edge or when overlapping), and some sickle cells
  may look rounder in a particular orientation. The heuristic will make
  mistakes in both directions.
- **Scale and stain sensitivity.** Cell diameter, focus, magnification,
  staining, and illumination all affect segmentation. If the input differs
  from the training-style images, tune the cell diameter and thresholds or
  treat the counts as unreliable.
- **The classifier and the count can disagree.** The classifier looks at the
  whole field's appearance; the count looks at segmented object shapes. The
  app surfaces both rather than forcing them to agree.
- **Not clinically validated.** No claim is made that these counts correspond
  to a laboratory measurement or to a patient's sickle-cell burden.

## Tuning the app

The sidebar's **Counting settings** expander exposes the main controls:

- **Analysis resolution**: higher preserves small cells but is slower.
- **Approximate cell diameter**: the most important segmentation control. If
  the overlay misses many cells, try a smaller diameter; if one mask covers
  several cells, try a larger one.
- **Sickle shape thresholds**: raise the eccentricity threshold and/or lower
  the circularity threshold to be stricter.
- **Cellpose thresholds**: lower the cell-probability threshold to include
  fainter objects; raise the flow threshold to keep more masks.

## Possible future work

A real per-cell detector requires exhaustive bounding-box or mask
annotations. The sparse boxes in the primary dataset's `Positive/Labelled`
images are not sufficient as-is. That gap is now addressed by the
SKDrepaData-trained YOLOv8 detector in `src/detector_counting.py` - see
[`skdrepadata.md`](skdrepadata.md). The heuristic here remains useful as an
independent, training-free cross-check and for datasets where no detector
weights are available.

Further options include manually annotating a subset of the primary dataset's
fields so the detector can be fine-tuned on this project's exact stain and
magnification, and training a per-cell shape classifier on cropped cells for
comparison.
