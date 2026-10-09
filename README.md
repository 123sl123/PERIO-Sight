# PERIO-Sight: AI Enabled Multidimensional Periodontal Diagnosis and Longitudinal Monitoring Using Intraoral Optical Scanning Data

This repository contains selected research implementations for PERIO-Sight, with an emphasis on the DM-YOLO periodontal detection component and the processing of multi-view images derived from intraoral optical scans.

The public implementation includes the combined Mamba + DySample model architecture, training and OBB evaluation scripts, core preprocessing operations, and a representative 34-view input with stored deployment outputs. It is a partial research release, not the complete PERIO-Sight clinical software system. Diagnostic aggregation, visualization, and longitudinal monitoring implementations are not included.

## Method overview

DM-YOLO builds on YOLO11x with an oriented bounding-box (OBB) detection head. The backbone and detection head are retained, while the neck incorporates DySample-based upsampling and Mamba-based feature modeling. DySample replaces the upsampling operations from P5 to P4 and from P4 to P3. C3k2Mamba modules are incorporated into the top-down P4 and bottom-up P4/P5 feature-fusion blocks.

The multi-view processing example constructs tooth-associated regions, crops images, performs per-view detection, and assigns detections to annotated teeth through one-to-one intersection-over-union matching. A function-level interface illustrates conditional two-stage processing using an externally supplied diagnostic aggregator. The aggregator is not part of the public implementation.

## Repository structure

```text
model/
  architectures/          # Combined Mamba + DySample architecture
  ultralytics_patch/nn/   # Model parser and module implementations
  install_patch.py        # Installation into a compatible framework checkout
training/
  train_obb.py
  cfgs/                   # Training and dataset configurations
testing/
  evaluate_obb.py          # OBB detection evaluation
  check_bundle.py          # Source and example-input checks
inference/
  region_core.py          # Region construction and cropping
  matching_core.py        # Detection-to-tooth assignment
  multiview_core.py       # Per-view inference and aggregation interface
  run_example.py          # Validation of the public input example
preprocessing/
  render_core.py          # Single-view rendering and vertex-label coloring
  annotation_core.py      # Annotation filtering and OBB target conversion
DATA/D0064/                # 34 PNG images with paired JSON annotations
RESULTS/D0064/             # Corresponding diagnosis JSON and six CAM images
figures/                  # Supplied architecture figure
```

## Software requirements and installation

The model components require a complete Ultralytics checkout with version `8.4.146`. The supplied parser and module registration files are version-dependent and should be installed into a separate, compatible checkout.

The reference software environment comprises Python `3.10.19`, PyTorch `2.7.1+cu118`, Mamba `2.3.2.post1`, and causal-conv1d `1.6.2.post1`. Dependencies are listed in `requirements.txt`. CUDA extensions must be compatible with the installed PyTorch and CUDA versions.

Execute the following commands from the repository root after preparing the compatible Ultralytics checkout:

```bash
python -m pip install -r requirements.txt
python -m pip install causal-conv1d==1.6.2.post1 mamba-ssm==2.3.2.post1 --no-build-isolation
python model/install_patch.py --package-dir /path/to/ultralytics/ultralytics
python -m pip install -e /path/to/ultralytics
```

The `--package-dir` argument identifies the Python package directory containing `nn/`; the editable installation path identifies its parent project directory. The installer verifies the framework version and backs up replaced files with a `.before_dm` suffix. The model files in this repository are not a standalone Ultralytics distribution.

CUDA is required for execution of the complete DM architecture. The Mamba branch is bypassed during CPU model construction; CPU execution must not be used to assess the full model. Mesh rendering additionally requires Open3D and a compatible graphics context; see [Preprocessing](preprocessing/README.md).

## Data organization

Training and evaluation expect patient-exclusive dataset partitions with the following structure:

```text
dataset/
  train/
    images/
    labels/
  val/
    images/
    labels/
  test/
    images/
    labels/
```

Each OBB annotation contains a class index followed by four normalized corner coordinates:

```text
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

The detection classes are `0: periodontitis` and `1: target_region`. These class indices identify detection targets and must not be interpreted as healthy/diseased patient labels.

The public example contains 32 ordered tooth-view slots followed by lower- and upper-jaw views. Slots may use neighboring-tooth substitute views; a slot does not establish tooth presence. Tooth identities are defined by the paired JSON annotations. The example does not include clinical reference labels and cannot be used to calculate diagnostic performance. See [Example input](DATA/README.md).

## Model training

1. Place the YOLO11x-OBB initialization checkpoint at `weights/yolo11x-obb.pt`.
2. Set the dataset root in `training/cfgs/yolov11_obb_data.yaml`.
3. Execute:

```bash
CUDA_VISIBLE_DEVICES=3 python training/train_obb.py \
  --config training/cfgs/yolov11_obb.yaml
```

The supplied configuration specifies Adam optimization, an initial learning rate of 0.001, batch size 16, input size 640 pixels, a maximum of 200 epochs, and early-stopping patience of 50 epochs. The complete configuration, including augmentation settings and random seed, is provided in `training/cfgs/yolov11_obb.yaml`.

The architecture is defined in `model/architectures/yolo11x_obb_mamba_dysample.yaml`. Training outputs are written under `outputs/training/`. Relative paths are resolved from the repository root.

## Detection evaluation

Evaluate a compatible trained checkpoint on the test partition:

```bash
CUDA_VISIBLE_DEVICES=3 python testing/evaluate_obb.py \
  --weights /path/to/dm_best.pt \
  --data training/cfgs/yolov11_obb_data.yaml \
  --split test --device 0
```

Evaluation outputs are written under `outputs/testing/` and include plots and a `metrics.json` file with per-class precision, recall, mAP at IoU 0.50, and mAP averaged over IoU thresholds from 0.50 to 0.95.

These are OBB detection metrics. They are distinct from patient- and tooth-level diagnostic metrics, which require clinical reference labels and the corresponding diagnostic aggregation procedure.

## Multi-view processing example

Validate the 34-view example without loading model weights:

```bash
python inference/run_example.py --check-only
```

The command checks input ordering and image/annotation pairing. It does not perform diagnostic inference, export patient predictions, or generate visualization images.

At the function level, `prepare_views()` constructs and crops image regions, and `run_stage()` returns per-view tooth-matching statistics. `run_two_stage()` illustrates stage routing and requires a caller-supplied `aggregate` function. No implementation or fallback for diagnostic aggregation is provided.

### Stored example outputs

[D0064 deployment outputs](RESULTS/D0064/) contain the unchanged `diagnosis_results.json`, four selected-view CAM images (`1_cam.png` through `4_cam.png`), and lower- and upper-jaw CAM images. These seven files were saved by the same existing deployment run. Its logged input filenames match the 34-view example, whose image pixels are unchanged from that input set.

The JSON contains the patient-level diagnostic decision and score, together with tooth-level scores keyed by FDI tooth number. These are model outputs, not clinical reference labels; absent tooth keys must not be interpreted as verified healthy or missing teeth. The stored outputs are not regenerated by the public input-validation command, and their inclusion does not release the proprietary aggregation or visualization implementation. They are not presented as an evaluation of the supplied DM-YOLO architecture.

The supplied [architecture figure](figures/DM_YOLO_architecture.png) is rendered from the source PDF at 300 dpi without modifying its content.

## Preprocessing

Selected preprocessing functions are provided for single-view rendering, vertex-label coloring, annotation filtering, selection of clinically positive teeth, and OBB target conversion. They operate on caller-supplied meshes or in-memory annotations.

The complete data preparation pipeline is not included. Mesh alignment, clinical spreadsheet processing, study-specific view tables, batch scheduling, and patient partition generation must be supplied separately. See [Preprocessing](preprocessing/README.md) for function-level examples.

## Implementation checks

```bash
python testing/check_bundle.py
```

This command checks Python syntax, English-only text and filenames, the 34 ordered image/annotation pairs, and agreement between image dimensions and annotation metadata. These checks assess source and input integrity, not model performance or clinical validity.

## Code availability

This repository provides a selected implementation of the DM-YOLO architecture, training and detection evaluation, core preprocessing, and multi-view processing interfaces. The complete proprietary software system is not distributed. Diagnostic aggregation, visualization, and longitudinal monitoring implementations are excluded from this release.

The public code does not independently reproduce the complete patient- and tooth-level diagnostic workflow or all study-level results. Such reproduction additionally requires the excluded computational components, study-specific data partitions, reference labels, and trained checkpoints.

## Data availability

A representative 34-view PNG/JSON input and its stored deployment outputs are included for demonstrating input and output organization. Clinical records, complete study datasets, clinical reference labels, and trained checkpoints are not distributed with this repository.

## License and attribution

Ultralytics copyright notices and the accompanying AGPL-3.0 license are retained in `model/ULTRALYTICS_LICENSE`. Existing third-party license notices apply to their respective components.

The upsampling implementation adapts the DySample approach described in *Learning to Upsample by Learning to Sample* (ICCV 2023). The state-space modules use the Mamba implementation. These dependencies and their underlying methods should be acknowledged when using the corresponding components.

## Intended use

The code is provided for research purposes. It is not a standalone clinical diagnostic system and should not be used as a substitute for professional clinical assessment.
