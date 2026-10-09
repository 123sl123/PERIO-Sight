# PERIO-Sight: AI Enabled Multidimensional Periodontal Diagnosis and Longitudinal Monitoring Using Intraoral Optical Scanning Data

This repository contains selected research implementations for PERIO-Sight, with periodontal diagnosis as the primary task and gingivitis classification as an auxiliary task. The primary implementation focuses on DM-YOLO periodontal detection and the processing of multi-view images derived from intraoral optical scans.

The public implementation includes the combined Mamba + DySample model architecture, training and OBB evaluation scripts, core preprocessing operations, and a representative 34-view input with stored deployment outputs. The auxiliary gingivitis release includes ResNet50 classifier training and inference, an aggregation interface, architecture materials and display images. This repository is a selected research release. Access to LCAM-specific components and the complete diagnostic aggregation, visualization and longitudinal monitoring implementations may be requested from the research team; see [Access requests](#access-requests).

## Method overview

DM-YOLO builds on YOLO11x with an oriented bounding-box (OBB) detection head. The backbone and detection head are retained, while the neck incorporates DySample-based upsampling and Mamba-based feature modeling. DySample replaces the upsampling operations from P5 to P4 and from P4 to P3. C3k2Mamba modules are incorporated into the top-down P4 and bottom-up P4/P5 feature-fusion blocks.

The multi-view processing example constructs tooth-associated regions, crops images, performs per-view detection, and assigns detections to annotated teeth through one-to-one intersection-over-union matching. A function-level interface illustrates conditional two-stage processing using an externally supplied diagnostic aggregator. Access to the complete aggregator may be requested from the research team.

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
gingivitis/
  training/               # Baseline and V4 ResNet training
  inference/              # Tooth-crop inference and aggregation interface
  display_figures/        # Historical gingivitis visualization examples
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

At the function level, `prepare_views()` constructs and crops image regions, and `run_stage()` returns per-view tooth-matching statistics. `run_two_stage()` illustrates stage routing and requires a caller-supplied `aggregate` function. The public example requires this external implementation; access to the complete diagnostic aggregation component may be requested from the research team.

### Stored example outputs

[D0064 deployment outputs](RESULTS/D0064/) contain the unchanged `diagnosis_results.json`, four selected-view CAM images (`1_cam.png` through `4_cam.png`), and lower- and upper-jaw CAM images. These seven files were saved by the same existing deployment run. Its logged input filenames match the 34-view example, whose image pixels are unchanged from that input set.

The JSON contains the patient-level diagnostic decision and score, together with tooth-level scores keyed by FDI tooth number. These are model outputs, not clinical reference labels; absent tooth keys must not be interpreted as verified healthy or missing teeth. The stored outputs are not regenerated by the public input-validation command. Access to the complete aggregation and visualization implementations may be requested from the research team. These example outputs are not presented as an evaluation of the supplied DM-YOLO architecture.

The supplied [architecture figure](figures/DM_YOLO_architecture.png) is rendered from the source PDF at 300 dpi without modifying its content.

## Auxiliary gingivitis classification

The [gingivitis module](gingivitis/README.md) complements the primary periodontal workflow. The accompanying architecture materials present LCAM-ResNet50. The publicly released code provides the ResNet50 classifier component, with a cross-entropy baseline and a V4 variant using weighted sampling, class-weighted focal modulation and label smoothing. Access to the LCAM-specific implementation may be requested from the research team. Public inference exports image-level scores from tooth crops and provides an interface for a separately supplied tooth/patient aggregator.

Model output indices are `0: Gingivitis`, `1: Normal`; the public image-prediction export uses the clinical convention `0: healthy`, `1: gingivitis`. The supplied training precision, recall and F1 use model class 1 (Normal) as positive and are not gingivitis sensitivity. Final training reports re-evaluate the validation partition rather than an independent test partition.

The [gingivitis architecture figure](figures/gingivitis_architecture.png) is available in `figures/`. Representative CAM visualizations and projected scan images are organized in `gingivitis/display_figures/`. See the [module README](gingivitis/README.md) for implementation details, configuration, usage and validation checks.

## Preprocessing

Selected preprocessing functions are provided for single-view rendering, vertex-label coloring, annotation filtering, selection of clinically positive teeth, and OBB target conversion. They operate on caller-supplied meshes or in-memory annotations.

The public preprocessing functions operate on caller-supplied inputs. Access to the complete data preparation pipeline, including mesh alignment, clinical spreadsheet processing, study-specific view tables, batch scheduling and patient partition generation, may be requested from the research team. See [Preprocessing](preprocessing/README.md) for function-level examples.

## Implementation checks

```bash
python testing/check_bundle.py
```

This command checks Python syntax, English-only text and filenames, the 34 ordered image/annotation pairs, and agreement between image dimensions and annotation metadata. These checks assess source and input integrity, not model performance or clinical validity.

## Code availability

This repository provides a selected implementation of the DM-YOLO architecture, training and detection evaluation, core preprocessing, multi-view processing interfaces, and the ResNet50 classifier component of the auxiliary gingivitis module. Access to LCAM-specific components and the complete diagnostic aggregation, visualization and longitudinal monitoring implementations may be requested from the research team.

Reproducing the complete patient- and tooth-level diagnostic workflow and study-level results additionally requires the corresponding aggregation components, study-specific data partitions, reference labels and trained checkpoints. Requests for access to these materials are handled as described below.

## Data availability

A representative 34-view PNG/JSON input and its stored deployment outputs are included for demonstrating input and output organization. Historical gingivitis display images are included separately and are not an evaluation dataset. Access to study datasets, clinical reference labels and trained checkpoints may be requested from the research team, subject to the access conditions below. Clinical records require the applicable institutional and confidentiality approvals.

## Access requests

Materials beyond the public release may be requested from the research team for research purposes. To initiate a request, open a GitHub issue specifying the requested components or materials, intended use and institutional affiliation. Do not include patient information or other confidential data in a public issue; the team will arrange a private follow-up channel as appropriate.

Requests are reviewed individually. Access is subject to availability, applicable institutional and ethical approvals, patient confidentiality, licensing and intellectual-property conditions, and any required data-use or collaboration agreements. Submitting a request does not guarantee access.

## License and attribution

Ultralytics copyright notices and the accompanying AGPL-3.0 license are retained in `model/ULTRALYTICS_LICENSE`. Existing third-party license notices apply to their respective components.

The upsampling implementation adapts the DySample approach described in *Learning to Upsample by Learning to Sample* (ICCV 2023). The state-space modules use the Mamba implementation. These dependencies and their underlying methods should be acknowledged when using the corresponding components.

## Intended use

The code is provided for research purposes. It is not a standalone clinical diagnostic system and should not be used as a substitute for professional clinical assessment.
