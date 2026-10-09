# Auxiliary Gingivitis Classification

This module complements the primary periodontal diagnosis component of PERIO-Sight. This selected public release provides ResNet50 classifier training, portable tooth-crop inference, an aggregation interface, LCAM-ResNet50 architecture materials and visualization examples. Access to LCAM-specific components and the complete clinical aggregation and visualization implementations may be requested from the research team; see [Access requests](../README.md#access-requests).

## Implementation and architecture

The accompanying [architecture figure](../figures/gingivitis_architecture.png) presents LCAM-ResNet50. The public code provides the underlying ResNet50 classifier component, using torchvision ResNet50 with a two-output fully connected layer. Access to the LCAM-specific implementation may be requested from the research team; the code and configuration below describe the publicly released classifier component.

The two public training scripts retain the source model, augmentation, optimization and checkpoint-selection procedures. Server-specific paths have been replaced by environment settings, forced GPU indices and disabled TLS verification have been removed, and the final validation re-evaluation is no longer described as an independent test. A two-class validation split is required for AUC-based model selection.

## Directory structure

```text
gingivitis/
  training/
    train.py                         # Original cross-entropy training baseline
    train_v4_focal_labelsmooth.py     # Supplied V4 training implementation
  inference/
    core.py                          # Crop inference and aggregation interface
    predict.py                       # Image-level JSON export
  display_figures/
    cam_examples/                    # Historical crop-level CAM images
    projection_examples/             # Historical projected scan images
    workflow_examples/               # Separately supplied workflow displays
```

The three display collections retain their original pixels and patient-oriented filenames and are provided as illustrative visualization examples. The two CAM collections are kept separate to preserve their respective renderings. Access to the complete CAM-generation implementation may be requested from the research team. Diagnostic performance is assessed using clinical reference labels and prediction records, separately from these display materials.

## Class and score conventions

The checkpoint output order is fixed as `0: Gingivitis`, `1: Normal`, matching the source training folder order. This order is distinct from the clinical binary convention `0: healthy`, `1: gingivitis` used by the public inference export.

`prob_gingivitis` is softmax output column 0. `prob_normal` is column 1. `prediction` is the clinically encoded image-level argmax decision, not a tooth-level or patient-level decision. Training precision, recall and F1 follow the supplied scripts and use model class 1 (Normal) as the positive class; they must not be presented as gingivitis sensitivity. Validation AUC also uses the Normal score with the corresponding class-1 labels.

Reference labels are read only from explicitly named input class directories. They remain null for unlabeled input and are never inferred from predictions. Missing crops are not assigned fabricated scores or diagnoses.

## Training configuration

| Setting | Baseline | V4 |
|---|---|---|
| Default model | ResNet50 | ResNet50 |
| Input | RGB, 224 x 224 | RGB, 224 x 224 |
| Batch size | 64 | 64 |
| Maximum epochs | 100 | 100 |
| Optimizer | AdamW | AdamW |
| Initial learning rate | 0.001 | 0.001 |
| Weight decay | 0.0001 | 0.0001 |
| Scheduler | CosineAnnealingLR | CosineAnnealingLR |
| Best checkpoint | Highest validation AUC | Highest validation AUC |
| Early-stopping patience | 15 epochs | 15 epochs |
| Seed | 42 | 42 |
| Sampling | Shuffled original distribution | Inverse-frequency weighted sampling with replacement |
| Loss | Cross entropy | Class-weighted focal modulation of smoothed cross entropy |
| Focal gamma | Not applicable | 2.0 |
| Smoothing | Not applicable | 0.1 |
| Checkpoint interval | Every epoch | Every five epochs |

Both versions apply horizontal and vertical flips with probability 0.5 and random rotation within 15 degrees during training. Validation preprocessing is deterministic. ImageNet normalization uses mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`.

V4 sampling weights are inversely proportional to training class counts. Its loss alpha weights are the normalized inverse frequencies multiplied by the number of classes. The smoothed target assigns 0.9 to the reference class and 0.1 to the other class; focal modulation is computed from the smoothed cross entropy. This is the supplied implementation, not an assertion of an independently validated improvement.

## Usage

Install `requirements.txt` from the repository root. The gingivitis component uses PyTorch and torchvision; the Mamba CUDA extensions are not required to run this component alone.

Provide patient-exclusive, externally established partitions:

```text
dataset/gingivitis/
  train/Gingivitis/...
  train/Normal/...
  val/Gingivitis/...
  val/Normal/...
```

From the repository root:

```bash
CUDA_VISIBLE_DEVICES=3 \
GINGIVITIS_DATA_DIR=/path/to/gingivitis_dataset \
GINGIVITIS_OUTPUT_DIR=outputs/gingivitis/training \
python gingivitis/training/train_v4_focal_labelsmooth.py
```

Use `gingivitis/training/train.py` for the supplied baseline. Set `GINGIVITIS_INIT_WEIGHTS=/path/to/best_model.pth` to load a local initialization checkpoint; otherwise the source ImageNet initialization is used. Set `GINGIVITIS_DEVICE=cpu` to request CPU training. Loading an initialization checkpoint is fine-tuning, not resuming the optimizer and epoch state.

The final ROC, confusion matrix and summary from either training script re-evaluate the selected model on `val/`. An independently held-out test evaluation must be performed separately. The scripts do not generate partitions or verify patient exclusivity.

Run image-level inference on tooth crops:

```bash
CUDA_VISIBLE_DEVICES=3 python gingivitis/inference/predict.py \
  --weights /path/to/gingivitis_best_model.pth \
  --input /path/to/tooth_crops \
  --device cuda:0 \
  --output outputs/gingivitis/inference/image_predictions.json
```

Supported crop names include `D0279_Y-70deg_36.jpg` and names containing `tooth36`. Whole-jaw views and CAM overlays are not classifier inputs. Use `--longitudinal` to preserve separate visit units; visits are not periodontal severity stages. Patient-folder suffixes `-1`, `-2`, `-3` take precedence over filename visit suffixes.

At the function level, `run_pipeline(..., aggregate=callback)` permits integration with a separately supplied tooth/patient aggregator. The project includes base, internal-cohort and visit-aware inference variants. Access to their complete voting and threshold-search implementations may be requested from the research team. The public CLI exports image-level predictions; tooth/patient decisions require an implementation supplied through the aggregation interface.

Threshold selection must use an explicitly designated development partition and be fixed before held-out or external evaluation. The supplied source history does not establish that a particular optimized threshold was selected independently of external evaluation labels. No performance claim or threshold-validation claim is made by this release.

## Scope and checks

Run `python testing/check_bundle.py` for text and source integrity, and `python testing/check_gingivitis.py` in an environment with the module dependencies for CPU smoke tests. These checks do not reproduce clinical evaluation or validate the historical CAM images.

This public release includes the classifier implementation, configuration and usage documentation, architecture materials and illustrative display images. Access to the LCAM-specific implementation, complete aggregation/visualization components, trained weights and study datasets may be requested from the research team under the [access conditions](../README.md#access-requests). Study metrics are computed from clinical reference labels and prediction records rather than display images.
