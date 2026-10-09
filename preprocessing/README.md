# Core Preprocessing Operations

This directory provides function-level implementations of selected preprocessing operations used in the research workflow. It is not an end-to-end data preparation pipeline.

## Included Functions

| Module | Functions | Purpose |
| --- | --- | --- |
| `render_core.py` | `create_gray_mesh_from_labels`, `setup_camera_and_render` | Encode per-vertex tooth labels and render an aligned mesh at specified X/Y angles |
| `annotation_core.py` | `polygon_area`, `shape_area`, `filter_shapes` | Measure annotation area and apply caller-supplied view exclusions |
| `annotation_core.py` | `retain_positive_teeth` | Select teeth with supplied clinical label 1 while retaining `target_region` |
| `annotation_core.py` | `convert_labelme_to_yolo_rotated` | Export normalized four-corner YOLO OBB targets |

Install the optional rendering and conversion dependencies:

```bash
python -m pip install -r preprocessing/requirements.txt
```

## Annotation Processing

```python
from preprocessing.annotation_core import (
    filter_shapes,
    retain_positive_teeth,
    convert_labelme_to_yolo_rotated,
)

# Supply exclusions for the current view and clinical labels independently.
filtered = filter_shapes(
    segmentation["shapes"],
    excluded_labels=excluded_teeth_for_view,
    min_area=1500,
)
positive = retain_positive_teeth(filtered, clinical_tooth_labels)
targets = convert_labelme_to_yolo_rotated(
    {"shapes": positive},
    img_width=image_width,
    img_height=image_height,
)
```

The example variables must be supplied by the caller. Clinical labels are integer 0/1 values; unknown teeth are not relabeled as positive. Apply tooth filtering before adding the enclosing `target_region`, or preserve that region separately before clinical selection. OBB export assigns class 0 to the selected periodontitis targets and class 1 to `target_region`. These detection classes are distinct from clinical healthy/diseased labels.

The functions operate on in-memory annotations and do not read or overwrite files. Filtering returns copies of the retained shapes.

## Rendering

```python
from preprocessing.render_core import setup_camera_and_render

image = setup_camera_and_render(
    aligned_centered_mesh,
    x_angle_degrees=0,
    y_angle_degrees=0,
    for_grayscale=False,
)
```

The mesh must already be aligned and centered. Vertex labels must correspond one-to-one with mesh vertices. Rendering uses Open3D's visualizer and requires a compatible graphics context; this minimal example does not configure a headless rendering backend.

## Scope

Access to mesh alignment, study-specific view-exclusion tables, clinical spreadsheet processing, patient partitioning, batch orchestration and the complete preprocessing pipeline may be requested from the research team; see [Access requests](../README.md#access-requests). Users of the public functions supply the corresponding inputs and enforce patient-exclusive training, validation and test partitions.
