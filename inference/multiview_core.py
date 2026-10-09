"""Multi-view detection and integration points for private diagnostic aggregation."""

from .matching_core import match_yolo_result_to_teeth
from .region_core import crop_image_by_det_region_in_memory, process_labelme_json_in_memory


def prepare_views(images, annotations):
    """Prepare 32 tooth views followed by lower- and upper-jaw views in memory."""
    if len(images) != 34 or len(annotations) != 34:
        raise ValueError("Exactly 34 paired views are required")
    prepared = []
    for index, (image, annotation) in enumerate(zip(images, annotations)):
        if image is None:
            raise ValueError(f"Unreadable image at index {index}")
        boxes = process_labelme_json_in_memory(
            annotation, img_width=image.shape[1], img_height=image.shape[0],
            apply_vertical_adjustment=index < 32,
        )
        cropped, labels, _ = crop_image_by_det_region_in_memory(
            image, boxes, intersection_threshold=0.8,
        )
        prepared.append({"image": cropped, "annotation": labels})
    return prepared


def run_stage(prepared_views, model, device="0"):
    """Return per-view tooth-matching statistics without diagnostic aggregation."""
    results = list(model.predict(
        source=[view["image"] for view in prepared_views],
        conf=0.3, iou=0.5, imgsz=640, device=device, verbose=False,
    ))
    if len(results) != len(prepared_views):
        raise ValueError("Detection result count does not match the input views")
    statistics = [
        match_yolo_result_to_teeth(result, view["annotation"], iou_threshold=0.3)["tooth_stats"]
        for result, view in zip(results, prepared_views)
    ]
    return statistics


def run_two_stage(prepared_views, stage_one_model, stage_two_model, aggregate, device="0"):
    """Illustrate stage routing with a caller-supplied diagnostic aggregator.

    aggregate(view_statistics) must return a dictionary containing
    patient_periodontitis.is_diseased. Its implementation is not distributed.
    """
    first = aggregate(run_stage(prepared_views, stage_one_model, device))
    if first["patient_periodontitis"]["is_diseased"]:
        return aggregate(run_stage(prepared_views, stage_two_model, device))
    return first
