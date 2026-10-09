"""One-to-one IoU assignment of class-0 detections to annotated teeth."""

import numpy as np
from shapely.geometry import Polygon, box as Box


def parse_json_bbox(shape):
    """Parse a box from a JSON shape.."""
    label = shape['label']
    points = shape['points']
    x1, y1 = points[0]
    x2, y2 = points[1]
    min_x, max_x = (min(x1, x2), max(x1, x2))
    min_y, max_y = (min(y1, y2), max(y1, y2))
    polygon = Box(min_x, min_y, max_x, max_y)
    return (label, polygon)


def calculate_iou(poly1, poly2):
    """Compute intersection over union for two polygons.."""
    try:
        if not poly1.is_valid:
            poly1 = poly1.buffer(0)
        if not poly2.is_valid:
            poly2 = poly2.buffer(0)
        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        if union == 0:
            return 0.0
        return intersection / union
    except Exception as e:
        print(f'    Warning: IoUComputation failed - {e}')
        return 0.0


def compute_iou_matrix(predictions, teeth_dict):
    """Compute IoU for every prediction/tooth pair.."""
    teeth_labels = list(teeth_dict.keys())
    n_pred = len(predictions)
    n_teeth = len(teeth_labels)
    iou_matrix = np.zeros((n_pred, n_teeth))
    for i, (pred_poly, _) in enumerate(predictions):
        for j, tooth_label in enumerate(teeth_labels):
            tooth_poly = teeth_dict[tooth_label]
            iou_matrix[i, j] = calculate_iou(pred_poly, tooth_poly)
    return (iou_matrix, teeth_labels)


def match_predictions_to_teeth_one_to_one(predictions, teeth_dict, threshold=0.3):
    """Greedily match predictions to teeth by decreasing IoU, using each prediction and tooth at most once.."""
    if not predictions or not teeth_dict:
        return ([], [(i, conf, 0.0) for i, (_, conf) in enumerate(predictions)])
    iou_matrix, teeth_labels = compute_iou_matrix(predictions, teeth_dict)
    candidates = []
    for i in range(len(predictions)):
        for j in range(len(teeth_labels)):
            iou_score = iou_matrix[i, j]
            if iou_score >= threshold:
                candidates.append((iou_score, i, j))
    candidates.sort(reverse=True, key=lambda x: x[0])
    matched_predictions = set()
    matched_teeth = set()
    matches = []
    for iou_score, pred_idx, tooth_idx in candidates:
        if pred_idx not in matched_predictions and tooth_idx not in matched_teeth:
            tooth_label = teeth_labels[tooth_idx]
            confidence = predictions[pred_idx][1]
            matches.append((pred_idx, tooth_label, confidence, iou_score))
            matched_predictions.add(pred_idx)
            matched_teeth.add(tooth_idx)
    unmatched_predictions = []
    for pred_idx, (_, confidence) in enumerate(predictions):
        if pred_idx not in matched_predictions:
            best_score = np.max(iou_matrix[pred_idx, :]) if iou_matrix.shape[1] > 0 else 0.0
            unmatched_predictions.append((pred_idx, confidence, best_score))
    return (matches, unmatched_predictions)


def match_predictions_to_teeth_core(predictions, teeth_dict, iou_threshold=0.3):
    """Match parsed prediction boxes to tooth annotations.."""
    matches, unmatched_predictions = match_predictions_to_teeth_one_to_one(predictions, teeth_dict, threshold=iou_threshold)
    tooth_stats = {}
    for tooth_label in teeth_dict.keys():
        tooth_stats[tooth_label] = {'matched': False, 'iou': 0.0, 'confidence': 0.0}
    for pred_idx, tooth_label, confidence, iou_score in matches:
        if tooth_label in tooth_stats:
            tooth_stats[tooth_label]['matched'] = True
            tooth_stats[tooth_label]['iou'] = iou_score
            tooth_stats[tooth_label]['confidence'] = confidence
    return {'matches': matches, 'unmatched_predictions': unmatched_predictions, 'total_predictions': len(predictions), 'total_teeth': len(teeth_dict), 'tooth_stats': tooth_stats}


def match_yolo_result_to_teeth(yolo_result, json_data, iou_threshold=0.3):
    """Match an in-memory YOLO result to tooth annotations.."""
    teeth_dict = {}
    for shape in json_data.get('shapes', []):
        label = shape['label']
        if label != 'target_region':
            _, polygon = parse_json_bbox(shape)
            teeth_dict[label] = polygon
    predictions = []
    if hasattr(yolo_result, 'obb') and yolo_result.obb is not None and (len(yolo_result.obb) > 0):
        for obb in yolo_result.obb:
            cls_id = int(obb.cls[0])
            if cls_id == 0:
                confidence = float(obb.conf[0])
                xyxyxyxy = obb.xyxyxyxy[0].cpu().numpy()
                points = [[float(x), float(y)] for x, y in xyxyxyxy]
                polygon = Polygon(points)
                predictions.append((polygon, confidence))
    elif hasattr(yolo_result, 'boxes') and yolo_result.boxes is not None and (len(yolo_result.boxes) > 0):
        for box in yolo_result.boxes:
            cls_id = int(box.cls[0])
            if cls_id == 0:
                confidence = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                polygon = Box(x1, y1, x2, y2)
                predictions.append((polygon, confidence))
    return match_predictions_to_teeth_core(predictions, teeth_dict, iou_threshold)
