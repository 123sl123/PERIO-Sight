"""In-memory annotation filtering and YOLO OBB export.

Callers provide angle-dependent exclusions and clinical tooth labels.
These functions do not infer clinical labels or modify files.
"""

import copy
import math
from typing import Dict, List


def polygon_area(points):
    """Return polygon area in square pixels using the shoelace formula."""
    if len(points) < 3:
        return 0.0
    return abs(sum(
        x * points[(i + 1) % len(points)][1]
        - points[(i + 1) % len(points)][0] * y
        for i, (x, y) in enumerate(points)
    )) / 2.0


def shape_area(shape):
    """Measure a LabelMe polygon, two-corner rectangle or circle."""
    points = shape.get("points", [])
    kind = shape.get("shape_type", "polygon")
    if kind == "rectangle" and len(points) == 2:
        return abs((points[1][0] - points[0][0]) * (points[1][1] - points[0][1]))
    if kind == "circle" and len(points) == 2:
        return math.pi * sum((a - b) ** 2 for a, b in zip(points[0], points[1]))
    if kind in {"polygon", "rotation"}:
        return polygon_area(points)
    return 0.0


def filter_shapes(shapes, excluded_labels=(), min_area=1500):
    """Filter tooth annotations by caller-supplied angle exclusions and area."""
    if not math.isfinite(min_area) or min_area < 0:
        raise ValueError("min_area must be finite and nonnegative")
    excluded = {str(label) for label in excluded_labels}
    return copy.deepcopy([
        shape for shape in shapes
        if str(shape.get("label", "")) not in excluded
        and shape_area(shape) >= min_area
    ])


def retain_positive_teeth(shapes, tooth_labels):
    """Keep clinically positive teeth and target_region; omit unknown labels."""
    labels = {str(tooth): label for tooth, label in tooth_labels.items()}
    return copy.deepcopy([
        shape for shape in shapes
        if shape.get("label") == "target_region"
        or labels.get(str(shape.get("label"))) == 1
    ])


def convert_labelme_to_yolo_rotated(json_data: Dict, img_width: int, img_height: int, class_id: int = 0) -> List[str]:
    """Export normalized four-corner OBB labels: periodontitis class 0, target_region class 1."""
    if img_width <= 0 or img_height <= 0:
        raise ValueError("Image dimensions must be positive")
    import cv2
    import numpy as np

    yolo_annotations = []
    
    if not json_data or 'shapes' not in json_data:
        return yolo_annotations
    
    for shape in json_data.get('shapes', []):
        label = shape.get('label', '')
        
        # Set class IDs from labels.
        if label == 'target_region':
            current_class_id = 1  # The target_region class ID is 1.
        else:
            current_class_id = class_id  # The periodontitis class ID is 0.
        
        shape_type = shape.get('shape_type')
        points = shape.get('points', [])
        
        if not points or len(points) < 2:
            continue
        
        # Handle rotation annotations with four corners.
        if shape_type == 'rotation' and len(points) == 4:
            # Normalize the four corners directly.
            normalized_points = []
            for point in points:
                x_norm = max(0.0, min(1.0, point[0] / img_width))
                y_norm = max(0.0, min(1.0, point[1] / img_height))
                normalized_points.extend([x_norm, y_norm])
            
            # YOLO OBB format: class_id x1 y1 x2 y2 x3 y3 x4 y4.
            yolo_line = f"{current_class_id} " + " ".join([f"{coord:.6f}" for coord in normalized_points])
            yolo_annotations.append(yolo_line)
        
        # Convert rectangle annotations to four corners.
        elif shape_type == 'rectangle' and len(points) == 2:
            x1, y1 = points[0]
            x2, y2 = points[1]
            
            # Four rectangle corners in cyclic order.
            rect_points = [
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2]
            ]
            
            # Normalize the four corners.
            normalized_points = []
            for point in rect_points:
                x_norm = max(0.0, min(1.0, point[0] / img_width))
                y_norm = max(0.0, min(1.0, point[1] / img_height))
                normalized_points.extend([x_norm, y_norm])
            
            # YOLO OBB format: class_id x1 y1 x2 y2 x3 y3 x4 y4.
            yolo_line = f"{current_class_id} " + " ".join([f"{coord:.6f}" for coord in normalized_points])
            yolo_annotations.append(yolo_line)
        
        # Handle four-point polygons.
        elif shape_type == 'polygon' and len(points) == 4:
            # Use the four corners directly.
            normalized_points = []
            for point in points:
                x_norm = max(0.0, min(1.0, point[0] / img_width))
                y_norm = max(0.0, min(1.0, point[1] / img_height))
                normalized_points.extend([x_norm, y_norm])
            
            # YOLO OBB format: class_id x1 y1 x2 y2 x3 y3 x4 y4.
            yolo_line = f"{current_class_id} " + " ".join([f"{coord:.6f}" for coord in normalized_points])
            yolo_annotations.append(yolo_line)
        
        # Fit a minimum-area rotated rectangle to polygons with more than four points.
        elif shape_type == 'polygon' and len(points) > 4:
            # Convert to a NumPy array.
            pts = np.array(points, dtype=np.float32)
            
            # Use OpenCV to compute the minimum-area rotated rectangle.
            rect = cv2.minAreaRect(pts)
            # Get the four corners.
            box_points = cv2.boxPoints(rect)
            
            # Normalize the four corners.
            normalized_points = []
            for point in box_points:
                x_norm = max(0.0, min(1.0, point[0] / img_width))
                y_norm = max(0.0, min(1.0, point[1] / img_height))
                normalized_points.extend([x_norm, y_norm])
            
            # YOLO OBB format: class_id x1 y1 x2 y2 x3 y3 x4 y4.
            yolo_line = f"{current_class_id} " + " ".join([f"{coord:.6f}" for coord in normalized_points])
            yolo_annotations.append(yolo_line)
        
        else:
            # Skip unsupported annotation types.
            continue
    
    return yolo_annotations
