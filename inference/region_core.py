"""In-memory tooth-region construction, cropping and annotation filtering."""

import numpy as np
from PIL import Image
from shapely.geometry import Polygon


def rect_to_rotated_box(points):
    """Convert two opposite rectangle corners into four cyclically ordered vertices.."""
    x1, y1 = points[0]
    x2, y2 = points[1]
    x_min, x_max = (min(x1, x2), max(x1, x2))
    y_min, y_max = (min(y1, y2), max(y1, y2))
    return [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]


def get_all_box_points(shapes):
    """Collect vertices from all annotation boxes.."""
    all_points = []
    for shape in shapes:
        if shape.get('shape_type') == 'rectangle':
            points = shape['points']
            rotated = rect_to_rotated_box(points)
            all_points.extend(rotated)
        elif shape.get('shape_type') == 'rotation':
            all_points.extend(shape['points'])
    return np.array(all_points, dtype=np.float32) if all_points else np.array([])


def compute_minimum_rotated_rect(points):
    """Compute the minimum-area rotated rectangle enclosing the points.."""
    if len(points) < 3:
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        return [[float(x_min), float(y_min)], [float(x_max), float(y_min)], [float(x_max), float(y_max)], [float(x_min), float(y_max)]]
    try:
        import cv2
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        box = box.astype(int)
        return [[float(x), float(y)] for x, y in box]
    except ImportError:
        print('Warning: Not installedOpenCV, Use an axis-aligned box instead of a minimum-area rectangle')
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        margin_x = (x_max - x_min) * 0.05
        margin_y = (y_max - y_min) * 0.05
        x_min -= margin_x
        y_min -= margin_y
        x_max += margin_x
        y_max += margin_y
        return [[float(x_min), float(y_min)], [float(x_max), float(y_min)], [float(x_max), float(y_max)], [float(x_min), float(y_max)]]


def scale_rotated_box(box_points, scale_factor):
    """Scale a rotated box about its center.."""
    points = np.array(box_points, dtype=np.float32)
    center = points.mean(axis=0)
    scaled_points = center + (points - center) * scale_factor
    return [[float(x), float(y)] for x, y in scaled_points]


def clip_rotated_box_to_image(box_points, img_width, img_height):
    """Clip rotated-box vertices to image boundaries.."""
    clipped_points = []
    for x, y in box_points:
        x_clipped = max(0, min(img_width, float(x)))
        y_clipped = max(0, min(img_height, float(y)))
        clipped_points.append([x_clipped, y_clipped])
    return clipped_points


def process_labelme_json_in_memory(labelme_data, img_width=None, img_height=None, apply_vertical_adjustment=True):
    """Deep-copy LabelMe data and convert tooth polygons to boxes."""
    import copy
    processed_data = copy.deepcopy(labelme_data)
    if img_width is None:
        img_width = processed_data.get('imageWidth')
    if img_height is None:
        img_height = processed_data.get('imageHeight')
    if img_width is None or img_height is None:
        raise ValueError('Cannot determine image dimensions, Please provideimg_widthandimg_heightParameters, or ensureJSONcontainsimageWidthandimageHeightfield')
    new_shapes = []
    for shape in processed_data.get('shapes', []):
        if shape['shape_type'] not in ['polygon', 'rectangle']:
            continue
        points = shape['points']
        if not points:
            continue
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        x_min, x_max = (min(x_coords), max(x_coords))
        y_min, y_max = (min(y_coords), max(y_coords))
        width = x_max - x_min
        extend_x = width / 16
        x_min -= extend_x
        x_max += extend_x
        x_min = max(0, x_min)
        x_max = min(img_width, x_max)
        if apply_vertical_adjustment:
            label = shape['label']
            try:
                label_value = int(label)
                height = y_max - y_min
                if 11 <= label_value <= 28:
                    y_min_extend = y_min - height / 2
                    y_max_contract = y_max - height / 2
                    y_min = max(0, y_min_extend)
                    y_max = max(0, y_max_contract)
                elif 31 <= label_value <= 48:
                    y_min_contract = y_min + height / 2
                    y_max_extend = y_max + height / 2
                    y_min = min(img_height, y_min_contract)
                    y_max = min(img_height, y_max_extend)
            except (ValueError, TypeError):
                pass
        bbox_shape = {'label': shape['label'], 'points': [[x_min, y_min], [x_max, y_max]], 'group_id': shape.get('group_id', None), 'shape_type': 'rectangle', 'flags': shape.get('flags', {})}
        new_shapes.append(bbox_shape)
    processed_data['shapes'] = new_shapes
    if new_shapes:
        new_shapes = [s for s in new_shapes if s.get('label') not in ['target_region', 'det_region']]
        processed_data['shapes'] = new_shapes
        all_points = get_all_box_points(new_shapes)
        if len(all_points) > 0:
            target_box = compute_minimum_rotated_rect(all_points)
            target_box = clip_rotated_box_to_image(target_box, img_width, img_height)
            target_shape = {'label': 'target_region', 'points': target_box, 'group_id': None, 'shape_type': 'rotation', 'flags': {}}
            det_box = scale_rotated_box(target_box, scale_factor=1.2)
            det_box = clip_rotated_box_to_image(det_box, img_width, img_height)
            det_shape = {'label': 'det_region', 'points': det_box, 'group_id': None, 'shape_type': 'rotation', 'flags': {}}
            processed_data['shapes'].insert(0, target_shape)
            processed_data['shapes'].insert(0, det_shape)
    return processed_data


def get_rotated_box_aabb(box_points):
    """Return (x_min, y_min, x_max, y_max) enclosing the rotated box.."""
    points = np.array(box_points, dtype=np.float32)
    x_min = float(points[:, 0].min())
    y_min = float(points[:, 1].min())
    x_max = float(points[:, 0].max())
    y_max = float(points[:, 1].max())
    return (x_min, y_min, x_max, y_max)


def adjust_shape_coordinates(shape, offset_x, offset_y):
    """Subtract crop offsets from annotation coordinates.."""
    adjusted_shape = shape.copy()
    adjusted_points = []
    for point in shape['points']:
        new_x = point[0] - offset_x
        new_y = point[1] - offset_y
        adjusted_points.append([new_x, new_y])
    adjusted_shape['points'] = adjusted_points
    return adjusted_shape


def shrink_box_width(box_points, shrink_ratio=0.2):
    """Reduce box width by shrink_ratio while preserving its center.."""
    points = np.array(box_points, dtype=np.float32)
    center_x = points[:, 0].mean()
    center_y = points[:, 1].mean()
    sorted_indices = np.argsort(points[:, 0])
    left_indices = sorted_indices[:2]
    right_indices = sorted_indices[2:]
    left_x_avg = points[left_indices, 0].mean()
    right_x_avg = points[right_indices, 0].mean()
    current_width = right_x_avg - left_x_avg
    move_distance = current_width * shrink_ratio / 2
    new_points = points.copy()
    new_points[left_indices, 0] += move_distance
    new_points[right_indices, 0] -= move_distance
    return [[float(x), float(y)] for x, y in new_points]


def calculate_polygon_area(box_points):
    """Compute polygon area using the shoelace formula.."""
    try:
        poly = Polygon(box_points)
        return poly.area
    except:
        return 0.0


def calculate_intersection_area(box1_points, box2_points):
    """Compute the intersection area of two polygons.."""
    try:
        poly1 = Polygon(box1_points)
        poly2 = Polygon(box2_points)
        if not poly1.is_valid or not poly2.is_valid:
            return 0.0
        intersection = poly1.intersection(poly2)
        return intersection.area
    except:
        return 0.0


def convert_to_polygon_points(points):
    """Convert a two-corner rectangle to four vertices; accept existing polygons.."""
    if len(points) == 2:
        x1, y1 = points[0]
        x2, y2 = points[1]
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    else:
        return points


def calculate_intersection_ratio(box_points, target_box_points):
    """Return intersection area divided by the candidate box area.."""
    try:
        box_poly_points = convert_to_polygon_points(box_points)
        target_poly_points = convert_to_polygon_points(target_box_points)
        poly_box = Polygon(box_poly_points)
        poly_target = Polygon(target_poly_points)
        if not poly_box.is_valid:
            poly_box = poly_box.buffer(0)
        if not poly_target.is_valid:
            poly_target = poly_target.buffer(0)
        box_area = poly_box.area
        if box_area <= 0:
            return 0.0
        intersection = poly_box.intersection(poly_target)
        intersection_area = intersection.area
        ratio = intersection_area / box_area
        return ratio
    except Exception as e:
        return 0.0


def crop_image_by_det_region_in_memory(image, json_data, intersection_threshold=0.8):
    """Crop the image to the det_region AABB and shift annotations."""
    import copy
    data = copy.deepcopy(json_data)
    if hasattr(image, 'shape'):
        import cv2
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        img = Image.fromarray(image_rgb)
    else:
        img = image
    det_region = None
    for shape in data.get('shapes', []):
        if shape.get('label') == 'det_region':
            det_region = shape
            break
    if not det_region:
        raise ValueError('Not founddet_region, Cannot crop')
    det_points = det_region['points']
    x_min, y_min, x_max, y_max = get_rotated_box_aabb(det_points)
    x_min, y_min = (int(max(0, x_min)), int(max(0, y_min)))
    x_max, y_max = (int(x_max), int(y_max))
    try:
        cropped_img = img.crop((x_min, y_min, x_max, y_max))
        crop_width, crop_height = cropped_img.size
    except Exception as e:
        raise ValueError(f'Image cropping failed: {e}')
    adjusted_shapes = []
    target_region_points = None
    for shape in data.get('shapes', []):
        label = shape.get('label')
        if label == 'det_region':
            continue
        adjusted_shape = adjust_shape_coordinates(shape, x_min, y_min)
        if label == 'target_region':
            adjusted_shape['points'] = shrink_box_width(adjusted_shape['points'], shrink_ratio=0.2)
            target_region_points = adjusted_shape['points']
        adjusted_shapes.append(adjusted_shape)
    new_shapes = []
    pending_shapes = []
    filtered_count = 0
    secondary_threshold = 0.4
    for shape in adjusted_shapes:
        label = shape.get('label')
        if label == 'target_region':
            new_shapes.append(shape)
            continue
        if target_region_points is None:
            new_shapes.append(shape)
            continue
        intersection_ratio = calculate_intersection_ratio(shape['points'], target_region_points)
        if intersection_ratio >= intersection_threshold:
            new_shapes.append(shape)
        elif intersection_ratio >= secondary_threshold:
            pending_shapes.append((shape, intersection_ratio))
        else:
            filtered_count += 1
    if pending_shapes:
        kept_tooth_areas = []
        for shape in new_shapes:
            if shape.get('label') != 'target_region':
                area = calculate_polygon_area(shape['points'])
                if area > 0:
                    kept_tooth_areas.append(area)
        if kept_tooth_areas:
            min_kept_area = min(kept_tooth_areas)
            dynamic_area_threshold = min_kept_area * 0.7
            for shape, intersection_ratio in pending_shapes:
                tooth_area = calculate_polygon_area(shape['points'])
                intersection_area = tooth_area * intersection_ratio
                if intersection_area >= dynamic_area_threshold:
                    new_shapes.append(shape)
                else:
                    filtered_count += 1
        else:
            filtered_count += len(pending_shapes)
    data['shapes'] = new_shapes
    data['imageWidth'] = crop_width
    data['imageHeight'] = crop_height
    crop_info = {'crop_box': (x_min, y_min, x_max, y_max), 'crop_size': (crop_width, crop_height), 'bbox_count': len(new_shapes), 'filtered_count': filtered_count}
    return (cropped_img, data, crop_info)
