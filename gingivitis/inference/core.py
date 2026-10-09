"""Image-level ResNet inference and a caller-supplied aggregation interface."""

from pathlib import Path
import re

import torch
from PIL import Image
from torchvision import models, transforms


MODEL_CLASSES = ("Gingivitis", "Normal")
CLINICAL_LABELS = {"Normal": 0, "Gingivitis": 1}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
FDI_TEETH = {str(q * 10 + t) for q in range(1, 5) for t in range(1, 9)}


def parse_view(path, longitudinal=False):
    """Extract a case, FDI tooth and optional visit from a tooth-crop filename."""
    path = Path(path)
    case_id = path.stem.split("_")[0]
    explicit = re.search(r"tooth(\d+)", path.stem, re.IGNORECASE)
    candidates = [explicit.group(1)] if explicit else reversed(path.stem.split("_"))
    tooth = next((item for item in candidates if item in FDI_TEETH), None)
    if tooth is None:
        raise ValueError(f"No valid permanent FDI tooth in filename: {path.name}")
    visit = None
    if longitudinal:
        folder = re.fullmatch(r"(.+)[-_]([123])", path.parent.name)
        if folder:
            case_id, visit = folder.group(1), int(folder.group(2))
        else:
            suffix = re.search(r"_([12])$", path.stem)
            visit = int(suffix.group(1)) + 1 if suffix else 1
    return case_id, tooth, visit


def collect_inputs(root, longitudinal=False):
    """Enumerate crops without deriving reference labels from predictions."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise ValueError(f"No input images: {root}")
    inputs = []
    for path in files:
        case, tooth, visit = parse_view(path, longitudinal)
        relative = path.relative_to(root)
        label = CLINICAL_LABELS.get(relative.parts[0]) if len(relative.parts) > 1 else None
        inputs.append({"path": path, "image_name": relative.as_posix(), "case_id": case,
                       "tooth_id": tooth, "visit": visit, "reference_label": label})
    return inputs


def load_model(weights, device):
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def predict_views(model, inputs, device="cpu", batch_size=100):
    """Return scores for each crop; model class 0 denotes gingivitis."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    records = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            batch = inputs[start:start + batch_size]
            tensors = []
            for item in batch:
                with Image.open(item["path"]) as image:
                    tensors.append(transform(image.convert("RGB")))
            probabilities = model(torch.stack(tensors).to(device)).softmax(dim=1).cpu()
            if probabilities.shape != (len(batch), 2) or not torch.isfinite(probabilities).all():
                raise ValueError("Expected finite probabilities for two model classes")
            for item, probabilities_row in zip(batch, probabilities):
                record = {key: value for key, value in item.items() if key != "path"}
                record.update({"prob_gingivitis": float(probabilities_row[0]),
                               "prob_normal": float(probabilities_row[1]),
                               "prediction": CLINICAL_LABELS[MODEL_CLASSES[int(probabilities_row.argmax())]]})
                records.append(record)
    return records


def run_pipeline(model, inputs, device="cpu", batch_size=100, aggregate=None):
    """Keep tooth/patient decisions outside the public image-level implementation."""
    records = predict_views(model, inputs, device, batch_size)
    if aggregate is None:
        return {"image_predictions": records}
    if not callable(aggregate):
        raise TypeError("aggregate must be callable")
    return {"image_predictions": records, "diagnostic_results": aggregate(records)}
