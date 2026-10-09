"""CPU checks for gingivitis preprocessing, score encoding and release integrity."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


core = load_module("gingivitis_core", ROOT / "gingivitis/inference/core.py")
v4 = load_module("gingivitis_v4", ROOT / "gingivitis/training/train_v4_focal_labelsmooth.py")


class FixedClassifier(torch.nn.Module):
    def forward(self, images):
        return images.new_tensor([2.0, -1.0]).repeat(len(images), 1)


class GingivitisChecks(unittest.TestCase):
    def test_tooth_and_visit_parsing(self):
        self.assertEqual(core.parse_view("D0279_Y-70deg_36.jpg"), ("D0279", "36", None))
        self.assertEqual(core.parse_view("Gingivitis/IB-001-2/IB-001_tooth36.png", True),
                         ("IB-001", "36", 2))
        self.assertEqual(core.parse_view("IB-001_tooth36_2.png", True), ("IB-001", "36", 3))
        with self.assertRaises(ValueError):
            core.parse_view("upper_whole_jaw.png")

    def test_inference_and_optional_aggregation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in ("Gingivitis/D0064/D0064_Y+00deg_11.png",
                             "Normal/D0065/D0065_Y+00deg_21.png"):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (32, 24), "white").save(path)
            inputs = core.collect_inputs(root)
            self.assertEqual([item["reference_label"] for item in inputs], [1, 0])
            result = core.run_pipeline(FixedClassifier(), inputs, batch_size=1)
            self.assertEqual(set(result), {"image_predictions"})
            for record in result["image_predictions"]:
                self.assertEqual(record["prediction"], 1)
                self.assertGreater(record["prob_gingivitis"], record["prob_normal"])
                self.assertAlmostEqual(record["prob_gingivitis"] + record["prob_normal"], 1.0, places=6)
                self.assertNotIn("path", record)
            json.dumps(result, allow_nan=False)
            result = core.run_pipeline(FixedClassifier(), inputs, aggregate=lambda rows: {"count": len(rows)})
            self.assertEqual(result["diagnostic_results"], {"count": 2})
            with self.assertRaises(TypeError):
                core.run_pipeline(FixedClassifier(), inputs, aggregate=42)
            with self.assertRaises(ValueError):
                core.predict_views(FixedClassifier(), inputs, batch_size=0)

    def test_unlabeled_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (8, 8)).save(root / "D0064_Y+00deg_11.png")
            self.assertIsNone(core.collect_inputs(root)[0]["reference_label"])

    def test_focal_loss_matches_source_formula(self):
        logits = torch.tensor([[2.0, -1.0], [-0.2, 0.7]], requires_grad=True)
        labels = torch.tensor([0, 1])
        alpha = torch.tensor([1.5, 0.5])
        target = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
        cross_entropy = -(target * logits.log_softmax(dim=1)).sum(dim=1)
        expected = (alpha[labels] * (1 - cross_entropy.exp().reciprocal()) ** 2 * cross_entropy).mean()
        actual = v4.FocalLossWithLabelSmoothing(alpha, gamma=2, smoothing=0.1)(logits, labels)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_resnet_checkpoint_compatibility(self):
        model = v4.create_model(pretrained=False)
        self.assertEqual(model.fc.out_features, 2)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "test_model.pth"
            torch.save(model.state_dict(), checkpoint)
            restored = core.load_model(checkpoint, "cpu")
            restored.eval()
            with torch.inference_mode():
                self.assertEqual(tuple(restored(torch.zeros(1, 3, 224, 224)).shape), (1, 2))

    def test_display_images(self):
        root = ROOT / "gingivitis/display_figures"
        self.assertEqual(len(list(root.rglob("*.png"))), 625)
        self.assertFalse(list(root.rglob("desktop.ini")))
        for path in [*root.rglob("*.png"), ROOT / "figures/gingivitis_architecture.png"]:
            with Image.open(path) as image:
                image.verify()


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
