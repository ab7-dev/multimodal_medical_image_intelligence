import json
from types import SimpleNamespace
from pathlib import Path

import cv2
import numpy as np
import pytest
from jsonschema import Draft202012Validator

from vision.quality import check_quality
from vision.check_data import main as check_data_main
from vision.tune_quality import main as tune_quality_main
from vision.tune_quality import blur_values
from vision.check_model import choose_gradcam_layer, configure_progress_encoding, main as check_model_main, model_factory_kwargs, preprocess_real_image, summarize_scores
from vision import analyze, gradcam
from vision.evaluate import main as evaluate_main, missing_inputs, write_not_evaluated_report
from vision.make_mocks import demo_images, main as make_mocks_main


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "contracts" / "vision_result.schema.json").read_text())


@pytest.mark.parametrize(
    "filename",
    [
        "vision_result.example.json",
        "vision_result.poor_quality.example.json",
        "vision_result.not_xray.example.json",
    ],
)
def test_contract_examples_validate(filename):
    payload = json.loads((ROOT / "contracts" / filename).read_text())
    Draft202012Validator(SCHEMA).validate(payload)


def test_schema_enforces_low_score_null_outputs():
    example = json.loads((ROOT / "contracts" / "vision_result.example.json").read_text())
    example["findings"][0]["score"] = 0.29
    example["findings"][0]["region"] = None
    example["findings"][0]["heatmap_path"] = None
    Draft202012Validator(SCHEMA).validate(example)
    example["findings"][0]["region"] = {"x": 0, "y": 0, "w": 1, "h": 1}
    with pytest.raises(Exception):
        Draft202012Validator(SCHEMA).validate(example)


def test_check_quality_marks_gaussian_blur_poor():
    rng = np.random.default_rng(2)
    image = rng.integers(0, 256, size=(512, 512), dtype=np.uint8)
    blurred = cv2.GaussianBlur(image, (21, 21), 0)
    result = check_quality(blurred)
    assert result["status"] == "poor"
    assert "blurry" in result["issues"]


def test_check_quality_blur_similar_at_two_resolutions():
    rng = np.random.default_rng(3)
    image = rng.integers(0, 256, size=(512, 512), dtype=np.uint8)
    high_resolution = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    full_blur = check_quality(image)["blur"]
    high_blur = check_quality(high_resolution)["blur"]
    assert abs(full_blur - high_blur) / max(full_blur, high_blur) <= 0.25


@pytest.mark.parametrize("intensity", [0, 255])
def test_check_quality_black_and_white_are_poor(intensity):
    result = check_quality(np.full((256, 256), intensity, dtype=np.uint8))
    assert result["status"] == "poor"


def test_check_quality_rejects_color_noise():
    rng = np.random.default_rng(4)
    image = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    result = check_quality(image)
    assert result["is_chest_xray"] is False
    assert result["status"] == "poor"


def test_blur_values_reads_supported_image_files(tmp_path):
    image = np.full((32, 32), 120, dtype=np.uint8)
    image_path = tmp_path / "sample.png"
    assert cv2.imwrite(str(image_path), image)
    values = blur_values(tmp_path)
    assert len(values) == 1
    assert values[0][0] == image_path
    assert values[0][1] >= 0


def test_check_data_main_reports_missing_metadata(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("vision.check_data.ROOT", tmp_path)
    check_data_main()
    output = capsys.readouterr().out
    assert "Data_Entry_2017.csv" in output
    assert "BBox_List_2017.csv" in output


def test_tune_quality_main_handles_missing_samples(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("vision.tune_quality.ROOT", tmp_path)
    tune_quality_main()
    assert "BLUR_THRESHOLD unchanged" in capsys.readouterr().out


def test_model_factory_kwargs_uses_inspected_weights_parameter():
    def factory(weights=None):
        return weights

    assert model_factory_kwargs(factory) == {"weights": "densenet121-res224-all"}


def test_choose_gradcam_layer_selects_last_convolution():
    features = __import__("torch").nn.Sequential(
        __import__("torch").nn.Conv2d(1, 2, 3),
        __import__("torch").nn.ReLU(),
        __import__("torch").nn.Conv2d(2, 1, 1),
    )
    assert choose_gradcam_layer(features) == "2"


def test_summarize_scores_reports_shape_range_and_scale():
    torch = __import__("torch")
    summary = summarize_scores(torch.tensor([[0.2, 0.8]]))
    assert summary == {"shape": (1, 2), "min": pytest.approx(0.2), "max": pytest.approx(0.8), "within_zero_one": True}


def test_preprocess_real_image_uses_library_crop_and_resize():
    class Transforms:
        def XRayCenterCrop(self):
            return lambda image: image

        def XRayResizer(self, size):
            return lambda image: np.zeros((1, size, size), dtype=np.float32)

    fake_xrv = SimpleNamespace(
        utils=SimpleNamespace(load_image=lambda _path: np.zeros((1, 12, 12), dtype=np.float32)),
        datasets=Transforms(),
    )
    assert tuple(preprocess_real_image(fake_xrv, "sample.png").shape) == (1, 1, 224, 224)


def test_configure_progress_encoding_sets_utf8(monkeypatch):
    encodings = []
    fake_stdout = SimpleNamespace(reconfigure=lambda **kwargs: encodings.append(kwargs))
    monkeypatch.setattr("vision.check_model.sys.stdout", fake_stdout)
    configure_progress_encoding()
    assert encodings == [{"encoding": "utf-8", "errors": "backslashreplace"}]


def test_check_model_main_uses_mock_installed_library(tmp_path, monkeypatch, capsys):
    import torch
    from torch import nn

    class FakeModel(nn.Module):
        pathologies = ["Pneumonia", "Effusion", "Pneumothorax"]

        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(nn.Conv2d(1, 1, 1))

        def forward(self, image):
            return torch.full((image.shape[0], 3), 0.5, device=image.device)

    xrv_module = SimpleNamespace(
        models=SimpleNamespace(DenseNet=lambda weights: FakeModel()),
        utils=SimpleNamespace(load_image=lambda path: np.zeros((1, 8, 8), dtype=np.float32)),
        datasets=SimpleNamespace(XRayCenterCrop=lambda: None, XRayResizer=lambda size: None),
    )
    monkeypatch.setattr("vision.check_model.ROOT", tmp_path)
    check_model_main(xrv_module)
    output = capsys.readouterr().out
    assert "GRADCAM_LAYER = 0" in output
    assert "missing labels = []" in output


def test_demo_images_lists_supported_files(tmp_path):
    (tmp_path / "sample.jpg").write_bytes(b"image placeholder")
    (tmp_path / "ignore.txt").write_text("not an image")
    assert [path.name for path in demo_images(tmp_path)] == ["sample.jpg"]


def test_make_mocks_main_skips_empty_demo_folder(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("vision.make_mocks.ROOT", tmp_path)
    make_mocks_main()
    assert "No demo images found" in capsys.readouterr().out


def test_missing_inputs_reports_exact_csvs(tmp_path):
    assert missing_inputs(tmp_path / "metadata.csv", tmp_path / "boxes.csv") == [
        "data/nih/Data_Entry_2017.csv",
        "data/nih/BBox_List_2017.csv",
    ]


def test_write_not_evaluated_report_has_required_disclosure(tmp_path):
    report_path = tmp_path / "eval_results.md"
    write_not_evaluated_report(report_path, ["data/nih/Data_Entry_2017.csv"])
    report = report_path.read_text()
    assert report.startswith("The pretrained weights may have been trained on NIH data, so these numbers are probably inflated.")
    assert "AUROC: not evaluated." in report


def test_evaluate_main_writes_report_for_missing_csvs(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("vision.evaluate.METADATA", tmp_path / "Data_Entry_2017.csv")
    monkeypatch.setattr("vision.evaluate.BOXES", tmp_path / "BBox_List_2017.csv")
    monkeypatch.setattr("vision.evaluate.REPORT", tmp_path / "outputs" / "eval_results.md")
    evaluate_main()
    assert "not run" in capsys.readouterr().out
    assert (tmp_path / "outputs" / "eval_results.md").is_file()


def test_normalize_heatmap_scales_to_zero_one():
    result = gradcam.normalize_heatmap(np.array([[-1.0, 2.0], [1.0, 0.0]]))
    assert result.min() == pytest.approx(0.0)
    assert result.max() == pytest.approx(1.0)


def test_largest_region_returns_normalized_bbox():
    heatmap = np.zeros((10, 20), dtype=np.float32)
    heatmap[2:6, 5:11] = 1.0
    assert gradcam.largest_region(heatmap) == {"x": 0.25, "y": 0.2, "w": 0.3, "h": 0.4}


def test_localization_guard_accepts_compact_interior_heatmap():
    heatmap = np.zeros((100, 100), dtype=np.float32)
    heatmap[30:60, 30:60] = 1.0
    assert gradcam.localization_is_usable(heatmap)


def test_localization_guard_rejects_broad_edge_heatmap():
    heatmap = np.zeros((100, 100), dtype=np.float32)
    heatmap[:40, :95] = 1.0
    assert not gradcam.localization_is_usable(heatmap)


def test_generate_gradcam_returns_normalized_map():
    import torch
    from torch import nn

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 1, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Linear(1, 2)

        def forward(self, image):
            activation = torch.relu(self.conv(image))
            return torch.sigmoid(self.head(self.pool(activation).flatten(1)))

    model = TinyModel()
    heatmap = gradcam.generate_gradcam(model, torch.rand(1, 1, 8, 8), model.conv, 0)
    assert heatmap.shape == (8, 8)
    assert 0 <= heatmap.min() <= heatmap.max() <= 1


def test_preprocess_image_uses_torchxrayvision_transforms(monkeypatch, tmp_path):
    import torch

    class Transforms:
        def XRayCenterCrop(self):
            return lambda image: image

        def XRayResizer(self, size):
            return lambda image: np.zeros((1, size, size), dtype=np.float32)

    fake_xrv = SimpleNamespace(
        utils=SimpleNamespace(load_image=lambda path: np.zeros((1, 12, 12), dtype=np.float32)),
        datasets=Transforms(),
    )
    monkeypatch.setattr(analyze, "MODEL", object())
    monkeypatch.setattr(analyze, "XRV", fake_xrv)
    result = analyze.preprocess_image(tmp_path / "sample.png")
    assert result.shape == (1, 1, 224, 224)
    assert result.dtype == torch.float32


def test_model_scores_to_label_map_preserves_selected_scores():
    import torch

    class CallableModel:
        pathologies = ["Effusion", "Pneumonia", "Pneumothorax"]

        def __call__(self, tensor):
            return torch.tensor([[0.2, 0.8, 0.4]])

    result, indices = analyze.model_scores_to_label_map(CallableModel(), torch.zeros(1, 1, 224, 224))
    assert result == {"Pneumonia": pytest.approx(0.8), "Effusion": pytest.approx(0.2), "Pneumothorax": pytest.approx(0.4)}
    assert indices["Pneumonia"] == 1


def test_model_scores_to_label_map_rejects_out_of_range_scores():
    import torch

    class CallableModel:
        pathologies = ["Pneumonia", "Effusion", "Pneumothorax"]

        def __call__(self, tensor):
            return torch.tensor([[0.2, 1.2, 0.4]])

    with pytest.raises(ValueError, match="refusing to rescale"):
        analyze.model_scores_to_label_map(CallableModel(), torch.zeros(1, 1, 224, 224))


def test_load_pretrained_model_caches_and_selects_layer(monkeypatch):
    import sys
    from torch import nn

    class FakeModel(nn.Module):
        pathologies = ["Pneumonia", "Effusion", "Pneumothorax"]

        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(nn.Conv2d(1, 1, 1))

        def forward(self, image):
            return image

    fake_xrv = SimpleNamespace(models=SimpleNamespace(DenseNet=lambda weights: FakeModel()))
    monkeypatch.setitem(sys.modules, "torchxrayvision", fake_xrv)
    monkeypatch.setattr(analyze, "MODEL", None)
    monkeypatch.setattr(analyze, "XRV", None)
    monkeypatch.setattr(analyze, "GRADCAM_LAYER", None)
    loaded = analyze.load_pretrained_model()
    assert loaded is analyze.MODEL
    assert analyze.GRADCAM_LAYER == "0"


def test_save_heatmap_overlay_writes_png(monkeypatch, tmp_path):
    image = np.zeros((16, 20, 3), dtype=np.uint8)
    heatmap = np.ones((8, 10), dtype=np.float32)
    monkeypatch.setattr(analyze, "ROOT", tmp_path)
    path = analyze.save_heatmap_overlay(image, heatmap, "sample", "Effusion")
    assert path == "outputs/heatmaps/sample_Effusion.png"
    assert (tmp_path / path).is_file()


def test_analyze_image_rejects_non_xray_and_matches_schema(tmp_path):
    rng = np.random.default_rng(5)
    image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    image_path = tmp_path / "color_noise.png"
    assert cv2.imwrite(str(image_path), image)
    result = analyze.analyze_image(image_path)
    Draft202012Validator(SCHEMA).validate(result)
    assert result["image_id"] == "color_noise"
    assert result["quality"]["is_chest_xray"] is False
    assert result["findings"] == []


def test_analyze_image_leaves_low_scores_without_regions_or_heatmaps(monkeypatch, tmp_path):
    import torch
    from torch import nn

    image_path = tmp_path / "xray.png"
    assert cv2.imwrite(str(image_path), np.tile(np.arange(256, dtype=np.uint8), (256, 1)))

    class FakeModel(nn.Module):
        pathologies = ["Pneumonia", "Effusion", "Pneumothorax"]

        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(nn.Conv2d(1, 1, 1))

        def forward(self, image):
            return torch.tensor([[0.2, 0.1, 0.05]], device=image.device)

    model = FakeModel()
    monkeypatch.setattr(analyze, "MODEL", model)
    monkeypatch.setattr(analyze, "GRADCAM_LAYER", "0")
    monkeypatch.setattr(analyze, "preprocess_image", lambda _path: torch.zeros(1, 1, 224, 224))
    result = analyze.analyze_image(image_path)
    Draft202012Validator(SCHEMA).validate(result)
    assert all(item["region"] is None and item["heatmap_path"] is None for item in result["findings"])


def test_analyze_image_suppresses_broad_cam_for_high_score(monkeypatch, tmp_path):
    import torch
    from torch import nn

    image_path = tmp_path / "xray.png"
    image = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
    assert cv2.imwrite(str(image_path), image)

    class FakeModel(nn.Module):
        pathologies = ["Pneumonia", "Effusion", "Pneumothorax"]

        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(nn.Conv2d(1, 1, 1))

        def forward(self, input_tensor):
            return torch.tensor([[0.1, 0.1, 0.5]], device=input_tensor.device)

    broad_cam = np.zeros((8, 8), dtype=np.float32)
    broad_cam[:3, :] = 1.0
    monkeypatch.setattr(analyze, "MODEL", FakeModel())
    monkeypatch.setattr(analyze, "GRADCAM_LAYER", "0")
    monkeypatch.setattr(analyze, "preprocess_image", lambda _path: torch.zeros(1, 1, 224, 224))
    monkeypatch.setattr(analyze, "generate_gradcam", lambda *_args: broad_cam)

    result = analyze.analyze_image(image_path)
    Draft202012Validator(SCHEMA).validate(result)
    pneumo = next(item for item in result["findings"] if item["label"] == "Pneumothorax")
    assert pneumo["score"] == pytest.approx(0.5)
    assert pneumo["region"] is None
    assert pneumo["heatmap_path"] is None


SAMPLE_IMAGES = [
    path for path in (ROOT / "data" / "images").glob("*")
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
]


@pytest.mark.skipif(not SAMPLE_IMAGES, reason="sample images missing from data/images/")
def test_analyze_real_sample_matches_schema():
    result = analyze.analyze_image(SAMPLE_IMAGES[0])
    Draft202012Validator(SCHEMA).validate(result)
