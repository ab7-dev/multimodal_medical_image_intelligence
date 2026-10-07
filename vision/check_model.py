"""Inspect the requested pretrained DenseNet and its installed runtime behavior."""

import inspect
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
MODEL_WEIGHTS = "densenet121-res224-all"
LABELS = ("Pneumonia", "Effusion", "Pneumothorax")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def model_factory_kwargs(factory):
    parameters = inspect.signature(factory).parameters
    if "weights" not in parameters:
        raise RuntimeError(f"Installed DenseNet factory has no 'weights' parameter: {list(parameters)}")
    return {"weights": MODEL_WEIGHTS}


def choose_gradcam_layer(features):
    convolution_layers = [
        (name, module)
        for name, module in features.named_modules()
        if name and isinstance(module, torch.nn.Conv2d)
    ]
    if not convolution_layers:
        raise RuntimeError("No Conv2d module found under model.features")
    return convolution_layers[-1][0]


def summarize_scores(scores):
    values = scores.detach().float()
    minimum = float(values.min())
    maximum = float(values.max())
    return {
        "shape": tuple(values.shape),
        "min": minimum,
        "max": maximum,
        "within_zero_one": minimum >= 0.0 and maximum <= 1.0,
    }


def preprocess_real_image(xrv_module, image_path):
    image = xrv_module.utils.load_image(str(image_path))
    image = xrv_module.datasets.XRayCenterCrop()(image)
    image = xrv_module.datasets.XRayResizer(224)(image)
    return torch.from_numpy(np.asarray(image, dtype=np.float32)).unsqueeze(0)


def configure_progress_encoding():
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="backslashreplace")


def main(xrv_module=None):
    if xrv_module is None:
        import torchxrayvision as xrv_module

    factory = xrv_module.models.DenseNet
    configure_progress_encoding()
    model = factory(**model_factory_kwargs(factory))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    pathologies = list(model.pathologies)
    present = [label for label in LABELS if label in pathologies]
    missing = [label for label in LABELS if label not in pathologies]
    print(f"model.pathologies = {pathologies}")
    print(f"present labels = {present}")
    print(f"missing labels = {missing}")
    if missing:
        raise SystemExit("STOP: requested labels missing from pretrained model")

    with torch.no_grad():
        random_scores = model(torch.randn(1, 1, 224, 224, device=device))
    random_summary = summarize_scores(random_scores)
    print(f"random input output = {random_summary}")
    if getattr(model, "op_threshs", None) is not None:
        print("output interpretation = operating-point-normalized model scores; not calibrated probabilities")
    else:
        print(f"output lies within [0, 1] = {random_summary['within_zero_one']}; probability calibration not established")

    image_paths = sorted(
        path for path in (ROOT / "data" / "images").rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if image_paths:
        tensor = preprocess_real_image(xrv_module, image_paths[0]).to(device)
        with torch.no_grad():
            real_scores = model(tensor)
        print(f"real image = {image_paths[0]}")
        print(f"real input output = {summarize_scores(real_scores)}")
    else:
        print("real image output = not evaluated; data/images/ has no image files")

    print(f"device = {device}; CUDA available = {torch.cuda.is_available()}")
    print(f"top-level modules = {[name for name, _ in model.named_children()]}")
    print(f"model.features top-level modules = {[name for name, _ in model.features.named_children()]}")
    print(f"last model.features modules = {list(model.features.named_modules())[-12:]}")
    print(f"GRADCAM_LAYER = {choose_gradcam_layer(model.features)}")
    print(f"real-image loader signature = {inspect.signature(xrv_module.utils.load_image)}")
    print(f"XRayCenterCrop signature = {inspect.signature(xrv_module.datasets.XRayCenterCrop)}")
    print(f"XRayResizer signature = {inspect.signature(xrv_module.datasets.XRayResizer)}")


if __name__ == "__main__":
    main()
