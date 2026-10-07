"""Pretrained chest X-ray scoring and Grad-CAM visualization."""

from pathlib import Path

import cv2
import numpy as np
import torch

if __package__:
    from vision.check_model import LABELS, MODEL_WEIGHTS, choose_gradcam_layer, configure_progress_encoding
    from vision.gradcam import generate_gradcam, largest_region, localization_is_usable
    from vision.quality import check_quality
else:
    from check_model import LABELS, MODEL_WEIGHTS, choose_gradcam_layer, configure_progress_encoding
    from gradcam import generate_gradcam, largest_region, localization_is_usable
    from quality import check_quality


ROOT = Path(__file__).resolve().parents[1]
MODEL = None
XRV = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRADCAM_LAYER = None


def load_pretrained_model():
    global MODEL, XRV, GRADCAM_LAYER
    if MODEL is None:
        import torchxrayvision as xrv

        configure_progress_encoding()
        model = xrv.models.DenseNet(weights=MODEL_WEIGHTS)
        model.eval()
        model.to(DEVICE)
        pathologies = list(model.pathologies)
        missing = [label for label in LABELS if label not in pathologies]
        if missing:
            raise RuntimeError(f"STOP: requested labels missing from pretrained model: {missing}")
        GRADCAM_LAYER = choose_gradcam_layer(model.features)
        XRV = xrv
        MODEL = model
    return MODEL


def preprocess_image(image_path):
    load_pretrained_model()
    image = XRV.utils.load_image(str(image_path))
    image = XRV.datasets.XRayCenterCrop()(image)
    image = XRV.datasets.XRayResizer(224)(image)
    return torch.from_numpy(np.asarray(image, dtype=np.float32)).unsqueeze(0).to(DEVICE)


def model_scores_to_label_map(model, input_tensor):
    with torch.no_grad():
        output = model(input_tensor)[0].detach().cpu().numpy()
    if not np.isfinite(output).all() or output.min() < 0 or output.max() > 1:
        # Scores are not rescaled; a non-probability output must be calibrated explicitly.
        raise ValueError("Model output is outside [0, 1]; refusing to rescale model scores")
    pathology_indices = {label: model.pathologies.index(label) for label in LABELS}
    return {label: float(output[pathology_indices[label]]) for label in LABELS}, pathology_indices


def save_heatmap_overlay(image, heatmap, image_id, label):
    height, width = image.shape[:2]
    resized = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_LINEAR)
    colored = cv2.applyColorMap(np.uint8(np.clip(resized, 0, 1) * 255), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image[:, :, :3], 0.55, colored, 0.45, 0)
    destination = ROOT / "outputs" / "heatmaps" / f"{image_id}_{label}.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), overlay):
        raise OSError(f"Could not save heatmap overlay: {destination}")
    return destination.relative_to(ROOT).as_posix()


def analyze_image(image_path):
    image_path = Path(image_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    quality = check_quality(image)
    image_id = image_path.stem
    if not quality["is_chest_xray"]:
        return {"image_id": image_id, "quality": quality, "findings": []}

    model = load_pretrained_model()
    input_tensor = preprocess_image(image_path)
    scores, pathology_indices = model_scores_to_label_map(model, input_tensor)
    target_layer = model.features.get_submodule(GRADCAM_LAYER)
    findings = []
    for label in LABELS:
        score = scores[label]
        region = None
        heatmap_path = None
        if score >= 0.3:
            cam = generate_gradcam(model, input_tensor, target_layer, pathology_indices[label])
            cam = cv2.resize(cam, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
            if localization_is_usable(cam):
                region = largest_region(cam)
                heatmap_path = save_heatmap_overlay(image, cam, image_id, label)
        findings.append({
            "label": label,
            "score": score,
            "region": region,
            "heatmap_path": heatmap_path,
        })
    return {"image_id": image_id, "quality": quality, "findings": findings}
