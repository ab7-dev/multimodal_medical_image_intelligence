"""Generate Person B's JSON artifacts when demo images are supplied."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def demo_images(folder):
    return sorted(
        path for path in Path(folder).rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def main():
    images = demo_images(ROOT / "data" / "demo_images")
    if not images:
        print("No demo images found; no mock JSON files written.")
        return
    if __package__:
        from .analyze import analyze_image
    else:
        from analyze import analyze_image

    output_dir = ROOT / "data" / "mock_vision"
    output_dir.mkdir(parents=True, exist_ok=True)
    print("image_id\tstatus\tfindings")
    for image_path in images:
        result = analyze_image(image_path)
        destination = output_dir / f"{result['image_id']}.json"
        destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        positives = ", ".join(
            f"{item['label']}={item['score']:.3f}"
            for item in result["findings"] if item["score"] >= 0.3
        ) or "none >= 0.3"
        print(f"{result['image_id']}\t{result['quality']['status']}\t{positives}")


if __name__ == "__main__":
    main()
