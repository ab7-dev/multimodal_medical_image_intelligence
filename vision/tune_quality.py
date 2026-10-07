"""Report sharp/blurry sample blur values and tune the quality threshold."""

from pathlib import Path

import cv2

try:
    from .quality import check_quality
except ImportError:
    from quality import check_quality


ROOT = Path(__file__).resolve().parents[1]


def blur_values(folder):
    values = []
    for path in sorted(Path(folder).glob("*")):
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is not None:
                values.append((path, check_quality(image)["blur"]))
    return values


def main():
    sharp = blur_values(ROOT / "data" / "tune" / "sharp")
    blurry = blur_values(ROOT / "data" / "tune" / "blurry")
    for group_name, group in (("sharp", sharp), ("blurry", blurry)):
        print(f"{group_name} images: {len(group)}")
        for path, blur in group:
            print(f"  {path.name}: {blur:.3f}")
    if sharp and blurry:
        sharp_floor = min(value for _, value in sharp)
        blurry_ceiling = max(value for _, value in blurry)
        if blurry_ceiling < sharp_floor:
            threshold = (blurry_ceiling + sharp_floor) / 2
            print(f"Set BLUR_THRESHOLD = {threshold:.3f}")
        else:
            print("Groups overlap; BLUR_THRESHOLD unchanged.")
    else:
        print("BLUR_THRESHOLD unchanged: sharp and blurry sample images are both required.")


if __name__ == "__main__":
    main()
