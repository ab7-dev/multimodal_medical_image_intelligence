"""Inspect locally supplied NIH metadata without assuming CSV column names."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LABELS = ("Pneumonia", "Effusion", "Pneumothorax")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def main():
    image_root = ROOT / "data" / "nih"
    images = [path for path in image_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES] if image_root.exists() else []
    print(f"NIH images present on disk: {len(images)}")
    for csv_name in ("Data_Entry_2017.csv", "BBox_List_2017.csv"):
        csv_path = ROOT / "data" / "nih" / csv_name
        if not csv_path.is_file():
            print(f"Missing: {csv_path.relative_to(ROOT)}")
            continue
        table = pd.read_csv(csv_path)
        print(f"\n{csv_name} columns: {list(table.columns)}")
        print(table.head(3).to_string(index=False))
        print("Label/image counts not computed: confirm the label and image filename columns from these headers.")
    for label in LABELS:
        print(f"{label}: image count not evaluated; boxed image count not evaluated (<30 flag not evaluated)")
    if not images:
        print("Missing image folder contents: data/nih/ must contain the NIH image files referenced by the metadata.")


if __name__ == "__main__":
    main()
