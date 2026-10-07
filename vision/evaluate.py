"""Run the requested NIH evaluation only after its source files are supplied."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "outputs" / "eval_results.md"
METADATA = ROOT / "data" / "nih" / "Data_Entry_2017.csv"
BOXES = ROOT / "data" / "nih" / "BBox_List_2017.csv"


def missing_inputs(metadata_path=METADATA, boxes_path=BOXES):
    missing = []
    if not Path(metadata_path).is_file():
        missing.append("data/nih/Data_Entry_2017.csv")
    if not Path(boxes_path).is_file():
        missing.append("data/nih/BBox_List_2017.csv")
    return missing


def write_not_evaluated_report(report_path, missing):
    report = [
        "The pretrained weights may have been trained on NIH data, so these numbers are probably inflated.",
        "",
        "## Evaluation status",
        "",
        "AUROC: not evaluated.",
        "Degraded-copy AUROC and drops: not evaluated.",
        "Calibration: not evaluated.",
        "Grad-CAM box IoU and hit rate: not evaluated.",
        "",
        "## Missing inputs",
        "",
    ]
    report.extend(f"- `{item}`" for item in missing)
    report.extend([
        "- NIH image files referenced by the metadata, somewhere under `data/nih/`.",
        "- A verified model package/weight load and inspected CSV column mapping are required before evaluation can run.",
        "",
    ])
    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(report), encoding="utf-8")


def main():
    missing = missing_inputs()
    if missing:
        write_not_evaluated_report(REPORT, missing)
        print(f"Evaluation not run; report written to {REPORT}")
        return
    raise RuntimeError(
        "Metadata is present, but column mapping and model inspection must be completed from the real files before evaluation."
    )


if __name__ == "__main__":
    main()
