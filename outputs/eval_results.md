The pretrained weights may have been trained on NIH data, so these numbers are probably inflated.

## Evaluation status

AUROC: not evaluated.
Degraded-copy AUROC and drops: not evaluated.
Calibration: not evaluated.
Grad-CAM box IoU and hit rate: not evaluated.

## Missing inputs

- `data/nih/Data_Entry_2017.csv`
- `data/nih/BBox_List_2017.csv`
- NIH image files referenced by the metadata, somewhere under `data/nih/`.
- CSV columns and labels cannot be inspected until the CSVs are supplied; NIH image files are also absent.
