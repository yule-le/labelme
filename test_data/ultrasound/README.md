# Ultrasound test data

Place each de-identified ultrasound image next to its expected Labelme JSON
file using the same stem:

```text
transverse_001.png
transverse_001.json
sagittal_001.png
sagittal_001.json
```

Generated predictions do not belong here. Write them to `test_outputs/`.

`transverse_ema_golden.jpg` is the metadata-stripped `536 x 536` ROI from a
held-out transverse test sample. Its matching JSON contains expert-reviewed
`EMA`, `fat`, and `skin` polygons for Step 2 and Step 3 integration checks. The
raw `800 x 600` source is intentionally not copied because its JPEG application
metadata contains patient-related fields.
