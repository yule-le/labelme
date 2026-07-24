# Ultrasound auto-annotation

This directory contains design and operating documentation for the ultrasound
extension. General Labelme documentation remains in the parent `docs/`
directory.

The extension supports two strategies:

1. `boundary_derived`: predict skin and eye-muscle regions, then construct the
   fat polygon between the skin lower boundary and eye-muscle upper boundary.
1. `multilabel`: convert the transverse model's eye-muscle, fat, and skin
   channels directly into polygons.

Both strategies must return the framework-independent
`AnnotationPrediction` contract before any Qt or Labelme canvas code is
invoked.

## Ownership boundaries

- `labelme/_ultrasound/adapters/`: model loading and inference adapters.
- `labelme/_ultrasound/strategies/`: boundary-derived and multilabel logic.
- `labelme/_ultrasound/contracts.py`: Qt-independent output contracts.
- `configs/ultrasound/`: local model and strategy configuration.
- `tests/ultrasound/`: extension-specific automated tests.
- `test_data/ultrasound/`: de-identified image/JSON pairs.
- `test_outputs/`: generated local outputs; never committed.

Copy `configs/ultrasound/models.example.yaml` to
`configs/ultrasound/models.local.yaml` when machine-specific artifact paths
are needed. The local file and `model_artifacts/` directory are intentionally
ignored by Git.
