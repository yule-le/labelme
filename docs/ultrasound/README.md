# Ultrasound auto-annotation

The extension keeps model inference separate from Labelme's canvas and JSON
code. It uses two published, independent transverse models and exports three
native editable Labelme polygons: `EMA`, `fat`, and `skin`.

The production strategy is:

1. run `seg_eye_muscle__transverse__v1` to produce `EMA`;
2. run the separate `seg_skin__transverse__v1` model to produce `skin`;
3. derive the `fat` polygon between the skin lower boundary and EMA upper
   boundary.

The experimental `seg_multilabel__transverse__v1` artifact is not used by the
production auto-label workflow.

## Step 3 scope

Step 3 runs the eye-muscle and skin models independently. After a user edits
those two polygons, `fat` is derived from the current canvas boundaries rather
than the original model masks. Selected model results and each derived-fat
update are separate canvas transactions, so one Undo restores the previous
state. Running the current-image model command again replaces existing shapes
with the selected labels. The results use ordinary Labelme polygons with empty
flags and descriptions, so the Annotation List and exported JSON do not
distinguish them from manually drawn shapes.

For dense model polygons, select exactly one polygon in **Edit Shapes**, then
hold **Shift** and drag with the left mouse button to draw a temporary point
selection box. Selected vertices are highlighted. Press **Delete** to remove
them and reconnect the surrounding retained vertices with straight segments,
**Esc** to cancel the selection, or **Ctrl+Z** to undo a deletion. The box is
never saved to JSON. Deletion is blocked when it would leave fewer than three
polygon vertices.

## Toolbar tasks

The existing **AI-Assisted Annotation** toolbar position contains the
transverse-ultrasound controls:

- `EMA` runs `seg_eye_muscle__transverse__v1`.
- `skin` runs `seg_skin__transverse__v1`.
- **Generate Fat** requires exactly one edited `EMA` polygon and one edited
  `skin` polygon. It uses the skin lower boundary and EMA upper boundary,
  replacing only an existing `fat` polygon. It does not rerun either model or
  use the independent fat artifact.

**Run Current** applies the checked tasks to the open image. **Run Folder**
processes every image listed from the open directory, creates JSON only for
images without an existing annotation file, continues past per-image failures,
and reports created/skipped/failed totals. Folder runs generate only the checked
EMA/skin model tasks; fat is generated after per-image boundary review. The
configured annotation output directory is respected; otherwise JSON is written
beside each image.

The toolbar's separate **Review** panel replaces AI Text-to-Annotation for this
workflow. **Needs expert review** marks the small number of uncertain
annotations that require expert review, while **Model failure** marks unusable
or clearly failed model output. Both image-level checkboxes default off, remain
independent, and are never enabled automatically by current-image or folder
inference. When either is checked, the JSON carries an extensible top-level
object:

```json
"ultrasoundReview": {
  "required": true,
  "modelFailure": false,
  "reasons": [],
  "note": ""
}
```

When both are unchecked, the review object is removed. Shape flags and
descriptions remain empty.

## Configuration

Copy `configs/ultrasound/models.example.yaml` to
`configs/ultrasound/models.local.yaml`, then set `weights_path` and
`config_path` to the published artifact files. The local file and
`model_artifacts/` are intentionally ignored by Git.

Set the optional local `annotation_root` to the parent directory containing
the ROI batch folders. **Open Folder** will start there on every invocation,
making it possible to choose the required batch directly. If the setting is
absent, null, or its directory is unavailable, Labelme falls back to its normal
previous/current-directory behavior.

The transverse ROI contract accepts either:

- the raw `800 x 600` image and crops `(x=92, y=31, w=536, h=536)`; or
- a prepared ROI image, using its complete width and height.

Set `accept_variable_cropped_size: true` only when the annotation workflow
opens images that have already been cropped to the ultrasound ROI. Prepared
batches may have different crop-template dimensions—for example, the
`roi_t_b006` batch is `599 x 538`. Every accepted ROI is resized to the
model's `512 x 512` input and predictions are mapped back to its original
dimensions. An exact `800 x 600` input is always treated as a raw frame and
uses the configured rectangle instead of the variable-size path.

## Runtime

PyTorch and torchvision are imported only when ultrasound inference is first
requested. Labelme can start and retain its normal annotation features without
them; clicking the ultrasound command produces a clear error until they are
installed. Install the runtime with:

```powershell
uv sync --extra ultrasound
```

Use **Run Current** in the top **AI-Assisted Annotation** panel after opening an
image. Inference runs on a persistent `QThread`, and a request ID plus normalized
image path prevents a completed result from being applied after the user changes
images.

## Ownership boundaries

- `labelme/_ultrasound/adapters/`: artifact loading and model execution.
- `labelme/_ultrasound/preprocessing.py`: input conversion and reversible ROI
  geometry.
- `labelme/_ultrasound/postprocessing.py`: mask cleanup and polygon extraction.
- `labelme/_ultrasound/worker.py`: background execution without UI access.
- `labelme/_ultrasound/labelme_adapter.py`: native `Shape` conversion.
- `tests/ultrasound/`: framework-independent tests.
- `tests/e2e/`: Labelme canvas, undo, save, and reopen checks.
- `test_data/ultrasound/`: de-identified golden fixtures.
