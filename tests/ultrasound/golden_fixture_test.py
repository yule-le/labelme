from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

_FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "test_data/ultrasound"


def test_transverse_ema_fixture_is_deidentified_and_matches_json() -> None:
    image_path = _FIXTURE_ROOT / "transverse_ema_golden.jpg"
    json_path = _FIXTURE_ROOT / "transverse_ema_golden.json"
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    with Image.open(image_path) as image:
        assert image.size == (536, 536)
    assert payload["imagePath"] == image_path.name
    assert (payload["imageWidth"], payload["imageHeight"]) == (536, 536)
    assert [shape["label"] for shape in payload["shapes"]] == [
        "EMA",
        "fat",
        "skin",
    ]
    assert len(payload["shapes"][0]["points"]) == 18

    image_bytes = image_path.read_bytes()
    for field in (
        b"PATIENT_ID",
        b"PATIENT_NAME",
        b"CURRENT_USER",
        b"CLINIC_NAME",
    ):
        assert field not in image_bytes
