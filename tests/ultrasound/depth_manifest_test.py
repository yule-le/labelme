from __future__ import annotations

from pathlib import Path

import pytest

from labelme._ultrasound.depth_manifest import DepthManifest
from labelme._ultrasound.depth_manifest import DepthManifestError


def test_manifest_matches_filename_and_depth_ocr_mm_to_cropped_image(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        "sample_id,filename,depth_ocr_text,depth_ocr_mm,ocr_status\n"
        "s_123,2026Jul29-10.50.59.jpg,10,100,ok\n",
        encoding="utf-8",
    )

    manifest = DepthManifest.load(manifest_path)
    entry = manifest.lookup(tmp_path / "cropped" / "2026Jul29-10.50.59.jpg")

    assert entry is not None
    assert entry.depth_setting_mm == 100.0
    assert entry.original_image_path == "2026Jul29-10.50.59.jpg"


def test_manifest_keeps_legacy_columns_compatible(tmp_path: Path) -> None:
    manifest_path = tmp_path / "legacy.csv"
    manifest_path.write_text(
        "original_image_path,depth_setting\n"
        "backend/data/raw/batch/image.jpg,90\n",
        encoding="utf-8",
    )

    entry = DepthManifest.load(manifest_path).lookup("cropped/image.jpg")

    assert entry is not None
    assert entry.depth_setting_mm == 90.0
    assert entry.original_image_path.startswith("backend/data/raw/")


def test_manifest_rejects_conflicting_duplicate_filenames(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        "filename,depth_ocr_mm\n"
        "shared.jpg,100\n"
        "shared.jpg,120\n",
        encoding="utf-8",
    )

    with pytest.raises(DepthManifestError, match="conflicting"):
        DepthManifest.load(manifest_path)


@pytest.mark.parametrize("depth", ["", "zero", "0", "-1", "nan"])
def test_manifest_rejects_invalid_depth(tmp_path: Path, depth: str) -> None:
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        f"filename,depth_ocr_mm\nimage.jpg,{depth}\n",
        encoding="utf-8",
    )

    with pytest.raises(DepthManifestError):
        DepthManifest.load(manifest_path)
