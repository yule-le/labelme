"""Load per-image ultrasound depth settings from an exported CSV manifest."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path


class DepthManifestError(ValueError):
    """Raised when a depth manifest cannot be used safely."""


@dataclass(frozen=True)
class DepthManifestEntry:
    filename: str
    original_image_path: str
    depth_setting_mm: float
    row_number: int


@dataclass(frozen=True)
class DepthManifest:
    source_path: Path
    entries_by_filename: dict[str, DepthManifestEntry]

    @classmethod
    def load(cls, csv_path: str | Path) -> DepthManifest:
        source_path = Path(csv_path)
        try:
            stream = source_path.open("r", encoding="utf-8-sig", newline="")
        except OSError as exc:
            raise DepthManifestError(f"Could not open CSV: {exc}") from exc

        entries: dict[str, DepthManifestEntry] = {}
        with stream:
            reader = csv.DictReader(stream)
            fieldnames = set(reader.fieldnames or ())
            if {"filename", "depth_ocr_mm"} <= fieldnames:
                image_column = "filename"
                depth_column = "depth_ocr_mm"
            elif {"original_image_path", "depth_setting"} <= fieldnames:
                image_column = "original_image_path"
                depth_column = "depth_setting"
            else:
                raise DepthManifestError(
                    "CSV must contain filename and depth_ocr_mm "
                    "(or legacy original_image_path and depth_setting) columns."
                )
            for row_number, row in enumerate(reader, start=2):
                original = (row.get(image_column) or "").strip()
                raw_depth = (row.get(depth_column) or "").strip()
                if not original:
                    raise DepthManifestError(
                        f"Row {row_number}: {image_column} is empty."
                    )
                filename = original.replace("\\", "/").rsplit("/", 1)[-1]
                if not filename:
                    raise DepthManifestError(
                        f"Row {row_number}: {image_column} has no filename."
                    )
                try:
                    depth = float(raw_depth)
                except ValueError as exc:
                    raise DepthManifestError(
                        f"Row {row_number}: invalid {depth_column} {raw_depth!r}."
                    ) from exc
                if not math.isfinite(depth) or depth <= 0.0:
                    raise DepthManifestError(
                        f"Row {row_number}: {depth_column} must be greater than zero."
                    )

                key = filename.casefold()
                entry = DepthManifestEntry(
                    filename=filename,
                    original_image_path=original,
                    depth_setting_mm=depth,
                    row_number=row_number,
                )
                previous = entries.get(key)
                if previous is not None:
                    if (
                        previous.depth_setting_mm != entry.depth_setting_mm
                        or previous.original_image_path != entry.original_image_path
                    ):
                        raise DepthManifestError(
                            f"Rows {previous.row_number} and {row_number} contain "
                            f"conflicting entries for filename {filename!r}."
                        )
                    continue
                entries[key] = entry

        if not entries:
            raise DepthManifestError("CSV contains no image depth rows.")
        return cls(source_path=source_path, entries_by_filename=entries)

    def lookup(self, image_path: str | Path) -> DepthManifestEntry | None:
        return self.entries_by_filename.get(Path(image_path).name.casefold())

    def matched_count(self, image_paths: list[str]) -> int:
        return sum(self.lookup(path) is not None for path in image_paths)
