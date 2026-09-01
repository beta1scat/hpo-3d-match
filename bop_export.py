"""Export structured pose predictions to the BOP19 CSV result format."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


OBJECT_NAME_TO_ID = {
    "bracket_planar": 5,
    "screw_black": 24,
    "star": 25,
}
OBJECT_ID_TO_NAME = {obj_id: name for name, obj_id in OBJECT_NAME_TO_ID.items()}
BOP19_FIELDS = ("scene_id", "im_id", "obj_id", "score", "R", "t", "time")
IMAGE_TIME_TOLERANCE_SEC = 1e-3


class BOPExportError(ValueError):
    """Raised when prediction data cannot be represented as a BOP19 result."""


@dataclass(frozen=True)
class BOPResult:
    """One validated BOP19 pose result with translation in millimetres."""

    scene_id: int
    im_id: int
    obj_id: int
    object_name: str
    score: float
    rotation: tuple[float, ...]
    translation_mm: tuple[float, ...]
    time: float


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BOPExportError(f"{field} must be an integer")
    if value < 0:
        raise BOPExportError(f"{field} must be non-negative")
    return value


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BOPExportError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise BOPExportError(f"{field} must be finite")
    return result


def _number_sequence(value: Any, field: str, length: int) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise BOPExportError(f"{field} must be an array with {length} entries")
    if len(value) != length:
        raise BOPExportError(
            f"{field} must contain exactly {length} entries, got {len(value)}"
        )
    return tuple(
        _finite_number(item, f"{field}[{index}]")
        for index, item in enumerate(value)
    )


def _object_identity(record: Mapping[str, Any]) -> tuple[int, str]:
    raw_obj_id = record.get("obj_id")
    raw_name = record.get("object_name")
    if raw_obj_id is None and raw_name is None:
        raise BOPExportError("one of obj_id or object_name is required")

    obj_id = None if raw_obj_id is None else _integer(raw_obj_id, "obj_id")
    if raw_name is not None:
        if not isinstance(raw_name, str) or raw_name not in OBJECT_NAME_TO_ID:
            raise BOPExportError(
                "object_name must be one of "
                f"{', '.join(sorted(OBJECT_NAME_TO_ID))}"
            )
        mapped_id = OBJECT_NAME_TO_ID[raw_name]
        if obj_id is not None and obj_id != mapped_id:
            raise BOPExportError(
                f"obj_id {obj_id} does not match object_name {raw_name!r} "
                f"(expected {mapped_id})"
            )
        return mapped_id, raw_name

    assert obj_id is not None
    if obj_id not in OBJECT_ID_TO_NAME:
        raise BOPExportError(
            f"obj_id must be one of {', '.join(map(str, sorted(OBJECT_ID_TO_NAME)))}"
        )
    return obj_id, OBJECT_ID_TO_NAME[obj_id]


def prediction_to_bop_result(record: Mapping[str, Any]) -> BOPResult:
    """Validate one structured prediction and convert it to a BOP result."""

    if not isinstance(record, Mapping):
        raise BOPExportError("prediction must be a JSON object")
    record_type = record.get("record_type")
    if record_type is not None and record_type != "prediction":
        raise BOPExportError(
            f"record_type must be 'prediction', got {record_type!r}"
        )

    pose = record.get("pose")
    if not isinstance(pose, Mapping):
        raise BOPExportError("pose must be an object containing R and t")
    obj_id, object_name = _object_identity(record)
    return BOPResult(
        scene_id=_integer(record.get("scene_id"), "scene_id"),
        im_id=_integer(record.get("im_id"), "im_id"),
        obj_id=obj_id,
        object_name=object_name,
        score=_finite_number(record.get("score"), "score"),
        rotation=_number_sequence(pose.get("R"), "pose.R", 9),
        translation_mm=_number_sequence(pose.get("t"), "pose.t", 3),
        time=_finite_number(record.get("time"), "time"),
    )


def read_prediction_jsonl(path: str | Path) -> list[BOPResult]:
    """Read and validate structured prediction records from a JSONL file."""

    source = Path(path)
    results = []
    try:
        with source.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise BOPExportError(
                        f"invalid JSON at {source}:{line_number}: {exc.msg}"
                    ) from exc
                try:
                    results.append(prediction_to_bop_result(record))
                except BOPExportError as exc:
                    raise BOPExportError(
                        f"invalid prediction at {source}:{line_number}: {exc}"
                    ) from exc
    except OSError as exc:
        raise BOPExportError(f"cannot read prediction JSONL {source}: {exc}") from exc
    return results


def _validated_results(
    predictions: Iterable[BOPResult | Mapping[str, Any]],
) -> list[BOPResult]:
    results = []
    image_times: dict[tuple[int, int], float] = {}
    for index, prediction in enumerate(predictions):
        try:
            if isinstance(prediction, BOPResult):
                prediction = {
                    "scene_id": prediction.scene_id,
                    "im_id": prediction.im_id,
                    "obj_id": prediction.obj_id,
                    "object_name": prediction.object_name,
                    "score": prediction.score,
                    "pose": {
                        "R": prediction.rotation,
                        "t": prediction.translation_mm,
                    },
                    "time": prediction.time,
                }
            result = prediction_to_bop_result(prediction)
        except BOPExportError as exc:
            raise BOPExportError(f"invalid prediction at index {index}: {exc}") from exc

        image_key = (result.scene_id, result.im_id)
        previous_time = image_times.setdefault(image_key, result.time)
        if abs(result.time - previous_time) > IMAGE_TIME_TOLERANCE_SEC:
            raise BOPExportError(
                f"time must agree within {IMAGE_TIME_TOLERANCE_SEC} seconds for "
                f"scene_id={result.scene_id}, "
                f"im_id={result.im_id}: got {previous_time!r} and {result.time!r}"
            )
        results.append(replace(result, time=previous_time))
    return results


def _format_number(value: float) -> str:
    return format(value, ".17g")


def write_bop19_csv(
    predictions: Iterable[BOPResult | Mapping[str, Any]], output_path: str | Path
) -> None:
    """Validate and write predictions in the seven-column BOP19 CSV format.

    Results are ordered by ascending scene, image, and object IDs, then by
    descending score. Python's stable sort preserves input order for equal keys.
    """

    results = _validated_results(predictions)
    results.sort(
        key=lambda item: (item.scene_id, item.im_id, item.obj_id, -item.score)
    )

    destination = Path(output_path)
    try:
        with destination.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=BOP19_FIELDS)
            writer.writeheader()
            for result in results:
                writer.writerow(
                    {
                        "scene_id": result.scene_id,
                        "im_id": result.im_id,
                        "obj_id": result.obj_id,
                        "score": _format_number(result.score),
                        "R": " ".join(map(_format_number, result.rotation)),
                        "t": " ".join(map(_format_number, result.translation_mm)),
                        "time": _format_number(result.time),
                    }
                )
    except OSError as exc:
        raise BOPExportError(f"cannot write BOP19 CSV {destination}: {exc}") from exc


def export_prediction_jsonl(
    input_path: str | Path, output_path: str | Path
) -> None:
    """Convert a structured prediction JSONL file to a BOP19 CSV file."""

    write_bop19_csv(read_prediction_jsonl(input_path), output_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export structured prediction JSONL records as BOP19 CSV."
    )
    parser.add_argument("input_jsonl", type=Path, help="structured prediction JSONL")
    parser.add_argument("output_csv", type=Path, help="destination BOP19 CSV")
    args = parser.parse_args(argv)
    try:
        export_prediction_jsonl(args.input_jsonl, args.output_csv)
    except BOPExportError as exc:
        parser.error(str(exc))
    return 0


__all__ = [
    "BOP19_FIELDS",
    "BOPExportError",
    "BOPResult",
    "OBJECT_ID_TO_NAME",
    "OBJECT_NAME_TO_ID",
    "export_prediction_jsonl",
    "prediction_to_bop_result",
    "read_prediction_jsonl",
    "write_bop19_csv",
]


if __name__ == "__main__":
    raise SystemExit(main())
