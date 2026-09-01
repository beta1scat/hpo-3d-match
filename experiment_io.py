"""Experiment manifests and structured record logging.

The record writers accept arbitrary mappings.  The field constants below are
recommended schemas, not validation rules.  CSV headers are inferred from the
first record; later records may omit fields but may not introduce new ones.
CSV writing is not concurrency-safe and must not be used for concurrent trial
logs. Concurrent trial logging must use JSONL; no cross-process lock is added.
"""

from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import enum
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import secrets
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


MANIFEST_SCHEMA_VERSION = 1

COMMON_RECORD_FIELDS = (
    "run_id",
    "timestamp_utc",
    "record_type",
)
TRIAL_RECORD_FIELDS = (
    "timestamp_utc",
    "record_type",
    "study_name",
    "trial_number",
    "state",
    "model_name",
    "objective_version",
    "sampler_name",
    "pruner_name",
    "budget",
    "repeat",
    "seed",
    "direction",
    "resumed_from_trials",
    "params",
    "objective_value",
    "duration_sec",
    "datetime_start",
    "datetime_complete",
    "intermediate_values",
    "user_attrs",
    "system_attrs",
)
PREDICTION_RECORD_FIELDS = COMMON_RECORD_FIELDS + (
    "scene_id",
    "im_id",
    "obj_id",
    "object_name",
    "score",
    "pose",
    "time",
)
ASSOCIATION_RECORD_FIELDS = COMMON_RECORD_FIELDS + (
    "trial_number",
    "scene_id",
    "object_name",
    "prediction_id",
    "ground_truth_id",
    "matched",
    "position_error_mm",
    "rotation_error_deg",
    "cost",
)


def utc_now() -> str:
    """Return the current UTC time in an ISO 8601 representation."""
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def generate_run_id(prefix: str = "run", now: dt.datetime | None = None) -> str:
    """Generate a sortable run identifier with a random collision suffix."""
    current = now or dt.datetime.now(dt.timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=dt.timezone.utc)
    current = current.astimezone(dt.timezone.utc)
    timestamp = current.strftime("%Y%m%dT%H%M%S%fZ")
    suffix = secrets.token_hex(4)
    return f"{prefix}-{timestamp}-{suffix}" if prefix else f"{timestamp}-{suffix}"


def to_jsonable(value: Any) -> Any:
    """Recursively convert common scientific Python values to strict JSON data."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        result = value.isoformat()
        if isinstance(value, dt.datetime) and value.tzinfo == dt.timezone.utc:
            result = result.replace("+00:00", "Z")
        return result
    if isinstance(value, enum.Enum):
        return to_jsonable(value.value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        converted = [to_jsonable(item) for item in value]
        return sorted(converted, key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, bytes):
        return value.hex()

    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            item = item_method()
        except (TypeError, ValueError):
            pass
        else:
            if item is not value:
                return to_jsonable(item)

    tolist_method = getattr(value, "tolist", None)
    if callable(tolist_method):
        return to_jsonable(tolist_method())

    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    """Calculate a file SHA-256 digest without loading the file into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def collect_input_files(
    input_files: Mapping[str, str | os.PathLike[str]]
    | Iterable[str | os.PathLike[str]],
) -> list[dict[str, Any]]:
    """Collect resolved paths, sizes, and SHA-256 digests for input files."""
    if isinstance(input_files, Mapping):
        named_paths = input_files.items()
    else:
        named_paths = ((Path(path).name, path) for path in input_files)

    result = []
    for name, raw_path in named_paths:
        path = Path(raw_path).expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"Input path is not a file: {path}")
        result.append(
            {
                "name": str(name),
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return result


def collect_package_versions(
    package_names: Iterable[str] | None = None,
) -> dict[str, str | None]:
    """Collect installed versions for all distributions or selected packages."""
    if package_names is None:
        versions = {}
        for distribution in importlib.metadata.distributions():
            name = distribution.metadata.get("Name")
            if name:
                versions[name] = distribution.version
        return dict(sorted(versions.items(), key=lambda item: item[0].lower()))

    versions = {}
    for name in package_names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def collect_runtime_info(
    package_names: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Collect Python, operating-system, hardware, and package information."""
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "description": platform.platform(),
        },
        "packages": collect_package_versions(package_names),
    }


def _run_git(repo_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo_path), *args],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
    )


def collect_git_info(
    repo_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Collect the Git commit and whole-worktree dirty state without raising."""
    path = Path(repo_path or Path(__file__).resolve().parent).resolve()
    try:
        commit_result = _run_git(path, "rev-parse", "HEAD")
        root_result = _run_git(path, "rev-parse", "--show-toplevel")
        status_result = _run_git(path, "status", "--porcelain", "--untracked-files=normal")
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "available": False,
            "commit": None,
            "dirty": None,
            "root": None,
            "error": str(exc),
        }

    results = (commit_result, root_result, status_result)
    if any(result.returncode != 0 for result in results):
        error = next(
            (result.stderr.strip() for result in results if result.returncode != 0),
            "Unable to inspect Git repository",
        )
        return {
            "available": False,
            "commit": None,
            "dirty": None,
            "root": None,
            "error": error,
        }

    return {
        "available": True,
        "commit": commit_result.stdout.strip(),
        "dirty": bool(status_result.stdout),
        "root": root_result.stdout.strip(),
        "error": None,
    }


def collect_manifest(
    *,
    run_id: str | None = None,
    cli_config: Any = None,
    search_space: Any = None,
    roi: Any = None,
    object_mapping: Any = None,
    input_files: Mapping[str, str | os.PathLike[str]]
    | Iterable[str | os.PathLike[str]] = (),
    package_names: Iterable[str] | None = None,
    repo_path: str | os.PathLike[str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a self-contained experiment manifest.

    ``cli_config`` may be an argparse Namespace, mapping, or dataclass.  If an
    arbitrary object with ``__dict__`` is supplied, its public attributes are
    recorded.  Input file errors are intentionally propagated because an
    incomplete input inventory is not reproducible.
    """
    if cli_config is not None and not isinstance(cli_config, Mapping):
        if not dataclasses.is_dataclass(cli_config) and hasattr(cli_config, "__dict__"):
            cli_config = vars(cli_config)

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id or generate_run_id(),
        "created_at_utc": utc_now(),
        "runtime": collect_runtime_info(package_names),
        "git": collect_git_info(repo_path),
        "cli_config": cli_config,
        "search_space": search_space,
        "roi": roi,
        "object_mapping": object_mapping,
        "input_files": collect_input_files(input_files),
    }
    if extra:
        overlap = set(manifest).intersection(extra)
        if overlap:
            raise ValueError(f"extra cannot replace manifest fields: {sorted(overlap)}")
        manifest.update(extra)
    return to_jsonable(manifest)


def atomic_write_json(
    path: str | os.PathLike[str],
    data: Any,
    *,
    indent: int | None = 2,
) -> None:
    """Atomically replace a JSON file with strict, UTF-8 encoded JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(
                to_jsonable(data),
                temporary,
                ensure_ascii=True,
                allow_nan=False,
                indent=indent,
                sort_keys=True,
            )
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _prepare_record(record: Mapping[str, Any], record_type: str | None) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")
    prepared = dict(record)
    if record_type is not None:
        existing_type = prepared.setdefault("record_type", record_type)
        if existing_type != record_type:
            raise ValueError(
                f"record_type must be {record_type!r}, got {existing_type!r}"
            )
    prepared.setdefault("timestamp_utc", utc_now())
    return to_jsonable(prepared)


def append_jsonl(
    path: str | os.PathLike[str],
    record: Mapping[str, Any],
    *,
    record_type: str | None = None,
) -> None:
    """Append one strict JSON record using a single append-mode OS write."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    prepared = _prepare_record(record, record_type)
    line = json.dumps(
        prepared,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    descriptor = os.open(destination, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        written = os.write(descriptor, line)
        if written != len(line):
            raise OSError(f"Short JSONL write: {written} of {len(line)} bytes")
    finally:
        os.close(descriptor)


def _csv_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return value


def append_csv(
    path: str | os.PathLike[str],
    record: Mapping[str, Any],
    *,
    record_type: str | None = None,
) -> None:
    """Append a non-concurrent record, inferring the CSV header when empty.

    CSV output is not concurrency-safe. Concurrent trial logs must use JSONL.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    prepared = _prepare_record(record, record_type)

    file_exists = destination.exists() and destination.stat().st_size > 0
    if file_exists:
        with destination.open("r", encoding="utf-8-sig", newline="") as source:
            fieldnames = next(csv.reader(source), None)
        if not fieldnames:
            file_exists = False
            fieldnames = list(prepared)
        else:
            unknown_fields = set(prepared).difference(fieldnames)
            if unknown_fields:
                raise ValueError(
                    "CSV record contains fields absent from the existing header: "
                    f"{sorted(unknown_fields)}"
                )
    else:
        fieldnames = list(prepared)

    with destination.open("a", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="raise")
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: _csv_cell(value) for key, value in prepared.items()})


def append_record(
    path: str | os.PathLike[str],
    record: Mapping[str, Any],
    *,
    record_type: str | None = None,
    format: str | None = None,
) -> None:
    """Append a CSV or JSONL record, inferring format from the file suffix."""
    selected_format = (format or Path(path).suffix.lstrip(".")).lower()
    if selected_format == "csv":
        append_csv(path, record, record_type=record_type)
    elif selected_format in {"jsonl", "ndjson"}:
        append_jsonl(path, record, record_type=record_type)
    else:
        raise ValueError("format must be 'csv', 'jsonl', or 'ndjson'")


def append_trial_record(
    path: str | os.PathLike[str], record: Mapping[str, Any], *, format: str | None = None
) -> None:
    """Append a trial record; concurrent trial logs must use JSONL."""
    append_record(path, record, record_type="trial", format=format)


def append_prediction_record(
    path: str | os.PathLike[str], record: Mapping[str, Any], *, format: str | None = None
) -> None:
    """Append a prediction with rotation ``pose.R`` and millimetre ``pose.t``."""
    append_record(path, record, record_type="prediction", format=format)


def append_association_record(
    path: str | os.PathLike[str], record: Mapping[str, Any], *, format: str | None = None
) -> None:
    """Append an association record with ``record_type='association'``."""
    append_record(path, record, record_type="association", format=format)


__all__ = [
    "ASSOCIATION_RECORD_FIELDS",
    "COMMON_RECORD_FIELDS",
    "MANIFEST_SCHEMA_VERSION",
    "PREDICTION_RECORD_FIELDS",
    "TRIAL_RECORD_FIELDS",
    "append_association_record",
    "append_csv",
    "append_jsonl",
    "append_prediction_record",
    "append_record",
    "append_trial_record",
    "atomic_write_json",
    "collect_git_info",
    "collect_input_files",
    "collect_manifest",
    "collect_package_versions",
    "collect_runtime_info",
    "generate_run_id",
    "sha256_file",
    "to_jsonable",
    "utc_now",
]
