"""Report whether this checkout can reproduce P0/P1 training without bypassing provenance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from src.config import PROJECT_ROOT


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON mapping: {path}")
    return value


def inspect_training_readiness(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    root = root.resolve()
    split_path = root / "data" / "splits" / "outer_0.json"
    preprocessing_path = root / "configs" / "preprocessing.yaml"
    blockers: list[str] = []
    if not split_path.is_file():
        blockers.append("missing data/splits/outer_0.json")
        required_sources: list[dict[str, Any]] = []
    else:
        required_sources = list(_read_json(split_path).get("sources") or [])
    if not preprocessing_path.is_file():
        blockers.append("missing configs/preprocessing.yaml")
        source_root = root / "data" / "raw" / "uta_rldd"
    else:
        preprocessing = yaml.safe_load(preprocessing_path.read_text(encoding="utf-8")) or {}
        source_root = _resolve(root, preprocessing.get("dataset_root", "data/raw/uta_rldd"))

    missing_sources = []
    present_sources = []
    for source in required_sources:
        source_id = str(source.get("video_id", ""))
        relative = source.get("relative_path")
        path = _resolve(source_root, relative) if isinstance(relative, str) else source_root / "__missing__"
        (present_sources if path.is_file() else missing_sources).append(source_id)
    if missing_sources:
        blockers.append(f"missing {len(missing_sources)} of {len(required_sources)} frozen raw videos")

    raw_features = root / "data" / "processed" / "raw_features"
    missing_feature_pairs = [
        str(source.get("video_id", ""))
        for source in required_sources
        if not (raw_features / f"{source.get('video_id')}.parquet").is_file()
        or not (raw_features / f"{source.get('video_id')}.metadata.json").is_file()
    ]
    if missing_feature_pairs:
        blockers.append(f"missing {len(missing_feature_pairs)} frozen raw-feature pairs")

    snapshot = root / "runs" / "phase8" / "snapshot" / "snapshot.json"
    p0_report = root / "runs" / "phase11" / "derived" / "P0.json"
    p1_report = root / "runs" / "phase11" / "derived" / "P1.json"
    if not snapshot.is_file():
        blockers.append("missing runs/phase8/snapshot/snapshot.json")
    if not p0_report.is_file():
        blockers.append("missing runs/phase11/derived/P0.json")

    manifests = []
    processed = root / "data" / "processed"
    for path in sorted(processed.glob("derived_*/dataset_manifest.json")):
        try:
            manifest = _read_json(path)
            referenced = [manifest.get("split_path"), manifest.get("snapshot_path"), manifest.get("raw_dir")]
            portable = all(value is not None and _resolve(root, value).exists() for value in referenced)
            manifests.append({
                "path": str(path),
                "mode": manifest.get("mode"),
                "outer_index": manifest.get("outer_index"),
                "portable_references": portable,
            })
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            manifests.append({"path": str(path), "error": str(exc), "portable_references": False})
    if manifests and not any(item.get("portable_references") for item in manifests):
        blockers.append("all derived manifests reference missing/non-portable frozen paths")
    elif not manifests:
        blockers.append("no derived dataset manifests found")

    checkpoints = sorted(str(path) for path in (root / "models" / "checkpoints").glob("*/integrity.json"))
    return {
        "status": "ready" if not blockers else "blocked",
        "project_root": str(root),
        "required_raw_videos": len(required_sources),
        "present_raw_videos": len(present_sources),
        "missing_raw_video_ids": sorted(missing_sources),
        "missing_raw_feature_pair_ids": sorted(missing_feature_pairs),
        "phase8_snapshot": str(snapshot) if snapshot.is_file() else None,
        "p0_derived_report": str(p0_report) if p0_report.is_file() else None,
        "p1_derived_report": str(p1_report) if p1_report.is_file() else None,
        "derived_manifests": manifests,
        "installed_checkpoints": checkpoints,
        "blockers": blockers,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    args = parser.parse_args(argv)
    report = inspect_training_readiness(args.project_root)
    encoded = json.dumps(report, indent=2, ensure_ascii=False)
    print(encoded)
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    return 0 if report["status"] == "ready" else 3


if __name__ == "__main__":
    raise SystemExit(main())
