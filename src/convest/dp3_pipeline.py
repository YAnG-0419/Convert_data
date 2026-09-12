"""Transactional ROS bag to uncropped DP3 Zarr conversion, grouped by quality TXT."""
from pathlib import Path
import fcntl
import hashlib
import json
import shutil
import subprocess
import time

import yaml

from convest.config import WORKSPACE, atomic_json, digest
from convest.formats import dp3_zarr
from convest.recipes import dp3
from convest.registry import get_source
from convest.selection import select_episodes


OWNER = "convest-dp3"
VERSION = "dp3-uncropped-1"


def git_revision():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=WORKSPACE, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def source_sha256(path):
    checksums = {}
    for file in sorted(Path(path).iterdir()):
        if not file.is_file():
            continue
        checksum = hashlib.sha256()
        with file.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                checksum.update(block)
        checksums[file.name] = checksum.hexdigest()
    return checksums


def load_records(root):
    return [json.loads(path.read_text())
            for path in sorted((Path(root) / "conversion/records").glob("*.json"))]


def quality_list(config, quality):
    groups = config.get("quality_groups", {})
    if quality not in groups:
        raise ValueError(f"Unknown quality {quality!r}; available: {', '.join(groups)}")
    path = Path(groups[quality]).expanduser()
    return (WORKSPACE / path).resolve() if not path.is_absolute() else path.resolve()


def quality_root(config, quality):
    return Path(config["output_root"]).resolve() / quality


def _fingerprint(config, contract, quality):
    schemas = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(config["schema_dir"]).glob("*.msg"))
    }
    return digest({
        "config": config, "contract": contract, "schemas": schemas,
        "quality": quality, "converter_version": VERSION,
    })


def _validate_paths(root, source):
    root, source = root.resolve(), source.resolve()
    if not root.is_relative_to(WORKSPACE) or root == WORKSPACE:
        raise ValueError("Output must be a child of this workspace")
    if root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError("Source and output trees must not overlap")


def _validate_committed(source, records):
    for record in records:
        if source.snapshot(record["source"]) != record["source_snapshot"]:
            raise ValueError(f"Previously converted source changed: {record['source']}")


def convert_quality(config, quality, *, resume=False, skip_ineligible=False, limit=None):
    """Convert one quality list into one append-only Zarr dataset."""
    dp3.validate_config(config)
    source = get_source(config["source_format"])
    source_root = Path(config["source_root"]).resolve()
    root = quality_root(config, quality)
    _validate_paths(root, source_root)
    contract = yaml.safe_load(Path(config["contract"]).read_text())
    dp3.validate_contract(contract, config)
    list_path = quality_list(config, quality)
    candidates = source.discover(source_root)
    selected, selection = select_episodes(candidates, list_path, skip_ineligible)
    if limit is not None:
        selected = selected[:limit]
        selection["selected_episodes"] = [Path(item["path"]).name for item in selected]
        selection["limit"] = limit
    if not selected:
        raise ValueError(f"No eligible episodes selected for {quality}")

    marker = root / "conversion/manifest.json"
    fingerprint = _fingerprint(config, contract, quality)
    if root.exists() and (not root.is_dir() or (any(root.iterdir()) and (not resume or not marker.is_file()))):
        raise FileExistsError(f"Output exists: {root}; reopen an owned dataset with --resume")
    (root / "conversion").mkdir(parents=True, exist_ok=True)

    with (root / "conversion/lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if marker.exists():
            manifest = json.loads(marker.read_text())
            if not resume or manifest.get("owner") != OWNER:
                raise FileExistsError("Only an owned DP3 dataset can be reopened with --resume")
            if manifest.get("fingerprint") != fingerprint:
                raise ValueError("Conversion config/schema differs from the existing dataset; use a new output")
        else:
            atomic_json(marker, {
                "owner": OWNER, "fingerprint": fingerprint, "converter_version": VERSION,
                "converter_git_commit": git_revision(),
                "target_format": "dp3_uncropped_zarr", "quality": quality,
                "config": config, "quality_list": str(list_path),
            })

        records = load_records(root)
        _validate_committed(source, records)
        store = dp3_zarr.recover(root, records, config)
        completed = {record["source"] for record in records}
        remaining = [item for item in selected if item["path"] not in completed]
        selection.update(
            already_converted=[Path(item["path"]).name for item in selected if item["path"] in completed],
            to_convert=[Path(item["path"]).name for item in remaining],
            retained_bags_not_in_current_selection=sorted(
                completed - {item["path"] for item in selected}
            ),
        )
        atomic_json(root / "conversion/discovery.json", candidates)
        atomic_json(root / "conversion/selection.json", selection)
        atomic_json(root / f"conversion/selections/{time.time_ns()}.json", selection)
        errors = []
        previous_count = len(records)
        print(f"[{quality}] {len(remaining)} bag(s) remaining -> {root}", flush=True)

        for item in remaining:
            name = Path(item["path"]).name
            started = time.monotonic()
            try:
                if shutil.disk_usage(root).free < config["min_free_gb"] * 1e9:
                    raise OSError("Available disk space below configured reserve")
                before = source.snapshot(item["path"])
                episodes = []
                with source.Bag(item["path"], config["schema_dir"]) as bag:
                    streams, aligned = dp3.prepare(source, bag, item, config, contract)
                    for segment, bounds in enumerate(aligned.segments):
                        result = dp3_zarr.append_segment(
                            store, bag, streams, aligned, bounds, config, contract, item["path"]
                        )
                        result.update(
                            episode_index=sum(len(record["episodes"]) for record in records) + len(episodes),
                            segment_index=segment,
                            source_recording_id=item["source_recording_id"],
                        )
                        episodes.append(result)
                checksums = source_sha256(item["path"])
                if source.snapshot(item["path"]) != before:
                    raise ValueError("Source changed during conversion; recording may still be active")
                record = {
                    "source": item["path"], "source_snapshot": before,
                    "source_sha256": checksums,
                    "source_recording_id": item["source_recording_id"],
                    "alignment": aligned.report, "episodes": episodes,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                }
                atomic_json(root / f"conversion/records/{len(records):06d}.json", record)
                records.append(record)
                print(
                    f"[{quality} {len(records)}/{len(selected)}] {name}: "
                    f"committed {len(episodes)} segment(s)", flush=True,
                )
            except Exception as exc:
                store = dp3_zarr.recover(root, records, config)
                errors.append({"source": item["path"], "error": f"{type(exc).__name__}: {exc}"})
                atomic_json(root / "conversion/errors.json", errors)
                print(f"[{quality}] FAILED {name}: {errors[-1]['error']}", flush=True)
                if isinstance(exc, OSError):
                    raise

        complete = not errors
        dp3_zarr.finalize(root, records, config, contract, quality, complete=complete)
        atomic_json(root / "conversion/errors.json", errors)
        summary = {
            "quality": quality, "selected_bags": len(selected), "converted_bags": len(records),
            "added_bags": len(records) - previous_count,
            "episodes": sum(len(record["episodes"]) for record in records),
            "frames": sum(ep["length"] for record in records for ep in record["episodes"]),
            "skipped": selection["skipped"], "errors": errors,
            "ready_for_cropping": complete,
        }
        atomic_json(root / "conversion/summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return 1 if errors else 0
