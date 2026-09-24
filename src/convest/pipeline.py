from pathlib import Path
import fcntl
import hashlib
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from collections import deque
import multiprocessing
from importlib import import_module

import yaml

from convest.config import WORKSPACE, atomic_json, digest, validate_repo_id
from convest.selection import select_episodes
from convest.registry import get_source, get_target, get_writer


def inspect(config, report):
    bags = get_source(config["source_format"]).discover(config["source_root"])
    result = {"source_root": config["source_root"], "bags": bags,
              "total": len(bags), "eligible": sum(b["eligible"] for b in bags),
              "source_bytes": sum(b["bytes"] for b in bags),
              "eligible_seconds": sum((b["source_end_ns"] - b["source_start_ns"]) / 1e9 for b in bags if b["eligible"])}
    atomic_json(report, result)
    print(f"Found {result['total']} bags, {result['eligible']} eligible, "
          f"{result['eligible_seconds'] / 3600:.2f} validated hours -> {report}", flush=True)
    return result


def load_records(root):
    return [json.loads(p.read_text()) for p in sorted((root / "conversion/records").glob("*.json"))]


def conversion_fingerprint(config, contract, schemas, converter_version):
    return digest({"config": config, "contract": contract, "schemas": schemas,
                   "converter_version": converter_version})


def _config_without_locations(config):
    return {key: value for key, value in config.items() if key not in {"source_root", "output_root"}}


def _resolved_record(record, index, relocations):
    source, source_snapshot = record["source"], record["source_snapshot"]
    key = f"{index:06d}"
    for relocation in relocations:
        override = relocation.get("records", {}).get(key)
        if override:
            source, source_snapshot = override["source"], override["source_snapshot"]
    return source, source_snapshot


def _map_relocated_path(path, old_config, new_config):
    path = Path(path).resolve()
    for key in ("source_root", "output_root"):
        old_root = Path(old_config[key]).resolve()
        if path.is_relative_to(old_root):
            return Path(new_config[key]).resolve() / path.relative_to(old_root)
    return path


def _file_sha256(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def _verify_same_source_files(old_path, new_path):
    old_files = {p.name: p for p in Path(old_path).iterdir() if p.is_file()}
    new_files = {p.name: p for p in Path(new_path).iterdir() if p.is_file()}
    if old_files.keys() != new_files.keys():
        raise ValueError(f"Relocated source file list differs: {old_path} -> {new_path}")
    for name in sorted(old_files):
        old_file, new_file = old_files[name], new_files[name]
        if old_file.stat().st_size != new_file.stat().st_size:
            raise ValueError(f"Relocated source file size differs: {old_file} -> {new_file}")
        if _file_sha256(old_file) != _file_sha256(new_file):
            raise ValueError(f"Relocated source file content differs: {old_file} -> {new_file}")


def prepare_relocation(manifest, config, records, source_adapter):
    """Verify a location-only move and return an auditable manifest event."""
    old_config = manifest["config"]
    if _config_without_locations(old_config) != _config_without_locations(config):
        raise ValueError("Relocation permits only source_root and output_root changes")
    changes = {
        key: {"from": old_config[key], "to": config[key]}
        for key in ("source_root", "output_root") if old_config[key] != config[key]
    }
    if not changes:
        raise ValueError("Relocation requested but source_root and output_root are unchanged")
    relocations = manifest.get("relocations", [])
    if not isinstance(relocations, list):
        raise ValueError("Invalid relocation history in conversion manifest")

    overrides = {}
    resolved_paths = set()
    verification_counts = {"sha256": 0, "exact_snapshot": 0}
    for index, record in enumerate(records):
        current_source, current_snapshot = _resolved_record(record, index, relocations)
        current_path = Path(current_source).resolve()
        relocated_path = _map_relocated_path(current_path, old_config, config)
        if not relocated_path.is_dir():
            raise ValueError(f"Relocated source is missing: {relocated_path}")
        relocated_snapshot = source_adapter.snapshot(relocated_path)
        if current_path == relocated_path:
            if relocated_snapshot != current_snapshot:
                raise ValueError(f"Previously converted source changed: {current_path}")
            verification_counts["exact_snapshot"] += 1
        elif current_path.is_dir():
            if source_adapter.snapshot(current_path) != current_snapshot:
                raise ValueError(f"Previously converted source changed before relocation: {current_path}")
            print(f"Relocation check [{index + 1}/{len(records)}] {current_path.name}: SHA-256", flush=True)
            _verify_same_source_files(current_path, relocated_path)
            verification_counts["sha256"] += 1
        elif relocated_snapshot != current_snapshot:
            # A directory moved with the output tree is verifiable from the original
            # snapshot only when file names, sizes and mtimes were preserved exactly.
            raise ValueError(f"Cannot verify moved source without its original: {current_path}")
        else:
            print(f"Relocation check [{index + 1}/{len(records)}] {current_path.name}: exact snapshot", flush=True)
            verification_counts["exact_snapshot"] += 1
        resolved = str(relocated_path)
        if resolved in resolved_paths:
            raise ValueError(f"Relocation maps multiple records to one source: {resolved}")
        resolved_paths.add(resolved)
        overrides[f"{index:06d}"] = {
            "source": resolved,
            "source_snapshot": relocated_snapshot,
        }
    return {
        "timestamp_ns": time.time_ns(),
        "changes": changes,
        "verification": "exact snapshot or SHA-256 comparison against the previous source",
        "verification_counts": verification_counts,
        "records": overrides,
    }


def recover(root, records, target_format="lerobot_v21"):
    # Only operate inside an already identified convest dataset, under our own file patterns.
    referenced = {p for record in records for ep in record["episodes"] for p in ep["paths"]}
    for pattern in get_target(target_format).file_patterns:
        for path in root.glob(pattern):
            if str(path.relative_to(root)) not in referenced:
                path.unlink()
    stage = root / "conversion/staging"
    if stage.exists():
        shutil.rmtree(stage)


def prepare_bag(item, config, contract, stage):
    source = get_source(config["source_format"])
    recipe = import_module(get_target(config["target_format"]).recipe)
    writer = get_writer(config["target_format"])
    before = source.snapshot(item["path"])
    started = time.monotonic()
    name = Path(item["path"]).name
    print(f"{name}: indexing source headers and joint streams", flush=True)
    if shutil.disk_usage(stage.parent).free < config["min_free_gb"] * 1e9:
        raise OSError("Available disk space below configured reserve")
    stage.mkdir()
    episodes, global_index = [], 0
    with source.Bag(item["path"], config["schema_dir"]) as bag:
        streams, aligned = recipe.prepare(source, bag, item, config, contract)
        logical_segments = (recipe.output_segments(item, aligned, config)
                            if hasattr(recipe, "output_segments") else
                            [{"bounds": bounds, "task": config["task"]}
                             for bounds in aligned.segments])
        print(f"{name}: {aligned.report['retained_frames']} frames, {len(logical_segments)} output episode(s); encoding RGB", flush=True)
        for segment, spec in enumerate(logical_segments):
            if shutil.disk_usage(stage).free < config["min_free_gb"] * 1e9:
                raise OSError("Available disk space below configured reserve")
            result = writer.write_segment(stage, bag, streams, aligned, spec["bounds"], segment,
                                          global_index, config, contract, spec)
            episodes.append(result)
            global_index += result["length"]
    if source.snapshot(item["path"]) != before:
        raise ValueError("Source changed during conversion; recording may still be active")
    return {"source": item["path"], "source_snapshot": before, "alignment": aligned.report,
            "episodes": episodes, "elapsed_seconds": round(time.monotonic() - started, 3)}


def prepared_in_order(items, config, contract, staging, workers):
    """Bound staging storage to worker count, preserving deterministic source order."""
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        pending = deque()
        iterator = iter(items)

        def submit():
            item = next(iterator, None)
            if item is None:
                return
            stage = staging / digest(item["path"])[:20]
            pending.append((item, stage, pool.submit(prepare_bag, item, config, contract, stage)))

        for _ in range(workers):
            submit()
        while pending:
            item, stage, future = pending.popleft()
            try:
                record, error = future.result(), None
            except Exception as exc:
                record, error = None, exc
            yield item, stage, record, error
            submit()


def convert(config, resume=False, limit=None, episode=None, workers=1, *, episode_list=None,
            skip_ineligible=False, relocate=False):
    validate_repo_id(config["repo_id"])
    source_adapter = get_source(config["source_format"])
    target_spec = get_target(config["target_format"])
    recipe = import_module(target_spec.recipe)
    recipe.validate_config(config)
    writer = get_writer(config["target_format"])
    if episode_list and (episode or limit is not None):
        raise ValueError("--episode-list cannot be combined with --episode or --limit")
    if skip_ineligible and not episode_list:
        raise ValueError("--skip-ineligible requires --episode-list")
    if relocate and not resume:
        raise ValueError("--relocate requires --resume")
    root = Path(config["output_root"]).resolve()
    source = Path(config["source_root"]).resolve()
    if not root.is_relative_to(WORKSPACE) or root == WORKSPACE:
        raise ValueError("Output must be a child of this workspace; existing projects are read-only")
    if root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError("Source and output trees must not overlap")
    marker = root / "conversion/manifest.json"
    if root.exists() and (not root.is_dir() or (any(root.iterdir()) and (not resume or not marker.is_file()))):
        raise FileExistsError(f"Output exists: {root}; only an owned dataset can be reopened with --resume")
    candidates = source_adapter.discover(source)
    selection = None
    if episode_list:
        selected, selection = select_episodes(candidates, episode_list, skip_ineligible)
    else:
        selected = [bag for bag in candidates if bag["eligible"]]
    if episode:
        selected = [bag for bag in selected if Path(bag["path"]).name == episode]
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError("No eligible selected bags")
    contract = yaml.safe_load(Path(config["contract"]).read_text())
    recipe.validate_contract(contract, config)
    schemas = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(config["schema_dir"]).glob("*.msg"))}
    fingerprint = conversion_fingerprint(config, contract, schemas, target_spec.version)
    (root / "conversion").mkdir(parents=True, exist_ok=True)
    with (root / "conversion/lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if marker.exists():
            manifest = json.loads(marker.read_text())
            if not resume or manifest.get("owner") != "convest-data":
                raise FileExistsError("Only an owned dataset can be reopened with --resume")
            if manifest.get("selection_mode") == "episode_list" and not episode_list:
                raise ValueError("This dataset was created from a TXT list; --resume requires --episode-list to prevent accidental full conversion")
            relocation_needed = manifest["fingerprint"] != fingerprint
            if relocation_needed and not relocate:
                raise ValueError("Conversion config/schema differs from existing dataset; use a new output")
        else:
            if relocate:
                raise ValueError("--relocate requires an existing converted dataset")
            atomic_json(marker, {"owner": "convest-data", "fingerprint": fingerprint, "config": config,
                                 "selection_mode": "episode_list" if episode_list else "discovery"})
            manifest = json.loads(marker.read_text())
            relocation_needed = False
        records = load_records(root)
        if relocation_needed:
            old_fingerprint = conversion_fingerprint(manifest["config"], contract, schemas, target_spec.version)
            if manifest["fingerprint"] != old_fingerprint:
                raise ValueError("Existing config/schema/converter changed; location-only relocation refused")
            relocation_event = prepare_relocation(manifest, config, records, source_adapter)
            manifest = dict(manifest)
            manifest["config"] = config
            manifest["fingerprint"] = fingerprint
            manifest["relocations"] = [*manifest.get("relocations", []), relocation_event]
            atomic_json(marker, manifest)
            print(f"Relocated {len(records)} committed source record(s): "
                  f"{', '.join(relocation_event['changes'])}", flush=True)
        relocations = manifest.get("relocations", [])
        completed = set()
        for index, record in enumerate(records):
            resolved_source, resolved_snapshot = _resolved_record(record, index, relocations)
            if source_adapter.snapshot(resolved_source) != resolved_snapshot:
                raise ValueError(f"Previously converted source changed: {resolved_source}")
            completed.add(resolved_source)
            for ep in record["episodes"]:
                for path in ep["paths"]:
                    if not (root / path).is_file():
                        raise ValueError(f"Committed output file missing: {path}")
        recover(root, records, config["target_format"])
        writer.write_metadata(root, records, config)
        task_indices = {}
        for record in records:
            for ep in record["episodes"]:
                task = ep["tasks"][0]
                index = ep.get("task_index", 0)
                if task in task_indices and task_indices[task] != index:
                    raise ValueError("Inconsistent task index in committed records")
                if index in task_indices.values() and task not in task_indices:
                    raise ValueError("Committed task index refers to multiple task strings")
                task_indices[task] = index
        previous_count = len(records)
        errors = []
        atomic_json(root / "conversion/discovery.json", candidates)
        remaining = [item for item in selected if item["path"] not in completed]
        retained = sorted(completed - {item["path"] for item in selected})
        if selection is not None:
            selection["already_converted"] = [Path(item["path"]).name for item in selected if item["path"] in completed]
            selection["to_convert"] = [Path(item["path"]).name for item in remaining]
            selection["retained_bags_not_in_current_selection"] = retained
            atomic_json(root / "conversion/selection.json", selection)
            atomic_json(root / f"conversion/selections/{time.time_ns()}.json", selection)
            for skipped in selection["skipped"]:
                print(f"SKIPPED {skipped['episode']}: {skipped['reason']}", flush=True)
            if retained:
                print(f"Keeping {len(retained)} previously converted bag(s) absent from the current list; resume never deletes committed data", flush=True)
        staging = root / "conversion/staging"
        staging.mkdir()
        print(f"{len(remaining)} remaining bags; {workers} worker(s)", flush=True)
        for item, stage, record, error in prepared_in_order(remaining, config, contract, staging, workers):
            try:
                if error is not None:
                    raise error
                episodes = record["episodes"]
                first_index = sum(len(r["episodes"]) for r in records)
                global_index = sum(ep["length"] for r in records for ep in r["episodes"])
                reference = records[0]["episodes"][0]["features"] if records else episodes[0]["features"]
                for segment, result in enumerate(episodes):
                    if result["features"] != reference:
                        raise ValueError("Camera/feature shape differs from existing dataset")
                    task = result["tasks"][0]
                    task_index = task_indices.setdefault(task, len(task_indices))
                    writer.reindex_segment(stage, result, first_index + segment, global_index, task_index)
                    global_index += result["length"]
                # Commit journal is authoritative. Interrupted file moves are recovered on resume.
                for ep in episodes:
                    for relative in ep["paths"]:
                        target = root / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        (stage / "final" / relative).replace(target)
                atomic_json(root / f"conversion/records/{len(records):06d}.json", record)
                records.append(record)
                shutil.rmtree(stage)
                writer.write_metadata(root, records, config)
                print(f"[{len(records)}/{len(selected)}] {Path(item['path']).name}: committed {len(episodes)} episode(s), {record['elapsed_seconds']:.1f}s", flush=True)
            except Exception as exc:
                errors.append({"source": item["path"], "error": f"{type(exc).__name__}: {exc}"})
                if stage.exists():
                    shutil.rmtree(stage)
                print(f"  FAILED: {errors[-1]['error']}", flush=True)
                atomic_json(root / "conversion/errors.json", errors)
                if isinstance(exc, OSError):
                    raise
        recover(root, records, config["target_format"])
        atomic_json(root / "conversion/errors.json", errors)
        summary = {"converted_bags": len(records), "episodes": sum(len(r["episodes"]) for r in records),
                   "frames": sum(ep["length"] for r in records for ep in r["episodes"]),
                   "repo_id": config["repo_id"], "added_bags": len(records) - previous_count,
                   "selected_bags": len(selected), "skipped_selected_bags": selection["skipped"] if selection else [],
                   "retained_bags_not_in_current_selection": retained,
                   "skipped_by_collection_state": (len(selection["skipped"]) if selection else sum(not c["eligible"] for c in candidates)),
                   "errors": errors}
        atomic_json(root / "conversion/summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return 1 if errors else 0
