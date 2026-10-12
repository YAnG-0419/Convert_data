"""Preserve the original Pi05 causal alignment and source contract."""
import numpy as np

from convest.align import align


def validate_config(config):
    if config["window"] != "validated":
        raise ValueError("Only the validated source-time window is supported")
    if config["max_staleness_ms"] <= 0:
        raise ValueError("Invalid freshness limit")
    gap_frames = config.get("camera_short_gap_max_frames", 0)
    gap_age = config.get("camera_short_gap_max_age_ms", 250)
    reject_frames = config.get("camera_gap_reject_min_frames", 0)
    if type(reject_frames) is not int or reject_frames not in (0, 3):
        raise ValueError("camera_gap_reject_min_frames must be 0 (off) or 3")
    if type(gap_frames) is not int or gap_frames not in (0, 1, 2):
        raise ValueError("camera_short_gap_max_frames must be 0, 1, or 2")
    if (not isinstance(gap_age, (int, float)) or not np.isfinite(gap_age) or gap_age <= 0
            or (gap_frames and gap_age < config["max_staleness_ms"])):
        raise ValueError("Invalid camera_short_gap_max_age_ms")
    if not 0 <= config["video_crf"] <= 51 or config["video_threads"] < 1:
        raise ValueError("Invalid encoder settings")
    if config.get("segments", "all") not in {"all", "full", "milestones", "first-milestone-or-full", "subtasks", "subtasks-only", "both"}:
        raise ValueError("segments must be all, full, milestones, first-milestone-or-full, subtasks, subtasks-only, or both")
    tasks = config.get("segment_tasks", {})
    if not isinstance(tasks, dict) or any(not isinstance(key, str) or not key.strip()
                                           or not isinstance(value, str) or not value.strip()
                                           for key, value in tasks.items()):
        raise ValueError("segment_tasks must map nonempty segment IDs to nonempty task strings")
    subtask_tasks = config.get("subtask_tasks", {})
    if not isinstance(subtask_tasks, dict) or any(
        not isinstance(key, str) or not key.strip() or not isinstance(value, str)
        or not value.strip() or not value.isascii()
        for key, value in subtask_tasks.items()
    ):
        raise ValueError("subtask_tasks must map nonempty source labels to nonempty English task strings")
    if config.get("segments", "all") in {"subtasks", "subtasks-only", "both"} and not subtask_tasks:
        raise ValueError("subtasks export requires subtask_tasks translations")


def validate_contract(contract, config):
    if contract["features"]["observation.state"]["dimension"] != 108 or contract["features"]["action"]["dimension"] != 54:
        raise ValueError("Pi05 FR3/Wuji requires state=108, action=54 in the source contract order")


def assess_candidate(item, config):
    if config.get("segments", "all") not in {"subtasks", "subtasks-only", "both"} or not item["eligible"]:
        return item
    result = dict(item)
    subtasks = item.get("subtasks", [])
    if len(subtasks) != 4:
        result.update(eligible=False, reason=f"subtasks export requires exactly 4 phases; found {len(subtasks)}")
        return result
    missing = [part["task"] for part in subtasks if part["task"] not in config.get("subtask_tasks", {})]
    if missing:
        result.update(eligible=False, reason=f"missing English subtask translation: {missing}")
    return result


def prepare(source, bag, item, config, contract):
    streams = source.read_streams(bag, contract)
    max_age_ns = int(config["max_staleness_ms"] * 1e6)
    gap_frames = config.get("camera_short_gap_max_frames", 0)
    reject_frames = config.get("camera_gap_reject_min_frames", 0)
    # The new bounded policy must inspect real image timestamps. The legacy
    # compact-boundary helper inserts retimestamped copies and could hide age.
    compact_holds = {} if gap_frames or reject_frames else source.bridge_compact_image_boundaries(
        streams, bag.compact_time_map, item["source_start_ns"], item["source_end_ns"],
        config["fps"], max_age_ns,
    )
    aligned = align(streams, item["source_start_ns"], item["source_end_ns"], config["fps"],
                    max_age_ns, config["min_segment_frames"],
                    camera_short_gap_max_frames=gap_frames,
                    camera_short_gap_max_age_ns=int(config.get("camera_short_gap_max_age_ms", 250) * 1e6),
                    camera_gap_reject_min_frames=reject_frames)
    if compact_holds:
        aligned.report["compact_boundary_image_holds"] = compact_holds
    if item.get("source_time_adapter"):
        aligned.report["source_time_adapter"] = item["source_time_adapter"]
    aligned.report["status_timestamp_policy"] = {
        kind: source.timestamp_policy(kind, bag.compact_time_map)
        for kind in sorted(source.STATUS_MESSAGE_TYPES)
    }
    return streams, aligned


def _milestones(item):
    markers = item.get("milestones", [])
    if not isinstance(markers, list):
        raise ValueError("milestones must be a list")
    result, seen, previous = [], {"full"}, 0
    for marker in markers:
        if not isinstance(marker, dict):
            raise ValueError("each milestone must be an object")
        marker_id, timestamp_ns = marker.get("id"), marker.get("timestamp_ns")
        if not isinstance(marker_id, str) or not marker_id.strip() or marker_id in seen:
            raise ValueError("milestone IDs must be nonempty, unique, and not 'full'")
        if marker.get("clock") != "ros":
            raise ValueError(f"{marker_id}: milestone clock must be ros")
        if type(timestamp_ns) is not int or timestamp_ns <= previous:
            raise ValueError("milestone timestamps must be positive, increasing integer nanoseconds")
        result.append((marker_id, timestamp_ns))
        seen.add(marker_id)
        previous = timestamp_ns
    return result


def output_segments(item, aligned, config):
    """Return logical dataset episodes, including overlapping milestone prefixes."""
    mode = config.get("segments", "all")
    if mode == "both":
        raise ValueError("both requires separate dataset export through pipeline.convert")
    if mode in {"subtasks", "subtasks-only"}:
        return _subtask_segments(item, aligned, config, include_full=mode == "subtasks")
    if mode == "full" and item.get("subtasks"):
        return _subtask_segments(item, aligned, config, include_phases=False)
    markers = [] if mode == "full" else _milestones(item)
    tasks = config.get("segment_tasks", {})
    recording_id = item.get("source_recording_id") or item["path"]

    def description(segment_id):
        return tasks.get(segment_id, config["task"] if segment_id == "full" else segment_id)

    if mode == "milestones" and not markers:
        raise ValueError("milestones export requested but this bag has no milestones")
    selected_markers = (markers[:1] if mode == "first-milestone-or-full"
                        else [] if mode == "full" else markers)
    if selected_markers and mode != "first-milestone-or-full" and aligned.segments != [(0, len(aligned.timeline))]:
        raise ValueError("A bag with milestones must have one continuous valid aligned interval")

    result = []
    for marker_id, marker_ns in selected_markers:
        if not int(aligned.timeline[0]) < marker_ns < int(item["source_end_ns"]):
            raise ValueError(f"{marker_id}: milestone is outside the validated source-time interval")
        end = int(np.searchsorted(aligned.timeline, marker_ns, side="right"))
        if end < config["min_segment_frames"]:
            raise ValueError(f"{marker_id}: milestone prefix is shorter than min_segment_frames")
        start = 0
        if mode == "first-milestone-or-full":
            prefix_segments = [(a, min(b, end)) for a, b in aligned.segments if a < end and b > 0]
            if len(prefix_segments) != 1 or prefix_segments[0][1] != end:
                raise ValueError(f"{marker_id}: milestone prefix contains an internal invalid interval")
            start = prefix_segments[0][0]
        result.append({"bounds": (start, end), "segment_id": marker_id,
                       "milestone_timestamp_ns": marker_ns, "source_recording_id": recording_id,
                       "task": description(marker_id)})

    if mode in {"all", "full"} or (mode == "first-milestone-or-full" and not markers):
        many = len(aligned.segments) > 1
        for index, bounds in enumerate(aligned.segments, 1):
            segment_id = f"full_part_{index}" if many else "full"
            result.append({"bounds": bounds, "segment_id": segment_id,
                           "milestone_timestamp_ns": None, "source_recording_id": recording_id,
                           "task": description("full")})
    return result


def _subtask_segments(item, aligned, config, *, include_full=True, include_phases=True):
    """Create full-task and phase-task episodes from persistent-language annotations."""
    subtasks = item.get("subtasks", [])
    if include_phases and len(subtasks) != 4:
        raise ValueError(f"subtasks export requires exactly 4 phases; found {len(subtasks)}")
    recording_id = item.get("source_recording_id") or item["path"]
    translations = config.get("subtask_tasks", {})
    result = []

    full_start = int(np.searchsorted(aligned.timeline, subtasks[0]["start_ns"], side="left"))
    full_end = int(np.searchsorted(aligned.timeline, subtasks[-1]["end_ns"], side="left"))
    full_pieces = [(max(a, full_start), min(b, full_end)) for a, b in aligned.segments
                   if a < full_end and b > full_start]
    full_pieces = [(a, b) for a, b in full_pieces if b - a >= config["min_segment_frames"]]
    if not full_pieces:
        raise ValueError("Full task has no continuous valid aligned frames")
    for part, bounds in enumerate(full_pieces if include_full else [], 1):
        result.append({
            "bounds": bounds,
            "segment_id": f"full_part_{part}" if len(full_pieces) > 1 else "full",
            "source_recording_id": recording_id,
            "task": config["task"],
        })

    for subtask in subtasks if include_phases else []:
        start = int(np.searchsorted(aligned.timeline, subtask["start_ns"], side="left"))
        end = int(np.searchsorted(aligned.timeline, subtask["end_ns"], side="left"))
        pieces = [(max(a, start), min(b, end)) for a, b in aligned.segments
                  if a < end and b > start]
        pieces = [(a, b) for a, b in pieces if b - a >= config["min_segment_frames"]]
        base_id = f"subtask_{subtask['index']}"
        if not pieces:
            raise ValueError(f"{base_id}: no continuous valid aligned frames")
        if subtask["task"] not in translations:
            raise ValueError(f"{base_id}: missing English translation for {subtask['task']!r}")
        for part, bounds in enumerate(pieces, 1):
            segment_id = f"{base_id}_part_{part}" if len(pieces) > 1 else base_id
            result.append({
                "bounds": bounds,
                "segment_id": segment_id,
                "subtask_index": subtask["index"],
                "subtask_start_ns": subtask["start_ns"],
                "subtask_end_ns": subtask["end_ns"],
                "source_recording_id": recording_id,
                "task": translations[subtask["task"]],
            })
    return result
