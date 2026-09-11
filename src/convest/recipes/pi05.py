"""Preserve the original Pi05 causal alignment and source contract."""
import numpy as np

from convest.align import align


def validate_config(config):
    if config["window"] != "validated":
        raise ValueError("Only the validated source-time window is supported")
    if config["max_staleness_ms"] <= 0:
        raise ValueError("Invalid freshness limit")
    if not 0 <= config["video_crf"] <= 51 or config["video_threads"] < 1:
        raise ValueError("Invalid encoder settings")
    if config.get("segments", "all") not in {"all", "full", "milestones"}:
        raise ValueError("segments must be all, full, or milestones")
    tasks = config.get("segment_tasks", {})
    if not isinstance(tasks, dict) or any(not isinstance(key, str) or not key.strip()
                                           or not isinstance(value, str) or not value.strip()
                                           for key, value in tasks.items()):
        raise ValueError("segment_tasks must map nonempty segment IDs to nonempty task strings")


def validate_contract(contract, config):
    if contract["features"]["observation.state"]["dimension"] != 108 or contract["features"]["action"]["dimension"] != 54:
        raise ValueError("Pi05 FR3/Wuji requires state=108, action=54 in the source contract order")


def prepare(source, bag, item, config, contract):
    streams = source.read_streams(bag, contract)
    aligned = align(streams, item["source_start_ns"], item["source_end_ns"], config["fps"],
                    int(config["max_staleness_ms"] * 1e6), config["min_segment_frames"])
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
    markers = [] if mode == "full" else _milestones(item)
    tasks = config.get("segment_tasks", {})
    recording_id = item.get("source_recording_id") or item["path"]

    def description(segment_id):
        return tasks.get(segment_id, config["task"] if segment_id == "full" else segment_id)

    if mode == "milestones" and not markers:
        raise ValueError("milestones export requested but this bag has no milestones")
    selected_markers = [] if mode == "full" else markers
    if selected_markers and aligned.segments != [(0, len(aligned.timeline))]:
        raise ValueError("A bag with milestones must have one continuous valid aligned interval")

    result = []
    for marker_id, marker_ns in selected_markers:
        if not int(aligned.timeline[0]) < marker_ns < int(item["source_end_ns"]):
            raise ValueError(f"{marker_id}: milestone is outside the validated source-time interval")
        end = int(np.searchsorted(aligned.timeline, marker_ns, side="right"))
        if end < config["min_segment_frames"]:
            raise ValueError(f"{marker_id}: milestone prefix is shorter than min_segment_frames")
        result.append({"bounds": (0, end), "segment_id": marker_id,
                       "milestone_timestamp_ns": marker_ns, "source_recording_id": recording_id,
                       "task": description(marker_id)})

    if mode in {"all", "full"}:
        many = len(aligned.segments) > 1
        for index, bounds in enumerate(aligned.segments, 1):
            segment_id = f"full_part_{index}" if many else "full"
            result.append({"bounds": bounds, "segment_id": segment_id,
                           "milestone_timestamp_ns": None, "source_recording_id": recording_id,
                           "task": description("full")})
    return result
