"""ACT nearest-neighbor sampling; never differentiate across discarded gaps."""
from dataclasses import dataclass
import json
import math

import numpy as np

from convest.align import valid_segments

CAMERAS = {"cam0": "head", "cam1": "left_wrist", "cam2": "right_wrist"}


@dataclass
class ActEpisode:
    timeline: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    action: np.ndarray
    image_indices: dict
    segments: list
    report: dict


def validate_config(config):
    if config["state_dim"] not in (14, 54):
        raise ValueError("ACT state_dim must be 14 or 54")
    if config["window"] != "trimmed_validated":
        raise ValueError("ACT window must be trimmed_validated")
    if config["qvel_policy"] not in ("prefer_source", "zeros"):
        raise ValueError("qvel_policy must be prefer_source or zeros")
    for key in ("trim_start_sec", "trim_end_sec", "min_free_gb"):
        if not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    for key in ("image_max_delta_ms", "arm_max_delta_ms", "hand_max_delta_ms", "max_joint_step_rad"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("image_width", "image_height"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["compression"] not in (None, "lzf", "gzip"):
        raise ValueError("compression must be null, lzf or gzip")


def group_names(contract, dimension, action=False):
    names = contract["source_joint_names"]
    groups = [names[f"{'validated' if action else 'measured'}_{side}_arm"] for side in ("left", "right")]
    if dimension == 54:
        groups += [names[f"{side}_hand"] for side in ("left", "right")]
    return groups


def validate_contract(contract, config):
    for action in (False, True):
        groups = group_names(contract, config["state_dim"], action)
        if [len(g) for g in groups] != ([7, 7, 20, 20] if config["state_dim"] == 54 else [7, 7]):
            raise ValueError("ACT requires fixed joint groups of 7, 7, 20, 20 (or 7, 7)")
        flat = sum(groups, [])
        if len(set(flat)) != len(flat):
            raise ValueError("ACT joint names must be unique")


def nearest_indices(times, timeline):
    right = np.clip(np.searchsorted(times, timeline, side="left"), 0, len(times) - 1)
    left = np.maximum(right - 1, 0)
    # Deterministic ties go to the earlier message.
    return np.where(np.abs(times[left] - timeline) <= np.abs(times[right] - timeline), left, right)


def align(streams, start, end, config):
    fps = config["fps"]
    count = (end - start) * fps // 1_000_000_000 + 1
    if count < 2:
        raise ValueError("ACT source-time interval is empty or too short after trimming")
    timeline = start + np.arange(count, dtype=np.int64) * 1_000_000_000 // fps
    valid = np.ones(count, dtype=bool)
    unmatched, deltas, indices = {}, {}, {}
    for key, series in streams.items():
        index = nearest_indices(series.times, timeline)
        delta = np.abs(series.times[index] - timeline)
        category = "image" if key in CAMERAS else ("arm" if key.endswith("arm") else "hand")
        good = delta <= round(config[f"{category}_max_delta_ms"] * 1e6)
        valid &= good
        unmatched[key] = int((~good).sum())
        deltas[key] = float(delta[good].max() / 1e6) if good.any() else None
        indices[key] = index
    segments = valid_segments(valid, config["min_segment_frames"])
    if not segments:
        raise ValueError(f"No continuous ACT segments within nearest-neighbor limits: {unmatched}")
    groups = ["left_arm", "right_arm"] + (["left_hand", "right_hand"] if config["state_dim"] == 54 else [])
    states = [np.asarray(streams[f"state.{g}"].values)[indices[f"state.{g}"]] for g in groups]
    qpos = np.concatenate([s[:, :s.shape[1] // 2] for s in states], axis=1).astype(np.float32)
    qvel = np.concatenate([s[:, s.shape[1] // 2:] for s in states], axis=1).astype(np.float32)
    action = np.concatenate([np.asarray(streams[f"action.{g}"].values)[indices[f"action.{g}"]]
                             for g in groups], axis=1).astype(np.float32)
    estimated = 0
    for a, b in segments:
        if config["qvel_policy"] == "zeros":
            qvel[a:b] = 0
        else:
            missing = ~np.isfinite(qvel[a:b])
            estimated += int(missing.sum())
            # Subtract epoch before floating conversion to retain nanosecond precision.
            seconds = (timeline[a:b] - timeline[a]).astype(np.float64) / 1e9
            velocity = np.gradient(qpos[a:b].astype(np.float64), seconds, axis=0)
            qvel[a:b] = np.where(missing, velocity, qvel[a:b])
    retained = sum(b - a for a, b in segments)
    return ActEpisode(timeline, qpos, qvel, action, {c: indices[c] for c in CAMERAS}, segments,
                      {"grid_frames": count, "retained_frames": retained, "invalid_frames": int((~valid).sum()),
                       "short_segment_frames": int(valid.sum()) - retained, "segment_count": len(segments),
                       "unmatched": unmatched, "max_delta_ms": deltas, "estimated_velocity_values": estimated,
                       "source_start_ns": int(start), "source_end_ns": int(end), "alignment": "nearest"})


def prepare(source, bag, item, config, contract):
    streams = source.read_act_streams(bag, contract, config["state_dim"])
    report = json.loads((bag.path / "collection_state.json").read_text())["validation_report"]
    # The collection report's common source window is before its own edge trim.
    # Intersect, rather than subtracting a second second from effective bounds.
    common_start = report.get("source_common_start_time_ns", max(int(s.times[0]) for s in streams.values()))
    common_end = report.get("source_common_end_time_ns", min(int(s.times[-1]) for s in streams.values()))
    start = max(item["source_start_ns"], common_start + round(config["trim_start_sec"] * 1e9))
    end = min(item["source_end_ns"], common_end - round(config["trim_end_sec"] * 1e9))
    aligned = align(streams, start, end, config)
    aligned.report["window"] = {"policy": "trimmed_validated", "common_start_ns": common_start,
                               "common_end_ns": common_end, "validated_start_ns": item["source_start_ns"],
                               "validated_end_ns": item["source_end_ns"],
                               "trim_start_sec": config["trim_start_sec"], "trim_end_sec": config["trim_end_sec"]}
    return streams, aligned
