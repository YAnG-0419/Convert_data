"""Causal sampling with optional bounded, camera-only short-gap tolerance."""
from dataclasses import dataclass

import numpy as np

from convest.quality import RecordingRejected

GROUPS = ("left_arm", "right_arm", "left_hand", "right_hand")
CAMERAS = ("cam0", "cam1", "cam2")


@dataclass
class AlignedEpisode:
    timeline: np.ndarray
    state: np.ndarray
    action: np.ndarray
    engaged: np.ndarray
    image_indices: dict
    segments: list
    report: dict
    source_indices: dict | None = None


def valid_segments(valid, minimum):
    edges = np.diff(np.r_[False, valid, False].astype(np.int8))
    return [(int(a), int(b)) for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
            if b - a >= minimum]


def align(streams, start, end, fps, max_age_ns, min_frames=2, camera_keys=CAMERAS, *,
          camera_short_gap_max_frames=0, camera_short_gap_max_age_ns=250_000_000,
          camera_gap_reject_min_frames=0):
    if type(camera_short_gap_max_frames) is not int or camera_short_gap_max_frames not in (0, 1, 2):
        raise ValueError("camera_short_gap_max_frames must be 0, 1, or 2")
    if not np.isfinite(camera_short_gap_max_age_ns) or camera_short_gap_max_age_ns <= 0:
        raise ValueError("Camera short-gap age limit must be finite and positive")
    if camera_short_gap_max_frames and camera_short_gap_max_age_ns < max_age_ns:
        raise ValueError("Camera short-gap age limit must be >= the normal freshness limit")
    if type(camera_gap_reject_min_frames) is not int or camera_gap_reject_min_frames not in (0, 3):
        raise ValueError("camera_gap_reject_min_frames must be 0 (off) or 3")
    count = (end - start) * fps // 1_000_000_000 + 1
    if count < 1:
        raise ValueError("Empty source-time interval")
    timeline = start + np.arange(count, dtype=np.int64) * 1_000_000_000 // fps
    valid = np.ones(count, dtype=bool)
    unmatched = {}
    ages = {}
    source_indices = {}

    def sample(key):
        series = streams[key]
        index = series.indices(timeline)
        age = timeline - series.times[np.maximum(index, 0)]
        good = (index >= 0) & (age >= 0) & (age <= max_age_ns)
        valid[:] &= good
        unmatched[key] = int((~good).sum())
        ages[key] = float(age[good].max() / 1e6) if good.any() else None
        source_indices[key] = index
        return index, np.asarray(series.values)[np.maximum(index, 0)]

    states, actions, engaged = [], [], []
    for group, dimension in zip(GROUPS, (7, 7, 20, 20)):
        _, state = sample(f"state.{group}")
        states.append(state)
        status_index, status = sample(f"status.{group}")
        active = status[:, 0].astype(bool)
        valid &= status[:, 1].astype(bool)
        source = streams[f"status.{group}"]
        active_source = np.asarray(source.values)[:, 0].astype(bool)
        transitions = np.r_[0, np.flatnonzero(active_source[1:] != active_source[:-1]) + 1]
        run = np.searchsorted(transitions, np.maximum(status_index, 0), side="right") - 1
        transition_time = source.times[transitions[run]]
        command = streams[f"action.{group}"]
        command_index = command.indices(timeline)
        command_time = command.times[np.maximum(command_index, 0)]
        active_good = ((command_index >= 0) & (timeline - command_time <= max_age_ns)
                       & (command_time + 20_000_000 >= transition_time))
        action = np.asarray(command.values)[np.maximum(command_index, 0)].copy()
        # A disengaged group holds the target at the actual status transition.
        hold_index = command.indices(transition_time)
        held = np.asarray(command.values)[np.maximum(hold_index, 0)]
        measured = streams[f"state.{group}"]
        seed_index = measured.indices(transition_time)
        seeds = np.asarray(measured.values)[np.maximum(seed_index, 0), :dimension]
        # If capture began inactive before its first state, seed once from first measured state.
        held = np.where((hold_index >= 0)[:, None], held, seeds)
        action[~active] = held[~active]
        selected_command_index = command_index.copy()
        selected_command_index[~active] = hold_index[run][~active]
        source_indices[f"action.{group}"] = selected_command_index
        good = ~active | active_good
        valid &= good
        unmatched[f"action.{group}"] = int((~good).sum())
        actions.append(action)
        engaged.append(active)
    image_indices, camera_good, camera_ages = {}, {}, {}
    for camera in camera_keys:
        s = streams[camera]
        index = s.indices(timeline)
        age = timeline - s.times[np.maximum(index, 0)]
        good = (index >= 0) & (age >= 0) & (age <= max_age_ns)
        camera_good[camera] = good
        camera_ages[camera] = age
        image_indices[camera] = index
        source_indices[camera] = index
    tolerated = []
    if camera_good:
        # Count the UNION of stale intervals across cameras, not each camera
        # separately: staggered short holes must not conceal a long scene gap.
        visual_good = np.logical_and.reduce(list(camera_good.values()))
        visual_gaps = valid_segments(~visual_good, 1)
        rejected_gaps = [
            {"bounds": [a, b], "frames": b - a,
             "source_start_ns": int(timeline[a]), "source_end_ns": int(timeline[b]),
             "cameras": [c for c, good in camera_good.items() if not good[a:b].all()]}
            for a, b in visual_gaps
            if camera_gap_reject_min_frames and 0 < a < b < count
            and b - a >= camera_gap_reject_min_frames
        ]
        if rejected_gaps:
            raise RecordingRejected(
                "Whole recording excluded: internal camera gap of 3 or more output frames",
                {"policy": "camera_gap_reject_min_frames", "threshold": camera_gap_reject_min_frames,
                 "fps": fps, "gaps": rejected_gaps},
            )
        for a, b in visual_gaps if camera_short_gap_max_frames else []:
            if a == 0 or b == count or b - a > camera_short_gap_max_frames:
                continue
            if not valid[a - 1:b + 1].all():
                continue  # Never bridge bad state/action/status or their boundary.
            if any((image_indices[c][a:b] < 0).any()
                   or (camera_ages[c][a:b] < 0).any()
                   or (camera_ages[c][a:b] > camera_short_gap_max_age_ns).any()
                   for c in camera_good):
                continue  # All-or-nothing: no partial repair of a rejected gap.
            details = {}
            for camera, good in camera_good.items():
                indices = np.flatnonzero(~good[a:b]) + a
                if len(indices):
                    details[camera] = {
                        "frame_indices": indices.tolist(),
                        "image_source_timestamps_ns": streams[camera].times[image_indices[camera][indices]].tolist(),
                        "image_age_ms": (camera_ages[camera][indices] / 1e6).tolist(),
                    }
                    good[a:b] = True
            tolerated.append({"bounds": [a, b], "cameras": details})
    for camera, good in camera_good.items():
        valid &= good
        unmatched[camera] = int((~good).sum())
        age = camera_ages[camera]
        ages[camera] = float(age[good].max() / 1e6) if good.any() else None
    segments = valid_segments(valid, min_frames)
    retained = sum(b - a for a, b in segments)
    if not segments:
        raise ValueError(f"No continuous valid segments: {unmatched}")
    return AlignedEpisode(timeline, np.concatenate(states, axis=1).astype(np.float32),
                          np.concatenate(actions, axis=1).astype(np.float32),
                          np.stack(engaged, axis=1).astype(np.float32), image_indices, segments,
                          {"grid_frames": count, "retained_frames": retained,
                           "invalid_frames": int((~valid).sum()), "short_segment_frames": int(valid.sum()) - retained,
                           "segment_count": len(segments), "unmatched": unmatched, "max_age_ms": ages,
                           "camera_short_gap_tolerance": {
                               "max_frames": camera_short_gap_max_frames,
                               "max_image_age_ms": camera_short_gap_max_age_ns / 1e6,
                               "reject_min_frames": camera_gap_reject_min_frames,
                               "accepted_frames": sum(x["bounds"][1] - x["bounds"][0] for x in tolerated),
                               "gaps": tolerated,
                           },
                           "source_start_ns": int(start), "source_end_ns": int(end)}, source_indices)
