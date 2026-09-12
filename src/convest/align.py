"""Causal sample-and-hold on an integer nanosecond grid; never bridge gaps."""
from dataclasses import dataclass

import numpy as np

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


def align(streams, start, end, fps, max_age_ns, min_frames=2, camera_keys=CAMERAS):
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
    image_indices = {}
    for camera in camera_keys:
        s = streams[camera]
        index = s.indices(timeline)
        age = timeline - s.times[np.maximum(index, 0)]
        good = (index >= 0) & (age >= 0) & (age <= max_age_ns)
        valid &= good
        unmatched[camera] = int((~good).sum())
        ages[camera] = float(age[good].max() / 1e6) if good.any() else None
        image_indices[camera] = index
        source_indices[camera] = index
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
                           "source_start_ns": int(start), "source_end_ns": int(end)}, source_indices)
