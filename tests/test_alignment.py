from types import SimpleNamespace
import struct

import numpy as np
import pytest

from convest.align import align, GROUPS, valid_segments
from convest.sources.gello_rosbag2 import (CompactTimeMap, Series,
                                           bridge_compact_image_boundaries,
                                           header_ns, joint_values)
from convest.recipes.pi05 import output_segments


def streams():
    times = np.arange(21, dtype=np.int64) * 100_000_000 + 1_000_000_000
    result = {}
    for group, n in zip(GROUPS, (7, 7, 20, 20)):
        result[f"state.{group}"] = Series(times, [np.full(2 * n, t, np.float32) for t in range(21)])
        result[f"action.{group}"] = Series(times, [np.full(n, 100 + t, np.float32) for t in range(21)])
        result[f"status.{group}"] = Series(times, [(True, True)] * 21)
    for camera in ("cam0", "cam1", "cam2"):
        result[camera] = Series(times, list(range(21)))
    return result


def test_causal_sampling_and_dimensions():
    raw = streams()
    result = align(raw, 1_050_000_000, 2_950_000_000, 30, 150_000_000)
    assert result.state.shape[1] == 108 and result.action.shape[1] == 54
    assert result.state[0, 0] == 0  # Never use the future 1.1s state.
    assert result.action[0, 0] == 100
    for camera in result.image_indices:
        assert np.all(raw[camera].times[result.image_indices[camera]] <= result.timeline)
    assert np.max(np.abs(np.diff(result.timeline) - 1e9 / 30)) <= 1


def test_camera_gap_splits_without_compressing_time():
    raw = streams()
    raw["cam0"] = Series(raw["cam0"].times[[0, 1, 2, 15, 16, 17, 18, 19, 20]], list(range(9)))
    result = align(raw, 1_000_000_000, 3_000_000_000, 10, 150_000_000)
    assert result.segments == [(0, 4), (15, 21)]
    assert result.report["invalid_frames"] == 11


def test_inactive_holds_then_requires_new_command():
    raw = streams()
    t = raw["status.left_arm"].times
    raw["status.left_arm"] = Series(t, [(True, True)] * 4 + [(False, True)] * 8 + [(True, True)] * 9)
    raw["action.left_arm"] = Series(t[:4], [np.full(7, i, np.float32) for i in range(4)])
    result = align(raw, int(t[0]), int(t[-1]), 10, 150_000_000)
    assert np.all(result.action[4:12, :7] == 3)
    assert result.segments == [(0, 12)]  # Reactivation cannot reuse a stale command.


def test_inactive_at_start_uses_measured_seed():
    raw = streams()
    t = raw["status.left_arm"].times
    raw["status.left_arm"] = Series(t, [(False, True)] * 21)
    raw["action.left_arm"] = Series(t[-1:], [np.ones(7)])
    raw["state.left_arm"].values[0] = np.full(14, 42, np.float32)
    result = align(raw, int(t[0]), int(t[-1]), 10, 150_000_000)
    assert np.all(result.action[:, :7] == 42)


def test_stale_status_cannot_authorize_hold():
    raw = streams()
    raw["status.left_arm"] = Series(raw["status.left_arm"].times[:1], [(False, True)])
    result = align(raw, 1_000_000_000, 3_000_000_000, 10, 150_000_000)
    assert result.segments == [(0, 2)]


def test_joint_name_reordering_and_rejections():
    msg = SimpleNamespace(name=["b", "a"], position=[2, 1], velocity=[4, 3])
    np.testing.assert_array_equal(joint_values(msg, ["a", "b"], ["position", "velocity"]), [1, 2, 3, 4])
    assert joint_values(msg, ["c"], ["position"], allow_absent=True) is None
    with pytest.raises(ValueError, match="Partial"):
        joint_values(msg, ["a", "c"], ["position"], allow_absent=True)
    msg.velocity = []
    with pytest.raises(ValueError, match="length"):
        joint_values(msg, ["a", "b"], ["velocity"])


def test_cdr_and_duplicate_header_policy():
    assert header_ns(b"\x00\x01\x00\x00" + struct.pack("<iI", 123, 4)) == 123000000004
    assert header_ns(b"\x00\x00\x00\x00" + struct.pack(">iI", 123, 4)) == 123000000004
    with pytest.raises(ValueError):
        header_ns(b"\x00" * 12)
    s = Series.build([(3, "c"), (1, "a"), (1, "last")])
    assert s.times.tolist() == [1, 3] and s.values == ["last", "c"]
    assert valid_segments(np.array([True, False, True, True]), 2) == [(2, 4)]


@pytest.mark.parametrize("milestones,error", [
    ([{"id": "milestone_1", "timestamp_ns": 1_500_000_000, "clock": "wall"}], "clock"),
    ([{"id": "full", "timestamp_ns": 1_500_000_000, "clock": "ros"}], "IDs"),
    ([{"id": "milestone_1", "timestamp_ns": 999_000_000, "clock": "ros"}], "outside"),
])
def test_invalid_milestones_are_rejected(milestones, error):
    aligned = align(streams(), 1_000_000_000, 3_000_000_000, 10, 150_000_000)
    item = {"path": "/bags/episode1", "source_end_ns": 3_000_000_000,
            "source_recording_id": "recording-1", "milestones": milestones}
    config = {"segments": "all", "segment_tasks": {}, "task": "Full task", "min_segment_frames": 2}
    with pytest.raises(ValueError, match=error):
        output_segments(item, aligned, config)


@pytest.mark.parametrize("milestones,expected", [
    ([], [("full", (0, 21))]),
    ([{"id": "milestone_1", "timestamp_ns": 1_500_000_000, "clock": "ros"}],
     [("milestone_1", (0, 6))]),
    ([{"id": "milestone_1", "timestamp_ns": 1_500_000_000, "clock": "ros"},
      {"id": "milestone_2", "timestamp_ns": 2_000_000_000, "clock": "ros"}],
     [("milestone_1", (0, 6))]),
])
def test_first_milestone_or_full_mode(milestones, expected):
    aligned = align(streams(), 1_000_000_000, 3_000_000_000, 10, 150_000_000)
    item = {"path": "/bags/episode1", "source_end_ns": 3_000_000_000,
            "source_recording_id": "recording-1", "milestones": milestones}
    config = {"segments": "first-milestone-or-full", "segment_tasks": {},
              "task": "Full task", "min_segment_frames": 2}
    result = output_segments(item, aligned, config)
    assert [(segment["segment_id"], segment["bounds"]) for segment in result] == expected


def test_first_milestone_mode_trims_leading_invalid_but_rejects_internal_gap():
    aligned = align(streams(), 1_000_000_000, 3_000_000_000, 10, 150_000_000)
    item = {"path": "/bags/episode1", "source_end_ns": 3_000_000_000,
            "milestones": [{"id": "milestone_1", "timestamp_ns": 1_500_000_000, "clock": "ros"}]}
    config = {"segments": "first-milestone-or-full", "segment_tasks": {},
              "task": "Task", "min_segment_frames": 2}
    aligned.segments = [(1, len(aligned.timeline))]
    assert output_segments(item, aligned, config)[0]["bounds"] == (1, 6)
    aligned.segments = [(1, 3), (4, len(aligned.timeline))]
    with pytest.raises(ValueError, match="internal invalid interval"):
        output_segments(item, aligned, config)


def test_compact_boundary_hold_is_inserted_only_for_a_stale_grid_hole():
    series = Series(np.array([200_000_000, 600_000_000], dtype=np.int64), ["before", "after"])
    mapping = CompactTimeMap(((400_000_000, 500_000_000, 100_000_000),), frozenset(), "trim.json")
    inserted = bridge_compact_image_boundaries(
        {"cam0": series}, mapping, 0, 1_000_000_000, 10, 150_000_000, camera_keys=("cam0",),
    )
    assert inserted == {"cam0": 1}
    assert series.times.tolist() == [200_000_000, 400_000_000, 600_000_000]
    assert series.values == ["before", "before", "after"]

    fresh = Series(np.array([350_000_000, 450_000_000], dtype=np.int64), ["before", "after"])
    assert bridge_compact_image_boundaries(
        {"cam0": fresh}, mapping, 0, 1_000_000_000, 10, 150_000_000, camera_keys=("cam0",),
    ) == {}
