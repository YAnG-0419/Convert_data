"""Two-grid-frame image tolerance, whole-recording rejection, and audit logs."""
import json
import pickle
import sqlite3

import numpy as np
import pytest

from convest import pipeline
from convest.align import align, GROUPS
from convest.config import WORKSPACE, load_config
from convest.quality import RecordingRejected
from convest.recipes import pi05
from convest.sources.gello_rosbag2 import Series, snapshot
from convest.verify import verify
from test_export_modes import cleaned_fixture


def data():
    times = 1_000_000_000 + np.arange(31, dtype=np.int64) * 1_000_000_000 // 30
    raw = {}
    for group, n in zip(GROUPS, (7, 7, 20, 20)):
        raw[f"state.{group}"] = Series(times.copy(), [np.full(2 * n, i, np.float32) for i in range(31)])
        raw[f"action.{group}"] = Series(times.copy(), [np.full(n, 100 + i, np.float32) for i in range(31)])
        raw[f"status.{group}"] = Series(times.copy(), [(True, True)] * 31)
    for camera in ("cam0", "cam1", "cam2"):
        raw[camera] = Series(times.copy(), list(range(31)))
    return raw


def camera_hole(raw, frames, camera="cam0", last_good=6):
    # Last image at grid 6 becomes stale at grid 11 (age ~166.7 ms).
    s = raw[camera]
    keep = [i for i in range(31) if i <= last_good or i >= last_good + 5 + frames]
    raw[camera] = Series(s.times[keep], [s.values[i] for i in keep])


def run(raw, **kwargs):
    return align(raw, 1_000_000_000, 2_000_000_000, 30, 150_000_000,
                 camera_short_gap_max_frames=2, **kwargs)


@pytest.mark.parametrize("frames", [1, 2])
def test_short_holes_only_reuse_past_images_without_touching_vectors(frames):
    raw = data()
    camera_hole(raw, frames)
    old = align(raw, 1_000_000_000, 2_000_000_000, 30, 150_000_000)
    before = raw["cam0"].times.copy()
    fixed = run(raw, camera_gap_reject_min_frames=3)
    assert fixed.segments == [(0, 31)]
    np.testing.assert_array_equal(fixed.state, old.state)
    np.testing.assert_array_equal(fixed.action, old.action)
    np.testing.assert_array_equal(fixed.image_indices["cam0"], old.image_indices["cam0"])
    np.testing.assert_array_equal(raw["cam0"].times, before)
    assert (raw["cam0"].times[fixed.image_indices["cam0"]] <= fixed.timeline).all()
    audit = fixed.report["camera_short_gap_tolerance"]
    assert audit["accepted_frames"] == frames
    assert audit["gaps"][0]["bounds"] == [11, 11 + frames]
    assert audit["gaps"][0]["cameras"]["cam0"]["image_source_timestamps_ns"] == [1_200_000_000] * frames


def test_three_frame_gap_is_never_partially_filled_and_can_reject_whole_recording():
    raw = data()
    camera_hole(raw, 3)
    fixed = run(raw)
    assert fixed.segments == [(0, 11), (14, 31)]
    assert fixed.report["camera_short_gap_tolerance"]["accepted_frames"] == 0
    with pytest.raises(RecordingRejected) as result:
        run(raw, camera_gap_reject_min_frames=3)
    assert result.value.details["gaps"][0]["bounds"] == [11, 14]
    restored = pickle.loads(pickle.dumps(result.value))
    assert restored.reason == result.value.reason
    assert restored.details == result.value.details


def test_staggered_camera_holes_count_as_one_union_gap():
    raw = data()
    camera_hole(raw, 2, "cam0", last_good=6)  # [11,13)
    camera_hole(raw, 2, "cam1", last_good=8)  # [13,15)
    with pytest.raises(RecordingRejected) as result:
        run(raw, camera_gap_reject_min_frames=3)
    assert result.value.details["gaps"][0]["bounds"] == [11, 15]
    assert result.value.details["gaps"][0]["cameras"] == ["cam0", "cam1"]


def test_two_separate_two_frame_holes_are_not_counted_cumulatively():
    raw = data()
    camera_hole(raw, 2, "cam0", last_good=6)
    camera_hole(raw, 2, "cam1", last_good=16)
    fixed = run(raw, camera_gap_reject_min_frames=3)
    assert fixed.segments == [(0, 31)]
    assert fixed.report["camera_short_gap_tolerance"]["accepted_frames"] == 4


def test_age_limit_applies_to_entire_hole_not_just_first_tick():
    raw = data()
    camera_hole(raw, 2)
    result = run(raw, camera_short_gap_max_age_ns=190_000_000)
    assert result.segments == [(0, 11), (13, 31)]
    assert result.report["camera_short_gap_tolerance"]["accepted_frames"] == 0


@pytest.mark.parametrize("age", [float("nan"), float("inf"), 0, -1])
def test_alignment_rejects_invalid_age_caps(age):
    with pytest.raises(ValueError, match="finite and positive"):
        run(data(), camera_short_gap_max_age_ns=age)


@pytest.mark.parametrize("kind", ["state.left_arm", "action.left_arm", "status.left_arm"])
def test_nonvisual_faults_cannot_be_repaired_by_camera_tolerance(kind):
    raw = data()
    camera_hole(raw, 2)
    if kind.startswith("status"):
        raw[kind].values[11] = (True, False)
    else:
        series = raw[kind]
        keep = [i for i in range(31) if i <= 6 or i >= 13]
        raw[kind] = Series(series.times[keep], [series.values[i] for i in keep])
    fixed = run(raw)
    assert fixed.report["camera_short_gap_tolerance"]["accepted_frames"] == 0
    assert len(fixed.segments) == 2


@pytest.mark.parametrize("edge", ["start", "end"])
def test_edges_are_not_filled_or_rejected_as_internal_gaps(edge):
    raw = data()
    s = raw["cam0"]
    raw["cam0"] = Series(s.times[3:], s.values[3:]) if edge == "start" else Series(s.times[:24], s.values[:24])
    result = run(raw, camera_gap_reject_min_frames=3)
    assert result.report["camera_short_gap_tolerance"]["accepted_frames"] == 0
    assert len(result.segments) == 1


@pytest.mark.parametrize("overrides", [
    {"camera_short_gap_max_frames": 3}, {"camera_short_gap_max_frames": True},
    {"camera_short_gap_max_age_ms": float("nan")}, {"camera_short_gap_max_age_ms": 100},
    {"camera_gap_reject_min_frames": 2}, {"camera_gap_reject_min_frames": True},
])
def test_invalid_policy_config(overrides):
    config = load_config(WORKSPACE / "configs/gello_pi05.yaml")
    with pytest.raises(ValueError):
        pi05.validate_config(dict(config, **overrides))


def test_new_policy_does_not_invoke_legacy_retimestamped_image_holds(monkeypatch):
    from types import SimpleNamespace
    raw = data()
    camera_hole(raw, 2)

    def forbidden(*args, **kwargs):
        pytest.fail("Legacy image timestamp rewriting must not run")

    source = SimpleNamespace(read_streams=lambda *args: raw, bridge_compact_image_boundaries=forbidden,
                             STATUS_MESSAGE_TYPES=())
    config = load_config(WORKSPACE / "configs/gello_pi05.yaml")
    _, result = pi05.prepare(source, SimpleNamespace(compact_time_map=None),
                            {"source_start_ns": 1_000_000_000, "source_end_ns": 2_000_000_000}, config, {})
    assert result.segments == [(0, 31)]


def test_pipeline_excludes_all_full_and_phase_outputs_and_audits_reason(tmp_path, monkeypatch):
    source = tmp_path / "bags"
    for number, deleted_images in ((1, 1), (2, 2)):
        path = source / f"episode{number}"
        cleaned_fixture(path)
        with sqlite3.connect(path / "capture_0.db3") as db:
            db.execute("DELETE FROM messages WHERE topic_id=(SELECT id FROM topics WHERE name='/cam0/color/image_raw') "
                       "AND timestamp>=? AND timestamp<?",
                       (10_605_000_000, 10_605_000_000 + deleted_images * 100_000_000))
    before = {p.name: snapshot(p) for p in source.iterdir()}
    config = load_config(WORKSPACE / "configs/gello_pi05.yaml")
    config.update(source_root=str(source), output_root=str(tmp_path / "result"), segments="both",
                  repo_id="fr3_wuji/gap_test", subtask_tasks={f"Phase {i}": f"English {i}" for i in range(1, 5)})
    monkeypatch.setattr(pipeline, "WORKSPACE", tmp_path)
    for resume in (False, True):
        assert pipeline.convert(config, resume=resume, workers=2) == 0
        for name, count in (("full", 1), ("subtasks", 4)):
            output = tmp_path / "result" / name
            assert verify(output, full_video=True)["episodes"] == count
            records = pipeline.load_records(output)
            assert len(records) == 1 and records[0]["source"].endswith("episode1")
            assert records[0]["alignment"]["camera_short_gap_tolerance"]["accepted_frames"] == 1
            rejected = json.loads((output / "conversion/rejections.json").read_text())
            assert len(rejected) == 1 and rejected[0]["source"].endswith("episode2")
            assert rejected[0]["details"]["gaps"][0]["frames"] >= 3
            assert json.loads((output / "conversion/errors.json").read_text()) == []
            assert json.loads((output / "conversion/summary.json").read_text())["excluded_by_quality"] == 1
    assert {p.name: snapshot(p) for p in source.iterdir()} == before
