"""ACT schema, nearest alignment, source order, velocity fallback and safe resume."""
import json
from pathlib import Path
import sqlite3

import h5py
import numpy as np
import pytest

from convest import pipeline
from convest.config import WORKSPACE, load_config
from convest.formats.act_hdf5 import check_vectors, resize_rgb, verify
from convest.recipes.act import align, nearest_indices, prepare
from convest.sources.gello_rosbag2 import Bag, Series, snapshot
from convest import registry
from test_pipeline import make_bag


def config():
    return load_config(WORKSPACE / "configs/gello_act.yaml")


def streams(times, dimension=54):
    groups = [("left_arm", 7), ("right_arm", 7)]
    if dimension == 54:
        groups += [("left_hand", 20), ("right_hand", 20)]
    result = {}
    seconds = (times - times[0]) / 1e9
    for group, n in groups:
        result[f"state.{group}"] = Series(times, [np.r_[np.full(n, t * 2), np.full(n, np.nan)] for t in seconds])
        result[f"action.{group}"] = Series(times, [np.full(n, t + 1) for t in seconds])
    for camera in ("cam0", "cam1", "cam2"):
        result[camera] = Series(times, list(range(len(times))))
    return result


def test_nearest_uses_future_and_ties_choose_past():
    times = np.array([100, 200, 300], dtype=np.int64)
    assert nearest_indices(times, np.array([0, 100, 150, 151, 200, 999])).tolist() == [0, 0, 0, 1, 1, 2]


def test_act_thresholds_split_and_finite_difference_stays_inside_segments():
    cfg = config()
    times = 1_788_505_900_000_000_000 + np.arange(10, dtype=np.int64) * 50_000_000
    raw = streams(times)
    keep = np.array([0, 1, 2, 7, 8, 9])
    raw["action.left_arm"] = Series(times[keep], [raw["action.left_arm"].values[i] for i in keep])
    # Large position change across an invalid region must not affect qvel.
    for key in raw:
        if key.startswith("state."):
            n = len(raw[key].values[0]) // 2
            for value in raw[key].values[7:]:
                value[:n] += 100
    result = align(raw, int(times[0]), int(times[-1]), cfg)
    assert result.segments == [(0, 3), (7, 10)]
    assert result.report["invalid_frames"] == 4
    for a, b in result.segments:
        np.testing.assert_allclose(result.qvel[a:b], 2, atol=1e-3)
    raw = streams(times)
    raw["state.left_arm"].times = times + 20_000_000
    assert align(raw, int(times[0]), int(times[-1]), cfg).report["invalid_frames"] == 0
    raw["state.left_arm"].times = times + 20_000_001
    with pytest.raises(ValueError, match="No continuous ACT"):
        align(raw, int(times[0]), int(times[-1]), cfg)


def test_source_velocity_is_preferred_and_zero_policy_explicit():
    cfg = config()
    times = np.arange(3, dtype=np.int64) * 50_000_000 + 10_000_000_000
    raw = streams(times)
    for value in raw["state.left_arm"].values:
        value[7:] = 42
    result = align(raw, int(times[0]), int(times[-1]), cfg)
    np.testing.assert_array_equal(result.qvel[:, :7], 42)
    np.testing.assert_allclose(result.qvel[:, 7:], 2)
    result = align(raw, int(times[0]), int(times[-1]), dict(cfg, qvel_policy="zeros"))
    assert not result.qvel.any()


def test_head_padding_rgb_and_wrist_resize():
    rgb = np.full((400, 640, 3), [220, 40, 10], dtype=np.uint8)
    resized = resize_rgb(rgb, 320, 240)
    assert resized.shape == (240, 320, 3) and resized.dtype == np.uint8
    assert not resized[:19].any() and not resized[-19:].any()
    np.testing.assert_array_equal(resized[100, 100], [220, 40, 10])
    wrist = resize_rgb(np.full((480, 640, 3), [220, 40, 10], dtype=np.uint8), 320, 240)
    np.testing.assert_array_equal(wrist[0, 0], [220, 40, 10])


def test_quality_rejects_all_zero_nonfinite_and_large_jumps():
    value = np.ones((3, 14), dtype=np.float32)
    with pytest.raises(ValueError, match="all-zero"):
        check_vectors(value, value, value * 0, 1)
    bad = value.copy()
    bad[0, 0] = np.inf
    with pytest.raises(ValueError, match="NaN/inf"):
        check_vectors(bad, value, value, 1)
    bad[0, 0] = 3
    with pytest.raises(ValueError, match="jump"):
        check_vectors(value, value, bad, 1)


def remove_optional_streams_and_velocities(path):
    store = Bag(path, WORKSPACE / "schemas").store
    with sqlite3.connect(path / "capture_0.db3") as db:
        # 14D ACT must not require hands or the Pi05 status topics.
        db.execute("DELETE FROM messages WHERE topic_id IN (SELECT id FROM topics WHERE name LIKE '%wuji%' OR name LIKE '%status%')")
        db.execute("DELETE FROM topics WHERE name LIKE '%wuji%' OR name LIKE '%status%'")
        rows = db.execute("SELECT messages.id, data FROM messages JOIN topics ON topic_id=topics.id WHERE type='sensor_msgs/msg/JointState'").fetchall()
        for ident, payload in rows:
            msg = store.deserialize_cdr(payload, "sensor_msgs/msg/JointState")
            msg.velocity = np.array([], dtype=np.float64)
            db.execute("UPDATE messages SET data=? WHERE id=?", (bytes(store.serialize_cdr(msg, msg.__msgtype__)), ident))


@pytest.mark.parametrize("dimension", [14, 54])
def test_act_cdr_parallel_append_resume_and_verify(tmp_path, monkeypatch, dimension):
    source, output, listing = tmp_path / "bags", tmp_path / "output", tmp_path / "episodes.txt"
    for n in (9, 10):
        make_bag(source / f"episode{n}")
        if dimension == 14:
            remove_optional_streams_and_velocities(source / f"episode{n}")
    before = {p.name: snapshot(p) for p in source.iterdir()}
    cfg = dict(config(), source_root=str(source), output_root=str(output), state_dim=dimension,
               fps=10, trim_start_sec=0.1, trim_end_sec=0.1)
    monkeypatch.setattr(pipeline, "WORKSPACE", tmp_path)
    listing.write_text("10")
    assert pipeline.convert(cfg, episode_list=listing, resume=True) == 0
    old = output / "episode_0.hdf5"
    old_bytes, old_mtime = old.read_bytes(), old.stat().st_mtime_ns
    with h5py.File(old, "r") as f:
        assert f["observations/qpos"].shape == (17, dimension)
        np.testing.assert_array_equal(f["observations/qpos"][0, :7], np.arange(7) + 2)
        np.testing.assert_array_equal(f["action"][0, :14], np.arange(14) + 102)
        np.testing.assert_allclose(f["observations/qvel"][:], 10 if dimension == 14 else 0.1, atol=1e-5)
        np.testing.assert_array_equal(f["observations/images/head"][0, 120, 160], [123, 77, 10])
        if dimension == 54:
            np.testing.assert_array_equal(f["observations/qpos"][0, 14:34], np.arange(20) + 2)
    listing.write_text("9-10")
    orphan = output / "episode_999.hdf5"
    orphan.write_bytes(b"interrupted")
    assert pipeline.convert(cfg, episode_list=listing, resume=True, workers=2) == 0
    assert not orphan.exists()
    assert old.read_bytes() == old_bytes and old.stat().st_mtime_ns == old_mtime
    assert verify(output)["episodes"] == 2
    records = pipeline.load_records(output)
    assert [Path(r["source"]).name for r in records] == ["episode10", "episode9"]
    assert before == {p.name: snapshot(p) for p in source.iterdir()}
    assert pipeline.convert(cfg, episode_list=listing, resume=True) == 0
    with pytest.raises(ValueError, match="differs"):
        pipeline.convert(dict(cfg, fps=20), episode_list=listing, resume=True)
    with pytest.raises(ValueError, match="workspace"):
        pipeline.convert(dict(cfg, output_root=str(tmp_path.parent / "other")))
    with h5py.File(output / "episode_1.hdf5", "r+") as f:
        f["observations/images/head"][0] = 0
    with pytest.raises(ValueError, match="all-black"):
        verify(output)


def test_trim_intersection_does_not_double_crop(tmp_path):
    path = tmp_path / "episode103"
    make_bag(path)
    state_path = path / "collection_state.json"
    state = json.loads(state_path.read_text())
    state["validation_report"].update(source_common_start_time_ns=10_000_000_000,
                                      source_common_end_time_ns=12_000_000_000)
    state_path.write_text(json.dumps(state))
    source = registry.get_source("gello_rosbag2")
    item = source.discover(path)[0]
    cfg = dict(config(), fps=10, trim_start_sec=0.2, trim_end_sec=0.2)
    import yaml
    contract = yaml.safe_load(Path(cfg["contract"]).read_text())
    with Bag(path, cfg["schema_dir"]) as bag:
        _, result = prepare(source, bag, item, cfg, contract)
    assert result.timeline[[0, -1]].tolist() == [10_200_000_000, 11_800_000_000]
