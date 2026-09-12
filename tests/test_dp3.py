from types import SimpleNamespace

import numpy as np
import pytest
import yaml
import zarr

from convest.config import WORKSPACE, atomic_json, load_config
from convest import dp3_pipeline
from convest.formats import dp3_zarr
from convest.recipes.dp3 import validate_config
from convest.sources.gello_rosbag2 import Series
from test_pipeline import make_bag


def config():
    return {
        "window": "validated", "state_dim": 54, "fps": 10,
        "max_staleness_ms": 150, "min_segment_frames": 2,
        "depth_scale_m": 0.001, "min_depth_m": 0.001, "max_depth_m": 10.0,
        "max_joint_step_rad": 1.0, "source_point_frame": "depth",
        "point_frame": "workcell", "calibration_version": "test-v1",
        "T_point_from_depth_camera": [
            [1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1],
        ],
        "task": "test task",
    }


def test_uncropped_point_geometry_order_and_recovery(tmp_path):
    cfg = config()
    validate_config(cfg)
    contract = yaml.safe_load((WORKSPACE / "schemas/gello_contract.yaml").read_text())
    times = np.array([1_000_000_000, 1_100_000_000], dtype=np.int64)
    receive = times + 5_000_000
    info = SimpleNamespace(height=2, width=2, d=[], k=[1, 0, 0, 0, 1, 0, 0, 0, 1])
    streams = {
        key: Series(times, [0, 1], receive)
        for key in dp3_zarr.SOURCE_COLUMNS
    }
    streams["cam0_depth"].values = ["depth0", "depth1"]
    streams["cam0_depth_info"].values = [info, info]

    state = np.zeros((2, 108), dtype=np.float32)
    state[:, 0:7], state[:, 14:21] = 1, 2
    state[:, 28:48], state[:, 68:88] = 3, 4
    action = np.zeros((2, 54), dtype=np.float32)
    action[:, 0:7], action[:, 7:14] = 11, 12
    action[:, 14:34], action[:, 34:54] = 13, 14
    aligned = SimpleNamespace(
        timeline=times, state=state, action=action,
        image_indices={"cam0_depth": np.array([0, 1]), "cam0_depth_info": np.array([0, 1])},
        source_indices={key: np.array([0, 1]) for key in dp3_zarr.SOURCE_COLUMNS},
    )

    class FakeBag:
        def depth(self, reference):
            assert reference in ("depth0", "depth1")
            return np.array([[1000, 0], [2000, 1000]], dtype=np.uint16)

    store = dp3_zarr.open_dataset(tmp_path, cfg)
    episode = dp3_zarr.append_segment(
        store, FakeBag(), streams, aligned, (0, 2), cfg, contract, "/bags/episode1"
    )
    records = [{
        "source": "/bags/episode1", "source_sha256": {}, "source_recording_id": "recording-1",
        "alignment": {"invalid_frames": 0, "short_segment_frames": 0, "unmatched": {}},
        "elapsed_seconds": 0.1, "episodes": [episode],
    }]
    atomic_json(tmp_path / "conversion/manifest.json", {"config": cfg})
    atomic_json(tmp_path / "conversion/records/000000.json", records[0])
    dp3_zarr.finalize(tmp_path, records, cfg, contract, "high")
    assert dp3_zarr.verify(tmp_path)["frames"] == 2

    result = zarr.open_group(str(tmp_path / "dataset_uncropped.zarr"), mode="r")
    np.testing.assert_array_equal(result["data/point_cloud_offsets"][:], [0, 3, 6])
    np.testing.assert_allclose(result["data/point_cloud_xyz"][:3], [
        [1, 2, 4], [1, 4, 5], [2, 3, 4],
    ])
    np.testing.assert_array_equal(result["data/state"][0], [*([1] * 7), *([3] * 20), *([2] * 7), *([4] * 20)])
    np.testing.assert_array_equal(result["data/action"][0], [*([11] * 7), *([13] * 20), *([12] * 7), *([14] * 20)])
    np.testing.assert_array_equal(result["time/source_receive_ns"][:, 0], receive)

    # Simulate interruption after arrays grew but before a journal record committed.
    writable = dp3_zarr.open_dataset(tmp_path, cfg)
    dp3_zarr.append_segment(writable, FakeBag(), streams, aligned, (0, 2), cfg, contract, "/bags/episode2")
    recovered = dp3_zarr.recover(tmp_path, records, cfg)
    assert recovered["data/state"].shape == (2, 54)
    assert recovered["data/point_cloud_xyz"].shape == (6, 3)


def test_nonidentity_frame_requires_calibration_transform():
    cfg = config()
    cfg["T_point_from_depth_camera"] = np.eye(4).tolist()
    with pytest.raises(ValueError, match="actual calibrated transform"):
        validate_config(cfg)


def test_float_depth_is_already_metres():
    info = SimpleNamespace(height=1, width=1, d=[], k=[1, 0, 0, 0, 1, 0, 0, 0, 1])
    np.testing.assert_allclose(
        dp3_zarr.point_cloud(np.array([[1.0]], dtype=np.float32), info, config()),
        [[1, 2, 4]],
    )


def test_quality_pipeline_real_cdr_resume_and_public_manifests(tmp_path, monkeypatch):
    source, output = tmp_path / "bags", tmp_path / "output"
    make_bag(source / "episode9")
    episode_list = tmp_path / "high.txt"
    episode_list.write_text("9\n")
    cfg = load_config(WORKSPACE / "configs/gello_dp3_sample.yaml")
    cfg.update(
        source_root=str(source), output_root=str(output), min_free_gb=0,
        quality_groups={"high": str(episode_list)},
    )
    monkeypatch.setattr(dp3_pipeline, "WORKSPACE", tmp_path)
    before = dp3_pipeline.get_source(cfg["source_format"]).snapshot(source / "episode9")
    assert dp3_pipeline.convert_quality(cfg, "high") == 0
    root = output / "high"
    assert dp3_zarr.verify(root)["episodes"] == 1
    assert (root / "manifest.json").is_file()
    assert (root / "episode_manifest.json").is_file()
    assert (root / "conversion_report.json").is_file()
    assert dp3_pipeline.convert_quality(cfg, "high", resume=True) == 0
    assert len(dp3_pipeline.load_records(root)) == 1
    assert before == dp3_pipeline.get_source(cfg["source_format"]).snapshot(source / "episode9")
