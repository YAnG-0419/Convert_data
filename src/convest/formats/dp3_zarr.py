"""Append-only, recoverable Zarr writer for synchronized uncropped DP3 data."""
from pathlib import Path
import json

from numcodecs import Blosc
import numpy as np
import zarr

from convest.config import atomic_json


SOURCE_COLUMNS = (
    "cam0_depth", "cam0_depth_info",
    "state.left_arm", "state.left_hand", "state.right_arm", "state.right_hand",
    "action.left_arm", "action.left_hand", "action.right_arm", "action.right_hand",
    "status.left_arm", "status.left_hand", "status.right_arm", "status.right_hand",
)


def _create(group, name, shape, chunks, dtype, compressor, fill_value=0):
    if name in group:
        array = group[name]
        expected_tail = tuple(shape[1:])
        if array.dtype != np.dtype(dtype) or array.shape[1:] != expected_tail:
            raise ValueError(
                f"Existing {array.path} has {array.shape}/{array.dtype}; "
                f"expected (*,{expected_tail})/{np.dtype(dtype)}"
            )
        return array
    return group.create_dataset(
        name, shape=shape, chunks=chunks, dtype=dtype, compressor=compressor,
        fill_value=fill_value,
    )


def open_dataset(root, config):
    path = Path(root) / "dataset_uncropped.zarr"
    store = zarr.open_group(str(path), mode="a")
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    data, times, meta = store.require_group("data"), store.require_group("time"), store.require_group("meta")
    dimension = config["state_dim"]
    _create(data, "point_cloud_xyz", (0, 3), (262144, 3), "f4", compressor)
    _create(data, "point_cloud_offsets", (1,), (4096,), "i8", compressor)
    _create(data, "state", (0, dimension), (1024, dimension), "f4", compressor)
    _create(data, "action", (0, dimension), (1024, dimension), "f4", compressor)
    _create(times, "timestamp_ns", (0,), (4096,), "i8", compressor)
    _create(times, "source_header_ns", (0, len(SOURCE_COLUMNS)), (1024, len(SOURCE_COLUMNS)), "i8", compressor, -1)
    _create(times, "source_receive_ns", (0, len(SOURCE_COLUMNS)), (1024, len(SOURCE_COLUMNS)), "i8", compressor, -1)
    _create(meta, "episode_ends", (0,), (1024,), "i8", compressor)
    if data["point_cloud_offsets"].shape == (1,):
        data["point_cloud_offsets"][0] = 0
    transform = np.asarray(config["T_point_from_depth_camera"], dtype=np.float64)
    transformed = (config["source_point_frame"] != config["point_frame"]
                   or not np.allclose(transform, np.eye(4), atol=1e-8))
    store.attrs.update(
        schema="bimanual_dp3_uncropped_xyz/v1",
        stage="coordinate_transformed_uncropped" if transformed else "camera_native_uncropped",
        target_hz=config["fps"], point_unit="m", point_frame=config["point_frame"],
        source_point_frame=config["source_point_frame"], depth_scale_m=config["depth_scale_m"],
        T_point_from_depth_camera=config["T_point_from_depth_camera"],
        calibration_version=config["calibration_version"], timestamp_columns=list(SOURCE_COLUMNS),
        point_cloud_layout="packed_xyz_with_frame_offsets", point_cloud_channels=3,
        crop_min=None, crop_max=None, voxel_size_m=None, pixel_stride=1,
        timestamp_policy="validated_header_time_causal_asof",
        state_dim=dimension, action_dim=dimension, coordinate_transform_applied=transformed,
        crop_applied=False, sampling="none", num_points=None, training_ready=False,
    )
    return store


def _resize(array, shape):
    array.resize(tuple(shape))


def recover(root, records, config):
    store = open_dataset(root, config)
    episodes = [episode for record in records for episode in record["episodes"]]
    frames = sum(episode["length"] for episode in episodes)
    points = sum(episode["point_count"] for episode in episodes)
    data, times, meta = store["data"], store["time"], store["meta"]
    _resize(data["point_cloud_xyz"], (points, 3))
    _resize(data["point_cloud_offsets"], (frames + 1,))
    for name in ("state", "action"):
        _resize(data[name], (frames, config["state_dim"]))
    _resize(times["timestamp_ns"], (frames,))
    for name in ("source_header_ns", "source_receive_ns"):
        _resize(times[name], (frames, len(SOURCE_COLUMNS)))
    _resize(meta["episode_ends"], (len(episodes),))
    store.attrs.update(frames=frames, points=points, episodes=len(episodes), conversion_complete=False)
    return store


def _append(array, values):
    values = np.asarray(values, dtype=array.dtype)
    start = array.shape[0]
    _resize(array, (start + len(values), *array.shape[1:]))
    array[start:] = values


def _vectors(aligned, start, end):
    state = aligned.state[start:end]
    action = aligned.action[start:end]
    # Canonical DP3 order: left arm, left hand, right arm, right hand.
    state = np.concatenate((state[:, 0:7], state[:, 28:48], state[:, 14:21], state[:, 68:88]), axis=1)
    action = np.concatenate((action[:, 0:7], action[:, 14:34], action[:, 7:14], action[:, 34:54]), axis=1)
    return state.astype(np.float32), action.astype(np.float32)


def _source_times(aligned, streams, start, end, receive=False):
    result = np.full((end - start, len(SOURCE_COLUMNS)), -1, dtype=np.int64)
    for column, key in enumerate(SOURCE_COLUMNS):
        indices = aligned.source_indices[key][start:end]
        valid = indices >= 0
        source = streams[key].receive_times if receive else streams[key].times
        if source is not None:
            result[valid, column] = source[indices[valid]]
    return result


def point_cloud(depth, info, config):
    if depth.ndim != 2:
        raise ValueError("Depth must be a single-channel image")
    if (int(info.height), int(info.width)) != depth.shape:
        raise ValueError("Depth image and CameraInfo dimensions differ")
    distortion = np.asarray(info.d, dtype=np.float64)
    if distortion.size and not np.allclose(distortion, 0, atol=1e-9):
        raise ValueError("Raw distorted depth is unsupported; provide rectified depth/CameraInfo")
    intrinsics = np.asarray(info.k, dtype=np.float64).reshape(3, 3)
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    if not np.isfinite(intrinsics).all() or min(fx, fy) <= 0:
        raise ValueError("Invalid depth intrinsics")
    # ROS 32FC1 depth is already metres; the configured scale applies to integer depth.
    scale = config["depth_scale_m"] if np.issubdtype(depth.dtype, np.integer) else 1.0
    z = depth.astype(np.float32) * np.float32(scale)
    valid = np.isfinite(z) & (z >= config["min_depth_m"]) & (z <= config["max_depth_m"])
    v, u = np.nonzero(valid)
    z = z[valid]
    if not len(z):
        raise ValueError("Depth frame has no valid points")
    xyz = np.column_stack(((u - cx) * z / fx, (v - cy) * z / fy, z)).astype(np.float32)
    transform = np.asarray(config["T_point_from_depth_camera"], dtype=np.float32)
    return (xyz @ transform[:3, :3].T + transform[:3, 3]).astype(np.float32)


def append_segment(store, bag, streams, aligned, bounds, config, contract, source_bag):
    del contract
    start, end = bounds
    state, action = _vectors(aligned, start, end)
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("State/action contains NaN or infinity")
    max_step = max(float(np.abs(np.diff(state, axis=0)).max(initial=0)),
                   float(np.abs(np.diff(action, axis=0)).max(initial=0)))
    if max_step > config["max_joint_step_rad"]:
        raise ValueError(f"Joint jump {max_step:.6f} exceeds {config['max_joint_step_rad']} rad/tick")

    data, times, meta = store["data"], store["time"], store["meta"]
    first_frame, first_point = data["state"].shape[0], data["point_cloud_xyz"].shape[0]
    offsets = []
    cached_index = None
    cached_points = None
    for output_index in range(start, end):
        depth_index = aligned.image_indices["cam0_depth"][output_index]
        info_index = aligned.image_indices["cam0_depth_info"][output_index]
        cache_key = (int(depth_index), int(info_index))
        if cache_key != cached_index:
            depth = bag.depth(streams["cam0_depth"].values[depth_index])
            info = streams["cam0_depth_info"].values[info_index]
            cached_points = point_cloud(depth, info, config)
            cached_index = cache_key
        _append(data["point_cloud_xyz"], cached_points)
        offsets.append(data["point_cloud_xyz"].shape[0])

    _append(data["point_cloud_offsets"], np.asarray(offsets, dtype=np.int64))
    _append(data["state"], state)
    _append(data["action"], action)
    _append(times["timestamp_ns"], aligned.timeline[start:end])
    _append(times["source_header_ns"], _source_times(aligned, streams, start, end))
    _append(times["source_receive_ns"], _source_times(aligned, streams, start, end, receive=True))
    _append(meta["episode_ends"], [first_frame + end - start])
    return {
        "length": end - start, "point_count": data["point_cloud_xyz"].shape[0] - first_point,
        "source_bag": source_bag, "source_start_ns": int(aligned.timeline[start]),
        "source_end_ns": int(aligned.timeline[end - 1]), "max_joint_step_rad": max_step,
        "paths": ["dataset_uncropped.zarr"],
    }


def joint_names(contract):
    names = contract["contract_joint_names"]
    return names["left_arm"] + names["left_hand"] + names["right_arm"] + names["right_hand"]


def finalize(root, records, config, contract, quality, complete=True):
    store = recover(root, records, config)
    names = joint_names(contract)
    store.attrs.update(
        quality=quality, state_names=names, action_names=names,
        state_unit="rad", action_unit="rad", action_semantics="absolute_joint_target",
        task=config["task"],
        conversion_complete=complete, ready_for_cropping=complete, training_ready=False,
    )
    episodes = [episode for record in records for episode in record["episodes"]]
    dataset = {
        "schema": store.attrs["schema"], "quality": quality, "frames": store.attrs["frames"],
        "points": store.attrs["points"], "episodes": len(episodes),
        "point_frame": config["point_frame"], "state_dim": config["state_dim"],
        "action_dim": config["state_dim"], "ready_for_cropping": complete,
    }
    root = Path(root)
    atomic_json(root / "conversion/dataset.json", dataset)
    internal = json.loads((root / "conversion/manifest.json").read_text())
    public = dict(store.attrs)
    public.update(
        source_bags=[Path(record["source"]).name for record in records],
        source_checksums={Path(record["source"]).name: record.get("source_sha256", {}) for record in records},
        source_topic_contract=contract["topics"], converter_git_commit=internal.get("converter_git_commit"),
    )
    atomic_json(root / "manifest.json", public)
    atomic_json(root / "episode_manifest.json", episodes)
    atomic_json(root / "invalid_segments.json", [{
        "source_bag": Path(record["source"]).name,
        "invalid_frames": record["alignment"]["invalid_frames"],
        "short_segment_frames": record["alignment"]["short_segment_frames"],
        "unmatched": record["alignment"]["unmatched"],
    } for record in records])
    atomic_json(root / "conversion_report.json", {
        **dataset,
        "conversion_complete": complete,
        "bags": [{
            "source_bag": Path(record["source"]).name,
            "source_recording_id": record.get("source_recording_id"),
            "alignment": record["alignment"],
            "elapsed_seconds": record["elapsed_seconds"],
            "output_segments": len(record["episodes"]),
        } for record in records],
    })


def verify(root, full_video=False):
    del full_video
    root = Path(root)
    manifest = json.loads((root / "conversion/manifest.json").read_text())
    config = manifest["config"]
    records = [json.loads(path.read_text()) for path in sorted((root / "conversion/records").glob("*.json"))]
    store = zarr.open_group(str(root / "dataset_uncropped.zarr"), mode="r")
    frames = sum(ep["length"] for record in records for ep in record["episodes"])
    points = sum(ep["point_count"] for record in records for ep in record["episodes"])
    offsets = store["data/point_cloud_offsets"][:]
    ends = store["meta/episode_ends"][:]
    if offsets.shape != (frames + 1,) or offsets[0] != 0 or offsets[-1] != points or not np.all(np.diff(offsets) > 0):
        raise ValueError("Invalid point_cloud_offsets")
    expected_ends = np.cumsum([ep["length"] for record in records for ep in record["episodes"]])
    if not np.array_equal(ends, expected_ends):
        raise ValueError("Invalid episode_ends")
    dimension = config["state_dim"]
    expected = {
        "data/point_cloud_xyz": ((points, 3), np.float32),
        "data/state": ((frames, dimension), np.float32),
        "data/action": ((frames, dimension), np.float32),
        "time/timestamp_ns": ((frames,), np.int64),
        "time/source_header_ns": ((frames, len(SOURCE_COLUMNS)), np.int64),
        "time/source_receive_ns": ((frames, len(SOURCE_COLUMNS)), np.int64),
    }
    for key, (shape, dtype) in expected.items():
        array = store[key]
        if array.shape != shape or array.dtype != np.dtype(dtype):
            raise ValueError(f"{key}: expected {shape}/{np.dtype(dtype)}, got {array.shape}/{array.dtype}")
        if np.issubdtype(array.dtype, np.floating):
            for start in range(0, len(array), max(1, array.chunks[0])):
                if not np.isfinite(array[start:start + array.chunks[0]]).all():
                    raise ValueError(f"{key} contains NaN/inf")
    start = 0
    timestamps = store["time/timestamp_ns"][:]
    for end in ends:
        if end - start > 1 and not np.all(np.diff(timestamps[start:end]) > 0):
            raise ValueError("Timestamps must increase strictly within each episode")
        start = int(end)
    result = {"status": "passed", "target_format": "dp3_uncropped_zarr",
              "quality": store.attrs["quality"], "episodes": len(ends), "frames": frames, "points": points}
    atomic_json(root / "conversion/verification.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result
