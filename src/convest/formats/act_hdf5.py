"""ACT episode HDF5 writer and streaming schema/quality verification."""
from pathlib import Path
import json
import shutil

import av
import h5py
import numpy as np

from convest.config import atomic_json
from convest.recipes.act import CAMERAS, group_names


def resize_rgb(rgb, width, height):
    """Centered black padding to target aspect ratio, followed by area resize.

    640x400 becomes 640x480 with 40 black rows on each side; wrist images
    at 640x480 pass directly to resize. All operations keep RGB channel order.
    """
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected RGB uint8 HxWx3 image")
    h, w = rgb.shape[:2]
    if h < 1 or w < 1:
        raise ValueError("Empty image")
    padded_h = max(h, (w * height + width - 1) // width)
    padded_w = max(w, (h * width + height - 1) // height)
    top, left = (padded_h - h) // 2, (padded_w - w) // 2
    if (padded_h, padded_w) != (h, w):
        rgb = np.pad(rgb, ((top, padded_h - h - top), (left, padded_w - w - left), (0, 0)))
    frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")
    return frame.reformat(width=width, height=height, format="rgb24", interpolation="AREA").to_ndarray(format="rgb24")


def check_vectors(qpos, qvel, action, max_step):
    for name, values in (("qpos", qpos), ("qvel", qvel), ("action", action)):
        if not np.isfinite(values).all():
            raise ValueError(f"ACT {name} contains NaN/inf")
    if not np.any(action):
        raise ValueError("ACT episode has all-zero action")
    steps = {name: float(np.max(np.abs(np.diff(values, axis=0))))
             for name, values in (("qpos", qpos), ("action", action))}
    if any(step > max_step for step in steps.values()):
        raise ValueError(f"ACT joint jump exceeds {max_step} rad/timestep: {steps}")
    return steps


def write_segment(root, bag, streams, aligned, bounds, ep_index, global_index, config, contract,
                  segment=None):
    a, b = bounds
    length, dimension = b - a, config["state_dim"]
    qpos, qvel, action = aligned.qpos[a:b], aligned.qvel[a:b], aligned.action[a:b]
    for values in (qpos, qvel, action):
        if values.shape != (length, dimension):
            raise ValueError("ACT vector shape does not match configured dimension")
    steps = check_vectors(qpos, qvel, action, config["max_joint_step_rad"])
    relative = f"episode_{ep_index}.hdf5"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    names = {"qpos": sum(group_names(contract, dimension), []),
             "action": sum(group_names(contract, dimension, action=True), [])}
    shape = [config["image_height"], config["image_width"], 3]
    features = {"state_dim": dimension, "action_dim": dimension, "joint_names": names,
                "cameras": {name: shape for name in CAMERAS.values()}, "color": "RGB", "dtype": "float32"}
    with h5py.File(path, "w") as f:
        f.attrs.update(sim=False, compress=False, fps=config["fps"], episode_index=ep_index,
                       source_bag=str(bag.path), color="RGB", joint_names_json=json.dumps(names))
        observations = f.create_group("observations")
        observations.create_dataset("qpos", data=qpos, dtype="f4")
        observations.create_dataset("qvel", data=qvel, dtype="f4")
        f.create_dataset("action", data=action, dtype="f4")
        f.create_dataset("source_timestamp_ns", data=aligned.timeline[a:b], dtype="i8")
        images = observations.create_group("images")
        for camera, name in CAMERAS.items():
            dataset = images.create_dataset(name, (length, *shape), dtype="u1", chunks=(1, *shape),
                                            compression=config["compression"])
            last, rgb, source_shape = None, None, None
            for frame_index, index in enumerate(aligned.image_indices[camera][a:b]):
                if frame_index % 100 == 0 and shutil.disk_usage(root).free < config["min_free_gb"] * 1e9:
                    raise OSError("Available disk space below configured reserve")
                if index != last:
                    raw = bag.image(streams[camera].values[index])
                    if source_shape is not None and raw.shape != source_shape:
                        raise ValueError(f"{camera}: source resolution changed within episode")
                    source_shape = raw.shape
                    if not np.any(raw):
                        raise ValueError(f"{camera}: all-black source frame at output frame {frame_index}")
                    rgb = resize_rgb(raw, config["image_width"], config["image_height"])
                    last = index
                dataset[frame_index] = rgb
    return {"episode_index": ep_index, "length": length, "tasks": [config["task"]],
            "source_start_ns": int(aligned.timeline[a]), "source_end_ns": int(aligned.timeline[b - 1]),
            "features": features, "quality": {"max_joint_step_rad": steps}, "paths": [relative]}


def reindex_segment(stage, episode, episode_index, global_index, task_index=0):
    old = stage / episode["paths"][0]
    relative = f"episode_{episode_index}.hdf5"
    final = stage / "final" / relative
    final.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(old, "r+") as f:
        f.attrs["episode_index"] = episode_index
    old.replace(final)
    episode.update(episode_index=episode_index, paths=[relative])


def write_metadata(root, records, config):
    episodes = [ep for record in records for ep in record["episodes"]]
    atomic_json(root / "conversion/dataset.json", {
        "target_format": "act_hdf5", "fps": config["fps"], "task": config["task"],
        "total_episodes": len(episodes), "total_frames": sum(ep["length"] for ep in episodes),
        "features": episodes[0]["features"] if episodes else None})


def verify(root, full_video=False):
    # HDF5 images are always fully read in bounded blocks; full_video is a CLI compatibility argument.
    root = Path(root)
    manifest = json.loads((root / "conversion/manifest.json").read_text())
    config = manifest["config"]
    if manifest.get("owner") != "convest-data" or config["target_format"] != "act_hdf5":
        raise ValueError("Not an owned ACT dataset")
    records = [json.loads(p.read_text()) for p in sorted((root / "conversion/records").glob("*.json"))]
    episodes = [ep for record in records for ep in record["episodes"]]
    if not episodes:
        raise ValueError("ACT dataset contains no committed episodes")
    expected = {f"episode_{i}.hdf5" for i in range(len(episodes))}
    if {p.name for p in root.glob("episode_*.hdf5")} != expected:
        raise ValueError("Missing or uncommitted ACT episode files")
    total = 0
    reference = episodes[0]["features"]
    for index, ep in enumerate(episodes):
        length, dimension = ep["length"], config["state_dim"]
        if ep["episode_index"] != index or ep["features"] != reference or ep["paths"] != [f"episode_{index}.hdf5"]:
            raise ValueError("Inconsistent ACT episode record/schema")
        with h5py.File(root / ep["paths"][0], "r") as f:
            if bool(f.attrs.get("sim", True)) or bool(f.attrs.get("compress", True)):
                raise ValueError("ACT requires sim=False and unencoded RGB arrays")
            if f.attrs.get("color") != "RGB" or f.attrs.get("fps") != config["fps"] or f.attrs.get("episode_index") != index:
                raise ValueError("ACT attributes mismatch")
            if json.loads(f.attrs["joint_names_json"]) != reference["joint_names"]:
                raise ValueError("ACT joint order changed")
            vectors = []
            for key in ("observations/qpos", "observations/qvel", "action"):
                d = f[key]
                if d.shape != (length, dimension) or d.dtype not in (np.dtype("float32"), np.dtype("float64")):
                    raise ValueError(f"ACT invalid shape/dtype: {key}")
                vectors.append(d[:])
            check_vectors(*vectors, config["max_joint_step_rad"])
            times = f["source_timestamp_ns"][:]
            if (times.shape != (length,) or times.dtype != np.int64 or length < 2
                    or times[0] != ep["source_start_ns"] or times[-1] != ep["source_end_ns"]
                    or not np.all(np.abs(np.diff(times) - 1e9 / config["fps"]) <= 1)):
                raise ValueError("ACT timestamps contain a gap or mismatch")
            for camera in CAMERAS.values():
                d = f[f"observations/images/{camera}"]
                if d.shape != (length, config["image_height"], config["image_width"], 3) or d.dtype != np.uint8:
                    raise ValueError(f"ACT invalid image shape/dtype: {camera}")
                for start in range(0, length, 16):
                    if not np.all(np.any(d[start:start + 16], axis=(1, 2, 3))):
                        raise ValueError(f"ACT all-black frame: {camera}")
        total += length
    info = json.loads((root / "conversion/dataset.json").read_text())
    if info["total_episodes"] != len(episodes) or info["total_frames"] != total or info["features"] != reference:
        raise ValueError("ACT dataset metadata mismatch")
    result = {"status": "passed", "target_format": "act_hdf5", "episodes": len(episodes),
              "frames": total, "full_image_decode": True}
    atomic_json(root / "conversion/verification.json", result)
    print(json.dumps(result), flush=True)
    return result
