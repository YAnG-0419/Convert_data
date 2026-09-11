"""Standard LeRobot v2.1 Parquet/MP4 writer, without a torch/ROS dependency."""
from fractions import Fraction
import json

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from convest.config import atomic_json
from convest.align import CAMERAS

DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


def feature_schema(contract):
    features = {}
    for key in ("observation.state", "action"):
        spec = contract["features"][key]
        names = []
        for group in spec["order"]:
            field = group["name"].split(".")[-1]
            names.extend(f"{joint}.{field}" for joint in group["joints"])
        features[key] = {"dtype": "float32", "shape": [spec["dimension"]], "names": names}
    features["observation.engaged"] = {"dtype": "float32", "shape": [4], "names": ["left_arm", "right_arm", "left_hand", "right_hand"]}
    for key in ("timestamp", "frame_index", "episode_index", "index", "task_index", "source_timestamp_ns"):
        features[key] = {"dtype": "float32" if key == "timestamp" else "int64", "shape": [1], "names": None}
    return features


def array_stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {**{k: np.atleast_1d(v).tolist() for k, v in {
        "min": values.min(axis=0), "max": values.max(axis=0),
        "mean": values.mean(axis=0), "std": values.std(axis=0)}.items()}, "count": [len(values)]}


def write_video(path, images, config, expected_count):
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = []
    # Bounded and deterministic RGB statistics, matching the v2.1 Cx1x1 convention.
    sample_indices = set(np.linspace(0, expected_count - 1, min(100, expected_count), dtype=int).tolist())
    shape = None
    count = 0
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=config["fps"])
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(config["video_crf"]), "preset": config["video_preset"]}
        stream.codec_context.thread_count = config["video_threads"]
        stream.codec_context.gop_size = config["fps"]
        for index, rgb in enumerate(images):
            if shape is None:
                shape = list(rgb.shape)
                stream.height, stream.width = shape[:2]
                if stream.height % 2 or stream.width % 2:
                    raise ValueError("H264 yuv420p requires even image dimensions")
            if list(rgb.shape) != shape:
                raise ValueError("Camera resolution changed within episode")
            if index in sample_indices:
                samples.append(rgb[::max(1, shape[0] // 64), ::max(1, shape[1] // 64)].reshape(-1, 3))
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, config["fps"])
            for packet in stream.encode(frame):
                container.mux(packet)
            count += 1
        for packet in stream.encode():
            container.mux(packet)
    if count != expected_count:
        raise ValueError(f"Video count mismatch: {count} != {expected_count}")
    stats = array_stats(np.concatenate(samples).astype(np.float64) / 255)
    stats = {k: np.asarray(v).reshape(3, 1, 1).tolist() if k != "count" else [len(samples)] for k, v in stats.items()}
    feature = {"dtype": "video", "shape": shape, "names": ["height", "width", "channel"],
               "info": {"video.height": shape[0], "video.width": shape[1], "video.codec": "h264",
                        "video.pix_fmt": "yuv420p", "video.fps": config["fps"], "video.channels": 3,
                        "video.is_depth_map": False, "has_audio": False}}
    return feature, stats


def write_segment(root, bag, streams, aligned, bounds, ep_index, global_index, config, contract,
                  segment=None):
    start, end = bounds
    length = end - start
    path_args = {"episode_chunk": ep_index // 1000, "episode_index": ep_index}
    columns = {"observation.state": aligned.state[start:end], "action": aligned.action[start:end],
               "observation.engaged": aligned.engaged[start:end],
               "timestamp": (np.arange(length) / config["fps"]).astype(np.float32),
               "frame_index": np.arange(length, dtype=np.int64),
               "episode_index": np.full(length, ep_index, dtype=np.int64),
               "index": np.arange(global_index, global_index + length, dtype=np.int64),
               "task_index": np.zeros(length, dtype=np.int64), "source_timestamp_ns": aligned.timeline[start:end]}
    features = feature_schema(contract)
    stats = {key: array_stats(value) for key, value in columns.items()}
    paths = []
    for camera in CAMERAS:
        key = f"observation.images.{camera}"
        relative = VIDEO_PATH.format(**path_args, video_key=key)

        def images():
            last_index, rgb = None, None
            for index in aligned.image_indices[camera][start:end]:
                if index != last_index:
                    rgb = bag.image(streams[camera].values[index])
                    last_index = index
                yield rgb

        features[key], stats[key] = write_video(root / relative, images(), config, length)
        paths.append(relative)
    arrays = {}
    for key, values in columns.items():
        arrays[key] = (pa.FixedSizeListArray.from_arrays(pa.array(values.ravel()), values.shape[1])
                       if values.ndim == 2 else pa.array(values))
    relative = DATA_PATH.format(**path_args)
    (root / relative).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(arrays), root / relative, compression="zstd")
    paths.append(relative)
    segment = segment or {"task": config["task"]}
    result = {"episode_index": ep_index, "length": length, "tasks": [segment["task"]],
            "task_index": 0,
            "source_start_ns": int(aligned.timeline[start]), "source_end_ns": int(aligned.timeline[end - 1]),
            "features": features, "stats": stats, "paths": paths}
    for key in ("segment_id", "source_recording_id", "milestone_timestamp_ns"):
        if key in segment:
            result[key] = segment[key]
    return result


def write_metadata(root, records, config):
    episodes = [ep for record in records for ep in record["episodes"]]
    if not episodes:
        return
    reference = episodes[0]["features"]
    for episode in episodes:
        if episode["features"] != reference:
            raise ValueError("Feature schema or camera resolution differs between episodes")
    task_rows = {}
    for episode in episodes:
        task_index = episode.get("task_index", 0)
        task = episode["tasks"][0]
        if task_index in task_rows and task_rows[task_index] != task:
            raise ValueError("A task index refers to multiple task strings")
        task_rows[task_index] = task
    if sorted(task_rows) != list(range(len(task_rows))):
        raise ValueError("Task indices must be contiguous from zero")
    info = {"codebase_version": "v2.1", "robot_type": "dual_fr3_wuji",
            "total_episodes": len(episodes), "total_frames": sum(e["length"] for e in episodes),
            "total_tasks": len(task_rows), "total_videos": 3 * len(episodes), "total_chunks": (len(episodes) + 999) // 1000,
            "chunks_size": 1000, "fps": config["fps"], "splits": {"train": f"0:{len(episodes)}"},
            "data_path": DATA_PATH, "video_path": VIDEO_PATH, "features": reference}

    def jsonl(name, rows):
        path = root / "meta" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        temp.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))
        temp.replace(path)

    jsonl("tasks.jsonl", [{"task_index": index, "task": task_rows[index]} for index in sorted(task_rows)])
    episode_keys = ("episode_index", "tasks", "length", "task_index", "segment_id",
                    "source_recording_id", "milestone_timestamp_ns", "source_start_ns", "source_end_ns")
    jsonl("episodes.jsonl", [{k: ep[k] for k in episode_keys if k in ep} for ep in episodes])
    jsonl("episodes_stats.jsonl", [{"episode_index": ep["episode_index"], "stats": ep["stats"]} for ep in episodes])
    atomic_json(root / "meta/info.json", info)


def reindex_segment(stage, episode, episode_index, global_index, task_index=0):
    """Assign final indices after a parallel bag finishes, preserving commit order."""
    old = {"episode_chunk": episode["episode_index"] // 1000, "episode_index": episode["episode_index"]}
    new = {"episode_chunk": episode_index // 1000, "episode_index": episode_index}
    old_data = DATA_PATH.format(**old)
    table = pq.read_table(stage / old_data)
    for key, value in (("episode_index", np.full(len(table), episode_index, dtype=np.int64)),
                       ("index", np.arange(global_index, global_index + len(table), dtype=np.int64)),
                       ("task_index", np.full(len(table), task_index, dtype=np.int64))):
        table = table.set_column(table.schema.get_field_index(key), key, pa.array(value))
        episode["stats"][key] = array_stats(value)
    # A separate final/ tree avoids collisions when local and global indices overlap.
    destination = stage / "final"
    new_data = DATA_PATH.format(**new)
    (destination / new_data).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, destination / new_data, compression="zstd")
    paths = [new_data]
    for camera in CAMERAS:
        key = f"observation.images.{camera}"
        old_video = VIDEO_PATH.format(**old, video_key=key)
        new_video = VIDEO_PATH.format(**new, video_key=key)
        (destination / new_video).parent.mkdir(parents=True, exist_ok=True)
        (stage / old_video).replace(destination / new_video)
        paths.append(new_video)
    episode["episode_index"], episode["paths"], episode["task_index"] = episode_index, paths, task_index
