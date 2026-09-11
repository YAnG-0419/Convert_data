from pathlib import Path
import json

import av
import numpy as np
import pyarrow.parquet as pq

from convest.config import atomic_json


def verify(root, full_video=False):
    root = Path(root)
    info = json.loads((root / "meta/info.json").read_text())
    assert info["codebase_version"] == "v2.1"
    episodes = [json.loads(s) for s in (root / "meta/episodes.jsonl").read_text().splitlines()]
    stats = [json.loads(s) for s in (root / "meta/episodes_stats.jsonl").read_text().splitlines()]
    tasks = {row["task_index"]: row["task"] for row in
             (json.loads(s) for s in (root / "meta/tasks.jsonl").read_text().splitlines())}
    assert len(episodes) == len(stats) == info["total_episodes"]
    assert len(tasks) == info["total_tasks"] and sorted(tasks) == list(range(len(tasks)))
    assert info["features"]["observation.state"]["shape"] == [108]
    assert info["features"]["action"]["shape"] == [54]
    total = 0
    for index, episode in enumerate(episodes):
        assert episode["episode_index"] == stats[index]["episode_index"] == index
        args = {"episode_index": index, "episode_chunk": index // info["chunks_size"]}
        table = pq.read_table(root / info["data_path"].format(**args))
        length = len(table)
        assert length == episode["length"]
        task_index = episode.get("task_index", 0)
        assert tasks[task_index] == episode["tasks"][0]
        for key, expected in (("frame_index", np.arange(length)), ("index", np.arange(total, total + length)),
                              ("episode_index", np.full(length, index)),
                              ("task_index", np.full(length, task_index))):
            np.testing.assert_array_equal(table[key].to_numpy(), expected)
        np.testing.assert_allclose(table["timestamp"].to_numpy(), np.arange(length) / info["fps"], atol=1e-5, rtol=1e-7)
        delta = np.diff(table["source_timestamp_ns"].to_numpy())
        assert np.all(np.abs(delta - 1e9 / info["fps"]) <= 1)
        for key in ("observation.state", "action", "observation.engaged"):
            values = np.asarray(table[key].to_pylist(), dtype=np.float32)
            assert values.shape == (length, info["features"][key]["shape"][0]) and np.isfinite(values).all()
            np.testing.assert_allclose(values.mean(axis=0, dtype=np.float64), stats[index]["stats"][key]["mean"], atol=1e-6)
        for key, feature in info["features"].items():
            if feature["dtype"] != "video":
                continue
            with av.open(str(root / info["video_path"].format(**args, video_key=key))) as container:
                stream = container.streams.video[0]
                assert stream.frames == length and float(stream.average_rate) == info["fps"]
                assert [stream.height, stream.width, 3] == feature["shape"]
                if full_video:
                    count = 0
                    for frame in container.decode(stream):
                        assert abs(float(frame.pts * frame.time_base) - count / info["fps"]) < 1e-6
                        count += 1
                    assert count == length
                else:
                    first = next(container.decode(stream))
                    assert float(first.pts * first.time_base) == 0
        total += length
    assert total == info["total_frames"]
    result = {"status": "passed", "episodes": len(episodes), "frames": total, "full_video_decode": full_video}
    atomic_json(root / "conversion/verification.json", result)
    print(json.dumps(result), flush=True)
    return result
