"""Compare sampled converted rows and decoded RGB against their original bag messages."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import av
import numpy as np
import pyarrow.parquet as pq
import yaml

from convest.config import atomic_json
from convest.sources.gello_rosbag2 import Bag, read_streams
from convest.align import GROUPS


def decoded_at(path, frame_index, fps):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        timestamp = frame_index / fps
        container.seek(int(timestamp / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            if float(frame.pts * frame.time_base) >= timestamp - 1e-5:
                assert abs(float(frame.pts * frame.time_base) - timestamp) < 1e-5
                return frame.to_ndarray(format="rgb24")
        raise ValueError("Requested video frame not found")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--bags", type=int, default=3)
    args = parser.parse_args()
    config = json.loads((args.root / "conversion/manifest.json").read_text())["config"]
    contract = yaml.safe_load(Path(config["contract"]).read_text())
    info = json.loads((args.root / "meta/info.json").read_text())
    files = sorted((args.root / "conversion/records").glob("*.json"))
    results = []
    for position in np.unique(np.linspace(0, len(files) - 1, min(len(files), args.bags), dtype=int)):
        record = json.loads(files[position].read_text())
        with Bag(record["source"], config["schema_dir"]) as bag:
            streams = read_streams(bag, contract)
            for ep in record["episodes"]:
                fmt = {"episode_index": ep["episode_index"], "episode_chunk": ep["episode_index"] // info["chunks_size"]}
                table = pq.read_table(args.root / info["data_path"].format(**fmt))
                for row_index in sorted(set([0, len(table) // 2, len(table) - 1])):
                    row = table.slice(row_index, 1).to_pylist()[0]
                    t = row["source_timestamp_ns"]
                    expected_state = []
                    for group in GROUPS:
                        s = streams[f"state.{group}"]
                        index = int(np.searchsorted(s.times, t, side="right")) - 1
                        assert index >= 0
                        expected_state.extend(s.values[index])
                    np.testing.assert_array_equal(np.asarray(row["observation.state"], dtype=np.float32), expected_state)
                    # Engaged groups must match the last named absolute command exactly.
                    offset = 0
                    for group, dim, engaged in zip(GROUPS, (7, 7, 20, 20), row["observation.engaged"]):
                        if engaged:
                            s = streams[f"action.{group}"]
                            index = int(np.searchsorted(s.times, t, side="right")) - 1
                            np.testing.assert_array_equal(np.asarray(row["action"][offset:offset + dim], dtype=np.float32), s.values[index])
                        offset += dim
                    maes = {}
                    for camera in ("cam0", "cam1", "cam2"):
                        s = streams[camera]
                        index = int(np.searchsorted(s.times, t, side="right")) - 1
                        source = bag.image(s.values[index])
                        relative = info["video_path"].format(**fmt, video_key=f"observation.images.{camera}")
                        rgb = decoded_at(args.root / relative, row_index, info["fps"])
                        mae = float(np.abs(rgb.astype(np.int16) - source.astype(np.int16)).mean())
                        if mae >= 15:
                            raise AssertionError(f"Large source/video difference: {camera}, row={row_index}, MAE={mae}")
                        maes[camera] = round(mae, 4)
                    results.append({"bag": record["source"], "episode_index": ep["episode_index"],
                                    "frame_index": row_index, "vectors_exact": True, "rgb_mae_255": maes})
        print(f"Source comparison passed: {Path(record['source']).name}", flush=True)
    atomic_json(args.root / "conversion/source_verification.json", {"status": "passed", "samples": results,
                "scope": "sampled states, engaged absolute actions and source-aligned RGB; inactive holds covered by unit tests"})


if __name__ == "__main__":
    main()
