import json

import numpy as np
import yaml

from convest.align import align
from convest.config import WORKSPACE, load_config
from convest.formats.lerobot_v21 import write_metadata, write_segment, reindex_segment
from convest.verify import verify
from test_alignment import streams


def test_writer_roundtrip_all_video_timestamps_and_statistics(tmp_path):
    raw = streams()
    aligned = align(raw, 1_000_000_000, 3_000_000_000, 30, 150_000_000)

    class Images:
        @staticmethod
        def image(index):
            image = np.zeros((16, 24, 3), dtype=np.uint8)
            image[:, :, 0] = index * 10
            image[:, :, 1] = 77
            return image

    config = load_config(WORKSPACE / "configs/gello_pi05.yaml")
    contract = yaml.safe_load((WORKSPACE / "schemas/gello_contract.yaml").read_text())
    result = write_segment(tmp_path, Images(), raw, aligned, aligned.segments[0], 0, 0, config, contract)
    write_metadata(tmp_path, [{"episodes": [result]}], config)
    report = verify(tmp_path, full_video=True)
    assert report["frames"] == 61 and report["episodes"] == 1
    info = json.loads((tmp_path / "meta/info.json").read_text())
    assert info["features"]["observation.images.cam0"]["shape"] == [16, 24, 3]
    reindex_segment(tmp_path, result, 1001, 1234)
    import pyarrow.parquet as pq
    data = pq.read_table(tmp_path / "final/data/chunk-001/episode_001001.parquet")
    assert data["episode_index"].to_pylist() == [1001] * 61
    assert data["index"].to_pylist() == list(range(1234, 1295))
    assert result["stats"]["index"]["min"] == [1234]
