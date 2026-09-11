"""End-to-end standard CDR/SQLite fixture, parallel commit, resume and immutability."""
from dataclasses import fields
import json
import shutil
import sqlite3

import numpy as np
import pytest
import yaml
import pyarrow.parquet as pq

from convest.config import WORKSPACE, load_config
from convest import pipeline
from convest.sources.gello_rosbag2 import Bag, snapshot
from convest.verify import verify


def make_bag(path):
    path.mkdir(parents=True)
    contract = yaml.safe_load((WORKSPACE / "schemas/gello_contract.yaml").read_text())
    store = Bag(path, WORKSPACE / "schemas").store
    types = store.types
    db = sqlite3.connect(path / "capture_0.db3")
    db.executescript("CREATE TABLE topics(id INTEGER PRIMARY KEY,name TEXT,type TEXT,serialization_format TEXT);"
                     "CREATE TABLE messages(id INTEGER PRIMARY KEY,topic_id INTEGER,timestamp INTEGER,data BLOB);")
    topic_ids = {}

    def append(topic, message, time):
        if topic not in topic_ids:
            ident = len(topic_ids) + 1
            topic_ids[topic] = ident
            db.execute("INSERT INTO topics VALUES(?,?,?,?)", (ident, topic, message.__msgtype__, "cdr"))
        db.execute("INSERT INTO messages(topic_id,timestamp,data) VALUES(?,?,?)",
                   (topic_ids[topic], time + 5000000, bytes(store.serialize_cdr(message, message.__msgtype__))))

    names = contract["source_joint_names"]
    for frame in range(21):
        time = 10_000_000_000 + frame * 100_000_000
        stamp = types["builtin_interfaces/msg/Time"](time // 1000000000, time % 1000000000)
        header = types["std_msgs/msg/Header"](stamp, "")

        def joints(group, value):
            # Reverse wire ordering to prove named reordering in the full source adapter.
            n = len(group)
            return types["sensor_msgs/msg/JointState"](header, list(reversed(group)),
                np.arange(n, dtype=np.float64)[::-1] + value,
                np.full(n, 0.1, dtype=np.float64), np.array([], dtype=np.float64))

        for side in ("left", "right"):
            for part in ("arm", "hand"):
                group = names[f"measured_{side}_arm" if part == "arm" else f"{side}_hand"]
                append(contract["topics"][f"{side}_{part}_state"]["topic"], joints(group, frame), time)
            append(contract["topics"][f"{side}_hand_action"]["topic"], joints(names[f"{side}_hand"], frame + 100), time)
        arm_names = names["validated_left_arm"] + names["validated_right_arm"]
        append(contract["topics"]["validated_arm_action"]["topic"], joints(arm_names, frame + 100), time)
        status = types["teleop_interfaces/msg/ArmCommandStatus"](header, "gello", "test", frame, ["left", "right"], [])
        append(contract["topics"]["arm_command_status"]["topic"], status, time)
        for side in ("left", "right"):
            cls = types["teleop_interfaces/msg/HandTelemetryStatus"]
            kwargs = {}
            for f in fields(cls):
                if f.name.startswith("__"):
                    continue
                kwargs[f.name] = {"int": 0, "float": 0.0, "str": "test", "bool": True}.get(f.type)
            kwargs.update(header=header, source_wall_stamp=stamp, side=side)
            append(contract["topics"]["hand_telemetry_status"]["topic"], cls(**kwargs), time)
        for camera in ("cam0", "cam1", "cam2"):
            rows = np.zeros((16, 76), dtype=np.uint8)
            rows[:, :72] = np.tile([frame * 5, 77, 123], (16, 24))
            image = types["sensor_msgs/msg/Image"](header, 16, 24, "bgr8", 0, 76, rows.ravel())
            append(contract["topics"][camera]["topic"], image, time)
    db.commit()
    db.close()
    (path / "metadata.yaml").write_text(yaml.safe_dump({"rosbag2_bagfile_information": {
        "storage_identifier": "sqlite3", "relative_file_paths": ["capture_0.db3"]}}))
    (path / "collection_state.json").write_text(json.dumps({"state": "finalized", "finalized": True, "failures": [],
        "validation_report": {"source_effective_start_time_ns": 10200000000, "source_effective_end_time_ns": 11800000000}}))


def test_real_cdr_parallel_pipeline_resume_and_read_only(tmp_path, monkeypatch):
    source, output = tmp_path / "bags", tmp_path / "output"
    make_bag(source / "episode9")
    shutil.copytree(source / "episode9", source / "episode10")
    shutil.copytree(source / "episode9", source / "episode11")
    (source / "episode11/collection_state.json").write_text('{"state":"discarded"}')
    before = {p.name: snapshot(p) for p in source.iterdir()}
    config = load_config(WORKSPACE / "configs/gello_pi05.yaml")
    config.update(source_root=str(source), output_root=str(output))
    monkeypatch.setattr(pipeline, "WORKSPACE", tmp_path)
    assert pipeline.convert(config, workers=2) == 0
    assert verify(output, full_video=True)["episodes"] == 2
    records = pipeline.load_records(output)
    assert [r["source"].split("/")[-1] for r in records] == ["episode9", "episode10"]
    row = pq.read_table(output / "data/chunk-000/episode_000000.parquet").slice(0, 1).to_pylist()[0]
    np.testing.assert_array_equal(row["observation.state"][:7], np.arange(7) + 2)
    np.testing.assert_allclose(row["observation.state"][7:14], 0.1)
    np.testing.assert_array_equal(row["action"][:14], np.arange(14) + 102)
    np.testing.assert_array_equal(row["action"][14:34], np.arange(20) + 102)
    with Bag(source / "episode9", config["schema_dir"]) as bag:
        t, ref = next(bag.records("/cam0/color/image_raw", image=True))
        assert bag.image(ref)[0, 0].tolist() == [123, 77, 0]
    with pytest.raises(FileExistsError):
        pipeline.convert(config)
    orphan = output / "data/chunk-000/episode_999999.parquet"
    orphan.write_bytes(b"interrupted and uncommitted")
    assert pipeline.convert(config, resume=True) == 0
    assert not orphan.exists()
    assert len(pipeline.load_records(output)) == 2
    assert before == {p.name: snapshot(p) for p in source.iterdir()}
    changed = dict(config, fps=20)
    with pytest.raises(ValueError, match="differs"):
        pipeline.convert(changed, resume=True)
    with pytest.raises(ValueError, match="workspace"):
        pipeline.convert(dict(config, output_root=str(tmp_path.parent / "unrelated")))


def test_milestone_bag_exports_overlapping_prefix_and_full_episodes(tmp_path, monkeypatch):
    source, output = tmp_path / "bags", tmp_path / "output"
    bag_path = source / "episode476"
    make_bag(bag_path)
    state_path = bag_path / "collection_state.json"
    state = json.loads(state_path.read_text())
    state.update(source_recording_id="recording-476", milestones=[{
        "id": "milestone_1", "timestamp_ns": 11_000_000_000, "clock": "ros",
    }])
    state_path.write_text(json.dumps(state))
    config = load_config(WORKSPACE / "configs/gello_pi05.yaml")
    config.update(source_root=str(source), output_root=str(output), fps=10,
                  segment_tasks={"milestone_1": "Pick up", "full": "Pick up and place"})
    monkeypatch.setattr(pipeline, "WORKSPACE", tmp_path)

    assert pipeline.convert(config, episode="episode476") == 0
    assert verify(output, full_video=True) == {
        "status": "passed", "episodes": 2, "frames": 26, "full_video_decode": True,
    }
    episodes = [json.loads(line) for line in (output / "meta/episodes.jsonl").read_text().splitlines()]
    assert [(ep["segment_id"], ep["length"], ep["task_index"]) for ep in episodes] == [
        ("milestone_1", 9, 0), ("full", 17, 1),
    ]
    assert all(ep["source_recording_id"] == "recording-476" for ep in episodes)
    prefix = pq.read_table(output / "data/chunk-000/episode_000000.parquet")
    full = pq.read_table(output / "data/chunk-000/episode_000001.parquet")
    assert prefix["source_timestamp_ns"][-1].as_py() == 11_000_000_000
    for key in ("source_timestamp_ns", "observation.state", "action"):
        assert prefix[key].to_pylist() == full[key].slice(0, len(prefix)).to_pylist()
