"""Independent full/phase datasets and cleaned status-clock regression coverage."""
import json
import hashlib
import sqlite3
import struct
from dataclasses import replace

import numpy as np
import pyarrow.parquet as pq
import pytest
import yaml

from convest import pipeline, registry
from convest.config import WORKSPACE, load_config
from convest.recipes.pi05 import output_segments, assess_candidate, validate_config
from convest.sources.gello_rosbag2 import CompactTimeMap, STATUS_MESSAGE_TYPES, sample_timestamp, snapshot
from convest.verify import verify
from test_alignment import streams
from convest.align import align
from test_pipeline import make_bag


@pytest.mark.parametrize("kind", sorted(STATUS_MESSAGE_TYPES))
def test_status_clock_is_scoped_to_unmodified_cleaned_status(kind):
    mapping = CompactTimeMap(((1_000, 2_000, 1_000),), frozenset({kind}), "trim.json")
    assert sample_timestamp(9_000, 3_000, kind, mapping) == 3_000
    assert sample_timestamp(9_000, 3_000, kind, None) == 9_000
    assert sample_timestamp(9_000, 3_000, kind, replace(mapping, unmodified_payload_types=frozenset())) == 9_000
    for other in ("sensor_msgs/msg/JointState", "sensor_msgs/msg/Image"):
        assert sample_timestamp(9_000, 3_000, other, mapping) == 9_000
        assert sample_timestamp(9_000, 3_000, other, replace(mapping, unmodified_payload_types=frozenset({other}))) == 8_000


def phase_item():
    return {"path": "/bags/episode1", "eligible": True, "subtasks": [
        {"index": i + 1, "task": f"Phase {i + 1}",
         "start_ns": 1_000_000_000 + i * 500_000_000,
         "end_ns": 1_000_000_000 + (i + 1) * 500_000_000}
        for i in range(4)
    ]}


@pytest.mark.parametrize("split", [False, True])
def test_full_and_phase_outputs_partition_legacy_mixed_output(split):
    aligned = align(streams(), 1_000_000_000, 3_000_000_000, 10, 150_000_000)
    if split:
        aligned.segments = [(0, 2), (3, 21)]
    config = {"segments": "subtasks", "task": "Full task", "min_segment_frames": 2,
              "subtask_tasks": {f"Phase {i}": f"English {i}" for i in range(1, 5)}}
    mixed = output_segments(phase_item(), aligned, config)
    full = output_segments(phase_item(), aligned, dict(config, segments="full", subtask_tasks={}))
    phases = output_segments(phase_item(), aligned, dict(config, segments="subtasks-only"))
    assert full + phases == mixed
    assert all(x["segment_id"].startswith("full") and x["task"] == "Full task" for x in full)
    assert all(x["segment_id"].startswith("subtask_") for x in phases)
    assert full[-1]["bounds"][1] == 20  # Annotation [start,end), same in both datasets.


def test_full_does_not_require_every_phase_to_survive():
    aligned = align(streams(), 1_000_000_000, 3_000_000_000, 10, 150_000_000)
    aligned.segments = [(5, 21)]
    config = {"segments": "full", "task": "Full task", "min_segment_frames": 2,
              "subtask_tasks": {f"Phase {i}": f"English {i}" for i in range(1, 5)}}
    assert output_segments(phase_item(), aligned, config)[0]["bounds"] == (5, 20)
    with pytest.raises(ValueError, match="subtask_1"):
        output_segments(phase_item(), aligned, dict(config, segments="subtasks-only"))


@pytest.mark.parametrize("mode", ["subtasks", "subtasks-only", "both"])
def test_phase_modes_require_four_translated_phases(mode):
    config = load_config(WORKSPACE / "configs/gello_pi05.yaml")
    config.update(segments=mode, subtask_tasks={})
    with pytest.raises(ValueError, match="translations"):
        validate_config(config)
    assert not assess_candidate(phase_item(), config)["eligible"]
    item = phase_item()
    item["subtasks"] = item["subtasks"][:3]
    assert "exactly 4" in assess_candidate(item, config)["reason"]


def cleaned_fixture(path):
    make_bag(path)
    # Simulate stale custom headers from an earlier cleaning pass, while the
    # latest report has no remaining deletion interval to explain their offset.
    with sqlite3.connect(path / "capture_0.db3") as db:
        rows = db.execute("SELECT m.id,m.data FROM messages m JOIN topics t ON m.topic_id=t.id "
                          "WHERE t.type IN (?,?)", tuple(sorted(STATUS_MESSAGE_TYPES))).fetchall()
        for ident, payload in rows:
            seconds, nanos = struct.unpack("<iI", payload[4:12])
            db.execute("UPDATE messages SET data=? WHERE id=?",
                       (payload[:4] + struct.pack("<iI", seconds + 5, nanos) + payload[12:], ident))
    (path / "trim_report.json").write_text(json.dumps({
        "source_start_ns": "10200000000", "compact": True,
        "timestamps": "compact_shared_time_mapping", "mapping": [],
        "unmodified_payload_types": {kind: 21 for kind in STATUS_MESSAGE_TYPES},
    }))
    (path / "subtasks.yaml").write_text(yaml.safe_dump({
        "version": 1, "time_basis": "bag_record_timestamp",
        "timestamps": "compact_shared_time_mapping", "interval_semantics": "[start,end)",
        "recommended_lerobot_column": "language_persistent", "coverage": "gap_free",
        "subtasks": [{"index": i + 1, "subtask": f"Phase {i + 1}",
                      "output_start": i * .4, "output_end": (i + 1) * .4} for i in range(4)],
    }))


def test_two_datasets_cdr_video_resume_and_version_guard(tmp_path, monkeypatch):
    source = tmp_path / "bags/episode1"
    cleaned_fixture(source)
    before = snapshot(source)
    config = load_config(WORKSPACE / "configs/gello_pi05.yaml")
    config.update(source_root=str(source.parent), output_root=str(tmp_path / "separate"),
                  repo_id="fr3_wuji/test", segments="both",
                  subtask_tasks={f"Phase {i}": f"English {i}" for i in range(1, 5)})
    monkeypatch.setattr(pipeline, "WORKSPACE", tmp_path)
    assert pipeline.convert(config) == 0
    output = tmp_path / "separate"
    assert verify(output / "full", full_video=True)["episodes"] == 1
    assert verify(output / "subtasks", full_video=True)["episodes"] == 4
    for child, mode in (("full", "full"), ("subtasks", "subtasks-only")):
        manifest = json.loads((output / child / "conversion/manifest.json").read_text())
        assert manifest["config"]["segments"] == mode
        assert manifest["config"]["repo_id"] == f"fr3_wuji/test_{child}"
        record = pipeline.load_records(output / child)[0]
        assert set(record["alignment"]["status_timestamp_policy"].values()) == {"bag_record_timestamp"}
    full = pq.read_table(output / "full/data/chunk-000/episode_000000.parquet")
    phases = [pq.read_table(p) for p in sorted((output / "subtasks/data/chunk-000").glob("*.parquet"))]
    for column in ("source_timestamp_ns", "observation.state", "action"):
        assert full[column].to_pylist() == [row for part in phases for row in part[column].to_pylist()]
    assert len(full) == 48
    np.testing.assert_array_equal(full["action"][0].as_py()[:14], np.arange(14) + 102)
    assert pipeline.convert(config, resume=True) == 0
    assert snapshot(source) == before
    # Simulate an old converter manifest: fail before updating either child.
    marker = output / "subtasks/conversion/manifest.json"
    manifest = json.loads(marker.read_text())
    contract = yaml.safe_load((WORKSPACE / "schemas/gello_contract.yaml").read_text())
    schemas = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (WORKSPACE / "schemas").glob("*.msg")}
    manifest["fingerprint"] = pipeline.conversion_fingerprint(manifest["config"], contract, schemas, "0.3.0")
    marker.write_text(json.dumps(manifest))
    first_summary = (output / "full/conversion/summary.json").read_bytes()
    with pytest.raises(ValueError, match="converter differs"):
        pipeline.convert(config, resume=True)
    assert (output / "full/conversion/summary.json").read_bytes() == first_summary
    assert registry.get_target("lerobot_v21").version != "0.3.0"


def test_both_refuses_existing_mixed_dataset(tmp_path, monkeypatch):
    output = tmp_path / "mixed"
    (output / "conversion").mkdir(parents=True)
    (output / "conversion/manifest.json").write_text("{}")
    config = load_config(WORKSPACE / "configs/gello_pi05.yaml")
    config.update(output_root=str(output), source_root=str(tmp_path / "bags"), segments="both")
    monkeypatch.setattr(pipeline, "WORKSPACE", tmp_path)
    with pytest.raises(FileExistsError, match="new parent directory"):
        pipeline.convert(config, resume=True)
    assert not (output / "full").exists()


@pytest.mark.parametrize("mode", ["full", "subtasks-only", "both"])
def test_cli_accepts_export_choices(mode, tmp_path, monkeypatch):
    import sys
    from convest.cli import main

    calls = []

    def capture(config, *args, **kwargs):
        calls.append(config)
        return 0

    monkeypatch.setattr(pipeline, "convert", capture)
    monkeypatch.setattr(sys, "argv", ["convest", "convert", "--segments", mode,
                                    "--source-root", str(tmp_path / "A"),
                                    "--source-root", str(tmp_path / "B"),
                                    "--output", str(tmp_path / "result"),
                                    "--repo-id", "fr3_wuji/new", "--resume"])
    with pytest.raises(SystemExit) as result:
        main()
    assert result.value.code == 0
    assert calls[0]["segments"] == mode
    assert calls[0]["source_root"] == [str(tmp_path / "A"), str(tmp_path / "B")]
    assert calls[0]["repo_id"] == "fr3_wuji/new"


@pytest.mark.parametrize("workers,accepted", [(1, True), (8, True), (12, True),
                                             (0, False), (-1, False), (13, False)])
def test_cli_worker_limits(workers, accepted, monkeypatch):
    import sys
    from convest.cli import main

    calls = []

    def capture(config, resume, limit, episode, worker_count, **kwargs):
        calls.append(worker_count)
        return 0

    monkeypatch.setattr(pipeline, "convert", capture)
    monkeypatch.setattr(sys, "argv", ["convest", "convert", "--workers", str(workers)])
    with pytest.raises(SystemExit) as result:
        main()
    assert result.value.code == (0 if accepted else 2)
    assert calls == ([workers] if accepted else [])
