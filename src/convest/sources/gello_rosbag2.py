"""Read sqlite ROS2 bags in read-only mode; no ROS installation is required.

Image payloads stay in SQLite. Indexing reads only the CDR header prefix, then
selected images are fetched on demand. Custom message definitions are local snapshots.
"""
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
import json
import re
import sqlite3
import struct

import numpy as np
import yaml
from rosbags.typesys import Stores, get_typestore, get_types_from_msg


STATUS_MESSAGE_TYPES = frozenset({
    "teleop_interfaces/msg/ArmCommandStatus",
    "teleop_interfaces/msg/HandTelemetryStatus",
})


def timestamp_policy(kind, time_map):
    if time_map and kind in time_map.unmodified_payload_types:
        # These custom headers can retain timestamps from earlier cleaning passes.
        # The latest trim report cannot reconstruct that history, whereas bag
        # record timestamps already live on the final compacted time axis.
        return "bag_record_timestamp" if kind in STATUS_MESSAGE_TYPES else "mapped_header_timestamp"
    return "header_timestamp"


def sample_timestamp(header_timestamp, received, kind, time_map):
    policy = timestamp_policy(kind, time_map)
    if policy == "bag_record_timestamp":
        return received
    if policy == "mapped_header_timestamp":
        return time_map.apply(header_timestamp)
    return header_timestamp


def natural_key(path):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", str(path))]


@dataclass(frozen=True)
class CompactTimeMap:
    """Map original absolute timestamps onto a cleaner's compacted time axis."""

    intervals: tuple[tuple[int, int, int], ...]
    unmodified_payload_types: frozenset[str]
    report_path: str

    @property
    def compact_boundaries(self):
        removed_before = 0
        result = []
        for start, _, removed in self.intervals:
            result.append(start - removed_before)
            removed_before += removed
        return tuple(result)

    def apply(self, timestamp_ns):
        removed_before = 0
        for start, end, removed in self.intervals:
            if timestamp_ns < start:
                break
            if timestamp_ns < end:
                # The cleaner clamps typed timestamps that fall inside removed ranges.
                return start - removed_before
            removed_before += removed
        return timestamp_ns - removed_before


def load_compact_time_map(path):
    path = Path(path)
    report_path = path / "trim_report.json"
    if not report_path.is_file():
        return None
    report = json.loads(report_path.read_text())
    if report.get("compact") is not True or report.get("timestamps") != "compact_shared_time_mapping":
        raise ValueError(f"Unsupported trim report time policy: {report_path}")
    raw_mapping = report.get("mapping")
    if not isinstance(raw_mapping, list):
        raise ValueError(f"Missing compact timestamp mapping: {report_path}")
    intervals = []
    previous_end = None
    for index, entry in enumerate(raw_mapping):
        try:
            start = int(entry["source_start_ns"])
            end = int(entry["source_end_ns"])
            removed = int(entry["removed_ns"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid compact mapping entry {index}: {report_path}") from exc
        if start >= end or removed != end - start or (previous_end is not None and start < previous_end):
            raise ValueError(f"Invalid/overlapping compact mapping entry {index}: {report_path}")
        intervals.append((start, end, removed))
        previous_end = end
    payload_types = report.get("unmodified_payload_types", {})
    if not isinstance(payload_types, dict) or any(not isinstance(name, str) for name in payload_types):
        raise ValueError(f"Invalid unmodified_payload_types: {report_path}")
    return CompactTimeMap(tuple(intervals), frozenset(payload_types), str(report_path.resolve()))


def _map_state_timestamps(value, time_map, key=None):
    if isinstance(value, dict):
        return {name: _map_state_timestamps(item, time_map, name) for name, item in value.items()}
    if isinstance(value, list):
        return [_map_state_timestamps(item, time_map, key) for item in value]
    is_absolute_time = key == "timestamp_ns" or (isinstance(key, str) and key.endswith(("_time_ns", "_header_ns")))
    return time_map.apply(value) if is_absolute_time and type(value) is int else value


def load_collection_state(path, time_map=None):
    """Load raw state, or adapt a cleaner-preserved source state without rewriting the bag."""
    path = Path(path)
    time_map = time_map if time_map is not None else load_compact_time_map(path)
    state_path = path / "collection_state.json"
    if state_path.is_file():
        return json.loads(state_path.read_text()), time_map, state_path.name
    source_state_path = path / "source_collection_state.json"
    if not source_state_path.is_file():
        raise FileNotFoundError(f"Missing collection_state.json: {path}")
    if time_map is None:
        raise ValueError(f"source_collection_state.json requires trim_report.json: {path}")
    state = json.loads(source_state_path.read_text())
    return _map_state_timestamps(state, time_map), time_map, source_state_path.name


def load_subtasks(path):
    """Load gap-free cleaner annotations on their compact output timeline."""
    path = Path(path)
    annotation_path = path / "subtasks.yaml"
    if not annotation_path.is_file():
        return None
    report_path = path / "trim_report.json"
    if not report_path.is_file():
        raise ValueError(f"subtasks.yaml requires trim_report.json: {path}")
    annotation = yaml.safe_load(annotation_path.read_text())
    report = json.loads(report_path.read_text())
    if not isinstance(annotation, dict) or annotation.get("version") != 1:
        raise ValueError(f"Unsupported subtask annotation version: {annotation_path}")
    expected = {
        "time_basis": "bag_record_timestamp",
        "timestamps": "compact_shared_time_mapping",
        "interval_semantics": "[start,end)",
        "recommended_lerobot_column": "language_persistent",
        "coverage": "gap_free",
    }
    for key, value in expected.items():
        if annotation.get(key) != value:
            raise ValueError(f"Unsupported {key} in {annotation_path}")
    try:
        origin_ns = int(report["source_start_ns"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing source_start_ns: {report_path}") from exc
    raw = annotation.get("subtasks")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Subtask list is empty: {annotation_path}")
    subtasks, previous_end = [], None
    for position, entry in enumerate(raw, 1):
        try:
            index = entry["index"]
            task = entry["subtask"].strip()
            output_start = float(entry["output_start"])
            output_end = float(entry["output_end"])
        except (KeyError, AttributeError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid subtask {position}: {annotation_path}") from exc
        if type(index) is not int or index != position or not task:
            raise ValueError(f"Invalid subtask index/task at position {position}: {annotation_path}")
        if not np.isfinite([output_start, output_end]).all() or output_end <= output_start:
            raise ValueError(f"Invalid subtask interval at position {position}: {annotation_path}")
        if previous_end is None:
            if abs(output_start) > 1e-6:
                raise ValueError(f"Gap-free subtask coverage must start at zero: {annotation_path}")
        elif abs(output_start - previous_end) > 1e-6:
            raise ValueError(f"Subtask coverage has a gap/overlap at position {position}: {annotation_path}")
        start_ns = origin_ns + round(output_start * 1_000_000_000)
        end_ns = origin_ns + round(output_end * 1_000_000_000)
        subtasks.append({"index": index, "task": task, "start_ns": start_ns, "end_ns": end_ns,
                         "output_start": output_start, "output_end": output_end})
        previous_end = output_end
    return {
        "items": subtasks,
        "source_recording_id": report.get("source") or str(path),
        "annotation_file": str(annotation_path.resolve()),
    }


def discover(root):
    root = Path(root).resolve()
    paths = [root / "metadata.yaml"] if (root / "metadata.yaml").exists() else root.rglob("metadata.yaml")
    results = []
    for metadata in sorted(paths, key=natural_key):
        path = metadata.parent
        try:
            time_map = load_compact_time_map(path)
            subtask_annotation = load_subtasks(path)
            try:
                state, time_map, state_file = load_collection_state(path, time_map)
                eligible = state.get("finalized") is True and state.get("state") == "finalized" and not state.get("failures")
                reason = None if eligible else f"collection state: {state.get('state')}; failures={state.get('failures', [])}"
                report = state.get("validation_report", {})
                start, end = (report.get(f"source_effective_{edge}_time_ns") for edge in ("start", "end"))
                if eligible and (start is None or end is None or end <= start):
                    eligible, reason = False, "missing/empty validated source-time window"
                milestones = state.get("milestones", [])
                source_recording_id = state.get("source_recording_id") or str(path)
                eligibility_source = "collection_state"
            except FileNotFoundError:
                if subtask_annotation is None or time_map is None:
                    raise
                subtasks = subtask_annotation["items"]
                eligible, reason = True, None
                start, end = subtasks[0]["start_ns"], subtasks[-1]["end_ns"]
                milestones = []
                source_recording_id = subtask_annotation["source_recording_id"]
                state_file = None
                eligibility_source = "gap_free_subtasks"
            source_time_adapter = ({"kind": "compact_trim_report", "state_file": state_file,
                                    "trim_report": time_map.report_path,
                                    "mapped_payload_types": sorted(time_map.unmodified_payload_types - STATUS_MESSAGE_TYPES),
                                    "record_timestamp_payload_types": sorted(time_map.unmodified_payload_types & STATUS_MESSAGE_TYPES)}
                                   if time_map is not None else None)
            subtasks = subtask_annotation["items"] if subtask_annotation else []
            subtask_annotation_file = subtask_annotation["annotation_file"] if subtask_annotation else None
        except (OSError, ValueError) as exc:
            eligible, reason, start, end = False, str(exc), None, None
            milestones, source_recording_id = [], str(path)
            source_time_adapter = None
            subtasks, subtask_annotation_file = [], None
            eligibility_source = None
        results.append({"path": str(path), "eligible": eligible, "reason": reason,
                        "source_start_ns": start, "source_end_ns": end,
                        "source_recording_id": source_recording_id, "milestones": milestones,
                        "subtasks": subtasks, "subtask_annotation_file": subtask_annotation_file,
                        "eligibility_source": eligibility_source,
                        "source_time_adapter": source_time_adapter,
                        "bytes": sum(p.stat().st_size for p in path.glob("*.db3"))})
    if not results:
        raise ValueError(f"No ROS2 bag metadata found under {root}")
    return results


def snapshot(path):
    return [{"name": p.name, "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for p in sorted(Path(path).iterdir()) if p.is_file()]


def header_ns(payload):
    if len(payload) < 12 or payload[:2] not in (b"\x00\x01", b"\x00\x00"):
        raise ValueError("Expected ROS2 CDR1 payload with a std_msgs/Header first field")
    sec, ns = struct.unpack_from("<iI" if payload[1] else ">iI", payload, 4)
    if ns >= 1_000_000_000 or (sec == 0 and ns == 0):
        raise ValueError("Invalid/zero source header timestamp; receive-time fallback is disabled")
    return sec * 1_000_000_000 + ns


@dataclass
class Series:
    times: np.ndarray
    values: object
    receive_times: np.ndarray | None = None

    @classmethod
    def build(cls, samples):
        if not samples:
            raise ValueError("Required stream contains no samples")
        # Stable sort permits bag delivery reordering; equal headers use last arrival.
        samples.sort(key=lambda x: x[0])
        has_receive_time = len(samples[0]) == 3
        if any((len(sample) == 3) != has_receive_time for sample in samples):
            raise ValueError("Series cannot mix samples with and without receive timestamps")
        unique = {
            sample[0]: (sample[1], sample[2] if has_receive_time else None)
            for sample in samples
        }
        return cls(
            np.array(list(unique), dtype=np.int64),
            [value for value, _ in unique.values()],
            (np.array([received for _, received in unique.values()], dtype=np.int64)
             if has_receive_time else None),
        )

    def indices(self, timeline):
        return np.searchsorted(self.times, timeline, side="right") - 1


def bridge_compact_image_boundaries(streams, time_map, start, end, fps, max_age_ns,
                                    camera_keys=("cam0", "cam1", "cam2")):
    """Reset a held image at a compact splice only when a grid frame would be stale.

    Trim ranges are selected in bag-receive time. Standard Image header stamps are
    compacted separately, so their receive/header latency can leave a one- or two-grid
    hole immediately before the splice. Re-stamping the last causal image at the first
    stale grid point preserves sample-and-hold semantics without admitting a future
    image or relaxing freshness away from cleaner-created boundaries.
    """
    if time_map is None:
        return {}
    count = (end - start) * fps // 1_000_000_000 + 1
    timeline = start + np.arange(count, dtype=np.int64) * 1_000_000_000 // fps
    inserted = {}
    for key in camera_keys:
        series = streams[key]
        count_for_key = 0
        for boundary in time_map.compact_boundaries:
            position = int(np.searchsorted(series.times, boundary, side="left"))
            if position <= 0 or position >= len(series.times):
                continue
            previous_time = int(series.times[position - 1])
            next_time = int(series.times[position])
            if not previous_time < boundary <= next_time:
                continue
            first = int(np.searchsorted(timeline, previous_time + max_age_ns, side="right"))
            if first >= len(timeline) or int(timeline[first]) >= next_time:
                continue
            hold_time = int(timeline[first])
            if hold_time > boundary or boundary - hold_time > max_age_ns:
                continue
            hold_position = int(np.searchsorted(series.times, hold_time, side="left"))
            if hold_position < len(series.times) and series.times[hold_position] == hold_time:
                continue
            series.times = np.insert(series.times, hold_position, hold_time)
            series.values.insert(hold_position, series.values[hold_position - 1])
            if series.receive_times is not None:
                series.receive_times = np.insert(series.receive_times, hold_position, hold_time)
            count_for_key += 1
        if count_for_key:
            inserted[key] = count_for_key
    return inserted


class Bag:
    def __init__(self, path, schema_dir):
        self.path = Path(path)
        self.store = get_typestore(Stores.ROS2_HUMBLE)
        for name in ("ArmCommandStatus", "HandTelemetryStatus"):
            self.store.register(get_types_from_msg((Path(schema_dir) / f"{name}.msg").read_text(),
                                                  f"teleop_interfaces/msg/{name}"))
        self.connections = []
        self.topics = {}
        self.stack = ExitStack()
        self.compact_time_map = load_compact_time_map(self.path)

    def __enter__(self):
        try:
            meta = yaml.safe_load((self.path / "metadata.yaml").read_text())["rosbag2_bagfile_information"]
            if meta["storage_identifier"] != "sqlite3" or meta.get("compression_format"):
                raise ValueError("Only uncompressed sqlite3 ROS2 bags are supported")
            for relative in meta["relative_file_paths"]:
                file = (self.path / relative).resolve()
                if not file.is_relative_to(self.path.resolve()):
                    raise ValueError("Bag file path escapes source directory")
                db = sqlite3.connect(file.as_uri() + "?mode=ro", uri=True)
                self.stack.callback(db.close)
                db.execute("PRAGMA query_only=ON")
                self.connections.append(db)
                for ident, topic, kind, serialization in db.execute("SELECT id,name,type,serialization_format FROM topics"):
                    if serialization != "cdr":
                        raise ValueError(f"Unsupported serialization {serialization}")
                    self.topics.setdefault(topic, []).append((db, ident, kind))
            return self
        except BaseException:
            self.stack.close()
            raise

    def __exit__(self, *args):
        self.stack.close()

    def records(self, topic, image=False, include_receive=False):
        if topic not in self.topics:
            raise ValueError(f"Missing topic: {topic}")
        for db, ident, kind in self.topics[topic]:
            expected = "sensor_msgs/msg/Image" if image else None
            if expected and kind != expected:
                raise ValueError(f"{topic}: expected {expected}, got {kind}")
            field = "substr(data,1,12)" if image else "data"
            for row_id, received, payload in db.execute(
                f"SELECT id,timestamp,{field} FROM messages WHERE topic_id=? ORDER BY timestamp,id", (ident,)
            ):
                time = sample_timestamp(header_ns(payload), received, kind, self.compact_time_map)
                value = (db, row_id) if image else self.store.deserialize_cdr(payload, kind)
                yield (time, value, received) if include_receive else (time, value)

    def image(self, reference):
        db, row_id = reference
        payload = db.execute("SELECT data FROM messages WHERE id=?", (row_id,)).fetchone()[0]
        msg = self.store.deserialize_cdr(payload, "sensor_msgs/msg/Image")
        encoding = msg.encoding.lower()
        channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}.get(encoding)
        if channels is None:
            raise ValueError(f"Unsupported image encoding {encoding}")
        rows = np.asarray(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
        image = rows[:, :msg.width * channels].reshape(msg.height, msg.width, channels)
        if encoding.startswith("bgr"):
            image = image[..., [2, 1, 0]]
        elif channels == 1:
            image = np.repeat(image, 3, axis=-1)
        else:
            image = image[..., :3]
        return np.ascontiguousarray(image)

    def depth(self, reference):
        db, row_id = reference
        payload = db.execute("SELECT data FROM messages WHERE id=?", (row_id,)).fetchone()[0]
        msg = self.store.deserialize_cdr(payload, "sensor_msgs/msg/Image")
        encoding = msg.encoding.lower()
        dtype = {"16uc1": np.dtype("u2"), "32fc1": np.dtype("f4")}.get(encoding)
        if dtype is None:
            raise ValueError(f"Unsupported depth encoding {encoding}")
        dtype = dtype.newbyteorder(">" if msg.is_bigendian else "<")
        row_bytes = msg.width * dtype.itemsize
        if msg.step < row_bytes or len(msg.data) != msg.height * msg.step:
            raise ValueError("Depth Image data/step/shape mismatch")
        rows = np.asarray(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
        return np.frombuffer(rows[:, :row_bytes].copy(), dtype=dtype).reshape(msg.height, msg.width)


def joint_values(msg, names, fields, allow_absent=False):
    if len(set(msg.name)) != len(msg.name):
        raise ValueError("Duplicate JointState joint names")
    by_name = {name: index for index, name in enumerate(msg.name)}
    overlap = set(names) & by_name.keys()
    if not overlap and allow_absent:
        return None
    if any(name not in by_name for name in names):
        raise ValueError(f"Partial/missing joint group: {set(names) - by_name.keys()}")
    result = []
    for field in fields:
        values = getattr(msg, field)
        if len(values) != len(msg.name):
            raise ValueError(f"JointState {field} length mismatch (no zero filling)")
        result.extend(values[by_name[name]] for name in names)
    result = np.asarray(result, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite joint values")
    return result


def read_streams(bag, contract):
    topics = contract["topics"]
    names = contract["source_joint_names"]
    streams = {}
    for part in ("arm", "hand"):
        for side in ("left", "right"):
            joints = names[f"measured_{side}_arm" if part == "arm" else f"{side}_hand"]
            streams[f"state.{side}_{part}"] = Series.build([
                (t, joint_values(m, joints, ["position", "velocity"]))
                for t, m in bag.records(topics[f"{side}_{part}_state"]["topic"])])
    actions = {side: [] for side in ("left", "right")}
    for t, m in bag.records(topics["validated_arm_action"]["topic"]):
        for side in actions:
            v = joint_values(m, names[f"validated_{side}_arm"], ["position"], allow_absent=True)
            if v is not None:
                actions[side].append((t, v))
    for side in actions:
        # Finalized bags must have engaged each group at least once.
        streams[f"action.{side}_arm"] = Series.build(actions[side])
        streams[f"action.{side}_hand"] = Series.build([
            (t, joint_values(m, names[f"{side}_hand"], ["position"]))
            for t, m in bag.records(topics[f"{side}_hand_action"]["topic"])])
    statuses = {group: [] for group in ("left_arm", "right_arm", "left_hand", "right_hand")}
    for t, m in bag.records(topics["arm_command_status"]["topic"]):
        for side in ("left", "right"):
            statuses[f"{side}_arm"].append((t, (side in m.accepted_sides, not bool(m.faults))))
    for t, m in bag.records(topics["hand_telemetry_status"]["topic"]):
        if m.side not in ("left", "right"):
            raise ValueError(f"Unknown hand side {m.side}")
        valid = m.state_valid and (not m.engaged or m.command_valid)
        statuses[f"{m.side}_hand"].append((t, (bool(m.engaged), bool(valid))))
    streams.update({f"status.{g}": Series.build(samples) for g, samples in statuses.items()})
    for camera in ("cam0", "cam1", "cam2"):
        streams[camera] = Series.build(list(bag.records(topics[camera]["topic"], image=True)))
    return streams


def read_act_streams(bag, contract, state_dim=54):
    """Named positions and optional velocities; ACT uses direct command samples.

    NaN is an internal missing-velocity marker only, replaced inside each aligned
    continuous segment. Invalid positions and malformed velocity arrays fail.
    """
    topics, names = contract["topics"], contract["source_joint_names"]
    groups = ("left_arm", "right_arm") + (("left_hand", "right_hand") if state_dim == 54 else ())
    streams = {}
    for group in groups:
        joints = names[f"measured_{group}" if group.endswith("arm") else group]
        samples = []
        for time, msg in bag.records(topics[f"{group}_state"]["topic"]):
            position = joint_values(msg, joints, ["position"])
            if len(msg.velocity) == 0:
                velocity = np.full(len(joints), np.nan, dtype=np.float32)
            else:
                if len(msg.velocity) != len(msg.name):
                    raise ValueError(f"{group}: JointState velocity length mismatch")
                by_name = {name: index for index, name in enumerate(msg.name)}
                velocity = np.array([msg.velocity[by_name[n]] for n in joints], dtype=np.float32)
                velocity[~np.isfinite(velocity)] = np.nan
            samples.append((time, np.r_[position, velocity]))
        streams[f"state.{group}"] = Series.build(samples)
    actions = {side: [] for side in ("left", "right")}
    for time, msg in bag.records(topics["validated_arm_action"]["topic"]):
        for side in actions:
            value = joint_values(msg, names[f"validated_{side}_arm"], ["position"], allow_absent=True)
            if value is not None:
                actions[side].append((time, value))
    for side, samples in actions.items():
        streams[f"action.{side}_arm"] = Series.build(samples)
    for group in groups[2:]:
        streams[f"action.{group}"] = Series.build([
            (time, joint_values(msg, names[group], ["position"]))
            for time, msg in bag.records(topics[f"{group}_action"]["topic"])])
    for camera in ("cam0", "cam1", "cam2"):
        streams[camera] = Series.build(list(bag.records(topics[camera]["topic"], image=True)))
    return streams


def read_dp3_streams(bag, contract):
    """Read DP3 vectors, health gates, depth and intrinsics with both clocks."""
    topics, names = contract["topics"], contract["source_joint_names"]

    def series(topic_key, convert, *, image=False):
        return Series.build([
            (time, convert(value), received)
            for time, value, received in bag.records(
                topics[topic_key]["topic"], image=image, include_receive=True
            )
        ])

    streams = {}
    for part in ("arm", "hand"):
        for side in ("left", "right"):
            group = f"{side}_{part}"
            joints = names[f"measured_{side}_arm" if part == "arm" else f"{side}_hand"]
            streams[f"state.{group}"] = series(
                f"{group}_state", lambda msg, joints=joints: joint_values(msg, joints, ["position", "velocity"])
            )

    actions = {side: [] for side in ("left", "right")}
    for time, msg, received in bag.records(
        topics["validated_arm_action"]["topic"], include_receive=True
    ):
        for side in actions:
            value = joint_values(msg, names[f"validated_{side}_arm"], ["position"], allow_absent=True)
            if value is not None:
                actions[side].append((time, value, received))
    for side, samples in actions.items():
        streams[f"action.{side}_arm"] = Series.build(samples)
        streams[f"action.{side}_hand"] = series(
            f"{side}_hand_action",
            lambda msg, side=side: joint_values(msg, names[f"{side}_hand"], ["position"]),
        )

    statuses = {group: [] for group in ("left_arm", "right_arm", "left_hand", "right_hand")}
    for time, msg, received in bag.records(
        topics["arm_command_status"]["topic"], include_receive=True
    ):
        for side in ("left", "right"):
            statuses[f"{side}_arm"].append(
                (time, (side in msg.accepted_sides, not bool(msg.faults)), received)
            )
    for time, msg, received in bag.records(
        topics["hand_telemetry_status"]["topic"], include_receive=True
    ):
        if msg.side not in ("left", "right"):
            raise ValueError(f"Unknown hand side {msg.side}")
        valid = msg.state_valid and (not msg.engaged or msg.command_valid)
        statuses[f"{msg.side}_hand"].append(
            (time, (bool(msg.engaged), bool(valid)), received)
        )
    streams.update({f"status.{group}": Series.build(samples) for group, samples in statuses.items()})
    streams["cam0_depth"] = series("cam0_depth", lambda value: value, image=True)
    streams["cam0_depth_info"] = series("cam0_depth_info", lambda value: value)
    return streams
