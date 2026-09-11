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


def natural_key(path):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", str(path))]


def discover(root):
    root = Path(root).resolve()
    paths = [root / "metadata.yaml"] if (root / "metadata.yaml").exists() else root.rglob("metadata.yaml")
    results = []
    for metadata in sorted(paths, key=natural_key):
        path = metadata.parent
        state_path = path / "collection_state.json"
        try:
            state = json.loads(state_path.read_text())
            eligible = state.get("finalized") is True and state.get("state") == "finalized" and not state.get("failures")
            reason = None if eligible else f"collection state: {state.get('state')}; failures={state.get('failures', [])}"
            report = state.get("validation_report", {})
            start, end = (report.get(f"source_effective_{edge}_time_ns") for edge in ("start", "end"))
            if eligible and (start is None or end is None or end <= start):
                eligible, reason = False, "missing/empty validated source-time window"
            milestones = state.get("milestones", [])
            source_recording_id = state.get("source_recording_id") or str(path)
        except (OSError, ValueError) as exc:
            eligible, reason, start, end = False, str(exc), None, None
            milestones, source_recording_id = [], str(path)
        results.append({"path": str(path), "eligible": eligible, "reason": reason,
                        "source_start_ns": start, "source_end_ns": end,
                        "source_recording_id": source_recording_id, "milestones": milestones,
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

    @classmethod
    def build(cls, samples):
        if not samples:
            raise ValueError("Required stream contains no samples")
        # Stable sort permits bag delivery reordering; equal headers use last arrival.
        samples.sort(key=lambda x: x[0])
        unique = {time: value for time, value in samples}
        return cls(np.array(list(unique), dtype=np.int64), list(unique.values()))

    def indices(self, timeline):
        return np.searchsorted(self.times, timeline, side="right") - 1


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

    def records(self, topic, image=False):
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
                time = header_ns(payload)
                yield time, ((db, row_id) if image else self.store.deserialize_cdr(payload, kind))

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
