"""Causal 10 Hz alignment for uncropped DP3 point-cloud delivery."""
import math

import numpy as np

from convest.align import align


def validate_config(config):
    if config.get("window") != "validated":
        raise ValueError("DP3 window must be validated")
    if config.get("state_dim") != 54:
        raise ValueError("This FR3/Wuji DP3 converter requires state_dim=54")
    for key in ("max_staleness_ms", "depth_scale_m", "min_depth_m", "max_depth_m", "max_joint_step_rad"):
        value = config.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
    if config["max_staleness_ms"] <= 0 or config["depth_scale_m"] <= 0:
        raise ValueError("Freshness and depth scale must be positive")
    if not 0 <= config["min_depth_m"] < config["max_depth_m"]:
        raise ValueError("Depth range is invalid")
    transform = np.asarray(config.get("T_point_from_depth_camera"), dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("T_point_from_depth_camera must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError("Transform bottom row must be [0, 0, 0, 1]")
    rotation = transform[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-4) or not np.isclose(
        np.linalg.det(rotation), 1, atol=1e-4
    ):
        raise ValueError("Transform rotation must be orthonormal with determinant +1")
    if not str(config.get("source_point_frame", "")).strip() or not str(config.get("point_frame", "")).strip():
        raise ValueError("source_point_frame and point_frame are required")
    if not str(config.get("calibration_version", "")).strip():
        raise ValueError("calibration_version is required")
    if (config["source_point_frame"] != config["point_frame"]
            and np.allclose(transform, np.eye(4), atol=1e-8)):
        raise ValueError("A non-identity point frame requires the actual calibrated transform")
    if config.get("require_coordinate_transform", False) and np.allclose(transform, np.eye(4), atol=1e-8):
        raise ValueError("Formal delivery requires the actual non-identity camera-to-workcell transform")


def validate_contract(contract, config):
    del config
    names = contract["source_joint_names"]
    expected = {
        "measured_left_arm": 7, "measured_right_arm": 7,
        "validated_left_arm": 7, "validated_right_arm": 7,
        "left_hand": 20, "right_hand": 20,
    }
    if any(len(names[key]) != size for key, size in expected.items()):
        raise ValueError("DP3 requires FR3/Wuji joint groups 7, 7, 20, 20")
    for key in ("cam0_depth", "cam0_depth_info"):
        if key not in contract["topics"]:
            raise ValueError(f"DP3 contract is missing {key}")


def prepare(source, bag, item, config, contract):
    streams = source.read_dp3_streams(bag, contract)
    aligned = align(
        streams,
        item["source_start_ns"],
        item["source_end_ns"],
        config["fps"],
        round(config["max_staleness_ms"] * 1e6),
        config["min_segment_frames"],
        camera_keys=("cam0_depth", "cam0_depth_info"),
    )
    return streams, aligned
