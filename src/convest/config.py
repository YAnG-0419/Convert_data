from pathlib import Path
import json
import hashlib
import re
from importlib import import_module

import yaml

WORKSPACE = Path(__file__).resolve().parents[2]


def load_config(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    # Relative paths are resolved against the project, independently of shell cwd.
    for key in ("source_root", "output_root", "contract", "schema_dir"):
        p = Path(config[key]).expanduser()
        config[key] = str((WORKSPACE / p).resolve() if not p.is_absolute() else p.resolve())
    from convest.registry import SOURCES, get_target
    if config["source_format"] not in SOURCES:
        raise ValueError(f"Unknown source_format: {config['source_format']}")
    target = get_target(config["target_format"])
    if not isinstance(config["fps"], int) or not 1 <= config["fps"] <= 120:
        raise ValueError("fps must be an integer in [1, 120]")
    if not str(config["task"]).strip():
        raise ValueError("task must be a nonempty language instruction")
    validate_repo_id(config["repo_id"])
    if config["min_segment_frames"] < 2:
        raise ValueError("Invalid freshness/segment limits")
    import_module(target.recipe).validate_config(config)
    return config


def validate_repo_id(repo_id):
    part = r"[A-Za-z0-9_]+(?:[.-][A-Za-z0-9_]+)*"
    if not isinstance(repo_id, str) or not re.fullmatch(f"{part}/{part}", repo_id):
        raise ValueError("repo_id must be namespace/dataset, e.g. fr3_wuji/tomato_high_quality")
    return repo_id


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)
