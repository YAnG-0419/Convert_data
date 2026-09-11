"""Run with Pi05's existing Python. Redirect every cache into this workspace."""
from pathlib import Path
import argparse
import os

WORKSPACE = Path(__file__).resolve().parents[1]
os.environ["HF_HOME"] = str(WORKSPACE / ".cache/huggingface")
os.environ["HF_DATASETS_CACHE"] = str(WORKSPACE / ".cache/huggingface/datasets")
os.environ["XDG_CACHE_HOME"] = str(WORKSPACE / ".cache")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"

import json
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from openpi.policies.fr3_wuji_policy import Fr3WujiInputs
from openpi.models.model import ModelType


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--backend", choices=["pyav", "torchcodec"], default="pyav")
    args = parser.parse_args()
    manifest = json.loads((args.root / "conversion/manifest.json").read_text())
    info = json.loads((args.root / "meta/info.json").read_text())
    repo_id = manifest["config"]["repo_id"]
    ds = LeRobotDataset(repo_id, root=args.root, video_backend=args.backend,
                        delta_timestamps={"action": [i / info["fps"] for i in range(50)]})
    transform = Fr3WujiInputs(ModelType.PI05)
    checked = []
    for ep in range(min(args.episodes, ds.meta.total_episodes)):
        start = int(ds.episode_data_index["from"][ep])
        end = int(ds.episode_data_index["to"][ep])
        for index in sorted(set([start, (start + end) // 2, end - 1])):
            row = ds[index]
            data = {"observation/state": row["observation.state"].numpy(),
                    "actions": row["action"].numpy(), "prompt": row["task"],
                    "observation/image": row["observation.images.cam0"].numpy(),
                    "observation/left_wrist_image": row["observation.images.cam1"].numpy(),
                    "observation/right_wrist_image": row["observation.images.cam2"].numpy()}
            transformed = transform(data)
            assert transformed["state"].shape == (54,)
            assert transformed["actions"].shape == (50, 54)
            assert all(im.dtype == np.uint8 for im in transformed["image"].values())
            assert all(ds.hf_dataset[i]["episode_index"].item() == ep for i in range(index, min(index + 50, end)))
            if index == end - 1:
                assert row["action_is_pad"][1:].all()
            checked.append(index)
    result = {"status": "passed", "loader": "Pi05 installed LeRobotDataset (v2.1)", "video_backend": ds.video_backend,
              "repo_id": repo_id, "transform": "Fr3WujiInputs", "episodes": ds.meta.total_episodes,
              "frames": len(ds), "sampled_indices": checked, "action_horizon": 50}
    (args.root / "conversion/pi05_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
