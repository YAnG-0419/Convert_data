"""Compute Pi05 stats from vectors only, with identical horizon and delta transforms.

No RGB decoding is necessary for state/action statistics. Runs with Pi05 Python.
"""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json

import numpy as np
import pyarrow.parquet as pq
from openpi.policies.fr3_wuji_policy import _reorder_actions, _reorder_state
from openpi.shared import normalize
from openpi import transforms
from openpi.training import config as training_config

WORKSPACE = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, help="Default: outputs/pi05_assets/pi05_fr3_wuji/<dataset repo_id>")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    manifest = json.loads((args.root / "conversion/manifest.json").read_text())
    repo_id = manifest["config"]["repo_id"]
    output = (args.output or WORKSPACE / "outputs/pi05_assets/pi05_fr3_wuji" / repo_id).resolve()
    if not output.is_relative_to(WORKSPACE / "outputs"):
        raise ValueError("Normalization output must remain under this workspace's outputs/")
    lock_path = args.root / "conversion/lock"
    with lock_path.open("r") as lock:
        # Refuse to compute stats against a dataset that is currently being appended.
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        info_bytes = (args.root / "meta/info.json").read_bytes()
        info = json.loads(info_bytes)
        episodes = [json.loads(s) for s in (args.root / "meta/episodes.jsonl").read_text().splitlines()]
        cfg = training_config.get_config("pi05_fr3_wuji")
        horizon = cfg.model.action_horizon
        delta = transforms.DeltaActions(transforms.make_bool_mask(7, -20, 7, -20))
        stats = {key: normalize.RunningStats() for key in ("state", "actions")}
        for ep in episodes:
            path = args.root / info["data_path"].format(episode_chunk=ep["episode_index"] // info["chunks_size"],
                                                       episode_index=ep["episode_index"])
            table = pq.read_table(path, columns=["observation.state", "action"])
            state = _reorder_state(np.asarray(table["observation.state"].to_pylist(), dtype=np.float32))
            action = _reorder_actions(np.asarray(table["action"].to_pylist(), dtype=np.float32))
            for start in range(0, len(table), args.batch_size):
                stop = min(len(table), start + args.batch_size)
                indices = np.minimum(np.arange(start, stop)[:, None] + np.arange(horizon), len(table) - 1)
                batch = {"state": state[start:stop], "actions": action[indices]}
                if cfg.data.extra_delta_transform:
                    batch = delta(batch)
                for key in stats:
                    # Accumulate in float64 to avoid cancellation for nearly static joints.
                    stats[key].update(batch[key].astype(np.float64))
            print(f"stats episode {ep['episode_index'] + 1}/{len(episodes)}", flush=True)
        result = {key: value.get_statistics() for key, value in stats.items()}
        for value in result.values():
            for field in ("mean", "std", "q01", "q99"):
                assert np.isfinite(getattr(value, field)).all()
        normalize.save(output, result)
        provenance = {"dataset": str(args.root.resolve()), "repo_id": repo_id,
                      "dataset_info_sha256": hashlib.sha256(info_bytes).hexdigest(),
                      "frames": info["total_frames"], "episodes": len(episodes), "action_horizon": horizon,
                      "extra_delta_transform": cfg.data.extra_delta_transform,
                      "method": "all rows, episode-clipped horizon, Pi05 reorder + DeltaActions + RunningStats, float64 accumulation"}
        (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        print(f"Saved Pi05 normalization statistics -> {output / 'norm_stats.json'}")


if __name__ == "__main__":
    main()
