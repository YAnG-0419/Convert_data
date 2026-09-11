"""Process-scoped Pi05 entrypoint; does not change its source or environment."""
from pathlib import Path
import os
import runpy
import sys
import argparse

WORKSPACE = Path(__file__).resolve().parents[1]
PI05 = Path(os.environ.get("PI05_ROOT", "/home/user/lpy/Pi05")).resolve()
os.environ["HF_HOME"] = str(WORKSPACE / ".cache/huggingface")
os.environ["HF_DATASETS_CACHE"] = str(WORKSPACE / ".cache/huggingface/datasets")
os.environ["HF_LEROBOT_HOME"] = str(WORKSPACE / "outputs")
os.environ["XDG_CACHE_HOME"] = str(WORKSPACE / ".cache")
os.environ["OPENPI_DATA_HOME"] = str(WORKSPACE / ".cache/openpi")
sys.path[:0] = [str(PI05 / "src"), str(PI05 / "packages/openpi-client/src")]


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("check", "norm", "train"):
        raise SystemExit("Usage: scripts/pi05 {check|norm|train} [arguments]")
    mode, arguments = sys.argv[1], sys.argv[2:]
    if mode != "train":
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["HF_HUB_OFFLINE"] = "1"
    import lerobot.common.datasets.lerobot_dataset as dataset_module
    # This local process chooses the already-working backend. No package file is patched.
    dataset_module.get_safe_default_codec = lambda: "pyav"
    if mode == "check":
        entry = WORKSPACE / "scripts/check_pi05.py"
    elif mode == "norm":
        entry = WORKSPACE / "scripts/compute_pi05_norm.py"
    else:
        options = argparse.ArgumentParser(add_help=False)
        options.add_argument("--repo-id", default="fr3_wuji/tomato")
        known, arguments = options.parse_known_args(arguments)
        entry = PI05 / "scripts/train.py"
        arguments = ["pi05_fr3_wuji", "--assets-base-dir", str(WORKSPACE / "outputs/pi05_assets"),
                     "--checkpoint-base-dir", str(WORKSPACE / "outputs/checkpoints"),
                     "--data.repo-id", known.repo_id, *arguments]
    sys.argv = [str(entry), *arguments]
    runpy.run_path(str(entry), run_name="__main__")


if __name__ == "__main__":
    main()
