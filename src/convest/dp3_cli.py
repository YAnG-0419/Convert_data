"""Command line entrypoint for quality-grouped uncropped DP3 conversion."""
import argparse
from pathlib import Path

from convest.config import WORKSPACE, load_config
from convest.dp3_pipeline import convert_quality, quality_root
from convest.formats.dp3_zarr import verify


def main():
    parser = argparse.ArgumentParser(description="ROS2 bag -> coordinate-transformed uncropped DP3 Zarr")
    sub = parser.add_subparsers(dest="command", required=True)
    convert = sub.add_parser("convert")
    convert.add_argument("--config", type=Path, default=WORKSPACE / "configs/gello_dp3.yaml")
    convert.add_argument("--quality", required=True, help="Quality key from config, or all")
    convert.add_argument("--source-root", type=Path)
    convert.add_argument("--output-root", type=Path)
    convert.add_argument("--resume", action="store_true")
    convert.add_argument("--skip-ineligible", action="store_true")
    convert.add_argument("--limit", type=int, help="Smoke-test only: convert the first N selected bags")
    check = sub.add_parser("verify")
    check.add_argument("--config", type=Path, default=WORKSPACE / "configs/gello_dp3.yaml")
    check.add_argument("--quality", required=True, help="Quality key from config, or all")
    check.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        if getattr(args, "source_root", None):
            config["source_root"] = str(args.source_root.expanduser().resolve())
        if args.output_root:
            config["output_root"] = str(args.output_root.expanduser().resolve())
        groups = config.get("quality_groups", {})
        if not groups or (args.quality != "all" and args.quality not in groups):
            raise ValueError(f"Unknown quality {args.quality!r}; available: {', '.join(groups)}")
        qualities = tuple(groups) if args.quality == "all" else (args.quality,)
        if args.command == "verify":
            for quality in qualities:
                verify(quality_root(config, quality))
            return
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be positive")
        failed = 0
        for quality in qualities:
            failed |= convert_quality(
                config, quality, resume=args.resume,
                skip_ineligible=args.skip_ineligible, limit=args.limit,
            )
        raise SystemExit(failed)
    except (ValueError, FileExistsError, FileNotFoundError, BlockingIOError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
