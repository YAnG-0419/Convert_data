import argparse
from pathlib import Path

from convest.config import load_config, validate_repo_id, WORKSPACE


def main():
    parser = argparse.ArgumentParser(description="Offline robot dataset conversion")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "convert"):
        command = sub.add_parser(name)
        command.add_argument("--config", type=Path, default=WORKSPACE / "configs/gello_pi05.yaml")
        command.add_argument("--source-root", type=Path, action="append",
                             help="Input root; repeat to combine multiple non-overlapping roots")
        if name == "inspect":
            command.add_argument("--report", type=Path, default=WORKSPACE / "reports/inventory.json")
        else:
            command.add_argument("--resume", action="store_true")
            command.add_argument("--relocate", action="store_true",
                                 help=("With --resume, explicitly migrate an existing dataset after only "
                                       "source_root/output_root changed; verifies committed sources first"))
            selection = command.add_mutually_exclusive_group()
            selection.add_argument("--limit", type=int)
            selection.add_argument("--episode")
            selection.add_argument("--episode-list", type=Path, help="TXT containing episode numbers")
            command.add_argument("--skip-ineligible", action="store_true", help="Explicitly omit listed bags that failed collection validation")
            command.add_argument("--output", type=Path)
            command.add_argument("--repo-id", help="Immutable dataset id: namespace/dataset")
            command.add_argument("--segments", choices=("all", "full", "milestones", "first-milestone-or-full", "subtasks", "subtasks-only", "both"),
                                 help=("Export prefixes plus full, full only, prefixes only, the first "
                                       "milestone prefix with full fallback, full+subtasks mixed (legacy), "
                                       "subtasks only, or two independent full/ and subtasks/ datasets"))
            command.add_argument("--workers", type=int, default=1,
                                 help="Parallel bags (1-12); each worker uses two H264 encoder threads")
    command = sub.add_parser("check-list", help="Read-only list and collection-state preflight")
    command.add_argument("episode_list", type=Path)
    command.add_argument("--config", type=Path, default=WORKSPACE / "configs/gello_pi05.yaml")
    command.add_argument("--source-root", type=Path, action="append",
                         help="Input root; repeat to combine multiple non-overlapping roots")
    command = sub.add_parser("verify")
    command.add_argument("root", type=Path)
    command.add_argument("--full-video", action="store_true")
    args = parser.parse_args()
    if args.command == "verify":
        import json
        from importlib import import_module
        from convest.registry import get_target
        marker = args.root / "conversion/manifest.json"
        target_format = json.loads(marker.read_text())["config"]["target_format"] if marker.exists() else "lerobot_v21"
        verifier = import_module(get_target(target_format).verifier)
        verifier.verify(args.root, args.full_video)
        return
    config = load_config(args.config)
    if getattr(args, "source_root", None):
        roots = [str(path.expanduser().resolve()) for path in args.source_root]
        config["source_root"] = roots[0] if len(roots) == 1 else roots
    if args.command == "check-list":
        from convest.selection import select_episodes
        from convest.registry import get_source
        import json
        try:
            from convest.pipeline import discover_sources
            source = get_source(config["source_format"])
            _, selection = select_episodes(discover_sources(source, config["source_root"]),
                                           args.episode_list, skip_ineligible=True)
        except (ValueError, OSError) as exc:
            parser.exit(2, f"Error: {exc}\n")
        print(json.dumps(selection, ensure_ascii=False, indent=2))
        raise SystemExit(2 if selection["skipped"] else 0)
    if args.command == "inspect":
        from convest.pipeline import inspect
        inspect(config, args.report)
    else:
        from convest.pipeline import convert
        if config["target_format"] == "dp3_uncropped_zarr":
            parser.error("DP3 quality-group conversion uses scripts/dp3 convert --quality <group>")
        if args.output:
            config["output_root"] = str(args.output.expanduser().resolve())
        if args.repo_id:
            try:
                config["repo_id"] = validate_repo_id(args.repo_id)
            except ValueError as exc:
                parser.error(str(exc))
        if args.segments:
            config["segments"] = args.segments
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be positive")
        if not 1 <= args.workers <= 12:
            parser.error("--workers must be in [1, 12]")
        if args.skip_ineligible and not args.episode_list:
            parser.error("--skip-ineligible requires --episode-list")
        try:
            result = convert(config, args.resume, args.limit, args.episode, args.workers,
                             episode_list=args.episode_list, skip_ineligible=args.skip_ineligible,
                             relocate=args.relocate)
        except (ValueError, FileExistsError, FileNotFoundError, BlockingIOError) as exc:
            parser.exit(2, f"Error: {exc}\n")
        raise SystemExit(result)


if __name__ == "__main__":
    main()
