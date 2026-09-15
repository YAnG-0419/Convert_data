"""CLI kept separate from the existing ACT/Pi05/DP3 interfaces."""
import argparse
import json
from pathlib import Path

from convest.config import WORKSPACE
from convest.dp.convert import convert, load_config, verify


def main():
    parser = argparse.ArgumentParser(description='ROS2 episodes -> Diffusion Policy RGB-D Zarr v2')
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('convert')
    run.add_argument('--config', type=Path, default=WORKSPACE / 'configs/franka_dp.yaml')
    run.add_argument('--source-root', type=Path)
    run.add_argument('--output', type=Path)
    selection = run.add_mutually_exclusive_group()
    selection.add_argument('--episode')
    selection.add_argument('--episode-list', type=Path)
    run.add_argument('--cache-source', action='store_true', help='Temporarily cache each source bag locally for remote mounts')
    run.add_argument('--workers', type=int, default=1, help='Number of episode conversion processes (default: 1)')
    run.add_argument('--limit', type=int)
    run.add_argument('--resume', action='store_true')
    run.add_argument('--skip-ineligible', action='store_true')
    check = sub.add_parser('verify')
    check.add_argument('output', type=Path)
    args = parser.parse_args()
    try:
        if getattr(args, 'workers', 1) < 1:
            raise ValueError('--workers must be a positive integer')
        if args.command == 'verify':
            result = verify(args.output)
        else:
            if args.limit is not None and args.limit < 1:
                raise ValueError('--limit must be positive')
            config = load_config(args.config)
            for arg, key in ((args.source_root, 'source_root'), (args.output, 'output_root')):
                if arg:
                    config[key] = str(arg.expanduser().resolve())
            result = convert(config, episode=args.episode, episode_list=args.episode_list,
                             limit=args.limit, resume=args.resume, skip_ineligible=args.skip_ineligible, workers=args.workers, cache_source=args.cache_source)
        print(json.dumps(result, indent=2))
        if result.get('errors'):
            raise SystemExit(1)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
