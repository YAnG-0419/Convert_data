"""Batch-convert selected bags at 30 Hz, resume commits, then verify the output."""
import argparse
import json
from pathlib import Path

from convest.config import WORKSPACE
from convest.dp.convert import convert, load_config, verify


def main():
    parser = argparse.ArgumentParser(
        description='Batch ROS2 -> DP at 30 Hz; resumes committed bags and verifies the output')
    parser.add_argument('--config', type=Path, default=WORKSPACE / 'configs/franka_dp.yaml')
    parser.add_argument('--source-root', type=Path, help='Recursively discover episode folders below this directory')
    parser.add_argument('--output', type=Path, help='Dataset directory inside Convert_data')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--all', action='store_true', help='Convert all discovered episodes (default)')
    selection.add_argument('--episodes', nargs='+', help='Several episodes in one batch, e.g. episode60 episode62 63')
    selection.add_argument('--episode-list', type=Path, help='Optional TXT list of episode numbers/ranges')
    parser.add_argument('--skip-ineligible', action='store_true', help='Explicitly omit bags failing collection validation')
    parser.add_argument('--segment-mode', choices=['full', 'first_stage'], default='full', help='Keep full validated recordings or only the prefix before the milestone')
    parser.add_argument('--cache-source', action='store_true', help='Temporarily cache each source bag locally for remote mounts')
    parser.add_argument('--workers', type=int, default=1, help='Number of episode conversion processes (default: 1)')
    parser.add_argument('--limit', type=int, help='Convert only the first N selected bags, for a trial run')
    args = parser.parse_args()
    try:
        if getattr(args, 'workers', 1) < 1:
            raise ValueError('--workers must be a positive integer')
        if args.limit is not None and args.limit < 1:
            raise ValueError('--limit must be positive')
        config = load_config(args.config)
        # This entrypoint always produces the requested 30 Hz action/observation grid.
        config['fps'] = 30
        config['segment_mode'] = args.segment_mode
        for arg, key in ((args.source_root, 'source_root'), (args.output, 'output_root')):
            if arg:
                config[key] = str(arg.expanduser().resolve())
        print(f"Batch conversion: 30 Hz, source={config['source_root']}, output={config['output_root']}", flush=True)
        result = convert(config, episode_list=args.episode_list, episodes=args.episodes, resume=True,
                         skip_ineligible=args.skip_ineligible, limit=args.limit, workers=args.workers, cache_source=args.cache_source)
        print(json.dumps(result, indent=2), flush=True)
        # Even if a bag failed, verify all successfully committed data; preserve error exit status.
        if result['frames']:
            print(json.dumps(verify(config['output_root']), indent=2), flush=True)
        if result['errors']:
            raise SystemExit(1)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
