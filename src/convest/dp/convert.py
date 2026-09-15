"""Bounded-memory Zarr v2 writer with per-bag commits and resumable rollback."""
from pathlib import Path
from collections import deque
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import multiprocessing
import tempfile
import fcntl
import hashlib
import json
import math
import shutil
import time

import av
import numpy as np
import yaml
import zarr
from numcodecs import Blosc

from convest.config import WORKSPACE, atomic_json, digest
from convest.selection import select_episodes
from convest.sources.gello_rosbag2 import Bag, discover, snapshot
from convest.dp import source

VERSION = 'dp-rgbd-1'


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    for key in ('source_root', 'output_root', 'contract', 'schema_dir'):
        p = Path(config[key]).expanduser()
        config[key] = str((WORKSPACE / p).resolve() if not p.is_absolute() else p.resolve())
    for key in ('fps', 'image_width', 'image_height', 'min_segment_frames'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if config['fps'] > 120 or config['min_segment_frames'] < 2:
        raise ValueError('fps must be <=120 and min_segment_frames >=2')
    for key in ('depth_scale', 'depth_max_m', 'state_max_age_ms', 'command_max_age_ms',
                'status_max_age_ms', 'image_max_age_ms', 'max_joint_step_rad', 'min_free_gb'):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if type(config['include_depth']) is not bool:
        raise ValueError('include_depth must be a boolean')
    return config


def output_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(WORKSPACE) or path == WORKSPACE:
        raise ValueError('DP output must be a child directory of Convert_data')
    return path


def shape_meta(config, contract):
    shape = [3, config['image_height'], config['image_width']]
    dim = len(source.joint_names(contract))
    obs = {'joint_pos': {'shape': [dim], 'type': 'low_dim'}}
    obs.update({key: {'shape': shape, 'type': 'rgb'} for key in source.CAMERAS})
    if config['include_depth']:
        obs['head_depth'] = {'shape': [1, *shape[1:]], 'type': 'depth'}
    return {'obs': obs, 'action': {'shape': [dim]}}


def resize(array, width, height, depth=False):
    """Letterbox to preserve aspect; depth uses nearest pixel centres, never blends."""
    h, w = array.shape[:2]
    ph, pw = max(h, math.ceil(w * height / width)), max(w, math.ceil(h * width / height))
    top, left = (ph-h)//2, (pw-w)//2
    padding = ((top, ph-h-top), (left, pw-w-left)) + (() if depth else ((0, 0),))
    array = np.pad(array, padding)
    if depth:
        ys = np.minimum(((np.arange(height)+0.5)*ph/height).astype(int), ph-1)
        xs = np.minimum(((np.arange(width)+0.5)*pw/width).astype(int), pw-1)
        return array[ys[:, None], xs[None, :], None]
    frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(array), format='rgb24')
    return frame.reformat(width=width, height=height, format='rgb24', interpolation='AREA').to_ndarray(format='rgb24')


def depth_metres(raw, scale):
    values = raw.astype(np.float32)
    if raw.dtype.kind == 'u':
        values *= scale
    values[~np.isfinite(values) | (values <= 0)] = 0
    return values


def records(root):
    return [json.loads(p.read_text()) for p in sorted((root / 'conversion/records').glob('*.json'))]


def recover(store, committed):
    episodes = [ep for r in committed for ep in r['episodes']]
    ends = np.array([e['end'] for e in episodes], dtype='i8')
    total = int(ends[-1]) if len(ends) else 0
    for group in ('data', 'provenance'):
        for _, arr in store.require_group(group).arrays():
            if len(arr) < total:
                raise ValueError('Committed array is truncated; restore the dataset from backup')
            arr.resize((total, *arr.shape[1:]))
    meta = store.require_group('meta')
    if 'episode_ends' not in meta:
        meta.create_dataset('episode_ends', shape=(0,), chunks=(1024,), dtype='i8', compressor=None)
    meta['episode_ends'].resize(len(ends))
    meta['episode_ends'][:] = ends
    return total


def append_array(group, key, values, start, image=False):
    shape = values.shape[1:]
    if key not in group:
        if start:
            raise ValueError(f'Unexpected new array after first episode: {key}')
        group.create_dataset(key, shape=(0, *shape), chunks=(1 if image else 256, *shape),
                             dtype=values.dtype, compressor=Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE))
    arr = group[key]
    if arr.shape[1:] != shape or arr.dtype != values.dtype:
        raise ValueError(f'Array schema changed: {key}')
    arr.resize((start + len(values), *shape))
    arr[start:] = values


def append_segment(store, bag, streams, aligned, bounds, config):
    a, b = bounds
    n = b-a
    data, prov = store['data'], store['provenance']
    start = len(data['action']) if 'action' in data else 0
    quality = {}
    for key, values in (('joint_pos', aligned['state'][a:b]), ('action', aligned['action'][a:b])):
        step = float(np.abs(np.diff(values, axis=0)).max())
        if not np.isfinite(values).all() or step > config['max_joint_step_rad']:
            raise ValueError(f'{key}: nonfinite values or joint step {step} exceeds limit')
        quality[key + '_max_step_rad'] = step
        append_array(data, key, values, start)
    if not np.any(aligned['action'][a:b]):
        raise ValueError('All-zero action segment')
    for key in ('engaged', 'action_from_state', 'source_time_ns', 'receive_time_ns'):
        append_array(prov, key, aligned[key][a:b], start)
    append_array(prov, 'timestamp_ns', aligned['timeline'][a:b], start)
    for key in list(source.CAMERAS) + (['head_depth'] if config['include_depth'] else []):
        last, converted, raw_shape = None, None, None
        valid_pixels = 0
        for offset, index in enumerate(aligned['indices'][key][a:b]):
            if offset % 100 == 0 and shutil.disk_usage(config['output_root']).free < config['min_free_gb'] * 1e9:
                raise OSError('Available disk space below configured reserve')
            if index != last:
                raw = bag.depth(streams[key].values[index]) if key == 'head_depth' else bag.image(streams[key].values[index])
                if raw_shape is not None and raw.shape != raw_shape:
                    raise ValueError(f'{key}: source resolution changed')
                raw_shape = raw.shape
                if key == 'head_depth':
                    raw = depth_metres(raw, config['depth_scale'])
                if not np.any(raw):
                    raise ValueError(f'{key}: all-black or entirely invalid source image')
                converted = resize(raw, config['image_width'], config['image_height'], key == 'head_depth')
                if not np.any(converted):
                    raise ValueError(f'{key}: no valid pixels after resizing')
                last = index
            append_array(data, key, converted[None], start + offset, image=True)
            if key == 'head_depth':
                valid_pixels += int(np.count_nonzero(converted))
        if key == 'head_depth':
            quality['depth_valid_fraction'] = valid_pixels / (n * config['image_width'] * config['image_height'])
    ends = store['meta/episode_ends']
    ends.append(np.array([start+n], dtype='i8'))
    return dict(start=start, end=start+n, length=n, source_start_ns=int(aligned['timeline'][a]),
                source_end_ns=int(aligned['timeline'][b-1]), quality=quality)


def sha256_files(path):
    result = {}
    for p in sorted(Path(path).iterdir()):
        if p.is_file():
            h = hashlib.sha256()
            with p.open('rb') as stream:
                for block in iter(lambda: stream.read(8*1024*1024), b''):
                    h.update(block)
            result[p.name] = h.hexdigest()
    return result


def select_inputs(candidates, *, episode_list=None, episodes=None, skip_ineligible=False):
    """Select all discovered bags, a TXT list, or several explicitly named episodes."""
    if episode_list is not None and episodes is not None:
        raise ValueError('Use either episode_list or episodes')
    if episode_list is not None:
        return select_episodes(candidates, episode_list, skip_ineligible)
    requested = None
    if episodes is not None:
        requested = []
        for value in episodes:
            number = str(value).removeprefix('episode')
            if not number.isascii() or not number.isdecimal():
                raise ValueError(f'Invalid episode {value!r}; use episode60 or 60')
            name = f'episode{int(number)}'
            if name in requested:
                raise ValueError(f'Duplicate episode: {name}')
            requested.append(name)
        if not requested:
            raise ValueError('Episode selection must not be empty')
        for name in requested:
            matches = [c for c in candidates if Path(c['path']).name == name]
            if not matches:
                raise ValueError(f'{name}: no bag found below source_root')
            if len(matches) > 1:
                raise ValueError(f'{name}: ambiguous name; narrow --source-root')
        candidates = [c for c in candidates if Path(c['path']).name in requested]
    skipped = [c for c in candidates if not c['eligible']]
    if skipped and not skip_ineligible:
        raise ValueError(f'{len(skipped)} ineligible bags; use --skip-ineligible to explicitly omit them')
    selection = {'skipped': skipped, 'requested_episodes': requested,
                 'selection_mode': 'episodes' if requested is not None else 'all'}
    return [c for c in candidates if c['eligible']], selection


def convert_bag(store, item, config, contract, cached_path=None):
    """Read and convert one bag; only the caller commits the returned record."""
    started = time.monotonic()
    before = snapshot(item['path'])
    with Bag(cached_path or item['path'], config['schema_dir']) as bag:
        for connection in bag.connections:
            connection.execute('PRAGMA cache_size=-131072')
            connection.execute('PRAGMA temp_store=MEMORY')
        audit = source.inspect(bag, contract)
        start, end, window = source.selected_window(item, config)
        streams = source.read_streams(bag, contract, config, end_ns=end)
        aligned = source.align(streams, start, end, config)
        if 'stream_keys' in store.attrs and list(store.attrs['stream_keys']) != aligned['stream_keys']:
            raise ValueError('Provenance stream order changed')
        store.attrs['stream_keys'] = aligned['stream_keys']
        episodes = []
        for bounds in aligned['segments']:
            result = append_segment(store, bag, streams, aligned, bounds, config)
            result['source_recording_id'] = item['source_recording_id']
            episodes.append(result)
    checksums = sha256_files(cached_path or item['path'])
    if snapshot(item['path']) != before:
        raise ValueError('Source changed during conversion')
    record = dict(source=item['path'], source_snapshot=before, source_sha256=checksums,
                  source_recording_id=item['source_recording_id'], episodes=episodes,
                  alignment=aligned['report'], inspection=audit, selection_window=window,
                  elapsed_seconds=round(time.monotonic()-started, 3))
    return record


def init_worker():
    # Each process has its own decoder and compressor; avoid N x codec threads.
    from numcodecs import blosc
    blosc.set_nthreads(1)


def stage_bag(item, config, contract, stage, cache_source=False):
    """Process entrypoint: no shared SQLite handles or writes to the final store."""
    stage = Path(stage)
    stage.mkdir(parents=True, exist_ok=False)
    store = zarr.open_group(str(stage / 'replay_buffer.zarr'), mode='w')
    recover(store, [])
    cached_path = None
    started = time.monotonic()
    if cache_source:
        before = snapshot(item['path'])
        source_bytes = sum(x['size'] for x in before)
        if shutil.disk_usage(config['output_root']).free < source_bytes + config['min_free_gb']*1e9:
            raise OSError('Insufficient space for one temporary source cache')
        cached_path = stage / 'source'
        print(f"Caching {item['path']} ({source_bytes/1e9:.2f} GB)", flush=True)
        shutil.copytree(item['path'], cached_path, copy_function=shutil.copyfile)
        if snapshot(item['path']) != before:
            raise ValueError('Source changed while it was being cached')
        print(f"Cached {item['path']}; converting locally", flush=True)
    result = convert_bag(store, item, config, contract, cached_path=cached_path)
    result['elapsed_seconds'] = round(time.monotonic()-started, 3)
    return result


def ordered_jobs(executor, items, config, contract, staging, workers, worker=stage_bag):
    """Keep <= workers running/ready stages; commit order follows source order."""
    items = iter(enumerate(items))
    pending = deque()

    def submit_next():
        entry = next(items, None)
        if entry is not None:
            index, item = entry
            stage = Path(staging) / f'{index:06d}'
            pending.append((item, stage, executor.submit(worker, item, config, contract, stage)))

    for _ in range(workers):
        submit_next()
    while pending:
        yield pending.popleft()
        # The caller commits/removes the yielded stage before resuming this generator.
        submit_next()


def append_staged(store, stage, record, config):
    """Append a private stage once, shifting all exclusive episode end offsets.

    Image chunks span exactly one frame, so compressed bytes can be copied
    without decoding or compressing a second time. Small vector arrays use
    bounded reads because episode offsets need not align to their chunks.
    """
    staged = zarr.open_group(str(Path(stage) / 'replay_buffer.zarr'), mode='r')
    keys = list(staged.attrs['stream_keys'])
    if 'stream_keys' in store.attrs and list(store.attrs['stream_keys']) != keys:
        raise ValueError('Provenance stream order changed')
    store.attrs['stream_keys'] = keys
    ends = staged['meta/episode_ends'][:]
    if not len(ends) or not np.array_equal(ends, [ep['end'] for ep in record['episodes']]):
        raise ValueError('Invalid staged episode boundaries')
    total = int(ends[-1])
    offset = len(store['data/action']) if 'action' in store['data'] else 0
    for group in ('data', 'provenance'):
        if offset and set(store[group].array_keys()) != set(staged[group].array_keys()):
            raise ValueError(f'{group}: staged array keys changed')
        for key, src in staged[group].arrays():
            if len(src) != total:
                raise ValueError(f'{key}: incomplete staged array')
            if src.ndim != 4:
                for start in range(0, total, 256):
                    append_array(store[group], key, src[start:start+256], offset+start)
                continue
            if src.chunks != (1, *src.shape[1:]):
                raise ValueError(f'{key}: expected one complete image per chunk')
            if key not in store[group]:
                if offset:
                    raise ValueError(f'Unexpected new image: {key}')
                store[group].create_dataset(key, shape=(0, *src.shape[1:]), chunks=src.chunks,
                                            dtype=src.dtype, compressor=src.compressor,
                                            filters=src.filters, fill_value=src.fill_value, order=src.order)
            dst = store[group][key]
            if (dst.shape[1:] != src.shape[1:] or dst.dtype != src.dtype or dst.chunks != src.chunks
                    or dst.compressor != src.compressor or dst.filters != src.filters or dst.order != src.order):
                raise ValueError(f'{key}: incompatible compressed image schema')
            dst.resize((offset+total, *src.shape[1:]))
            for frame in range(total):
                if frame % 100 == 0 and shutil.disk_usage(config['output_root']).free < config['min_free_gb']*1e9:
                    raise OSError('Available disk space below configured reserve')
                # Zarr v2 chunk keys include group paths and the dimension separator.
                src_key = src._chunk_key((frame, 0, 0, 0))
                dst_key = dst._chunk_key((offset+frame, 0, 0, 0))
                dst.chunk_store[dst_key] = src.chunk_store[src_key]
    store['meta/episode_ends'].append(ends + offset)
    shifted = dict(record)
    shifted['episodes'] = [dict(ep, start=ep['start']+offset, end=ep['end']+offset)
                           for ep in record['episodes']]
    return shifted


def convert(config, *, episode=None, episode_list=None, episodes=None, resume=False, skip_ineligible=False, limit=None, workers=1, cache_source=False):
    if type(workers) is not int or workers < 1:
        raise ValueError('workers must be a positive integer')
    root = output_path(config['output_root'])
    source_root = Path(config['source_root']).resolve()
    if root.is_relative_to(source_root) or source_root.is_relative_to(root):
        raise ValueError('Source and output directories overlap')
    contract = yaml.safe_load(Path(config['contract']).read_text())
    source.validate_contract(contract)
    if episode:
        source_root = (source_root / episode).resolve()
        if not source_root.is_relative_to(Path(config['source_root'])) or not (source_root / 'metadata.yaml').is_file():
            raise ValueError('Episode must name a bag below source_root')
    candidates = discover(source_root)
    selected, selection = select_inputs(candidates, episode_list=episode_list,
                                        episodes=episodes, skip_ineligible=skip_ineligible)
    if limit:
        selected = selected[:limit]
    if not selected:
        raise ValueError('No eligible bags selected')
    schemas = {p.name: p.read_text() for p in Path(config['schema_dir']).glob('*.msg')}
    fingerprint = digest(dict(config=config, contract=contract, schemas=schemas, version=VERSION))
    marker = root / 'conversion/manifest.json'
    if root.exists() and any(root.iterdir()) and (not resume or not marker.exists()):
        raise FileExistsError('Output exists; use --resume on an owned DP dataset, or a new output')
    (root / 'conversion').mkdir(parents=True, exist_ok=True)
    with (root / 'conversion/lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if marker.exists():
            manifest = json.loads(marker.read_text())
            if not resume or manifest.get('owner') != VERSION or manifest.get('fingerprint') != fingerprint:
                raise ValueError('Resume rejected: converter/config/contract differs')
        else:
            atomic_json(marker, dict(owner=VERSION, fingerprint=fingerprint, config=config, contract=contract))
        committed = records(root)
        for r in committed:
            if snapshot(r['source']) != r['source_snapshot']:
                raise ValueError(f"Previously converted source changed: {r['source']}")
        store = zarr.open_group(str(root / 'replay_buffer.zarr'), mode='a')
        recover(store, committed)
        store.attrs.update(format=VERSION, fps=config['fps'], shape_meta=shape_meta(config, contract),
                           joint_names=source.joint_names(contract), action_joint_names=source.joint_names(contract, True),
                           group_order=list(source.GROUPS), action_semantics='absolute_joint_position_radians',
                           depth_units='metres', depth_scale_16uc1=config['depth_scale'], depth_max_m=config['depth_max_m'],
                           image_transform='center_letterbox; RGB area; depth nearest; invalid depth=0')
        selection['selected'] = [item['path'] for item in selected]
        atomic_json(root / 'conversion/selection.json', selection)
        done = {r['source'] for r in committed}
        remaining = []
        for item in selected:
            if item['path'] in done:
                print(f"Already committed: {item['path']}", flush=True)
            else:
                remaining.append(item)
        errors = []
        print(f"{len(remaining)} bag(s) remaining, workers={workers}", flush=True)
        # Exit order matters: wait for child processes before cleaning their stages.
        with ExitStack() as resources:
            if (workers > 1 or cache_source) and remaining:
                stage_root = root / 'conversion/staging'
                stage_root.mkdir(exist_ok=True)
                staging = resources.enter_context(tempfile.TemporaryDirectory(prefix='run-', dir=stage_root))
                active_workers = min(workers, len(remaining))
                executor = resources.enter_context(ProcessPoolExecutor(
                    max_workers=active_workers, mp_context=multiprocessing.get_context('spawn'),
                    initializer=init_worker))
                jobs = ordered_jobs(executor, remaining, config, contract, staging, active_workers,
                                    worker=partial(stage_bag, cache_source=cache_source))
            else:
                jobs = ((item, None, None) for item in remaining)
            for item, stage, future in jobs:
                print(f"Converting {item['path']}", flush=True)
                try:
                    if future is None:
                        record = convert_bag(store, item, config, contract)
                    else:
                        record = future.result()
                        if snapshot(item['path']) != record['source_snapshot']:
                            raise ValueError('Source changed before staged data could be committed')
                        record = append_staged(store, stage, record, config)
                    atomic_json(root / f'conversion/records/{len(committed):06d}.json', record)
                    committed.append(record)
                    episodes = record['episodes']
                    print(f"Committed {len(episodes)} segment(s), {sum(e['length'] for e in episodes)} frames", flush=True)
                except Exception as exc:
                    recover(store, committed)
                    errors.append({'source': item['path'], 'error': str(exc)})
                    print(f"FAILED: {item['path']}: {exc}", flush=True)
                finally:
                    if stage is not None and stage.exists():
                        shutil.rmtree(stage)
        result = dict(bags=len(committed), episodes=sum(len(r['episodes']) for r in committed),
                      frames=int(store['meta/episode_ends'][-1]) if committed else 0, errors=errors, workers=workers)
        atomic_json(root / 'conversion_report.json', result)
        (root / 'shape_meta.yaml').write_text(yaml.safe_dump(shape_meta(config, contract), sort_keys=False))
        return result


def verify(root):
    root = output_path(root)
    with (root / 'conversion/lock').open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        store = zarr.open_group(str(root / 'replay_buffer.zarr'), mode='r')
        config = json.loads((root / 'conversion/manifest.json').read_text())['config']
        contract = json.loads((root / 'conversion/manifest.json').read_text())['contract']
        if (store.attrs.get('format') != VERSION or store.attrs.get('fps') != config['fps']
                or store.attrs.get('shape_meta') != shape_meta(config, contract)
                or store.attrs.get('joint_names') != source.joint_names(contract)
                or store.attrs.get('action_joint_names') != source.joint_names(contract, True)):
            raise ValueError('Stored schema/joint order differs from conversion manifest')
        committed = records(root)
        episodes = [e for r in committed for e in r['episodes']]
        expected = np.array([e['end'] for e in episodes], dtype='i8')
        ends = store['meta/episode_ends'][:]
        if not len(ends) or ends.dtype != np.int64 or not np.array_equal(ends, expected) or np.any(np.diff(np.r_[0, ends]) <= 0):
            raise ValueError('Invalid/uncommitted episode boundaries; resume conversion first')
        total = int(ends[-1])
        specs = {'action': (tuple(store.attrs['shape_meta']['action']['shape']), np.dtype('f4'))}
        for key, spec in store.attrs['shape_meta']['obs'].items():
            shape = tuple(spec['shape'])
            if spec['type'] in ('rgb', 'depth'):
                shape = (*shape[1:], shape[0])
            specs[key] = (shape, np.dtype('u1' if spec['type'] == 'rgb' else 'f4'))
        if set(store['data'].array_keys()) != set(specs):
            raise ValueError('Unexpected or missing data arrays')
        for key, (shape, dtype) in specs.items():
            arr = store['data'][key]
            if arr.shape != (total, *shape) or arr.dtype != dtype:
                raise ValueError(f'{key}: invalid shape/dtype')
            for i in range(0, total, 64):
                x = arr[i:i+64]
                if not np.isfinite(x).all() or (key == 'head_depth' and np.any(x < 0)):
                    raise ValueError(f'{key}: nonfinite/negative depth')
                if x.ndim == 4 and np.any(~np.any(x, axis=(1,2,3))):
                    raise ValueError(f'{key}: empty image')
        stream_keys = list(store.attrs['stream_keys'])
        prov_specs = {'timestamp_ns': ((), 'i8'), 'source_time_ns': ((len(stream_keys),), 'i8'),
                      'receive_time_ns': ((len(stream_keys),), 'i8'), 'engaged': ((4,), 'u1'),
                      'action_from_state': ((4,), 'u1')}
        if set(store['provenance'].array_keys()) != set(prov_specs):
            raise ValueError('Missing or unexpected provenance arrays')
        for key, (shape, dtype) in prov_specs.items():
            arr = store['provenance'][key]
            if arr.shape != (total, *shape) or arr.dtype != np.dtype(dtype):
                raise ValueError(f'{key}: invalid provenance shape/dtype')
        previous = 0
        for ep in episodes:
            a,b = ep['start'], ep['end']
            if a != previous or b-a != ep['length'] or b-a < config['min_segment_frames']:
                raise ValueError('Noncontiguous segment metadata')
            previous = b
            stamps = store['provenance/timestamp_ns'][a:b]
            expected_stamps = ep['source_start_ns'] + np.arange(b-a, dtype='i8')*1_000_000_000//config['fps']
            # Integer rational grids may differ by one ns after a segment cut.
            if np.any(np.abs(stamps-expected_stamps)>1) or stamps[-1] != ep['source_end_ns']:
                raise ValueError('A segment bridges a time gap')
            times = store['provenance/source_time_ns'][a:b]
            if np.any(times > stamps[:,None]) or np.any(times <= 0):
                raise ValueError('Future or missing source sample')
            engaged = store['provenance/engaged'][a:b].astype(bool)
            for j, key in enumerate(stream_keys):
                if key.startswith('state.'):
                    limit = config['state_max_age_ms']
                elif key.startswith('status.'):
                    limit = config['status_max_age_ms']
                elif key.startswith('action.'):
                    limit = config['command_max_age_ms']
                else:
                    limit = config['image_max_age_ms']
                good = stamps-times[:,j] <= round(limit*1e6)
                if key.startswith('action.'):
                    good |= ~engaged[:,source.GROUPS.index(key.removeprefix('action.'))]
                if not good.all():
                    raise ValueError(f'{key}: source sample exceeds freshness limit')
            for key in ('joint_pos', 'action'):
                x = store['data'][key][a:b]
                if np.abs(np.diff(x, axis=0)).max() > config['max_joint_step_rad']:
                    raise ValueError(f'{key}: joint jump')
                if key == 'action' and not np.any(x):
                    raise ValueError('All-zero action segment')
        result = dict(ok=True, bags=len(committed), episodes=len(ends), frames=total,
                      action_dim=specs['action'][0][0], full_image_read=True,
                      arrays={k:list(store['data'][k].shape) for k in specs})
        atomic_json(root / 'verification.json', result)
        return result
