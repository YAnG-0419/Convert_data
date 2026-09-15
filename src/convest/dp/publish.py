"""Local bounded conversion, verified ZIP upload, and resumable dataset manifest."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import deque
import fcntl
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil
import time
import zipfile

import zarr

from convest.config import WORKSPACE, atomic_json, digest
from convest.dp.convert import convert, load_config, output_path, verify, records
from convest.dp.source import selected_window
from convest.sources.gello_rosbag2 import discover, snapshot

FORMAT = 'franka-dp-zarr-shards-v1'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def convert_part(item, config, local):
    local = Path(local)
    local.mkdir(parents=True, exist_ok=True)
    with (local/'job.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ready = local/'part.json'
        if ready.exists():
            info = json.loads(ready.read_text())
            archive = local/'replay_buffer.zarr.zip'
            if (info['record']['source_snapshot'] != snapshot(item['path'])
                    or sha256(archive) != info['sha256']):
                raise ValueError('Retained local artifact or source changed')
            return info
        # A previous interrupted job can leave a source cache with no commit.
        stale = local/'dataset/conversion/staging'
        if stale.exists():
            shutil.rmtree(stale)
        return _convert_part(item, config, local)


def _convert_part(item, config, local):
    cfg = dict(config, source_root=item['path'], output_root=str(local/'dataset'))
    result = convert(cfg, cache_source=True, resume=True)
    if result['errors'] or result['bags'] != 1:
        raise ValueError(f"Conversion failed: {result}")
    check = verify(cfg['output_root'])
    rec = records(Path(cfg['output_root']))[0]
    window = rec['selection_window']
    store = zarr.open_group(str(local/'dataset/replay_buffer.zarr'), mode='r')
    stamps = store['provenance/timestamp_ns'][:]
    if stamps.max() > window['end_ns_inclusive']:
        raise ValueError('Output contains a frame outside selected window')
    if snapshot(item['path']) != item['snapshot']:
        raise ValueError('Source differs from frozen inventory')
    archive = local/'replay_buffer.zarr.zip'
    with zipfile.ZipFile(archive.with_suffix('.partial'), 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as out:
        root = local/'dataset/replay_buffer.zarr'
        for p in sorted(root.rglob('*')):
            if p.is_file():
                out.write(p, p.relative_to(root).as_posix())
    archive.with_suffix('.partial').replace(archive)
    with zipfile.ZipFile(archive) as out:
        bad = out.testzip()
        if bad:
            raise ValueError(f'ZIP CRC failed: {bad}')
    info = dict(episode=Path(item['path']).name, archive=f"episodes/{Path(item['path']).name}/replay_buffer.zarr.zip",
                bytes=archive.stat().st_size, sha256=sha256(archive), frames=check['frames'],
                record=rec, verification=check, attrs=dict(store.attrs),
                arrays={k:dict(shape=list(a.shape),dtype=a.dtype.str) for k,a in store['data'].arrays()})
    atomic_json(local/'part.json', info)
    # Only this job's newly generated, now archived intermediate is removed.
    shutil.rmtree(local/'dataset')
    return info


def upload_part(info, local, destination):
    target = destination/info['archive']
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size != info['bytes'] or sha256(target) != info['sha256']:
            raise ValueError(f'Refusing to overwrite different existing archive: {target}')
    else:
        temporary = target.with_suffix('.uploading')
        with (Path(local)/'replay_buffer.zarr.zip').open('rb') as src, temporary.open('wb', buffering=8*1024*1024) as dst:
            shutil.copyfileobj(src, dst, length=8*1024*1024)
            dst.flush()
        if temporary.stat().st_size != info['bytes'] or sha256(temporary) != info['sha256']:
            raise ValueError('Uploaded archive SHA-256 mismatch; local artifact retained')
        temporary.replace(target)
    atomic_json(target.parent/'record.json', info['record'])
    atomic_json(target.parent/'verification.json', info['verification'])
    return dict(info, upload_verified=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,default=WORKSPACE.parent/'company-remote/tomato')
    parser.add_argument('--destination',type=Path,default=WORKSPACE.parent/'company-remote/tomato-dp')
    parser.add_argument('--local-root',type=Path,default=WORKSPACE/'outputs/dp_first_stage_publish')
    parser.add_argument('--config',type=Path,default=WORKSPACE/'configs/franka_dp.yaml')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if args.workers<1 or (args.limit is not None and args.limit<1):parser.error('Counts must be positive')
    src,dest=args.source_root.resolve(),args.destination.resolve()
    allowed=WORKSPACE.parent/'company-remote/tomato-dp'
    if not dest.is_relative_to(allowed.resolve()):parser.error('Destination must be inside the authorized tomato-dp directory')
    if dest.is_relative_to(src) or src.is_relative_to(dest):parser.error('Source and destination overlap')
    local=output_path(args.local_root);local.mkdir(parents=True,exist_ok=True)
    cfg=load_config(args.config);cfg.update(fps=30,segment_mode='first_stage',source_root=str(src))
    cfg['output_root']=str(local)
    selected=[];skipped=[]
    for item in discover(src):
        if not item['eligible']:
            skipped.append(item);continue
        try:
            start,end,window=selected_window(item,cfg)
            grid_frames=(end-start)*30//1_000_000_000+1
            if grid_frames<cfg['min_segment_frames']:
                raise ValueError(f'Selected window has only {grid_frames} frames; minimum is {cfg["min_segment_frames"]}')
        except ValueError as exc:
            skipped.append(dict(item,eligible=False,collection_eligible=item['eligible'],reason=str(exc)));continue
        selected.append(dict(item,window=window,expected_grid_frames=(end-start)*30//1_000_000_000+1,
                             snapshot=snapshot(item['path'])))
    if args.limit:selected=selected[:args.limit]
    contract=json.loads(json.dumps(__import__('yaml').safe_load(Path(cfg['contract']).read_text())))
    signature=digest(dict(config=cfg,contract=contract,selection=selected))
    inventory=dict(format=FORMAT,fingerprint=signature,selected=selected,skipped=skipped,config=cfg,
                   single_stage=sum(not x['milestones']for x in selected),prefix_stage=sum(bool(x['milestones'])for x in selected),
                   source_bytes=sum(x['bytes']for x in selected),expected_grid_frames=sum(x['expected_grid_frames']for x in selected))
    atomic_json(local/'inventory.json',inventory)
    print(json.dumps({k:v for k,v in inventory.items() if k not in ('selected','skipped','config')},indent=2),flush=True)
    print(f'Selected {len(selected)}; skipped {len(skipped)}; destination={dest}',flush=True)
    if args.dry_run:return
    dest.mkdir(parents=True,exist_ok=True)
    with (local/'publish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest_path=dest/'dataset_manifest.json'
        if manifest_path.exists():
            manifest=json.loads(manifest_path.read_text())
            if manifest['fingerprint']!=signature:raise ValueError('Destination belongs to a different selection/configuration')
        else:
            if any(dest.iterdir()):raise ValueError('Destination is not empty and has no owned manifest')
            manifest=dict(format=FORMAT,fingerprint=signature,status='running',fps=30,segment_mode='first_stage',
                          expected_recordings=len(selected),parts=[],errors=[],created_unix=time.time())
        atomic_json(dest/'inventory.json',inventory)
        completed={p['episode']for p in manifest['parts']}
        for part in manifest['parts']:
            archive=dest/part['archive']
            if not archive.is_file() or archive.stat().st_size!=part['bytes']:
                raise ValueError(f'Previously uploaded archive missing/truncated: {archive}')
        remaining_items=[x for x in selected if Path(x['path']).name not in completed]
        # Exercise a real milestone prefix early, before the long single-stage backlog.
        if not any(p['record']['selection_window']['milestone'] for p in manifest['parts']):
            first_prefix=next((i for i,x in enumerate(remaining_items) if x['milestones']),None)
            if first_prefix is not None:
                remaining_items.insert(0,remaining_items.pop(first_prefix))
        remaining=deque(remaining_items)
        budget=shutil.disk_usage(local).free-8*10**9
        if budget<=0:raise OSError('Need at least 8 GB free reserve')
        pending=deque();reserved=0;manifest['errors']=[]
        def save():
            manifest.update(recordings=len(manifest['parts']),frames=sum(p['frames']for p in manifest['parts']),
                            uploaded_bytes=sum(p['bytes']for p in manifest['parts']),updated_unix=time.time())
            atomic_json(local/'progress.json',manifest);atomic_json(manifest_path,manifest)
        save()
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            while remaining or pending:
                while remaining and len(pending)<args.workers:
                    item=remaining[0];cost=int(item['bytes']*1.6)+10**9
                    if reserved+cost>budget:
                        if not pending:raise OSError(f'Not enough local space for {item["path"]}; estimated working bytes {cost}')
                        break
                    remaining.popleft();name=Path(item['path']).name;job=local/'jobs'/name
                    print(f'LOCAL START {name}; queued={len(remaining)}',flush=True)
                    future=pool.submit(convert_part,item,cfg,str(job));pending.append((item,job,cost,future));reserved+=cost
                item,job,cost,future=pending.popleft()
                try:
                    info=future.result()
                    print(f'UPLOAD {info["episode"]}: {info["frames"]} frames, {info["bytes"]/1e9:.2f} GB',flush=True)
                    part=upload_part(info,job,dest)
                    manifest['parts'].append(part)
                    save()  # Remote commitment precedes local cleanup.
                    shutil.rmtree(job)
                    print(f'UPLOADED + VERIFIED {part["episode"]}; {len(manifest["parts"])}/{len(selected)}',flush=True)
                except Exception as exc:
                    manifest['errors'].append(dict(source=item['path'],error=str(exc)))
                    save();print(f'FAILED {item["path"]}: {exc}',flush=True)
                    # Keep failed artifacts for diagnosis; stop if storage is tight.
                    if shutil.disk_usage(local).free<8*10**9:raise
                finally:
                    reserved-=cost
        if any(snapshot(item['path'])!=item['snapshot'] for item in selected):
            manifest['errors'].append(dict(error='Source snapshot changed since inventory'))
        manifest['status']='complete' if len(manifest['parts'])==len(selected) and not manifest['errors'] else 'incomplete'
        manifest['source_snapshots_unchanged']=not any(snapshot(item['path'])!=item['snapshot'] for item in selected)
        save();print(f'FINISHED {manifest["status"]}: {manifest["recordings"]} recordings, {manifest["frames"]} frames',flush=True)
        if manifest['status']!='complete':raise SystemExit(1)


if __name__=='__main__':main()
