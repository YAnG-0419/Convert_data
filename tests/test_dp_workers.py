"""Parallel DP tests use tiny generated CDR bags, never the user's recordings."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import shutil
import sqlite3

import numpy as np
import pytest
import zarr

from convest.config import WORKSPACE
from convest.dp import convert as dp
from convest.sources.gello_rosbag2 import snapshot
from test_pipeline import make_bag


def fixture_config(tmp_path, monkeypatch):
    cfg = dp.load_config(WORKSPACE / 'configs/franka_dp.yaml')
    cfg.update(source_root=str(tmp_path/'bags'), output_root=str(tmp_path/'parallel'),
               image_width=24, image_height=16, min_free_gb=.001)
    monkeypatch.setattr(dp, 'WORKSPACE', tmp_path)
    return cfg


@pytest.mark.parametrize("cache_source", [False, True])
def test_parallel_matches_serial_and_resumes_with_a_different_worker_count(tmp_path, monkeypatch, cache_source):
    cfg = fixture_config(tmp_path, monkeypatch)
    bags = Path(cfg['source_root'])
    make_bag(bags/'episode9')
    shutil.copytree(bags/'episode9', bags/'episode10')
    before = {p.name: snapshot(p) for p in bags.iterdir()}
    result = dp.convert(cfg, workers=2, cache_source=cache_source)
    assert result['errors'] == [] and result['bags'] == 2 and result['frames'] == 98
    assert dp.verify(Path(cfg['output_root']))['ok']
    parallel = zarr.open_group(str(Path(cfg['output_root'])/'replay_buffer.zarr'), mode='r')
    serial_cfg = dict(cfg, output_root=str(tmp_path/'serial'))
    assert dp.convert(serial_cfg, workers=1)['frames'] == 98
    serial = zarr.open_group(str(tmp_path/'serial/replay_buffer.zarr'), mode='r')
    for group in ('data','provenance','meta'):
        for key, arr in serial[group].arrays():
            np.testing.assert_array_equal(parallel[group][key][:], arr[:])
    assert before == {p.name: snapshot(p) for p in bags.iterdir()}
    writable = zarr.open_group(str(Path(cfg['output_root'])/'replay_buffer.zarr'), mode='a')
    writable['data/action'].resize((101,54))  # interrupted, uncommitted tail
    result = dp.convert(cfg, workers=3, resume=True)
    assert result['bags'] == 2 and result['frames'] == 98
    assert not list((Path(cfg['output_root'])/'conversion/staging').iterdir())
    assert dp.verify(Path(cfg['output_root']))['ok']
    np.testing.assert_array_equal(writable['data/action'][:], serial['data/action'][:])


def test_worker_failure_is_rolled_back_and_other_episodes_commit(tmp_path, monkeypatch):
    cfg = fixture_config(tmp_path, monkeypatch)
    bags = Path(cfg['source_root'])
    make_bag(bags/'episode9')
    for name in ('episode10','episode11'):
        shutil.copytree(bags/'episode9', bags/name)
    with sqlite3.connect(bags/'episode10/capture_0.db3') as db:
        db.execute("DELETE FROM topics WHERE name='/cam0/color/image_raw'")
    result = dp.convert(cfg, workers=2)
    assert result['bags'] == 2 and result['frames'] == 98
    assert len(result['errors']) == 1 and result['errors'][0]['source'].endswith('episode10')
    records = dp.records(Path(cfg['output_root']))
    assert [Path(r['source']).name for r in records] == ['episode9','episode11']
    assert records[0]['episodes'][0]['end'] == records[1]['episodes'][0]['start'] == 49
    assert not list((Path(cfg['output_root'])/'conversion/staging').iterdir())
    assert dp.verify(Path(cfg['output_root']))['ok']


def test_broken_compressed_chunk_can_be_rolled_back(tmp_path, monkeypatch):
    import yaml
    cfg = fixture_config(tmp_path, monkeypatch)
    make_bag(Path(cfg['source_root'])/'episode9')
    Path(cfg['output_root']).mkdir()
    item = dp.discover(cfg['source_root'])[0]
    contract = yaml.safe_load(Path(cfg['contract']).read_text())
    stage = tmp_path/'stage'
    record = dp.stage_bag(item, cfg, contract, stage)
    final = zarr.open_group(str(tmp_path/'merge.zarr'), mode='w')
    dp.recover(final, [])
    committed = dp.append_staged(final, stage, record, cfg)
    image = zarr.open_group(str(stage/'replay_buffer.zarr'), mode='r')['data/head_rgb']
    target = final['data/head_rgb']
    assert target.chunk_store[target._chunk_key((0,0,0,0))] == image.chunk_store[image._chunk_key((0,0,0,0))]
    # Make the stage incomplete only after the first successful append.
    (stage/'replay_buffer.zarr'/image._chunk_key((0,0,0,0))).unlink()
    with pytest.raises(KeyError):
        dp.append_staged(final, stage, record, cfg)
    dp.recover(final, [committed])
    np.testing.assert_array_equal(final['meta/episode_ends'][:], [49])
    assert all(len(a)==49 for _,a in final['data'].arrays())


def simultaneous_job(item, config, contract, stage):
    """First two jobs require one another to start, proving concurrent execution."""
    import os
    import time
    stage.mkdir()
    marker = stage.parent / f"started-{item['index']}"
    marker.write_text(str(os.getpid()))
    deadline = time.monotonic()+15
    while item['index'] < 2 and len(list(stage.parent.glob('started-*'))) < 2:
        if time.monotonic() > deadline:
            raise RuntimeError('The two jobs did not run concurrently')
        time.sleep(.01)
    return os.getpid()


def test_scheduler_is_concurrent_ordered_and_has_bounded_staging(tmp_path):
    items = [{'index':i} for i in range(5)]
    pids, order = set(), []
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as executor:
        for item, stage, future in dp.ordered_jobs(executor, items, {}, {}, tmp_path, 2, worker=simultaneous_job):
            pids.add(future.result())
            order.append(item['index'])
            assert len([p for p in tmp_path.iterdir() if p.is_dir()]) <= 2
            shutil.rmtree(stage)
    assert len(pids) == 2 and order == list(range(5))


def test_workers_must_be_positive():
    with pytest.raises(ValueError, match='workers'):
        dp.convert({}, workers=0)
