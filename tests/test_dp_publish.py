import copy
import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from convest.dp import convert as dp
from convest.dp.publish import convert_part, upload_part, sha256
from convest.dp.source import selected_window
from convest.sources.gello_rosbag2 import discover, snapshot
from test_pipeline import make_bag
from test_dp_workers import fixture_config


def test_milestone_exclusive_and_single_stage_unchanged():
    item=dict(source_start_ns=100,source_end_ns=1000,milestones=[])
    assert selected_window(item,{'segment_mode':'first_stage'})[:2]==(100,1000)
    item['milestones']=[dict(clock='ros',timestamp_ns=700,id='milestone_1')]
    assert selected_window(item,{'segment_mode':'first_stage'})[:2]==(100,699)
    assert selected_window(item,{})[:2]==(100,1000)
    for stamp in [100,1001,700.0,True]:
        bad=copy.deepcopy(item);bad['milestones'][0]['timestamp_ns']=stamp
        with pytest.raises(ValueError):selected_window(bad,{'segment_mode':'first_stage'})
    for markers in ['invalid',[{},{}],[dict(clock='wall',timestamp_ns=700)]]:
        with pytest.raises(ValueError):selected_window(dict(item,milestones=markers),{'segment_mode':'first_stage'})


def test_local_crop_archive_upload_exact_bytes_and_no_source_writes(tmp_path,monkeypatch):
    cfg=fixture_config(tmp_path,monkeypatch);cfg['segment_mode']='first_stage'
    bag=Path(cfg['source_root'])/'episode9';make_bag(bag)
    p=bag/'collection_state.json';state=json.loads(p.read_text());state['milestones']=[dict(clock='ros',id='milestone_1',timestamp_ns=11_000_000_000)];p.write_text(json.dumps(state))
    item=discover(bag)[0];item['snapshot']=snapshot(bag)
    local=tmp_path/'job';info=convert_part(item,cfg,local)
    assert info['frames']==24
    assert convert_part(item,cfg,local)==info  # Reuse a verified archive after interruption.
    with zarr.ZipStore(str(local/'replay_buffer.zarr.zip'),mode='r') as zipstore:
        group=zarr.open_group(zipstore,mode='r')
        stamps=group['provenance/timestamp_ns'][:]
        assert stamps.min()==10_200_000_000 and stamps.max()<11_000_000_000
        assert np.all(group['provenance/source_time_ns'][:]<11_000_000_000)
        assert group['data/action'].shape==(24,54)
    dest=tmp_path/'remote';part=upload_part(info,local,dest)
    assert part['upload_verified'] and sha256(dest/info['archive'])==info['sha256']
    assert upload_part(info,local,dest)==part
    assert snapshot(bag)==item['snapshot']
    (dest/info['archive']).write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='overwrite'):upload_part(info,local,dest)


def test_shard_reader_cross_boundaries_and_pickling(tmp_path):
    import pickle
    from convest.dp.shards import ShardedReplayBuffer
    parts=[]
    for i,n in enumerate([5,7]):
        path=tmp_path/f'{i}.zip'
        with zarr.ZipStore(str(path),mode='w') as zipstore:
            group=zarr.group(zipstore)
            group.attrs.update(shape_meta={},depth_max_m=2)
            data=group.create_group('data');group.create_group('meta').create_dataset('episode_ends',data=np.array([n],dtype='i8'))
            data.create_dataset('action',data=np.arange(i*5,i*5+n,dtype='f4')[:,None])
        parts.append(dict(archive=path.name,frames=n,bytes=path.stat().st_size,upload_verified=True,
                          attrs={'shape_meta':{},'depth_max_m':2},arrays={'action':dict(shape=[n,1],dtype='<f4')},
                          record={'episodes':[dict(start=0,end=n,length=n,source_recording_id=str(i))]}))
    manifest=dict(format='franka-dp-zarr-shards-v1',status='complete',parts=parts)
    (tmp_path/'dataset_manifest.json').write_text(json.dumps(manifest))
    replay=ShardedReplayBuffer(tmp_path)
    np.testing.assert_array_equal(replay['action'][3:9],np.arange(3,9)[:,None])
    np.testing.assert_array_equal(replay.episode_ends,[5,12])
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(replay))['action'][:],np.arange(12)[:,None])
    assert replay['action'][-1]==11
    assert replay['action'][9:3].shape==(0,1)
    replay.close()
    manifest['status']='running';(tmp_path/'dataset_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='incomplete'):ShardedReplayBuffer(tmp_path)
