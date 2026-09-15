import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from convest.dp import source
from convest.dp.convert import depth_metres, load_config, recover, resize, output_path
from convest.sources.gello_rosbag2 import Series


def config():
    cfg = load_config(Path(__file__).resolve().parents[1] / 'configs/franka_dp.yaml')
    cfg['fps'] = 20  # Existing fixtures explicitly use a 50 ms source grid.
    return cfg


def streams(times):
    result = {}
    for group in source.GROUPS:
        for kind in ('state', 'action'):
            result[f'{kind}.{group}'] = Series.build([(t,np.array([i+1],dtype='f4'),t+100) for i,t in enumerate(times)])
        result[f'status.{group}'] = Series.build([(t,(True,True),t+100) for t in times])
    for key in list(source.CAMERAS)+['head_depth']:
        result[key] = Series.build([(t,('reference',i),t+100) for i,t in enumerate(times)])
    return result


def test_causal_selection_and_gap_split():
    cfg = config()
    cfg['min_segment_frames'] = 2
    base = 1_700_000_000_000_000_000
    times = base+np.arange(20,dtype='i8')*50_000_000
    s = streams(times)
    # At t0 there is no camera sample; later observations must use the preceding one.
    s['head_rgb'] = Series.build([(int(t+1),('ref',i),int(t+100)) for i,t in enumerate(times) if i not in (5,6,7,8)])
    a = source.align(s,int(times[0]),int(times[-1]),cfg)
    assert len(a['segments']) == 2
    for lo,hi in a['segments']:
        assert np.all(a['source_time_ns'][lo:hi] <= a['timeline'][lo:hi,None])
    assert a['indices']['head_rgb'][1] == 0
    assert all(not (lo <= 7 < hi) for lo,hi in a['segments'])


def test_inactive_hold_is_fixed_at_transition():
    cfg = config(); cfg['min_segment_frames'] = 2
    t = 1_000_000_000+np.arange(20)*50_000_000
    s = streams(t)
    flags = [True]*3+[False]*5+[True]*4+[False]*8
    s['status.left_arm'] = Series.build([(x,(flag,True),x+100) for x,flag in zip(t,flags)])
    a = source.align(s,int(t[0]),int(t[-1]),cfg)
    np.testing.assert_array_equal(a['action'][3:8,0], [4]*5)
    np.testing.assert_array_equal(a['action'][12:,0], [13]*8)
    col = a['stream_keys'].index('action.left_arm')
    np.testing.assert_array_equal(a['source_time_ns'][12:,col], [t[12]]*8)


def test_initial_inactive_fallback_never_uses_future_measurement():
    cfg=config(); cfg['min_segment_frames']=2
    t=1_000_000_000+np.arange(20)*50_000_000
    s=streams(t)
    s['action.left_arm'] = Series.build([(t[10],np.array([7],dtype='f4'),t[10]+100)])
    s['status.left_arm'] = Series.build([(x,(False,True),x+100) for x in t])
    a=source.align(s,int(t[0]),int(t[-1]),cfg)
    assert np.all(a['action'][:,0]==1)
    assert np.all(a['action_from_state'][:,0]==1)
    s['state.left_arm'] = Series.build([(x,np.array([1],dtype='f4'),x+100) for x in t[1:]])
    with pytest.raises(ValueError,match='No sufficiently long'):
        source.align(s,int(t[0]),int(t[-1]),cfg)


def test_status_fault_excludes_timesteps():
    cfg=config(); cfg['min_segment_frames']=2
    t=1_000_000_000+np.arange(20)*50_000_000
    s=streams(t)
    s['status.right_hand']=Series.build([(x,(True,i!=10),x+100) for i,x in enumerate(t)])
    a=source.align(s,int(t[0]),int(t[-1]),cfg)
    assert a['segments']==[(0,10),(11,20)]


def test_depth_units_invalid_values_and_nearest_padding():
    np.testing.assert_allclose(depth_metres(np.array([[0,1000,2000]],dtype='>u2'),.001),[[0,1,2]])
    np.testing.assert_array_equal(depth_metres(np.array([[np.nan,np.inf,-1,1]],dtype='f4'),.001),[[0,0,0,1]])
    raw=np.ones((4,8),dtype='f4')*2
    out=resize(raw,4,4,depth=True)
    assert out.shape==(4,4,1)
    assert set(np.unique(out)) == {0,2}
    assert np.all(out[0]==0) and np.all(out[-1]==0)
    rgb=np.zeros((4,8,3),dtype='u1'); rgb[...,0]=255
    resized=resize(rgb,4,4)
    assert resized.shape==(4,4,3) and not np.any(resized[...,1:])


def test_recovery_rolls_back_only_uncommitted_tail(tmp_path):
    store=zarr.open_group(str(tmp_path/'replay.zarr'),mode='w')
    data=store.create_group('data')
    data.create_dataset('action',data=np.arange(12,dtype='f4').reshape(6,2))
    prov=store.create_group('provenance')
    prov.create_dataset('timestamp_ns',data=np.arange(5,dtype='i8'))
    meta=store.create_group('meta'); meta.create_dataset('episode_ends',data=np.array([4,6],dtype='i8'))
    committed=[{'episodes':[{'end':4}]}]
    recover(store,committed)
    assert store['data/action'].shape==(4,2)
    np.testing.assert_array_equal(store['meta/episode_ends'][:],[4])
    np.testing.assert_array_equal(store['data/action'][:],np.arange(8).reshape(4,2))
    store['data/action'].resize((3,2))
    with pytest.raises(ValueError,match='truncated'):
        recover(store,committed)


def test_output_cannot_escape_workspace():
    with pytest.raises(ValueError,match='child directory'):
        output_path('/home/descfly/lcl_work/diffusion_policy/new_data')


def test_config_rejects_invalid_values(tmp_path):
    import yaml
    cfg=config(); cfg['depth_scale']=float('nan')
    p=tmp_path/'config.yaml'; p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match='depth_scale'):
        load_config(p)


def test_30hz_grid_has_no_rounding_drift():
    cfg = config(); cfg['fps'] = 30
    base = 1_700_000_000_000_000_000
    times = base + np.arange(91, dtype='i8') * 1_000_000_000 // 30
    s = streams(times)
    # A 20 Hz camera is held causally on the faster output grid.
    camera_times = base + np.arange(61, dtype='i8') * 50_000_000
    s['head_rgb'] = Series.build([(int(t), ('ref', i), int(t+100))
                                 for i,t in enumerate(camera_times)])
    aligned = source.align(s, int(times[0]), int(times[-1]), cfg)
    assert aligned['segments'] == [(0, 91)]
    np.testing.assert_array_equal(aligned['timeline'], times)
    assert set(np.diff(aligned['timeline'])) == {33_333_333, 33_333_334}
    assert int(aligned['timeline'][-1] - aligned['timeline'][0]) == 3_000_000_000
    assert np.all(aligned['source_time_ns'] <= times[:, None])
    assert aligned['indices']['head_rgb'][0] == aligned['indices']['head_rgb'][1] == 0


def test_batch_selects_multiple_episodes_and_preserves_boundaries_in_input_list():
    from convest.dp.convert import select_inputs
    bags = [{'path':f'/bags/episode{i}', 'eligible':True} for i in (60,62,63,64)]
    selected, report = select_inputs(bags, episodes=['episode60', '62', 'episode63'])
    assert [Path(x['path']).name for x in selected] == ['episode60','episode62','episode63']
    assert report['requested_episodes'] == ['episode60','episode62','episode63']
    assert report['selection_mode'] == 'episodes'
    assert select_inputs(bags)[0] == bags
    for names, match in [(['60','episode60'],'Duplicate'), (['61'],'no bag'), (['../60'],'Invalid')]:
        with pytest.raises(ValueError,match=match):
            select_inputs(bags, episodes=names)
    with pytest.raises(ValueError,match='ambiguous'):
        select_inputs(bags+[{'path':'/other/episode60','eligible':True}], episodes=['60'])


def test_batch_does_not_silently_drop_ineligible_inputs():
    from convest.dp.convert import select_inputs
    bags = [{'path':'/bags/episode60','eligible':True},
            {'path':'/bags/episode64','eligible':False,'reason':'incomplete'}]
    with pytest.raises(ValueError,match='ineligible'):
        select_inputs(bags)
    selected, report = select_inputs(bags, episodes=['60','64'], skip_ineligible=True)
    assert selected == bags[:1]
    assert report['skipped'] == bags[1:]
    assert select_inputs(bags,episodes=['60'])[0] == bags[:1]
