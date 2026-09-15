"""Optional DP-environment tests: python -m unittest discover -s tests -p test_dp_dataset.py."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import zarr

try:
    from convest.dp.dataset import FrankaImageDataset, split_by_recording
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


@unittest.skipUnless(AVAILABLE, 'Run in a Diffusion Policy environment with torch')
class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.shape = {'action': {'shape':[1]}, 'obs': {
            'joint_pos': {'shape':[1], 'type':'low_dim'},
            'head_rgb': {'shape':[3,4,4], 'type':'rgb'},
            'head_depth': {'shape':[1,4,4], 'type':'depth'}}}
        store = zarr.open_group(str(self.root/'replay_buffer.zarr'),mode='w')
        store.attrs.update(shape_meta=self.shape, depth_max_m=2)
        data=store.create_group('data'); meta=store.create_group('meta')
        # Source A has two segments, B has one; values identify each recording.
        values=np.r_[np.arange(20),np.arange(20),100+np.arange(20)].astype('f4')[:,None]
        data.create_dataset('action',data=values)
        data.create_dataset('joint_pos',data=values)
        data.create_dataset('head_rgb',data=np.full((60,4,4,3),255,dtype='u1'))
        data.create_dataset('head_depth',data=np.full((60,4,4,1),4,dtype='f4'))
        meta.create_dataset('episode_ends',data=np.array([20,40,60],dtype='i8'))
        directory=self.root/'conversion/records'; directory.mkdir(parents=True)
        for i,(identity,ends) in enumerate([('A',[20,40]),('B',[60])]):
            (directory/f'{i:06d}.json').write_text(json.dumps({'episodes':[
                {'end':end,'source_recording_id':identity} for end in ends]}))

    def tearDown(self):
        self.tmp.cleanup()

    def test_group_split_and_train_only_statistics(self):
        ds=FrankaImageDataset(self.root,self.shape,val_ratio=.4)
        self.assertEqual(bool(ds.val_mask[0]),bool(ds.val_mask[1]))
        self.assertNotEqual(bool(ds.val_mask[0]),bool(ds.val_mask[2]))
        train_ids={g for g,keep in zip(ds.recording_ids,ds.train_mask) if keep}
        val_ids={g for g,keep in zip(ds.recording_ids,ds.val_mask) if keep}
        self.assertFalse(train_ids & val_ids)
        stats=ds.get_normalizer()['action'].get_input_stats()
        self.assertEqual(float(stats['min'][0]),0 if 'A' in train_ids else 100)
        self.assertEqual(float(stats['max'][0]),19 if 'A' in train_ids else 119)

    def test_padding_depth_and_no_cross_episode_sampling(self):
        ds=FrankaImageDataset(self.root,self.shape,val_ratio=0)
        for i in range(len(ds)):
            sample=ds[i]
            x=sample['action'].numpy()[:,0]
            self.assertLessEqual(np.abs(np.diff(x)).max(),1)
            self.assertEqual(tuple(sample['obs']['head_depth'].shape),(2,1,4,4))
            self.assertTrue(np.all(sample['obs']['head_depth'].numpy()==1))
            self.assertTrue(np.all(sample['obs']['head_rgb'].numpy()==1))
        np.testing.assert_array_equal(ds[0]['obs']['joint_pos'].numpy(),[[0],[0]])
        self.assertEqual(len(ds.get_validation_dataset()),0)

    def test_wrong_dimension_and_uncommitted_store_rejected(self):
        import copy
        wrong=copy.deepcopy(self.shape); wrong['action']['shape']=[2]
        with self.assertRaisesRegex(ValueError,'action dimension'):
            FrankaImageDataset(self.root,wrong)
        record=self.root/'conversion/records/000001.json'
        record.unlink()
        with self.assertRaisesRegex(ValueError,'Uncommitted'):
            FrankaImageDataset(self.root,self.shape)

    def test_single_recording_has_no_validation_leakage(self):
        self.assertFalse(split_by_recording(['A','A'],.2,0).any())
        with self.assertRaises(ValueError):
            split_by_recording(['A'],1,0)

    def test_explicit_recording_split(self):
        ds=FrankaImageDataset(self.root,self.shape,val_recording_ids=['B'])
        np.testing.assert_array_equal(ds.val_mask,[False,False,True])
        self.assertEqual(float(ds.get_normalizer()['action'].get_input_stats()['max'][0]),19)
        with self.assertRaisesRegex(ValueError,'Unknown validation'):
            FrankaImageDataset(self.root,self.shape,val_recording_ids=['missing'])
        with self.assertRaisesRegex(ValueError,'no training'):
            FrankaImageDataset(self.root,self.shape,val_recording_ids=['A','B'])

    def test_sharded_dataset_matches_single_store_and_worker_loading(self):
        import pickle
        import torch
        reference=FrankaImageDataset(self.root,self.shape,val_recording_ids=['B'])
        original=zarr.open_group(str(self.root/'replay_buffer.zarr'),mode='r')
        parts=[]
        for i,(a,b,ends,identity) in enumerate([(0,40,[20,40],'A'),(40,60,[20],'B')]):
            path=self.root/f'shard{i}.zip'
            with zarr.ZipStore(str(path),mode='w') as zipstore:
                group=zarr.group(zipstore);group.attrs.update(dict(original.attrs))
                data=group.create_group('data');group.create_group('meta').create_dataset('episode_ends',data=np.array(ends,dtype='i8'))
                for key,arr in original['data'].arrays():data.create_dataset(key,data=arr[a:b])
                specs={key:dict(shape=list(arr.shape),dtype=arr.dtype.str) for key,arr in data.arrays()}
            starts=[0]+ends[:-1]
            parts.append(dict(archive=path.name,frames=b-a,bytes=path.stat().st_size,upload_verified=True,
                              attrs=dict(original.attrs),arrays=specs,record={'episodes':[
                                  dict(start=x,end=y,length=y-x,source_recording_id=identity)for x,y in zip(starts,ends)]}))
        (self.root/'dataset_manifest.json').write_text(json.dumps(dict(format='franka-dp-zarr-shards-v1',status='complete',parts=parts)))
        ds=FrankaImageDataset(self.root,self.shape,val_recording_ids=['B'])
        for index in [0,20,len(ds)-1]:
            for key,value in reference[index]['obs'].items():
                np.testing.assert_array_equal(ds[index]['obs'][key],value)
            np.testing.assert_array_equal(ds[index]['action'],reference[index]['action'])
        np.testing.assert_array_equal(ds.get_validation_dataset()[0]['action'],reference.get_validation_dataset()[0]['action'])
        self.assertEqual(float(ds.get_normalizer()['action'].get_input_stats()['max'][0]),19)
        restored=pickle.loads(pickle.dumps(ds))
        np.testing.assert_array_equal(restored[0]['action'],ds[0]['action'])
        loader=torch.utils.data.DataLoader(ds,batch_size=2,num_workers=2,multiprocessing_context='spawn')
        self.assertEqual(tuple(next(iter(loader))['action'].shape),(2,16,1))
        ds.replay_buffer.close();restored.replay_buffer.close()
