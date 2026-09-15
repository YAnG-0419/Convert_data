"""Lazy concatenation of standard Zarr ZIP stores for the official DP sampler."""
from collections import OrderedDict
import json
import os
from pathlib import Path

import numpy as np
import zarr


class ShardedArray:
    def __init__(self, replay, key):
        self.replay, self.key = replay, key
        spec = replay.parts[0]['arrays'][key]
        self.shape = (replay.n_steps, *spec['shape'][1:])
        self.dtype = np.dtype(spec['dtype'])
        self.ndim = len(self.shape)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        if isinstance(index, (int, np.integer)):
            index = int(index)
            if index < 0:index += len(self)
            if not 0 <= index < len(self):raise IndexError(index)
            return self[index:index+1][0]
        if not isinstance(index, slice):raise TypeError('Use an integer or contiguous slice')
        start, stop, step = index.indices(len(self))
        if step != 1:raise ValueError('Only contiguous reads are supported')
        if stop <= start:return np.empty((0,*self.shape[1:]),dtype=self.dtype)
        pieces=[]
        while start < stop:
            part = int(np.searchsorted(self.replay.part_ends, start, side='right'))
            base = int(self.replay.part_starts[part]);end=min(stop,int(self.replay.part_ends[part]))
            pieces.append(self.replay.group(part)['data'][self.key][start-base:end-base])
            start=end
        return pieces[0] if len(pieces)==1 else np.concatenate(pieces,axis=0)


class ShardedReplayBuffer:
    """Read-only ReplayBuffer interface; never joins images in RAM or across episodes."""
    def __init__(self, root):
        self.root=Path(root).resolve()
        manifest=json.loads((self.root/'dataset_manifest.json').read_text())
        if manifest.get('format')!='franka-dp-zarr-shards-v1' or manifest.get('status')!='complete':
            raise ValueError('Uncommitted/incomplete sharded dataset; finish publishing first')
        self.parts=manifest['parts']
        if not self.parts:raise ValueError('Empty sharded dataset')
        self.attrs=self.parts[0]['attrs']
        self.part_ends=np.cumsum([p['frames']for p in self.parts],dtype='i8')
        self.part_starts=np.r_[0,self.part_ends[:-1]]
        self.n_steps=int(self.part_ends[-1])
        self.episodes=[];self._cache=OrderedDict();self._pid=os.getpid()
        first=self.parts[0]
        seen=set()
        for offset,p in zip(self.part_starts,self.parts):
            archive=(self.root/p['archive']).resolve()
            if not archive.is_relative_to(self.root):raise ValueError('Archive path escapes dataset')
            if str(archive) in seen:raise ValueError('Duplicate archive')
            seen.add(str(archive))
            if not archive.is_file() or archive.stat().st_size!=p['bytes'] or not p.get('upload_verified'):
                raise ValueError(f'Missing/truncated/unverified archive: {archive}')
            if p['attrs']!=self.attrs or p['arrays'].keys()!=first['arrays'].keys():
                raise ValueError('Inconsistent shard schemas')
            for key,spec in p['arrays'].items():
                other=first['arrays'][key]
                if spec['shape'][0]!=p['frames'] or spec['shape'][1:]!=other['shape'][1:] or spec['dtype']!=other['dtype']:
                    raise ValueError('Inconsistent shard array')
            previous=0
            for ep in p['record']['episodes']:
                if ep['start']!=previous or ep['end']-ep['start']!=ep['length']:
                    raise ValueError('Invalid shard episode offsets')
                previous=ep['end']
                self.episodes.append(dict(ep,start=int(offset)+ep['start'],end=int(offset)+ep['end']))
            if previous!=p['frames']:raise ValueError('Uncommitted shard tail')
        self.episode_ends=np.array([e['end']for e in self.episodes],dtype='i8')
        self.n_episodes=len(self.episode_ends)
        self.arrays={key:ShardedArray(self,key) for key in first['arrays']}

    def group(self, index):
        if self._pid!=os.getpid():
            self.close();self._pid=os.getpid()
        if index not in self._cache:
            store=zarr.ZipStore(str(self.root/self.parts[index]['archive']),mode='r')
            group=zarr.open_group(store,mode='r')
            if dict(group.attrs)!=self.attrs or not np.array_equal(group['meta/episode_ends'][:],
                    [e['end']for e in self.parts[index]['record']['episodes']]):
                store.close();raise ValueError('ZIP metadata differs from manifest')
            self._cache[index]=(store,group)
            if len(self._cache)>8:self._cache.popitem(last=False)[1][0].close()
        self._cache.move_to_end(index)
        return self._cache[index][1]

    def close(self):
        for store,_ in self._cache.values():store.close()
        self._cache.clear()

    def __getstate__(self):
        state=self.__dict__.copy();state['_cache']=OrderedDict();state['_pid']=None
        return state

    def keys(self):return self.arrays.keys()
    def __getitem__(self,key):return self.arrays[key]
    def __contains__(self,key):return key in self.arrays
