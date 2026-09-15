"""DP BaseImageDataset adapter. Importable in the official Python 3.9+ environment.

Conversion dependencies (ROS bags, PyAV) are deliberately not imported here.
Images stay on disk; only requested observation frames are decompressed.
"""
import copy
import json
from pathlib import Path

import numpy as np
import torch
import zarr
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.common.normalize_util import get_image_range_normalizer


def split_by_recording(recording_ids, val_ratio, seed):
    if not 0 <= val_ratio < 1:
        raise ValueError('val_ratio must be in [0, 1)')
    groups = sorted(set(recording_ids))
    count = min(max(1, round(len(groups)*val_ratio)), len(groups)-1) if val_ratio else 0
    val = set(np.random.default_rng(seed).choice(groups, count, replace=False))
    return np.array([key in val for key in recording_ids], dtype=bool)


class FrankaImageDataset(BaseImageDataset):
    def __init__(self, dataset_path, shape_meta, horizon=16, pad_before=1, pad_after=7,
                 n_obs_steps=2, seed=42, val_ratio=0.1, val_recording_ids=None):
        if horizon < 1 or not 1 <= n_obs_steps <= horizon:
            raise ValueError('Require 1 <= n_obs_steps <= horizon')
        if not 0 <= pad_before < horizon or not 0 <= pad_after < horizon:
            raise ValueError('Padding must be in [0, horizon)')
        root = Path(dataset_path).expanduser().resolve()
        self.path = str(root / 'replay_buffer.zarr')
        sharded = (root / 'dataset_manifest.json').exists()
        if sharded:
            from convest.dp.shards import ShardedReplayBuffer
            self.replay_buffer = ShardedReplayBuffer(root)
            attrs = self.replay_buffer.attrs
            episodes = self.replay_buffer.episodes
        else:
            store = zarr.open_group(self.path, mode='r')
            attrs = store.attrs
            self.replay_buffer = ReplayBuffer(store)
            committed = [json.loads(p.read_text()) for p in sorted((root / 'conversion/records').glob('*.json'))]
            episodes = [ep for r in committed for ep in r['episodes']]
        actual = attrs['shape_meta']
        # Allow RGB-only ablations by selecting a subset of stored observations.
        self.shape_meta = {'action': {'shape': list(shape_meta['action']['shape'])},
                           'obs': {k: {'shape': list(v['shape']), 'type': v.get('type', 'low_dim')}
                                   for k,v in shape_meta['obs'].items()}}
        if self.shape_meta['action'] != actual['action']:
            raise ValueError('Configured action dimension differs from converted data')
        for key, spec in self.shape_meta['obs'].items():
            if actual['obs'].get(key) != spec:
                raise ValueError(f'Configured observation differs from converted data: {key}')
        ends = self.replay_buffer.episode_ends[:]
        if not len(ends) or not np.array_equal(ends, [ep['end'] for ep in episodes]):
            raise ValueError('Uncommitted/empty dataset; finish or resume conversion before training')
        for key, spec in self.shape_meta['obs'].items():
            shape = tuple(spec['shape'])
            if spec['type'] in ('rgb', 'depth'):
                shape = (*shape[1:], shape[0])
            arr = self.replay_buffer[key]
            dtype = np.uint8 if spec['type'] == 'rgb' else np.float32
            if arr.shape != (int(ends[-1]), *shape) or arr.dtype != dtype:
                raise ValueError(f'{key}: stored shape/dtype differs from shape_meta')
        action = self.replay_buffer['action']
        if action.shape != (int(ends[-1]), *self.shape_meta['action']['shape']) or action.dtype != np.float32:
            raise ValueError('action: stored shape/dtype differs from shape_meta')
        self.recording_ids = [ep['source_recording_id'] for ep in episodes]
        if val_recording_ids is None:
            self.val_mask = split_by_recording(self.recording_ids, val_ratio, seed)
        else:
            requested = set(val_recording_ids)
            missing = requested - set(self.recording_ids)
            if missing:
                raise ValueError(f'Unknown validation recording IDs: {sorted(missing)}')
            self.val_mask = np.array([key in requested for key in self.recording_ids], dtype=bool)
            if self.val_mask.all():
                raise ValueError('Validation selection leaves no training recordings')
        self.train_mask = ~self.val_mask
        self.horizon, self.n_obs_steps = horizon, n_obs_steps
        self.pad_before, self.pad_after = pad_before, pad_after
        self.depth_max_m = float(attrs['depth_max_m'])
        self.sampler = self._sampler(self.train_mask)
        if not len(self.sampler):
            raise ValueError('No training sequences: check segment lengths/horizon/padding')

    def _sampler(self, mask):
        # Official sampler handles action padding and guarantees episode boundaries.
        return SequenceSampler(self.replay_buffer, sequence_length=self.horizon,
                               pad_before=self.pad_before, pad_after=self.pad_after,
                               keys=['action'], episode_mask=mask)

    def get_validation_dataset(self):
        result = copy.copy(self)
        result.sampler = self._sampler(self.val_mask)
        return result

    def get_normalizer(self, **kwargs):
        normalizer = LinearNormalizer()
        ends = self.replay_buffer.episode_ends[:]
        starts = np.r_[0, ends[:-1]]
        # Only training recordings contribute vector statistics.
        for key in ['action'] + [k for k,v in self.shape_meta['obs'].items() if v['type'] == 'low_dim']:
            values = np.concatenate([self.replay_buffer[key][int(a):int(b)]
                                     for a,b,keep in zip(starts,ends,self.train_mask) if keep], axis=0)
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(values, **kwargs)
        for key, spec in self.shape_meta['obs'].items():
            if spec['type'] in ('rgb', 'depth'):
                normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self):
        return torch.from_numpy(self.replay_buffer['action'][:])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, index):
        action = self.sampler.sample_sequence(index)['action'].astype(np.float32)
        start, end, sample_start, _ = self.sampler.indices[index]
        selected = np.clip(start + np.arange(self.n_obs_steps) - sample_start, start, end-1)
        low, high = int(selected[0]), int(selected[-1])+1
        obs = {}
        for key, spec in self.shape_meta['obs'].items():
            # A contiguous disk read followed by indexing duplicates padded edges.
            x = self.replay_buffer[key][low:high][selected-low].astype(np.float32)
            if spec['type'] == 'rgb':
                x = np.moveaxis(x, -1, 1) / 255.0
            elif spec['type'] == 'depth':
                x = np.moveaxis(np.clip(x, 0, self.depth_max_m), -1, 1) / self.depth_max_m
            obs[key] = torch.from_numpy(np.ascontiguousarray(x))
        return {'obs': obs, 'action': torch.from_numpy(action)}
