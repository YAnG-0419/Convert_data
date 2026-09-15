#!/usr/bin/env python3
"""Exercise the real DP Dataset, encoder, diffusion loss/backward and prediction."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT / 'src'), os.environ.get('DP_ROOT', str(ROOT.parent / 'diffusion_policy'))]
os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT / '.cache/numba'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.report and not args.report.resolve().is_relative_to(ROOT):
        parser.error('Report must be inside Convert_data')
    import numpy as np
    import torch
    import zarr
    from torch.utils.data import DataLoader
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
    from convest.dp.dataset import FrankaImageDataset
    from convest.dp.encoder import RGBDObsEncoder
    torch.set_num_threads(2)
    torch.manual_seed(42)
    store = zarr.open_group(str(args.dataset / 'replay_buffer.zarr'), mode='r')
    shape = store.attrs['shape_meta']
    dataset = FrankaImageDataset(args.dataset, shape, horizon=16, n_obs_steps=2)
    batch = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))
    normalizer = dataset.get_normalizer()
    # A reduced U-Net exercises the same official code at a practical CPU cost.
    policy = DiffusionUnetImagePolicy(
        shape_meta=shape, noise_scheduler=DDPMScheduler(num_train_timesteps=10,
        beta_schedule='squaredcos_cap_v2', prediction_type='epsilon', clip_sample=True),
        obs_encoder=RGBDObsEncoder(shape), horizon=16, n_action_steps=8, n_obs_steps=2,
        num_inference_steps=2, diffusion_step_embed_dim=32, down_dims=(64,128), n_groups=8)
    policy.set_normalizer(normalizer)
    policy.to(args.device)
    batch = {'obs': {k:v.to(args.device) for k,v in batch['obs'].items()}, 'action': batch['action'].to(args.device)}
    loss = policy.compute_loss(batch)
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite diffusion loss')
    loss.backward()
    gradient_values = [p.grad for p in policy.parameters() if p.grad is not None]
    if not gradient_values or not all(torch.isfinite(g).all() for g in gradient_values):
        raise ValueError('Missing or nonfinite gradients')
    policy.eval()
    with torch.no_grad():
        result = policy.predict_action(batch['obs'])
    if result['action'].shape != (1,8,shape['action']['shape'][0]) or not torch.isfinite(result['action']).all():
        raise ValueError('Invalid predicted action')
    # Verify boundary padding uses the same observation as the corresponding action time.
    first = dataset[0]
    if not torch.equal(first['obs']['joint_pos'][0], first['obs']['joint_pos'][1]):
        raise ValueError('Beginning-of-episode observation padding mismatch')
    report = dict(ok=True, dataset=str(args.dataset.resolve()), train_sequences=len(dataset),
                  validation_sequences=len(dataset.get_validation_dataset()),
                  obs_shapes={k:list(v.shape) for k,v in batch['obs'].items()},
                  action_shape=list(batch['action'].shape), prediction_shape=list(result['action'].shape),
                  loss=float(loss.detach().cpu()), finite_gradients=True,
                  torch_version=torch.__version__, numpy_version=np.__version__,
                  device=args.device, test_unet_down_dims=[64,128], inference_steps=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
