#!/usr/bin/env python3
"""Run the official workspace with the local RGB-D task and encoder.

Run with a working official DP Python environment, not the conversion venv.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('CONVERT_ROOT', str(ROOT))
os.environ.setdefault('DP_ROOT', str(ROOT.parent / 'diffusion_policy'))
os.environ.setdefault('DP_DATASET', str(ROOT / 'outputs/dp/tomato_30hz'))
os.environ.setdefault('WANDB_CACHE_DIR', str(ROOT / '.cache/wandb'))
os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT / '.cache/numba'))
sys.path[:0] = [str(ROOT / 'src'), os.environ['DP_ROOT']]
sys.dont_write_bytecode = True

import hydra
from omegaconf import OmegaConf

OmegaConf.register_new_resolver('eval', eval, replace=True)


@hydra.main(version_base=None, config_path=str(ROOT / 'configs'), config_name='train_franka_dp')
def main(cfg):
    workspace = hydra.utils.get_class(cfg._target_)(cfg)
    if cfg.get('resume_checkpoint'):
        workspace.load_checkpoint(path=cfg.resume_checkpoint, map_location='cpu', weights_only=False)
    workspace.run()
    if cfg.get('save_final_checkpoint', False):
        import json
        if workspace._saving_thread is not None:
            workspace._saving_thread.join()
        checkpoint = workspace.save_checkpoint(tag='final', use_thread=False)
        rows=[json.loads(line) for line in Path(workspace.output_dir,'logs.json.txt').read_text().splitlines()]
        optimizer_updates=len({r['global_step'] for r in rows if 'train_loss' in r})
        Path(workspace.output_dir, 'training_complete.json').write_text(json.dumps(
            dict(epoch=workspace.epoch, global_step=workspace.global_step, optimizer_updates=optimizer_updates,
                 final_checkpoint=checkpoint), indent=2)+'\n')


if __name__ == '__main__':
    main()
