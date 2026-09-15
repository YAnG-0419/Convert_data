#!/usr/bin/env python3
"""Offline held-out action evaluation; camera replay never represents a robot rollout."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'outputs/dp_pilot_20260915'
DATA=ROOT/'outputs/dp/pilot_high_30hz'
TRAIN=ROOT/'outputs/dp_training/pilot_high_30hz'
sys.path[:0]=[str(ROOT/'src'),str(ROOT.parent/'diffusion_policy')]
sys.dont_write_bytecode=True
os.environ.setdefault('CONVERT_ROOT',str(ROOT))
os.environ.setdefault('DP_DATASET',str(DATA))
os.environ.setdefault('NUMBA_CACHE_DIR',str(ROOT/'.cache/numba'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',type=Path,default=TRAIN/'checkpoints/final.ckpt')
    parser.add_argument('--contexts-per-recording',type=int,default=96)
    parser.add_argument('--inference-steps',type=int,default=100,help='Use all DDPM steps for the installed legacy scheduler')
    parser.add_argument('--output',type=Path,default=ART)
    args=parser.parse_args()
    if args.contexts_per_recording < 1:parser.error('--contexts-per-recording must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    import numpy as np
    import torch
    import hydra
    import dill
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver('eval',eval,replace=True)
    torch.set_num_threads(2);torch.manual_seed(123)
    payload=torch.load(args.checkpoint,map_location='cpu',pickle_module=dill,weights_only=False)
    cfg=payload['cfg']
    if args.inference_steps is not None:
        if not 1 <= args.inference_steps <= cfg.policy.noise_scheduler.num_train_timesteps:
            parser.error('--inference-steps must be between 1 and the training diffusion step count')
        cfg.policy.num_inference_steps=args.inference_steps
    checkpoint_epoch=dill.loads(payload['pickles']['epoch'])
    checkpoint_global_step=dill.loads(payload['pickles']['global_step'])
    dataset=hydra.utils.instantiate(cfg.task.dataset)
    validation=dataset.get_validation_dataset()
    policy=hydra.utils.instantiate(cfg.policy)
    policy.set_normalizer(dataset.get_normalizer())
    weight_key='ema_model' if cfg.training.use_ema else 'model'
    policy.load_state_dict(payload['state_dicts'][weight_key]);del payload
    policy.cuda().eval()
    records=[json.loads(p.read_text()) for p in sorted((DATA/'conversion/records').glob('*.json'))]
    selection=json.loads((ART/'selection.json').read_text())
    validation_ids=set(selection['validation_recording_ids'])
    ends=dataset.replay_buffer.episode_ends[:];starts=np.r_[0,ends[:-1]]
    train_actions=np.concatenate([dataset.replay_buffer['action'][int(a):int(b)]
                                  for a,b,keep in zip(starts,ends,dataset.train_mask) if keep],axis=0)
    mean_action=train_actions.mean(axis=0)
    indexes=validation.sampler.indices
    chosen=[]
    for record in records:
        if record['source_recording_id'] not in validation_ids:continue
        candidates=[]
        for ep in record['episodes']:
            mask=(indexes[:,0]>=ep['start'])&(indexes[:,1]<=ep['end'])&(indexes[:,2]==0)&(indexes[:,3]==cfg.horizon)
            candidates.extend(np.flatnonzero(mask).tolist())
        positions=np.unique(np.linspace(0,len(candidates)-1,min(args.contexts_per_recording,len(candidates))).astype(int))
        for pos in positions:
            idx=candidates[int(pos)]
            current=int(indexes[idx,0]+cfg.n_obs_steps-1)
            ep=next(e for e in record['episodes'] if e['start']<=current<e['end'])
            chosen.append(dict(dataset_index=idx,global_index=current,episode=Path(record['source']).name,
                               source_recording_id=record['source_recording_id'],
                               time_s=(current-ep['start'])/30))
    predictions=[];truth=[];holds=[]
    started=time.monotonic()
    for offset in range(0,len(chosen),8):
        rows=chosen[offset:offset+8]
        samples=[validation[r['dataset_index']] for r in rows]
        obs={key:torch.stack([s['obs'][key] for s in samples]).cuda() for key in samples[0]['obs']}
        with torch.no_grad():result=policy.predict_action(obs)
        predictions.append(result['action'].cpu().numpy())
        truth.append(np.stack([s['action'].numpy()[cfg.n_obs_steps-1:cfg.n_obs_steps-1+cfg.n_action_steps] for s in samples]))
        state=np.stack([s['obs']['joint_pos'][-1].numpy() for s in samples])
        holds.append(np.repeat(state[:,None,:],cfg.n_action_steps,axis=1))
        print(f"Evaluated {min(offset+8,len(chosen))}/{len(chosen)} held-out contexts",flush=True)
    pred,gt,hold=np.concatenate(predictions),np.concatenate(truth),np.concatenate(holds)
    if not np.isfinite(pred).all():
        raise ValueError('Nonfinite policy prediction')
    average=np.broadcast_to(mean_action,gt.shape)
    groups={'left_arm':(0,7),'right_arm':(7,14),'left_hand':(14,34),'right_hand':(34,54),'all_joints':(0,54)}
    def metrics(values,mask=None):
        error=values-gt
        if mask is not None:error=error[mask]
        return {name:{'mae_rad':float(np.abs(error[:,:,a:b]).mean()),
                      'rmse_rad':float(np.sqrt(np.square(error[:,:,a:b]).mean()))} for name,(a,b) in groups.items()}
    summary={name:metrics(values) for name,values in [('diffusion_policy',pred),('hold_current_position',hold),('training_mean_action',average)]}
    per_recording={name:{k:metrics(v,np.array([r['episode']==name for r in chosen])) for k,v in
                        [('diffusion_policy',pred),('hold_current_position',hold),('training_mean_action',average)]}
                   for name in sorted(set(r['episode'] for r in chosen))}
    report=dict(checkpoint=str(args.checkpoint),checkpoint_epoch=checkpoint_epoch,checkpoint_global_step=checkpoint_global_step,weights=weight_key,contexts=len(chosen),prediction_horizon_steps=cfg.n_action_steps,
                prediction_horizon_seconds=cfg.n_action_steps/30,units='radians',fps=30,seed=123,
                inference_steps=cfg.policy.num_inference_steps,summary=summary,per_recording=per_recording,
                duration_seconds=round(time.monotonic()-started,2),
                limitations=['Held-out validation recordings were also used for validation loss; this is not an independent test set.',
                             'Open-loop action errors do not measure task success, safety, or closed-loop robot execution.'])
    np.savez_compressed(args.output/'predictions.npz',prediction=pred,ground_truth=gt,hold=hold,mean_action=mean_action,
                        global_index=np.array([r['global_index'] for r in chosen]),
                        episode=np.array([r['episode'] for r in chosen]),time_s=np.array([r['time_s'] for r in chosen]))
    (args.output/'evaluation_contexts.json').write_text(json.dumps(chosen,indent=2)+'\n')
    (args.output/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
