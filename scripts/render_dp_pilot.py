#!/usr/bin/env python3
"""Render learning curves and a clearly labelled recorded-view/action-prediction demo."""
import argparse
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'outputs/dp_pilot_20260915'
TRAIN=ROOT/'outputs/dp_training/pilot_high_30hz'
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image,ImageDraw
import zarr
from preview_dp_pilot import camera_row,font,write_video


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--curves-only",action="store_true")
    args=parser.parse_args()
    logs=[]
    for line in (TRAIN/'logs.json.txt').read_text().splitlines():
        try:logs.append(json.loads(line))
        except ValueError:pass
    unique={}
    for r in logs:
        if 'train_loss' in r:unique.setdefault(r['global_step'],r)
    step_number={g:i+1 for i,g in enumerate(unique)}
    train=[dict(r,global_step=step_number[g]) for g,r in unique.items()]
    val=[dict(r,global_step=step_number[r['global_step']]) for r in logs if 'val_loss' in r]
    fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained')
    x=np.array([r['global_step'] for r in train]);y=np.array([r['train_loss'] for r in train])
    ax.plot(x,y,color='#74adbd',alpha=.35,lw=.8,label='Train batch loss')
    if len(y)>=30:ax.plot(x[29:],np.convolve(y,np.ones(30)/30,mode='valid'),color='#137d82',lw=2,label='Train loss (30-step mean)')
    ax.plot([r['global_step'] for r in val],[r['val_loss'] for r in val],'o-',color='#d17424',label='Held-out validation loss')
    ax.set(xlabel='Optimizer updates',ylabel='Diffusion noise prediction MSE',title='Franka + Wuji | 9 training / 3 validation recordings')
    ax.grid(alpha=.2);ax.legend();fig.savefig(ART/'training_curves.png',dpi=180);plt.close(fig)
    if args.curves_only or not (ART/'predictions.npz').exists():return
    values=np.load(ART/'predictions.npz')
    report=json.loads((ART/'evaluation.json').read_text())
    groups=['left_arm','right_arm','left_hand','right_hand']
    fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained')
    for i,(method,label,color) in enumerate([('diffusion_policy','Diffusion Policy','#147d92'),('hold_current_position','Hold current position','#c99145'),('training_mean_action','Training mean action','#7a8697')]):
        ax.bar(np.arange(4)+(i-1)*.25,[report['summary'][method][g]['mae_rad'] for g in groups],width=.24,label=label,color=color)
    ax.set_xticks(np.arange(4),['Left arm','Right arm','Left hand','Right hand']);ax.set(ylabel='Mean absolute joint error (rad)',title='8-step action predictions on held-out validation recordings')
    ax.legend();ax.grid(axis='y',alpha=.2);fig.savefig(ART/'evaluation_errors.png',dpi=180);plt.close(fig)
    store=zarr.open_group(str(ROOT/'outputs/dp/pilot_high_30hz/replay_buffer.zarr'),mode='r')
    pred,gt=values['prediction'],values['ground_truth']
    # Select two informative joints per limb/hand using only ground-truth variation.
    groups_dims=[('Left arm',0,7),('Right arm',7,14),('Left hand',14,34),('Right hand',34,54)]
    plotted=[]
    for label,a,b in groups_dims:
        joints=np.argsort(gt[:,:,a:b].std(axis=(0,1)))[-2:]+a
        plotted.extend([(label,int(j)) for j in joints])
    def frames():
        for i,index in enumerate(values['global_index']):
            canvas=Image.new('RGB',(1280,800),'#101923');d=ImageDraw.Draw(canvas)
            name=str(values['episode'][i]);seconds=float(values['time_s'][i])
            d.text((20,12),f"HELD-OUT VALIDATION | {name} | recorded t={seconds:.1f}s",font=font(23),fill='#e5f1f8')
            d.text((20,47),'Top: recorded cameras. Bottom: model action prediction vs recorded target. No robot rollout.',font=font(17),fill='#a3bacb')
            cams=camera_row(store,int(index),400);canvas.paste(cams,(40,85))
            for px,label in zip((50,450,850),('Head RGB','Left wrist RGB','Right wrist RGB')):
                d.text((px,391),label,font=font(15),fill='#abc5d6')
            for k,(label,joint) in enumerate(plotted):
                x=24+(k%4)*314;y=425+(k//4)*154;w=294;h=108
                d.text((x,y),f'{label} | joint {joint+1}',font=font(15),fill='#dce8f0')
                all_values=np.r_[pred[:, :, joint].ravel(),gt[:, :, joint].ravel()]
                lo,hi=float(all_values.min()),float(all_values.max());margin=max((hi-lo)*.08,.02);lo-=margin;hi+=margin
                d.rectangle((x,y+25,x+w,y+25+h),outline='#344c5d')
                for values_i,color in ((gt[i,:,joint],'#efbd6d'),(pred[i,:,joint],'#71ddc7')):
                    pts=[(x+t*w/(len(values_i)-1),y+25+h-(float(v)-lo)*h/(hi-lo)) for t,v in enumerate(values_i)]
                    d.line(pts,fill=color,width=3)
                d.text((x+4,y+27),f'{hi:.2f} rad',font=font(11),fill='#8ba4b6')
                d.text((x+w-100,y+113),'0 → 0.233 s',font=font(11),fill='#8ba4b6')
            mae=float(np.abs(pred[i]-gt[i]).mean())
            d.text((24,754),'Recorded target',font=font(18),fill='#efbd6d')
            d.text((235,754),'DP prediction',font=font(18),fill='#71ddc7')
            d.text((460,754),f'54-joint MAE: {mae:.3f} rad | 8 predicted steps at 30 Hz',font=font(18),fill='#b5cbd9')
            yield canvas
    write_video(ART/'prediction_demo.mp4',frames(),fps=6)
    print('Saved training_curves.png, evaluation_errors.png and prediction_demo.mp4')


if __name__=='__main__':main()
