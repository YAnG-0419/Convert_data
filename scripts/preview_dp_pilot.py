#!/usr/bin/env python3
"""Export inspectable RGB-D mosaics and recorded-video excerpts from committed data."""
import argparse
from functools import lru_cache
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import zarr

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'outputs/dp_pilot_20260915'
DATA=ROOT/'outputs/dp/pilot_high_30hz'
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


@lru_cache(maxsize=16)
def font(size): return ImageFont.truetype(FONT,size)


def camera_row(store,index,width=320):
    panels=[]
    for key in ('head_rgb','left_wrist_rgb','right_wrist_rgb'):
        img=Image.fromarray(store['data'][key][index])
        img=img.resize((width,width*3//4),Image.Resampling.BILINEAR)
        panels.append(img)
    row=Image.new('RGB',(width*3,width*3//4),'#101923')
    for i,panel in enumerate(panels): row.paste(panel,(width*i,0))
    return row


def write_video(path,frames,fps=10):
    output=av.open(str(path),'w')
    stream=output.add_stream('libx264',rate=fps)
    stream.pix_fmt='yuv420p'; stream.options={'crf':'21','preset':'fast'}
    for i,frame in enumerate(frames):
        a=np.asarray(frame)
        if i==0: stream.width=a.shape[1];stream.height=a.shape[0]
        f=av.VideoFrame.from_ndarray(a,format='rgb24')
        for packet in stream.encode(f): output.mux(packet)
    for packet in stream.encode(): output.mux(packet)
    output.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--video',action='store_true');args=parser.parse_args()
    selection=json.loads((ART/'selection.json').read_text())
    records=[json.loads(p.read_text()) for p in sorted((DATA/'conversion/records').glob('*.json'))]
    if not records: raise SystemExit('No committed data yet')
    store=zarr.open_group(str(DATA/'replay_buffer.zarr'),mode='r')
    mapping={Path(r['source']).name:r for r in records}
    sheet=Image.new('RGB',(1920,650),'#101923');draw=ImageDraw.Draw(sheet)
    draw.text((20,14),'SELECTED HUMAN-REVIEWED DATA | 3 batches x 4 recordings',font=font(25),fill='#e6f1f7')
    draw.text((20,48),'Recorded head / left wrist / right wrist views at each recording midpoint',font=font(18),fill='#a8bfce')
    for i,row in enumerate(selection['selected']):
        x=(i%4)*480;y=90+(i//4)*185
        rec=mapping.get(row['episode'])
        label=f"B{row['batch']} {row['episode']} | {row['split']} | score {row['technical_score']:.1f}"
        draw.text((x+8,y),label,font=font(15),fill='#ffca86' if row['split']=='validation' else '#70dac9')
        if rec:
            ep=max(rec['episodes'],key=lambda x:x['length'])
            sheet.paste(camera_row(store,(ep['start']+ep['end'])//2,160),(x,y+30))
        else: draw.text((x+10,y+75),'Conversion pending',font=font(18),fill='#8199a7')
    sheet.save(ART/'contact_sheet.png')
    if args.video:
        def frames():
            for batch in selection['batches']:
                for name in (batch['episodes'][0],batch['episodes'][-1]):
                    if name not in mapping:continue
                    row=next(r for r in selection['selected'] if r['episode']==name)
                    ep=max(mapping[name]['episodes'],key=lambda x:x['length'])
                    start=max(ep['start'],ep['start']+int(ep['length']*.55)-90)
                    for index in range(start,min(start+180,ep['end']),3):
                        canvas=Image.new('RGB',(960,336),'#101923');d=ImageDraw.Draw(canvas)
                        seconds=(index-ep['start'])/30
                        d.text((12,8),f"RECORDED DATA | Batch {row['batch']} | {name} | {row['split']} | t={seconds:.1f}s",font=font(19),fill='#e6f1f7')
                        canvas.paste(camera_row(store,index),(0,44))
                        for x,label in zip((12,332,652),('Head RGB','Left wrist RGB','Right wrist RGB')):
                            d.text((x,294),label,font=font(17),fill='#9ebacb')
                        yield canvas
        write_video(ART/'data_preview.mp4',frames())
    print(json.dumps({'contact_sheet':str(ART/'contact_sheet.png'),'committed':len(records),'video':args.video}))


if __name__=='__main__':main()
