#!/usr/bin/env python3
"""Local read-only dashboard for the selected-batch DP experiment."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'outputs/dp_pilot_20260915'
DATA=ROOT/'outputs/dp/pilot_high_30hz'
TRAIN=ROOT/'outputs/dp_training/pilot_high_30hz'


def read_json(path, default=None):
    try: return json.loads(path.read_text())
    except (OSError,ValueError): return default


def status():
    selection=read_json(ART/'selection.json',{})
    records=[read_json(p,{}) for p in sorted((DATA/'conversion/records').glob('*.json'))]
    stages=[]
    for p in sorted((DATA/'conversion/staging').glob('run-*/*/replay_buffer.zarr')):
        arrays={}
        for key in ('action','head_rgb','left_wrist_rgb','right_wrist_rgb','head_depth'):
            meta=read_json(p/'data'/key/'.zarray',{})
            arrays[key]=meta.get('shape',[0])[0]
        stages.append({'job':p.parent.name,'arrays':arrays,
                       'cached_gb':round(sum(f.stat().st_size for f in (p.parent/'source').glob('*.db3'))/1e9,2)})
    logs=[]
    try:
        for line in (TRAIN/'logs.json.txt').read_text().splitlines():
            try: logs.append(json.loads(line))
            except ValueError: pass
    except OSError: pass
    return dict(selection=selection,converted_bags=len(records),
                frames=sum(sum(e['length'] for e in r.get('episodes',[])) for r in records),
                completed=[Path(r.get('source','')).name for r in records], stages=stages,
                training=logs,optimizer_updates=len({r['global_step'] for r in logs if 'train_loss' in r}),metrics=read_json(ART/'evaluation.json'),
                plan=read_json(ART/'experiment.json',{}),
                conversion=read_json(DATA/'conversion_report.json'),
                training_complete=(TRAIN/'training_complete.json').exists(),
                assets={name:int((ART/name).stat().st_mtime) for name in
                        ('contact_sheet.png','training_curves.png','data_preview.mp4','prediction_demo.mp4','evaluation_errors.png')
                        if (ART/name).exists()})


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.split('?')[0]=='/status.json':
            data=json.dumps(status(),ensure_ascii=False,allow_nan=False).encode()
            self.send_response(200); self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Cache-Control','no-store'); self.end_headers(); self.wfile.write(data)
        else: super().do_GET()
    def log_message(self,*args): pass


if __name__=='__main__':
    ART.mkdir(parents=True,exist_ok=True)
    print('DP experiment dashboard: http://127.0.0.1:8765',flush=True)
    ThreadingHTTPServer(('127.0.0.1',8765),partial(Handler,directory=str(ART))).serve_forever()
