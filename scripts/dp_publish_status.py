#!/usr/bin/env python3
"""Local progress page for conversion and verified upload."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]/'outputs/dp_first_stage_publish'
PAGE='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DP 首阶段数据转换</title><style>body{background:#101923;color:#e4eef4;font:17px/1.7 system-ui;margin:40px auto;max-width:1000px;padding:20px}h1{font-size:30px}.cards{display:flex;gap:24px;flex-wrap:wrap}.card,section{background:#192b38;padding:20px;border-radius:12px;margin:15px 0}.number{font-size:34px;color:#7de0cf}progress{width:100%;height:24px}pre{white-space:pre-wrap}a{color:#80d6f2}.muted{color:#aac1d0}</style><h1>单阶段 + 记录点之前 → Diffusion Policy</h1><p class="muted">130 条单阶段 · 93 条双阶段前缀 · 30 Hz · 本机转换、逐条校验上传</p><progress id="bar" max="223"></progress><div class="cards"><div class="card">已上传并校验<div id="count" class="number">—</div></div><div class="card">有效帧<div id="frames" class="number">—</div></div><div class="card">已上传体积<div id="size" class="number">—</div></div></div><section><h2 id="phase">读取状态</h2><pre id="jobs"></pre><p id="last"></p></section><section><h2>转换规则</h2><p>单阶段保留采集校验后的有效窗口。双阶段仅使用时间戳严格早于 milestone_1 的数据。保留三路 RGB、头部深度及 54 维关节观测与目标动作。</p><p>每条 Zarr ZIP 经完整图像读取检查及上传 SHA-256 校验；原始 tomato 目录只读。</p></section><pre id="errors"></pre><p class="muted" id="time"></p><script>async function update(){try{let s=await(await fetch('/status.json')).json();document.getElementById('bar').value=s.recordings;document.getElementById('count').textContent=s.recordings+' / '+s.expected;document.getElementById('frames').textContent=s.frames.toLocaleString();document.getElementById('size').textContent=(s.bytes/1e9).toFixed(2)+' GB';document.getElementById('phase').textContent=s.status==='complete'?'全部完成':'本机转换 / 上传中';document.getElementById('jobs').textContent=s.jobs.map(j=>j.episode+'：'+j.state).join('\n');document.getElementById('last').textContent='最近上传：'+s.last.join('、');document.getElementById('errors').textContent=s.errors.length?JSON.stringify(s.errors,null,2):'';document.getElementById('time').textContent='每 10 秒刷新 · '+new Date().toLocaleString()}catch(e){document.getElementById('time').textContent=e.message}}update();setInterval(update,10000)</script></html>'''


def status():
    p=json.loads((ROOT/'progress.json').read_text())
    jobs=[]
    for folder in sorted((ROOT/'jobs').glob('*')):
        archives=list(folder.glob('*.zip'))
        caches=list(folder.glob('dataset/conversion/staging/run-*/*/source/*.db3'))
        if archives:state=f'已归档 {sum(x.stat().st_size for x in archives)/1e9:.2f} GB，等待或正在上传'
        elif caches:state=f'本机缓存 {sum(x.stat().st_size for x in caches)/1e9:.2f} GB，读取 / 转换中'
        else:state='转换 / 校验中'
        jobs.append(dict(episode=folder.name,state=state))
    return dict(status=p['status'],recordings=p['recordings'],expected=p['expected_recordings'],
                frames=p['frames'],bytes=p['uploaded_bytes'],last=[x['episode']for x in p['parts'][-6:]],jobs=jobs,errors=p['errors'])


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            isjson=self.path.split('?')[0]=='/status.json'
            body=(json.dumps(status(),ensure_ascii=False) if isjson else PAGE).encode()
            self.send_response(200);self.send_header('Content-Type','application/json; charset=utf-8' if isjson else 'text/html; charset=utf-8')
            self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
        except (OSError,ValueError):self.send_error(503)
    def log_message(self,*args):pass


if __name__=='__main__':
    print('http://127.0.0.1:8766',flush=True)
    ThreadingHTTPServer(('127.0.0.1',8766),Handler).serve_forever()
