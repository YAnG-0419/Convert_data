#!/usr/bin/env python3
"""Wait for the publisher and write a self-contained completion/error report."""
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from convest.config import atomic_json

LOCAL=ROOT/'outputs/dp_first_stage_publish'
DEST=ROOT.parent/'company-remote/tomato-dp'


def main():
    publishers=[]
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args=path.read_bytes().split(b'\0')
            if b'convest.dp.publish' in args:publishers.append(int(path.parent.name))
        except (OSError,ValueError):pass
    if len(publishers)!=1:raise RuntimeError(f'Expected one active publisher, found {publishers}')
    pid=publishers[0]
    while True:
        progress=json.loads((LOCAL/'progress.json').read_text())
        if progress['status'] in ('complete','incomplete'):break
        try:os.kill(pid,0)
        except ProcessLookupError:
            progress['status']='interrupted';break
        time.sleep(30)
    parts=progress['parts']
    errors=list(progress['errors'])
    for p in parts:
        w=p['record']['selection_window']
        if not p['upload_verified'] or w['mode']!='first_stage':
            errors.append({'episode':p['episode'],'error':'Missing verification/selection policy'})
        if w['milestone'] and any(e['source_end_ns']>=w['milestone']['timestamp_ns']for e in p['record']['episodes']):
            errors.append({'episode':p['episode'],'error':'Milestone boundary exceeded'})
    complete=progress['status']=='complete' and not errors and progress.get('source_snapshots_unchanged') is True
    summary=dict(complete=complete,status=progress['status'],recordings=len(parts),expected_recordings=progress['expected_recordings'],
                 single_stage=sum(p['record']['selection_window']['milestone'] is None for p in parts),
                 milestone_prefixes=sum(p['record']['selection_window']['milestone'] is not None for p in parts),
                 segments=sum(len(p['record']['episodes']) for p in parts),frames=sum(p['frames']for p in parts),fps=30,
                 uploaded_bytes=sum(p['bytes']for p in parts),source_snapshots_unchanged=progress.get('source_snapshots_unchanged',False),
                 errors=errors,completed_unix=time.time())
    atomic_json(LOCAL/'completion.json',summary);atomic_json(DEST/'completion.json',summary)
    text=f'''# 首阶段数据转换上传结果

状态：{'全部完成' if complete else '尚未全部完成，请检查错误或恢复任务'}。

- 已转换并上传：{summary['recordings']} / {summary['expected_recordings']} 条原始采集。
- 单阶段：{summary['single_stage']} 条；记录点之前的前缀：{summary['milestone_prefixes']} 条。
- 30 Hz，共 {summary['frames']:,} 帧，{summary['segments']} 个连续片段。
- 上传体积：{summary['uploaded_bytes']/1e9:.2f} GB。
- 原始文件快照保持不变：{summary['source_snapshots_unchanged']}。
- 每条数据均经过完整图像/动作/时间检查、ZIP CRC 校验和上传后的 SHA-256 校验。

每条采集的来源、保留时间边界、校验和详见 `dataset_manifest.json`。
完整选择及跳过原因见 `inventory.json`；训练读取方法见 `TRAINING.md`。

错误：

```json
{json.dumps(errors,ensure_ascii=False,indent=2)}
```
'''
    (LOCAL/'CONVERSION_REPORT.md').write_text(text);(DEST/'CONVERSION_REPORT.md').write_text(text)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    if not complete:raise SystemExit(1)


if __name__=='__main__':main()
