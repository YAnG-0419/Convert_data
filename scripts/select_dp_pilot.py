#!/usr/bin/env python3
"""Choose three reproducible quality batches from existing human-reviewed data."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from convest.config import atomic_json
from convest.selection import read_episode_list
from convest.sources.gello_rosbag2 import discover


def main():
    inventory = discover(ROOT.parent/'company-remote/tomato')
    human = set(read_episode_list(ROOT/'lists/high_quality.txt')['requested_episodes'])
    ranked = []
    for bag in inventory:
        name = Path(bag['path']).name
        if not bag['eligible'] or name not in human:
            continue
        state = json.loads(Path(bag['path'],'collection_state.json').read_text())
        report = state['validation_report']
        gaps, nonmono = {}, 0
        for topic, info in report['streams'].items():
            source = info.get('source_timing', {})
            gap = float(source.get('max_internal_gap_ms', 150))
            if 'image_raw' in topic:
                limit = 100
            elif 'joint_states' in topic:
                limit = 75
            else:
                limit = 150
            gaps[topic] = {'gap_ms':gap,'ratio':gap/limit}
            nonmono += int(source.get('nonmonotonic_count', 0))
        telemetry = report.get('telemetry', {})
        bad_packets = sum(int(v) for k,v in telemetry.items() if k.startswith('delta_') and k.endswith('_packets'))
        worst = max(x['ratio'] for x in gaps.values())
        score = max(0.,100-30*worst-5*nonmono-min(20,bad_packets*.2))
        ranked.append(dict(episode=name, source=bag['path'], source_recording_id=bag['source_recording_id'],
                           duration_s=round((bag['source_end_ns']-bag['source_start_ns'])/1e9,2),
                           manual_quality='high', technical_score=round(score,3), worst_gap_ratio=worst,
                           nonmonotonic_samples=nonmono, telemetry_packet_events=bad_packets, gaps=gaps,
                           milestones=len(state.get('milestones',[]))))
    ranked.sort(key=lambda x:int(x['episode'].removeprefix('episode')))
    batches=[]
    for i,group in enumerate(np.array_split(np.arange(len(ranked)),3),1):
        members=[ranked[int(j)] for j in group]
        chosen=sorted(members,key=lambda x:(-x['technical_score'], int(x['episode'].removeprefix('episode'))))[:4]
        chosen.sort(key=lambda x:int(x['episode'].removeprefix('episode')))
        # One source per batch is reserved for validation; never split its fragments.
        validation=chosen[-1]['source_recording_id']
        for row in chosen:
            row['batch']=i
            row['split']='validation' if row['source_recording_id']==validation else 'train'
        batches.append({'batch':i,'episodes':[r['episode'] for r in chosen], 'validation_recording_id':validation})
        (ROOT/f'lists/dp_pilot_batch{i}.txt').write_text('\n'.join(x['episode'] for x in chosen)+'\n')
    chosen=[r for r in ranked if 'batch' in r]
    (ROOT/'lists/dp_pilot_all.txt').write_text('\n'.join(r['episode'] for r in chosen)+'\n')
    result=dict(total_bags=len(inventory), eligible_bags=sum(x['eligible'] for x in inventory),
                human_high_candidates=len(ranked), selection_rule='human high-quality list; three capture-order bands; top four by source continuity and telemetry in each band',
                score_formula='100 - 30 * max(source_gap / conversion_age_limit) - 5 * nonmonotonic_count - min(20, 0.2 * telemetry_packet_events)',
                limitation='Technical continuity score is not a task-success score; task quality comes from the existing human high-quality list.',
                batches=batches, selected=chosen, all_high_candidates=ranked,
                selected_duration_s=sum(r['duration_s'] for r in chosen),
                validation_recording_ids=[x['validation_recording_id'] for x in batches])
    atomic_json(ROOT/'outputs/dp_pilot_20260915/selection.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('selected','all_high_candidates')},indent=2))


if __name__=='__main__':
    main()
