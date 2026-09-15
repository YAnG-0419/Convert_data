"""Named joints and causal alignment; source bags remain SQLite read-only."""
import numpy as np

from convest.align import valid_segments
from convest.sources.gello_rosbag2 import Series, joint_values

GROUPS = ('left_arm', 'right_arm', 'left_hand', 'right_hand')
CAMERAS = {'head_rgb': 'cam0', 'left_wrist_rgb': 'cam1', 'right_wrist_rgb': 'cam2'}


def joint_names(contract, action=False):
    names = contract['source_joint_names']
    return sum([names[(('validated_' if action else 'measured_') + group)
                      if group.endswith('arm') else group] for group in GROUPS], [])


def validate_contract(contract):
    for action in (False, True):
        names = joint_names(contract, action)
        if not names or len(names) != len(set(names)):
            raise ValueError('Joint names must be nonempty and globally unique')
    for side in ('left', 'right'):
        for prefix in ('measured', 'validated'):
            if len(contract['source_joint_names'][f'{prefix}_{side}_arm']) != 7:
                raise ValueError('Each Franka arm requires seven named joints')
        if not contract['source_joint_names'][f'{side}_hand']:
            raise ValueError('Hand joint list must not be empty')


def selected_window(item, config):
    """Inclusive source window; first_stage excludes the milestone itself."""
    start, end = item['source_start_ns'], item['source_end_ns']
    mode = config.get('segment_mode', 'full')
    if mode not in ('full', 'first_stage'):
        raise ValueError('segment_mode must be full or first_stage')
    markers = item.get('milestones', [])
    marker = None
    if mode == 'first_stage':
        if not isinstance(markers, list) or len(markers) > 1:
            raise ValueError('first_stage requires zero or one milestone')
        if markers:
            marker = markers[0]
            if not isinstance(marker, dict) or marker.get('clock') != 'ros':
                raise ValueError('Milestone must use the ROS source clock')
            stamp = marker.get('timestamp_ns')
            if type(stamp) is not int or not start < stamp <= end:
                raise ValueError('Milestone must lie within the validated source window')
            end = stamp - 1
    return start, end, dict(mode=mode, milestone=marker, start_ns=start,
                           end_ns_inclusive=end, validated_end_ns=item['source_end_ns'])


def read_streams(bag, contract, config, end_ns=None):
    topics, names = contract['topics'], contract['source_joint_names']
    streams = {}

    def read(key, topic, convert, image=False):
        streams[key] = Series.build([
            (t, convert(msg), received) for t, msg, received in bag.records(
                topics[topic]['topic'], image=image, include_receive=True)
            if end_ns is None or t <= end_ns])

    def positions(msg, joints, hand=False):
        if hand and set(msg.name) != set(joints):
            raise ValueError('Hand joints differ from contract; supply the actual hand mapping')
        return joint_values(msg, joints, ['position'])

    for group in GROUPS:
        joints = names['measured_' + group if group.endswith('arm') else group]
        read('state.' + group, group + '_state',
             lambda msg, j=joints, g=group: positions(msg, j, g.endswith('hand')))
    actions = {side: [] for side in ('left', 'right')}
    for t, msg, received in bag.records(topics['validated_arm_action']['topic'], include_receive=True):
        if end_ns is not None and t > end_ns:
            continue
        for side in actions:
            value = joint_values(msg, names[f'validated_{side}_arm'], ['position'], allow_absent=True)
            if value is not None:
                actions[side].append((t, value, received))
    for side in actions:
        streams[f'action.{side}_arm'] = Series.build(actions[side])
        read(f'action.{side}_hand', f'{side}_hand_action',
             lambda msg, j=names[f'{side}_hand']: positions(msg, j, True))
    statuses = {g: [] for g in GROUPS}
    for t, msg, received in bag.records(topics['arm_command_status']['topic'], include_receive=True):
        if end_ns is not None and t > end_ns:
            continue
        for side in ('left', 'right'):
            statuses[f'{side}_arm'].append((t, (side in msg.accepted_sides, not bool(msg.faults)), received))
    for t, msg, received in bag.records(topics['hand_telemetry_status']['topic'], include_receive=True):
        if end_ns is not None and t > end_ns:
            continue
        if msg.side not in ('left', 'right'):
            raise ValueError(f'Unknown hand status side: {msg.side}')
        statuses[f'{msg.side}_hand'].append((t, (msg.engaged, msg.state_valid and
                                                (not msg.engaged or msg.command_valid)), received))
    for group, values in statuses.items():
        streams['status.' + group] = Series.build(values)
    for key, topic in CAMERAS.items():
        read(key, topic, lambda value: value, image=True)
    if config['include_depth']:
        read('head_depth', 'cam0_depth', lambda value: value, image=True)
    return streams


def align(streams, start, end, config):
    """Last source-header sample at/before t. Gaps split independent episodes.

    Inactive groups hold the last command at disengagement, or the measured
    position at disengagement if no command existed. Never seed from the future.
    """
    fps = config['fps']
    count = (end - start) * fps // 1_000_000_000 + 1
    if count < config['min_segment_frames']:
        raise ValueError('Validated source window is too short')
    timeline = start + np.arange(count, dtype=np.int64) * 1_000_000_000 // fps
    valid = np.ones(count, dtype=bool)
    indices, misses, ages = {}, {}, {}

    def sample(key, limit):
        series = streams[key]
        index = series.indices(timeline)
        age = timeline - series.times[np.maximum(index, 0)]
        good = (index >= 0) & (age >= 0) & (age <= round(config[limit] * 1e6))
        valid[:] &= good
        indices[key] = np.maximum(index, 0)
        misses[key] = int((~good).sum())
        ages[key] = float(age[good].max() / 1e6) if good.any() else None
        return np.asarray(series.values)[np.maximum(index, 0)]

    states, actions, engaged, origins = [], [], [], []
    for group in GROUPS:
        measured = streams['state.' + group]
        states.append(sample('state.' + group, 'state_max_age_ms'))
        status = sample('status.' + group, 'status_max_age_ms')
        active = status[:, 0].astype(bool)
        valid &= status[:, 1].astype(bool)
        statuses = streams['status.' + group]
        flags = np.asarray(statuses.values)[:, 0].astype(bool)
        transitions = np.r_[0, np.flatnonzero(flags[1:] != flags[:-1]) + 1]
        run = np.searchsorted(transitions, indices['status.' + group], side='right') - 1
        transition_time = statuses.times[transitions[run]]
        command = streams['action.' + group]
        command_index = command.indices(timeline)
        command_time = command.times[np.maximum(command_index, 0)]
        good = (command_index >= 0) & (timeline - command_time <= round(config['command_max_age_ms'] * 1e6))
        # Commands and status headers may differ slightly; allow 20 ms at engagement.
        good &= command_time + 20_000_000 >= transition_time
        hold_index = command.indices(transition_time)
        seed_index = measured.indices(transition_time)
        use_seed = (~active) & (hold_index < 0)
        selected = np.where(active, command_index, hold_index)
        values = np.asarray(command.values)[np.maximum(selected, 0)].copy()
        values[use_seed] = np.asarray(measured.values)[np.maximum(seed_index[use_seed], 0)]
        good = np.where(active, good, (hold_index >= 0) | (seed_index >= 0))
        valid &= good
        indices['action.' + group] = np.maximum(selected, 0)
        misses['action.' + group] = int((~good).sum())
        actions.append(values)
        engaged.append(active)
        # Actual timestamp provenance for measured fallback (origin=1).
        origins.append(use_seed.astype(np.uint8))
        indices['seed.' + group] = np.maximum(seed_index, 0)
    image_keys = list(CAMERAS) + (['head_depth'] if config['include_depth'] else [])
    for key in image_keys:
        # Avoid np.asarray on SQLite image references.
        s = streams[key]
        idx = s.indices(timeline)
        age = timeline - s.times[np.maximum(idx, 0)]
        good = (idx >= 0) & (age >= 0) & (age <= round(config['image_max_age_ms'] * 1e6))
        valid &= good
        indices[key] = np.maximum(idx, 0)
        misses[key] = int((~good).sum())
        ages[key] = float(age[good].max() / 1e6) if good.any() else None
    segments = valid_segments(valid, config['min_segment_frames'])
    if not segments:
        raise ValueError(f'No sufficiently long continuous segments: {misses}')
    keys = [k for k in streams]
    source_times, receive_times = [], []
    for key in keys:
        s, idx = streams[key], indices[key]
        source_times.append(s.times[idx].copy())
        receive_times.append(s.receive_times[idx].copy())
        if key.startswith('action.'):
            group = key.removeprefix('action.')
            mask = origins[GROUPS.index(group)].astype(bool)
            measured = streams['state.' + group]
            seed = indices['seed.' + group]
            source_times[-1][mask] = measured.times[seed[mask]]
            receive_times[-1][mask] = measured.receive_times[seed[mask]]
    return dict(timeline=timeline, state=np.concatenate(states, axis=1).astype('f4'),
                action=np.concatenate(actions, axis=1).astype('f4'),
                engaged=np.stack(engaged, axis=1).astype('u1'),
                action_from_state=np.stack(origins, axis=1), indices=indices, segments=segments,
                source_time_ns=np.stack(source_times, axis=1),
                receive_time_ns=np.stack(receive_times, axis=1), stream_keys=keys,
                report=dict(grid_frames=count, retained_frames=sum(b-a for a,b in segments),
                            invalid_frames=int((~valid).sum()), segments=segments,
                            unmatched=misses, max_age_ms=ages, alignment='causal_source_header'))


def inspect(bag, contract):
    """Read a single message per configured stream; no large payload dump."""
    result = {}
    for key, spec in contract['topics'].items():
        if spec['topic'] not in bag.topics:
            result[key] = {'missing': spec['topic']}
            continue
        records = bag.records(spec['topic'], image=spec['type'] == 'sensor_msgs/msg/Image')
        try:
            stamp, msg = next(records)
            value = dict(topic=spec['topic'], source_timestamp_ns=stamp)
            if spec['type'] == 'sensor_msgs/msg/Image':
                db, row_id = msg
                raw = db.execute('SELECT data FROM messages WHERE id=?', (row_id,)).fetchone()[0]
                image = bag.store.deserialize_cdr(raw, spec['type'])
                value.update(width=image.width, height=image.height, encoding=image.encoding)
            elif hasattr(msg, 'name'):
                value.update(joint_names=list(msg.name), position_dim=len(msg.position))
            elif hasattr(msg, 'k'):
                value.update(width=msg.width, height=msg.height, K=list(msg.k))
            result[key] = value
        finally:
            records.close()
    return result
