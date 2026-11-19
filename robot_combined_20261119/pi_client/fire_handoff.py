"""Stationary approach snapshot and same-session launcher handoff. No hardware I/O."""
from copy import deepcopy
from pathlib import Path
from statistics import median
import json
import uuid
import os
import math
import hashlib
import time
from aiming import alignment, solution
from core import valid_distance, valid_yaw
from fire_review import FireRangeReview
from fire_navigation import fire_alignment_tolerance


class HandoffCapture:
    def __init__(self):
        self.reset()

    def reset(self):
        self.range = FireRangeReview()
        self.began = None

    def update(self, mission, now, distance, yaw, seq, vision, leveled, link_ok):
        cfg = mission.cfg
        mission.observe_fire_sonar(now, distance, seq)
        confidence = mission.fire_range_diagnostic(now)
        if not (leveled and link_ok and mission.active and not mission.rescue
                and confidence['arrival_ready']
                and mission.state == 'FIRE_POSITION_READY' and mission.last_action == 'stop'
                and mission.fresh(vision, now) and valid_yaw(yaw) and valid_distance(distance)
                and cfg['threshold_cm'] < distance <= cfg['fire_target_cm']
                and not any(d['label']=='person' and d['confidence'] >= .35 for d in vision['detections'])
                and mission.fire_box):
            self.reset()
            return None
        dx, tolerance, _ = alignment(mission.fire_box, vision, cfg)
        tolerance = fire_alignment_tolerance(vision['width'])
        if abs(dx) > tolerance:
            self.reset()
            return None
        if self.began is None:
            self.began = now
        readings = self.range.update(now, distance, seq, False)
        if (readings is None or vision['capture_mono'] < self.began
                or max(readings) > cfg['fire_target_cm']):
            return None
        distance = median(readings)
        plan = solution(mission.fire_box, distance, vision['width'], vision['height'], cfg,
                        vision.get('camera_intrinsics'))
        return dict(version=2, id=uuid.uuid4().hex, pi_monotonic=now,
                    created_unix=time.time(), config_sha256=config_signature(cfg),
                    sonar_raw_cm=distance, sonar_samples_cm=readings,
                    range_consistency=deepcopy(confidence),
                    yaw_deg=yaw, vision_seq=vision['seq'],
                    width=vision['width'], height=vision['height'],
                    camera_intrinsics=deepcopy(vision.get('camera_intrinsics')),
                    target_box=list(mission.fire_box), plan=deepcopy(plan),
                    range_change_limit_cm=20., yaw_change_limit_deg=4.,
                    physical_launch_verified=False)


def durable_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    target = path if exclusive else path.with_suffix('.tmp')
    with target.open('x' if exclusive else 'w', encoding='utf-8') as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    if not exclusive:
        target.replace(path)
    if os.name == 'posix':
        fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def save_snapshot(snapshot, directory):
    directory = Path(directory)
    target = directory / (snapshot['id'] + '.json')
    durable_json(target, snapshot)
    durable_json(directory/'latest.json', snapshot)
    return target


def config_signature(cfg):
    excluded = {'mission_profile', 'angle_output_enabled', 'fire_output_enabled'}
    data = {k:v for k,v in cfg.items() if k not in excluded}
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()


def gravity_unit(acceleration):
    try:
        values = [float(acceleration[k]) for k in ('x','y','z')]
        norm = math.sqrt(sum(v*v for v in values))
        if not all(math.isfinite(v) for v in values) or not 7.5 < norm < 12:
            return None
        return [v/norm for v in values]
    except (KeyError, TypeError, ValueError):
        return None


def tilt_error(reference, acceleration):
    unit = gravity_unit(acceleration)
    saved = reference.get('gravity_unit')
    if unit is None or saved is None:
        return 'Fresh stable IMU acceleration required'
    angle = math.degrees(math.acos(max(-1., min(1., sum(a*b for a,b in zip(unit,saved))))))
    return 'Chassis tilt changed by more than 4 degrees' if angle > 4. else None


def used_path(path, snapshot):
    return Path(path).parent/'used'/(snapshot['id']+'.json')


def load_snapshot(path, cfg):
    path = Path(path)
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    # Reject nonfinite numbers anywhere, including the saved plan.
    json.dumps(data, allow_nan=False)
    if data.get('version') != 2 or len(data.get('id','')) != 32:
        raise ValueError('Unsupported handoff file; save a new position with client1')
    if any(c not in '0123456789abcdef' for c in data['id']):
        raise ValueError('Invalid snapshot id')
    if data.get('config_sha256') != config_signature(cfg):
        raise ValueError('Geometry/config changed; save a new approach position')
    if not cfg['threshold_cm'] < data['sonar_raw_cm'] <= cfg['fire_target_cm']:
        raise ValueError('Saved range outside approach interval')
    box=data['target_box'];gravity=data['gravity_unit']
    if (len(box)!=4 or not 0<=box[0]<box[2]<=data['width']
            or not 0<=box[1]<box[3]<=data['height'] or len(gravity)!=3
            or abs(sum(v*v for v in gravity)-1)>0.01):
        raise ValueError('Invalid saved target or attitude')
    if used_path(path,data).exists():
        raise ValueError('This snapshot already reserved a shot; save a NEW position in client1')
    return data


def claim_shot(path, snapshot):
    # Persist BEFORE channel 14. An interrupted/failed attempt remains used.
    durable_json(used_path(path,snapshot), dict(id=snapshot['id'],
        state='shot_reserved', unix=time.time(), physical_launch_verified=False), exclusive=True)


class ManualRecheck:
    def __init__(self, reference, cfg):
        self.reference, self.cfg = deepcopy(reference), cfg
        self.reason = 'Waiting for new stopped readings'
        self.generation = 0
        self.reset()

    def reset(self):
        self.range = FireRangeReview()
        self.began = None
        self.accepted = None

    def fail(self, reason):
        self.reason = reason
        self.reset()
        return None

    def update(self, now, distance, yaw, seq, vision, box, overlap, acceleration):
        ref = self.reference
        if not valid_distance(distance) or not valid_yaw(yaw):
            return self.fail('Fresh sonar and heading required')
        if abs(distance-ref['sonar_raw_cm']) > 20.:
            return self.fail('Range differs from saved position by more than 20 cm')
        error = tilt_error(ref, acceleration)
        if error:
            return self.fail(error)
        if (vision['width']!=ref['width'] or vision['height']!=ref['height']
                or vision.get('camera_intrinsics')!=ref['camera_intrinsics']):
            return self.fail('Camera geometry differs from saved position')
        if overlap(box,ref['target_box']) < .5:
            return self.fail('Saved fire target does not match current detection')
        if self.accepted is not None:
            error=snapshot_error(self.accepted,vision,distance,yaw,overlap)
            return self.fail(error) if error else self.accepted
        if self.began is None:
            self.began=now
        readings=self.range.update(now,distance,seq,False)
        if readings is None or vision['capture_mono']<=self.began:
            self.reason='Waiting for five new stable sonar readings after start'
            return None
        measured=median(readings)
        plan=solution(box,measured,vision['width'],vision['height'],self.cfg,vision.get('camera_intrinsics'))
        self.accepted=deepcopy(ref)
        # Monotonic time and relative yaw from the previous Pi process are NOT reusable.
        self.accepted.update(pi_monotonic=self.began, yaw_deg=yaw, sonar_raw_cm=measured,
                             target_box=list(box), plan=plan, sonar_samples_cm=readings)
        self.reason='Saved target rechecked; using freshly recomputed plan'
        self.generation+=1
        return self.accepted


def snapshot_error(snapshot, vision, distance, yaw, overlap):
    if not valid_distance(distance) or not valid_yaw(yaw):
        return 'Fresh sonar and heading required'
    if abs(distance-snapshot['sonar_raw_cm']) > snapshot['range_change_limit_cm']:
        return 'Range changed from saved approach position'
    yaw_delta = (yaw-snapshot['yaw_deg']+180.) % 360. - 180.
    if abs(yaw_delta) > snapshot['yaw_change_limit_deg']:
        return 'Heading changed from saved approach position'
    if (not vision or vision['capture_mono'] <= snapshot['pi_monotonic']
            or vision['width'] != snapshot['width'] or vision['height'] != snapshot['height']
            or vision.get('camera_intrinsics') != snapshot['camera_intrinsics']):
        return 'Waiting for matching fresh camera geometry after handoff'
    if not any(d['label']=='fire' and overlap(d['box'], snapshot['target_box']) >= .5
               for d in vision['detections']):
        return 'Saved fire target not visible'
    return None
