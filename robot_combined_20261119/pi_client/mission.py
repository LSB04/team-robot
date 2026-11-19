"""Pure mission arbitration. No hardware, network or inference imports."""
import math
from copy import deepcopy
from fire_handoff import snapshot_error, tilt_error
from person_review import PersonReview
from core import valid_distance, valid_yaw
from navigation import NavigationScan
from aiming import solution, alignment
from fire_review import FireReview, FireRangeReview
from launcher_review import LauncherFireReview
from launcher_range import LauncherRange
from fire_navigation import FireDistance, FireAlignment, FireRangeConsistency, fire_alignment_tolerance, VERSION as FIRE_NAVIGATION_VERSION


def iou(a, b):
    inter = max(0, min(a[2], b[2])-max(a[0], b[0]))*max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union > 0 else 0.


# Preserve the public name used by existing imports and tests.
MissionScan = NavigationScan

class Mission:
    def __init__(self, cfg):
        self.cfg = cfg
        self.handoff_snapshot = None
        self.manual_recheck = None
        self.current_acceleration = None
        self.scan = MissionScan(threshold=cfg['threshold_cm'])
        self.active = False
        self.state, self.reason = 'IDLE', ''
        self.last_seq = -1
        self.fire_count = self.person_count = 0
        self.fire_box = None
        self.plan = None
        self.aim_at = None
        self.last_aim = None
        self.shots = 0
        self.rescue = False
        self.last_fire_at = -math.inf
        self.person_review = PersonReview(hold_while_visible=cfg.get('mission_profile') == 2)
        self.launcher_review = LauncherFireReview(cfg)
        self.launcher_range = LauncherRange()
        self.fire_review = FireReview(cfg)
        self.fire_range_review = FireRangeReview()
        self.fire_distance = FireDistance(cfg['threshold_cm'])
        self.fire_consistency = FireRangeConsistency(cfg['fire_target_cm'], cfg['threshold_cm'])
        self.fire_range_started_at = None
        self.fire_alignment = FireAlignment()
        self.fire_range_mode = None
        self.fire_approach_verified = False
        self.fire_position_reached = False
        self.fire_initial_aligned = False
        self.fire_arrival_confirmed = False
        self.fire_arrival_locked = False
        self.fire_arrival_locked_at = None
        self.fire_route_box = None
        self.fire_route_generation = 0
        self.fire_route_target_id = None
        self.fire_initial_leveled = self.fire_final_leveled = False
        self.last_action = 'stop'

    def reset_manual_recheck(self):
        if self.manual_recheck:
            self.manual_recheck.reset()
            self.handoff_snapshot = None
            self.aim_at = self.last_aim = None

    def stop(self, reason='Operator stop'):
        self.fire_arrival_locked = False
        self.fire_arrival_locked_at = None
        self.reset_manual_recheck()
        self.active = False
        self.scan.stop(reason)
        self.state, self.reason = 'IDLE', reason
        self.aim_at = None
        self.fire_review.reset()
        self.fire_range_review.reset()
        self.launcher_review.reset()
        self.launcher_range.reset()
        self.fire_distance.reset()
        self.fire_alignment.reset()
        self.fire_range_mode = None
        self.fire_approach_verified = False
        self.fire_position_reached = False
        self.reset_fire_route()
        self.last_action = 'stop'

    def start(self, now, yaw):
        self.reset_manual_recheck()
        if not valid_yaw(yaw):
            return False
        self.fire_arrival_locked = False
        self.fire_arrival_locked_at = None
        self.scan.stop()
        self.scan.start(now, yaw)
        self.active, self.rescue = True, False
        self.aim_at = None
        self.fire_count = self.person_count = 0
        self.fire_box = None
        self.last_seq = -1
        self.last_fire_at = -math.inf
        self.person_review = PersonReview(hold_while_visible=self.cfg.get('mission_profile') == 2)
        self.launcher_review.reset()
        self.launcher_range.reset()
        self.fire_review.reset()
        self.fire_range_review.reset()
        self.fire_position_reached = False
        self.reset_fire_route()
        self.fire_distance.reset()
        self.fire_range_started_at = now
        self.fire_alignment.reset()
        self.fire_range_mode = None
        self.fire_approach_verified = False
        self.last_action = 'stop'
        # Shot count is deliberately not reset by start.
        return True

    def fresh(self, vision, now):
        return (vision is not None and isinstance(vision.get('capture_mono'), (float, int))
                and math.isfinite(vision['capture_mono'])
                and 0 <= now-vision['capture_mono'] <= self.cfg['vision_max_age_seconds'])

    def people(self, vision):
        return [d for d in vision.get('detections', [])
                if d['label'] == 'person' and d['confidence'] >= self.cfg['person_conf']]

    def fire_candidates(self, vision):
        if self.cfg.get('mission_profile') == 1:
            return self.fire_review.candidates(vision)
        if self.cfg.get('mission_profile') == 2:
            return self.launcher_review.candidates(vision)
        return [d for d in (vision or {}).get('detections', [])
                if d['label'] == 'fire' and d['confidence'] >= self.cfg['fire_conf']]

    def fire_motion_visible(self, vision):
        if self.cfg.get('mission_profile') == 1:
            return self.fire_review.confirmed and self.fire_review.visible_match(vision, iou)
        return bool(self.fire_candidates(vision))

    def refresh_fire_guard(self, vision, now):
        # Leveling/gait blocks the outer mission loop. Consume fresh images here
        # so a normal 1.5s leveling operation is not mistaken for a camera gap.
        if self.cfg.get('mission_profile') == 1 and self.fresh(vision, now):
            self.fire_review.update(vision, now, iou)

    def reset_fire_route(self):
        self.fire_consistency.reset()
        self.fire_range_started_at = None
        self.fire_initial_aligned = self.fire_arrival_confirmed = False
        self.fire_route_box = None
        self.fire_route_target_id = None
        self.fire_initial_leveled = self.fire_final_leveled = False
        self.fire_route_generation += 1

    def fire_phase(self):
        if self.fire_position_reached:
            return 'READY'
        if self.fire_arrival_locked:
            return 'FINAL_ALIGN'
        if not self.fire_initial_aligned:
            return 'INITIAL_ALIGN'
        return 'FINAL_ALIGN' if self.fire_arrival_confirmed else 'CRUISE'

    def observe_fire_sonar(self, now, distance, seq, history=()):
        if not self.active or self.cfg.get('mission_profile') != 1:
            return
        if self.fire_range_started_at is None:
            self.fire_range_started_at = now
        for sample in history:
            stamp = sample['pi_monotonic']
            if self.fire_range_started_at <= stamp <= now:
                value = sample['raw_cm'] if sample['read_status'] == 'ok' else None
                self.fire_consistency.observe(stamp, value, sample['seq'])
        self.fire_consistency.observe(now, distance, seq)

    def fire_range_diagnostic(self, now):
        return self.fire_consistency.diagnostic(now, self.fire_range_review.stopped_at)

    def fire_approach_range_stop(self, now, distance):
        # Missing/split readings are diagnostics, not a travel interlock.
        # They never count as arrival, clear space, or shot authorization.
        if valid_distance(distance) and distance <= self.cfg['threshold_cm']:
            return 'Obstacle at or below stop threshold'
        self.lock_fire_arrival(now)
        if self.fire_arrival_locked:
            return 'Stable fire approach target reached; hold position'
        return None

    def lock_fire_arrival(self, now):
        if (not self.fire_arrival_locked and self.active
                and self.cfg.get('mission_profile') == 1 and self.fire_review.confirmed
                and self.fire_consistency.approach_stop_ready(now)):
            self.fire_arrival_locked = True
            self.fire_arrival_locked_at = now
            self.fire_approach_verified = False
            self.fire_range_review.reset()
            self.plan = None
            return True
        return False

    def required_fire_level(self):
        if (not self.active or self.cfg.get('mission_profile') != 1
                or self.state not in ('FIRE_ALIGN', 'FIRE_APPROACH', 'FIRE_POSITION_READY', 'FIRE_RANGE_RECHECK')):
            return None
        if not self.fire_initial_leveled:
            return 'INITIAL'
        if self.fire_arrival_locked and not self.fire_final_leveled:
            return 'FINAL'
        return None

    def after_fire_level(self, stage):
        # Only a new target/start can require INITIAL leveling again. Temporary
        # range/vision waits must never erase this completed stage.
        if stage == 'INITIAL' and not self.fire_initial_leveled:
            self.fire_initial_leveled = True
            self.fire_initial_aligned = self.fire_arrival_confirmed = False
        elif stage == 'FINAL' and self.fire_arrival_locked and not self.fire_final_leveled:
            self.fire_final_leveled = True
        else:
            return False
        # A posture change invalidates pre-level range and the pre-level pixel
        # center. Require fresh final alignment/range before saving a position.
        self.fire_position_reached = False
        self.fire_distance.reset()
        self.fire_range_review.reset()
        self.fire_range_mode = None
        self.fire_approach_verified = False
        self.last_action = 'stop'
        return True

    def fire_level_diagnostic(self):
        return dict(route_generation=self.fire_route_generation,
                    target_id=self.fire_route_target_id, phase=self.fire_phase(),
                    initial_done=self.fire_initial_leveled, final_done=self.fire_final_leveled,
                    required=self.required_fire_level())

    def live_fire_alignment(self, vision):
        matches = [d for d in self.fire_candidates(vision)
                   if self.fire_box and iou(self.fire_box, d['box']) >= .2]
        if not matches:
            return None
        return alignment(max(matches, key=lambda d:d['confidence'])['box'], vision, self.cfg)

    def fire_alignment_complete(self, vision, action):
        current = self.live_fire_alignment(vision)
        if current is None:
            return False
        dx, tolerance, _ = current
        if self.cfg.get('mission_profile') == 1:
            tolerance = fire_alignment_tolerance(vision['width'], far=not self.fire_alignment.final)
        return abs(dx) <= tolerance or (action == 'right' and dx < 0) or (action == 'left' and dx > 0)

    def fire_approach_off_center(self, vision):
        current = self.live_fire_alignment(vision)
        return current is not None and abs(current[0]) > fire_alignment_tolerance(vision['width'], far=True)

    def _review_fire_navigation(self, now, distance, sample_id, vision):
        chosen = self.fire_review.update(vision, now, iou)
        if not self.fire_review.engaged:
            return None
        self.scan.stop('Fire priority')
        if chosen is None:
            self.fire_range_review.reset()
            return self.result('FIRE_RECHECK', 'Stopped for fresh fire review; start clears the target')
        if (self.fire_route_target_id is not None
                and self.fire_route_target_id != self.fire_review.target_id):
            if self.fire_arrival_locked:
                self.fire_position_reached = False
                self.plan = None
                return self.result('FIRE_RECHECK', 'Arrival locked; target changed; start required for a new approach')
            self.reset_fire_route()
            self.fire_position_reached = False
            self.fire_range_review.reset()
            self.fire_distance.reset()
            self.fire_approach_verified = False
            self.observe_fire_sonar(now, distance, sample_id)
        self.fire_route_box = list(chosen['box'])
        self.fire_route_target_id = self.fire_review.target_id
        self.fire_box = list(chosen['box'])
        if valid_distance(distance) and distance <= self.cfg['threshold_cm']:
            self.fire_range_review.reset()
            return self.result('FIRE_TOO_CLOSE', 'Obstacle stop threshold has priority over approach')
        width, height = vision['width'], vision['height']
        box = chosen['box']
        dx, tolerance, focal = alignment(box, vision, self.cfg)
        target = self.cfg.get('fire_target_cm', 200.)
        control_distance = self.fire_distance.update(now, distance, sample_id)
        confidence = self.fire_range_diagnostic(now)
        # Evaluate distance BEFORE choosing any alignment movement. This lock
        # survives sonar noise, vision pauses and leveling until operator restart.
        if self.lock_fire_arrival(now):
            return self.result('FIRE_RANGE_RECHECK', 'Arrival locked at 200cm or nearer; forward disabled')
        if not confidence['near_consistent']:
            self.fire_position_reached = False
            self.fire_arrival_confirmed = False
            self.plan = None
        # The arrival lock selects final alignment even during later dropouts.
        # Saving still requires agreement over the longer stationary history.
        alignment_stage = self.fire_arrival_locked
        # Test9 width-scaled bands, including the first approach. Only confirmed
        # arrival switches to the narrow band; a stray near pulse cannot do so.
        tolerance = fire_alignment_tolerance(width, far=not alignment_stage)
        should_align = self.fire_alignment.update(dx, tolerance, final=alignment_stage, outer_multiplier=1.)
        if should_align:
            self.fire_range_review.reset()
            self.fire_distance.reset()
            self.fire_approach_verified = False
            result = self.result('FIRE_ALIGN', 'Aligning confirmed fire horizontally')
            result.update(action='right' if dx > 0 else 'left', turn_degrees=min(4., max(.5,
                abs(math.degrees(math.atan2(dx, focal))))))
            return result
        if not self.fire_initial_aligned:
            self.fire_initial_aligned = True
        if self.fire_position_reached:
            self.plan = solution(box, distance, width, height, self.cfg, vision.get('camera_intrinsics'))
            return self.result('FIRE_POSITION_READY', 'Position reached; hold until operator restart', plan=self.plan)
        if not self.fire_arrival_locked:
            self.fire_range_review.reset()
            self.fire_range_mode = 'nonblocking_approach'
            self.fire_approach_verified = True
            result = self.result('FIRE_APPROACH', confidence['code']+
                ': visual approach continues; arrival not confirmed')
            result['action'] = 'forward'
            return result
        # Stop only after a consistent near history; then confirm while stopped.
        # No near/far cluster selection and no fabricated replacement distance.
        if not valid_distance(distance) or not self.cfg['threshold_cm'] < distance <= target:
            self.fire_range_review.reset()
            return self.result('FIRE_RANGE_RECHECK', 'Arrival locked; aligned; waiting for valid save distance')
        range_mode = 'arrival_raw'
        if range_mode != self.fire_range_mode:
            self.fire_range_review.reset()
            self.fire_range_mode = range_mode
        review_distance = distance
        values = self.fire_range_review.update(now, review_distance, sample_id, self.last_action != 'stop')
        if values is None or min(values) <= target < max(values):
            return self.result('FIRE_RANGE_RECHECK', 'Stopped to verify five fresh consistent sonar readings')
        if max(values) <= target:
            confidence = self.fire_range_diagnostic(now)
            if not confidence['arrival_ready']:
                return self.result('FIRE_RANGE_RECHECK', confidence['code']+': arrival/save not confirmed')
            self.fire_arrival_confirmed = True
            tolerance = fire_alignment_tolerance(width)
            self.fire_alignment.update(dx, tolerance, final=True, outer_multiplier=1.)
            if abs(dx) > tolerance:
                self.fire_range_review.reset()
                self.fire_distance.reset()
                self.fire_approach_verified = False
                result = self.result('FIRE_ALIGN', 'Stopped range confirmed; final fire alignment')
                result.update(action='right' if dx > 0 else 'left', turn_degrees=min(4., max(.5,
                    abs(math.degrees(math.atan2(dx, focal))))))
                return result
            self.fire_position_reached = True
            self.fire_approach_verified = False
            self.plan = solution(box, distance, width, height, self.cfg, vision.get('camera_intrinsics'))
            return self.result('FIRE_POSITION_READY', f'Fire centered within {target:g} cm; hold pose', plan=self.plan)
        return self.result('FIRE_RANGE_RECHECK', 'Arrival locked; forward disabled; waiting for save distance')

    def result(self, state, reason='', **extra):
        self.state, self.reason = state, reason
        return dict(state=state, reason=reason, action='stop', **extra)

    def firing_permitted(self, now, vision, distance, yaw, target_box, link_fresh):
        if self.manual_recheck and (self.manual_recheck.accepted is None
                or tilt_error(self.manual_recheck.reference, self.current_acceleration)):
            return False
        if self.handoff_snapshot and snapshot_error(self.handoff_snapshot, vision, distance, yaw, iou):
            return False
        if not (self.active and not self.rescue and link_fresh and self.fresh(vision, now)
                and not self.people(vision) and valid_yaw(yaw) and valid_distance(distance)
                and self.cfg['fire_min_cm'] <= distance <= self.cfg['fire_max_cm']):
            return False
        if self.cfg.get('mission_profile') == 2:
            # Even one weak person candidate inhibits channel 14 immediately,
            # including during an active pulse before the next mission step.
            if (self.person_review.hold_at is not None
                    or any(d['label'] == 'person' and d['confidence'] >= .35
                           for d in vision['detections'])
                    or not self.launcher_review.confirmed):
                return False
        return any(d['label'] == 'fire' and d['confidence'] >= self.cfg['fire_conf']
                   and iou(d['box'], target_box) >= .5
                   and abs(alignment(d['box'], vision, self.cfg)[0])
                   <= alignment(d['box'], vision, self.cfg)[1]
                   for d in vision['detections'])

    def step(self, now, distance, yaw, sample_id, vision, armed=False):
        if self.cfg.get('mission_profile') == 2 and self.active:
            self.launcher_range.update(now,distance,sample_id)
        self.observe_fire_sonar(now, distance, sample_id)
        result = self._step(now, distance, yaw, sample_id, vision, armed)
        if self.manual_recheck and result['state'] in ('PERSON_CONFIRM','PERSON_RECHECK','RESCUE_HOLD','GYRO_WAIT','VISION_WAIT','FIRE_RANGE_WAIT','FIRE_ALIGN','FIRE_CONFIRM','LAUNCHER_WAIT','FIRE_TOO_CLOSE','FIRE_APPROACH'):
            self.reset_manual_recheck()
        self.last_action = result['action']
        if self.cfg.get('mission_profile') == 1:
            if result['action'] == 'forward':
                self.fire_final_leveled = False
            if result['state'] not in ('FIRE_ALIGN','FIRE_APPROACH','FIRE_RANGE_RECHECK','FIRE_POSITION_READY'):
                self.fire_distance.reset()
                self.fire_alignment.reset()
                self.fire_approach_verified = False
            result['fire_review'] = self.fire_review.diagnostic()
            result['fire_range_review'] = self.fire_range_review.diagnostic()
            result['fire_navigation'] = dict(version=FIRE_NAVIGATION_VERSION,
                alignment_policy='test9_width_scaled_120_20',
                range_policy='nonblocking_missing_or_split_stable_arrival_required',
                phase=self.fire_phase(), initial_aligned=self.fire_initial_aligned,
                arrival_confirmed=self.fire_arrival_confirmed,
                arrival_locked=self.fire_arrival_locked, arrival_locked_at=self.fire_arrival_locked_at,
                leveling=self.fire_level_diagnostic(),
                range_consistency=self.fire_range_diagnostic(now),
                distance=self.fire_distance.diagnostic(), alignment=self.fire_alignment.diagnostic(),
                range_review_mode=self.fire_range_mode, approach_verified=self.fire_approach_verified)
        if self.cfg.get('mission_profile') == 2:
            result['launcher_review'] = self.launcher_review.diagnostic()
            result['launcher_range'] = self.launcher_range.diagnostic()
        return result

    def _step(self, now, distance, yaw, sample_id, vision, armed=False):
        if not self.active:
            return self.result('IDLE', self.reason)
        if self.rescue:
            return self.result('RESCUE_HOLD', 'Person detected; operator start required to resume')
        if not valid_yaw(yaw):
            self.launcher_review.reset()
            self.aim_at = None
            self.fire_review.pause(now)
            self.fire_range_review.reset()
            return self.result('GYRO_WAIT', 'No fresh gyro; movement stopped')
        if not self.fresh(vision, now):
            self.launcher_review.reset()
            self.fire_review.pause(now)
            self.fire_range_review.reset()
            self.aim_at = None
            self.fire_count = self.person_count = 0
            return self.result('VISION_WAIT', 'Waiting for fresh camera inference')
        people = self.people(vision)
        fires = [d for d in vision['detections'] if d['label'] == 'fire' and d['confidence'] >= self.cfg['fire_conf']]
        matches = [d for d in fires if self.fire_box and iou(self.fire_box, d['box']) >= .2]
        chosen = max(matches or fires, key=lambda d: d['confidence']) if fires else None
        if self.cfg.get('mission_profile') == 2:
            chosen = self.launcher_review.update(vision, iou)
        if vision['seq'] != self.last_seq:
            self.last_seq = vision['seq']
            self.person_count = self.person_count+1 if people else 0
            self.fire_count = (self.fire_count+1 if self.fire_box and chosen and
                               iou(self.fire_box, chosen['box']) >= .3 else 1 if chosen else 0)
            self.fire_box = chosen['box'] if chosen else None
        review = (self.person_review.update(vision, now, iou)
                  if self.cfg.get('mission_profile') == 2 else None)
        if people:
            self.launcher_review.reset()
            self.fire_review.pause(now)
            self.fire_range_review.reset()
            self.aim_at = None
            self.scan.stop('Person priority')
            if self.person_count >= self.cfg['confirm_frames']:
                self.rescue = True
                return self.result('RESCUE_HOLD', 'Rescue request: person detected', rescue_event=True)
            return self.result('PERSON_CONFIRM', 'Person candidate; movement and firing stopped')
        if self.cfg.get('mission_profile') in (1, 2):
            if self.cfg.get('mission_profile') == 1:
                review = self.person_review.update(vision, now, iou)
            if review in ('hold', 'confirmed'):
                self.launcher_review.reset()
                self.fire_review.pause(now)
                self.fire_range_review.reset()
                self.aim_at = None
                self.scan.stop('Repeated person candidate')
                if review == 'confirmed':
                    self.rescue = True
                    return self.result('RESCUE_HOLD', 'Repeated person confirmed while stopped', rescue_event=True)
                return self.result('PERSON_RECHECK', 'Stopped to recheck repeated person candidate',
                    person_recheck_frames=self.person_review.confirm)
            if review == 'clear':
                self.scan.stop('Person recheck cleared; restart from current heading')
                self.launcher_review.reset()
                self.aim_at = None
                if self.cfg.get('mission_profile') == 2:
                    return self.result('LAUNCHER_WAIT', 'Person recheck cleared; reacquire fresh fire target')
        if self.shots:
            return self.result('SHOT_COMPLETE', 'One shot used; inspect/reload and restart client for another shot', plan=self.plan)
        if self.cfg.get('mission_profile') == 1:
            reviewed = self._review_fire_navigation(now, distance, sample_id, vision)
            if reviewed is not None:
                return reviewed
        if chosen:
            self.last_fire_at = now
            self.scan.stop('Fire priority')
            confirmed = (self.launcher_review.confirmed if self.cfg.get('mission_profile') == 2
                         else self.fire_count >= self.cfg['confirm_frames'])
            if not confirmed:
                self.aim_at = None
                return self.result('FIRE_CONFIRM', 'Confirming consistent fire detection')
            if not valid_distance(distance):
                self.aim_at = None
                return self.result('FIRE_RANGE_WAIT', 'No fresh sonar distance')
            if self.cfg.get('mission_profile') == 1 and distance <= self.cfg['threshold_cm']:
                return self.result('FIRE_TOO_CLOSE', 'Fire target at or below obstacle stop threshold')
            width, height = vision['width'], vision['height']
            box = chosen['box']
            if self.handoff_snapshot and iou(box, self.handoff_snapshot['target_box']) < .5:
                self.aim_at = None
                return self.result('FIRE_RECHECK', 'Waiting for the saved fire target')
            dx, tolerance, focal = alignment(box, vision, self.cfg)
            if abs(dx) > tolerance:
                self.aim_at = None
                result = self.result('FIRE_ALIGN', 'Fire off center; navigation held' if self.cfg.get('mission_profile') == 2 else 'Aligning fire horizontally')
                result.update(action=('stop' if self.cfg.get('mission_profile') == 2 else 'right' if dx > 0 else 'left'), turn_degrees=min(8., max(2.,
                              abs(math.degrees(math.atan2(dx, focal))))))
                return result
            if self.cfg.get('mission_profile') == 1:
                target = self.cfg.get('fire_target_cm', 280.)
                if distance > target:
                    result = self.result('FIRE_APPROACH', 'Approaching aligned fire target')
                    result['action'] = 'forward'
                    return result
                self.plan = solution(box, distance, width, height, self.cfg, vision.get('camera_intrinsics'))
                return self.result('FIRE_POSITION_READY', f'Fire aligned within {target:g} cm; hold pose for client2', plan=self.plan)
            if distance > self.cfg['fire_max_cm']:
                self.aim_at = None
                result = self.result('FIRE_APPROACH', 'Fire outside configured range; navigation held' if self.cfg.get('mission_profile') == 2 else 'Approaching aligned fire')
                result['action'] = 'stop' if self.cfg.get('mission_profile') == 2 else 'forward'
                return result
            if distance < self.cfg['fire_min_cm']:
                self.aim_at = None
                return self.result('FIRE_TOO_CLOSE', 'Below configured firing distance')
            if self.manual_recheck:
                self.handoff_snapshot = self.manual_recheck.update(now,distance,yaw,sample_id,
                    vision,box,iou,self.current_acceleration)
                if self.handoff_snapshot is None:
                    self.aim_at = self.last_aim = None
                    return self.result('FIRE_RECHECK', self.manual_recheck.reason)
            if self.handoff_snapshot:
                error = snapshot_error(self.handoff_snapshot, vision, distance, yaw, iou)
                if error:
                    self.aim_at = None
                    return self.result('FIRE_RECHECK', error)
                self.plan = deepcopy(self.handoff_snapshot['plan'])
            else:
                aim_distance = self.launcher_range.value if self.cfg.get('mission_profile') == 2 else distance
                if aim_distance is None:
                    self.aim_at = None
                    return self.result('FIRE_RANGE_WAIT', 'Collecting five agreeing fresh sonar readings for angle')
                self.plan = solution(box, aim_distance, width, height, self.cfg, vision.get('camera_intrinsics'))
            if not armed:
                return self.result('FIRE_READY_DISARMED', 'Angle preview; physical launcher not armed', plan=self.plan)
            if self.plan['angle_pwm'] is None:
                self.aim_at = None
                return self.result('FIRE_CALIBRATION_BLOCK', self.plan['calibration_error'], plan=self.plan)
            if self.last_aim is None or abs(self.plan['angle_deg']-self.last_aim) > 1.0:
                self.aim_at, self.last_aim = now, self.plan['angle_deg']
            if self.aim_at is None:
                self.aim_at = now
            ready = now-self.aim_at >= self.cfg['aim_settle_seconds']
            if self.cfg.get('mission_profile') == 2:
                ready = ready and self.firing_permitted(now, vision, distance, yaw, box, True)
            return self.result('FIRE_READY' if ready else 'AIM_SETTLE', plan=self.plan, fire=ready)
        self.aim_at = None
        if now-self.last_fire_at < 1.0:
            return self.result('FIRE_LOST_WAIT', 'Fire lost; waiting before resuming navigation')
        if self.cfg.get('mission_profile') == 2:
            return self.result('LAUNCHER_WAIT', 'Stationary mode; waiting for fire target')
        if self.scan.state == 'IDLE':
            self.scan.start(now, yaw)
        action = self.scan.step(now, distance, yaw, sample_id)
        result = self.result(self.scan.state, self.scan.reason, scan_results=dict(self.scan.results))
        result['action'] = action
        result['navigation'] = self.scan.diagnostic(yaw)
        return result
