"""Pure navigation: direct turns and bounded left sonar retries.

Positive continuous yaw is left, as in the existing HR22 adapter.
Tuning defaults are provisional; no hardware access occurs in this module.
"""
from collections import deque
import statistics
from core import Scan, valid_distance, valid_yaw

TURN_TOLERANCE_DEG = None
# Stop within four degrees while approaching the target, or after crossing it.
# No post-turn tolerance checks or reverse correction movements are applied.
TURN_STOP_MARGIN_DEG = 4.0
TRIM_TARGET_BAND_DEG = ()
MAX_TURN_CORRECTIONS = 0
LEFT_RETRY_OFFSETS_DEG = (0., -12., 12.)
SAMPLE_SECONDS = 3.
LEFT_DRIFT_DEADBAND_DEG = 5.0
MAX_LEFT_DRIFT_COMPENSATION_DEG = 20.0
NAVIGATION_VERSION = 'direct_turns_v11_left_reference_forward_hold'


class DirectTurn:
    """Stop within four degrees of the target, or after crossing it."""
    def __init__(self, target, yaw, now):
        self.target = target
        self.phase = 'coarse'
        self.trims = 0
        self.reason = ''
        self.final_yaw = None
        self.sign = 1 if target >= yaw else -1
        self.initial_error = abs(target-yaw)

    def fail(self, reason):
        self.phase, self.reason = 'failed', reason
        return 'stop'

    def motion_finished(self, yaw):
        return self.phase != 'coarse' or self.sign*(self.target-yaw) <= TURN_STOP_MARGIN_DEG

    def step(self, yaw, now):
        if self.phase in ('done','failed'):
            return 'stop'
        if not valid_yaw(yaw):
            return self.fail('Heading missing during turn')
        if self.motion_finished(yaw):
            self.phase, self.final_yaw = 'done', yaw
            return 'stop'
        remaining = abs(self.target-yaw)
        if remaining > self.initial_error+12:
            return self.fail('Turn moved away from target; check direction sign')
        return 'left' if self.sign>0 else 'right'

    def speed(self, yaw):
        remaining = abs(self.target-yaw)
        return max(65,min(100,int(65+35*remaining/40)))


class NavigationScan(Scan):
    def __init__(self, threshold=50., angle=90., **kwargs):
        super().__init__(threshold=threshold,angle=angle,**kwargs)
        self.job = None
        self.pre_scan_level = False
        self.forward_heading = None
        self.left_target = None
        self.left_drift_compensation = 0.
        self.events = deque(maxlen=128)
        self.left_attempt = self.position_rechecks = 0
        self.sample_trims = 0
        self.last_job_phase = None

    def emit(self, event, now, **values):
        self.events.append(dict(event=event,pi_monotonic=now,**values))

    def start(self, now, yaw):
        if not super().start(now,yaw):
            return False
        self.job = None
        self.pre_scan_level = False
        self.forward_heading = yaw
        self.left_target = None
        self.left_drift_compensation = 0.
        self.left_attempt = self.position_rechecks = 0
        self.sample_trims = 0
        return True

    def stop(self, reason='Operator stop'):
        super().stop(reason)
        self.job = None
        self.pre_scan_level = False

    def fail(self, reason):
        self.job = None
        self.pre_scan_level = False
        return super().fail(reason)

    @property
    def needs_level(self):
        return self.pre_scan_level

    @property
    def deadline(self):
        # Compatibility for callers of Scan; navigation turns have no timer.
        return float('inf')

    def turn(self, state, target, yaw, now):
        self.state, self.reason = state, 'Turning toward fixed heading'
        self.job = DirectTurn(target,yaw,now)
        self.sample_trims = 0
        self.last_job_phase = None
        self.emit('TURN_TARGET',now,state=state,target_yaw_deg=target,start_yaw_deg=yaw)

    def turn_finished(self, yaw):
        return self.job is None or self.job.motion_finished(yaw)

    def turn_speed(self, yaw):
        return self.job.speed(yaw) if self.job else 40

    def level_complete(self, yaw, now):
        if not valid_yaw(yaw):
            return self.fail('Heading unavailable after leveling')
        if self.pre_scan_level:
            self.pre_scan_level = False
            self.origin = yaw
            # Keep the RIGHT target tied to the current, leveled origin as before.
            # Only remove accumulated LEFT drift from the first LEFT target.
            drift=(yaw-self.forward_heading) if valid_yaw(self.forward_heading) else 0.
            self.left_drift_compensation=(min(MAX_LEFT_DRIFT_COMPENSATION_DEG,drift)
                                          if drift>LEFT_DRIFT_DEADBAND_DEG else 0.)
            self.left_target=self.origin+self.angle-self.left_drift_compensation
            self.emit('LEFT_REFERENCE',now,forward_heading_deg=self.forward_heading,
                      scan_origin_yaw_deg=self.origin,accumulated_left_drift_deg=drift,
                      left_target_yaw_deg=self.left_target,
                      left_drift_compensation_deg=self.left_drift_compensation,
                      right_target_yaw_deg=self.origin-self.angle)
            self.turn('LEFT_TURN',self.left_target,yaw,now)
        else:
            raise RuntimeError('Navigation did not request leveling')

    def begin_check(self, now, sample_id, heading=None):
        self.state, self.reason = 'CHECK_SAMPLE', 'Checking distance after turn'
        self.job = None
        self.sample_target = self.forward_heading if heading is None else heading
        self.ready_at = now+self.selected_settle
        self.sample_deadline = self.ready_at+SAMPLE_SECONDS
        self.samples, self.last_sample = [], sample_id

    def _sample_side(self, state, target, now, sample_id):
        self.job = None
        self.state, self.reason = state, 'Stationary sonar measurement'
        self.sample_target = target
        self.samples, self.last_sample = [], sample_id
        self.ready_at = now+.8
        self.sample_deadline = self.ready_at+SAMPLE_SECONDS

    def _finish_side(self, side, value, yaw, now, sample_id):
        self.results[side+'_cm'] = value
        self.results[side+'_yaw_deg'] = yaw-self.origin
        self.results[side+'_heading_deg'] = yaw
        self.emit('SIDE_RANGE',now,side=side,distance_cm=value,yaw_deg=yaw,
                  offset_deg=LEFT_RETRY_OFFSETS_DEG[self.left_attempt] if side=='left' else 0.)
        if side=='left':
            self.results.setdefault('left_attempts',[]).append(dict(
                offset_deg=LEFT_RETRY_OFFSETS_DEG[self.left_attempt],distance_cm=value,yaw_deg=yaw))
            if value is None and self.left_attempt+1<len(LEFT_RETRY_OFFSETS_DEG):
                self.left_attempt += 1
                self.position_rechecks = 0
                target = self.left_target+LEFT_RETRY_OFFSETS_DEG[self.left_attempt]
                self.turn('LEFT_RETRY_TURN',target,yaw,now)
                return
            self.position_rechecks = 0
            # Fixed origin avoids accumulating the left turn's residual error.
            self.turn('RIGHT_TURN',self.origin-self.angle,yaw,now)
            return
        candidates=[(self.results.get(s+'_cm'),s) for s in ('right','left')
                    if valid_distance(self.results.get(s+'_cm')) and self.results[s+'_cm']>self.threshold]
        if not candidates:
            self.fail('No measured clear side after bounded retries')
            return
        _,chosen=max(candidates,key=lambda x:x[0])
        self.results['selected_side']=chosen
        self.position_rechecks = 0
        target=self.results[chosen+'_heading_deg']
        self.emit('SIDE_SELECTED',now,side=chosen,target_yaw_deg=target)
        if chosen=='right':
            self.begin_check(now,sample_id,target)
        else:
            self.turn('SELECT_TURN',target,yaw,now)

    def diagnostic(self, yaw):
        target=self.job.target if self.job else getattr(self,'sample_target',None) if self.state.endswith('_SAMPLE') else None
        origin=getattr(self,'origin',None)
        return dict(phase='pre_level' if self.pre_scan_level else self.job.phase if self.job else 'sample' if self.state.endswith('_SAMPLE') else self.state.lower(),
                    target_yaw_deg=target,yaw_error_deg=None if target is None or not valid_yaw(yaw) else target-yaw,
                    scan_origin_yaw_deg=origin,target_relative_deg=None if target is None or origin is None else target-origin,
                    turn_tolerance_deg=TURN_TOLERANCE_DEG,
                    turn_stop_margin_deg=TURN_STOP_MARGIN_DEG,
                    turn_completion_mode='heading_target',
                    turn_timeout_seconds=None,no_progress_timeout_seconds=None,
                    trim_target_band_deg=list(TRIM_TARGET_BAND_DEG),
                    max_turn_corrections=MAX_TURN_CORRECTIONS,forward_mode='reduce_only_heading_hold',
                    forward_heading_deg=self.forward_heading,left_target_yaw_deg=self.left_target,
                    left_drift_compensation_deg=self.left_drift_compensation,
                    left_retry=self.left_attempt,trim_count=self.job.trims if self.job else self.sample_trims)

    def step(self, now, distance, yaw, sample_id):
        if self.state in ('IDLE','FAULT','DONE'):
            return 'stop'
        if not valid_yaw(yaw):
            return self.fail('Heading missing or stale')
        if self.state=='SONAR_WAIT':
            return super().step(now,distance,yaw,sample_id)
        if self.state=='FORWARD':
            if not valid_distance(distance):
                self.reason = 'No valid sonar range; forward continues by configuration'
                return 'forward'
            self.reason = ''
            if distance<=self.threshold:
                self.origin = yaw
                self.results = dict(front_cm=distance,left_attempts=[])
                self.left_attempt = self.position_rechecks = 0
                self.state, self.reason = 'LEFT_TURN','Leveling before scan origin is fixed'
                self.pre_scan_level = True
                return 'stop'
            return 'forward'
        if self.pre_scan_level:
            return 'stop'
        if self.job:
            action=self.job.step(yaw,now)
            self.sample_trims = self.job.trims
            if self.job.phase!=self.last_job_phase:
                self.last_job_phase=self.job.phase
                self.reason='Heading '+self.job.phase
                self.emit('TURN_PHASE',now,state=self.state,**self.diagnostic(yaw))
            if self.job.phase=='failed':
                return self.fail(self.job.reason)
            if self.job.phase=='done':
                state,target=self.state,self.job.target
                self.emit('TURN_COMPLETED',now,state=state,target_yaw_deg=target,
                          measured_yaw_deg=self.job.final_yaw,error_deg=target-self.job.final_yaw)
                if state in ('LEFT_TURN','LEFT_RETRY_TURN','RIGHT_TURN'):
                    self._sample_side(state.replace('_TURN','_SAMPLE'),target,now,sample_id)
                else:
                    self.begin_check(now,sample_id,target)
            return action
        if self.state in ('LEFT_SAMPLE','LEFT_RETRY_SAMPLE','RIGHT_SAMPLE','CHECK_SAMPLE'):
            if now<self.ready_at:
                self.last_sample=sample_id
                return 'stop'
            if sample_id!=self.last_sample:
                self.last_sample=sample_id
                if valid_distance(distance):
                    if self.state=='CHECK_SAMPLE' and distance<=self.threshold:
                        # Re-enter normal obstacle scan with the latest bearing.
                        self.forward_heading=self.sample_target
                        self.state='FORWARD'
                        return self.step(now,distance,yaw,sample_id)
                    self.samples.append(distance)
                else:
                    self.samples.clear()
            if len(self.samples)>=3:
                value=statistics.median(self.samples[-3:])
                if self.state=='CHECK_SAMPLE':
                    self.results['selected_current_cm']=value
                    self.forward_heading=self.sample_target
                    self.state,self.reason='FORWARD',''
                    self.position_rechecks=0
                    self.emit('FORWARD_READY',now,target_yaw_deg=self.forward_heading,distance_cm=value)
                else:
                    self._finish_side('right' if self.state=='RIGHT_SAMPLE' else 'left',value,yaw,now,sample_id)
            elif now>self.sample_deadline:
                if self.state=='CHECK_SAMPLE':
                    return self.fail('No fresh sonar after selected turn')
                self._finish_side('right' if self.state=='RIGHT_SAMPLE' else 'left',None,yaw,now,sample_id)
            return 'stop'
        return self.fail('Unknown navigation state')
