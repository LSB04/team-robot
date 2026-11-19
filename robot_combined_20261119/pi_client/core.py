"""Repeated sonar stop and farther-side navigation. No hardware imports."""
import math
import statistics


def valid_distance(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and 2 <= value < 400


def valid_yaw(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


class Scan:
    def __init__(self, threshold=47., angle=90., settle=.6, selected_settle=2.):
        self.threshold, self.angle, self.settle = threshold, angle, settle
        self.selected_settle = selected_settle
        self.sonar_dropouts = 0
        self.state = 'IDLE'
        self.results = {}
        self.reason = ''

    def start(self, now, yaw):
        if self.state not in ('IDLE', 'DONE', 'FAULT'):
            return False
        if not valid_yaw(yaw):
            self.fail('No fresh gyro value')
            return False
        self.state, self.reason, self.results = 'FORWARD', '', {}
        self.sonar_dropouts = 0
        return True

    def stop(self, reason='Operator stop'):
        self.state, self.reason = 'IDLE', reason

    def fail(self, reason):
        self.state, self.reason = 'FAULT', reason
        return 'stop'

    def turn(self, state, target, yaw, now):
        self.state, self.target, self.turn_start = state, target, yaw
        self.sign = 1 if target >= yaw else -1
        self.deadline = now + max(15., abs(target-yaw)/90*25)
        self.best, self.progress_at = 0., now
        self.turn_degrees = abs(target-yaw)
        self.base_speed = 100
        lead = max(2., min(self.turn_degrees / 90 * self.base_speed / 100 * 8., 15.))
        self.predict_offset = max(2., lead * .7) if self.sign > 0 else lead

    def begin_check(self, now, sample_id):
        self.state = 'CHECK_SAMPLE'
        self.ready_at, self.sample_deadline = now+self.selected_settle, now+self.selected_settle+10
        self.samples, self.last_sample = [], sample_id

    def sonar_lost(self, now, sample_id):
        if self.state == 'SONAR_WAIT':
            return
        self.sonar_dropouts += 1
        self.results['sonar_dropouts'] = self.sonar_dropouts
        if self.sonar_dropouts >= 3:
            self.fail('Repeated sonar loss (3); inspect sensor before explicit restart')
            return
        self.state = 'SONAR_WAIT'
        self.reason = 'Sonar missing; stopped while waiting for fresh readings'
        self.recovery_deadline = now+10.
        self.recovery_samples, self.recovery_sample_id = [], sample_id

    def forward_completed(self, now, sample_id):
        if self.state != 'FORWARD':
            return
        self.state = 'STEP_CHECK'
        self.reason = 'Gait phase finished; stationary sonar check'
        self.ready_at, self.sample_deadline = now+self.settle, now+self.settle+10
        self.samples, self.last_sample = [], sample_id

    def remaining(self, yaw):
        return max(0., self.turn_degrees - abs(yaw-self.turn_start))

    def turn_finished(self, yaw):
        return self.remaining(yaw) <= self.predict_offset

    def turn_speed(self, yaw):
        remaining = self.remaining(yaw)
        return max(65, min(self.base_speed, int(65+(self.base_speed-65)*remaining/40)))

    def step(self, now, distance, yaw, sample_id):
        if self.state in ('IDLE', 'DONE', 'FAULT'):
            return 'stop'
        if not valid_yaw(yaw):
            return self.fail('Gyro missing or stale')
        if self.state == 'SONAR_WAIT':
            if now >= self.recovery_deadline:
                return self.fail('Sonar unavailable for 10 seconds; explicit restart required')
            if sample_id != self.recovery_sample_id:
                self.recovery_sample_id = sample_id
                if not valid_distance(distance):
                    self.recovery_samples.clear()
                else:
                    self.recovery_samples.append(distance)
            elif not valid_distance(distance):
                self.recovery_samples.clear()
            if len(self.recovery_samples) >= 3:
                self.state, self.reason = 'FORWARD', ''
                # Re-evaluate the latest distance on the next tick before any movement.
            return 'stop'
        if self.state == 'FORWARD':
            if not valid_distance(distance):
                self.sonar_lost(now, sample_id)
                return 'stop'
            if distance <= self.threshold:
                self.origin = yaw
                self.results = {'front_cm': distance}
                self.turn('LEFT_TURN', yaw+self.angle, yaw, now)
                self.pause_until = now+self.settle
                return 'stop'
            return 'forward'
        if self.state.endswith('_TURN'):
            if now < getattr(self, 'pause_until', 0):
                return 'stop'
            progress = abs(yaw-self.turn_start)
            if progress > self.best+.5:
                self.best, self.progress_at = progress, now
            if self.turn_finished(yaw):
                if self.state == 'SELECT_TURN':
                    self.begin_check(now, sample_id)
                else:
                    self.state = self.state.replace('_TURN', '_SAMPLE')
                    self.ready_at, self.sample_deadline = now+self.settle, now+self.settle+3
                    self.samples, self.last_sample = [], sample_id
                return 'stop'
            if now > self.deadline or now-self.progress_at > 5:
                return self.fail('Turn timeout or no gyro progress')
            return 'left' if self.sign > 0 else 'right'
        if self.state == 'STEP_CHECK':
            if now >= self.sample_deadline:
                return self.fail('No stable sonar after gait phase for 10 seconds')
            if now < self.ready_at:
                self.last_sample = sample_id
                return 'stop'
            if sample_id != self.last_sample:
                self.last_sample = sample_id
                if valid_distance(distance):
                    self.samples.append(distance)
                else:
                    self.samples.clear()
            elif not valid_distance(distance):
                self.samples.clear()
            if len(self.samples) >= 3:
                self.results['stationary_cm'] = statistics.median(self.samples[-3:])
                self.state, self.reason = 'FORWARD', ''
            return 'stop'
        if self.state == 'CHECK_SAMPLE':
            if now > self.sample_deadline:
                return self.fail('No fresh sonar in selected direction')
            if now < self.ready_at:
                self.last_sample = sample_id
                return 'stop'
            if sample_id != self.last_sample:
                self.last_sample = sample_id
                if not valid_distance(distance):
                    self.samples.clear()
                elif distance <= self.threshold:
                    return self.fail(f'Selected direction blocked at {self.threshold:g} cm or less')
                else:
                    self.samples.append(distance)
            if len(self.samples) >= 3:
                self.results['selected_current_cm'] = statistics.median(self.samples[-3:])
                self.state = 'FORWARD'
            return 'stop'
        if self.state.endswith('_SAMPLE'):
            if now > self.sample_deadline:
                return self.fail('No fresh sonar samples at scan position')
            if now < self.ready_at:
                self.last_sample = sample_id
                return 'stop'
            if sample_id != self.last_sample:
                self.last_sample = sample_id
                if valid_distance(distance):
                    self.samples.append(distance)
                else:
                    self.samples.clear()
            if len(self.samples) >= 3:
                side = 'left' if self.state == 'LEFT_SAMPLE' else 'right'
                self.results[side+'_cm'] = statistics.median(self.samples[-3:])
                self.results[side+'_yaw_deg'] = yaw-self.origin
                if side == 'left':
                    self.turn('RIGHT_TURN', yaw-2*self.angle, yaw, now)
                else:
                    left, right = self.results['left_cm'], self.results['right_cm']
                    if max(left, right) <= self.threshold:
                        return self.fail(f'Both directions blocked at {self.threshold:g} cm or less')
                    self.results['selected_side'] = 'left' if left > right else 'right'
                    if left > right:
                        self.turn('SELECT_TURN', yaw+2*self.angle, yaw, now)
                    else:
                        self.begin_check(now, sample_id)
            return 'stop'
        return self.fail('Unknown state')
