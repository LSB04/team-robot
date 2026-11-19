"""Server1 fire approach filtering. No hardware access or shot authorization."""
from collections import deque
from statistics import median
from core import valid_distance

VERSION = 'fire_approach_v10_arrival_lock'


def fire_alignment_tolerance(width, far=False):
    """Test9 screen-width ratios: 120/640 in transit, 20/640 at arrival."""
    return (120. if far else 20.) * float(width) / 640.


class FireRangeConsistency:
    """Keep raw pulse history across short reviews, rotations and leveling.

    This checks agreement, not which echo belongs to the fire plane. Never
    select the farther cluster or substitute a measured reference distance.
    """
    WINDOW = 3.0
    MIN_SPAN = 1.2
    MIN_SAMPLES = 12
    STOPPED_SECONDS = 1.5
    MAX_SPREAD = 20.0
    SPLIT_SPREAD = 60.0
    MAX_AGE = .35

    def __init__(self, target_cm=200., obstacle_cm=50.):
        self.target_cm, self.obstacle_cm = target_cm, obstacle_cm
        self.reset()

    def reset(self):
        self.samples = deque(maxlen=128)
        self.last_seq = self.last_time = None

    def observe(self, now, raw, seq):
        # A replayed/stale sequence cannot refresh evidence or erase a conflict.
        if self.last_seq is not None and seq <= self.last_seq:
            return
        if self.last_time is not None:
            if now < self.last_time:
                return
            if now-self.last_time > self.MAX_AGE or seq > self.last_seq+1:
                self.samples.append((now, None))
        self.last_seq, self.last_time = seq, now
        self.samples.append((now, float(raw) if valid_distance(raw) else None))
        while self.samples and now-self.samples[0][0] > self.WINDOW:
            self.samples.popleft()

    def diagnostic(self, now, stopped_since=None):
        recent = [(t, v) for t, v in self.samples if 0 <= now-t <= self.WINDOW]
        values = [v for _, v in recent if v is not None]
        low, high = (min(values), max(values)) if values else (None, None)
        spread = high-low if values else None
        span = recent[-1][0]-recent[0][0] if recent else 0.
        age = now-self.last_time if self.last_time is not None else None
        fresh = age is not None and 0 <= age <= self.MAX_AGE
        split = bool(values and low <= self.target_cm < high and spread >= self.SPLIT_SPREAD)
        invalid = len(recent)-len(values)
        enough = len(values) >= self.MIN_SAMPLES and span >= self.MIN_SPAN
        consistent = bool(fresh and enough and not invalid and spread <= self.MAX_SPREAD)
        near = bool(consistent and self.obstacle_cm < low and high <= self.target_cm)
        stopped = now-stopped_since if stopped_since is not None else 0.
        ready = near and stopped >= self.STOPPED_SECONDS
        if not fresh:
            code = 'RANGE_STALE'
        elif split:
            code = 'RANGE_SPLIT'
        elif invalid:
            code = 'RANGE_GAP_OR_INVALID'
        elif not enough:
            code = 'RANGE_COLLECTING'
        elif spread > self.MAX_SPREAD:
            code = 'RANGE_UNSTABLE'
        elif not near:
            code = 'RANGE_NOT_ARRIVED'
        elif not ready:
            code = 'RANGE_STOP_SETTLING'
        else:
            code = 'RANGE_NEAR_CONFIRMED'
        return dict(code=code, split=split, near_consistent=near, arrival_ready=ready,
                    raw_min_cm=low, raw_max_cm=high, spread_cm=spread,
                    valid_samples=len(values), invalid_samples=invalid, span_seconds=span,
                    stopped_seconds=stopped, latest_age_seconds=age, last_seq=self.last_seq,
                    window_seconds=self.WINDOW, min_samples=self.MIN_SAMPLES,
                    min_span_seconds=self.MIN_SPAN, stopped_required_seconds=self.STOPPED_SECONDS,
                    max_spread_cm=self.MAX_SPREAD, split_spread_cm=self.SPLIT_SPREAD)

    def approach_stop_ready(self, now):
        # Moving range naturally decreases: use a short near window to stop,
        # then the full stationary history to authorize arrival/save.
        recent = [(t,v) for t,v in self.samples if 0 <= now-t <= .7]
        if (len(recent) < 5 or recent[-1][0]-recent[0][0] < .25
                or now-recent[-1][0] > self.MAX_AGE
                or any(v is None or not self.obstacle_cm < v <= self.target_cm for _,v in recent)):
            return False
        values = [v for _,v in recent]
        return max(values)-min(values) <= self.MAX_SPREAD and not self.diagnostic(now)['split']


class FireDistance:
    """Median of three distinct fresh pulses; hard obstacles bypass the filter."""
    MAX_AGE = .35

    def __init__(self, obstacle_cm=50.):
        self.obstacle_cm = obstacle_cm
        self.reset()

    def reset(self):
        self.samples = deque(maxlen=3)
        self.last_seq = None
        self.last_time = None
        self.raw = self.control = None

    def update(self, now, raw, seq):
        if not valid_distance(raw):
            self.reset()
            return None
        if self.last_time is not None and now-self.last_time > self.MAX_AGE:
            # An unchanged sequence must not become fresh simply by rereading it.
            self.samples.clear()
            self.control = None
            if seq == self.last_seq:
                return None
        if seq == self.last_seq:
            return self.control
        if self.last_seq is not None and seq < self.last_seq:
            self.samples.clear()
        self.last_seq, self.last_time, self.raw = seq, now, raw
        if raw <= self.obstacle_cm:
            self.samples.clear()
            self.control = raw
            return raw
        self.samples.append((now, raw))
        recent = [value for stamp, value in self.samples if 0 <= now-stamp <= self.MAX_AGE]
        self.control = median(recent) if len(recent) == 3 else None
        return self.control

    def diagnostic(self):
        return dict(raw_cm=self.raw, control_cm=self.control, samples=len(self.samples),
                    window=3, max_age_seconds=self.MAX_AGE, last_seq=self.last_seq)

    def stable_far(self, target):
        values = [value for _, value in self.samples]
        return (self.control is not None and len(values) == 3
                and min(values) > target and max(values)-min(values) <= 20.)


class FireAlignment:
    """Approach uses a wider entry band; final saved position stays precise."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.turning = False
        self.dx = self.inner = self.outer = None
        self.final = False

    def update(self, dx, tolerance, final=False, outer_multiplier=2.):
        self.dx, self.inner, self.outer = dx, tolerance, tolerance*outer_multiplier
        self.final = final
        if abs(dx) <= tolerance:
            self.turning = False
        elif final or abs(dx) > self.outer:
            self.turning = True
        return self.turning

    def diagnostic(self):
        return dict(turning=self.turning, error_px=self.dx, centered_px=self.inner,
                    realign_px=self.outer, final_alignment=self.final)
