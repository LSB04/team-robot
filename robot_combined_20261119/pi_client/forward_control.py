"""Smooth forward-only heading feedback. No turns, stops or hardware I/O."""
import math

FORWARD_VERSION = 'forward_hold_v2_reduce_only_6pct'
DEADBAND_DEG = 5.
MAX_TRIM = .06
GAIN_PER_DEG = .006
FILTER_SECONDS = 1.0
TRIM_RATE_PER_SECOND = .02


def finite(value):
    return isinstance(value,(int,float)) and math.isfinite(value)


class ForwardHeading:
    def __init__(self):
        self.reset()

    def reset(self):
        self.target = self.filtered = self.last_time = None
        self.trim = 0.
        self.enabled = False

    def update(self, active, target, yaw, now):
        if not active or not all(finite(v) for v in (target,yaw,now)):
            self.reset()
            return 0.
        if self.target != target or self.last_time is None:
            self.reset()
            self.target,self.filtered,self.last_time = target,yaw,now
            self.enabled = True
            return 0.
        dt=now-self.last_time
        if dt <= 0.:
            return self.trim
        # Long pauses cannot cause a sudden stride jump on the first resumed step.
        dt=min(dt,1.)
        self.last_time=now
        self.filtered+=(1.-math.exp(-dt/FILTER_SECONDS))*(yaw-self.filtered)
        error=self.target-self.filtered
        magnitude=min(MAX_TRIM,max(0.,abs(error)-DEADBAND_DEG)*GAIN_PER_DEG)
        desired=math.copysign(magnitude,error)
        change=TRIM_RATE_PER_SECOND*dt
        self.trim+=max(-change,min(change,desired-self.trim))
        # Existing gait: positive -> reduce LEFT, negative -> reduce RIGHT.
        # Neither side is increased; lift trajectory, phase order and 79% remain.
        return self.trim

    def diagnostic(self, yaw=None):
        return dict(enabled=self.enabled,mode='reduce_only_heading_hold',version=FORWARD_VERSION,
            target_yaw=self.target,filtered_yaw=self.filtered,
            error_deg=None if self.target is None or not finite(yaw) else self.target-yaw,
            filtered_error_deg=None if self.target is None else self.target-self.filtered,
            stride_boost=self.trim,left_stride_reduction=max(0.,self.trim),
            right_stride_reduction=max(0.,-self.trim),deadband_deg=DEADBAND_DEG,
            max_stride_reduction=MAX_TRIM,filter_seconds=FILTER_SECONDS)
