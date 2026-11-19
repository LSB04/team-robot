"""Bounded HR22-style PID/Kalman leveling using the existing guarded leg writer."""
import math
import time
from Kalman_filter import Kalman_filter


class LevelingError(RuntimeError):
    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        super().__init__('Level check failed: '+', '.join(diagnostics['failed_checks']))


def measurement_diagnostic(accel, dt):
    def finite(value):
        try:
            result = float(value)
            return result if math.isfinite(result) else None
        except (TypeError, ValueError, OverflowError):
            return None
    interval = finite(dt)
    values = {key: finite(accel.get(key)) if isinstance(accel, dict) else None
              for key in ('x', 'y', 'z')}
    norm = math.sqrt(sum(v*v for v in values.values())) if all(v is not None for v in values.values()) else None
    if norm is not None and not math.isfinite(norm):
        norm = None
    errors = []
    if interval is None:
        errors.append('DT_INVALID')
    elif interval < .01:
        errors.append('DT_TOO_SHORT')
    elif interval > .2:
        errors.append('DT_TOO_LONG')
    if norm is None:
        errors.append('ACCEL_MISSING_OR_INVALID')
    else:
        if not 7.5 < norm < 12.:
            errors.append('ACCEL_MAGNITUDE_OUT_OF_RANGE')
        if values['z'] < 7.:
            errors.append('ACCEL_Z_TOO_LOW')
    return dict(code=errors[0] if errors else 'OK', failed_checks=errors,
                dt_seconds=interval, dt_ms=None if interval is None else interval*1000.,
                acceleration_m_s2=values, acceleration_norm_m_s2=norm,
                limits=dict(dt_min_ms=10., dt_max_ms=200., norm_min_m_s2=7.5,
                            norm_max_m_s2=12., z_min_m_s2=7.))


class LevelController:
    def __init__(self):
        self.filters=[Kalman_filter(.001,.1),Kalman_filter(.001,.1)]
        self.previous=None
        self.offsets={1:0.,4:0.,7:0.,10:0.}
        self.last_diagnostic = None

    def update(self, accel, dt):
        self.last_diagnostic = measurement_diagnostic(accel, dt)
        if self.last_diagnostic['failed_checks']:
            raise LevelingError(dict(self.last_diagnostic))
        error=[-f.kalman(accel[k]) for f,k in zip(self.filters,('x','y'))]
        # HR22 Kp=.3, Kd=.1, Ki=0. Avoid the initial derivative spike.
        outputs=[.3*e+(.1*(e-p)/dt if self.previous is not None else 0) for e,p in zip(error,self.previous or error)]
        self.previous=error
        pitch,roll=outputs
        desired={1:150*(pitch+roll),4:150*(pitch-roll),7:150*(-pitch+roll),10:150*(-pitch-roll)}
        for channel,value in desired.items():
            target=max(-35.,min(35.,value))
            self.offsets[channel]+=max(-3.,min(3.,target-self.offsets[channel]))
        return dict(self.offsets)


def level_robot(motion, sensors, guard, sleep=time.sleep, clock=time.monotonic):
    guard()
    motion.gait.robot_stand()
    motion.phase, motion.previous, motion.stood=1,None,True
    motion.sleep(.15)
    control=LevelController()
    previous=clock()
    measurements=[]
    for iteration in range(1,16):
        guard();motion.sleep(.04)
        now=clock()
        try:
            offsets=control.update(sensors.acceleration(),now-previous)
        except LevelingError as exc:
            exc.diagnostics.update(iteration=iteration, total_iterations=15,
                                   previous_offsets_pwm=dict(control.offsets),
                                   prior_measurements=measurements)
            raise
        measurements.append(dict(iteration=iteration, **control.last_diagnostic))
        previous=now
        for ch,delta in offsets.items():
            guard()
            motion.write(ch,motion.neutral[ch]+delta)
    motion.sleep(.3)
    return dict(iterations=15,offsets=offsets,limited_pwm_delta=35,
                max_dt_ms=max(m['dt_ms'] for m in measurements),
                measurements=measurements)
