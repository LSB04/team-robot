"""Fresh sonar, HR22 magnetic/gyro heading and acceleration. No import-time hardware access."""
import math
from collections import deque
from statistics import median
from gyro_calibration import load_or_measure
import threading
import time


def sonar_read_status(raw_cm, error, echo_high_start, rises, falls):
    """Describe software observations, not a hardware fault diagnosis."""
    if error:
        return 'read_exception'
    if raw_cm is None:
        if echo_high_start and not rises and not falls:
            return 'echo_high_no_edges'
        if rises and not falls:
            return 'echo_fall_missing'
        if falls and not rises:
            return 'echo_rise_missing'
        if not rises and not falls:
            return 'no_echo_edges'
        return 'echo_measurement_missing'
    if not math.isfinite(raw_cm):
        return 'nonfinite_value'
    if raw_cm < 2:
        return 'below_min_range'
    if raw_cm >= 400:
        return 'max_range_or_clipped'
    return 'ok'


class SonarDiagnostics:
    """Mixin: observe the existing gpiozero read without altering its timing rules."""
    def __init__(self, **kwargs):
        self.latest = (None, 0., 0)
        self.diag_latest = None
        self.diag_context = {'state': 'INITIALIZING', 'action': 'stop', 'phase': None}
        self.diag_edges = (0, 0)
        self.diag_started = None
        self.diag_previous_start = None
        self.diag_records = deque(maxlen=2048)
        # Independent of the telemetry drain: control must see intervening echoes.
        self.range_records = deque(maxlen=128)
        self.diag_lock = threading.Lock()
        self.diag_dropped = 0
        self.diag_failures = 0
        super().__init__(partial=True, queue_len=1, max_distance=4, **kwargs)

    def _echo_changed(self, ticks, level):
        rises, falls = self.diag_edges
        self.diag_edges = (rises+int(bool(level)), falls+int(not level))
        super()._echo_changed(ticks, level)

    def _echo_level(self):
        try:
            return bool(self.pin.state)
        except Exception:
            return None

    def _read(self):
        started = time.monotonic()
        interval = None if self.diag_previous_start is None else (started-self.diag_previous_start)*1000
        self.diag_previous_start = self.diag_started = started
        self.diag_edges = (0, 0)
        context = dict(self.diag_context)
        high_start = self._echo_level()
        value, error = None, None
        try:
            value = super()._read()
            return value
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            finished = time.monotonic()
            raw_cm = None if value is None else value*400
            seq = self.latest[2]+1
            self.latest = (raw_cm, finished, seq)
            rises, falls = self.diag_edges
            status = sonar_read_status(raw_cm, error, high_start, rises, falls)
            self.diag_failures = 0 if status == 'ok' else self.diag_failures+1
            record = dict(seq=seq, pi_monotonic=finished, pi_unix=time.time(),
                          raw_cm=raw_cm if raw_cm is not None and math.isfinite(raw_cm) else None,
                          read_status=status, read_ms=(finished-started)*1000,
                          interval_ms=interval, echo_high_start=high_start,
                          echo_high_end=self._echo_level(), rising_edges=rises,
                          falling_edges=falls, consecutive_failures=self.diag_failures,
                          error=error, **context)
            with self.diag_lock:
                if len(self.diag_records) == self.diag_records.maxlen:
                    self.diag_dropped += 1
                self.diag_records.append(record)
                self.range_records.append(record)
                self.diag_latest = record
            self.diag_started = None

    def diagnostic(self):
        now = time.monotonic()
        with self.diag_lock:
            record = dict(self.diag_latest) if self.diag_latest else {}
            dropped = self.diag_dropped
        age_ms = (now-record['pi_monotonic'])*1000 if record else None
        status = record.get('read_status', 'no_sample_yet')
        if age_ms is not None and age_ms > 350:
            status = 'stale_update'
        started = self.diag_started
        return dict(record, status=status, sample_age_ms=age_ms,
                    read_in_progress_ms=None if started is None else (now-started)*1000,
                    buffered_records_dropped=dropped)

    def drain_diagnostics(self, limit=24):
        with self.diag_lock:
            return [self.diag_records.popleft() for _ in range(min(limit, len(self.diag_records)))]

    def range_history(self, after_seq=None):
        with self.diag_lock:
            return [dict(r) for r in self.range_records
                    if after_seq is None or r['seq'] > after_seq]


class Sensors:
    def __init__(self, trigger, echo, gyro_sign, recalibrate=False, imu_address=0x68, imu_bus=1, gyro_axis='z', mag_calibration=None):
        from gpiozero import DistanceSensor
        from heading_fusion import FusionIMU
        from pathlib import Path
        if gyro_axis != 'z':
            raise ValueError('HR22 fusion requires horizontal IMU, +Z up, gyro axis z')
        class Sonar(SonarDiagnostics, DistanceSensor):
            pass
        self.imu = self.sonar = None
        self.axis = 'z'
        try:
            self.imu = FusionIMU(imu_address, imu_bus)
            path = mag_calibration or Path(__file__).with_name(f'mag_MPU9250_{imu_bus}_{imu_address:02x}.json')
            print('IMU: loading magnetic calibration; keep still for stationary zero.', flush=True)
            self.imu.initialize_heading(path, gyro_sign)
            self.sonar = Sonar(trigger=trigger, echo=echo)
        except BaseException:
            self.close()
            raise

    def set_context(self, state, action, phase=None):
        self.sonar.diag_context = dict(state=state, action=action, phase=phase)

    def diagnostic(self):
        return self.sonar.diagnostic()

    def drain_diagnostics(self):
        return self.sonar.drain_diagnostics()

    def range_history(self, after_seq=None):
        return self.sonar.range_history(after_seq)

    def diagnostic_info(self):
        from importlib.metadata import version, PackageNotFoundError
        try:
            gpiozero_version = version('gpiozero')
        except PackageNotFoundError:
            gpiozero_version = 'unknown'
        factory = type(self.sonar.pin_factory)
        return dict(gpiozero_version=gpiozero_version,
                    imu_model=self.imu.model, imu_address=self.imu.address, gyro_axis=self.axis,
                    heading_mode='hr22_magnetic_gyro_continuous_relative',
                    magnetic_calibration=self.imu.calibration_path,
                    gyro_sign=self.imu.estimator.sign,
                    pin_factory=f'{factory.__module__}.{factory.__name__}',
                    sonar_stale_ms=350, minimum_cm=2, maximum_cm=400,
                    note='Echo edges are software callbacks, not oscilloscope measurements.')

    def heading_diagnostic(self):
        return self.imu.diagnostic()

    def acceleration(self):
        return self.imu.snapshot()[1]

    def snapshot(self):
        now = time.monotonic()
        distance, stamp, seq = self.sonar.latest
        yaw, _ = self.imu.snapshot()
        distance_ok = distance is not None and math.isfinite(distance) and 2 <= distance < 400
        return (distance if distance_ok and now-stamp <= .35 else None, yaw, seq)

    def close(self):
        if self.sonar:
            self.sonar.close()
        if self.imu:
            self.imu.close()
