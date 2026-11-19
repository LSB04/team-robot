"""HR22 MPU9250/AK8963 fusion, with continuous yaw and saved field calibration.

No import-time hardware access. HR22 uses origin minus magnetic heading, and
positive gyro Z for a left turn. --gyro-sign reverses both if mounting requires it.
"""
import json
import math
from pathlib import Path
import struct
import threading
import time
from collections import deque


class SampleNotReady(RuntimeError):
    """No new sample yet; retry without ending the measurement worker."""


def wrap(angle):
    return (angle + 180.) % 360. - 180.


def field_calibration(fields):
    if len(fields) < 50 or any(len(f) != 3 or not all(map(math.isfinite, f)) for f in fields):
        raise ValueError('At least 50 finite magnetic samples required')
    xs, ys, _ = zip(*fields)
    cx, cy = (min(xs)+max(xs))/2, (min(ys)+max(ys))/2
    sx, sy = max(xs)-min(xs), max(ys)-min(ys)
    if min(sx, sy) < 10 or len({(x>cx, y>cy) for x,y in zip(xs,ys)}) < 4:
        raise ValueError('Incomplete magnetic calibration: rotate a full level circle')
    # Reject a small arc whose extrema alone happen to cover four quadrants.
    angles = {int((math.degrees(math.atan2((y-cy)/sy, (x-cx)/sx))+180)//30)%12 for x,y in zip(xs,ys)}
    if len(angles) < 10:
        raise ValueError('Magnetic calibration needs coverage of a full circle')
    return dict(cx=cx, cy=cy, sx=sx, sy=sy)


def validate_calibration(data, address, bus):
    if data.get('version') != 1 or data.get('imu_address') != address or data.get('bus') != bus:
        raise ValueError('Magnetic calibration identity mismatch; run calibrate_heading.py')
    cal = {k: data[k] for k in ('cx','cy','sx','sy')}
    if not all(isinstance(x,(int,float)) and math.isfinite(x) for x in cal.values()) or min(cal['sx'],cal['sy']) < 10:
        raise ValueError('Invalid magnetic calibration')
    return cal


class Estimator:
    def __init__(self, calibration, bias, origin, sign, now):
        if sign not in (-1,1):
            raise ValueError('Invalid heading sign')
        self.cal, self.bias, self.origin, self.sign = calibration,bias,origin,sign
        self.yaw = 0.
        self.previous = self.last_mag = now
        self.rejected = 0

    def heading(self, field):
        c = self.cal
        return math.degrees(math.atan2((field[1]-c['cy'])/c['sy'], (field[0]-c['cx'])/c['sx']))

    def update(self, rate, field, now):
        dt = now-self.previous
        if not math.isfinite(rate):
            raise RuntimeError(f'Gyro rate nonfinite: {rate}')
        if abs(rate) >= 249:
            raise RuntimeError(f'Gyro rate saturated: {rate:.3f} deg/s')
        if not 0 < dt <= .2:
            raise RuntimeError(f'Gyro sample gap: {dt*1000:.1f} ms (limit 200 ms)')
        self.previous = now
        self.yaw += (rate-self.bias)*self.sign*dt
        if field is not None and all(map(math.isfinite,field)):
            c = self.cal
            radius = math.hypot((field[0]-c['cx'])/c['sx'], (field[1]-c['cy'])/c['sy'])
            innovation = wrap(self.sign*wrap(self.origin-self.heading(field))-self.yaw)
            if .2 <= radius <= .9 and abs(innovation) <= 25:
                self.yaw += (1-math.exp(-dt/2.))*innovation
                self.last_mag = now
            else:
                self.rejected += 1
        if now-self.last_mag > 1.5:
            raise RuntimeError('Magnetic samples missing or rejected for 1.5 seconds')
        # Intentionally unwrapped: Scan targets use continuous relative yaw.
        return self.yaw


class FusionIMU:
    model = 'MPU9250+AK8963'
    def __init__(self, address=0x68, bus_number=1, bus=None, sleep=time.sleep):
        self.address, self.bus_number, self.sleep = address,bus_number,sleep
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.thread = None
        self.latest = None
        self.error = None
        self.last_error = None
        self.error_count = self.recoveries = self.not_ready_count = 0
        self.last_rate_dps = self.last_gap_ms = None
        self.max_gap_ms = 0.
        self.last_read = None
        self.recovery = deque(maxlen=10)
        self.previous_mode = self.orig_user = self.orig_pins = None
        if address not in (0x68,0x69):
            raise ValueError('Invalid MPU9250 address')
        if bus is None:
            try:
                from smbus2 import SMBus
            except ImportError:
                from smbus import SMBus
            bus = SMBus(bus_number)
        self.bus = bus
        try:
            self.identity = bus.read_byte_data(address,0x75)
            if self.identity not in (0x70,0x71):
                raise RuntimeError(f'IMU WHO_AM_I=0x{self.identity:02X} at bus {bus_number}, address 0x{address:02X}; HR22 accepts 0x70 or 0x71. Check model/address.')
            self.model = 'MPU9250+AK8963' if self.identity == 0x71 else 'HR22-compatible IMU(0x70)+AK8963'
            self.orig_user = bus.read_byte_data(address,0x6A)
            self.orig_pins = bus.read_byte_data(address,0x37)
            bus.write_byte_data(address,0x6A,self.orig_user & ~0x27)
            bus.write_byte_data(address,0x37,self.orig_pins | 2)
            for reg,val in ((0x6B,1),(0x6C,0),(0x1B,0),(0x1C,0),(0x1A,3),(0x19,4),(0x38,1)):
                bus.write_byte_data(address,reg,val)
            sleep(.05)
            self.mag_identity = bus.read_byte_data(0x0C,0)
            if self.mag_identity != 0x48:
                raise RuntimeError(f'AK8963 ID=0x{self.mag_identity:02X} at 0x0C; expected 0x48. Magnetic fusion requires a working magnetometer.')
            self.previous_mode = bus.read_byte_data(0x0C,0x0A)
            self.mode(0); self.mode(0x0F)
            asa = bus.read_i2c_block_data(0x0C,0x10,3)
            if len(asa)!=3 or any(x in (0,255) for x in asa):
                raise RuntimeError('Invalid AK8963 factory calibration')
            self.factors = [.15*(1+(x-128)/256.) for x in asa]
            self.mode(0); self.mode(0x16)
        except BaseException:
            self.close()
            raise

    def mode(self, value):
        self.bus.write_byte_data(0x0C,0x0A,value)
        self.sleep(.01)

    def sample(self):
        with self.lock:
            if not self.bus.read_byte_data(self.address,0x3A) & 1:
                raise SampleNotReady('MPU9250 sample not ready')
            raw = bytes(self.bus.read_i2c_block_data(self.address,0x3B,14))
            if len(raw)!=14:
                raise RuntimeError('Incomplete MPU9250 sample')
            ax,ay,az,_,gx,gy,gz = struct.unpack('>hhhhhhh',raw)
            if abs(gz)>=32700:
                raise RuntimeError('Gyro saturated')
            accel = dict(zip(('x','y','z'),[v*9.80665/16384 for v in (ax,ay,az)]))
            field = None
            if self.bus.read_byte_data(0x0C,2)&1:
                data=bytes(self.bus.read_i2c_block_data(0x0C,3,7))
                if len(data)!=7:
                    raise RuntimeError('Incomplete AK8963 sample')
                if not data[6]&8:
                    field=tuple(v*f for v,f in zip(struct.unpack('<hhh',data[:6]),self.factors))
            return gz/131.,field,accel

    def initialize_heading(self, path, sign=1):
        path=Path(path)
        if not path.exists():
            raise RuntimeError('Missing magnetic calibration. Run: python3 calibrate_heading.py')
        cal=validate_calibration(json.loads(path.read_text(encoding='utf-8')),self.address,self.bus_number)
        readings=[]
        helper=Estimator(cal,0,0,sign,time.monotonic())
        deadline=time.monotonic()+4
        while len(readings)<75 and time.monotonic()<deadline:
            self.sleep(.02)
            try:
                rate,field,accel=self.sample()
            except SampleNotReady:
                continue
            if abs(rate)>4 or abs(accel['x'])>2.5 or abs(accel['y'])>2.5 or not 8<accel['z']<11:
                raise RuntimeError('Keep IMU still and level with +Z up during zeroing')
            if field is not None:
                readings.append((rate,helper.heading(field)))
        if len(readings)<75:
            raise RuntimeError('Insufficient stationary magnetic samples')
        bias=sum(r for r,_ in readings)/len(readings)
        radians=[math.radians(h) for _,h in readings]
        origin=math.degrees(math.atan2(sum(map(math.sin,radians)),sum(map(math.cos,radians))))
        if max(abs(wrap(h-origin)) for _,h in readings)>8:
            raise RuntimeError('Unstable magnetic heading during stationary zeroing')
        self.estimator=Estimator(cal,bias,origin,sign,time.monotonic())
        self.calibration_path=str(path)
        self.thread=threading.Thread(target=self.run,daemon=True)
        self.thread.start()

    def run(self):
        while not self.done.wait(.02):
            self.poll_once()

    def invalidate(self, exc):
        self.error = f'{type(exc).__name__}: {exc}'
        self.last_error = self.error
        self.error_count += 1
        self.recovery.clear()

    def poll_once(self):
        """Keep retrying reads; invalid heading never becomes usable stale data."""
        try:
            rate,field,accel = self.sample()
            self.accept_sample(rate,field,accel,time.monotonic())
        except SampleNotReady as exc:
            self.not_ready_count += 1
            stamp = self.latest[2] if self.latest else self.estimator.previous
            if time.monotonic()-stamp > .2:
                self.invalidate(exc)
        except Exception as exc:
            self.invalidate(exc)

    def accept_sample(self, rate, field, accel, now):
        self.last_rate_dps = rate if math.isfinite(rate) else None
        previous = self.last_read if self.last_read is not None else self.estimator.previous
        self.last_gap_ms = (now-previous)*1000
        self.max_gap_ms = max(self.max_gap_ms,self.last_gap_ms)
        self.last_read = now
        if self.error:
            return self.recover_sample(rate,field,accel,now)
        yaw = self.estimator.update(rate,field,now)
        self.latest = (yaw,accel,now)

    def recover_sample(self, rate, field, accel, now):
        """Re-anchor only from ten fresh, stationary, consistent magnetic samples.

        Never integrate a missed time interval or repeat a previous gyro value.
        The client rechecks navigation conditions before resuming a paused mission.
        """
        e = self.estimator
        stable = (math.isfinite(rate) and abs(rate)<249 and abs(rate-e.bias)<=4
                  and all(math.isfinite(accel[k]) for k in ('x','y','z'))
                  and abs(accel['x'])<=2.5 and abs(accel['y'])<=2.5 and 8<accel['z']<11
                  and field is not None and all(map(math.isfinite,field)))
        if not stable:
            self.recovery.clear()
            return
        c = e.cal
        radius = math.hypot((field[0]-c['cx'])/c['sx'],(field[1]-c['cy'])/c['sy'])
        correction = wrap(e.sign*wrap(e.origin-e.heading(field))-e.yaw)
        if not .2<=radius<=.9 or abs(correction)>25:
            self.recovery.clear()
            return
        yaw = e.yaw+correction
        if self.recovery and (not 0<now-self.recovery[-1][0]<=.1
                              or abs(yaw-self.recovery[0][1])>5):
            self.recovery.clear()
        self.recovery.append((now,yaw))
        if len(self.recovery)<10 or now-self.recovery[0][0]<.15:
            return
        e.yaw = sum(y for _,y in self.recovery)/len(self.recovery)
        e.previous = e.last_mag = now
        self.latest = (e.yaw,accel,now)
        self.error = None
        self.recoveries += 1
        self.recovery.clear()

    def diagnostic(self):
        now = time.monotonic()
        return dict(error=self.error,last_error=self.last_error,
                    sample_age_ms=None if self.latest is None else (now-self.latest[2])*1000,
                    worker_alive=bool(self.thread and self.thread.is_alive()),
                    last_rate_dps=self.last_rate_dps,last_gap_ms=self.last_gap_ms,
                    max_gap_ms=self.max_gap_ms,error_count=self.error_count,
                    recoveries=self.recoveries,recovery_samples=len(self.recovery),
                    not_ready_count=self.not_ready_count,rejected_fields=self.estimator.rejected,
                    magnetic_age_seconds=now-self.estimator.last_mag)

    def snapshot(self):
        sample=self.latest
        if self.error or sample is None or time.monotonic()-sample[2]>.2:
            return None,None
        return sample[0],dict(sample[1])

    def close(self):
        self.done.set()
        if self.thread:
            self.thread.join(1.)
            if self.thread.is_alive():
                return  # Do not close a bus still owned by a blocked worker.
        with self.lock:
            try:
                if self.previous_mode is not None:
                    self.mode(0)
                    self.mode(self.previous_mode)
                if self.orig_pins is not None:
                    self.bus.write_byte_data(self.address,0x37,self.orig_pins)
                if self.orig_user is not None:
                    self.bus.write_byte_data(self.address,0x6A,self.orig_user & ~7)
            finally:
                self.bus.close()
