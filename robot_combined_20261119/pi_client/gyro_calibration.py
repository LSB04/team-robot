"""Persist stationary Z-rate bias; yaw itself starts at zero each run."""
import json
import math
import statistics
import time
from pathlib import Path

DEFAULT_PATH = Path(__file__).with_name('gyro_calibration.json')


def load_or_measure(imu, path=DEFAULT_PATH, recalibrate=False, sleep=time.sleep, axis='z'):
    path = Path(path)
    identity = dict(model=getattr(imu, 'model', 'MPU6050'), address=getattr(imu, 'address', 104), axis=axis)
    if path.exists() and not recalibrate:
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            value = data['z_bias_dps']
            if data['schema'] != 2 or any(data.get(k) != v for k, v in identity.items()):
                raise ValueError('Calibration metadata mismatch')
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('Invalid stored bias')
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError('Invalid gyro calibration file. Use --recalibrate-gyro while stationary.') from exc
        print(f'GYRO: loaded saved bias {value:.6f} dps from {path}', flush=True)
        return value
    print('GYRO: keep still; measuring bias once.', flush=True)
    readings = []
    for _ in range(100):
        value = float(imu.get_gyro_data()[axis])
        if not math.isfinite(value):
            raise ValueError('Invalid gyro calibration sample')
        readings.append(value)
        sleep(.01)
    if max(readings)-min(readings) > 5:
        raise RuntimeError('Keep robot still during gyro calibration; previous file unchanged')
    bias = statistics.median(readings)
    data = dict(schema=2, **identity, z_bias_dps=bias, saved_unix=time.time())
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)
    print(f'GYRO: saved bias {bias:.6f} dps to {path}', flush=True)
    return bias
