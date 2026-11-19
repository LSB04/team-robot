"""Pure gait extracted from Desktop/test8/move.py; hardware writes are injected."""
import time
from types import SimpleNamespace
cfg = SimpleNamespace(GAIT_PIXEL=4, GAIT_REACH_WIGGLE=130, GAIT_MAX_WIGGLE=160, YAW_COMPENSATION=0.0, GAIT_DIAG_DELAY=0.008, GAIT_SUB_STEP_DELAY=0.012)

def _set(channel, value):
    raise RuntimeError("A guarded motor output must be attached before calling gait")

address = 0x40

pwm0  = 265

pwm1  = 380

pwm2  = 313

pwm3  = 375

pwm4  = 217

pwm5  = 260

pwm6  = 380

pwm7  = 241

pwm8  = 240

pwm9  = 350

pwm10 = 380

pwm11 = 360

pwm0_max  = 450

pwm0_min  = 150

pwm1_max  = 450

pwm1_min  = 150

pwm2_max  = 450

pwm2_min  = 150

pwm3_max  = 450

pwm3_min  = 150

pwm4_max  = 450

pwm4_min  = 150

pwm5_max  = 450

pwm5_min  = 150

pwm6_max  = 450

pwm6_min  = 150

pwm7_max  = 450

pwm7_min  = 150

pwm8_max  = 450

pwm8_min  = 150

pwm9_max  = 450

pwm9_min  = 150

pwm10_max = 450

pwm10_min = 150

pwm11_max = 450

pwm11_min = 150

pixel        = cfg.GAIT_PIXEL

reach_wiggle = cfg.GAIT_REACH_WIGGLE

max_wiggle   = cfg.GAIT_MAX_WIGGLE

turn_steady  = 0.8

def _clamp(ch, val):
    limits = [
        (pwm0_min, pwm0_max), (pwm1_min, pwm1_max), (pwm2_min, pwm2_max),
        (pwm3_min, pwm3_max), (pwm4_min, pwm4_max), (pwm5_min, pwm5_max),
        (pwm6_min, pwm6_max), (pwm7_min, pwm7_max), (pwm8_min, pwm8_max),
        (pwm9_min, pwm9_max), (pwm10_min, pwm10_max), (pwm11_min, pwm11_max),
    ]
    lo, hi = limits[ch]
    return max(lo, min(hi, int(val)))

def leg_tripod(name, pos, spot, wiggle):
    increase = spot / pixel
    if wiggle > 0:
        direction = 1
    else:
        direction = 0
        wiggle = -wiggle

    if name == 'I':
        if pos == 1:
            if direction:
                _set(0, pwm0 - wiggle + increase * wiggle)
                _set(1, pwm1 + wiggle / 4 - increase * wiggle * 5 / 4)
                _set(2, pwm2 - increase * max_wiggle)
            else:
                _set(0, pwm0 + wiggle - increase * wiggle)
                _set(1, pwm1 + wiggle - 2 * increase * wiggle)
                _set(2, pwm2 - reach_wiggle - increase * (max_wiggle - reach_wiggle))
        elif pos == 2:
            if direction:
                _set(0, pwm0 + increase * wiggle)
                _set(1, pwm1 - wiggle + increase * wiggle * 2)
                _set(2, pwm2 - max_wiggle + increase * (max_wiggle - reach_wiggle))
            else:
                _set(0, pwm0 - increase * wiggle)
                _set(1, pwm1 - wiggle + 5 * increase * wiggle / 4)
                _set(2, pwm2 - max_wiggle + increase * max_wiggle)
        elif pos == 3:
            if direction:
                _set(0, pwm0 + wiggle - increase * wiggle / 3)
                _set(1, pwm1 + wiggle - increase * wiggle / 6)
                _set(2, pwm2 - reach_wiggle + increase * reach_wiggle / 6)
            else:
                _set(0, pwm0 - wiggle + increase * wiggle / 3)
                _set(1, pwm1 + wiggle / 4 + increase * wiggle / 12)
                _set(2, pwm2 - increase * reach_wiggle / 6)
        elif pos == 4:
            if direction:
                _set(0, pwm0 + 2 * wiggle / 3 - increase * wiggle / 3)
                _set(1, pwm1 + 5 * wiggle / 6 - increase * wiggle / 6)
                _set(2, pwm2 - 5 * reach_wiggle / 6 + increase * reach_wiggle / 6)
            else:
                _set(0, pwm0 - 2 * wiggle / 3 + increase * wiggle / 3)
                _set(1, pwm1 + wiggle / 3 + increase * wiggle / 12)
                _set(2, pwm2 - reach_wiggle / 6 - increase * reach_wiggle / 6)
        elif pos == 5:
            if direction:
                _set(0, pwm0 + wiggle / 3 - increase * wiggle / 3)
                _set(1, pwm1 + 4 * wiggle / 6 - increase * wiggle / 6)
                _set(2, pwm2 - 2 * reach_wiggle / 3 + increase * reach_wiggle / 6)
            else:
                _set(0, pwm0 - wiggle / 3 + increase * wiggle / 3)
                _set(1, pwm1 + 5 * wiggle / 12 + increase * wiggle / 12)
                _set(2, pwm2 - reach_wiggle / 3 - increase * reach_wiggle / 6)
        elif pos == 6:
            if direction:
                _set(0, pwm0 - increase * wiggle / 3)
                _set(1, pwm1 + wiggle / 2 - increase * wiggle / 12)
                _set(2, pwm2 - reach_wiggle / 2 + increase * reach_wiggle / 6)
            else:
                _set(0, pwm0 + increase * wiggle / 3)
                _set(1, pwm1 + wiggle / 2 + increase * wiggle / 6)
                _set(2, pwm2 - reach_wiggle / 2 - increase * reach_wiggle / 6)
        elif pos == 7:
            if direction:
                _set(0, pwm0 - wiggle / 3 - increase * wiggle / 3)
                _set(1, pwm1 + 5 * wiggle / 12 - increase * wiggle / 12)
                _set(2, pwm2 - 2 * reach_wiggle / 6 + increase * reach_wiggle / 6)
            else:
                _set(0, pwm0 + wiggle / 3 + increase * wiggle / 3)
                _set(1, pwm1 + 2 * wiggle / 3 + increase * wiggle / 6)
                _set(2, pwm2 - 2 * reach_wiggle / 3 - increase * reach_wiggle / 6)
        elif pos == 8:
            if direction:
                _set(0, pwm0 - 2 * wiggle / 3 - increase * wiggle / 3)
                _set(1, pwm1 + 4 * wiggle / 12 - wiggle / 12)
                _set(2, pwm2 - reach_wiggle / 6 + increase * reach_wiggle / 6)
            else:
                _set(0, pwm0 + 2 * wiggle / 3 + increase * wiggle / 3)
                _set(1, pwm1 + 5 * wiggle / 6 + increase * wiggle / 6)
                _set(2, pwm2 - 5 * reach_wiggle / 6 - increase * reach_wiggle / 6)

    elif name == 'II':
        if pos == 1:
            if direction:
                _set(3, pwm3 + wiggle - increase * wiggle)
                _set(4, pwm4 - wiggle + increase * wiggle * 2)
                _set(5, pwm5 + reach_wiggle + increase * (max_wiggle - reach_wiggle))
            else:
                _set(3, pwm3 - wiggle + increase * wiggle)
                _set(4, pwm4 - wiggle / 4 + 5 * increase * wiggle / 4)
                _set(5, pwm5 + increase * max_wiggle)
        elif pos == 2:
            if direction:
                _set(3, pwm3 - increase * wiggle)
                _set(4, pwm4 + wiggle - 5 * increase * wiggle / 4)
                _set(5, pwm5 + max_wiggle - increase * max_wiggle)
            else:
                _set(3, pwm3 + increase * wiggle)
                _set(4, pwm4 + wiggle - 2 * increase * wiggle)
                _set(5, pwm5 + max_wiggle - increase * (max_wiggle - reach_wiggle))
        elif pos == 3:
            if direction:
                _set(3, pwm3 - wiggle + increase * wiggle / 3)
                _set(4, pwm4 - wiggle / 4 - increase * wiggle / 12)
                _set(5, pwm5 + increase * reach_wiggle / 6)
            else:
                _set(3, pwm3 + wiggle - increase * wiggle / 3)
                _set(4, pwm4 - wiggle + increase * wiggle / 6)
                _set(5, pwm5 + reach_wiggle - increase * reach_wiggle / 6)
        elif pos == 4:
            if direction:
                _set(3, pwm3 - 2 * wiggle / 3 + increase * wiggle / 3)
                _set(4, pwm4 - 4 * wiggle / 12 - increase * wiggle / 12)
                _set(5, pwm5 + reach_wiggle / 6 + increase * reach_wiggle / 6)
            else:
                _set(3, pwm3 + 2 * wiggle / 3 - increase * wiggle / 3)
                _set(4, pwm4 - 5 * wiggle / 6 + increase * wiggle / 6)
                _set(5, pwm5 + 5 * reach_wiggle / 6 - increase * reach_wiggle / 6)
        elif pos == 5:
            if direction:
                _set(3, pwm3 - wiggle / 3 + increase * wiggle / 3)
                _set(4, pwm4 - 5 * wiggle / 12 - increase * wiggle / 12)
                _set(5, pwm5 + reach_wiggle / 3 + increase * reach_wiggle / 6)
            else:
                _set(3, pwm3 + wiggle / 3 - increase * wiggle / 3)
                _set(4, pwm4 - 2 * wiggle / 3 + increase * wiggle / 6)
                _set(5, pwm5 + 2 * reach_wiggle / 3 - increase * reach_wiggle / 6)
        elif pos == 6:
            if direction:
                _set(3, pwm3 + increase * wiggle / 3)
                _set(4, pwm4 - wiggle / 2 - increase * wiggle / 6)
                _set(5, pwm5 + reach_wiggle / 2 + increase * reach_wiggle / 6)
            else:
                _set(3, pwm3 - increase * wiggle / 3)
                _set(4, pwm4 - wiggle / 2 + increase * wiggle / 12)
                _set(5, pwm5 + reach_wiggle / 2 - increase * reach_wiggle / 6)
        elif pos == 7:
            if direction:
                _set(3, pwm3 + wiggle / 3 + increase * wiggle / 3)
                _set(4, pwm4 - 4 * wiggle / 6 - increase * wiggle / 6)
                _set(5, pwm5 + 2 * reach_wiggle / 3 + increase * reach_wiggle / 6)
            else:
                _set(3, pwm3 - wiggle / 3 - increase * wiggle / 3)
                _set(4, pwm4 - 5 * wiggle / 12 + increase * wiggle / 12)
                _set(5, pwm5 + reach_wiggle / 3 - increase * reach_wiggle / 6)
        elif pos == 8:
            if direction:
                _set(3, pwm3 + 2 * wiggle / 3 + increase * wiggle / 3)
                _set(4, pwm4 - 5 * wiggle / 6 - increase * wiggle / 6)
                _set(5, pwm5 + 5 * reach_wiggle / 6 + increase * reach_wiggle / 6)
            else:
                _set(3, pwm3 - 2 * wiggle / 3 - increase * wiggle / 3)
                _set(4, pwm4 - wiggle / 3 + increase * wiggle / 12)
                _set(5, pwm5 + reach_wiggle / 6 - increase * reach_wiggle / 6)

    elif name == 'III':
        if pos == 1:
            if direction:
                _set(6, pwm6 + wiggle - increase * wiggle)
                _set(7, pwm7 - wiggle / 4 + 5 * increase * wiggle / 4)
                _set(8, pwm8 + increase * max_wiggle)
            else:
                _set(6, pwm6 - wiggle + increase * wiggle)
                _set(7, pwm7 - wiggle + 2 * increase * wiggle)
                _set(8, pwm8 + reach_wiggle + increase * (max_wiggle - reach_wiggle))
        elif pos == 2:
            if direction:
                _set(6, pwm6 - increase * wiggle)
                _set(7, pwm7 + wiggle - 2 * increase * wiggle)
                _set(8, pwm8 + max_wiggle - increase * (max_wiggle - reach_wiggle))
            else:
                _set(6, pwm6 + increase * wiggle)
                _set(7, pwm7 + wiggle - 5 * increase * wiggle / 4)
                _set(8, pwm8 + max_wiggle - increase * max_wiggle)
        elif pos == 3:
            if direction:
                _set(6, pwm6 - wiggle + increase * wiggle / 3)
                _set(7, pwm7 - wiggle + increase * wiggle / 6)
                _set(8, pwm8 + reach_wiggle - increase * reach_wiggle / 6)
            else:
                _set(6, pwm6 + wiggle - increase * wiggle / 3)
                _set(7, pwm7 - wiggle / 4 - increase * wiggle / 12)
                _set(8, pwm8 + increase * reach_wiggle / 6)
        elif pos == 4:
            if direction:
                _set(6, pwm6 - 2 * wiggle / 3 + increase * wiggle / 3)
                _set(7, pwm7 - 5 * wiggle / 6 + increase * wiggle / 6)
                _set(8, pwm8 + 5 * reach_wiggle / 6 - increase * reach_wiggle / 6)
            else:
                _set(6, pwm6 + 2 * wiggle / 3 - increase * wiggle / 3)
                _set(7, pwm7 - wiggle / 4 - increase * wiggle / 12)
                _set(8, pwm8 + reach_wiggle / 6 + increase * reach_wiggle / 6)
        elif pos == 5:
            if direction:
                _set(6, pwm6 - wiggle / 3 + increase * wiggle / 3)
                _set(7, pwm7 - 2 * wiggle / 3 + increase * wiggle / 6)
                _set(8, pwm8 + 2 * reach_wiggle / 3 - increase * reach_wiggle / 6)
            else:
                _set(6, pwm6 + wiggle / 3 - increase * wiggle / 3)
                _set(7, pwm7 - 5 * wiggle / 12 - increase * wiggle / 12)
                _set(8, pwm8 + reach_wiggle / 3 + increase * reach_wiggle / 6)
        elif pos == 6:
            if direction:
                _set(6, pwm6 + increase * wiggle / 3)
                _set(7, pwm7 - wiggle / 2 + increase * wiggle / 12)
                _set(8, pwm8 + reach_wiggle / 2 - increase * reach_wiggle / 6)
            else:
                _set(6, pwm6 - increase * wiggle / 3)
                _set(7, pwm7 - wiggle / 2 - increase * wiggle / 6)
                _set(8, pwm8 + reach_wiggle / 2 + increase * reach_wiggle / 6)
        elif pos == 7:
            if direction:
                _set(6, pwm6 + wiggle / 3 + increase * wiggle / 3)
                _set(7, pwm7 - 5 * wiggle / 12 + increase * wiggle / 12)
                _set(8, pwm8 + reach_wiggle / 3 - increase * reach_wiggle / 6)
            else:
                _set(6, pwm6 - wiggle / 3 - increase * wiggle / 3)
                _set(7, pwm7 - 2 * wiggle / 3 - increase * wiggle / 6)
                _set(8, pwm8 + 2 * reach_wiggle / 3 + increase * reach_wiggle / 6)
        elif pos == 8:
            if direction:
                _set(6, pwm6 + 2 * wiggle / 3 + increase * wiggle / 3)
                _set(7, pwm7 - 4 * wiggle / 12 + increase * wiggle / 12)
                _set(8, pwm8 + reach_wiggle / 6 - increase * reach_wiggle / 6)
            else:
                _set(6, pwm6 - 2 * wiggle / 3 - increase * wiggle / 3)
                _set(7, pwm7 - 5 * wiggle / 6 - increase * wiggle / 6)
                _set(8, pwm8 + 5 * reach_wiggle / 6 + increase * reach_wiggle / 6)

    elif name == 'IV':
        if pos == 1:
            if direction:
                _set(9,  pwm9  - wiggle + increase * wiggle)
                _set(10, pwm10 + wiggle - 2 * increase * wiggle)
                _set(11, pwm11 - reach_wiggle - increase * (max_wiggle - reach_wiggle))
            else:
                _set(9,  pwm9  + wiggle - increase * wiggle)
                _set(10, pwm10 + wiggle / 4 - 5 * increase * wiggle / 4)
                _set(11, pwm11 - increase * max_wiggle)
        elif pos == 2:
            if direction:
                _set(9,  pwm9  + increase * wiggle)
                _set(10, pwm10 - wiggle + 5 * increase * wiggle / 4)
                _set(11, pwm11 - max_wiggle + increase * max_wiggle)
            else:
                _set(9,  pwm9  - increase * wiggle)
                _set(10, pwm10 - wiggle + 2 * wiggle * increase)
                _set(11, pwm11 - max_wiggle + increase * (max_wiggle - reach_wiggle))
        elif pos == 3:
            if direction:
                _set(9,  pwm9  + wiggle - increase * wiggle / 3)
                _set(10, pwm10 + wiggle / 4 + increase * wiggle / 12)
                _set(11, pwm11 - increase * reach_wiggle / 6)
            else:
                _set(9,  pwm9  - wiggle + increase * wiggle / 3)
                _set(10, pwm10 + wiggle - increase * wiggle / 6)
                _set(11, pwm11 - reach_wiggle + increase * reach_wiggle / 6)
        elif pos == 4:
            if direction:
                _set(9,  pwm9  + 2 * wiggle / 3 - increase * wiggle / 3)
                _set(10, pwm10 + wiggle / 3 + increase * wiggle / 12)
                _set(11, pwm11 - reach_wiggle / 6 - increase * reach_wiggle / 6)
            else:
                _set(9,  pwm9  - 2 * wiggle / 3 + increase * wiggle / 3)
                _set(10, pwm10 + 5 * wiggle / 6 - increase * wiggle / 6)
                _set(11, pwm11 - 5 * reach_wiggle / 6 + increase * reach_wiggle / 6)
        elif pos == 5:
            if direction:
                _set(9,  pwm9  + wiggle / 3 - increase * wiggle / 3)
                _set(10, pwm10 + 5 * wiggle / 12 + increase * wiggle / 12)
                _set(11, pwm11 - reach_wiggle / 3 - increase * reach_wiggle / 6)
            else:
                _set(9,  pwm9  - wiggle / 3 + increase * wiggle / 3)
                _set(10, pwm10 + 2 * wiggle / 3 - increase * wiggle / 6)
                _set(11, pwm11 - 2 * reach_wiggle / 3 + increase * reach_wiggle / 6)
        elif pos == 6:
            if direction:
                _set(9,  pwm9  - increase * wiggle / 3)
                _set(10, pwm10 + wiggle / 2 + increase * wiggle / 6)
                _set(11, pwm11 - reach_wiggle / 2 - increase * reach_wiggle / 6)
            else:
                _set(9,  pwm9  + increase * wiggle / 3)
                _set(10, pwm10 + wiggle / 2 - increase * wiggle / 12)
                _set(11, pwm11 - reach_wiggle / 2 + increase * reach_wiggle / 6)
        elif pos == 7:
            if direction:
                _set(9,  pwm9  - wiggle / 3 - increase * wiggle / 3)
                _set(10, pwm10 + 4 * wiggle / 6 + increase * wiggle / 6)
                _set(11, pwm11 - 2 * reach_wiggle / 3 - increase * reach_wiggle / 6)
            else:
                _set(9,  pwm9  + wiggle / 3 + increase * wiggle / 3)
                _set(10, pwm10 + 5 * wiggle / 12 - increase * wiggle / 12)
                _set(11, pwm11 - reach_wiggle / 3 + increase * reach_wiggle / 6)
        elif pos == 8:
            if direction:
                _set(9,  pwm9  - 2 * wiggle / 3 - increase * wiggle / 3)
                _set(10, pwm10 + 5 * wiggle / 6 + increase * wiggle / 6)
                _set(11, pwm11 - 5 * reach_wiggle / 6 - increase * reach_wiggle / 6)
            else:
                _set(9,  pwm9  + 2 * wiggle / 3 + increase * wiggle / 3)
                _set(10, pwm10 + wiggle / 3 - increase * wiggle / 12)
                _set(11, pwm11 - reach_wiggle / 6 + increase * reach_wiggle / 6)

FORWARD_LEG_I_STRIDE_SCALE = 1.00


def dove_move_tripod(step, speed, command):
    step_I   = step
    step_II  = step + 2
    step_III = step + 4
    step_IV  = step + 6
    
    if step_II  > 8: step_II  -= 8
    if step_III > 8: step_III -= 8
    if step_IV  > 8: step_IV  -= 8

    if command in ('forward', 'backward'):
        sign      = 1 if command == 'forward' else -1
        yaw_boost = int(speed * cfg.YAW_COMPENSATION)
        left_speed  = speed - max(0, yaw_boost)
        right_speed = speed + min(0, yaw_boost)
        # Leg I starts the swing at phase 1 (user-observed left front).
        # Scale its full forward cycle consistently; preserve lift and phase.
        first_leg_speed = left_speed * (FORWARD_LEG_I_STRIDE_SCALE if command == 'forward' else 1.)

        for i in range(1, pixel + 1):
            leg_tripod('I',  step_I,  i, sign * first_leg_speed)
            leg_tripod('IV', step_IV, i, sign * right_speed)
            time.sleep(cfg.GAIT_DIAG_DELAY)
            leg_tripod('II',  step_II,  i, sign * left_speed)
            leg_tripod('III', step_III, i, sign * right_speed)
            if i < pixel:
                time.sleep(cfg.GAIT_SUB_STEP_DELAY)

    elif command in ('left', 'right'):
        # Equal opposing strides reduce the forward bias of the old 100:80 turn.
        # This is a gait displacement parameter, not a time or velocity setting.
        turn_stride = int(speed * turn_steady)
        if command == 'left':
            speed_I   = -turn_stride
            speed_II  = -turn_stride
            speed_III = turn_stride
            speed_IV  = turn_stride
        else:
            speed_I   = turn_stride
            speed_II  = turn_stride
            speed_III = -turn_stride
            speed_IV  = -turn_stride

        for i in range(1, pixel + 1):
            leg_tripod('I',   step_I,   i, speed_I)
            leg_tripod('IV',  step_IV,  i, speed_IV)
            time.sleep(cfg.GAIT_DIAG_DELAY)
            leg_tripod('II',  step_II,  i, speed_II)
            leg_tripod('III', step_III, i, speed_III)
            if i < pixel:
                time.sleep(cfg.GAIT_SUB_STEP_DELAY)

def robot_stand():
    _set(0, pwm0)
    _set(1, pwm1)
    _set(2, pwm2)
    time.sleep(0.15)
    _set(3, pwm3)
    _set(4, pwm4)
    _set(5, pwm5)
    time.sleep(0.15)
    _set(6, pwm6)
    _set(7, pwm7)
    _set(8, pwm8)
    time.sleep(0.15)
    _set(9, pwm9)
    _set(10, pwm10)
    _set(11, pwm11)
    time.sleep(0.15)
