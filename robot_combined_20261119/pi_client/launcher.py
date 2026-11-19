"""Explicitly armed launcher output; never used by default."""
from aiming import arm_check
import math


class Launcher:
    def __init__(self, board, cfg, allow_fire=True, report=None):
        if cfg['angle_channel'] != 13 or cfg['fire_channel'] != 14:
            raise ValueError('Expected angle channel 13 and fire channel 14')
        if cfg.get('angle_output_enabled') is not True:
            raise ValueError('Angle output is disabled')
        if allow_fire:
            arm_check(cfg)
        self.allow_fire = allow_fire
        self.board, self.cfg = board, cfg
        self.report = report
        self.last_readback = None
        self.active = False
        self.deadline = 0.
        self.pwm = None
        if self.allow_fire:
            self.stop()

    def write(self, channel, value):
        if channel not in (13, 14) or not isinstance(value, int):
            raise ValueError('Invalid launcher output')
        if channel == 13 and not 150 <= value <= 450:
            raise ValueError('Angle PWM outside 150..450')
        if channel == 14 and (not 150 <= value <= 492 or value not in
                             (self.cfg['fire_active_pwm'],self.cfg['fire_stop_pwm'])):
            raise ValueError('Channel 14 accepts only configured run/stop PWM')
        if channel == 14 and (not self.allow_fire or self.cfg.get('fire_output_enabled') is not True):
            raise PermissionError('Channel 14 is locked')
        if channel == 13 and self.cfg.get('angle_output_enabled') is not True:
            raise PermissionError('Channel 13 is disabled')
        self.board._write_channel(channel, [0, 0, value & 255, (value >> 8) & 15])
        # Verify registers only; this cannot confirm power or physical movement.
        from launcher_diagnostics import verify_write
        self.last_readback = verify_write(self.board, channel, value)
        if self.report:
            self.report(self.last_readback)
        if not self.last_readback['register_match'] or not self.last_readback['setup_50hz']:
            raise RuntimeError('Launcher PWM register/setup mismatch; physical output is not confirmed')

    def aim(self, pwm):
        if self.pwm != pwm:
            self.write(self.cfg['angle_channel'], pwm)
            self.pwm = pwm

    def fire(self, now):
        if not self.allow_fire or self.cfg.get('fire_output_enabled') is not True:
            raise PermissionError('Channel 14 is locked')
        if self.active:
            raise RuntimeError('A firing pulse is already active')
        if not math.isfinite(now):
            raise ValueError('Invalid firing timestamp')
        self.active = True
        self.deadline = now+self.cfg['fire_seconds']
        self.write(self.cfg['fire_channel'], self.cfg['fire_active_pwm'])

    def update(self, now, permitted):
        """Configured pulse without blocking reception of stop/person events.

        Channel 14: configured run PWM -> configured duration -> configured stop PWM.
        Returns the stop reason once; a stopped pulse never restarts itself.
        """
        if not self.active:
            return None
        if not math.isfinite(now) or not permitted:
            self.stop()
            return 'Interlock stopped pulse'
        if now >= self.deadline:
            self.stop()
            return 'Pulse complete'
        return None

    def stop(self):
        self.write(self.cfg['fire_channel'], self.cfg['fire_stop_pwm'])
        self.active = False

    def close(self):
        if self.allow_fire:
            self.stop()

