"""Leg PWM driver adapted from robot_walk_test/walk_test.py."""
import time

CHANNEL_MAP = {i: (12 if i == 5 else i) for i in range(12)}
LEG_CHANNELS = tuple(CHANNEL_MAP.values())
PROTECTED_CHANNELS = (5, 13, 14, 15)

class PCA9685:
    def __init__(self, bus, address):
        try:
            from smbus2 import SMBus
        except ImportError:
            from smbus import SMBus
        self.bus, self.address = SMBus(bus), address

    def setup(self):
        old = self.bus.read_byte_data(self.address, 0)
        if old & 0x40:
            raise RuntimeError('External clock mode is not supported by this test.')
        # Reuse the existing 50 Hz setup without touching unused and launcher channels.
        if (not old & 0x10 and self.bus.read_byte_data(self.address, 0xFE) == 121
                and self.bus.read_byte_data(self.address, 1) == 0x04):
            return
        # Shared board: refuse to change frequency while unused and launcher channels are active.
        for ch in PROTECTED_CHANNELS:
            data = [self.bus.read_byte_data(self.address, 6+4*ch+i) for i in range(4)]
            if not (data[3] & 0x10) and (data[1] & 0x10 or data[0:2] != data[2:4]):
                raise RuntimeError(f'Channel {ch} is active. Stop other robot programs and disable its output first.')
        self.off()
        self.bus.write_byte_data(self.address, 0, (old & 0x0F) | 0x30)
        self.bus.write_byte_data(self.address, 0xFE, 121)  # ~50 Hz, 25 MHz oscillator
        self.bus.write_byte_data(self.address, 1, 0x04)
        self.bus.write_byte_data(self.address, 0, (old & 0x0F) | 0x20)
        time.sleep(.005)
        self.bus.write_byte_data(self.address, 0, (old & 0x0F) | 0xA0)

    def write(self, ch, value):
        if ch not in LEG_CHANNELS or not 150 <= value <= 450:
            raise ValueError('Leg output outside allowed channel or pulse range')
        self._write_channel(ch, [0, 0, value & 255, (value >> 8) & 15])

    def _write_channel(self, ch, data):
        # Explicit register addresses also work when Adafruit leaves MODE1.AI clear.
        for offset, value in enumerate(data):
            self.bus.write_byte_data(self.address, 6+4*ch+offset, value)

    def off(self):
        errors = []
        for ch in LEG_CHANNELS:
            try:
                # Assert FULL_OFF directly. Clearing OFF_L first temporarily
                # changes a live 0x01xx pulse to 0x0100 (256 ticks), which can
                # command a different servo position before the output stops.
                # FULL_OFF overrides the count and FULL_ON (NXP PCA9685 7.3.3).
                # Leave the other count registers untouched.
                self.bus.write_byte_data(self.address, 9+4*ch, 0x10)
            except OSError as exc:
                errors.append(str(exc))
        if errors:
            raise OSError('PWM release failed; cut servo power if needed: '+errors[0])

    def close(self):
        self.bus.close()
