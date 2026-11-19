"""Set channel 13 raw PWM only. Channel 14 is never written. Stop robot clients first."""
import argparse
from aiming import load_config
from launcher import Launcher


def require_existing_setup(board):
    mode = board.bus.read_byte_data(board.address, 0)
    if (mode & 0x50 or board.bus.read_byte_data(board.address, 0xFE) != 121
            or board.bus.read_byte_data(board.address, 1) != 0x04):
        raise RuntimeError('PCA9685 is not in the expected 50 Hz setup. Run init_pose.py first; no outputs changed.')


def prepare_stationary_launcher(board):
    """Accept retained setup or initialize a power-cycled, inactive board. No leg pose."""
    try:
        require_existing_setup(board)
        return
    except RuntimeError:
        pass
    for ch in range(16):
        data = [board.bus.read_byte_data(board.address, 6+4*ch+i) for i in range(4)]
        if not (data[3] & 0x10) and (data[1] & 0x10 or data[:2] != data[2:]):
            raise RuntimeError('Unexpected live PWM with wrong clock setup; stop and power-cycle controller before retrying')
    board.setup()
    require_existing_setup(board)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pwm', type=int, required=True, help='Raw PWM; known horizontal value is 280')
    p.add_argument('--bus', type=int, default=1)
    p.add_argument('--address', type=lambda v:int(v,0), default=0x40)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if not 150 <= a.pwm <= 450:
        p.error('PWM must be 150..450 (software limit, not a measured mechanical limit)')
    cfg = load_config()
    if cfg.get('angle_output_enabled') is not True:
        p.error('Channel 13 is disabled')
    if a.dry_run:
        print(f'DRY RUN: channel 13 = {a.pwm}; no channel 14 output; no I2C opened')
        return
    from pwm_driver import PCA9685
    board = PCA9685(a.bus, a.address)
    try:
        require_existing_setup(board)
        angle = Launcher(board, cfg, allow_fire=False)
        angle.aim(a.pwm)
        angle.close()
        print(f'Channel 13 = {a.pwm}. PWM held after exit. Channel 14 untouched.')
    finally:
        board.close()


if __name__ == '__main__': main()
