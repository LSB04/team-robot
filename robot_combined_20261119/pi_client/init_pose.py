"""Set the mapped leg home pose, hold briefly, release leg PWM, and exit."""
import argparse
import time
import gait
from pwm_driver import PCA9685, CHANNEL_MAP

HOME_PWM = tuple(getattr(gait, f'pwm{i}') for i in range(12))


def set_home(board, sleep=time.sleep):
    for logical, value in enumerate(HOME_PWM):
        physical = CHANNEL_MAP[logical]
        board.write(physical, value)
        print(f'Logical {logical} -> PWM channel {physical}: {value}', flush=True)
        if logical % 3 == 2:
            sleep(.15)


class DryRun:
    def write(self, channel, value):
        pass


def move_and_release(board, hold=1.0, sleep=time.sleep):
    try:
        board.setup()
        set_home(board, sleep=sleep)
        print(f'Holding home pose for {hold:g} seconds.', flush=True)
        sleep(hold)
    finally:
        try:
            print('Releasing leg outputs directly (FULL_OFF; no zero-count pose).', flush=True)
            board.off()
            print('LEG PWM OFF. Servo holding force released.', flush=True)
        finally:
            board.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--hold-seconds', type=float, default=1.0)
    args = parser.parse_args()
    if not 0 <= args.hold_seconds <= 60:
        parser.error('--hold-seconds must be between 0 and 60')
    try:
        if args.dry_run:
            print('DRY RUN: no I2C access or servo output.')
            set_home(DryRun(), sleep=lambda _: None)
        else:
            print('Setting mapped leg home pose. Stop other motor programs first.', flush=True)
            board = PCA9685(1, 0x40)
            move_and_release(board, hold=args.hold_seconds)
            print('Finished. Python process exiting.', flush=True)
    except KeyboardInterrupt:
        print('Interrupted. Cleanup attempted; Python process exiting.', flush=True)


if __name__ == '__main__':
    main()
