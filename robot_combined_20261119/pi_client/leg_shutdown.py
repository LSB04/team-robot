"""Release mapped leg PWM without moving to another pose or touching the launcher."""


def release_leg_pwm(board=None, bus=1, address=0x40):
    owned = board is None
    if owned:
        from pwm_driver import PCA9685
        board = PCA9685(bus,address)
    try:
        # No setup/frequency change and no intermediate PWM=0 command.
        board.off()
    finally:
        if owned:
            board.close()
