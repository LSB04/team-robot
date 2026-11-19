"""Read PCA9685 registers only. Cannot verify servo power, OE, motion or launch."""
import argparse
import json


def read_status(board):
    read = lambda reg: board.bus.read_byte_data(board.address, reg)
    mode1, mode2, prescale = read(0), read(1), read(0xFE)
    channels = {}
    for ch in (13, 14):
        data = [read(6+4*ch+i) for i in range(4)]
        on = data[0] | ((data[1] & 15) << 8)
        off = data[2] | ((data[3] & 15) << 8)
        channels[str(ch)] = dict(on=on, off=off, full_on=bool(data[1]&16),
                                full_off=bool(data[3]&16), pulse_ticks=(off-on)%4096,
                                registers=data)
    return dict(address=hex(board.address), mode1=mode1, mode2=mode2, prescale=prescale,
                sleeping=bool(mode1&16), external_clock=bool(mode1&64),
                setup_50hz=not bool(mode1&0x50) and mode2==4 and prescale==121,
                channels=channels, motion_verified=False,
                note='Register readback only. Servo supply, wiring, OE pin and physical movement are not measured.')


def verify_write(board, channel, pwm):
    status = read_status(board)
    actual = status['channels'][str(channel)]
    status.update(command_channel=channel, command_pwm=pwm,
                  register_match=actual['on']==0 and actual['off']==pwm
                  and not actual['full_on'] and not actual['full_off'])
    return status


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bus',type=int,default=1)
    p.add_argument('--address',type=lambda s:int(s,0),default=0x40)
    a=p.parse_args()
    from pwm_driver import PCA9685
    board=PCA9685(a.bus,a.address)
    try:
        print(json.dumps(read_status(board),indent=2),flush=True)
    finally:
        board.close()


if __name__=='__main__':main()
