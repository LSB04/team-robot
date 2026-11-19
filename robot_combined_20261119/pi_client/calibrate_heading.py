"""Hand-rotate a level robot to calibrate AK8963. Never opens the PWM board."""
import argparse
import json
import time
from pathlib import Path
from heading_fusion import FusionIMU,field_calibration


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--imu-address',type=lambda x:int(x,0),default=0x68,choices=(0x68,0x69))
    parser.add_argument('--imu-bus',type=int,default=1)
    args=parser.parse_args()
    path=Path(__file__).with_name(f'mag_MPU9250_{args.imu_bus}_{args.imu_address:02x}.json')
    input('Stop all robot programs and disconnect servo power. Keep chassis level. Press Enter, then hand-rotate one full circle in 20 seconds: ')
    imu=FusionIMU(args.imu_address,args.imu_bus)
    try:
        fields=[];end=time.monotonic()+20
        while time.monotonic()<end:
            time.sleep(.02)
            _,field,accel=imu.sample()
            if abs(accel['x'])>2.5 or abs(accel['y'])>2.5 or not 8<accel['z']<11:
                continue
            if field is not None:fields.append(field)
        calibration=field_calibration(fields)
        data=dict(version=1,imu_address=args.imu_address,bus=args.imu_bus,samples=len(fields),**calibration)
        temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps(data,indent=2),encoding='utf-8')
        temporary.replace(path)
        print('Saved:',path,flush=True)
    finally:
        imu.close()


if __name__=='__main__':main()
