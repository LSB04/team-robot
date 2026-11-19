"""Pi sonar scan client. Default simulation; --execute enables real sensors and legs."""
import argparse
import json
import math
from pathlib import Path
import queue
import socket
import threading
import time
from datetime import datetime
from types import SimpleNamespace
from core import Scan, valid_distance
from pwm_driver import CHANNEL_MAP

DEFAULT_AMPLITUDE = 0.79
DEFAULT_SERVER = '10.153.209.28'


class InterruptMotion(Exception):
    pass


class Link:
    def __init__(self, conn):
        self.conn = conn
        self.last = time.monotonic()
        self.stopped = threading.Event()
        self.stopped.set()
        self.exited = threading.Event()
        self.commands = queue.Queue(maxsize=16)
        self.thread = threading.Thread(target=self.receive, daemon=True)
        self.thread.start()

    def receive(self):
        buffer = b''
        try:
            while not self.exited.is_set():
                try:
                    data = self.conn.recv(4096)
                except socket.timeout:
                    continue
                if not data:
                    break
                buffer += data
                if len(buffer) > 8192:
                    raise ValueError('Oversized command')
                while b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)
                    command = json.loads(line)['command']
                    if command not in ('start', 'stop', 'exit', 'heartbeat'):
                        raise ValueError('Unknown command')
                    self.last = time.monotonic()
                    if command in ('stop', 'exit'):
                        self.stopped.set()
                    if command == 'exit':
                        self.exited.set()
                        return
                    if command != 'heartbeat':
                        self.commands.put_nowait(command)
        except (OSError, ValueError, KeyError, TypeError, queue.Full):
            pass
        finally:
            self.stopped.set()
            self.exited.set()

    def fresh(self):
        return not self.exited.is_set() and time.monotonic()-self.last < 1.


class Motion:
    def __init__(self, args, guard):
        import gait
        from pwm_driver import PCA9685
        self.gait, self.guard, self.args = gait, guard, args
        self.board = PCA9685(args.bus, args.address)
        self.old_set, self.old_time = gait._set, gait.time
        try:
            self.board.setup()
        except BaseException:
            self.board.close()
            raise
        self.neutral = [getattr(gait, f'pwm{i}') for i in range(12)]
        gait._set = self.write
        gait.time = SimpleNamespace(sleep=self.sleep)
        gait.cfg.GAIT_DIAG_DELAY = .008*args.delay_scale
        gait.cfg.GAIT_SUB_STEP_DELAY = .012*args.delay_scale
        self.phase, self.previous = 1, None
        self.stood = False

    def write(self, ch, raw):
        self.guard()
        if ch not in range(12):
            raise ValueError('Only leg channels 0..11 allowed')
        value = int(self.neutral[ch]+(raw-self.neutral[ch])*self.args.amplitude)
        self.board.write(CHANNEL_MAP[ch], max(150, min(450, value)))

    def sleep(self, seconds):
        end = time.monotonic()+seconds
        while time.monotonic() < end:
            self.guard()
            time.sleep(min(.005, max(0, end-time.monotonic())))

    def step(self, command, turn_speed=100):
        if command not in ('forward', 'left', 'right'):
            return
        if not self.stood:
            self.gait.robot_stand()
            self.sleep(.5)
            self.stood = True
        if command != self.previous:
            self.phase = 1
        self.previous = command
        self.gait.dove_move_tripod(self.phase, 85 if command == 'forward' else turn_speed, command)
        self.phase = self.phase % 8+1
        self.sleep(.06*self.args.delay_scale)

    def close(self):
        self.gait._set, self.gait.time = self.old_set, self.old_time
        # Hold the last leg positions; no automatic lowering or PWM release.
        self.board.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--server', default=DEFAULT_SERVER)
    p.add_argument('--port', type=int, default=5582)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--trigger', type=int, default=11)
    p.add_argument('--echo', type=int, default=8)
    p.add_argument('--gyro-sign', type=int, choices=[-1, 1], default=-1)
    p.add_argument('--recalibrate-gyro', action='store_true')
    p.add_argument('--amplitude', type=float, default=DEFAULT_AMPLITUDE)
    p.add_argument('--delay-scale', type=float, default=1.)
    p.add_argument('--bus', type=int, default=1)
    p.add_argument('--address', type=lambda s: int(s, 0), default=0x40)
    a = p.parse_args(argv)
    if not math.isfinite(a.amplitude) or not .1 <= a.amplitude <= 1 or not math.isfinite(a.delay_scale) or not 1 <= a.delay_scale <= 5:
        p.error('amplitude 0.1..1 and delay-scale 1..5 required')
    machine = Scan()
    sensors = motion = link = None
    action = 'stop'
    sim_yaw, sim_front, previous_time = 0., 80., time.monotonic()
    log_dir = Path(__file__).resolve().parent/'logs'
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir/(datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.jsonl')
    with socket.create_connection((a.server, a.port), timeout=5) as conn, log_path.open('x', encoding='utf-8') as log:
        conn.settimeout(.2)
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        link = Link(conn)
        def report(event, **values):
            record = dict(event=event, mode='hardware' if a.execute else 'simulation',
                          state=machine.state, reason=machine.reason, results=machine.results,
                          monotonic=time.monotonic(), **values)
            line = json.dumps(record, allow_nan=False)+'\n'
            log.write(line); log.flush()
            if event != 'SONAR_SAMPLES':
                print(line.strip(), flush=True)
            conn.sendall(line.encode())
        def guard():
            if not link.fresh() or link.stopped.is_set():
                raise InterruptMotion('Heartbeat lost or operator stop')
            distance, yaw, sample_id = sensors.snapshot()
            if yaw is None:
                raise InterruptMotion('Gyro missing or stale')
            if action == 'forward' and not valid_distance(distance):
                machine.sonar_lost(time.monotonic(), sample_id)
                raise InterruptMotion('Sonar missing or stale during gait')
            if action == 'forward' and distance <= machine.threshold:
                raise InterruptMotion('Obstacle at or below distance threshold')
            if action in ('left', 'right'):
                if machine.turn_finished(yaw):
                    raise InterruptMotion('Turn angle reached')
                if time.monotonic() > machine.deadline:
                    raise InterruptMotion('Turn deadline reached')
        try:
            if a.execute:
                from sensors import Sensors
                report('GYRO INIT: reuse saved bias; keep still if first run or recalibrating')
                sensors = Sensors(a.trigger, a.echo, a.gyro_sign, recalibrate=a.recalibrate_gyro)
                report('SENSOR_INFO', trigger_bcm=a.trigger, echo_bcm=a.echo,
                       diagnostics=sensors.diagnostic_info())
            report('READY: continuous walking; stationary checks after turns only; press Enter on PC to start')
            last_report, last_state = 0., None
            while not link.exited.is_set():
                now = time.monotonic()
                dt, previous_time = min(.1, now-previous_time), now
                if sensors:
                    distance, yaw, sample_id = sensors.snapshot()
                else:
                    sim_yaw += (120 if action == 'left' else -120 if action == 'right' else 0)*dt
                    if action == 'forward':
                        sim_front = max(25., sim_front-30*dt)
                    distance = 140. if sim_yaw > 45 else 80. if sim_yaw < -45 else sim_front
                    yaw, sample_id = sim_yaw, int(now*10)
                if not link.fresh():
                    link.stopped.set()
                    machine.stop('Heartbeat timeout; explicit restart required')
                while True:
                    try:
                        command = link.commands.get_nowait()
                    except queue.Empty:
                        break
                    if command == 'stop':
                        machine.stop()
                    elif command == 'start' and link.fresh():
                        if machine.start(now, yaw):
                            link.stopped.clear()
                            report('START ACCEPTED')
                        else:
                            report('START REJECTED: check state and fresh gyro')
                if link.stopped.is_set():
                    if machine.state not in ('IDLE', 'DONE', 'FAULT'):
                        machine.stop()
                    action = 'stop'
                else:
                    action = machine.step(now, distance, yaw, sample_id)
                if sensors:
                    sensors.set_context(machine.state, action, motion.phase if motion else None)
                    batch = sensors.drain_diagnostics()
                    if batch:
                        report('SONAR_SAMPLES', samples=batch)
                if machine.state != last_state or now-last_report >= .5:
                    report('STATUS', action=action, sonar_raw_cm=distance, yaw_deg=yaw,
                           sonar_dropouts=machine.sonar_dropouts,
                           sonar_diagnostic=sensors.diagnostic() if sensors else None)
                    last_report, last_state = now, machine.state
                if a.execute and action != 'stop':
                    try:
                        if motion is None:
                            guard()
                            motion = Motion(a, guard)
                        sensors.set_context(machine.state, action, motion.phase)
                        motion.step(action, machine.turn_speed(yaw) if action in ('left', 'right') else 100)
                    except InterruptMotion as exc:
                        previous_action = action
                        action = 'stop'
                        sensors.set_context(machine.state, action, motion.phase if motion else None)
                        report('MOTION_INTERRUPTED', interrupted_action=previous_action,
                               interrupt_reason=str(exc), sonar_diagnostic=sensors.diagnostic())
                time.sleep(.02)
        except KeyboardInterrupt:
            pass
        except OSError as exc:
            print(f'ERROR: {type(exc).__name__}: {exc}', flush=True)
            raise
        finally:
            link.stopped.set(); link.exited.set()
            if motion:
                motion.close()
            if sensors:
                sensors.close()
                try:
                    while True:
                        batch = sensors.drain_diagnostics()
                        if not batch:
                            break
                        report('SONAR_SAMPLES', samples=batch)
                except OSError:
                    pass
            link.thread.join(.4)
            print('Stopped. Last leg PWM is held. No automatic restart.', flush=True)


if __name__ == '__main__':
    main()
