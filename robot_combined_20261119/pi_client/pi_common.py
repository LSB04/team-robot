"""Navigation, fresh stationary target recheck, and one-shot launcher control."""
import argparse
from pathlib import Path
from copy import deepcopy
from fire_handoff import HandoffCapture, save_snapshot, load_snapshot, ManualRecheck, gravity_unit, claim_shot
import math
import os
import queue
import socket
import time
from aiming import load_config, arm_check
from core import valid_distance, valid_yaw
from mission import Mission, iou
from combined_flow import switch_to_launcher
from legacy_motion import Motion, InterruptMotion, DEFAULT_AMPLITUDE
from transport import PiLink, Sender
from video import CameraSender
from navigation import NAVIGATION_VERSION
from forward_control import ForwardHeading, FORWARD_VERSION, DEADBAND_DEG, MAX_TRIM
from fire_review import VERSION as FIRE_REVIEW_VERSION
from launcher_review import VERSION as LAUNCHER_REVIEW_VERSION
from fire_navigation import VERSION as FIRE_NAVIGATION_VERSION


def main(argv=None, profile=1, combined=False):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--server', default='192.168.137.1', help='Laptop hotspot IPv4 (default: 192.168.137.1); not the Pi IP')
    p.add_argument('--port', type=int, default=5602 if combined else 5582 if profile == 1 else 5592)
    p.add_argument('--video-port', type=int, default=5603 if combined else 5583 if profile == 1 else 5593)
    p.add_argument('--camera-width', type=int, default=1280)
    p.add_argument('--camera-height', type=int, default=960)
    p.add_argument('--jpeg-quality', type=int, default=90)
    p.add_argument('--camera-view', choices=('balanced', 'full', 'legacy'), default='balanced')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--arm', action='store_true')
    p.add_argument('--auto-handoff', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--save-handoff', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--handoff-file', help=argparse.SUPPRESS)
    p.add_argument('--fresh-target', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--config')
    p.add_argument('--trigger', type=int, default=11)
    p.add_argument('--echo', type=int, default=8)
    p.add_argument('--gyro-sign', type=int, choices=(-1, 1), default=1)
    p.add_argument('--recalibrate-gyro', action='store_true', help='Stationary zero is always refreshed in HR22 fusion mode')
    p.add_argument('--mag-calibration', help='Saved AK8963 calibration JSON; default next to client')
    p.add_argument('--imu-address', type=lambda s: int(s, 0), choices=(0x68, 0x69), default=0x68)
    p.add_argument('--imu-bus', type=int, default=1)
    p.add_argument('--gyro-axis', choices=('x', 'y', 'z'), default='z')
    p.add_argument('--bus', type=int, default=1)
    p.add_argument('--address', type=lambda s: int(s, 0), default=0x40)
    p.add_argument('--amplitude', type=float, default=DEFAULT_AMPLITUDE)
    p.add_argument('--delay-scale', type=float, default=1.)
    a = p.parse_args(argv)
    if (a.camera_width*3 != a.camera_height*4
            or any(v % 2 or not 320 <= v <= 1920 for v in (a.camera_width,a.camera_height))
            or not 50 <= a.jpeg_quality <= 95):
        p.error('Use even 4:3 camera dimensions (320..1920) and JPEG quality 50..95')
    cfg = load_config(a.config)
    base_cfg = deepcopy(cfg)
    if combined:
        if profile != 1 or a.handoff_file or a.fresh_target or a.auto_handoff:
            p.error('Combined mode starts with navigation and automatically captures a fresh target')
        a.save_handoff = True
    if a.auto_handoff:
        p.error('--auto-handoff is disabled. Use client1 --save-handoff, then client2 --handoff-file latest')
    if a.save_handoff and profile != 1:
        p.error('--save-handoff is only for client1')
    if a.handoff_file and profile != 2:
        p.error('--handoff-file is only for client2')
    if a.fresh_target and (profile != 2 or a.handoff_file):
        p.error('--fresh-target requires client2 and cannot be combined with --handoff-file')
    loaded_snapshot = None
    snapshot_path = None
    if a.handoff_file:
        snapshot_path = (Path(__file__).with_name('handoffs')/'latest.json'
                         if a.handoff_file == 'latest' else Path(a.handoff_file))
        try:
            loaded_snapshot = load_snapshot(snapshot_path,cfg)
            if a.execute and loaded_snapshot.get('hardware') is not True:
                raise ValueError('Simulation snapshots cannot be used for hardware firing')
        except (OSError,ValueError,KeyError,TypeError) as exc:
            p.error('Cannot load handoff: '+str(exc))
    cfg['mission_profile'] = profile
    if profile == 1:
        cfg['angle_output_enabled'] = False
        cfg['fire_output_enabled'] = False
        if a.arm and not combined:
            p.error('Profile 1 is detection/navigation only; use client2 for launcher output')
    if a.arm:
        if not a.execute:
            p.error('--arm requires --execute')
        try:
            arm_check(base_cfg if combined else cfg)
        except ValueError as exc:
            p.error(str(exc))
    if not math.isfinite(a.amplitude) or not .1 <= a.amplitude <= 1 or not math.isfinite(a.delay_scale) or not 1 <= a.delay_scale <= 5:
        p.error('amplitude 0.1..1 and delay-scale 1..5 required')
    os.environ.setdefault('GPIOZERO_PIN_FACTORY', 'lgpio')
    handoff = HandoffCapture()
    mission = Mission(cfg)
    if loaded_snapshot:
        mission.manual_recheck = ManualRecheck(loaded_snapshot,cfg)
    saved_handoff = None
    recheck_generation = 0
    sensors = motion = launcher = camera = sender = link = angle_board = None
    action, mode = 'stop', 'hardware' if a.execute else 'simulation'
    align_origin, align_limit = None, None
    sim_yaw, sim_distance, previous = 0., 80., time.monotonic()
    firing_box = None
    fire_command_confirmed = False
    fire_stop_reason = None
    leveling_now = level_for_fire = False
    pending_level = None
    observed_scan_state = None
    previous_heading_error = None
    heading_paused = start_requested = False
    forward_heading = ForwardHeading()
    with socket.create_connection((a.server, a.port), timeout=5) as conn:
        conn.settimeout(.2)
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        link, sender = PiLink(conn), Sender(conn)

        def report(event, **extra):
            sender.put(dict(event=event, mode=mode, pi_monotonic=time.monotonic(),
                            **dict(dict(state=mission.state, mission_profile=profile), **extra)))

        def observe_range(now, distance, seq):
            if profile == 1 and sensors and mission.active:
                history = sensors.range_history(mission.fire_consistency.last_seq)
                mission.observe_fire_sonar(now, distance, seq, history)

        def guard():
            now = time.monotonic()
            if link.stopped.is_set() or not link.fresh():
                raise InterruptMotion('Operator stop or heartbeat timeout')
            if not mission.fresh(link.vision, now) or mission.people(link.vision):
                raise InterruptMotion('Vision stale or person candidate')
            distance, yaw, seq = sensors.snapshot()
            observe_range(now, distance, seq)
            if not valid_yaw(yaw):
                raise InterruptMotion('Gyro missing or stale')
            if not level_for_fire and mission.fresh(link.vision, now) and mission.fire_candidates(link.vision) and (leveling_now or action != 'stop') and mission.state not in ('FIRE_ALIGN', 'FIRE_APPROACH', 'FIRE_POSITION_READY'):
                if profile == 1:
                    # Keep a candidate seen inside a gait guard even if the next frame loses it.
                    mission.fire_review.update(link.vision, now, iou)
                raise InterruptMotion('Fire priority; navigation leveling/gait stopped')
            if mission.state in ('FIRE_ALIGN', 'FIRE_APPROACH') or level_for_fire:
                mission.refresh_fire_guard(link.vision, now)
            if (mission.state in ('FIRE_ALIGN', 'FIRE_APPROACH') or level_for_fire) and not mission.fire_motion_visible(link.vision):
                if profile == 1:
                    mission.fire_review.pause(now)
                    mission.fire_range_review.reset()
                raise InterruptMotion('Fire target lost; stop approach/alignment')
            if leveling_now:
                return
            if profile == 1 and mission.state in ('FIRE_ALIGN','FIRE_APPROACH'):
                if mission.lock_fire_arrival(now):
                    raise InterruptMotion('Arrival locked at 200cm or nearer; stop before final alignment')
                if action == 'forward' and mission.fire_arrival_locked:
                    raise InterruptMotion('Arrival locked; forward disabled')
            confidence = mission.fire_range_diagnostic(now) if profile == 1 else None
            if action == 'forward':
                nonblocking_fire = profile == 1 and mission.state == 'FIRE_APPROACH'
                if not valid_distance(distance) and not nonblocking_fire:
                    if mission.state == 'FORWARD':
                        # Navigation-only exception. Fire approach still requires range.
                        return
                    raise InterruptMotion('Sonar missing; stop for fresh readings')
                if valid_distance(distance) and distance <= cfg['threshold_cm']:
                    raise InterruptMotion('Obstacle at or below stop threshold')
                if mission.state == 'FIRE_APPROACH':
                    if profile == 1:
                        reason = mission.fire_approach_range_stop(now, distance)
                        if reason:
                            raise InterruptMotion(reason)
                    elif distance <= cfg.get('fire_target_cm', 200.):
                        raise InterruptMotion('Fire approach target reached; hold position')
                    if profile == 1 and mission.fire_approach_off_center(link.vision):
                        raise InterruptMotion('Fire off center; realign before approach')
            if action in ('left', 'right'):
                if align_origin is not None:
                    if profile == 1 and mission.fire_alignment_complete(link.vision, action):
                        raise InterruptMotion('Fire center reached; stop alignment')
                    if abs(yaw-align_origin) >= align_limit:
                        raise InterruptMotion('Alignment increment complete')
                elif mission.scan.turn_finished(yaw):
                    raise InterruptMotion('Scan heading target reached')

        try:
            if a.execute:
                from sensors import Sensors
                report('INITIALIZING', reason='Gyro bias load/measurement; keep still')
                sensors = Sensors(a.trigger, a.echo, a.gyro_sign, a.recalibrate_gyro,
                                  a.imu_address, a.imu_bus, a.gyro_axis, a.mag_calibration)
                report('SENSOR_INFO', diagnostics=sensors.diagnostic_info())
            camera = CameraSender(a.server, a.video_port, a.execute,
                                  size=(a.camera_width,a.camera_height), quality=a.jpeg_quality,
                                  view=a.camera_view, focal=cfg['focal_at_480_px'])
            camera_info_sent = False
            if loaded_snapshot:
                report('HANDOFF_LOADED', snapshot=loaded_snapshot, source_path=str(snapshot_path),
                       reason='Waiting for operator start and NEW measurements')
            report('READY', combined=combined, combined_phase='NAVIGATION' if combined else None, save_handoff=a.save_handoff, handoff_loaded=bool(loaded_snapshot), armed=a.arm, reason='Waiting for PC start and fresh vision', navigation_revision=NAVIGATION_VERSION,
                   fire_navigation_revision=FIRE_NAVIGATION_VERSION if profile==1 else None,
                   target_source=('saved_recheck' if loaded_snapshot else 'live_camera_sonar') if profile==2 else 'navigation',
                   launcher_revision='server2_live_v1',
                   launcher_operating_mode=cfg.get('launcher_operating_mode', 'calibrated'),
                   angle_pwm_mode=cfg.get('angle_pwm_mode', 'measured'),
                   angle_horizontal_pwm=cfg.get('angle_horizontal_pwm'),
                   angle_offset_deg=cfg.get('angle_offset_deg',0.),
                   fire_active_pwm=cfg['fire_active_pwm'],fire_stop_pwm=cfg['fire_stop_pwm'],
                   fire_seconds=cfg['fire_seconds'],
                   camera_size=[a.camera_width,a.camera_height],camera_view=a.camera_view,
                   jpeg_quality=a.jpeg_quality,
                   fire_review_revision=FIRE_REVIEW_VERSION if profile == 1 else None,
                   launcher_review_revision=LAUNCHER_REVIEW_VERSION if profile == 2 else None,
                   fire_candidate_conf=cfg.get('launcher_candidate_conf',.37) if profile==2 else cfg.get('fire_candidate_conf',.35),
                   fire_confirmation_conf=cfg.get('launcher_review_peak_conf',.37) if profile==2 else cfg.get('fire_review_peak_conf',.45),
                   firing_conf=cfg['fire_conf'] if profile==2 else None,
                   align_reference_px=cfg.get('launcher_align_tolerance_px_at_640',cfg['align_tolerance_px_at_640']) if profile==2 else cfg['align_tolerance_px_at_640'],
                   obstacle_stop_cm=cfg['threshold_cm'],fire_target_cm=cfg['fire_target_cm'],motion_amplitude=a.amplitude,
                   forward_control_revision=FORWARD_VERSION,
                   forward_deadband_deg=DEADBAND_DEG,forward_max_stride_reduction=MAX_TRIM)
            print('READY. Start from laptop. Ctrl+C stops. Launcher armed:', a.arm, flush=True)
            print('Navigation:',NAVIGATION_VERSION,flush=True)
            print('Motion amplitude:',a.amplitude,flush=True)
            last_report, last_state = 0., None
            while not link.exited.is_set() and not link.receiver.done.is_set() and not sender.done.is_set():
                now = time.monotonic()
                if camera.info is not None and not camera_info_sent:
                    report('CAMERA_INFO', camera=camera.info)
                    camera_info_sent = True
                dt, previous = min(.1, now-previous), now
                if sensors:
                    distance, yaw, seq = sensors.snapshot()
                else:
                    sim_yaw += (70 if action == 'left' else -70 if action == 'right' else 0)*dt
                    if action == 'forward':
                        sim_distance = max(25., sim_distance-15*dt)
                    distance = 140. if sim_yaw > 45 else 80. if sim_yaw < -45 else sim_distance
                    yaw, seq = sim_yaw, int(now*10)
                heading = sensors.heading_diagnostic() if sensors else None
                heading_error = (heading or {}).get('error')
                if heading_error != previous_heading_error:
                    report('HEADING_ERROR' if heading_error else 'HEADING_RECOVERED',
                           reason=heading_error or 'Fresh stationary heading recovered; navigation conditions will be rechecked',
                           heading_diagnostic=heading)
                    previous_heading_error = heading_error
                if mission.active and not valid_yaw(yaw) and not heading_paused:
                    if profile == 1:
                        heading_paused = True
                        mission.scan.stop('Heading unavailable; automatic recovery pending')
                        mission.aim_at = None
                    else:
                        mission.stop('Heading unavailable; explicit start required in launcher mode')
                        link.stopped.set()
                    pending_level = None
                    report('HEADING_STOP', reason='Heading unavailable; motion stopped', heading_diagnostic=heading)
                if not link.fresh():
                    link.stopped.set()
                    start_requested = False
                    mission.stop('Heartbeat timeout; explicit start required')
                if camera.error:
                    link.stopped.set()
                    start_requested = False
                    mission.stop('Camera failed: '+camera.error)
                while True:
                    try:
                        command = link.commands.get_nowait()
                    except queue.Empty:
                        break
                    if command == 'stop':
                        start_requested = heading_paused = False
                        mission.stop()
                    elif command == 'start':
                        if saved_handoff is not None and profile == 1:
                            report('START_REJECTED', reason='Position saved; exit client1 and rewire with power off before client2')
                            continue
                        rejection = ('Heartbeat not fresh' if not link.fresh() else
                                     'Camera failed: '+camera.error if camera.error else
                                     'Heading unavailable: '+(heading_error or 'sample missing or stale') if not valid_yaw(yaw) else
                                     'Waiting for fresh camera inference' if not mission.fresh(link.vision,now) else None)
                        if rejection:
                            if profile == 1 and link.fresh() and not camera.error:
                                start_requested = True
                                link.stopped.clear()
                                mission.stop(rejection)
                                report('START_WAIT', reason=rejection+'; start queued until sensors and vision recover', heading_diagnostic=heading)
                            else:
                                start_requested = False
                                mission.stop(rejection)
                                link.stopped.set()
                                report('START_REJECTED', reason=rejection, heading_diagnostic=heading)
                        elif mission.start(now, yaw):
                            start_requested = heading_paused = False
                            link.stopped.clear()
                            pending_level = 'START'
                            observed_scan_state = None
                            report('START_ACCEPTED')
                if link.stopped.is_set():
                    start_requested = heading_paused = False
                    mission.stop(mission.reason or 'Operator stop')
                if (start_requested and not link.stopped.is_set() and link.fresh() and not camera.error
                        and valid_yaw(yaw) and mission.fresh(link.vision,now)):
                    if mission.start(now,yaw):
                        start_requested = heading_paused = False
                        pending_level = 'START'
                        observed_scan_state = None
                        report('START_ACCEPTED',reason='Queued start accepted after sensor recovery')
                if heading_paused and mission.active and valid_yaw(yaw):
                    heading_paused = False
                    pending_level = 'START'
                    observed_scan_state = None
                    report('NAVIGATION_RESUMED',reason='Heading recovered; rechecking vision and front distance')
                mission.current_acceleration = sensors.acceleration() if sensors else dict(x=0.,y=0.,z=9.81)
                # An active pulse is never a blocking sleep. Recheck every loop.
                if launcher and launcher.active:
                    vision = link.vision
                    safe = mission.firing_permitted(now, vision, distance, yaw, firing_box, link.fresh())
                    stop_reason = launcher.update(now, safe)
                    if stop_reason:
                        fire_stop_reason = stop_reason
                        report('FIRE_STOPPED', reason=stop_reason)
                observe_range(now, distance, seq)
                result = mission.step(now, distance, yaw, seq, link.vision, a.arm)
                if not mission.active and not valid_yaw(yaw):
                    result.update(state='GYRO_WAIT',reason=heading_error or 'Heading sample missing or stale')
                if profile == 2:
                    result['target_source'] = 'saved_recheck' if loaded_snapshot else 'live_camera_sonar'
                result['navigation_enabled'] = bool(mission.active or start_requested)
                result['heading_auto_resume'] = bool(profile == 1 and (heading_paused or start_requested))
                if saved_handoff is not None and profile == 1:
                    result.update(state='HANDOFF_SAVED_HOLD', reason='Position saved. Stop programs and power off before rewiring.', action='stop')
                action = result['action']
                # Navigation forward only. Fire alignment/approach and both turns
                # retain their existing gait, and priority stops always win.
                hold_forward=(profile==1 and mission.active and action=='forward'
                              and mission.state=='FORWARD' and pending_level is None)
                stride_boost=forward_heading.update(hold_forward,mission.scan.forward_heading,yaw,now)
                result['forward_control']=forward_heading.diagnostic(yaw)

                # Apply HR22 leveling on the same motor-owning thread. No gait thread races.
                scan_state = mission.scan.state
                if mission.state == scan_state and mission.scan.needs_level:
                    pending_level = 'NAV'
                elif pending_level == 'NAV':
                    pending_level = None
                while mission.scan.events:
                    report('NAVIGATION',detail=mission.scan.events.popleft())
                fire_stage = mission.required_fire_level()
                if fire_stage:
                    pending_level = 'FIRE_'+fire_stage
                elif pending_level in ('FIRE_INITIAL', 'FIRE_FINAL'):
                    pending_level = None
                can_level = mission.active and mission.fresh(link.vision, now) and not mission.people(link.vision) and valid_yaw(yaw)
                context_matches = (pending_level == 'START' and mission.state == 'FORWARD'
                    or pending_level == 'NAV' and mission.state==scan_state and mission.scan.needs_level
                    or fire_stage is not None and pending_level == 'FIRE_'+fire_stage)
                if not a.execute and profile==1 and pending_level and can_level and context_matches:
                    report('LEVEL_START', reason=pending_level, simulated=True,
                           fire_leveling=mission.fire_level_diagnostic())
                    if pending_level=='NAV':
                        mission.scan.level_complete(yaw,now)
                    elif fire_stage:
                        mission.after_fire_level(fire_stage)
                    elif pending_level=='START':
                        mission.scan.forward_heading=yaw
                        forward_heading.reset()
                    report('LEVEL_DONE', reason=pending_level, simulated=True,
                           fire_leveling=mission.fire_level_diagnostic())
                    pending_level=None
                    continue
                if a.execute and profile == 1 and pending_level and can_level and context_matches:
                    from leveling import level_robot
                    reason = pending_level
                    level_context = dict(level_phase=reason, mission_state=mission.state,
                                         fire_leveling=mission.fire_level_diagnostic())
                    leveling_now, level_for_fire = True, reason in ('FIRE_INITIAL', 'FIRE_FINAL')
                    action = 'stop'
                    try:
                        guard()
                        if motion is None:
                            motion = Motion(a, guard)
                        report('LEVEL_START', reason=reason, **level_context)
                        details = level_robot(motion, sensors, guard)
                        pending_level = None
                        after = time.monotonic()
                        _, after_yaw, _ = sensors.snapshot()
                        if reason == 'NAV':
                            mission.scan.level_complete(after_yaw,after)
                        elif level_for_fire:
                            mission.after_fire_level(reason.removeprefix('FIRE_'))
                            mission.aim_at = None
                        elif reason == 'START':
                            mission.scan.forward_heading=after_yaw
                            forward_heading.reset()
                        report('LEVEL_DONE', reason=reason, details=details,
                               level_phase=reason, fire_leveling=mission.fire_level_diagnostic())
                    except InterruptMotion as exc:
                        report('LEVEL_INTERRUPTED', reason=str(exc), **level_context)
                    except Exception as exc:
                        failure = dict(getattr(exc, 'diagnostics', {}))
                        failure.setdefault('code', 'LEVEL_IO_OR_INTERNAL_ERROR')
                        failure['exception_type'] = type(exc).__name__
                        failure['heading'] = sensors.heading_diagnostic()
                        mission.stop('Leveling failed: '+str(exc))
                        link.stopped.set()
                        pending_level = None
                        report('LEVEL_FAILED', reason=str(exc), details=failure, **level_context)
                    finally:
                        leveling_now = level_for_fire = False
                    continue  # Refresh distances, priorities, target geometry after posture changes.
                if a.save_handoff and profile == 1 and saved_handoff is None:
                    snapshot = handoff.update(mission, now, distance, yaw, seq, link.vision,
                                               mission.fire_initial_leveled and mission.fire_final_leveled
                                               and pending_level is None,
                                               link.fresh() and not link.stopped.is_set())
                    if snapshot is not None:
                        gravity = gravity_unit(mission.current_acceleration)
                        if gravity is None:
                            handoff.reset()
                            continue
                        snapshot['gravity_unit'] = gravity
                        snapshot['hardware'] = a.execute
                        try:
                            saved_path = save_snapshot(snapshot, Path(__file__).with_name('handoffs'))
                        except OSError as exc:
                            mission.stop('Cannot save handoff: '+str(exc))
                            link.stopped.set()
                            report('HANDOFF_FAILED', reason=mission.reason)
                            handoff.reset()
                            continue
                        saved_handoff = snapshot
                        if combined:
                            # Stop/exit/person events may arrive during the snapshot write.
                            if (link.stopped.is_set() or not link.fresh() or camera.error
                                    or not mission.fresh(link.vision,time.monotonic())
                                    or any(d['label']=='person' and d['confidence']>=.35
                                           for d in link.vision['detections'])):
                                saved_handoff = None
                                handoff.reset()
                                report('COMBINED_TRANSITION_WAIT', reason='Fresh link, target and no person required')
                                continue
                            loaded_snapshot, snapshot_path = snapshot, saved_path
                            mission = switch_to_launcher(mission,base_cfg,snapshot,time.monotonic(),yaw)
                            cfg, profile = mission.cfg, 2
                            # Existing leg PWM holds the reached pose. No leg writes or
                            # frequency change occur as part of this phase transition.
                            pending_level = None
                            start_requested = heading_paused = False
                            align_origin = align_limit = None
                            forward_heading.reset()
                            action = 'stop'
                            report('HANDOFF_SAVED', snapshot=snapshot,saved_path=str(saved_path),
                                   reason='Approach saved; continuing to fresh launcher recheck')
                            report('COMBINED_PHASE',phase='LAUNCHER',previous_phase='NAVIGATION',
                                   reference_id=snapshot['id'],armed=a.arm,
                                   reason='Arrived and centered; recompute from fresh stopped observations')
                            report('STATUS',state='FIRE_RECHECK',action='stop',reason='Rechecking current fire and distance',
                                   combined=True,combined_phase='LAUNCHER',armed=a.arm,fire_active=False,
                                   sonar_raw_cm=distance,yaw_deg=yaw)
                            last_report,last_state=0.,None
                            continue
                        mission.stop('Position saved; manual power changeover required')
                        link.stopped.set()
                        pending_level = None
                        start_requested = heading_paused = False
                        action = 'stop'
                        report('HANDOFF_SAVED', snapshot=snapshot, saved_path=str(saved_path),
                               reason='Saved only. Exit client1, power off and rewire; then run client2.')
                        continue
                if mission.manual_recheck and mission.manual_recheck.generation != recheck_generation:
                    recheck_generation = mission.manual_recheck.generation
                    report('HANDOFF_RECHECKED', reference_id=loaded_snapshot['id'],
                           snapshot=mission.handoff_snapshot,
                           reason='New measurements accepted; plan recomputed for current observation')
                if result.get('rescue_event'):
                    report('RESCUE_REQUEST', reason='Person confirmed', vision=link.vision)
                plan = result.get('plan')
                if (a.execute and cfg.get('angle_output_enabled') is True and plan
                        and plan.get('angle_pwm') is not None
                        and mission.state in ('AIM_SETTLE', 'FIRE_READY', 'FIRE_READY_DISARMED')):
                    if launcher is None:
                        guard()
                        from launcher import Launcher
                        from pwm_driver import PCA9685
                        from angle_control import require_existing_setup, prepare_stationary_launcher
                        angle_board = PCA9685(a.bus, a.address)
                        if profile == 2:
                            # Standalone live targeting must also work after a power cycle.
                            # This only initializes an inactive board; it never sets a leg pose.
                            prepare_stationary_launcher(angle_board)
                        else:
                            require_existing_setup(angle_board)
                        launcher = Launcher(angle_board, cfg, allow_fire=a.arm,
                                            report=lambda data: report('LAUNCHER_PWM', diagnostics=data))
                    # Hold one calibrated PWM during the settle interval.
                    from aiming import pwm_for_angle, command_angle_for
                    calculated_angle=mission.last_aim if a.arm else plan['angle_deg']
                    command_angle=command_angle_for(calculated_angle,cfg)
                    command_pwm=pwm_for_angle(command_angle,cfg)
                    launcher.aim(command_pwm)
                    result.update(angle_command_deg=command_angle,angle_command_pwm=command_pwm,
                                  angle_calculated_deg=calculated_angle,angle_offset_deg=cfg.get('angle_offset_deg',0.),
                                  angle_measured=False)
                    if not a.arm:
                        result.update(state='FIRE_AIM_ONLY', reason='Channel 13 enabled; channel 14 disarmed (no --arm)')
                    if a.arm and result.get('fire') and not mission.shots:
                        guard()
                        # Inference may arrive while the angle register is written.
                        # Recheck the newest person/fire/range/link before channel 14.
                        current_distance, current_yaw, _ = sensors.snapshot()
                        if mission.firing_permitted(time.monotonic(), link.vision,
                                current_distance, current_yaw, mission.fire_box, link.fresh()):
                            if loaded_snapshot:
                                try:
                                    claim_shot(snapshot_path,loaded_snapshot)
                                except (OSError,ValueError) as exc:
                                    mission.stop('Snapshot shot reservation failed: '+str(exc))
                                    link.stopped.set()
                                    report('HANDOFF_SHOT_BLOCKED',reason=mission.reason)
                                    continue
                                report('HANDOFF_SHOT_RESERVED',reference_id=loaded_snapshot['id'])
                                mission.shots += 1
                                # Disk synchronization may take time: check fresh inputs AGAIN.
                                current_distance, current_yaw, _ = sensors.snapshot()
                                mission.current_acceleration = sensors.acceleration()
                                if (link.stopped.is_set() or not mission.firing_permitted(time.monotonic(),link.vision,
                                        current_distance,current_yaw,mission.fire_box,link.fresh())):
                                    mission.stop('Conditions changed after shot reservation; snapshot remains used')
                                    link.stopped.set()
                                    report('HANDOFF_SHOT_BLOCKED',reason=mission.reason)
                                    continue
                            firing_box = list(mission.fire_box)
                            if not loaded_snapshot:
                                mission.shots += 1
                            launcher.fire(time.monotonic())
                            fire_command_confirmed = True
                            report('FIRE_STARTED', plan=plan,angle_command_deg=command_angle,
                                   angle_calculated_deg=calculated_angle,angle_offset_deg=cfg.get('angle_offset_deg',0.),
                                   angle_command_pwm=command_pwm,angle_measured=False,
                                   pwm_readback=launcher.last_readback, physical_launch_verified=False)
                        else:
                            mission.aim_at = None
                            result.update(state='FIRE_RECHECK', fire=False,
                                          reason='Fresh person/fire/range/link check blocked firing')
                if sensors:
                    sensors.set_context(mission.state, action, motion.phase if motion else None)
                    batch = sensors.drain_diagnostics()
                    if batch:
                        report('SONAR_SAMPLES', samples=batch)
                if result['state'] != last_state or now-last_report >= .5:
                    report('STATUS', **result, sonar_raw_cm=distance, yaw_deg=yaw,
                           combined=combined, combined_phase=('NAVIGATION' if profile==1 else 'LAUNCHER') if combined else None,
                           armed=a.arm, fire_active=bool(launcher and launcher.active),
                           fire_command_confirmed=fire_command_confirmed, fire_confirmation_basis='channel14_register_readback',
                           fire_stop_reason=fire_stop_reason,
                           telemetry_dropped=sender.dropped,
                           heading_diagnostic=sensors.heading_diagnostic() if sensors else None,
                           sonar_diagnostic=sensors.diagnostic() if sensors else None)
                    last_report, last_state = now, result['state']
                if a.execute and action != 'stop':
                    align_origin = yaw if mission.state == 'FIRE_ALIGN' else None
                    align_limit = result.get('turn_degrees')
                    try:
                        guard()
                        if motion is None:
                            motion = Motion(a, guard)
                        speed = 65 if align_origin is not None else mission.scan.turn_speed(yaw) if action in ('left', 'right') else 100
                        motion.gait.cfg.YAW_COMPENSATION = stride_boost if action == "forward" else 0.
                        motion.step(action, speed)
                    except InterruptMotion as exc:
                        report('MOTION_INTERRUPTED', reason=str(exc))
                        action = 'stop'
                    finally:
                        align_origin = None
                time.sleep(.02)
        except KeyboardInterrupt:
            pass
        finally:
            link.stopped.set()
            try:
                if launcher:
                    launcher.close()
            finally:
                try:
                    if a.execute:
                        from leg_shutdown import release_leg_pwm
                        release_leg_pwm(motion.board if motion else angle_board, a.bus, a.address)
                        print('LEG PWM OFF. Holding force released.', flush=True)
                finally:
                    if angle_board:
                        angle_board.close()
                    if motion:
                        motion.close()
                    if sensors:
                        sensors.close()
                    if camera:
                        camera.close()
                    sender.close()
                    link.receiver.close()
            print('Stopped. Leg PWM release attempted. Launcher stop requested if initialized.', flush=True)


if __name__ == '__main__':
    main()
