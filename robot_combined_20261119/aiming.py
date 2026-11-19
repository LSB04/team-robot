"""Pure geometry and spring model; hardware calibration is separate."""
import json
import math
from pathlib import Path


def load_config(path=None):
    path = Path(path) if path else Path(__file__).with_name('mission_config.json')
    cfg = json.loads(path.read_text(encoding='utf-8-sig'))
    for key, value in cfg.items():
        if isinstance(value, (float, int)) and not math.isfinite(value):
            raise ValueError(f'Nonfinite configuration: {key}')
    if cfg['angle_channel'] != 13 or cfg['fire_channel'] != 14:
        raise ValueError('This package reserves channel 13 for angle and 14 for firing')
    angle_offset(cfg)
    if not 0 < cfg['fire_seconds'] <= 5:
        raise ValueError('fire_seconds must be in (0, 5]')
    if not (35 <= cfg['threshold_cm'] < 400 and 0 < cfg['fire_min_cm'] < cfg['fire_max_cm'] < 400):
        raise ValueError('Invalid navigation / firing distances')
    if not 0 < cfg['vision_max_age_seconds'] <= 2 or not 2 <= cfg['confirm_frames'] <= 20:
        raise ValueError('Invalid vision freshness / confirmation')
    if cfg['focal_at_480_px'] <= 0 or abs(cfg['camera_pitch_deg']) >= 60:
        raise ValueError('Invalid camera calibration')
    if not 0 < cfg['fire_conf'] < 1 or not 0 < cfg['person_conf'] < 1:
        raise ValueError('Invalid confidence thresholds')
    for key, upper in (('fire_stop_pwm',450), ('fire_active_pwm',492)):
        if not isinstance(cfg[key], int) or not 150 <= cfg[key] <= upper:
            raise ValueError(f'Invalid {key}')
    speed(cfg)
    return cfg


def speed(cfg):
    if cfg['measured_speed_m_s'] is not None:
        value = float(cfg['measured_speed_m_s'])
        if not math.isfinite(value) or value <= 0:
            raise ValueError('Invalid measured speed')
        return value, 'measured'
    compression = cfg['spring_free_m']-cfg['spring_compressed_m']
    if min(compression, cfg['spring_k_n_m'], cfg['ball_diameter_cm'], cfg['ball_density_g_cm3']) <= 0:
        raise ValueError('Invalid spring model')
    mass = 4/3*math.pi*(cfg['ball_diameter_cm']/2)**3*cfg['ball_density_g_cm3']/1000
    return math.sqrt(cfg['spring_k_n_m']*compression**2/mass), 'ideal_spring_not_measured'


def pwm_for_angle(angle, cfg):
    if not math.isfinite(angle) or not cfg['angle_min_deg'] <= angle <= cfg['angle_max_deg']:
        raise ValueError('Angle outside configured range')
    mode = cfg.get('angle_pwm_mode', 'measured')
    if mode == 'team_reference':
        # Same gear and PWM span as test12; current horizontal PWM and direction
        # were confirmed by the user. This is not a measured angle calibration.
        zero = cfg['angle_horizontal_pwm']
        span, degrees, ratio = (cfg[k] for k in ('angle_pwm_span', 'angle_servo_span_deg', 'angle_gear_ratio'))
        if (not isinstance(zero, int) or not 150 <= zero <= 450
                or not all(isinstance(v, (float, int)) and math.isfinite(v) and v > 0
                           for v in (span, degrees, ratio))):
            raise ValueError('Invalid team angle reference')
        value = round(zero + angle * ratio * span / degrees)
        if not 150 <= value <= 450:
            raise ValueError('Reference angle exceeds configured PWM limits')
        return value
    if mode != 'measured':
        raise ValueError('Unknown angle PWM mode')
    points = cfg['angle_pwm_points']
    if len(points) < 2:
        raise ValueError('Need at least two measured angle/PWM calibration points')
    for i, (a, pwm) in enumerate(points):
        if not math.isfinite(a) or not isinstance(pwm, int) or not 150 <= pwm <= 450:
            raise ValueError('Invalid angle calibration point')
        if i and a <= points[i-1][0]:
            raise ValueError('Angle calibration points must increase in angle')
        if i and pwm <= points[i-1][1]:
            raise ValueError('PWM must increase as barrel angle increases')
    if not cfg['angle_min_deg'] <= angle <= cfg['angle_max_deg']:
        raise ValueError('Angle outside allowed mechanical range')
    for (a, p), (b, q) in zip(points, points[1:]):
        if a <= angle <= b:
            return round(p+(q-p)*(angle-a)/(b-a))
    raise ValueError('Angle outside measured PWM calibration; no extrapolation')


def angle_offset(cfg):
    value=cfg.get('angle_offset_deg',0.)
    if (isinstance(value,bool) or not isinstance(value,(float,int))
            or not math.isfinite(value) or abs(value)>60):
        raise ValueError('Invalid angle offset')
    return float(value)


def command_angle_for(calculated_angle, cfg):
    """Add the user offset once, before converting the motor target to PWM."""
    if not isinstance(calculated_angle,(float,int)) or not math.isfinite(calculated_angle):
        raise ValueError('Invalid calculated angle')
    command=calculated_angle+angle_offset(cfg)
    if not cfg['angle_min_deg'] <= command <= cfg['angle_max_deg']:
        raise ValueError('Offset angle outside configured range; no clipping')
    return command


def arm_check(cfg):
    angle_offset(cfg)
    if cfg.get('fire_output_enabled') is not True:
        raise ValueError('Channel 14 firing is temporarily locked in mission_config.json')
    if cfg.get('angle_output_enabled') is not True:
        raise ValueError('Channel 13 angle output is disabled')
    mode = cfg.get('launcher_operating_mode', 'calibrated')
    if mode == 'team_reference':
        if not (cfg.get('angle_reference_confirmed') is True and cfg.get('fire_reference_confirmed') is True):
            raise ValueError('Team angle and firing references must be confirmed')
        if cfg.get('angle_pwm_mode') != 'team_reference':
            raise ValueError('Team reference operation requires the matching angle model')
        speed(cfg)
    elif mode == 'calibrated':
        if not all(cfg[k] is True for k in ('launcher_verified', 'geometry_verified', 'range_plane_verified')):
            raise ValueError('Launcher, geometry and sonar target-plane verification are required for calibrated mode')
        if cfg['measured_speed_m_s'] is None:
            raise ValueError('Measured projectile speed is required for calibrated mode')
    else:
        raise ValueError('Unknown launcher operating mode')
    pwm_for_angle(cfg['angle_min_deg'], cfg)
    pwm_for_angle(cfg['angle_max_deg'], cfg)


def camera_geometry(width, height, cfg, intrinsics=None):
    if intrinsics is None:
        return (cfg['focal_at_480_px']*width/640,
                cfg['focal_at_480_px']*height/480, width/2, height/2)
    values = tuple(intrinsics[k] for k in ('fx', 'fy', 'cx', 'cy'))
    if (not all(isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v) for v in values)
            or min(values[:2]) <= 0 or not 0 <= values[2] <= width or not 0 <= values[3] <= height):
        raise ValueError('Invalid frame camera geometry')
    return values


def alignment(box, vision, cfg):
    fx, _, cx, _ = camera_geometry(vision['width'], vision['height'], cfg, vision.get('camera_intrinsics'))
    dx = (box[0]+box[2])/2-cx
    reference_tolerance = (cfg.get('launcher_align_tolerance_px_at_640',cfg['align_tolerance_px_at_640'])
                           if cfg.get('mission_profile') == 2 else cfg['align_tolerance_px_at_640'])
    tolerance = reference_tolerance*fx/cfg['focal_at_480_px'] + 1e-9
    return dx, tolerance, fx


def solution(box, distance, width, height, cfg, intrinsics=None):
    depth = distance+cfg['sonar_forward_cm']
    cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
    fx, fy, principal_x, principal_y = camera_geometry(width, height, cfg, intrinsics)
    x = (cx-principal_x)*depth/fx-cfg['muzzle_right_cm']
    y = (principal_y-cy)*depth/fy
    pitch = math.radians(cfg['camera_pitch_deg'])
    up = y*math.cos(pitch)+depth*math.sin(pitch)
    forward = depth*math.cos(pitch)-y*math.sin(pitch)
    dy = up-cfg['muzzle_above_camera_cm']
    dz = forward-cfg['muzzle_forward_cm']
    horizontal = math.hypot(x, dz)/100
    velocity, basis = speed(cfg)
    a = 9.81*horizontal**2/(2*velocity**2)
    discriminant = horizontal**2-4*a*(dy/100+a)
    angle = None
    if dz > 0 and discriminant >= 0 and a > 0:
        angle = math.degrees(math.atan2(2*(dy/100+a), horizontal+math.sqrt(discriminant)))
    pwm, command_angle, why = None, None, ''
    if angle is not None:
        try:
            command_angle = command_angle_for(angle, cfg)
            pwm = pwm_for_angle(command_angle, cfg)
        except ValueError as exc:
            why = str(exc)
    else:
        why = 'No reachable forward ballistic solution'
    return dict(angle_deg=angle, angle_command_deg=command_angle,
                angle_offset_deg=angle_offset(cfg), angle_pwm=pwm, calibration_error=why,
                target_box=list(box), target_center_x_px=cx, target_center_y_px=cy,
                camera_intrinsics=dict(fx=fx, fy=fy, cx=principal_x, cy=principal_y),
                target_ground_height_cm=cfg['camera_height_cm']+up,
                height_from_muzzle_cm=dy, horizontal_cm=horizontal*100,
                camera_depth_cm=depth, speed_m_s=velocity, speed_basis=basis,
                angle_pwm_basis=cfg.get('angle_pwm_mode', 'measured'),
                launcher_operating_mode=cfg.get('launcher_operating_mode', 'calibrated'),
                estimate_only=(not cfg['geometry_verified'] or basis != 'measured'
                               or cfg.get('angle_pwm_mode') == 'team_reference'))
