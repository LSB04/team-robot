"""Pure phase transition from completed navigation to stationary launcher recheck."""
from copy import deepcopy
from core import valid_yaw
from fire_handoff import ManualRecheck, config_signature
from mission import Mission

VERSION = 'combined_navigation_launcher_v1'


def switch_to_launcher(previous, base_cfg, snapshot, now, yaw):
    if not (previous.cfg.get('mission_profile') == 1 and previous.active
            and not previous.rescue and previous.shots == 0
            and previous.state == 'FIRE_POSITION_READY' and previous.last_action == 'stop'
            and previous.fire_arrival_locked and previous.fire_position_reached
            and previous.fire_initial_leveled and previous.fire_final_leveled
            and valid_yaw(yaw)):
        raise ValueError('Navigation arrival and final alignment must complete before launcher mode')
    if (snapshot.get('version') != 2
            or snapshot.get('config_sha256') != config_signature(base_cfg)
            or not base_cfg['threshold_cm'] < snapshot['sonar_raw_cm'] <= base_cfg['fire_target_cm']
            or not 0 <= now-snapshot['pi_monotonic'] <= 1.5):
        raise ValueError('A fresh matching arrival snapshot is required')
    cfg=deepcopy(base_cfg)
    cfg['mission_profile']=2
    mission=Mission(cfg)
    mission.current_acceleration=deepcopy(previous.current_acceleration)
    if not mission.start(now,yaw):
        raise ValueError('Fresh heading is required for the launcher phase')
    mission.manual_recheck=ManualRecheck(snapshot,cfg)
    return mission
