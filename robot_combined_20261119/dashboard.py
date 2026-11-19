"""Truthful display fields; no hardware or inference dependencies."""
import math
from functools import lru_cache
from pathlib import Path
from aiming import solution, alignment
from fire_overlay import overlap


def dashboard_lines(status, vision, cfg, profile, now, alert=False):
    fresh = 0 <= now - status.get('_received_pc', -1e9) < 1.5
    distance = status.get('sonar_raw_cm') if fresh else None
    valid = isinstance(distance, (int, float)) and math.isfinite(distance) and 0 < distance < 400
    sonar = f'{distance:.1f} cm' if valid else '-- cm'
    targets = [d for d in (vision or {}).get('detections', []) if d['label'] in ('person', 'fire')]
    camera = f'{distance + cfg["sonar_forward_cm"]:.1f} cm' if valid and targets else '-- cm'
    plan = status.get('plan') or {} if fresh else {}
    angle = plan.get('angle_command_deg',plan.get('angle_deg'))
    angle_text = f'{angle:.1f} deg' if isinstance(angle, (float, int)) else '-- deg'
    height = plan.get('target_ground_height_cm')
    height_text = f'{height:.1f}cm' if isinstance(height, (float, int)) else '--cm'
    return ['z - ' + sonar.replace(' cm', 'cm'), 'c - ' + camera.replace(' cm', 'cm'),
            'h - ' + height_text, 'a - ' + angle_text.replace(' deg', 'deg')]


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def display_estimate(status, vision, cfg, profile, now):
    """Screen-only geometry from a fresh candidate; never sent to motor control."""
    if not (0 <= now-status.get('_received_pc', -1e9) < 1.5 and vision):
        return None
    # Pi capture and status clocks can be compared with each other, not with PC now.
    if ('pi_monotonic' in status and 'capture_mono' in vision
            and abs(status['pi_monotonic']-vision['capture_mono']) > cfg['vision_max_age_seconds']):
        return None
    if status.get('state') in ('PERSON_CONFIRM','PERSON_RECHECK','RESCUE_HOLD','FAULT','VISION_WAIT','GYRO_WAIT'):
        return None
    person_threshold = .35 if profile == 2 else cfg['person_conf']
    if any(d['label']=='person' and d['confidence']>=person_threshold for d in vision.get('detections',[])):
        return None
    distance = status.get('sonar_raw_cm')
    if not (_finite(distance) and 2 <= distance < 400):
        return None
    threshold = cfg.get('launcher_candidate_conf',.37) if profile==2 else cfg.get('fire_candidate_conf',.35)
    fires=[d for d in vision.get('detections',[]) if d['label']=='fire' and d['confidence']>=threshold]
    if not fires:
        return None
    plan=status.get('plan') or {}
    selected=plan.get('target_box') or (status.get('launcher_review') or {}).get('box')
    matches=[d for d in fires if selected and overlap(d['box'],selected)>=.3]
    obj=max(matches or fires,key=lambda d:d['confidence'])
    if plan.get('target_box') and overlap(plan['target_box'],obj['box'])>=.3:
        values=plan
        source='control_plan'
    elif profile==2:
        values=solution(obj['box'],distance,vision['width'],vision['height'],cfg,vision.get('camera_intrinsics'))
        source='candidate_display_only'
    else:
        return None
    dx,limit,_=alignment(obj['box'],vision,dict(cfg,mission_profile=profile))
    calculated_angle=values.get('angle_deg')
    offset=values.get('angle_offset_deg',cfg.get('angle_offset_deg',0.))
    command_angle=values.get('angle_command_deg')
    if source=='control_plan' and _finite(status.get('angle_command_deg')):
        command_angle=status['angle_command_deg']
        calculated_angle=status.get('angle_calculated_deg',calculated_angle)
        offset=status.get('angle_offset_deg',offset)
    elif 'angle_command_deg' not in values:
        command_angle=calculated_angle  # Compatibility with recordings predating the offset.
    return dict(source=source,display_only=source=='candidate_display_only',
                height_cm=values.get('target_ground_height_cm'),angle_deg=command_angle,
                calculated_angle_deg=calculated_angle,angle_offset_deg=offset,
                fire_score=obj['confidence'],target_box=list(obj['box']),
                horizontal_error_px=dx,alignment_tolerance_px=limit,
                sonar_raw_cm=distance,vision_seq=vision['seq'],
                note='Geometry estimate; not a firing command or a measured height.')


def presentation(status, vision, cfg, profile, now, alert=False):
    """Translate actual telemetry only. Missing values never become zero."""
    fresh = 0 <= now - status.get('_received_pc', -1e9) < 1.5
    distance = status.get('sonar_raw_cm') if fresh else None
    valid = _finite(distance) and 2 <= distance < 400
    objects = (vision or {}).get('detections', [])
    people = [d for d in objects if d['label'] == 'person' and d['confidence'] >= cfg['person_conf']]
    fire_threshold = cfg.get('launcher_candidate_conf', .37) if profile==2 else cfg.get('fire_candidate_conf', .35)
    fires = [d for d in objects if d['label'] == 'fire' and d['confidence'] >= fire_threshold]
    camera = distance + cfg['sonar_forward_cm'] if valid and (people or fires) else None
    estimate = display_estimate(status,vision,cfg,profile,now)
    angle = (estimate or {}).get('angle_deg')
    target_height = (estimate or {}).get('height_cm')
    result = dict(z=f'{distance:.1f}' if valid else '--',
                  c=f'{camera:.1f}' if camera is not None else '--',
                  h=f'{target_height:.1f}' if _finite(target_height) else '--',
                  a=f'{angle:.1f}' if _finite(angle) else '--',
                  estimate=estimate,
                  title='수신 대기', detail='새 상태 정보를 기다립니다', color='muted')
    if not fresh:
        return result
    state, action = status.get('state'), status.get('action')
    heading_error = (status.get('heading_diagnostic') or {}).get('error')
    if heading_error or state == 'GYRO_WAIT':
        result.update(title='정지 · 자이로 수신 오류',
                      detail=('센서 안정화 후 자동 재개 대기' if status.get('heading_auto_resume') else
                              '센서 복구 후 서버 터미널에 start 입력'),color='red')
        return result
    if state == 'HANDOFF_SAVED_HOLD':
        result.update(title='불 위치 저장 완료 · 정지',
                      detail='프로그램 종료 · 전원 차단 후 연결 변경', color='amber')
        return result
    if state == 'IDLE':
        result.update(title='정지 · 주행 시작 대기',
                      detail='서버 터미널: start + Enter  |  영상 창: Space',color='muted')
        return result
    if state in ('FAULT','VISION_WAIT'):
        result.update(title='정지 · 점검 필요' if state == 'FAULT' else '정지 · 영상 수신 대기',
                      detail='서버 터미널에서 정지 원인을 확인하세요',color='red')
        return result
    sides = status.get('scan_results') or {}
    selected = {'left': '왼쪽', 'right': '오른쪽'}.get(sides.get('selected_side'))
    def side(name):
        value = sides.get(name + '_cm')
        return f'{value:.1f} cm' if _finite(value) and 2 <= value < 400 else '-- cm'
    result['detail'] = f'왼쪽 {side("left")}     |     오른쪽 {side("right")}'
    if state == 'RESCUE_HOLD':
        result.update(title='사람 발견 · 구조 요청 전송', color='red')
    elif state == 'PERSON_RECHECK':
        result.update(title='사람 후보 · 정지 후 재확인', detail='새 영상으로 사람 여부 확인 중', color='amber')
    elif people:
        result.update(title='사람 감지 · 확인 중', color='red')
    elif state == 'FIRE_RECHECK':
        result.update(title='불 후보 · 정지 후 재확인',color='amber')
    elif state in ('FIRE_RANGE_RECHECK','FIRE_RANGE_WAIT'):
        result.update(title='불 거리 · 정지 후 재측정',color='amber')
    elif fires:
        fire_title = {'FIRE_ALIGN':'불 추적 · 방향 정렬', 'FIRE_APPROACH':'불 추적 · 200cm까지 접근', 'FIRE_POSITION_READY':'불 위치 확보 · 자세 유지', 'FIRE_TOO_CLOSE':'정지 · 불/장애물 거리 확인'}.get(state, '불 감지 · 거리 측정')
        if profile == 2:
            fire_title = ('발사 명령 출력 중' if status.get('fire_active') else
                          {'FIRE_ALIGN':'불 중심 정렬 필요', 'FIRE_APPROACH':'발사 거리 범위 밖',
                           'AIM_SETTLE':'발사각 조절 · 안정화', 'FIRE_READY':'발사 준비',
                           'FIRE_CALIBRATION_BLOCK':'발사각 범위 확인', 'SHOT_COMPLETE':'발사 명령 완료 · 동작 미확인',
                           'FIRE_READY_DISARMED':'불 높이 · 각도 계산',
                           'FIRE_AIM_ONLY':'발사각 조절 · 발사 비활성'}.get(state, fire_title))
        result.update(title=fire_title, color='amber')
    elif state == 'FORWARD':
        result.update(title=(f'{selected} 선택 → 전진 재개' if selected else '전진') if action == 'forward' else (f'{selected} 선택 · 전진 준비' if selected else '전진 준비'), color='green')
    elif state == 'FORWARD_TURN':
        result.update(title='전진 방향 확인·보정', color='blue')
    elif state in ('LEFT_RETRY_TURN','LEFT_RETRY_SAMPLE'):
        result.update(title='왼쪽 재탐색 · 거리 측정' if state.endswith('SAMPLE') else '왼쪽 재탐색 · 각도 변경',color='blue')
    elif state in ('LEFT_TURN', 'LEFT_SAMPLE'):
        result.update(title='왼쪽 거리 측정' if state == 'LEFT_SAMPLE' else '왼쪽 탐색 중' if action == 'left' else '왼쪽 탐색 준비', color='blue')
    elif state in ('RIGHT_TURN', 'RIGHT_SAMPLE'):
        result.update(title='오른쪽 거리 측정' if state == 'RIGHT_SAMPLE' else '오른쪽 탐색 중' if action == 'right' else '오른쪽 탐색 준비', color='blue')
    elif state == 'SELECT_TURN':
        result.update(title=f'{selected or "선택"} 방향으로 전환', color='blue')
    elif state == 'CHECK_SAMPLE':
        result.update(title=f'{selected or "선택 방향"} 선택 · 전방 재확인', color='blue')
    else:
        title = {'IDLE':'정지', 'STEP_CHECK':'정지 · 전방 거리 확인', 'SONAR_WAIT':'정지 · 초음파 수신 대기', 'GYRO_WAIT':'정지 · 자이로 수신 대기', 'VISION_WAIT':'정지 · 영상 수신 대기', 'FAULT':'정지 · 점검 필요', 'LAUNCHER_WAIT':'불 검출 대기'}.get(state, '정지')
        result.update(title=title, color='muted')
    nav=status.get('navigation') or {}
    if not people and not fires and state!='RESCUE_HOLD':
        if nav.get('phase') in ('level','verify','trim') and state.endswith('_TURN'):
            result['title']=('전진 방향 보정' if nav['phase']=='trim' else '전진 방향 확인') if state=='FORWARD_TURN' else '회전 후 각도 보정' if nav['phase']=='trim' else '회전 후 자세·각도 확인'
        if nav.get('phase') in ('level','verify','trim'):
            error=nav.get('yaw_error_deg')
            if _finite(error):
                tolerance=nav.get('turn_tolerance_deg')
                limit=f'±{tolerance:g}°' if _finite(tolerance) else '--°'
                result['detail']=f'방향 오차 {error:+.1f}°     |     허용 {limit}'
    if people or fires or state == 'RESCUE_HOLD':
        target = '사람' if people or state == 'RESCUE_HOLD' else '불'
        result['detail'] = f'{target} 거리 c - {result["c"]} cm'
        if target == '불':
            if profile == 1:
                fire_status = '방향 정렬 · 200cm 접근'
            elif status.get('fire_active'):
                fire_status = '발사 명령 출력 중'
            elif status.get('armed'):
                fire_status = '발사 활성 · 조건 확인'
            elif cfg.get('angle_pwm_mode') != 'team_reference' and len(cfg.get('angle_pwm_points', [])) < 2:
                fire_status = '보정 대기 · 발사 비활성'
            else:
                fire_status = '발사 비활성'
            if profile == 2 and cfg.get('launcher_operating_mode') == 'team_reference':
                fire_status += ' · 참조값 계산'
            result['detail'] += '     |     ' + fire_status
        elif state == 'RESCUE_HOLD':
            result['detail'] += '     |     경고음 · 로그 저장'
    if profile == 1 and state == 'FIRE_RECHECK' and not people:
        review = status.get('fire_review') or {}
        persistent_hits = review.get('persistent_hits',0)
        persistent_need = review.get('persistent_hits_required',8)
        peak = review.get('peak',0.)
        required = review.get('peak_required',.45)
        result['detail'] = f'반복 확인 {persistent_hits}/{persistent_need} · 최고 {peak:.2f}/{required:.2f} · 대상 해제: start'
    elif profile == 1 and state in ('FIRE_RANGE_RECHECK','FIRE_RANGE_WAIT') and not people:
        review = status.get('fire_range_review') or {}
        spread = review.get('spread_cm')
        spread_text = f'{spread:.1f}cm' if _finite(spread) else '--cm'
        result['detail'] = f'새 거리 {review.get("samples",0)}/5     |     변동폭 {spread_text}'
    if profile == 2 and state in ('PERSON_RECHECK', 'PERSON_CONFIRM', 'RESCUE_HOLD'):
        result['a'] = result['h'] = '--'
        if state == 'PERSON_RECHECK':
            result.update(title='사람 후보 · 정지 후 재확인', color='amber',
                          detail=f'정지 후 새 영상 {min(5,status.get("person_recheck_frames",0))}/5 · 발사 중지')
        elif state == 'PERSON_CONFIRM':
            result.update(title='사람 감지 · 확인 중', detail='사람 우선 확인 · 발사 중지', color='red')
    elif profile == 2 and state == 'FIRE_CONFIRM':
        review = status.get('launcher_review') or {}
        score=review.get('visible_score')
        score_text=f'{score:.2f}' if _finite(score) else '--'
        required=cfg.get('launcher_review_peak_conf',.37)
        result.update(title='불 후보 · 정지 후 재확인', color='amber',
                      detail=f'같은 불 {min(5,review.get("hits",0))}/5장 · 필요 3장 · 점수 {score_text}/{required:.2f}')
    elif profile == 2 and state == 'FIRE_ALIGN' and estimate:
        dx=estimate['horizontal_error_px'];limit=estimate['alignment_tolerance_px']
        result['detail']=f'불 중심 {"오른쪽" if dx>0 else "왼쪽"} {abs(dx):.0f}px · 허용 ±{limit:.0f}px · 발사 대기'
    elif profile == 2 and state == 'AIM_SETTLE' and estimate and estimate['fire_score']<cfg['fire_conf']:
        result.update(title='각도 유지 · 불 재확인',
                      detail=f'불 점수 {estimate["fire_score"]:.2f} · 발포 기준 {cfg["fire_conf"]:.2f}')
    if (profile == 2 and status.get('fire_command_confirmed') is True
            and state in ('FIRE_READY','SHOT_COMPLETE') and not people):
        result.update(title='발포 확인', color='green',
                      detail='14번 PWM 출력 기준 · ' + ('구동 중' if status.get('fire_active')
                            else '구동 완료' if status.get('fire_stop_reason')=='Pulse complete' else '구동 중지'))
    return result


@lru_cache(maxsize=8)
def _font(size, bold=False):
    from PIL import ImageFont
    candidates = [Path('C:/Windows/Fonts') / ('malgunbd.ttf' if bold else 'malgun.ttf'),
                  Path('/usr/share/fonts/truetype/nanum') / ('NanumGothicBold.ttf' if bold else 'NanumGothic.ttf')]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    raise RuntimeError('Korean UI font missing: install Malgun Gothic or Nanum Gothic on the laptop')


def header_height(width):
    return 2*math.ceil(200*max(1., width/960)/2)


def render_dashboard(frame, status, vision, cfg, profile, now, alert=False, values=None):
    """Readable three-card header outside the camera; scales with video width."""
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw
    values = presentation(status, vision, cfg, profile, now, alert) if values is None else values
    width = frame.shape[1]
    scale = max(1., width/960)
    top = header_height(width)
    def px(value):
        return round(value*scale)
    camera = frame
    image = Image.new('RGB', (width, top + camera.shape[0]), '#111821')
    image.paste(Image.fromarray(cv2.cvtColor(camera, cv2.COLOR_BGR2RGB)), (0, top))
    draw = ImageDraw.Draw(image)
    palette = dict(muted='#CAD3DF', green='#67E8A3', blue='#7CB8FF', amber='#FFD166', red='#FF8B95')
    accent = palette[values['color']]
    def line(x, y, text, size, fill, bold=False, available=None):
        available = width-x-px(16) if available is None else available
        size = px(size)
        font = _font(size,bold)
        while draw.textlength(text,font=font) > available and size > px(14):
            size -= 1
            font = _font(size,bold)
        if draw.textlength(text,font=font) > available:
            while text and draw.textlength(text+'…',font=font) > available:
                text = text[:-1]
            text += '…'
        draw.text((x,px(y)),text,font=font,fill=fill,anchor='lt')
    angle_label='a · 예상 명령각' if (values.get('estimate') or {}).get('display_only') else 'a · 명령 각도'
    for i, (key, label, unit) in enumerate((('z', 'z · 초음파', 'cm'), ('c', f'c · +{cfg["sonar_forward_cm"]:g} cm', 'cm'), ('a', angle_label, '°'))):
        stride = (width-13)//3
        x = 12 + i*stride
        draw.rounded_rectangle((x, px(8), x+stride-11, px(94)), radius=px(8), fill='#202C3A')
        line(x+px(10),16,label,18,'#F0F4FA',available=stride-px(28))
        value = values[key]
        draw.text((x+px(10),px(60 if value=='--' else 44)), value, font=_font(px(36), True), fill='#FFFFFF',anchor='lt')
        value_width = draw.textlength(value, font=_font(px(36), True))
        draw.text((x+px(16)+value_width,px(58)), unit, font=_font(px(18)), fill='#F0F4FA',anchor='lt')
    draw.rounded_rectangle((12, px(105), 17, px(188)), radius=2, fill=accent)
    line(px(24),105,values['title'],24,accent,True)
    line(px(24),140,values['detail'],18,'#F0F4FA')
    height_note = (f'h · 불 중심 높이 {values["h"]} cm · 추정'
                   if values['h']!='--' else 'h · 불 중심 높이 : 계산 대기')
    estimate=values.get('estimate') or {}
    if _finite(estimate.get('calculated_angle_deg')) and estimate.get('angle_offset_deg'):
        height_note+=f'  |  계산 {estimate["calculated_angle_deg"]:.1f}° {estimate["angle_offset_deg"]:+g}°'
    line(px(24),172,height_note,20,'#FFFFFF',True)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
