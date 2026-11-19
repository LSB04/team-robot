"""Laptop YOLO/TinyGrid inference, independent heartbeat, rescue alerts and automatic recording."""
import argparse
import hashlib
from dashboard import render_dashboard, presentation
from fire_overlay import FireCenterMemory
from aiming import load_config
from datetime import datetime
import json
from pathlib import Path
import queue
import socket
import threading
import time
from transport import Sender, Receiver, validate_vision
from video import VideoReceiver
from vision_review import predict_review, RememberedFireReview, REACQUIRE_VERSION

ROOT = Path(__file__).resolve().parent


def beep():
    try:
        import winsound
        winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
    except ImportError:
        print('\a', end='', flush=True)


def main(argv=None, profile=1, combined=False):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port', type=int, default=5602 if combined else 5582 if profile == 1 else 5592)
    p.add_argument('--video-port', type=int, default=5603 if combined else 5583 if profile == 1 else 5593)
    p.add_argument('--model', type=Path, default=ROOT/'models/yolo_fire_person.pt')
    p.add_argument('--yolo-size', type=int, default=640, help='YOLO inference size; camera size is unchanged')
    p.add_argument('--headless', action='store_true')
    p.add_argument('--max-frames', type=int, default=0)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--window-width', type=int, default=960, help='Initial resizable monitor width; recording remains native size')
    a = p.parse_args(argv)
    if not 640 <= a.window_width <= 2560:
        p.error('--window-width must be 640..2560')
    import cv2
    import numpy as np
    from lightdet import Detector
    backend = 'cpu' if a.model.suffix.lower()=='.pt' else 'ort:cpu'
    detector = Detector(str(a.model), conf=.35, device=backend, yolo_size=a.yolo_size, threads=a.threads)
    if detector.names != ['fire', 'person', 'furniture']:
        raise ValueError('This robot requires model classes 0=fire, 1=person, 2=furniture')
    # Complete lazy model setup before accepting a robot connection.
    detector.predict(np.zeros((960,1280,3),dtype=np.uint8))
    print('MODEL:', a.model.name, detector.backend_info, 'input:', detector.size, flush=True)
    cfg = load_config()
    runtime_profile = profile
    fire_threshold=cfg.get('launcher_candidate_conf',.37) if profile==2 else cfg.get('fire_candidate_conf',.35)
    center_memory = FireCenterMemory(fire_threshold)
    fire_reacquire = RememberedFireReview()
    out = ROOT/'recordings'/('combined' if combined else f'server{profile}')/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    (out/'session.json').write_text(json.dumps(dict(
        profile=profile, combined=combined, model=str(a.model.resolve()),
        model_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(),
        backend=detector.backend_info, confidence=.35, confidence_is_accuracy=False,
        fire_candidate_conf=fire_threshold,
        inference=dict(model_input_size=detector.size, passes=('navigation: 1 or 2; launcher: 2' if combined else 2 if profile==2 else '1, or 2 when reacquiring'),
                       detail='upper-central 60 percent, mapped to original frame' if profile==2 else REACQUIRE_VERSION),
        geometry=cfg, fire_config_enabled=bool(cfg.get('fire_output_enabled') and (profile == 2 or combined)), arming_required=True,
        recording='Processed received frames in raw and dashboard videos; fixed 5 FPS; see events.jsonl for real timing',
        recording_files=dict(raw='camera.mp4',ui='camera_ui.mp4'),
        ui_recording='Full camera below the scaled three-card header, detection boxes and latest telemetry at each processed frame',
        camera_range='Sonar-derived estimate, not independent camera ranging',
        fire_center_overlay='Current center in cyan; last seen in gray LAST. Image coordinates only. Reset on start. UI recording and VISION logs.',
    ), ensure_ascii=False, indent=2), encoding='utf-8')
    sender = receiver = writer = ui_writer = video = conn = None
    ui_size = None
    window_created = False
    window_title = 'Robot - navigation and launcher' if combined else f'Server {profile} - robot monitor'
    frames = 0
    console = queue.Queue()

    def keyboard():
        while True:
            try:
                line = input().strip().lower()
            except EOFError:
                return
            command = {'': 'start', 'start': 'start', 's': 'stop', 'stop': 'stop', 'q': 'exit', 'exit': 'exit'}.get(line)
            if command:
                # Console stop bypasses inference and disk activity too.
                if sender:
                    sender.put({'command': command})
                if command == 'exit':
                    console.put(command)
                    return

    with (out/'events.jsonl').open('x', encoding='utf-8', buffering=1) as log, socket.socket() as listener:
        def record(value):
            log.write(json.dumps(dict(pc_wall_time=datetime.now().isoformat(),
                                      pc_monotonic=time.monotonic(), **value), ensure_ascii=False, allow_nan=False)+'\n')
        try:
            video = VideoReceiver(a.video_port)
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(('0.0.0.0', a.port))
            listener.listen(1)
            print(f'Waiting on TCP {a.port} (control), {a.video_port} (video). Logs: {out}', flush=True)
            conn, peer = listener.accept()
            conn.settimeout(.2)
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sender, receiver = Sender(conn, heartbeat=True), Receiver(conn)
            record(dict(event='CONNECTED', peer=peer, model=str(a.model)))
            threading.Thread(target=keyboard, daemon=True).start()
            print('Enter/start = start | s/stop = stop | q = exit. GUI: SPACE, S, Q.', flush=True)
            seq, frames, alert = -1, 0, False
            status, display = {}, None
            while not receiver.done.is_set() and not sender.done.is_set():
                recorded_panel = None
                while not console.empty():
                    command = console.get_nowait()
                    sender.put({'command': command})
                    if command == 'exit':
                        return
                for source in (receiver.records, sender.events):
                    for _ in range(512):
                        try:
                            item = source.get_nowait()
                        except queue.Empty:
                            break
                        record(item)
                        if item.get('event') in ('HANDOFF_SAVED','HANDOFF_LOADED','HANDOFF_RECHECKED'):
                            name = {'HANDOFF_SAVED':'fire_handoff.json',
                                    'HANDOFF_LOADED':'fire_handoff_loaded.json',
                                    'HANDOFF_RECHECKED':'fire_handoff_rechecked.json'}[item['event']]
                            (out/name).write_text(json.dumps(item['snapshot'], ensure_ascii=False, indent=2), encoding='utf-8')
                            print(item, flush=True)
                        if item.get('event') in ('HANDOFF_FAILED','HANDOFF_SHOT_RESERVED','HANDOFF_SHOT_BLOCKED'):
                            print(item, flush=True)
                        if item.get('event') in ('STATUS','COMBINED_PHASE') and item.get('mission_profile') == 2:
                            if runtime_profile != 2:
                                fire_threshold=cfg.get('launcher_candidate_conf',.37)
                                center_memory=FireCenterMemory(fire_threshold)
                                fire_reacquire.clear()
                            runtime_profile = 2
                        if item.get('event') in ('COMBINED_PHASE','COMBINED_TRANSITION_WAIT'):
                            print(item,flush=True)
                        if item.get('event') == 'STATUS':
                            if (item.get('state'),item.get('reason')) != (status.get('state'),status.get('reason')):
                                print(item, flush=True)
                            status = item
                            status['_received_pc'] = time.monotonic()
                        if item.get('event') == 'START_ACCEPTED':
                            alert = False
                            center_memory.clear()
                            fire_reacquire.clear()
                        if item.get('event') in ('START_ACCEPTED','START_REJECTED','START_WAIT','HEADING_ERROR',
                                                'CAMERA_INFO','LAUNCHER_PWM','FIRE_STARTED','FIRE_STOPPED',
                                                'HEADING_RECOVERED','HEADING_STOP','NAVIGATION_RESUMED','LEVEL_START',
                                                'LEVEL_DONE','LEVEL_FAILED','LEVEL_INTERRUPTED'):
                            print(item, flush=True)
                        if item.get('event')=='NAVIGATION':
                            print(item,flush=True)
                        if item.get('event') == 'RESCUE_REQUEST':
                            alert = True
                            threading.Thread(target=beep, daemon=True).start()
                            print('RESCUE REQUEST: PERSON DETECTED. ROBOT STOPPED.', flush=True)
                            if display is not None:
                                ok, encoded = cv2.imencode('.jpg', display)
                                if ok:
                                    (out/f'rescue_{time.time_ns()}.jpg').write_bytes(encoded.tobytes())
                latest = video.latest
                if latest and latest[0].get('seq') != seq and time.monotonic()-latest[2] < 1.5:
                    meta, jpeg, received = latest
                    frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        raise ValueError('Invalid JPEG frame')
                    if frame.shape[:2] != (meta['height'], meta['width']):
                        raise ValueError('Frame/header size mismatch')
                    if writer is not None and frame.shape[:2] != raw_shape:
                        raise ValueError('Camera dimensions changed; restart recording')
                    # Save raw decoded video before inference or UI annotation.
                    if writer is None:
                        raw_shape = frame.shape[:2]
                        record(dict(event='VIDEO_FORMAT', width=meta['width'],height=meta['height'],
                                    camera_model=meta.get('camera_model'),
                                    camera_view=meta.get('camera_view'),
                                    camera_crop=meta.get('camera_crop'),
                                    camera_reference_crop=meta.get('camera_reference_crop'),
                                    camera_intrinsics=meta.get('camera_intrinsics'),
                                    jpeg_quality=meta.get('jpeg_quality')))
                        print(f'VIDEO: {meta["width"]}x{meta["height"]}, view={meta.get("camera_view", "legacy")}',flush=True)
                        writer = cv2.VideoWriter(str(out/'camera.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 5., (frame.shape[1], frame.shape[0]))
                        if not writer.isOpened():
                            raise RuntimeError('Could not open MP4 recording')
                    writer.write(frame)
                    if runtime_profile == 1:
                        camera_key = (meta.get('camera_model'), meta.get('camera_view'), tuple(meta.get('camera_crop') or ()))
                        detections, inference_ms, inference_review = fire_reacquire.predict(
                            detector, frame, meta['capture_mono'], meta['seq'], camera_key, fire_threshold)
                    else:
                        detections, inference_ms, inference_review = predict_review(detector, frame, detail=True,
                                                                                  fire_threshold=fire_threshold)
                    packet = validate_vision(dict(meta, detections=detections))
                    sender.put(dict(command='vision', vision=packet))
                    seq = meta['seq']
                    display = frame.copy()
                    for obj in detections:
                        x1, y1, x2, y2 = map(int, obj['box'])
                        color = (0, 0, 255) if obj['label'] == 'person' else (0, 200, 255)
                        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                    fresh_status = 0 <= time.monotonic()-status.get('_received_pc', -1e9) < 1.5
                    selected_box = ((status.get('plan') or {}).get('target_box')
                                    if runtime_profile == 2 and fresh_status else None)
                    fire_center = center_memory.update(packet, time.monotonic(), selected_box)
                    center_memory.draw(display, fire_center)
                    panel_time=time.monotonic()
                    panel_values=presentation(status,packet,cfg,runtime_profile,panel_time,alert)
                    recorded_panel = render_dashboard(display,status,packet,cfg,runtime_profile,panel_time,alert,values=panel_values)
                    panel_size = (recorded_panel.shape[1],recorded_panel.shape[0])
                    if ui_writer is None:
                        ui_size = panel_size
                        ui_writer = cv2.VideoWriter(str(out/'camera_ui.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),5.,ui_size)
                        if not ui_writer.isOpened():
                            raise RuntimeError('Could not open UI MP4 recording')
                    if panel_size != ui_size:
                        raise RuntimeError('UI recording dimensions changed')
                    ui_writer.write(recorded_panel)
                    record(dict(event='VISION', **packet, inference_ms=inference_ms,
                                inference_review=inference_review,
                                ui_estimate=panel_values.get('estimate'),
                                fire_center=fire_center,
                                recording_frame=frames,ui_recording_frame=frames,
                                mission_profile=runtime_profile, ui_state=status.get('state'),ui_status_pi_monotonic=status.get('pi_monotonic')))
                    frames += 1
                    # Render telemetry on every GUI refresh, not only new camera frames.
                    if a.max_frames and frames >= a.max_frames:
                        break
                if not a.headless:
                    if display is not None:
                        panel = recorded_panel if recorded_panel is not None else render_dashboard(display, status,
                            packet if latest and time.monotonic()-latest[2] < 1.5 else None,
                            cfg, runtime_profile, time.monotonic(), alert)
                        if not window_created:
                            cv2.namedWindow(window_title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                            cv2.resizeWindow(window_title,a.window_width,
                                             round(panel.shape[0]*a.window_width/panel.shape[1]))
                            window_created = True
                        cv2.imshow(window_title, panel)
                    key = cv2.waitKey(1) & 255
                    command = {32: 'start', ord('s'): 'stop', ord('q'): 'exit'}.get(key)
                    if command:
                        console.put(command)
                if video.error:
                    raise RuntimeError(video.error)
                time.sleep(.01)
        except KeyboardInterrupt:
            pass
        finally:
            if sender:
                sender.close(send_exit=True)
                record(dict(event='SESSION_END', max_heartbeat_gap_ms=sender.max_gap_ms,
                            sender_dropped=sender.dropped, receiver_dropped=receiver.dropped,
                            sender_error=sender.error, receiver_error=receiver.error,
                            paired_recording_frames=frames,ui_recording_size=ui_size))
            if receiver:
                receiver.close()
            if conn:
                conn.close()
            if video:
                video.close()
            if writer:
                writer.release()
            if ui_writer:
                ui_writer.release()
            if not a.headless:
                cv2.destroyAllWindows()
            print('Saved:', out, flush=True)


if __name__ == '__main__':
    main()
