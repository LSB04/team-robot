"""Separate TCP video path. Camera capture timestamps use the Pi monotonic clock."""
import json
import math
import socket
import struct
import threading
import time


def read_exact(conn, size, done):
    buffer = bytearray()
    while len(buffer) < size and not done.is_set():
        try:
            chunk = conn.recv(size-len(buffer))
        except socket.timeout:
            continue
        if not chunk:
            raise EOFError('Video disconnected')
        buffer.extend(chunk)
    if len(buffer) != size:
        raise EOFError('Video stopped')
    return bytes(buffer)


class VideoReceiver:
    def __init__(self, port):
        self.done = threading.Event()
        self.latest = None
        self.error = None
        self.listener = socket.socket()
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(('0.0.0.0', port))
        self.listener.listen(1)
        self.listener.settimeout(.2)
        self.conn = None
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        try:
            while not self.done.is_set():
                try:
                    self.conn, _ = self.listener.accept()
                    break
                except socket.timeout:
                    continue
            if self.conn is None:
                return
            with self.conn as conn:
                conn.settimeout(.2)
                while not self.done.is_set():
                    header_len, image_len = struct.unpack('!II', read_exact(conn, 8, self.done))
                    if not 1 <= header_len <= 2048 or not 1 <= image_len <= 4_000_000:
                        raise ValueError('Invalid video lengths')
                    header = json.loads(read_exact(conn, header_len, self.done))
                    jpeg = read_exact(conn, image_len, self.done)
                    self.latest = (header, jpeg, time.monotonic())
        except (OSError, EOFError, ValueError) as exc:
            self.error = str(exc)
        finally:
            self.done.set()

    def close(self):
        self.done.set()
        self.listener.close()
        if self.conn:
            try:
                self.conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self.thread.join(1)


def crop_rect(value):
    rect = tuple(int(v) for v in value)
    if len(rect) != 4 or min(rect[:2]) < 0 or min(rect[2:]) <= 0:
        raise ValueError('Invalid camera crop rectangle')
    return rect


def widest_mode(modes, size):
    candidates = [m for m in modes if 'crop_limits' in m and 'bit_depth' in m and m.get('fps', 0) >= 5]
    if not candidates:
        raise ValueError('Camera does not report usable sensor modes with crop limits')
    return min(candidates, key=lambda m: (
        -m['crop_limits'][2]*m['crop_limits'][3],
        not (m['size'][0] >= size[0] and m['size'][1] >= size[1]),
        m['size'][0]*m['size'][1], -m['bit_depth']))


def frame_intrinsics(reference_crop, current_crop, size, focal):
    """Transfer the existing 640x480 pinhole reference through sensor cropping.

    This preserves the old angular rays; it is not a new physical calibration.
    Crop rectangles and principal points use full sensor pixel coordinates.
    """
    rx, ry, rw, rh = crop_rect(reference_crop)
    x, y, w, h = crop_rect(current_crop)
    width, height = size
    return dict(fx=focal*rw/640*width/w, fy=focal*rh/480*height/h,
                cx=(rx+rw/2-x)*width/w, cy=(ry+rh/2-y)*height/h,
                basis='legacy_640x480_reference_transformed_by_sensor_crop')


def view_crop(maximum, reference, view):
    """Balanced view uses the central 80 percent of the full sensor width."""
    x, y, w, h = crop_rect(maximum)
    if view == 'legacy':
        return crop_rect(reference)
    if view == 'full':
        return x, y, w, h
    if view != 'balanced':
        raise ValueError('Unknown camera view')
    # Even 4:3 crop. Actual frame metadata remains authoritative for geometry.
    cw = int(min(w*.8, h*.8*4/3))//8*8
    ch = cw*3//4
    if cw <= 0:
        raise ValueError('Sensor crop is too small')
    return x+(w-cw)//4*2, y+(h-ch)//4*2, cw, ch


class CameraSender:
    def __init__(self, host, port, execute, size=(1280, 960), quality=90,
                 view='balanced', focal=935.):
        if (len(size) != 2 or any(not isinstance(v, int) or not 320 <= v <= 1920 or v % 2 for v in size)
                or size[0]*3 != size[1]*4 or not 50 <= quality <= 95
                or view not in ('full', 'balanced', 'legacy') or not math.isfinite(focal) or focal <= 0):
            raise ValueError('Expected even 4:3 camera size, quality 50..95 and full/balanced/legacy view')
        self.host, self.port, self.execute = host, port, execute
        self.size, self.quality, self.view, self.focal = size, quality, view, focal
        self.info = None
        self.done = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        camera = None
        try:
            import cv2
            import numpy as np
            if self.execute:
                from picamera2 import Picamera2
                camera = Picamera2()
                model = str(camera.camera_properties.get('Model', 'unknown'))
                modes = camera.sensor_modes
                # Recover the exact previous view before selecting the wider mode.
                # The configured focal reference was used with this 640x480 mode.
                camera.configure(camera.create_video_configuration(
                    main={'size': (640, 480), 'format': 'RGB888'}, buffer_count=2))
                legacy_config = camera.camera_configuration()
                camera.start()
                reference_crop = crop_rect(camera.capture_metadata()['ScalerCrop'])
                camera.stop()
                mode = widest_mode(modes, self.size)
                sensor = (dict(output_size=mode['size'], bit_depth=mode['bit_depth'])
                          if self.view != 'legacy' else legacy_config['sensor'])
                frame_time = round(1e6/min(30., float(mode['fps']))) if self.view != 'legacy' else 33333
                # Picamera2 RGB888 arrays have OpenCV-compatible BGR byte order.
                camera.configure(camera.create_video_configuration(
                    main={'size': self.size, 'format': 'RGB888'}, sensor=sensor,
                    controls={'FrameDurationLimits': (frame_time, frame_time)}, buffer_count=2,
                    queue=False))
                maximum = crop_rect(camera.camera_properties['ScalerCropMaximum'])
                requested_crop = view_crop(maximum, reference_crop, self.view)
                camera.set_controls({'ScalerCrop': requested_crop})
                camera.start()
                self.info = dict(model=model,view=self.view, output_size=list(self.size), jpeg_quality=self.quality,
                                 reference_crop=list(reference_crop), maximum_crop=list(maximum),
                                 requested_crop=list(requested_crop),
                                 sensor_size=list(sensor['output_size']), fps=5)
            else:
                reference_crop = (0, 0, 640, 480)
                self.info = dict(model='simulation',view='simulation', output_size=list(self.size), jpeg_quality=self.quality,
                                 reference_crop=list(reference_crop), fps=5)
            print('CAMERA_INFO:', json.dumps(self.info), flush=True)
            with socket.create_connection((self.host, self.port), timeout=5) as conn:
                conn.settimeout(1.)
                sequence = 0
                while not self.done.is_set():
                    started = time.monotonic()
                    if camera:
                        request = camera.capture_request()
                        try:
                            frame = request.make_array('main')
                            current_crop = crop_rect(request.get_metadata()['ScalerCrop'])
                        finally:
                            request.release()
                    else:
                        frame = np.zeros((self.size[1], self.size[0], 3), np.uint8)
                        current_crop = reference_crop
                    stamp = time.monotonic()
                    sequence += 1
                    ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                    if not ok:
                        raise ValueError('JPEG encoding failed')
                    header = json.dumps(dict(seq=sequence, capture_mono=stamp,
                                             width=frame.shape[1], height=frame.shape[0],
                                             camera_intrinsics=frame_intrinsics(reference_crop, current_crop,
                                                 (frame.shape[1], frame.shape[0]), self.focal),
                                             camera_crop=list(current_crop),
                                             camera_reference_crop=list(reference_crop),
                                             camera_model=self.info['model'],
                                             camera_view=self.info['view'], jpeg_quality=self.quality)).encode()
                    jpeg = encoded.tobytes()
                    conn.sendall(struct.pack('!II', len(header), len(jpeg))+header+jpeg)
                    self.done.wait(max(0., .2-(time.monotonic()-started)))
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'
        finally:
            if camera:
                try:
                    if camera.started:
                        camera.stop()
                finally:
                    camera.close()
            self.done.set()

    def close(self):
        self.done.set()
        self.thread.join(2)
