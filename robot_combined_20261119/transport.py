"""Bounded JSON transport. Heartbeats never depend on inference, logs or UI."""
from collections import deque
import json
import math
import queue
import socket
import threading
import time


def validate_vision(value):
    if not isinstance(value, dict):
        raise ValueError('Invalid vision packet')
    if not isinstance(value.get('seq'), int) or value['seq'] < 0:
        raise ValueError('Invalid frame sequence')
    if not isinstance(value.get('capture_mono'), (int, float)) or not math.isfinite(value['capture_mono']):
        raise ValueError('Invalid capture timestamp')
    for key in ('width', 'height'):
        if not isinstance(value.get(key), int) or not 1 <= value[key] <= 4096:
            raise ValueError('Invalid frame dimensions')
    if 'camera_intrinsics' in value:
        model = value['camera_intrinsics']
        if (not isinstance(model, dict) or not all(isinstance(model.get(k), (int, float))
                and not isinstance(model[k], bool) and math.isfinite(model[k]) for k in ('fx','fy','cx','cy'))
                or not 0 < model['fx'] <= 100000 or not 0 < model['fy'] <= 100000
                or not 0 <= model['cx'] <= value['width'] or not 0 <= model['cy'] <= value['height']):
            raise ValueError('Invalid frame camera intrinsics')
    detections = value.get('detections')
    if not isinstance(detections, list) or len(detections) > 100:
        raise ValueError('Invalid detections')
    for obj in detections:
        if not isinstance(obj, dict) or obj.get('label') not in ('fire', 'person', 'furniture'):
            raise ValueError('Invalid label')
        box, confidence = obj.get('box'), obj.get('confidence')
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError('Invalid bounding box')
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in [*box, confidence]):
            raise ValueError('Nonfinite detection')
        if not 0 <= confidence <= 1 or not 0 <= box[0] < box[2] <= value['width'] or not 0 <= box[1] < box[3] <= value['height']:
            raise ValueError('Detection outside bounds')
    return value


class Sender:
    def __init__(self, conn, heartbeat=False):
        self.conn, self.heartbeat = conn, heartbeat
        self.done = threading.Event()
        self.lock = threading.Lock()
        self.pending = deque(maxlen=512)
        self.vision = None
        self.events = queue.Queue(maxsize=4096)
        self.dropped = 0
        self.error = None
        self.max_gap_ms = 0.
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def put(self, value):
        with self.lock:
            if value.get('command') == 'vision':
                self.vision = value
            elif value.get('command') in ('stop', 'exit'):
                self.pending.clear()
                self.vision = None
                self.pending.appendleft(value)
            else:
                if len(self.pending) == self.pending.maxlen:
                    self.dropped += 1
                self.pending.append(value)

    def note(self, value):
        try:
            self.events.put_nowait(value)
        except queue.Full:
            self.dropped += 1

    def run(self):
        next_heartbeat, last_heartbeat = 0., None
        try:
            while not self.done.is_set():
                now = time.monotonic()
                packet = None
                with self.lock:
                    if self.pending and self.pending[0].get('command') in ('stop', 'exit'):
                        packet = self.pending.popleft()
                    elif self.heartbeat and now >= next_heartbeat:
                        packet = {'command': 'heartbeat'}
                    elif self.pending:
                        packet = self.pending.popleft()
                    elif self.vision is not None:
                        packet, self.vision = self.vision, None
                if packet is None:
                    self.done.wait(.005)
                    continue
                self.conn.sendall((json.dumps(packet, allow_nan=False)+'\n').encode())
                sent = time.monotonic()
                command = packet.get('command')
                if command == 'heartbeat':
                    gap = None if last_heartbeat is None else (sent-last_heartbeat)*1000
                    self.max_gap_ms = max(self.max_gap_ms, gap or 0.)
                    self.note(dict(event='HEARTBEAT_SENT', gap_ms=gap, pc_sent_monotonic=sent))
                    last_heartbeat, next_heartbeat = sent, sent+.2
                elif command and command != 'vision':
                    self.note(dict(event='COMMAND_SENT', command=command))
                if command == 'exit':
                    break
        except (OSError, ValueError, TypeError) as exc:
            self.error = f'{type(exc).__name__}: {exc}'
            self.note(dict(event='SEND_ERROR', error=self.error))
        finally:
            self.done.set()

    def close(self, send_exit=False):
        if send_exit and not self.done.is_set():
            self.put({'command': 'exit'})
            self.thread.join(.5)
        self.done.set()
        self.thread.join(.5)


class Receiver:
    def __init__(self, conn, callback=None):
        self.conn, self.callback = conn, callback
        self.done = threading.Event()
        self.records = queue.Queue(maxsize=8192)
        self.dropped = 0
        self.error = None
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        buffer = b''
        try:
            while not self.done.is_set():
                try:
                    data = self.conn.recv(16384)
                except socket.timeout:
                    continue
                if not data:
                    break
                buffer += data
                if len(buffer) > 262144:
                    raise ValueError('Oversized JSON message')
                while b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)
                    obj = json.loads(line)
                    if not isinstance(obj, dict):
                        raise ValueError('Expected JSON object')
                    if self.callback:
                        self.callback(obj)
                    else:
                        try:
                            self.records.put_nowait(obj)
                        except queue.Full:
                            self.dropped += 1
        except (OSError, ValueError, KeyError, TypeError, queue.Full) as exc:
            self.error = f'{type(exc).__name__}: {exc}'
        finally:
            self.done.set()

    def close(self):
        self.done.set()
        self.thread.join(.5)


class PiLink:
    def __init__(self, conn):
        self.last = time.monotonic()
        self.stopped = threading.Event()
        self.stopped.set()
        self.vision = None
        self.commands = queue.Queue(maxsize=16)
        self.exited = threading.Event()
        self.receiver = Receiver(conn, self.handle)

    def handle(self, packet):
        command = packet['command']
        if command not in ('heartbeat', 'vision', 'start', 'stop', 'exit'):
            raise ValueError('Unknown command')
        if command == 'vision':
            value = validate_vision(packet['vision'])
            if self.vision is None or value['seq'] > self.vision['seq']:
                self.vision = value
            return
        self.last = time.monotonic()
        if command in ('stop', 'exit'):
            self.stopped.set()
            # Invalidate any start that has not yet been handled.
            while not self.commands.empty():
                try:
                    self.commands.get_nowait()
                except queue.Empty:
                    break
        if command == 'exit':
            self.exited.set()
        elif command != 'heartbeat':
            self.commands.put_nowait(command)

    def fresh(self):
        return not self.exited.is_set() and not self.receiver.done.is_set() and time.monotonic()-self.last < 1.
