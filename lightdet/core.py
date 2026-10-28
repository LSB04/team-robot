from pathlib import Path
import json
import os
import time
import cv2
import numpy as np
from .runtime import DEFAULT_THREADS, configure_cpu


def read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or '.jpg', image)
    if not ok:
        raise OSError(f"Cannot encode image: {path}")
    encoded.tofile(str(path))


def letterbox(image, size):
    h, w = image.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = cv2.resize(image, (nw, nh))
    return canvas, (nw / w, nh / h, left, top)


def decode(raw, names, size, transform, shape, conf, iou):
    pred = 1 / (1 + np.exp(-np.clip(raw[0].transpose(1, 2, 0), -60, 60)))
    gh, gw, channels = pred.shape
    if channels != 5 + len(names):
        raise ValueError('Model output and metadata classes do not match')
    cls = pred[..., 5:].argmax(-1)
    score = pred[..., 4] * pred[..., 5:].max(-1)
    ys, xs = np.where(score >= conf)
    sx, sy, left, top = transform
    h, w = shape[:2]
    detections = []
    # Vectorize geometry; retain the same candidate order and classwise OpenCV NMS.
    selected = pred[ys, xs]
    cx, cy = (xs + selected[:, 0]) * size / gw, (ys + selected[:, 1]) * size / gh
    bw, bh = selected[:, 2] * size, selected[:, 3] * size
    boxes = np.stack([np.clip((cx - bw / 2 - left) / sx, 0, w),
                      np.clip((cy - bh / 2 - top) / sy, 0, h),
                      np.clip((cx + bw / 2 - left) / sx, 0, w),
                      np.clip((cy + bh / 2 - top) / sy, 0, h)], axis=1)
    for y, x, coordinates in zip(ys, xs, boxes):
        box = coordinates.tolist()
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        cid = int(cls[y, x])
        detections.append(dict(class_id=cid, label=names[cid], confidence=float(score[y, x]), box=box))
    kept = []
    for cid in sorted(set(d['class_id'] for d in detections)):
        group = [d for d in detections if d['class_id'] == cid]
        boxes = [[d['box'][0], d['box'][1], d['box'][2] - d['box'][0], d['box'][3] - d['box'][1]] for d in group]
        indices = cv2.dnn.NMSBoxes(boxes, [d['confidence'] for d in group], conf, iou)
        kept.extend(group[int(i)] for i in np.asarray(indices).reshape(-1))
    return sorted(kept, key=lambda d: d['confidence'], reverse=True)


class Detector:
    def __init__(self, model, conf=.35, iou=.45, device='auto', yolo_size=320, threads=DEFAULT_THREADS):
        self.threads = configure_cpu(threads, 'cpu')
        if device not in ('auto', 'cpu', 'intel:cpu', 'intel:gpu', 'cuda', 'cuda:0'):
            raise ValueError('Use auto, cpu, intel:cpu, intel:gpu or cuda:0')
        use_cuda = device in ('cuda', 'cuda:0')
        if yolo_size < 32 or yolo_size % 32:
            raise ValueError('--yolo-size must be positive and divisible by 32')
        self.path = Path(model)
        exported = Path(__file__).resolve().parents[1] / 'models' / 'best_openvino.xml'
        if self.path.resolve() == (exported.parents[1] / 'best.pt').resolve() and device in ('auto', 'intel:cpu', 'intel:gpu') and exported.is_file():
            self.path = exported
        if not self.path.is_file():
            raise FileNotFoundError(f'Model not found: {self.path}')
        if not 0 < conf < 1 or not 0 < iou < 1:
            raise ValueError('conf and iou must be between 0 and 1')
        self.conf, self.iou, self.device = conf, iou, 'cpu'
        self.yolo_size = yolo_size
        self.ov_backend = None
        self.cuda_backend = None
        self.backend_info = dict(backend='opencv', device='cpu', precision='FP32')
        if use_cuda and self.path.suffix.lower() != '.onnx':
            raise ValueError('CUDA inference currently supports TinyGrid .onnx only; use --model models/best.onnx')
        if self.path.suffix == '.xml':
            from .openvino_backend import OpenVINODetector
            self.ov_backend = OpenVINODetector(self.path, 'intel:cpu' if device == 'cpu' else device, self.threads)
            self.names, self.size = self.ov_backend.names, self.ov_backend.size
            self.teacher = False
            self.device = self.ov_backend.device
            self.backend_info = dict(backend='openvino', device=self.device)
            return
        if device.startswith('intel:'):
            raise ValueError('Export first: python export_openvino.py; then use --model models/best_openvino.xml')
        self.teacher = self.path.suffix.lower() == '.pt'
        if self.teacher:
            config_dir = Path(__file__).resolve().parents[1] / 'outputs' / 'ultralytics_config'
            if 'YOLO_CONFIG_DIR' not in os.environ:
                config_dir.mkdir(parents=True, exist_ok=True)
                os.environ['YOLO_CONFIG_DIR'] = str(config_dir)
            import torch
            from ultralytics import YOLO
            configure_cpu(self.threads, 'cpu', torch)
            self.net = YOLO(str(self.path))
            # Ultralytics resets CPU threads during predictor setup. Reapply after setup.
            self.net.add_callback('on_predict_start', lambda predictor: configure_cpu(self.threads, 'cpu', torch))
            self.names = [self.net.names[i] for i in range(len(self.net.names))]
            self.backend_info = dict(backend='ultralytics', device=self.device)
        else:
            metadata = json.loads(self.path.with_suffix('.json').read_text(encoding='utf-8'))
            self.architecture = metadata.get('architecture')
            if self.architecture not in ('tinygrid-v1', 'tinygrid-v2', 'tinygrid-v3'):
                raise ValueError('Only tinygrid-v1/v2/v3 ONNX exports are supported')
            self.names, self.size = metadata['names'], metadata['size']
            precision = metadata.get('precision', 'FP32')
            if precision != 'FP32' and not use_cuda:
                raise ValueError('Use --device cuda:0 for the FP16 deployment model, or select its FP32 original')
            if use_cuda:
                from .cuda_backend import CudaBackend
                self.cuda_backend = CudaBackend(self.path, self.size, len(self.names), self.threads,
                                                architecture=self.architecture, precision=precision,
                                                cuda_graph=metadata.get('cuda_graph', False))
                self.device = 'cuda:0'
                self.backend_info = self.cuda_backend.info
                print(f'TinyGrid CUDA: {self.backend_info}', flush=True)
                return
            # Buffer loading also works for Windows paths containing Korean text.
            self.net = cv2.dnn.readNetFromONNX(np.frombuffer(self.path.read_bytes(), dtype=np.uint8))
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def predict(self, frame):
        start = time.perf_counter()
        if self.ov_backend is not None:
            objects = self.ov_backend.predict(frame, self.conf, self.iou)
            return objects, (time.perf_counter() - start) * 1000
        if self.teacher:
            result = self.net.predict(frame, conf=self.conf, iou=self.iou, device=self.device,
                                      imgsz=self.yolo_size, verbose=False)[0]
            detections = [dict(class_id=int(b.cls.item()), label=self.names[int(b.cls.item())],
                               confidence=float(b.conf.item()), box=b.xyxy[0].cpu().tolist()) for b in result.boxes]
        else:
            square, transform = letterbox(frame, self.size)
            blob = cv2.dnn.blobFromImage(square, 1 / 255., swapRB=True)
            if self.cuda_backend is not None:
                raw = self.cuda_backend.forward(blob)
            else:
                self.net.setInput(blob)
                raw = self.net.forward()
            if self.architecture in ('tinygrid-v2', 'tinygrid-v3'):
                from .decode_v2 import decode_v2
                detections = decode_v2(raw, self.names, self.size, transform, frame.shape, self.conf, self.iou)
            else:
                detections = decode(raw, self.names, self.size, transform, frame.shape, self.conf, self.iou)
        return detections, (time.perf_counter() - start) * 1000


def annotate(frame, detections, elapsed_ms):
    result = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = map(round, d['box'])
        cv2.rectangle(result, (x1, y1), (x2, y2), (70, 220, 70), 2)
        # OpenCV's built-in font supports ASCII; full Unicode labels remain in JSON.
        label = d['label'] if d['label'].isascii() else f"class_{d['class_id']}"
        cv2.putText(result, f"{label} {d['confidence']:.2f}", (x1, max(18, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (70, 220, 70), 2)
    cv2.putText(result, f"{len(detections)} objects | inference {elapsed_ms:.1f} ms", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 230, 255), 2)
    return result


def common_args(parser):
    parser.add_argument('--model', default=str(Path(__file__).resolve().parent / 'models' / 'tinygrid480_fp32.onnx'), help='Existing YOLO .pt or trained TinyGrid .onnx')
    parser.add_argument('--conf', type=float, default=.35)
    parser.add_argument('--iou', type=float, default=.45)
    parser.add_argument('--device', default='cpu', choices=['auto', 'cpu', 'intel:cpu', 'intel:gpu', 'cuda', 'cuda:0'], help='cuda:0: TinyGrid ONNX on NVIDIA; auto retains existing CPU/Intel selection')
    parser.add_argument('--threads', type=int, default=DEFAULT_THREADS, help='CPU threads (default: up to 4)')
    parser.add_argument('--yolo-size', type=int, default=320, help='YOLO input size; TinyGrid uses exported size')
    parser.add_argument('--show', action='store_true', help='Open preview; press Q to close')


def make_detector(args):
    return Detector(args.model, args.conf, args.iou, args.device, args.yolo_size, args.threads)
