"""OpenVINO runtime for fixed-shape YOLOv8 detect exports (no embedded NMS)."""
import json
from pathlib import Path
import warnings
import cv2
import numpy as np


class OpenVINODetector:
    def __init__(self, path, device='auto', threads=4):
        import openvino as ov
        path = Path(path)
        meta = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        if meta.get('architecture') != 'yolov8-openvino-v1':
            raise ValueError('Use export_openvino.py to generate this model and metadata')
        self.names, self.size = meta['names'], meta['size']
        core = ov.Core()
        gpu = next((d for d in core.available_devices if d.startswith('GPU')), None)
        requested = {'auto': gpu or 'CPU', 'intel:gpu': gpu, 'intel:cpu': 'CPU'}[device]
        if requested is None:
            raise RuntimeError(f'Intel GPU unavailable: {core.available_devices}. Check Intel graphics driver or use --device intel:cpu')
        if device == 'auto' and not gpu:
            warnings.warn('Intel GPU not found; using OpenVINO CPU')
        model = core.read_model(str(path))
        def compile_for(target):
            options = {'PERFORMANCE_HINT': 'LATENCY'}
            if target == 'CPU':
                options['INFERENCE_NUM_THREADS'] = threads
            return core.compile_model(model, target, options)
        try:
            self.compiled = compile_for(requested)
        except RuntimeError as e:
            if device != 'auto' or requested == 'CPU':
                raise
            warnings.warn(f'Intel GPU compilation failed, using OpenVINO CPU: {e}')
            requested = 'CPU'
            self.compiled = compile_for(requested)
        self.device = requested
        self.execution_devices = list(self.compiled.get_property('EXECUTION_DEVICES'))
        self.request = self.compiled.create_infer_request()
        print(f'OpenVINO: {core.get_property(requested, "FULL_DEVICE_NAME")} | execution={self.execution_devices} | size={self.size}', flush=True)

    def predict(self, frame, conf, iou):
        from .core import letterbox
        square, (sx, sy, left, top) = letterbox(frame, self.size)
        blob = cv2.dnn.blobFromImage(square, 1 / 255., swapRB=True)
        outputs = self.request.infer({0: blob})
        raw = np.asarray(next(iter(outputs.values())))[0]
        if raw.shape[0] != 4 + len(self.names):
            raise ValueError(f'Unexpected YOLOv8 output: {raw.shape}')
        pred = raw.T
        labels = pred[:, 4:].argmax(1)
        scores = pred[:, 4:].max(1)
        mask = scores >= conf
        pred, labels, scores = pred[mask], labels[mask], scores[mask]
        boxes = pred[:, :4].copy()
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / sx
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / sy
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, frame.shape[1])
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, frame.shape[0])
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes, labels, scores = boxes[valid], labels[valid], scores[valid]
        result = []
        for cid in np.unique(labels):
            indices = np.flatnonzero(labels == cid)
            xywh = boxes[indices].copy()
            xywh[:, 2:] -= xywh[:, :2]
            keep = cv2.dnn.NMSBoxes(xywh.tolist(), scores[indices].tolist(), conf, iou)
            for k in np.asarray(keep).reshape(-1):
                j = indices[int(k)]
                result.append(dict(class_id=int(cid), label=self.names[int(cid)], confidence=float(scores[j]), box=boxes[j].tolist()))
        return sorted(result, key=lambda d: -d['confidence'])[:300]
