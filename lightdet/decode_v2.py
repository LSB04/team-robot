"""NumPy/OpenCV decoder shared by student training, evaluation and deployment."""
import cv2
import numpy as np


def decode_v2(output, names, size, transform, shape, conf=.35, iou=.45, max_det=300):
    p = output[0]
    cls = p[:, 5:].argmax(-1)
    scores = p[:, 4] * p[:, 5:].max(-1)
    indices = np.flatnonzero(scores >= conf)
    if len(indices) > 3000:
        indices = indices[np.argsort(-scores[indices])[:3000]]
    p, cls, scores = p[indices], cls[indices], scores[indices]
    sx, sy, left, top = transform
    h, w = shape[:2]
    xy1, xy2 = (p[:, :2] - p[:, 2:4] / 2) * size, (p[:, :2] + p[:, 2:4] / 2) * size
    boxes = np.stack((np.clip((xy1[:, 0] - left) / sx, 0, w), np.clip((xy1[:, 1] - top) / sy, 0, h),
                      np.clip((xy2[:, 0] - left) / sx, 0, w), np.clip((xy2[:, 1] - top) / sy, 0, h)), -1)
    good = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes, scores, cls = boxes[good], scores[good], cls[good]
    result = []
    for cid in np.unique(cls):
        mask = np.flatnonzero(cls == cid)
        group = boxes[mask]
        xywh = np.concatenate((group[:, :2], group[:, 2:] - group[:, :2]), -1)
        kept = cv2.dnn.NMSBoxes(xywh.tolist(), scores[mask].tolist(), conf, iou)
        for idx in np.asarray(kept).reshape(-1):
            n = mask[int(idx)]
            result.append(dict(class_id=int(cid), label=names[int(cid)], confidence=float(scores[n]), box=boxes[n].tolist()))
    return sorted(result, key=lambda d: -d['confidence'])[:max_det]
