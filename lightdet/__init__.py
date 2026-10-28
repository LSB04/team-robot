"""CPU-first TinyGrid object detection, with optional NVIDIA CUDA."""
from pathlib import Path
from .core import Detector, read_image, annotate, write_image

class TinyGrid(Detector):
    """Load bundled weights automatically. Call with a path or OpenCV BGR frame."""
    def __init__(self, model=None, device='cpu', conf=0.35, iou=0.45, threads=4):
        if model is None:
            model = Path(__file__).resolve().parent / 'models' / 'tinygrid480_fp32.onnx'
        super().__init__(model, conf=conf, iou=iou, device=device, threads=threads)
        self.last_inference_ms = None

    def predict(self, image):
        if isinstance(image, (str, Path)):
            image = read_image(image)
        detections, ms = super().predict(image)
        self.last_inference_ms = ms
        return detections, ms

    def __call__(self, image):
        return self.predict(image)[0]

__all__ = ['TinyGrid', 'Detector', 'read_image', 'annotate', 'write_image']
