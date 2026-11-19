"""Optional ONNX Runtime CPU inference. No PyTorch or CUDA dependency."""
from pathlib import Path
import numpy as np


class OrtCPUBackend:
    def __init__(self, model, size, classes, threads=4, architecture='tinygrid-v2'):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError('Install requirements-ort-cpu.txt for --device ort:cpu') from exc
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        # Idle sessions must not compete with another model or the video receiver.
        options.add_session_config_entry('session.intra_op.allow_spinning', '0')
        options.add_session_config_entry('session.inter_op.allow_spinning', '0')
        self.session = ort.InferenceSession(Path(model).read_bytes(), sess_options=options,
                                           providers=['CPUExecutionProvider'])
        self.session.disable_fallback()
        if self.session.get_providers() != ['CPUExecutionProvider']:
            raise RuntimeError('Explicit CPU provider was not honored')
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        self.shape = [1, 3, size, size]
        anchors = 1 if architecture == 'tinygrid-v3' else 3
        expected = ([1, 5 + classes, size // 16, size // 16] if architecture == 'tinygrid-v1'
                    else [1, anchors * sum((size // s) ** 2 for s in (8, 16, 32)), 5 + classes])
        if architecture not in ('tinygrid-v1', 'tinygrid-v2', 'tinygrid-v3'):
            raise ValueError('Unsupported TinyGrid architecture')
        if len(inputs) != 1 or inputs[0].type != 'tensor(float)' or inputs[0].shape != self.shape:
            raise ValueError('CPU ONNX input does not match FP32 metadata')
        if len(outputs) != 1 or outputs[0].type != 'tensor(float)' or outputs[0].shape != expected:
            raise ValueError('CPU ONNX output does not match metadata')
        self.input_name, self.output_name = inputs[0].name, outputs[0].name
        self.info = dict(backend='onnxruntime', device='cpu', precision='FP32',
                         version=ort.__version__, providers=self.session.get_providers(),
                         threads=threads, allow_spinning=False)

    def forward(self, blob):
        if blob.dtype != np.float32 or list(blob.shape) != self.shape:
            raise ValueError('CPU ONNX input shape/type changed')
        return self.session.run([self.output_name], {self.input_name: np.ascontiguousarray(blob)})[0]
