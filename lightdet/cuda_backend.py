"""Explicit TinyGrid ONNX CUDA inference; never silently substitute CPU."""
from pathlib import Path
import os
import sysconfig
import threading
import numpy as np

_DLL_HANDLES = []
_DLL_DIRECTORIES = set()


def _register_nvidia_dlls():
    if os.name != 'nt':
        return
    # cuDNN can load extra engine DLLs lazily, beyond ORT's preload list.
    root = Path(sysconfig.get_path('purelib')) / 'nvidia'
    for folder in sorted(root.glob('*/bin')):
        value = str(folder.resolve())
        if value not in _DLL_DIRECTORIES:
            _DLL_HANDLES.append(os.add_dll_directory(value))
            os.environ['PATH'] = value + os.pathsep + os.environ.get('PATH', '')
            _DLL_DIRECTORIES.add(value)


class FixedCudaRunner:
    """Stable CUDA input/output addresses for graph replay; serialize buffer updates."""
    def __init__(self, session):
        import onnxruntime as ort
        inputs, outputs = session.get_inputs(), session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1 or any(v.type != 'tensor(float)' for v in inputs + outputs):
            raise ValueError('CUDA graph runner requires one FP32 input/output')
        if any(not isinstance(n, int) or n <= 0 for v in inputs + outputs for n in v.shape):
            raise ValueError('CUDA graph runner requires fixed shapes')
        self.session, self.shape = session, inputs[0].shape
        self.input = ort.OrtValue.ortvalue_from_shape_and_type(inputs[0].shape, np.float32, 'cuda', 0)
        self.output = ort.OrtValue.ortvalue_from_shape_and_type(outputs[0].shape, np.float32, 'cuda', 0)
        self.binding = session.io_binding()
        self.binding.bind_ortvalue_input(inputs[0].name, self.input)
        self.binding.bind_ortvalue_output(outputs[0].name, self.output)
        self.lock = threading.Lock()

    def forward(self, blob):
        if list(blob.shape) != self.shape or blob.dtype != np.float32:
            raise ValueError('CUDA graph input shape/type changed')
        with self.lock:
            self.input.update_inplace(np.ascontiguousarray(blob))
            self.session.run_with_iobinding(self.binding)
            self.binding.synchronize_outputs()
            return self.output.numpy()


class CudaBackend:
    def __init__(self, model, size, classes, threads=4, profile_prefix=None, architecture='tinygrid-v1', precision='FP32', cuda_graph=False):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError('CUDA inference requires requirements-cuda-infer.txt in .venv-cuda-infer') from exc
        # Load pip-installed NVIDIA DLLs rather than relying on a system Toolkit.
        _register_nvidia_dlls()
        if hasattr(ort, 'preload_dlls'):
            ort.preload_dlls(directory='')
        if 'CUDAExecutionProvider' not in ort.get_available_providers():
            raise RuntimeError('ONNX Runtime CUDAExecutionProvider is unavailable')
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
        if profile_prefix:
            options.enable_profiling = True
            options.profile_file_prefix = str(profile_prefix)
        try:
            self.session = ort.InferenceSession(
                Path(model).read_bytes(), sess_options=options,
                providers=[('CUDAExecutionProvider', {'device_id': 0,
                            'cudnn_conv_algo_search': 'HEURISTIC',
                            # ORT 1.23 disables all cuDNN tensor-core plans when this
                            # is 0, including FP16. Allow them for the half model.
                            'use_tf32': int(precision.startswith('FP16')),
                            'enable_cuda_graph': int(cuda_graph)})])
        except Exception as exc:
            raise RuntimeError('CUDA session failed. Check GPU, driver and CUDA/cuDNN DLLs; CPU fallback is disabled.') from exc
        self.session.disable_fallback()
        if self.session.get_providers()[0] != 'CUDAExecutionProvider':
            raise RuntimeError('CUDA request did not activate CUDAExecutionProvider')
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (len(inputs) != 1 or inputs[0].type != 'tensor(float)'
                or inputs[0].shape != [1, 3, size, size]):
            raise ValueError('TinyGrid CUDA requires a fixed FP32 [1,3,size,size] ONNX input matching metadata')
        if architecture not in ('tinygrid-v1', 'tinygrid-v2', 'tinygrid-v3'):
            raise ValueError('Unsupported TinyGrid architecture')
        anchors_per_location = 1 if architecture == 'tinygrid-v3' else 3
        expected = ([1, 5 + classes, size // 16, size // 16] if architecture == 'tinygrid-v1'
                    else [1, anchors_per_location * sum((size // stride) ** 2 for stride in (8, 16, 32)), 5 + classes])
        if len(outputs) != 1 or outputs[0].shape != expected:
            raise ValueError('TinyGrid output shape does not match metadata')
        self.input_name, self.output_name = inputs[0].name, outputs[0].name
        self.graph_runner = FixedCudaRunner(self.session) if cuda_graph else None
        self.info = dict(backend='onnxruntime', version=ort.__version__, device='cuda:0',
                         providers=self.session.get_providers(), precision=precision,
                         use_tf32_option=int(precision.startswith('FP16')),
                         cuda_graph=cuda_graph,
                         cpu_node_fallback=False)

    def forward(self, blob):
        if self.graph_runner is not None:
            return self.graph_runner.forward(blob)
        # run() returns host output, so its timing includes completed device work and copies.
        return self.session.run([self.output_name], {self.input_name: blob})[0]
