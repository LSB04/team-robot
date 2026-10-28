"""Conservative CPU settings for the i7-1360P laptop."""
import os
import cv2

DEFAULT_THREADS = min(4, os.cpu_count() or 1)


def configure_cpu(threads=DEFAULT_THREADS, device='cpu', torch_module=None):
    if str(device).lower() != 'cpu':
        raise ValueError('This laptop profile uses CPU. Use --device cpu (Intel Iris Xe is not CUDA).')
    if not isinstance(threads, int) or threads < 1:
        raise ValueError('--threads must be a positive integer')
    threads = min(threads, os.cpu_count() or 1)
    cv2.setNumThreads(threads)
    cv2.ocl.setUseOpenCL(False)
    if torch_module is not None:
        torch_module.set_num_threads(threads)
    return threads
