"""2. Detect objects in a video; save annotated MP4 and per-frame JSONL."""
import argparse
import json
import math
from pathlib import Path
import cv2
from lightdet.core import common_args, make_detector, annotate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    common_args(p)
    p.add_argument('--input', required=True)
    p.add_argument('--output', default='outputs/video_result.mp4')
    a = p.parse_args()
    output = Path(a.output)
    if Path(a.input).resolve() == output.resolve():
        p.error('Input and output must be different')
    detector = make_detector(a)
    cap = cv2.VideoCapture(a.input)
    writer = None
    count = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not cap.isOpened():
            raise OSError(f'Cannot open video: {a.input}')
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            fps = 25.
        with output.with_suffix('.jsonl').open('w', encoding='utf-8') as log:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if writer is None:
                    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                             (frame.shape[1], frame.shape[0]))
                    if not writer.isOpened():
                        raise OSError(f'Cannot create video: {output}')
                objects, ms = detector.predict(frame)
                marked = annotate(frame, objects, ms)
                writer.write(marked)
                log.write(json.dumps(dict(frame=count, seconds=count / fps, inference_ms=ms,
                                          device=detector.device, model=str(detector.path),
                                          backend=detector.backend_info,
                                          detections=objects), ensure_ascii=False) + '\n')
                count += 1
                if a.show:
                    cv2.imshow('Video detection', marked)
                    if cv2.waitKey(1) & 255 == ord('q'):
                        break
        if count == 0:
            raise ValueError('Video contains no readable frames')
        print(f'Saved {count} frames: {output}')
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
