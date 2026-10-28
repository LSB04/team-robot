"""1. Detect objects in one photo; save boxes and JSON."""
import argparse
import json
from pathlib import Path
import cv2
from lightdet.core import common_args, make_detector, read_image, write_image, annotate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    common_args(p)
    p.add_argument('--input', required=True)
    p.add_argument('--output', default='outputs/image_result.jpg')
    a = p.parse_args()
    if Path(a.input).resolve() == Path(a.output).resolve():
        p.error('Input and output must be different')
    detector = make_detector(a)
    frame = read_image(a.input)
    objects, ms = detector.predict(frame)
    marked = annotate(frame, objects, ms)
    write_image(a.output, marked)
    report = dict(input=a.input, model=str(detector.path), device=detector.device,
                  backend=detector.backend_info, inference_ms=ms, detections=objects)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    Path(a.output).with_suffix('.json').write_text(text, encoding='utf-8')
    print(text)
    if a.show:
        cv2.imshow('Image detection', marked)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
