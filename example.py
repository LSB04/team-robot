"""python example.py image.jpg [--device cuda:0]"""
import argparse
from lightdet import TinyGrid, annotate, read_image, write_image

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('image')
    p.add_argument('--device', default='cpu', choices=['cpu', 'cuda:0'])
    p.add_argument('--output', default='outputs/example.jpg')
    a = p.parse_args()
    model = TinyGrid(device=a.device)
    frame = read_image(a.image)
    results = model(frame)
    print(results)
    write_image(a.output, annotate(frame, results, model.last_inference_ms))

if __name__ == '__main__':
    main()
