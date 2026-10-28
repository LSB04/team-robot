# TinyGrid 소방로봇 인식 라이브러리

**모델을 불러 사진 또는 영상 프레임을 넣으면 탐지 결과를 받는 Python 라이브러리**입니다.
CPU가 기본이고 NVIDIA CUDA는 선택할 수 있습니다. IP·네트워크·로봇 제어 설정은 하지 않습니다.

## 연구실 PC 설치

1. GitHub 로그인 후 **Code → Download ZIP**으로 내려받아 압축을 풉니다. 비공개 저장소이므로 접근 권한이 필요합니다.
2. Python **3.12 64비트**를 준비합니다.
3. 압축을 푼 폴더에서 PowerShell 또는 VS Code 터미널을 엽니다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install .
```

약 20MB의 모델 가중치가 패키지에 포함됩니다. 학습하거나 모델을 별도로 다운로드할 필요가 없습니다.
Git 사용자는 `git clone https://github.com/LSB04/team-robot.git`으로 받아도 됩니다.

## YOLO처럼 불러서 사용

```python
from lightdet import TinyGrid

model = TinyGrid()  # CPU 기본, 가중치 자동 로드
results = model("photo.jpg")
for obj in results:
    print(obj["label"], obj["confidence"], obj["box"])
```

출력은 딕셔너리 리스트입니다. Ultralytics의 Results 객체와 동일한 API는 아닙니다.

```python
{"class_id": 0, "label": "fire", "confidence": 0.82, "box": [10, 20, 180, 240]}
```

위 값은 형식 예시입니다. 클래스는 **0=fire / 1=person / 2=furniture**.
box는 입력 이미지 픽셀 기준 `[x1, y1, x2, y2]`입니다.
OpenCV BGR 프레임도 바로 넣을 수 있으므로 기존 영상 수신 코드에서 `results = model(frame)`으로 사용하세요.
RGB 배열을 사용한다면 먼저 BGR로 변환해야 합니다.

## 사진·영상 실행 예제

```powershell
.\.venv\Scripts\python.exe example.py "C:\사진폴더\사진.jpg"
.\.venv\Scripts\python.exe detect_image.py --input "C:\사진폴더\사진.jpg" --show
.\.venv\Scripts\python.exe detect_video.py --input "C:\영상폴더\영상.mp4" --show
```

입력 경로를 실제 경로로 바꾸세요. 사진/영상 결과와 JSON/JSONL은 `outputs`에 저장됩니다.
영상에서 직접 사용하려면:

```python
import cv2
from lightdet import TinyGrid

model = TinyGrid(conf=0.35)
cap = cv2.VideoCapture("video.mp4")
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        results = model(frame)
        print(results)
finally:
    cap.release()
```

영상 파일의 프레임을 순차 분석하는 예제이며, 추적·시간축 투표·로봇 지령은 포함하지 않습니다.
추론 시간도 필요하면 `results, elapsed_ms = model.predict(frame)`을 사용하세요.

## GPU 가속을 쓰려면

CPU 환경과 별도 환경에 설치하는 예시입니다. NVIDIA GPU와 호환 드라이버가 필요합니다.

```powershell
py -3.12 -m venv .venv-cuda
.\.venv-cuda\Scripts\python.exe -m pip install ".[cuda]"
.\.venv-cuda\Scripts\python.exe example.py "C:\사진폴더\사진.jpg" --device cuda:0
```

Python에서는 `model = TinyGrid(device="cuda:0")`로 바꾸면 됩니다.
CUDA를 명시했는데 사용할 수 없으면 오류를 표시하며 자동으로 CPU로 바꾸지 않습니다.
Intel Iris Xe는 CUDA 대상이 아니며 이 패키지는 Intel 노트북에서도 CPU로 사용할 수 있습니다.

## 들어 있는 모델과 검증 범위

- **TinyGrid480 FP32 기존 안정 배포본**, 약 504만 파라미터. 최신 실험 후보가 전반적으로 개선되지 않아 유지한 모델입니다.
- 기존 test361장 측정 AP50 **64.508%**, AP50:95 **29.095%**. YOLO 비교본보다 높다는 의미는 아닙니다.
- 위 수치는 재사용된 과거 평가셋 결과이며 실제 Pi 시연 정확도나 현재 PC의 속도를 보장하지 않습니다.
- 기존 모델 CPU/CUDA 실행 검증 이력이 있습니다. 이번 전달본의 실행 검사 결과는 `release_verification.json`을 참조하세요.
- 새 불꽃 일러스트 1,000장은 **미합병·미학습** 상태입니다. 추가 데이터 합병·비교 해석·학습은 진행하지 않았습니다.
- ONNX/메타데이터 원본을 그대로 포함했고 `manifest.json`에 SHA256을 기록했습니다.
- 현재 비공개 연구용 전달본입니다. 학습자료와 제3자 권리를 확인한 공개 배포 라이선스를 부여하는 것은 아닙니다.

## 파일 안내

- `lightdet/`: 추론 라이브러리와 가중치
- `example.py`: 가장 간단한 사진 실행 예제
- `detect_image.py`, `detect_video.py`: 결과 파일 저장용 실행 코드
- `pyproject.toml`: pip 설치 설정
- `release_verification.json`: 이 배포본의 실제 검증 기록
