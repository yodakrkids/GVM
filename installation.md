# GVM (Generative Video Matting) 설치 안내서

이 문서는 `demo.py` 실행을 위한 환경 설정 방법을 안내합니다.

## 1. 사전 요구 사항

- **Git:** 소스 코드 저장소를 복제하기 위해 필요합니다.
- **Conda:** 독립된 Python 환경을 생성하기 위해 권장됩니다.
- **NVIDIA GPU:** 코드가 `cuda` 사용을 전제로 하므로 CUDA 지원 GPU가 필요합니다.
- **huggingface-cli:** 모델 가중치를 다운로드하기 위해 필요합니다. pip으로 설치할 수 있습니다:
  ```bash
  pip install -U huggingface_hub
  ```

## 2. 환경 설정

먼저 Git을 사용하여 저장소를 복제합니다:

```bash
git clone https://github.com/aim-uofa/GVM.git
cd GVM
```

다음으로, Conda 가상 환경을 생성하고 활성화합니다. 프로젝트 원본은 Python 3.10을 권장했습니다.

```bash
conda create -n gvm python=3.10 -y
conda activate gvm
```

## 3. 의존성 패키지 설치 (권장 방법)

**⚠️ 중요:** `requirements.txt`를 직접 사용하면 NumPy 2.x 호환성 문제가 발생할 수 있습니다. 아래의 단계별 설치를 권장합니다.

### Step 1: PyTorch CUDA 버전 먼저 설치
```bash
# CUDA 지원 PyTorch 설치 (NumPy 1.x와 호환)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 2: OpenCV 및 호환 NumPy 설치
```bash
# OpenCV 안정 버전 설치
pip install opencv-python==4.8.1.78

# NumPy 호환성 확보 (OpenCV와 PyTorch 호환)
pip install "numpy<2"
```

### Step 3: ML 관련 패키지 설치
```bash
pip install diffusers==0.35.0 transformers accelerate peft
```

### Step 4: 기타 필수 패키지 설치
```bash
pip install easydict matplotlib PIMS imageio av
```

### Step 5: 프로젝트 설정 (선택사항)
```bash
# setup.py에 인코딩 문제가 있을 수 있으므로 선택사항
# python setup.py develop
```

## 4. GPU 가속을 위한 PyTorch 설치

**CUDA 지원 PyTorch 설치 (RTX 3090 등 NVIDIA GPU 사용자):**

먼저 기존 CPU 버전 PyTorch를 제거하고 CUDA 지원 버전을 설치합니다:

```bash
# 기존 PyTorch 제거
pip uninstall torch torchvision -y

# CUDA 12.1 지원 PyTorch 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

설치 후 GPU 인식 확인:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

## 5. 필수 시스템 패키지 설치

비디오 처리와 컴퓨터 비전을 위한 시스템 패키지들을 설치합니다:

```bash
# FFmpeg, OpenCV, av 패키지 설치
conda install ffmpeg av opencv -c conda-forge -y

# OpenCV Python 바인딩 설치
pip install opencv-python

# 호환성 문제 해결을 위한 Pillow 버전 조정
pip uninstall pillow -y
pip install pillow==10.4.0
```

## 6. 모델 가중치 다운로드

프로젝트 실행을 위해 사전 훈련된 모델 가중치가 필요합니다. `huggingface-cli`를 사용하여 `data/weights` 디렉터리에 다운로드합니다.

```bash
hf download geyongtao/gvm --local-dir data/weights
```

다운로드 후 디렉터리 구조는 다음과 같아야 합니다:

```
GVM/
|-- data/
|   |-- weights/
|       |-- vae/
|       |   |-- config.json
|       |   |-- diffusion_pytorch_model.safetensors
|       |-- unet/
|       |   |-- config.json
|       |   |-- diffusion_pytorch_model.safetensors
|       |-- scheduler/
|           |-- scheduler_config.json
|-- ... (다른 프로젝트 파일)
```

## 7. 설치 확인 및 테스트

모든 설치가 완료되면 다음 명령어로 테스트 실행이 가능합니다:

```bash
# 빠른 테스트 (60프레임만 처리)
python demo.py --model_base data/weights --unet_base data/weights/unet --lora_base data/weights/unet --mode matte --num_frames_per_batch 16 --max_frames 60 --denoise_steps 1 --decode_chunk_size 16 --max_resolution 1024 --pretrain_type svd --data_dir data/Sony_Lens_Test.mp4 --output_dir output_test
```

성공적으로 실행되면 `output_test` 폴더에 결과 비디오가 생성됩니다.

---

## 설치 과정 및 오류 해결 기록

이 섹션은 설치 과정에서 발생했던 문제들과 해결 과정을 기록한 것입니다.

### 1. `requirements.txt` 파일 오류

- **문제:** `pip install -r requirements.txt` 실행 시 `Invalid requirement: 'diffusers.egg==info'` 오류 발생.
- **해결:** `requirements.txt` 파일에서 해당 라인을 제거했습니다.

### 2. `diffusers` 개발 버전 설치 불가

- **문제:** `No matching distribution found for diffusers==0.35.0.dev0` 오류 발생. PyPI(Python Package Index)에서 해당 개발 버전을 찾을 수 없었습니다.
- **해결:** `requirements.txt` 파일의 `diffusers` 버전을 `0.35.0.dev0`에서 `0.35.0`으로 수정했습니다.

### 3. `numpy`와 Python 버전 비호환성

- **문제:** `numpy==2.3.2` 버전이 Python 3.11 이상을 요구하여, 초기 설정된 Python 3.10 환경과 충돌했습니다.
- **분석:** 이 과정에서 실제 환경의 Python 버전이 권장사항과 다른 3.13임을 파악했습니다. 모든 문제의 근본적인 원인이었습니다.
- **해결:**
    1. `numpy` 버전을 Python 3.13과 호환되는 `2.3.2`로 `requirements.txt`에 유지하기로 결정했습니다.
    2. 이전 단계에서 임시로 `1.26.4`로 낮췄던 `numpy` 버전을 다시 `2.3.2`로 복원했습니다.

### 4. `av` 패키지 빌드 실패

- **문제:** `av` 패키지 설치 중 C++ 링커(Linker)가 `avformat.lib` 파일을 찾지 못하는 `LNK1181` 오류가 발생했습니다.
- **원인:** `av` 패키지는 FFmpeg 라이브러리에 대한 Python 바인딩으로, 빌드를 위해서는 시스템에 FFmpeg 개발 라이브러리가 설치되어 있어야 합니다.
- **해결:**
    1. `conda`를 통해 FFmpeg 라이브러리가 포함된 `av` 패키지를 설치하여 빌드 문제를 우회했습니다. (`conda install av -c conda-forge`)
    2. `pip`가 `conda`로 설치된 `av`를 무시하고 재설치를 시도하는 문제를 막기 위해, `requirements.txt`에서 `av==11.0.0` 라인을 완전히 제거했습니다.

### 5. 최종 `numba` 의존성 경고

- **상황:** 모든 패키지 설치 완료 후, `numba 0.61.2 requires numpy<2.3`, 하지만 `numpy 2.3.2`가 설치되어 호환되지 않는다는 경고가 출력되었습니다.
- **결론:** `numba`는 이 프로젝트의 직접적인 의존성이 아니므로, `demo.py` 실행에 영향을 주지 않을 가능성이 높습니다. 따라서 이 경고는 현재로서는 무시하고 진행하기로 결정했습니다.

### 6. 실제 실행 중 발견된 추가 의존성 문제들 (2025년 9월 해결)

**6.1 OpenCV (`cv2`) 모듈 누락:**
- **문제:** `ModuleNotFoundError: No module named 'cv2'`
- **해결:** `pip install opencv-python` 실행

**6.2 PIL/Pillow DLL 로딩 오류:**
- **문제:** `ImportError: DLL load failed while importing _imaging`
- **원인:** Pillow 버전 충돌 (11.3.0 버전에서 DLL 호환성 문제)
- **해결:** 안정된 버전으로 다운그레이드
  ```bash
  pip uninstall pillow -y
  pip install pillow==10.4.0
  ```

**6.3 FFmpeg 및 비디오 처리 의존성:**
- **문제:** 비디오 읽기/쓰기 관련 오류
- **해결:** conda를 통해 FFmpeg 생태계 패키지 일괄 설치
  ```bash
  conda install ffmpeg av opencv -c conda-forge -y
  ```

**6.4 CUDA PyTorch 성능 최적화:**
- **목적:** RTX 3090 GPU 최대 활용을 위한 CUDA 지원 PyTorch 설치
- **방법:** CPU 버전 제거 후 CUDA 12.1 버전 설치
  ```bash
  pip uninstall torch torchvision -y
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```
- **결과:** CPU 대비 10-50배 속도 향상 달성

### 7. 최신 발견된 문제들 (2024년 9월 업데이트)

**7.1 NumPy 2.x 호환성 문제:**
- **문제:** `AttributeError: _ARRAY_API not found`, `numpy.core.multiarray failed to import`
- **원인:** OpenCV 4.8.1이 NumPy 2.x와 호환되지 않음
- **해결:** NumPy를 1.x 버전으로 다운그레이드
  ```bash
  pip install "numpy<2"
  ```

**7.2 환경 손상 시 완전 재구성:**
- **문제:** pip 자체가 작동하지 않는 환경 손상
- **해결:** 환경 완전 삭제 후 재생성
  ```bash
  conda env remove -n gvm -y
  conda create -n gvm python=3.10 -y
  ```

**7.3 av 패키지 누락:**
- **문제:** `ModuleNotFoundError: No module named 'av'`
- **해결:** av 패키지 개별 설치
  ```bash
  pip install av
  ```

**7.4 권장 설치 순서:**
1. PyTorch CUDA 버전 (NumPy 1.x 포함)
2. OpenCV 안정 버전
3. ML 패키지 (diffusers, transformers 등)
4. 기타 유틸리티 패키지

위의 과정을 통해 최종적으로 모든 필수 라이브러리 설치와 GPU 가속 환경 구성을 완료했습니다.