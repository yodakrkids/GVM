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

## 3. 의존성 패키지 설치

`requirements.txt` 파일을 사용하여 필요한 Python 라이브러리를 설치합니다:

```bash
pip install -r requirements.txt
```

그 후, `setup.py` 스크립트를 개발 모드로 실행하여 프로젝트를 올바르게 설정합니다:

```bash
python setup.py develop
```

## 4. 모델 가중치 다운로드

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

---

## 설치 과정 및 오류 해결 기록

이 섹션은 초기 설치 과정에서 발생했던 문제들과 해결 과정을 기록한 것입니다.

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

위의 과정을 통해 최종적으로 모든 필수 라이브러리 설치를 완료했습니다.