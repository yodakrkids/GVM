# GVM (Generative Video Matting) 완전 설치 및 실행 가이드

## 개요

**GVM(Generative Video Matting)**은 SIGGRAPH 2025에 발표된 딥러닝 기반 비디오 매팅(Video Matting) 모델입니다. 이 문서는 설치부터 실행, 최적화까지의 전 과정을 다룹니다.

---

## 1. 사전 요구 사항

### 필수 소프트웨어
- **Git**: 소스 코드 저장소 복제용
- **Anaconda/Miniconda**: Python 가상 환경 관리
- **NVIDIA GPU**: CUDA 지원 GPU 필수
- **CUDA Toolkit**: GPU 연산을 위한 CUDA 드라이버

### 권장 시스템 사양
- **GPU**: 8GB+ VRAM (RTX 3070 이상 권장)
- **RAM**: 16GB+ (32GB 권장)
- **Storage**: 10GB+ 여유 공간

---

## 2. 설치 과정

### Step 1: 저장소 복제
```bash
git clone https://github.com/aim-uofa/GVM.git
cd GVM
```

### Step 2: Python 환경 설정
```bash
# Conda 가상환경 생성
conda create -n gvm python=3.10 -y
conda activate gvm

# HuggingFace Hub 설치 (모델 다운로드용)
pip install -U huggingface_hub
```

### Step 3: 의존성 패키지 설치
```bash
# Python 라이브러리 설치
pip install -r requirements.txt

# 프로젝트 개발 모드 설치
python setup.py develop

# av 패키지 별도 설치 (FFmpeg 의존성 해결)
conda install av -c conda-forge
```

### Step 4: 모델 가중치 다운로드
```bash
# 사전 훈련된 모델 다운로드 (약 5GB)
hf download geyongtao/gvm --local-dir data/weights
```

### Step 5: 디렉토리 구조 확인
설치 완료 후 다음과 같은 구조가 생성되어야 합니다:
```
GVM/
├── data/
│   ├── weights/
│   │   ├── vae/
│   │   │   ├── config.json
│   │   │   └── diffusion_pytorch_model.safetensors
│   │   ├── unet/
│   │   │   ├── config.json
│   │   │   └── diffusion_pytorch_model.safetensors
│   │   └── scheduler/
│   │       └── scheduler_config.json
│   └── demo_videos/
└── ... (기타 프로젝트 파일들)
```

---

## 3. 실행 방법

### 기본 실행
```bash
python demo.py \
--model_base 'data/weights/' \
--unet_base data/weights/unet \
--lora_base data/weights/unet \
--mode 'matte' \
--num_frames_per_batch 8 \
--num_interp_frames 1 \
--num_overlap_frames 1 \
--denoise_steps 1 \
--decode_chunk_size 8 \
--max_resolution 960 \
--pretrain_type 'svd' \
--data_dir 'data/Sony_Lens_Test.mp4' \
--output_dir 'output'
```

---

## 4. 파라미터 상세 설명

### 🔗 필수 파라미터

| 파라미터 | 설명 | 예시 |
|---|---|---|
| --data_dir | 입력 비디오 파일 경로 | "video.mp4" |
| --output_dir | 출력 결과 저장 디렉토리 | "output" |

### 🤖 모델 관련 파라미터

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| --model_base | 'data/weights' | 기본 모델 가중치 경로 |
| --unet_base | None | UNet 모델 경로 (기본값: model_base) |
| --lora_base | None | LoRA 가중치 경로 (선택사항) |
| --pretrain_type | 'dav' | 사전훈련 모델 타입 ('svd', 'dav') |

### 🔄 성능 조정 파라미터

| 파라미터 | 기본값 | 설명 | 메모리 영향 |
|---|---|---|---|
| --num_frames_per_batch | 32 | 배치당 프레임 수 | 높음 |
| --decode_chunk_size | 16 | VAE 디코딩 청크 크기 | 중간 |
| --denoise_steps | 1 | 디노이징 스텝 수 (1-3) | 낮음 |
| --max_resolution | 1024 | 최대 해상도 | 높음 |
| --size | 720 | 리사이즈 크기 | 중간 |

### 🎬 비디오 처리 파라미터

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| --num_interp_frames | 16 | 보간 프레임 수 |
| --num_overlap_frames | 6 | 윈도우 겹침 프레임 수 |
| --max_frames | None | 최대 처리 프레임 (전체: 생략) |

---

## 5. 실행 시나리오별 설정

### 🔋 시나리오 1: 메모리 최적화 (저사양 GPU)
**적용 상황**: 8GB 미만 VRAM, RAM 16GB 미만
```bash
python demo.py \
--model_base 'data/weights/' \
--unet_base data/weights/unet \
--lora_base data/weights/unet \
--mode 'matte' \
--num_frames_per_batch 4 \
--num_interp_frames 1 \
--num_overlap_frames 1 \
--denoise_steps 1 \
--decode_chunk_size 4 \
--max_resolution 720 \
--size 512 \
--pretrain_type 'svd' \
--data_dir 'data/Sony_Lens_Test.mp4' \
--output_dir 'output_lowmem'
```

### 🚀 시나리오 2: 고품질 처리 (고사양 GPU)
**적용 상황**: 16GB+ VRAM, RAM 32GB+
```bash
python demo.py \
--model_base 'data/weights/' \
--unet_base data/weights/unet \
--lora_base data/weights/unet \
--mode 'matte' \
--num_frames_per_batch 16 \
--num_interp_frames 2 \
--num_overlap_frames 2 \
--denoise_steps 3 \
--decode_chunk_size 16 \
--max_resolution 1280 \
--pretrain_type 'svd' \
--data_dir 'data/Sony_Lens_Test.mp4' \
--output_dir 'output_hq'
```

### ⚡ 시나리오 3: 빠른 테스트 (일부 프레임만)
**적용 상황**: 빠른 결과 확인, 파라미터 테스트
```bash
python demo.py \
--model_base 'data/weights/' \
--unet_base data/weights/unet \
--lora_base data/weights/unet \
--mode 'matte' \
--num_frames_per_batch 8 \
--max_frames 50 \
--denoise_steps 1 \
--decode_chunk_size 8 \
--max_resolution 720 \
--pretrain_type 'svd' \
--data_dir 'data/Sony_Lens_Test.mp4' \
--output_dir 'output_test'
```

### 🖼️ 시나리오 4: 이미지 시퀀스만 출력
**적용 상황**: 후처리용 개별 이미지 필요
```bash
python demo.py \
--model_base 'data/weights/' \
--unet_base data/weights/unet \
--lora_base data/weights/unet \
--mode 'matte' \
--num_frames_per_batch 8 \
--denoise_steps 1 \
--decode_chunk_size 8 \
--max_resolution 960 \
--pretrain_type 'svd' \
--output_image_seq_only \
--data_dir 'data/Sony_Lens_Test.mp4' \
--output_dir 'output_images'
```

---

## 6. 성능 최적화 가이드

### 🚨 메모리 부족 문제 해결
**증상**: `CUDA out of memory` 오류
**해결 순서**:
1. `--num_frames_per_batch` 감소 (8 → 4 → 2)
2. `--decode_chunk_size` 감소 (8 → 4)
3. `--max_resolution` 감소 (960 → 720 → 512)
4. `--max_frames` 설정으로 일부만 처리

### ⚡ 속도 향상 방법
**목표**: 처리 시간 단축
**조정 방법**:
1. `--denoise_steps` 최소화 (1 사용)
2. `--num_interp_frames` 감소
3. `--max_resolution` 감소
4. `--size` 감소

### 🎯 품질 향상 방법
**목표**: 더 정확한 매팅 결과
**조정 방법**:
1. `--denoise_steps` 증가 (2-3)
2. `--num_interp_frames` 증가
3. `--num_overlap_frames` 증가
4. `--max_resolution` 증가

---

## 7. 문제 해결 (Troubleshooting)

### ❌ 설치 단계 주요 오류

**오류 1: `diffusers.egg==info` 오류**
- **원인**: `requirements.txt` 파일 오류
- **해결**: `requirements.txt`에서 해당 라인 제거

**오류 2: `diffusers==0.35.0.dev0` 설치 실패**
- **원인**: 개발 버전이 PyPI에 없음
- **해결**: `requirements.txt`에서 `0.35.0.dev0` → `0.35.0`로 수정

**오류 3: `av` 패키지 빌드 실패**
- **원인**: FFmpeg 라이브러리 부재
- **해결**:
  ```bash
  conda install av -c conda-forge
  ```
  > **참고**: `requirements.txt`에서 `av` 라인 제거

**오류 4: `numpy` 버전 호환성 문제**
- **원인**: Python 버전과 numpy 버전 불일치
- **해결**: Python 버전에 맞는 numpy 버전 설치

### ⚠️ 실행 단계 주요 오류

**오류 1: `CUDA out of memory`**
- **해결**: 위의 [메모리 최적화](#-메모리-부족-문제-해결) 방법 적용

**오류 2: 느린 처리 속도 / 높은 CPU 사용률**
- **원인**:
  - 전체 비디오가 RAM에 로드됨
  - CPU 집약적 전처리
  - GPU 대기 시간 발생
- **해결**:
  - 배치 크기 조정
  - 해상도 낮추기
  - 프레임 수 제한 (`--max_frames`)

**오류 3: 모델 로딩 실패**
- **해결**:
  - 모델 가중치 다운로드 확인
  - 경로 설정 재확인
  - HuggingFace 토큰 설정

---

## 8. 리소스 사용량 분석

### 💾 메모리 사용 패턴

| 구성요소 | 사용량 | 최적화 방법 |
|---|---|---|
| 전체 비디오 로딩 | 매우 높음 | `max_frames` 제한 |
| 모델 가중치 | 높음 | 모델 경량화 불가 |
| 배치 처리 | 중간 | 배치 크기 조정 |
| 전처리 | 중간 | 해상도 조정 |

### 🔄 CPU vs GPU 사용률

| 작업 | 주요 처리 장치 | 최적화 포인트 |
|---|---|---|
| 비디오 디코딩 | CPU | 압축률 낮은 포맷 사용 |
| 전처리 | CPU | GPU 전처리로 전환 |
| AI 추론 | GPU | 배치 크기 증가 |
| 후처리 | CPU | 병렬 처리 |

---

## 9. 성능 벤치마크

### 🖥️ 하드웨어별 예상 성능

| GPU 모델 | VRAM | 권장 배치 | 예상 FPS | 메모리 사용 |
|---|---|---|---|---|
| RTX 4090 | 24GB | 16 | 8-12 | 20GB |
| RTX 4080 | 16GB | 12 | 6-10 | 14GB |
| RTX 4070 | 12GB | 8 | 4-8 | 10GB |
| RTX 4060 | 8GB | 4 | 2-5 | 7GB |

### 📏 해상도별 성능 영향

| 해상도 | 처리 시간 | 메모리 사용량 | 품질 |
|---|---|---|---|
| 512px | 빠름 | 낮음 | 보통 |
| 720px | 보통 | 중간 | 좋음 |
| 1024px | 느림 | 높음 | 우수 |
| 1280px | 매우 느림 | 매우 높음 | 최고 |

---

## 10. 최적 설정 추천

### 🥇 일반적인 용도 (균형잡힌 설정)
```bash
--num_frames_per_batch 8
--decode_chunk_size 8
--max_resolution 960
--denoise_steps 1
--num_interp_frames 1
```

### 🏆 프로덕션 용도 (고품질 우선)
```bash
--num_frames_per_batch 16
--decode_chunk_size 16
--max_resolution 1280
--denoise_steps 3
--num_interp_frames 2
```

### 🚀 프로토타이핑 용도 (속도 우선)
```bash
--num_frames_per_batch 4
--decode_chunk_size 4
--max_resolution 512
--denoise_steps 1
--max_frames 100
```

---

## 11. 추가 정보

### 🔗 관련 링크
- **논문**: https://arxiv.org/abs/2508.07905
- **프로젝트 페이지**: https://yongtaoge.github.io/project/gvm
- **GitHub**: https://github.com/aim-uofa/GVM
- **HuggingFace 모델**: https://huggingface.co/geyongtao/gvm

### 📝 라이선스
- **학술적 사용**: 2-clause BSD License
- **상업적 사용**: `chhshen@gmail.com` 문의

---

> 💡 **팁**: 처음 실행 시에는 메모리 최적화 설정으로 시작하여 시스템에 맞는 최적 파라미터를 찾아가는 것이 좋습니다.