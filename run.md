# GVM (Generative Video Matting) 실행 가이드

## 파라미터 설명

### 필수 파라미터
- `--data_dir`: 입력 비디오 파일 경로 (mp4, mkv, gif 지원)
- `--output_dir`: 결과물을 저장할 출력 디렉토리

### 모델 관련 파라미터
- `--model_base`: 기본 모델 가중치 경로 (기본값: 'data/weights')
- `--unet_base`: UNet 모델 가중치 경로 (기본값: model_base와 동일)
- `--lora_base`: LoRA 가중치 경로 (선택사항)
- `--pretrain_type`: 사전 훈련 모델 타입 ('svd' 또는 'dav', 기본값: 'dav')

### 추론 성능 파라미터
- `--num_frames_per_batch`: 배치당 처리할 프레임 수 (기본값: 32, 메모리에 따라 조정)
- `--decode_chunk_size`: VAE 디코딩 청크 크기 (기본값: 16, 메모리에 따라 조정)
- `--denoise_steps`: 디노이징 스텝 수 (기본값: 1, 1-3 권장)
- `--max_resolution`: 최대 해상도 (기본값: 1024, 낮출수록 빠름)
- `--size`: 리사이즈 크기 (기본값: 720)

### 비디오 처리 파라미터
- `--num_interp_frames`: 보간 프레임 수 (기본값: 16)
- `--num_overlap_frames`: 윈도우 간 겹치는 프레임 수 (기본값: 6)
- `--max_frames`: 처리할 최대 프레임 수 (전체 비디오 처리 시 생략)

### 기타 파라미터
- `--mode`: 추론 모드 (기본값: 'matte')
- `--noise_type`: 노이즈 타입 ('gaussian' 또는 'zeros', 기본값: 'zeros')
- `--seed`: 랜덤 시드 (재현 가능한 결과를 위해 설정)
- `--output_image_seq_only`: 이미지 시퀀스만 출력 (비디오 파일 생성 안함)

---

## 실행 유즈케이스

### 1. 기본 실행 (원본)
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

### 2. 메모리 최적화 (저사양 GPU)
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

### 3. 고품질 처리 (고사양 GPU)
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

### 4. 빠른 테스트 (일부 프레임만)
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

### 5. 이미지 시퀀스만 출력
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

## 성능 튜닝 가이드

### 메모리 부족 시 조정할 파라미터:
1. `--num_frames_per_batch` 감소 (8 → 4 → 2)
2. `--decode_chunk_size` 감소 (8 → 4)
3. `--max_resolution` 감소 (960 → 720 → 512)
4. `--max_frames` 설정하여 일부만 처리

### 속도 향상을 위한 조정:
1. `--denoise_steps` 최소화 (1 사용)
2. `--num_interp_frames` 감소
3. `--max_resolution` 감소
4. `--size` 감소

### 품질 향상을 위한 조정:
1. `--denoise_steps` 증가 (2-3)
2. `--num_interp_frames` 증가
3. `--num_overlap_frames` 증가
4. `--max_resolution` 증가