# Claude Code 설정 파일

이 파일은 Claude Code가 GVM 프로젝트에서 개발 작업을 수행하는 데 필요한 설정과 컨텍스트 정보를 포함합니다.

## 프로젝트 개요
GVM(Generative Video Model) 프로젝트는 비디오 생성을 위한 머신러닝 파이프라인입니다.

## 주요 디렉토리 구조
- `gvm/` - 메인 모듈
  - `models/` - 모델 정의
  - `pipelines/` - 추론 파이프라인
  - `utils/` - 유틸리티 함수
- `demo.py` - 데모 실행 스크립트

## 개발 환경
- Python 환경 관리: Conda 사용
- 주요 의존성: PyTorch, transformers, 기타 ML 라이브러리 (`requirements.txt` 참조)

## 개발 시 유의사항
1. 파일 수정 시 기존 코드 스타일을 따를 것
2. 로깅은 기존 로깅 시스템 활용
3. 모델 파일은 `.gitignore`에 포함되어 있으므로 커밋하지 않음
4. `demo.py` 실행 전 필요한 모델 가중치 다운로드 확인

## 테스트 및 검증
- 코드 변경 후 `demo.py` 실행으로 기본 동작 확인
- 로그 파일 `demo.log` 확인으로 오류 검토

## 자주 사용되는 명령어
```bash
# 데모 실행
python demo.py

# 의존성 설치
pip install -r requirements.txt
```