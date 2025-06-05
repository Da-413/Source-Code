# 🗂️ 데이터 기반 프로젝트 모음집

이 레포지토리는 **5개의 완결형 데이터·AI 프로젝트**를 한곳에 모았습니다. 각 디렉터리에는 소스 코드, 데이터 파이프라인, 모델 노트북, 배포 스크립트, 그리고 상세한 `README.md`가 포함되어 있습니다. 이 최상위 문서는 빠르게 전반적인 구조를 파악할 수 있는 **로드맵** 역할을 합니다.

---
## 📂 프로젝트 개요

| 번호 | 디렉터리 | 프로젝트명 | 핵심 내용 | 주요 기술 |
|---|---|---|---|---|
| 1 | `semiconductor-anomaly-detection/` | **반도체 소자 이상탐지** | AutoEncoder·Isolation Forest·One‑Class SVM을 비교하고 베이지안 최적화로 성능 극대화 | TensorFlow, PyTorch, Bayesian Optimization |
| 2 | `premier-league-match-prediction/` | **프리미어리그 경기 결과 예측** | 경기 데이터 크롤링·전처리 후 앙상블 모델로 승·무·패 예측 | Python Selenium, XGBoost, LightGBM |
| 3 | `stock-simulator-autotrade/` | **주식 모의 투자 & AI 자동매매** | FastAPI 백엔드와 Kiwoom API를 연동한 실시간 매매 봇 | FastAPI, DRL‑UTrans, Docker, Jenkins |
| 4 | `mosaic-loan/` | **모자익론 P2P 재테크 플랫폼** | 투자자 자산 자동 분산·차입자 신용평가 AI·REST API 설계 | Spark ML, ResNet, Docker, REST·WebSocket |
| 5 | `financial-data-mining-valuation/` | **기업 가치 평가 & 금융 데이터 마이닝** | 301개 상장기업 재무데이터 수집·회귀 분석으로 내재가치 산출 | Python, R, Pandas, StatsModels |

> **Tip** : 각 프로젝트 폴더의 `README.md`에는 데이터 수집 방법, 모델링 전략, 결과 시각화, 실행/배포 방법 등 세부 내용이 정리되어 있습니다.

---
## 🛠️ 공통 기술 스택

- **Python 3.9 + Poetry** : 패키지 관리 및 의존성 고정
- **Docker & Docker Compose** : 환경 통일과 재현성 확보
- **CI/CD (Jenkins + GitHub Actions)** : 테스트·빌드·배포 자동화
- **MLOps** : 모델 버전 관리, 실험 추적(MLflow), 배포 자동화

---
## ✨ 기여 방법

1. 이슈 또는 기능 제안을 먼저 등록해 주세요.
2. 포크 → 새로운 브랜치 생성(`feature/<주제>`) → 커밋 후 PR 생성
3. 리뷰가 끝나면 메인 브랜치에 병합됩니다.

---
## 📜 라이선스

본 레포지토리는 MIT 라이선스를 따릅니다. 자유롭게 활용하되, 출처를 명시해 주세요.

---
> 문의: gyoo97413@gmail.com · LinkedIn [@your‑profile](https://linkedin.com/in/your-profile)
