# Data & AI Projects

데이터 수집과 분석부터 머신러닝 모델링, API 서빙까지 직접 구현한 프로젝트를 정리한 저장소입니다.

각 프로젝트 README는 **문제 → 접근 방법 → 구현 → 결과** 순서로 정리했습니다. 자세한 구현은 각 디렉터리의 소스 코드와 노트북에서 확인할 수 있습니다.

## Projects

| 프로젝트 | 핵심 내용 | 주요 기술 |
|---|---|---|
| [모자익론 P2P 대출 플랫폼](./mosaic-loan-p2p-platform) | 불균형 신용 데이터를 이용한 부도 예측과 모델 API 서빙 | PySpark, Spark ML, Keras, FastAPI |
| [주식 모의투자 & AI 자동매매](./stock-simulation-ai-bot) | 규칙 기반 전략과 DRL-UTrans 모델을 자동매매 서비스에 연결 | PyTorch, FastAPI, AsyncIO |
| [기업 가치 평가](./business-value-evaluate) | 재무 데이터 자동 수집과 회귀 모델 비교 | Python, Selenium, R, StatsModels |
| [Premier League 경기 예측](./premier-leaque-prediction) | 팀 통계·상대전적 기반 승/무/패 예측 | Selenium, Scikit-learn, LightGBM |
| [반도체 이상 탐지](./semiconductor-anomaly-detection) | 이미지 임베딩 기반 비지도 이상 탐지 모델 비교 | PyTorch, TensorFlow, Scikit-learn |

## Repository Structure

```text
Source-Code/
├── mosaic-loan-p2p-platform/
├── stock-simulation-ai-bot/
├── business-value-evaluate/
├── premier-leaque-prediction/
└── semiconductor-anomaly-detection/
```

프로젝트마다 목적과 데이터가 달라 하나의 공통 기술 스택을 적용하기보다, 문제에 맞는 도구와 모델을 선택했습니다.

## Contact

- Email: gyoo97413@gmail.com
- GitHub: https://github.com/Da-413
