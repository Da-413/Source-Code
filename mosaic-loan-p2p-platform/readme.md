
# 모자익론 (P2P 재태크 플랫폼)

> **설명**: 투자자의 자산을 ‘모자이크’처럼 여러 차입자에게 자동 분산하고, 차입자에게는 신용평가 기반 맞춤형 대출을 제공하는 P2P 금융 투자·대출 서비스입니다.  
> **역할**: 개인 신용평가 모델 & AI 서빙 파트 리드 (Spark ML + Keras + FastAPI)

---

## 목차
1. [프로젝트 개요](#프로젝트-개요)  
2. [주요 기능](#주요-기능)  
3. [시스템 구성도](#시스템-구성도)  
4. [AI 모델링 파이프라인](#ai-모델링-파이프라인)  
5. [기술 스택](#기술-스택)  
6. [실행 방법](#실행-방법)  
7. [도전 과제 및 해결 방안](#도전-과제-및-해결-방안)  
8. [결과 및 인사이트](#결과-및-인사이트)  
9. [참고 자료](#참고-자료)  

---

## 프로젝트 개요
- **기간**: 2025‑02‑01 ~ 2025‑04‑01 (6 주)  
- **인원**: 6명 (FE 2, BE 2, AI 2)  
- **목표**  
  1. 투자자의 입력(금액·목표수익률)만으로 자동 분산 투자 실행  
  2. 차입자의 금융·비금융 데이터를 활용해 부도 확률 예측  
  3. 투자·대출 현황을 실시간 대시보드로 시각화  
- **성과 지표**  
  | 지표 | 값 | 비고 |
  |---|---|---|
  | Ensemble AUC | **0.87** | 개인 신용평가 모델 검증 |
  | 예측 API Latency | **< 1 s** | FastAPI – Spark ML 서빙 |
  | 투자 손실률 감소 | **−25 %** | 분산 투자 적용 전후 시뮬레이션 |

---

## 주요 기능
### 2‑1. 자동 분산 투자
- 투자자는 **금액·목표수익률**만 입력 → 최적 포트폴리오 자동 생성  
- 클러스터링(차입자 등급) + 시장지표(금리·경기)로 분산 비중 조정  
- **위험 상한선**·**동일차입자 중복방지** 로직으로 손실 방어

### 2‑2. 차입자 신용평가
- Spark ML (GBDT·Logistic) 세 모델 + K‑Means 군집 결과를  
  Keras Dense Network에 **확률 피처**로 결합 → Ensemble 모델 구축  
- AUC 0.87, Accuracy 82 % 달성

### 2‑3. 실시간 투자·대출 대시보드
- Next.js + Tailwind UI, WebSocket 스트림으로 수익·상환 현황 실시간 반영  
- Docker Compose 로 프론트·백엔드·AI 컨테이너 오케스트레이션

---

## 시스템 구성도
![시스템 아키텍처](./images/시스템아키텍쳐.png)

> **Flow**  
> ① Investor / Borrower → Gateway(FastAPI)  
> ② 투자요청 → **Investment Engine** → PostgreSQL  
> ③ 대출요청 → **Credit Scoring API** → Redis Cache + Spark Cluster  
> ④ 데이터 → Grafana Dashboard

---

## AI 모델링 파이프라인
### 4‑1. 데이터 레이어  
| 서브셋 | 컬럼 수 | 설명 |
|---|---|---|
| Demographic | 14 | 나이·성별·직업·거주지 |
| Credit Record | 23 | 기존 대출·연체 이력 |
| Behavior | 31 | 카드·계좌 사용 패턴 |
| Timeseries | 12 | 월별 상환·소득 추이 |

### 4‑2. 모델 스택  
1. **Timeseries Model** – Spark LGBM  
2. **Behavior Model** – Spark GBTClassifier  
3. **Record Model** – Spark LogisticRegression  
4. **K‑Means** – 리스크 동질군 5개 도출  
5. **Keras Ensemble** – 위 4개 결과를 Dense 4‑32‑16‑1, AUC Loss 커스텀

> 전체 파이프라인은 `models/` 폴더에 저장, FastAPI startup 시 자동 로드

---

## 기술 스택
| 범주 | 사용 기술 |
|---|---|
| **Backend / API** | FastAPI, Uvicorn, Pydantic |
| **AI / ML** | PySpark 3, Spark MLlib, TensorFlow·Keras |
| **Data** | PostgreSQL, Redis, Parquet |
| **Infra & DevOps** | Docker, Docker Compose, Jenkins CI/CD |
| **Frontend** | Next.js, Tailwind CSS, Framer Motion |
| **협업 & 기타** | GitHub Projects, Slack, Figma |

---

## 실행 방법
### 환경 준비 (Docker Compose)
```bash
# 1) 레포 클론
git clone https://github.com/Da-413/MosaicLoan.git
cd MosaicLoan

# 2) 모델 체크포인트 다운로드 (예: S3, 구글드라이브)
./scripts/download_models.sh   # final_model.keras 등

# 3) 서비스 기동
docker compose up -d           # api:8001, fe:3000
```

### 로컬 개발 (선택)
```bash
# Python 가상환경
python -m venv venv && source venv/bin/activate
pip install -r api/requirements.txt

# Spark 세션 경량 실행
cd api
uvicorn main:app --reload --port 8001
```

---

## 도전 과제 및 해결 방안
| 과제 | 해결 방안 |
|---|---|
| Spark Driver 메모리 부족 | `spark.memory.fraction=0.6`, Kryo Serializer 적용 |
| 다중 모델 로딩 지연 | 모델 별 lazy load + Redis warm‑cache |
| 실시간 예측 <1 s | 예측 단계 Spark 사용 최소화 → Numpy 전환, batch size 1 inference |
| 컨테이너 헬스체크 | `/health` 엔드포인트 + Jenkins pipeline fail fast |

---

## 결과 및 인사이트
1. **투자 손실률 25 % 감소** – 분산 투자 시뮬레이션  
2. **AUC 0.87 / Latency <1 s** – 실시간 신용평가 API  
3. **자동화 배포** – PR 머지 → Jenkins 빌드 → Docker Hub → Prod 서버 배포 완료까지 3 분

> *“금융 리스크를 데이터로 관리하고, 투자·대출 경험을 모두 향상한다”* 는 목표를 달성했습니다.

---

## 참고 자료
- Kaggle Credit Risk Dataset  
- Spark MLlib Docs <https://spark.apache.org/docs/latest/ml-guide.html>  
- TensorFlow Addons AUC Loss 구현 <https://www.tensorflow.org/addons/api_docs/python/tfa/losses/AUCBinary>
