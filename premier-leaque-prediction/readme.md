# Premier League 경기 결과 예측

Premier League 팀 통계를 직접 수집하고, 팀 특성과 상대전적을 이용해 **승·무·패를 예측**한 머신러닝 프로젝트입니다.

## 한눈에 보기

| 항목 | 내용 |
|---|---|
| 대상 | Premier League 20개 팀 |
| 목표 | 경기 결과 3-class 분류 |
| 데이터 수집 | Selenium |
| 모델 | Logistic Regression, Random Forest, SVM, LightGBM |
| 최고 정확도 | 약 **55%** |

## 전체 흐름

```text
웹 데이터 수집
      ↓
전처리 · 표준화
      ↓
팀 / 변수 특성 분석
      ↓
클러스터링
      ↓
경기별 Feature 생성
      ↓
분류 모델 학습 · 비교
      ↓
승 / 무 / 패 예측
```

## 1. 데이터 수집

Selenium을 이용해 Premier League 팀 통계 데이터를 수집했습니다. 공격·수비 관련 팀 지표와 경기 결과 데이터를 분석 가능한 형태로 정리했습니다.

관련 구현은 `src/crawling.py`에서 확인할 수 있습니다.

## 2. Feature Engineering

단순히 두 팀의 원본 통계를 모델에 넣지 않고 홈팀과 원정팀의 차이를 경기 단위 feature로 만들었습니다.

- 팀 통계 표준화
- 공격/수비 성격에 따른 변수 처리
- 홈 어드밴티지 반영
- 상대전적(`relative_record`) 추가

이 과정을 통해 각 경기를 하나의 학습 데이터로 변환했습니다.

## 3. 클러스터링

모든 팀에 같은 feature 조합을 적용하기보다 팀 특성을 기준으로 20개 팀을 **3개 클러스터**로 나누었습니다.

클러스터별로 사용되는 주요 feature를 다르게 구성해 팀 특성에 맞는 예측 모델을 만들고자 했습니다.

## 4. 모델 비교

다음 네 가지 분류 모델을 비교했습니다.

| 모델 | Accuracy |
|---|---:|
| Logistic Regression | 약 52% |
| Random Forest | 약 **55%** |
| SVM | 약 53% |
| LightGBM | 약 54% |

훈련 과정에서는 train/test 분할과 Grid Search를 사용하고, Accuracy와 classification report, confusion matrix를 통해 결과를 확인했습니다.

## 5. 주요 Feature

모델 분석 과정에서 다음 변수들이 상대적으로 중요한 신호로 나타났습니다.

| Feature | Importance |
|---|---:|
| interceptions | 0.300 |
| relative_record | 0.233 |
| clearences | 0.100 |
| goals_from_outside_box | 0.100 |

특히 수비 관련 지표와 상대전적이 경기 결과 분류에 의미 있는 정보를 제공했습니다.

## 6. 예측

학습된 클러스터별 모델과 팀 데이터를 이용해 새로운 경기의 feature를 생성하고 승·무·패를 예측하도록 구현했습니다.

`src/modeling.py`에는 모델 학습과 평가 로직이, 예측 관련 코드에는 홈/원정 팀의 feature 생성 및 결과 변환 과정이 포함되어 있습니다.

## 기술 스택

| 영역 | 기술 |
|---|---|
| Crawling | Selenium |
| Data | Pandas, NumPy |
| ML | Scikit-learn, LightGBM |
| Analysis | SciPy, StatsModels |
| Visualization | Matplotlib, Seaborn |

## 핵심 경험

이 프로젝트는 스포츠 데이터를 단순히 모델에 넣는 것보다 **어떤 방식으로 경기 단위 feature를 만들 것인지**가 중요하다는 점을 경험한 프로젝트입니다.

팀별 통계, 홈 어드밴티지, 상대전적을 하나의 경기 데이터로 변환하고 여러 분류 모델을 비교하면서 데이터 구성 방식이 모델 결과에 미치는 영향을 확인했습니다.

## 향후 개선

현재 구조에 최근 경기 폼, 선수 출전 정보, 일정·휴식일 같은 시계열 요소를 추가하면 정적인 시즌 통계만 사용할 때보다 경기 시점의 상태를 더 잘 반영할 수 있습니다.
