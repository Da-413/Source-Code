# 반도체 이미지 이상 탐지

정상 이미지 중심의 데이터에서 이미지 특징을 추출하고 여러 비지도 이상 탐지 방법을 비교한 프로젝트입니다.

기존 README에 노트북 코드가 그대로 나열되어 있던 내용을 정리하고, 전체 분석 흐름과 모델 선택 이유를 중심으로 구성했습니다.

## 한눈에 보기

| 항목 | 내용 |
|---|---|
| 문제 | 이미지 이상 탐지 |
| 특징 추출 | Pretrained ResNet50 |
| 비교 모델 | Isolation Forest, One-Class SVM, AutoEncoder |
| 튜닝 | Bayesian Optimization |
| 주요 기술 | PyTorch, TensorFlow/Keras, Scikit-learn |

## 전체 흐름

```text
Image
  ↓
Resize / Normalize
  ↓
Pretrained ResNet50
  ↓
Embedding Vector
  ↓
┌─────────────────────────────┐
│ Isolation Forest            │
│ One-Class SVM               │
│ AutoEncoder                 │
└─────────────────────────────┘
  ↓
Normal / Anomaly
```

## 1. 이미지 Feature 추출

원본 이미지를 바로 비지도 모델에 입력하지 않고 ImageNet으로 사전 학습된 **ResNet50**을 feature extractor로 사용했습니다.

마지막 classification layer를 제거하고 이미지마다 embedding vector를 생성해 이후 이상 탐지 모델의 입력으로 사용했습니다.

```python
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[:-1]))
```

입력 이미지는 224×224로 resize한 뒤 ImageNet 기준 mean/std로 normalize했습니다.

## 2. 이상 탐지 모델

### Isolation Forest

정상 데이터가 주로 분포하는 영역과 떨어진 관측치를 분리하는 방식으로 이상 이미지를 탐지했습니다. 별도의 이상 클래스 학습 데이터 없이 적용할 수 있다는 점에서 baseline으로 사용했습니다.

### One-Class SVM

정상 데이터의 경계를 학습하는 One-Class SVM을 비교했습니다. RBF kernel을 사용하고, 이상치 비율을 조절하는 `nu` 값을 Bayesian Optimization으로 탐색했습니다.

튜닝 과정에서는 생성된 군집의 silhouette score를 목적값으로 사용했습니다.

### AutoEncoder

ResNet50에서 추출한 embedding을 입력으로 받아 다시 복원하도록 AutoEncoder를 학습했습니다.

정상 데이터에서 학습한 reconstruction error를 기준으로 정상 패턴에서 크게 벗어난 샘플을 이상으로 분류했습니다.

## 3. 모델 비교 관점

세 모델은 이상을 판단하는 방식이 서로 다릅니다.

| 모델 | 판단 방식 | 특징 |
|---|---|---|
| Isolation Forest | 데이터 공간에서의 고립 정도 | 빠른 baseline 구성 |
| One-Class SVM | 정상 데이터의 경계 | kernel 기반 비선형 경계 |
| AutoEncoder | reconstruction error | 신경망 기반 정상 패턴 학습 |

한 모델의 결과만 사용하는 대신 서로 다른 방식의 비지도 모델을 구현하고 비교하면서 데이터에 적합한 이상 탐지 방식을 탐색했습니다.

## 4. 코드 구성

- `src/model.py` — AutoEncoder 구조
- `src/train.py` — AutoEncoder 학습
- `src/evaluate.py` — Precision / Recall / F1 평가
- `notebook/semiconductor_anomaly_detection.ipynb` — 전체 실험 과정

## 기술 스택

| 영역 | 기술 |
|---|---|
| Feature Extraction | PyTorch, torchvision, ResNet50 |
| Anomaly Detection | Isolation Forest, One-Class SVM |
| Deep Learning | TensorFlow/Keras, PyTorch |
| Optimization | Bayesian Optimization |
| Evaluation | Scikit-learn |
| Data | Pandas, NumPy |

## 핵심 경험

라벨이 충분한 일반적인 분류 문제와 달리 정상 데이터 중심의 환경에서 **어떻게 이상을 정의하고 판단할 것인지**를 다뤘습니다.

사전 학습 모델의 이미지 표현을 활용하고, 전통적인 이상 탐지 모델과 AutoEncoder를 같은 embedding 공간에서 비교하면서 모델마다 이상을 구분하는 기준이 어떻게 달라지는지 확인했습니다.

## 실행 및 상세 실험

실험 전체 과정과 파라미터는 `notebook/semiconductor_anomaly_detection.ipynb`에서 확인할 수 있습니다. README에는 핵심 구조만 남기고 긴 실험 코드는 노트북과 `src/` 파일로 분리했습니다.
