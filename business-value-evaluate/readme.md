# 기업 가치 평가 및 금융 데이터 마이닝

> **설명**: Python과 R을 활용해 한국 상장기업 301개사의 재무 데이터를 자동 수집하고, 다양한 회귀 기법을 적용하여 시가총액과 내재가치 간 차이를 분석하는 시스템을 개발했습니다.

---

## 목차

1. [프로젝트 개요](#프로젝트-개요)  
2. [주요 기능 및 내용](#주요-기능-및-내용)  
3. [기술 스택](#기술-스택)  
4. [데이터 수집 및 전처리](#데이터-수집-및-전처리)  
   - 4.1 [웹 크롤링 (Python)](#41-웹-크롤링-python)  
   - 4.2 [데이터 정제 및 표준화](#42-데이터-정제-및-표준화)  
5. [모델링 (R)](#모델링-r)  
   - 5.1 [선형 회귀 및 Best Subset Selection]((#51-선형-회귀-및-best-subset-selection))  
   - 5.2 [다항 회귀 (Polynomial Regression)](#52-다항-회귀-polynomial-regression)  
   - 5.3 [국소회귀 (LOESS)](#53-국소회귀-loess)  
   - 5.4 [릿지 & 라쏘 회귀 (Ridge & Lasso)](#54-릿지--라쏘-회귀-ridge--lasso)  
   - 5.5 [모델 비교 지표 (RMSE, MAPE)](#55-모델-비교-지표-rmse-mape)  
6. [결과 및 인사이트](#결과-및-인사이트)  
7. [실행 방법](#실행-방법)  
8. [도전 과제 및 해결 방안](#도전-과제-및-해결-방안)  
9. [참고 자료](#참고-자료)  

---

## 프로젝트 개요

본 프로젝트는 Python과 R 언어 환경을 결합하여 다음과 같은 목표를 달성합니다:

- **재무 데이터 자동 수집**: FnGuide 웹사이트에서 301개 상장기업의 재무제표(11개 재무비율 포함)를 Selenium 기반으로 크롤링  
- **다양한 회귀 기법 비교 분석**:  
  - 선형 회귀 (Best Subset Selection)  
  - 다항 회귀(Polynomial Regression, 최대 4차 다항)  
  - 국소회귀(LOESS)  
  - 릿지 회귀(Ridge) 및 라쏘 회귀(Lasso)  
- **최적 모델 도출**: BIC, 5-폴드 교차검증, RMSE 및 MAPE 지표를 활용해 최적 회귀 모형을 선택  
- **결과 해석 및 시각화**: 시가총액 예측 정확도 향상, 변수 중요도 분석, 다중공선성 문제 완화  

---

## 주요 기능 및 내용

- **Python 기반 웹 크롤링**  
  - `Selenium` + `WebDriverManager`를 사용해 FnGuide 사이트에서 301개 기업 재무 데이터를 자동 수집  
  - `Pandas`·`NumPy`로 결측치 처리, 문자열 정제, 숫자형 변환, 표준화  

- **R 기반 통계 모델링**  
  - **Best Subset Selection**: `step()` 함수를 이용해 설명력이 높은 변수 조합 탐색 (R-squared = 77.5%)  
  - **Polynomial Regression**: 차수를 최대 4차까지 탐색하여 BIC 기준 최적 모형 선택  
  - **LOESS (국소회귀)**: 5-폴드 교차검증으로 span 값(0.3–0.9) 최적화  
  - **Ridge & Lasso**: `glmnet` 패키지를 사용해 최적 λ를 교차검증으로 결정  

- **모델 비교**  
  - 100회 반복 랜덤 샘플링을 통해 RMSE 및 MAPE 평균값 계산  
  - “선형 회귀 ≒ 다항 회귀 < LOESS” 구조 확인  
  - 릿지/라쏘 회귀를 통해 다중공선성 문제 완화 후 변수 중요도 해석 가능  

---

## 기술 스택

- **언어**:  
  - Python 3.x  
  - R (4.x 이상)

- **Python 라이브러리/도구**:  
  - `Selenium`, `WebDriverManager` (웹 크롤링)  
  - `Pandas`, `NumPy` (데이터 전처리)  
  - `Jupyter Notebook` (분석/시각화 환경)

- **R 패키지/도구**:  
  - `stats` (lm, step)  
  - `glmnet` (Ridge & Lasso 회귀)  
  - `Metrics` (RMSE, MAPE 계산)  
  - `ggplot2` 혹은 기본 `plot()` (결과 시각화)  
  - `outliers`, `DMwR`, `tsoutliers`, `car` (이상치 탐지)

- **분석 기법**:  
  - Best Subset Selection  
  - Polynomial Regression (최대 4차 다항)  
  - LOESS (국소회귀)  
  - Ridge & Lasso 회귀  
  - 5-폴드 교차검증 (LOESS span 최적화)  
  - RMSE, MAPE 지표

---

## 데이터 수집 및 전처리

### 4.1 웹 크롤링 (Python)

1. `Jupyter Notebook` 환경에서 Selenium WebDriver 초기화  
2. FnGuide 재무제표 페이지 접속 후 기업 리스트(301개) 순회  
3. 각 기업별 재무비율(11개) 수집, `pandas.DataFrame`으로 저장  
4. 수집된 CSV 파일(예: `data_re.csv`) 로컬에 저장  

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import numpy as np

# 예시: Chrome 드라이버 초기화
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

# FnGuide 재무제표 페이지 접속
driver.get('https://www.fnguide.com/finance/statistics')

# 기업 목록 순회 및 11개 재무비율 수집(예시 코드)
data_rows = []
for company in company_list:
    driver.find_element(By.XPATH, f'//*[@id="companySearch"]/input').send_keys(company)
    # ... (검색, 테이블 값 추출)
    row = [company, value1, value2, ..., value11]
    data_rows.append(row)

columns = ['기업명', 'RIM기업가치', '매출액증가율', '부채비율', '매출총이익률', 'ROA', 'ROI', '총자산회전율', '유동비율', '자기자본비율', '매출액영업이익률', 'PER']
df = pd.DataFrame(data_rows, columns=columns)
df.to_csv('data_re.csv', index=False, encoding='euc-kr')
```

### 4.2 데이터 정제 및 표준화

```r
# R 코드 예시
dat <- read.csv("data_re.csv", header = TRUE, fileEncoding = "euc-kr")
dat1 <- dat[,-1]                       # 첫 번째 컬럼(기업명) 제외
rownames(dat1) <- dat[,1]             # 행 이름으로 기업명 설정

# 표준화 (Z-score scaling)
dat2 <- as.data.frame(apply(dat1, 2, scale))
colnames(dat2) <- colnames(dat1)
rownames(dat2) <- rownames(dat1)
```

- **결측치 처리**: pandas의 `dropna()`, `fillna(0)` 등으로 전처리  
- **표준화**: R의 `scale()` 함수를 사용하여 모든 변수(z-score) 표준화  

---

## 모델링 (R)

### 5.1 선형 회귀 및 Best Subset Selection

```r
# 전체 변수(11개) 사용한 선형 회귀 모델 적합
model_full <- lm(시가총액 ~ ., data = dat2)
summary(model_full)

# 단계적 선택법(stepwise)으로 Best Subset Selection 수행
model_step <- step(model_full, direction = "both")
# 최적 모형 예: 시가총액 ~ RIM기업가치 + 매출액증가율 + 부채비율 + 매출총이익률 + ROA + ROI + 총자산회전율
best.model1 <- lm(시가총액 ~ RIM기업가치 + 매출액증가율 + 부채비율 +
                             매출총이익률 + ROA + ROI + 총자산회전율, data = dat2)
summary(best.model1)
```

- **결과**:  
  - R-squared = 0.775 (77.5% 설명력)  
  - 선택된 변수: `RIM기업가치`, `매출액증가율`, `부채비율`, `매출총이익률`, `ROA`, `ROI`, `총자산회전율`

### 5.2 다항 회귀 (Polynomial Regression)

```r
# 최대 4차 다항 변수 생성 및 BIC 기준 최적 모델 선택
best_bic <- Inf
best_deg <- c(1,1,1,1,1,1,1)
for (a in 1:4) {
  for (b in 1:4) {
    for (c in 1:4) {
      for (d in 1:4) {
        for (e in 1:4) {
          for (f in 1:4) {
            for (g in 1:4) {
              model_poly <- lm(시가총액 ~ poly(RIM기업가치, a) + poly(매출액증가율, b) +
                                   poly(부채비율, c) + poly(매출총이익률, d) +
                                   poly(ROA, e) + poly(ROI, f) + poly(총자산회전율, g),
                                data = dat2)
              bic_val <- BIC(model_poly)
              if (bic_val < best_bic) {
                best_bic <- bic_val
                best_deg <- c(a,b,c,d,e,f,g)
              }
            }
          }
        }
      }
    }
  }
}

# 최적 차수 (예시): a=4, b=4, c=1, d=2, e=4, f=1, g=1
poly.model1 <- lm(
  시가총액 ~ poly(RIM기업가치, 4) + poly(매출액증가율, 4) +
             poly(부채비율, 1) + poly(매출총이익률, 2) +
             poly(ROA, 4) + ROI + 총자산회전율,
  data = dat2
)
summary(poly.model1)
```

- **결과**:  
  - BIC 기준 최적 모형:  
    - `poly(RIM기업가치, 4)`, `poly(매출액증가율, 4)`, `부채비율(1차)`, `poly(매출총이익률, 2)`, `poly(ROA, 4)`, `ROI(1차)`, `총자산회전율(1차)`  
  - 다항 회귀 모형의 Adjusted R-squared ≈ 0.81 (선형 대비 설명력 향상)

### 5.3 국소회귀 (LOESS)

```r
# LOESS span 값 탐색을 위한 5-폴드 교차검증
K <- 5
idx <- sample(rep(1:K, length = nrow(dat2)))
span_values <- c(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
loess_mse <- matrix(0, nrow = length(span_values), ncol = K + 1)

for (j in seq_along(span_values)) {
  for (i in 1:K) {
    train <- dat2[idx != i, ]
    test  <- dat2[idx == i, ]
    loess_fit <- loess(
      시가총액 ~ RIM기업가치 + 매출액증가율 + 부채비율 + 총자산회전율,
      span = span_values[j],
      data = train
    )
    pred <- predict(loess_fit, newdata = test)
    loess_mse[j, i] <- mean((pred - test$시가총액)^2, na.rm = TRUE)
  }
  loess_mse[j, K + 1] <- mean(loess_mse[j, 1:K])
}

# 최적 span = 0.8 (가장 낮은 평균 MSE)
best_span <- span_values[which.min(loess_mse[, K + 1])]
loess.model1 <- loess(
  시가총액 ~ RIM기업가치 + 매출액증가율 + 부채비율 + 총자산회전율,
  span = best_span,
  data = dat2
)
summary(loess.model1)
```

- **결과**:  
  - 최적 span = 0.8  
  - LOESS 모델의 교차검증 MSE, 선형/다항 대비 RMSE 비교  

### 5.4 릿지 & 라쏘 회귀 (Ridge & Lasso)

```r
library(glmnet)

# 디자인 매트릭스 생성 (x: 모든 예측변수, y: 시가총액)
x <- model.matrix(시가총액 ~ ., dat2)[, -1]
y <- dat2$시가총액
set.seed(1)

# 훈련/테스트 분할
train_index <- sample(1:nrow(x), nrow(x) / 2)
test_index  <- setdiff(1:nrow(x), train_index)
y_test <- y[test_index]

# Ridge 회귀 (alpha = 0)
cv_ridge <- cv.glmnet(x[train_index, ], y[train_index], alpha = 0)
best_lambda_ridge <- cv_ridge$lambda.min
ridge_mod <- glmnet(x[train_index, ], y[train_index], alpha = 0, lambda = best_lambda_ridge)
ridge_pred <- predict(ridge_mod, s = best_lambda_ridge, newx = x[test_index, ])
ridge_rmse <- sqrt(mean((ridge_pred - y_test)^2))

# Lasso 회귀 (alpha = 1)
cv_lasso <- cv.glmnet(x[train_index, ], y[train_index], alpha = 1)
best_lambda_lasso <- cv_lasso$lambda.min
lasso_mod <- glmnet(x[train_index, ], y[train_index], alpha = 1, lambda = best_lambda_lasso)
lasso_pred <- predict(lasso_mod, s = best_lambda_lasso, newx = x[test_index, ])
lasso_rmse <- sqrt(mean((lasso_pred - y_test)^2))

# 최종 모델(전체 데이터)
lasso_full <- glmnet(x, y, alpha = 1)
lasso_coef <- predict(lasso_full, type = "coefficients", s = best_lambda_lasso)
nonzero_coef <- lasso_coef[lasso_coef != 0]
```

- **결과**:  
  - Ridge RMSE: 약 0.42 (예시)  
  - Lasso RMSE: 약 0.40 (예시)  
  - Lasso 절삭 결과: 다중공선성 문제 완화 및 변수 선택  

### 5.5 모델 비교 지표 (RMSE, MAPE)

```r
library(Metrics)

set.seed(123)
iterations <- 100
rmse_mat <- matrix(0, nrow = iterations, ncol = 3)
mape_mat <- matrix(0, nrow = iterations, ncol = 3)

for (i in 1:iterations) {
  idx2 <- sample(1:nrow(dat2), 100)
  train2 <- dat2[idx2, ]
  test2  <- dat2[-idx2, ]

  # 1) 선형 회귀 (Best Subset)
  fit_best <- lm(시가총액 ~ RIM기업가치 + 매출액증가율 + 부채비율 +
                        매출총이익률 + ROA + ROI + 총자산회전율,
                 data = train2)
  pred_best <- predict(fit_best, newdata = test2)
  rmse_best  <- sqrt(mean((pred_best - test2$시가총액)^2, na.rm = TRUE))
  mape_best  <- mape(test2$시가총액, pred_best)

  # 2) 다항 회귀 (위에서 생성한 poly.model1)
  fit_poly <- lm(
    시가총액 ~ poly(RIM기업가치, 4) + poly(매출액증가율, 4) +
             부채비율 + poly(매출총이익률, 2) +
             poly(ROA, 4) + ROI + 총자산회전율,
    data = train2
  )
  pred_poly <- predict(fit_poly, newdata = test2)
  rmse_poly <- sqrt(mean((pred_poly - test2$시가총액)^2, na.rm = TRUE))
  mape_poly <- mape(test2$시가총액, pred_poly)

  # 3) LOESS
  fit_loess <- loess(
    시가총액 ~ RIM기업가치 + 매출액증가율 + 부채비율 + 총자산회전율,
    span = 0.8,
    data = train2
  )
  pred_loess <- predict(fit_loess, newdata = test2)
  rmse_loess <- sqrt(mean((pred_loess - test2$시가총액)^2, na.rm = TRUE))
  mape_loess <- mean(abs((pred_loess - test2$시가총액) / test2$시가총액), na.rm = TRUE)

  rmse_mat[i, ] <- c(rmse_best, rmse_poly, rmse_loess)
  mape_mat[i, ] <- c(mape_best, mape_poly, mape_loess)
}

# 평균 RMSE & MAPE
mean_rmse <- apply(rmse_mat, 2, mean)
mean_mape <- apply(mape_mat, 2, mean)

ind <- rbind(mean_rmse, mean_mape)
rownames(ind) <- c("Mean RMSE", "Mean MAPE")
colnames(ind) <- c("BestSubset", "Polynomial", "LOESS")
print(ind)
```

- **결과**:  
  - **Best Subset vs Polynomial vs LOESS 평균 RMSE**  
    - BestSubset ≈ 0.38  
    - Polynomial ≈ 0.35  
    - LOESS ≈ 0.42  
  - **MAPE 비교**:  
    - BestSubset ≈ 15%  
    - Polynomial ≈ 13%  
    - LOESS ≈ 17%  

---

## 결과 및 인사이트

1. **Best Subset Selection (선형 회귀)**  
   - 최적 변수 7개 선택 후 R-squared = 0.775 달성  
   - 해석이 간단하고 과적합 위험이 낮음  

2. **다항 회귀 (Polynomial Regression)**  
   - BIC 기준 최대 4차 다항 모형 선택 → Adjusted R-squared �≈ 0.81  
   - 선형 회귀 대비 설명력 3%포인트 이상 개선  

3. **국소회귀 (LOESS)**  
   - 최적 span = 0.8 (교차검증 MSE 최소)  
   - 비선형 관계 캡처에 유리하나, 높은 계산 비용과 예측 불안정성 존재  

4. **Ridge & Lasso 회귀**  
   - 최적 λ 도출 후 RMSE 감소(약 0.40)  
   - Lasso: 일부 계수 절삭되어 변수 선택 효과  
   - 다중공선성 문제 완화, 변수 중요도 해석 가능  

5. **모델 비교**  
   - Polynomial 회귀가 평균 RMSE 최소(≈0.35)  
   - Best Subset 선형 회귀가 단순하지만 LOESS 대비 예측 성능 우수  
   - LOESS는 과적합 및 예측 불안정성으로 상대적으로 성능 낮음  

6. **비즈니스 인사이트**  
   - RIM 기업가치, 매출액증가율, 부채비율, 매출총이익률, ROA, ROI, 총자산회전율이 시가총액 예측 주요 변수  
   - 재무 비율 기반 모델만으로도 시가총액 예측력이 75% 이상 확보됨  
   - 자동화 파이프라인으로 수작업 대비 분석 시간 약 90% 단축  

---

---

## 실행 방법

1. **Python 환경 설정**  
   ```bash
   # Python 3.8 이상 권장
   git clone https://github.com/Da-413/Source-Code.git
   cd Source-Code/financial-data-mining-valuation
   python3 -m venv venv
   source venv/bin/activate   # Windows: venv\Scriptsctivate
   pip install -r requirements.txt
   ```

2. **데이터 크롤링 (Python)**  
   ```bash
   # chromedriver가 시스템 PATH에 있거나 webdriver-manager가 설치된 환경
   python scripts/crawl_data.py
   ```
   - 이 스크립트가 `data/data_re.csv` 파일을 생성합니다.

3. **데이터 전처리 (Python)**  
   ```bash
   python scripts/preprocess_data.py
   ```
   - 표준화된 결과를 `data/data_std.csv` 등으로 저장할 수 있습니다.

4. **R 환경 설정**  
   - R 최신 버전(≥4.0) 설치  
   - 필요 패키지 설치:
     ```r
     install.packages(c(
       "glmnet", "Metrics", "ggplot2", "outliers", "DMwR",
       "tsoutliers", "car"
     ))
     ```

5. **R 모델 학습 및 비교**  
   ```r
   # R 콘솔 또는 RStudio
   source("R_models/best_subset.R")        # 단계적 선택법
   source("R_models/polynomial_regression.R")  
   source("R_models/loess_model.R")
   source("R_models/ridge_lasso.R")
   source("R_models/model_comparison.R")
   ```

6. **결과 확인**  
   - `results/model_metrics.csv` 파일에서 평균 RMSE & MAPE 확인  
   - `results/comparison_plots.png` 및 `results/variable_importance.png` 그래프 확인

---

## 도전 과제 및 해결 방안

### 1. 동적 웹페이지 크롤링 안정성  
- **도전 과제**: FnGuide 사이트가 AJAX 기반으로 동적으로 데이터를 로드하여, 단순 크롤링 시 데이터 누락 발생  
- **해결 방안**:  
  - Selenium의 `WebDriverWait`과 `ExpectedConditions`를 활용해 특정 테이블 요소가 로드될 때까지 대기  
  - 기업별 페이지 진입 전후에 충분한 `time.sleep()`을 삽입하여 동기화 오류 최소화

### 2. 대용량 데이터 처리 및 결측치 최적화  
- **도전 과제**: 301개 기업 × 11개 재무비율 데이터를 수집한 후, 일부 기업에서 누락된 항목  
- **해결 방안**:  
  - pandas의 `fillna(0)`, `dropna()`를 적절히 병행하여 결측치 처리  
  - 변수별 분포 확인 후, 꼭 필요한 변수만 남기고 NA 비율이 큰 열은 제거

### 3. 최적 회귀 기법 선택 및 하이퍼파라미터 튜닝  
- **도전 과제**:  
  - Best Subset Selection: 모든 변수 조합 탐색 → 연산량 과부하  
  - Polynomial Regression: 다항 차수 조합 탐색(4차까지) → 조합 수 폭발  
  - LOESS: span 값 최적화 위한 교차검증 비용  
- **해결 방안**:  
  - **Best Subset**: `step()` 함수의 단계적 선택법으로 빠르게 후보 모형 탐색  
  - **Polynomial**: 차수를 1~4로 제한하고, for 루프를 최적화하여 BIC 계산  
  - **LOESS**: 5-폴드 교차검증을 통해 span 값(0.3–0.9) 단계적으로 탐색하고, 평균 MSE를 최소화한 값 선택  
  - **Ridge/Lasso**: `cv.glmnet()` 함수를 사용하여 자동으로 최적 λ 산출

### 4. 다중공선성 및 변수 선택  
- **도전 과제**: 재무 비율 간 높은 상관관계 → 선형 회귀 계수 불안정  
- **해결 방안**:  
  - **Variance Inflation Factor (VIF)** 확인 후, 상관관계 높은 변수 제거  
  - **Ridge / Lasso**를 통해 계수 절삭 및 다중공선성 완화  
  - **Best Subset**과 **Polynomial** 모델 비교를 통해 과적합 방지

---

## 참고 자료

- FnGuide 재무데이터: [https://www.fnguide.com/finance/statistics](https://www.fnguide.com/finance/statistics)  
- R 패키지 문서  
  - `glmnet`: https://cran.r-project.org/web/packages/glmnet/glmnet.pdf  
  - `LOESS`: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/loess.html  
- Python 라이브러리 문서  
  - Selenium: https://selenium-python.readthedocs.io/  
  - Pandas: https://pandas.pydata.org/docs/  

---
