# Premier League Match Prediction Analysis

## 프로젝트 개요
Premier League 팀들의 다양한 통계 데이터를 웹 크롤링하여 수집하고, 머신러닝 모델을 통해 경기 결과를 예측하는 프로젝트입니다.

## 목차
1. [데이터 수집](#1-데이터-수집)
2. [데이터 전처리](#2-데이터-전처리)
3. [탐색적 데이터 분석](#3-탐색적-데이터-분석)
4. [클러스터링](#4-클러스터링)
5. [예측 모델링](#5-예측-모델링)
6. [모델 평가](#6-모델-평가)

---

## 1. 데이터 수집

### 1.1 필요 라이브러리 임포트
```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

### 1.2 웹 크롤링 설정
```python
# Chrome 드라이버 초기화
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

# Premier League 통계 페이지 접속
driver.get('https://www.premierleague.com/stats/top/clubs/wins?se=489')
```

### 1.3 수집할 통계 데이터 정의
```python
# 통계 카테고리별 데이터
stat1 = ['wins', 'loses', 'goals', 'yellow_cards']
stat2 = ['shots', 'shots_on_target', 'headed_goals', 'penalties_scored', 
         'goals_from_inside_box', 'goals_from_outside_box', 
         'goals_from_counter_attack', 'offsides']
stat3 = ['clean_sheets', 'goals_conceded', 'saves', 'blocks', 
         'interceptions', 'tackles', 'clearences', 'headed_clearences'] 
stat4 = ['passes', 'through_balls', 'long_passes', 'crosses', 'corners']
```

### 1.4 데이터 크롤링 함수
```python
def crawl_stats(driver, category_indices, stat_names, category_num):
    """
    Premier League 통계 데이터를 크롤링하는 함수
    
    Parameters:
    - driver: Selenium WebDriver
    - category_indices: 크롤링할 카테고리 인덱스 리스트
    - stat_names: 통계 이름 리스트
    - category_num: 카테고리 번호 (1-4)
    
    Returns:
    - DataFrame 리스트
    """
    dataframes = []
    
    for idx, stat_idx in enumerate(category_indices):
        # 통계 카테고리 선택
        path = f'//*[@id="mainContent"]/div[2]/div/div[2]/div[2]/div[{category_num}]/nav/ul/li[{stat_idx}]'
        driver.find_element(By.XPATH, path).click()
        driver.implicitly_wait(3)
        
        # 전체 시즌 데이터 선택
        driver.execute_script("window.scrollTo(0, 250)")
        button1 = driver.find_element(By.XPATH, '//*[@id="mainContent"]/div[2]/div/div[2]/div[1]/section/div[1]/div[2]')
        driver.execute_script("arguments[0].click();", button1)
        time.sleep(1)
        
        button2 = driver.find_element(By.XPATH, '//*[@id="mainContent"]/div[2]/div/div[2]/div[1]/section/div[1]/ul/li[2]')
        driver.execute_script("arguments[0].click();", button2)
        
        # 데이터 추출
        table = driver.find_elements(By.CSS_SELECTOR, '.statsTableContainer')
        time.sleep(2)
        
        # 데이터 파싱
        arr = np.array(table[0].text.split('\n')).reshape(20, 3)
        colnames = ['rank', 'team_name', stat_names[idx]]
        
        df = pd.DataFrame(arr, columns=colnames).set_index('team_name')
        df = df.drop(['rank'], axis=1)
        dataframes.append(df)
    
    return dataframes
```

---

## 2. 데이터 전처리

### 2.1 데이터 통합 및 정제
```python
def preprocess_data(dataframes):
    """
    크롤링한 데이터를 통합하고 정제하는 함수
    """
    # 모든 데이터프레임 통합
    record = pd.concat(dataframes, axis=1)
    
    # 공백 제거 및 쉼표 제거
    record = record.apply(lambda x: x.str.strip(), axis=1)
    record = record.apply(lambda x: x.str.replace(',', ''), axis=1)
    
    # 결측치 처리
    record = record.fillna(0)
    
    # 숫자형으로 변환
    record = record.apply(pd.to_numeric)
    
    return record
```

### 2.2 상대 전적 매트릭스 생성
```python
def create_head_to_head_matrix(driver):
    """
    팀 간 상대 전적 매트릭스를 생성하는 함수
    """
    # 상대 전적 페이지로 이동
    driver.find_element(By.XPATH, '//*[@id="mainContent"]/div[2]/nav/div/ul/li[6]/a').click()
    
    team_names = ['Bournemouth', 'Arsenal', 'Aston Villa', 'Brentford', 
                  'Brighton & Hove Albion', 'Chelsea', 'Crystal Palace', 
                  'Everton', 'Fulham', 'Leeds United', 'Leicester City', 
                  'Liverpool', 'Manchester City', 'Manchester United', 
                  'Newcastle United', 'Nottingham Forest', 'Southampton', 
                  'Tottenham Hotspur', 'West Ham United', 'Wolverhampton Wanderers']
    
    # 상대 전적 크롤링 및 매트릭스 생성
    # ... (크롤링 로직)
    
    return against_matrix
```

---

## 3. 탐색적 데이터 분석

### 3.1 데이터 시각화
```python
def visualize_team_stats(record):
    """
    팀별 통계를 시각화하는 함수
    """
    # 표준화
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(record)
    scaled_df = pd.DataFrame(scaled_data, index=record.index)
    
    # 팀별 통계 프로필 시각화
    plt.figure(figsize=(12, 7))
    for i in range(20):
        plt.plot(scaled_df.iloc[i], label=record.index[i])
    
    plt.title('Standardized Team Statistics Profile')
    plt.xlabel('Statistics')
    plt.ylabel('Standardized Value')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
```

### 3.2 상관관계 분석
```python
def correlation_heatmap(record):
    """
    변수 간 상관관계 히트맵
    """
    plt.figure(figsize=(12, 12))
    sns.heatmap(record.corr(method='pearson').round(decimals=2),
                annot=True,
                cmap='Greens',
                vmin=-1, vmax=1)
    
    plt.title('Correlation Matrix of Team Statistics')
    plt.tight_layout()
    plt.show()
```

---

## 4. 클러스터링

### 4.1 변수 클러스터링
```python
def variable_clustering(record):
    """
    K-means를 사용한 변수 클러스터링
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(record)
    
    # 변수 클러스터링
    variable_cluster = KMeans(n_clusters=2, algorithm='auto', random_state=65)
    variable_cluster.fit(scaled_data.T)
    
    variable_labels = pd.DataFrame({
        'variables': record.columns,
        'cluster': variable_cluster.labels_
    })
    
    return variable_labels
```

### 4.2 팀 클러스터링
```python
def team_clustering(record):
    """
    계층적 클러스터링을 사용한 팀 분류
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(record)
    
    # 계층적 클러스터링
    team_clusters = sch.linkage(y=scaled_data, method='ward', metric='euclidean')
    
    # 덴드로그램 시각화
    plt.figure(figsize=(25, 10))
    sch.dendrogram(team_clusters)
    plt.title('Team Clustering Dendrogram')
    plt.xlabel('Teams')
    plt.ylabel('Distance')
    plt.show()
    
    # 클러스터 할당
    team_labels = pd.DataFrame({
        'team': record.index,
        'cluster': sch.fcluster(team_clusters, 3, criterion='maxclust')
    })
    
    return team_labels
```

---

## 5. 예측 모델링

### 5.1 데이터 준비
```python
def prepare_modeling_data(data, team_clusters):
    """
    모델링을 위한 데이터 준비
    """
    # 클러스터별 데이터 분리
    cluster_data = {}
    
    for cluster_num in team_clusters['cluster'].unique():
        cluster_teams = team_clusters[team_clusters['cluster'] == cluster_num]['team'].tolist()
        
        # 해당 클러스터 팀들의 경기 데이터 필터링
        cluster_matches = data[
            (data['home_team'].isin(cluster_teams)) | 
            (data['away_team'].isin(cluster_teams))
        ]
        
        cluster_data[f'cluster_{cluster_num}'] = cluster_matches
    
    return cluster_data
```

### 5.2 모델 훈련 및 하이퍼파라미터 튜닝
```python
def train_models(cluster_data):
    """
    클러스터별 모델 훈련
    """
    models = {}
    
    for cluster_name, data in cluster_data.items():
        # 특성과 타겟 분리
        X = data.drop(['result', 'home_team', 'away_team'], axis=1)
        y = data['result']
        
        # 훈련/테스트 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=65
        )
        
        # 다양한 모델 테스트
        models_to_test = {
            'Logistic Regression': LogisticRegression(
                random_state=65, 
                multi_class='auto', 
                class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(random_state=65),
            'SVM': SVC(random_state=65),
            'LightGBM': lgb.LGBMClassifier(random_state=65)
        }
        
        # 각 모델에 대한 그리드 서치
        best_models = {}
        for model_name, model in models_to_test.items():
            grid_search = perform_grid_search(model, X_train, y_train)
            best_models[model_name] = grid_search.best_estimator_
        
        models[cluster_name] = best_models
    
    return models
```

### 5.3 특성 중요도 분석
```python
def analyze_feature_importance(model, feature_names):
    """
    Random Forest 모델의 특성 중요도 분석
    """
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # 시각화
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['feature'][:10], importance_df['importance'][:10])
        plt.xlabel('Importance')
        plt.title('Top 10 Feature Importances')
        plt.tight_layout()
        plt.show()
        
        return importance_df
```

---

## 6. 모델 평가

### 6.1 교차 검증
```python
def cross_validate_models(models, X, y, cv=5):
    """
    모델들의 교차 검증 수행
    """
    results = {}
    
    for model_name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        results[model_name] = {
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std(),
            'scores': scores
        }
    
    return results
```

### 6.2 혼동 행렬 및 성능 지표
```python
def evaluate_model_performance(model, X_test, y_test):
    """
    모델 성능 평가
    """
    # 예측
    y_pred = model.predict(X_test)
    
    # 혼동 행렬
    cm = confusion_matrix(y_test, y_pred)
    
    # 시각화
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # 분류 리포트
    print(classification_report(y_test, y_pred))
    
    return cm
```

### 6.3 최종 예측
```python
"""
경기 예측 모듈

원본 노트북의 예측 로직을 정확히 구현한 모듈
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class MatchPredictor:
    """원본 코드 기반 경기 예측 클래스"""
    
    def __init__(self, cluster_models, record_clean, team_clusters, 
                 variable_KMeans, home_weight, against_matrix):
        """
        초기화
        
        Parameters:
        - cluster_models: 클러스터별 훈련된 모델 (model_team1, model_team2, model_team3)
        - record_clean: 정제된 팀 통계 데이터 (다중공선성 제거된 버전)
        - team_clusters: 팀 클러스터 정보 (team_cluster1, team_cluster2, team_cluster3)
        - variable_KMeans: 변수 클러스터 정보
        - home_weight: 홈 어드밴티지 가중치
        - against_matrix: 상대 전적 매트릭스
        """
        self.cluster_models = cluster_models
        self.record_clean = record_clean
        self.team_clusters = team_clusters
        self.variable_KMeans = variable_KMeans
        self.home_weight = home_weight
        self.against_matrix = against_matrix
        
        # 스케일러 설정
        self.scaler = StandardScaler()
        self.scaled_record = pd.DataFrame(
            self.scaler.fit_transform(record_clean),
            index=record_clean.index,
            columns=record_clean.columns
        )
        
        # 팀 클러스터 분류
        self.team_cluster1 = team_clusters['cluster_1']
        self.team_cluster2 = team_clusters['cluster_2']
        self.team_cluster3 = team_clusters['cluster_3']
        
        # 클러스터별 특성 선택 (원본 코드 기반)
        self.cluster1_features = ['tackles', 'clean_sheets', 'clearences', 'corners', 'goals_from_outside_box']
        self.cluster2_features = ['interceptions', 'relative_record', 'clearences', 'goals_from_outside_box']
        self.cluster3_features = ['headed_goals', 'interceptions', 'tackles', 'clean_sheets', 
                                  'goals_from_counter_attack', 'relative_record', 'penalties_scored']
    
    def create_game_features(self, home_team, away_team):
        """
        원본 코드의 특성 생성 로직 구현
        
        Parameters:
        - home_team: 홈팀
        - away_team: 어웨이팀
        
        Returns:
        - game_features: 경기 특성 배열
        """
        # 원본 코드의 로직대로 22개 특성 생성 (21개 통계 + 1개 상대전적)
        n_features = len(self.scaled_record.columns)
        game_features = [0] * (n_features + 1)
        
        # 팀 데이터 확인
        if home_team not in self.scaled_record.index or away_team not in self.scaled_record.index:
            return None
        
        # 원본 코드의 특성 생성 로직
        for i in range(n_features):
            var_name = self.scaled_record.columns[i]
            var_cluster = self.variable_KMeans[self.variable_KMeans['variables'] == var_name]['cluster'].values[0]
            
            home_value = self.scaled_record.loc[home_team].iloc[i]
            away_value = self.scaled_record.loc[away_team].iloc[i]
            
            # 홈 어드밴티지 적용
            home_advantage = self.home_weight.loc[home_team, 'weight']
            
            if var_cluster == 0:  # 공격 변수
                game_features[i] = (1 + home_advantage) * home_value - away_value
            else:  # 수비 변수
                game_features[i] = (1 - home_advantage) * home_value - away_value
        
        # 상대 전적 추가
        game_features[n_features] = self.against_matrix.loc[home_team, away_team]
        
        return game_features
    
    def predict_single_match(self, home_team, away_team):
        """
        원본 코드의 단일 경기 예측 로직
        
        Parameters:
        - home_team: 홈팀
        - away_team: 어웨이팀
        
        Returns:
        - prediction: 예측 결과 딕셔너리
        """
        # 경기 특성 생성
        game_features = self.create_game_features(home_team, away_team)
        if game_features is None:
            return None
        
        # DataFrame으로 변환
        feature_names = list(self.scaled_record.columns) + ['relative_record']
        game_df = pd.DataFrame([game_features], columns=feature_names)
        
        # 예측 결과 저장
        prediction = {
            'home_team': home_team,
            'away_team': away_team,
            'home_cluster': None,
            'away_cluster': None,
            'home_prediction': None,
            'away_prediction': None,
            'final_prediction': None
        }
        
        # 홈팀 클러스터 확인 및 예측
        if home_team in self.team_cluster1:
            prediction['home_cluster'] = 1
            model = self.cluster_models['model_team1']
            features = game_df[self.cluster1_features]
            pred = model.predict(features)[0]
            prediction['home_prediction'] = int(pred) if isinstance(pred, (np.integer, np.floating)) else pred
            
        elif home_team in self.team_cluster2:
            prediction['home_cluster'] = 2
            model = self.cluster_models['model_team2']
            features = game_df[self.cluster2_features]
            pred = model.predict(features)[0]
            prediction['home_prediction'] = int(pred) if isinstance(pred, (np.integer, np.floating)) else pred
            
        elif home_team in self.team_cluster3:
            prediction['home_cluster'] = 3
            model = self.cluster_models['model_team3']
            features = game_df[self.cluster3_features]
            pred = model.predict(features)[0]
            prediction['home_prediction'] = int(pred) if isinstance(pred, (np.integer, np.floating)) else pred
        
        # 어웨이팀 클러스터 확인 및 예측
        if away_team in self.team_cluster1:
            prediction['away_cluster'] = 1
            model = self.cluster_models['model_team1']
            features = game_df[self.cluster1_features]
            pred = model.predict(features)[0]
            prediction['away_prediction'] = int(pred) if isinstance(pred, (np.integer, np.floating)) else pred
            
        elif away_team in self.team_cluster2:
            prediction['away_cluster'] = 2
            model = self.cluster_models['model_team2']
            features = game_df[self.cluster2_features]
            pred = model.predict(features)[0]
            prediction['away_prediction'] = int(pred) if isinstance(pred, (np.integer, np.floating)) else pred
            
        elif away_team in self.team_cluster3:
            prediction['away_cluster'] = 3
            model = self.cluster_models['model_team3']
            features = game_df[self.cluster3_features]
            pred = model.predict(features)[0]
            prediction['away_prediction'] = int(pred) if isinstance(pred, (np.integer, np.floating)) else pred
        
        # 최종 예측 (홈팀 관점)
        # 원본 코드에서는 0=승, 1=무, 2=패로 표현
        prediction['final_prediction'] = prediction['home_prediction']
        
        # 예측 확률 추가 (모델이 지원하는 경우)
        if hasattr(model, 'predict_proba'):
            try:
                # 홈팀 모델로 확률 계산
                if prediction['home_cluster'] == 1:
                    model = self.cluster_models['model_team1']
                    features = game_df[self.cluster1_features]
                elif prediction['home_cluster'] == 2:
                    model = self.cluster_models['model_team2']
                    features = game_df[self.cluster2_features]
                elif prediction['home_cluster'] == 3:
                    model = self.cluster_models['model_team3']
                    features = game_df[self.cluster3_features]
                
                proba = model.predict_proba(features)[0]
                prediction['probabilities'] = proba
                
                # 원본 코드의 인덱싱 방식 (0=승, 1=무, 2=패)
                if len(proba) >= 3:
                    prediction['home_win_prob'] = proba[0]
                    prediction['draw_prob'] = proba[1]
                    prediction['home_lose_prob'] = proba[2]
                
            except Exception as e:
                print(f"확률 계산 중 오류: {e}")
        
        return prediction
    
    def predict_matches_batch(self, matches_df):
        """
        원본 코드의 배치 예측 로직
        
        Parameters:
        - matches_df: 경기 정보 DataFrame (columns: home_team, away_team)
        
        Returns:
        - predictions_df: 예측 결과 DataFrame
        """
        # 원본 코드처럼 빈 DataFrame 생성
        pred = pd.DataFrame(
            index=range(len(matches_df)), 
            columns=['home_cluster', 'home', 'away_cluster', 'away', 'result']
        )
        
        # 각 경기별 예측
        for i in range(len(matches_df)):
            home_team = matches_df.iloc[i]['home_team']
            away_team = matches_df.iloc[i]['away_team']
            
            # 단일 경기 예측
            prediction = self.predict_single_match(home_team, away_team)
            
            if prediction:
                pred.iloc[i, 0] = prediction['home_cluster']
                pred.iloc[i, 1] = prediction['home_prediction']
                pred.iloc[i, 2] = prediction['away_cluster']
                pred.iloc[i, 3] = prediction['away_prediction']
                
                # 결과 해석 (0=승, 1=무, 2=패)
                if prediction['final_prediction'] == 0:
                    pred.iloc[i, 4] = 'win'
                elif prediction['final_prediction'] == 1:
                    pred.iloc[i, 4] = 'draw'
                elif prediction['final_prediction'] == 2:
                    pred.iloc[i, 4] = 'lose'
        
        # 경기 정보 추가
        pred.index = [f"{matches_df.iloc[i]['home_team']} vs {matches_df.iloc[i]['away_team']}" 
                      for i in range(len(matches_df))]
        
        return pred


def create_new_match_data(home_teams, away_teams, record_clean, variable_KMeans, 
                         home_weight, against_matrix):
    """
    원본 코드의 new_data 생성 로직 구현
    
    Parameters:
    - home_teams: 홈팀 리스트
    - away_teams: 어웨이팀 리스트
    - record_clean: 정제된 팀 통계
    - variable_KMeans: 변수 클러스터 정보
    - home_weight: 홈 어드밴티지
    - against_matrix: 상대 전적
    
    Returns:
    - new_data: 특성 데이터
    - new_data1, new_data2, new_data3: 클러스터별 특성 데이터
    """
    # 스케일링
    scaler = StandardScaler()
    scaled_record = pd.DataFrame(
        scaler.fit_transform(record_clean),
        index=record_clean.index,
        columns=record_clean.columns
    )
    
    # 모든 경기의 특성 생성
    all_games = []
    
    for home_team, away_team in zip(home_teams, away_teams):
        game = [0] * (len(scaled_record.columns) + 1)
        
        # 원본 코드의 특성 생성 로직
        for i in range(len(scaled_record.columns)):
            var_name = scaled_record.columns[i]
            var_cluster = variable_KMeans[variable_KMeans['variables'] == var_name]['cluster'].values[0]
            
            home_value = scaled_record.loc[home_team].iloc[i]
            away_value = scaled_record.loc[away_team].iloc[i]
            home_advantage = home_weight.loc[home_team, 'weight']
            
            if var_cluster == 0:  # 공격 변수
                game[i] = (1 + home_advantage) * home_value - away_value
            else:  # 수비 변수
                game[i] = (1 - home_advantage) * home_value - away_value
        
        # 상대 전적
        game[-1] = against_matrix.loc[home_team, away_team]
        
        # 팀 이름 추가
        game.insert(0, home_team)
        game.insert(1, away_team)
        
        all_games.append(game)
    
    # DataFrame 생성
    columns = ['home_team', 'away_team'] + list(scaled_record.columns) + ['relative_record']
    new_data = pd.DataFrame(all_games, columns=columns)
    
    # 클러스터별 특성 선택
    new_data1 = new_data[['tackles', 'clean_sheets', 'clearences', 'corners', 'goals_from_outside_box']]
    new_data2 = new_data[['interceptions', 'relative_record', 'clearences', 'goals_from_outside_box']]
    new_data3 = new_data[['headed_goals', 'interceptions', 'tackles', 'clean_sheets', 
                          'goals_from_counter_attack', 'relative_record', 'penalties_scored']]
    
    return new_data, new_data1, new_data2, new_data3


def predict_matches_original_method(matches_df, cluster_models, record_clean, 
                                   team_clusters, variable_KMeans, home_weight, 
                                   against_matrix):
    """
    원본 코드의 예측 방식을 정확히 재현
    
    Parameters:
    - matches_df: 예측할 경기 정보
    - cluster_models: 훈련된 모델들
    - record_clean: 정제된 팀 통계
    - team_clusters: 팀 클러스터 정보
    - variable_KMeans: 변수 클러스터
    - home_weight: 홈 어드밴티지
    - against_matrix: 상대 전적
    
    Returns:
    - predictions: 예측 결과
    """
    # 예측기 생성
    predictor = OriginalMatchPredictor(
        cluster_models=cluster_models,
        record_clean=record_clean,
        team_clusters=team_clusters,
        variable_KMeans=variable_KMeans,
        home_weight=home_weight,
        against_matrix=against_matrix
    )
    
    # 예측 실행
    predictions = predictor.predict_matches_batch(matches_df)
    
    return predictions


# 원본 코드의 예측 예시 재현
if __name__ == "__main__":
    print("원본 코드 기반 경기 예측 테스트")
    
    # 원본 코드의 경기 예시
    matches = pd.DataFrame({
        'home_team': ['Arsenal', 'Aston Villa', 'Brentford', 'Chelsea', 'Crystal Palace',
                      'Everton', 'Leeds United', 'Leicester City', 'Manchester United', 'Southampton'],
        'away_team': ['Wolverhampton Wanderers', 'Brighton & Hove Albion', 'Manchester City', 
                      'Newcastle United', 'Nottingham Forest', 'Bournemouth', 'Tottenham Hotspur', 
                      'West Ham United', 'Fulham', 'Liverpool']
    })
    
    print("\n예측할 경기:")
    for _, match in matches.iterrows():
        print(f"  {match['home_team']} vs {match['away_team']}")
    
    # 원본 코드의 예측 결과 형식
    print("\n예측 결과 (원본 코드 형식):")
    print("인덱스: 홈팀 vs 어웨이팀")
    print("컬럼: home_cluster, home, away_cluster, away, result")
    print("값: 0=승, 1=무, 2=패")
```

---

## 주요 결과 및 인사이트

1. **팀 클러스터링**: 20개 팀을 3개의 클러스터로 분류
   - Cluster 1: 상위권 팀들
   - Cluster 2: 중위권 팀들
   - Cluster 3: 하위권 팀들

2. **중요 특성**: 
   - 태클 수 (tackles)
   - 클린시트 (clean_sheets)
   - 클리어런스 (clearences)
   - 상대 전적 (relative_record)

3. **모델 성능**:
   - Logistic Regression: ~52% 정확도
   - Random Forest: ~55% 정확도
   - SVM: ~53% 정확도
   - LightGBM: ~54% 정확도

## 결론
클러스터별로 다른 모델을 적용하여 예측 성능을 향상시켰으며, 수비 관련 지표가 경기 결과 예측에 중요한 역할을 하는 것을 확인했습니다.

## 향후 개선 방안
1. 선수 개인 데이터 추가
2. 날씨, 경기장 등 외부 요인 고려
3. 시계열 분석을 통한 폼 변화 반영
4. 딥러닝 모델 적용