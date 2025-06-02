"""
머신러닝 모델링 모듈

경기 결과 예측을 위한 모델 훈련, 평가, 예측 함수들
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import lightgbm as lgb
import xgboost as xgb
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class MatchPredictor:
    """경기 결과 예측 모델 클래스"""
    
    def __init__(self):
        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=65, 
                multi_class='auto', 
                class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(random_state=65),
            'SVM': SVC(random_state=65, probability=True),
            'LightGBM': lgb.LGBMClassifier(random_state=65, verbose=-1)
        }
        
        self.param_grids = {
            'Logistic Regression': {
                'C': [0.001, 0.01, 0.1, 1],
                'solver': ['newton-cg', 'lbfgs', 'saga'],
                'max_iter': [100, 500]
            },
            'Random Forest': {
                'n_estimators': [10, 50, 100],
                'max_depth': [1, 5, 10, 20],
                'max_features': ['sqrt', 'log2']
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'poly', 'sigmoid'],
                'gamma': ['scale', 'auto']
            },
            'LightGBM': {
                'num_leaves': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [10, 50, 100],
                'max_depth': [1, 5, 10]
            }
        }
        
        self.cluster_models = {}
        self.best_models = {}
    
    def train_cluster_models(self, ml_datasets, cv=5):
        """
        각 클러스터별 모델 훈련
        
        Parameters:
        - ml_datasets: 클러스터별 ML 데이터셋
        - cv: 교차검증 폴드 수
        
        Returns:
        - cluster_models: 클러스터별 훈련된 모델
        """
        cluster_models = {}
        
        for cluster_name, data in ml_datasets.items():
            print(f"\n{'='*50}")
            print(f"{cluster_name} 모델 훈련")
            print(f"{'='*50}")
            
            X = data['X']
            y = data['y']
            
            # 클래스 분포 확인
            print(f"\n클래스 분포:")
            print(y.value_counts())
            
            # 훈련/테스트 분할
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=65, stratify=y
            )
            
            best_models = {}
            best_scores = {}
            
            # 각 모델별 훈련
            for model_name, model in self.models.items():
                print(f"\n[{model_name}]")
                
                try:
                    # 그리드 서치
                    grid_search = GridSearchCV(
                        model, 
                        self.param_grids[model_name], 
                        cv=cv, 
                        scoring='accuracy',
                        n_jobs=-1
                    )
                    grid_search.fit(X_train, y_train)
                    
                    # 최적 모델 저장
                    best_models[model_name] = grid_search.best_estimator_
                    best_scores[model_name] = grid_search.best_score_
                    
                    # 테스트 성능
                    test_pred = grid_search.predict(X_test)
                    test_accuracy = accuracy_score(y_test, test_pred)
                    
                    print(f"최적 파라미터: {grid_search.best_params_}")
                    print(f"CV 정확도: {grid_search.best_score_:.3f}")
                    print(f"테스트 정확도: {test_accuracy:.3f}")
                    
                except Exception as e:
                    print(f"에러 발생: {e}")
                    continue
            
            # 최고 성능 모델 선택
            if best_scores:
                best_model_name = max(best_scores, key=best_scores.get)
                best_model = best_models[best_model_name]
                
                print(f"\n✅ {cluster_name} 최적 모델: {best_model_name}")
                print(f"   CV 정확도: {best_scores[best_model_name]:.3f}")
                
                cluster_models[cluster_name] = {
                    'model': best_model,
                    'model_name': best_model_name,
                    'all_models': best_models,
                    'scores': best_scores,
                    'X_test': X_test,
                    'y_test': y_test
                }
        
        self.cluster_models = cluster_models
        return cluster_models
    
    def evaluate_models(self, cluster_models=None, n_iterations=50):
        """
        모델 안정성 평가
        
        Parameters:
        - cluster_models: 평가할 모델들
        - n_iterations: 반복 횟수
        
        Returns:
        - evaluation_results: 평가 결과
        """
        if cluster_models is None:
            cluster_models = self.cluster_models
        
        evaluation_results = {}
        
        for cluster_name, model_info in cluster_models.items():
            print(f"\n{'='*50}")
            print(f"{cluster_name} 모델 평가 ({n_iterations}회 반복)")
            print(f"{'='*50}")
            
            # 데이터 준비
            X = model_info['all_models'][list(model_info['all_models'].keys())[0]]._X
            y = model_info['all_models'][list(model_info['all_models'].keys())[0]]._y
            
            model_scores = {model_name: [] for model_name in model_info['all_models'].keys()}
            
            # 여러 번 반복하여 안정성 평가
            for i in range(n_iterations):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                
                for model_name, model in model_info['all_models'].items():
                    try:
                        # 모델 재훈련
                        model_copy = model.__class__(**model.get_params())
                        model_copy.fit(X_train, y_train)
                        
                        # 예측 및 정확도 계산
                        y_pred = model_copy.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)
                        model_scores[model_name].append(accuracy)
                    except:
                        continue
            
            # 결과 정리
            results = pd.DataFrame()
            for model_name, scores in model_scores.items():
                if scores:
                    results[model_name] = [
                        np.mean(scores), 
                        np.std(scores), 
                        np.min(scores), 
                        np.max(scores)
                    ]
            
            results.index = ['평균 정확도', '표준편차', '최소값', '최대값']
            
            print("\n모델별 성능 통계:")
            print(results.round(3))
            
            evaluation_results[cluster_name] = results
        
        return evaluation_results
    
    def analyze_feature_importance(self, cluster_models=None, top_n=10):
        """
        특성 중요도 분석
        
        Parameters:
        - cluster_models: 분석할 모델들
        - top_n: 상위 N개 특성
        
        Returns:
        - importance_results: 특성 중요도 결과
        """
        if cluster_models is None:
            cluster_models = self.cluster_models
        
        importance_results = {}
        
        for cluster_name, model_info in cluster_models.items():
            rf_model = model_info['all_models'].get('Random Forest')
            
            if rf_model and hasattr(rf_model, 'feature_importances_'):
                print(f"\n{'='*50}")
                print(f"{cluster_name} 특성 중요도 (Random Forest)")
                print(f"{'='*50}")
                
                # 특성 이름 가져오기
                if hasattr(rf_model, 'feature_names_in_'):
                    feature_names = rf_model.feature_names_in_
                else:
                    feature_names = [f'Feature_{i}' for i in range(len(rf_model.feature_importances_))]
                
                # 중요도 DataFrame 생성
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': rf_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                print(f"\n상위 {top_n}개 중요 특성:")
                print(importance_df.head(top_n))
                
                # 시각화
                plt.figure(figsize=(10, 6))
                plt.barh(importance_df['feature'][:top_n][::-1], 
                        importance_df['importance'][:top_n][::-1])
                plt.xlabel('Feature Importance')
                plt.title(f'Top {top_n} Feature Importances - {cluster_name}')
                plt.tight_layout()
                plt.show()
                
                importance_results[cluster_name] = importance_df
        
        return importance_results
    
    def calculate_vif(self, X):
        """
        VIF(Variance Inflation Factor) 계산
        
        Parameters:
        - X: 특성 데이터
        
        Returns:
        - vif_df: VIF 결과
        """
        vif_data = pd.DataFrame()
        vif_data["features"] = X.columns
        vif_data["VIF Factor"] = [
            variance_inflation_factor(X.values, i) 
            for i in range(X.shape[1])
        ]
        
        return vif_data.sort_values('VIF Factor', ascending=False).reset_index(drop=True)
    
    def plot_confusion_matrices(self, cluster_models=None):
        """
        혼동 행렬 시각화
        
        Parameters:
        - cluster_models: 시각화할 모델들
        """
        if cluster_models is None:
            cluster_models = self.cluster_models
        
        n_clusters = len(cluster_models)
        fig, axes = plt.subplots(1, n_clusters, figsize=(6*n_clusters, 5))
        
        if n_clusters == 1:
            axes = [axes]
        
        for idx, (cluster_name, model_info) in enumerate(cluster_models.items()):
            # 예측
            best_model = model_info['model']
            X_test = model_info['X_test']
            y_test = model_info['y_test']
            y_pred = best_model.predict(X_test)
            
            # 혼동 행렬
            cm = confusion_matrix(y_test, y_pred)
            
            # 시각화
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'{cluster_name}\n{model_info["model_name"]}')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
            
            # 분류 리포트
            print(f"\n{'='*50}")
            print(f"{cluster_name} 분류 리포트")
            print(f"{'='*50}")
            print(classification_report(y_test, y_pred))
        
        plt.tight_layout()
        plt.show()
    
    def predict_match(self, home_team, away_team, features, team_clusters):
        """
        단일 경기 예측
        
        Parameters:
        - home_team: 홈팀
        - away_team: 어웨이팀
        - features: 예측에 사용할 특성
        - team_clusters: 팀 클러스터 정보
        
        Returns:
        - prediction: 예측 결과
        """
        # 홈팀의 클러스터 확인
        home_cluster = None
        for cluster_name, model_info in self.cluster_models.items():
            if home_team in model_info.get('teams', []):
                home_cluster = cluster_name
                break
        
        if home_cluster is None:
            print(f"⚠️ {home_team}의 클러스터를 찾을 수 없습니다.")
            return None
        
        # 예측
        model = self.cluster_models[home_cluster]['model']
        
        # 예측 확률
        pred_proba = model.predict_proba(features)[0]
        pred_class = model.predict(features)[0]
        
        # 클래스 순서 확인
        classes = model.classes_
        
        prediction = {
            'home_team': home_team,
            'away_team': away_team,
            'cluster': home_cluster,
            'predicted_result': pred_class,
            'probabilities': {}
        }
        
        for i, cls in enumerate(classes):
            prediction['probabilities'][cls] = pred_proba[i]
        
        return prediction
    
    def predict_matches_batch(self, matches_df, features_list, team_clusters):
        """
        여러 경기 일괄 예측
        
        Parameters:
        - matches_df: 경기 정보 DataFrame
        - features_list: 각 경기의 특성 리스트
        - team_clusters: 팀 클러스터 정보
        
        Returns:
        - predictions_df: 예측 결과 DataFrame
        """
        predictions = []
        
        for idx, (_, match) in enumerate(matches_df.iterrows()):
            if idx < len(features_list):
                prediction = self.predict_match(
                    match['home_team'],
                    match['away_team'],
                    features_list[idx],
                    team_clusters
                )
                
                if prediction:
                    predictions.append(prediction)
        
        # DataFrame으로 변환
        predictions_df = pd.DataFrame(predictions)
        
        # 확률 컬럼 추가
        for result in ['win', 'draw', 'lose']:
            predictions_df[f'{result}_prob'] = predictions_df['probabilities'].apply(
                lambda x: x.get(result, 0)
            )
        
        return predictions_df
    
    def visualize_predictions(self, predictions_df):
        """
        예측 결과 시각화
        
        Parameters:
        - predictions_df: 예측 결과 DataFrame
        """
        fig, ax = plt.subplots(figsize=(12, len(predictions_df)*0.8))
        
        y_pos = np.arange(len(predictions_df))
        bar_width = 0.8
        
        # 확률 막대 그래프
        win_bars = ax.barh(y_pos, predictions_df['win_prob'], 
                          bar_width, label='Win', color='green', alpha=0.7)
        
        draw_bars = ax.barh(y_pos, predictions_df['draw_prob'], 
                           bar_width, left=predictions_df['win_prob'],
                           label='Draw', color='yellow', alpha=0.7)
        
        lose_bars = ax.barh(y_pos, predictions_df['lose_prob'], 
                           bar_width, 
                           left=predictions_df['win_prob'] + predictions_df['draw_prob'],
                           label='Lose', color='red', alpha=0.7)
        
        # 경기 레이블
        match_labels = [
            f"{row['home_team']} vs {row['away_team']}" 
            for _, row in predictions_df.iterrows()
        ]
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(match_labels)
        ax.invert_yaxis()
        ax.set_xlabel('Probability')
        ax.set_title('Match Prediction Probabilities', fontsize=16)
        ax.legend()
        
        # 확률 텍스트 추가
        for i, (idx, row) in enumerate(predictions_df.iterrows()):
            if row['win_prob'] > 0.15:
                ax.text(row['win_prob']/2, i, f"{row['win_prob']:.1%}", 
                       ha='center', va='center')
            if row['draw_prob'] > 0.15:
                ax.text(row['win_prob'] + row['draw_prob']/2, i, 
                       f"{row['draw_prob']:.1%}", ha='center', va='center')
            if row['lose_prob'] > 0.15:
                ax.text(row['win_prob'] + row['draw_prob'] + row['lose_prob']/2, i, 
                       f"{row['lose_prob']:.1%}", ha='center', va='center')
        
        plt.tight_layout()
        plt.show()


class ModelOptimizer:
    """모델 최적화 클래스"""
    
    @staticmethod
    def perform_feature_selection(X, y, method='random_forest', n_features=10):
        """
        특성 선택
        
        Parameters:
        - X: 특성 데이터
        - y: 타겟 변수
        - method: 'random_forest' or 'mutual_info'
        - n_features: 선택할 특성 수
        
        Returns:
        - selected_features: 선택된 특성 이름
        - X_selected: 선택된 특성만 포함한 데이터
        """
        from sklearn.feature_selection import SelectKBest, mutual_info_classif
        
        if method == 'random_forest':
            # Random Forest 기반 특성 선택
            rf = RandomForestClassifier(n_estimators=100, random_state=65)
            rf.fit(X, y)
            
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            selected_features = importance_df.head(n_features)['feature'].tolist()
            
        elif method == 'mutual_info':
            # Mutual Information 기반 특성 선택
            selector = SelectKBest(mutual_info_classif, k=n_features)
            selector.fit(X, y)
            
            selected_indices = selector.get_support(indices=True)
            selected_features = X.columns[selected_indices].tolist()
        
        else:
            raise ValueError("method는 'random_forest' 또는 'mutual_info'여야 합니다")
        
        X_selected = X[selected_features]
        
        print(f"✅ {n_features}개 특성 선택 완료: {selected_features}")
        return selected_features, X_selected
    
    @staticmethod
    def ensemble_predict(models, X, weights=None):
        """
        앙상블 예측
        
        Parameters:
        - models: 모델 리스트
        - X: 예측할 데이터
        - weights: 각 모델의 가중치
        
        Returns:
        - ensemble_pred: 앙상블 예측 결과
        """
        if weights is None:
            weights = [1/len(models)] * len(models)
        
        # 각 모델의 예측 확률
        predictions = []
        for model in models:
            pred_proba = model.predict_proba(X)
            predictions.append(pred_proba)
        
        # 가중 평균
        ensemble_proba = np.average(predictions, weights=weights, axis=0)
        
        # 최종 예측
        ensemble_pred = np.argmax(ensemble_proba, axis=1)
        
        return ensemble_pred, ensemble_proba


class ModelEvaluator:
    """모델 평가 클래스"""
    
    @staticmethod
    def plot_roc_curves(models, X_test, y_test):
        """
        ROC 커브 그리기
        
        Parameters:
        - models: 모델 딕셔너리
        - X_test: 테스트 데이터
        - y_test: 테스트 레이블
        """
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize
        
        # 다중 클래스를 위한 이진화
        classes = np.unique(y_test)
        y_test_bin = label_binarize(y_test, classes=classes)
        
        plt.figure(figsize=(10, 8))
        
        for model_name, model in models.items():
            y_score = model.predict_proba(X_test)
            
            # 각 클래스별 ROC 커브
            for i, class_name in enumerate(classes):
                if y_test_bin.shape[1] > 1:
                    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                else:
                    fpr, tpr, _ = roc_curve(y_test == class_name, y_score[:, i])
                
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, 
                        label=f'{model_name} - {class_name} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.show()
    
    @staticmethod
    def plot_learning_curves(model, X, y, cv=5):
        """
        학습 곡선 그리기
        
        Parameters:
        - model: 평가할 모델
        - X: 특성 데이터
        - y: 타겟 변수
        - cv: 교차검증 폴드 수
        """
        from sklearn.model_selection import learning_curve
        
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv, 
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy'
        )
        
        plt.figure(figsize=(10, 6))
        
        # 평균과 표준편차 계산
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # 학습 곡선 그리기
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation score')
        
        # 표준편차 영역 표시
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                        alpha=0.1, color='blue')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                        alpha=0.1, color='red')
        
        plt.xlabel('Training Size')
        plt.ylabel('Accuracy Score')
        plt.title('Learning Curves')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()


def run_modeling_pipeline(ml_datasets, team_clusters, save_models=True):
    """
    전체 모델링 파이프라인 실행
    
    Parameters:
    - ml_datasets: 클러스터별 ML 데이터셋
    - team_clusters: 팀 클러스터 정보
    - save_models: 모델 저장 여부
    
    Returns:
    - results: 모델링 결과
    """
    print("="*70)
    print("머신러닝 모델링 파이프라인")
    print("="*70)
    
    # 1. 모델 훈련
    print("\n[1단계: 모델 훈련]")
    predictor = MatchPredictor()
    cluster_models = predictor.train_cluster_models(ml_datasets)
    
    # 2. 모델 평가
    print("\n[2단계: 모델 평가]")
    evaluation_results = predictor.evaluate_models(cluster_models, n_iterations=30)
    
    # 3. 특성 중요도 분석
    print("\n[3단계: 특성 중요도 분석]")
    importance_results = predictor.analyze_feature_importance(cluster_models)
    
    # 4. 혼동 행렬 시각화
    print("\n[4단계: 혼동 행렬 분석]")
    predictor.plot_confusion_matrices(cluster_models)
    
    # 5. VIF 분석
    print("\n[5단계: 다중공선성 분석]")
    vif_results = {}
    for cluster_name, data in ml_datasets.items():
        print(f"\n{cluster_name} VIF 분석:")
        vif_df = predictor.calculate_vif(data['X'])
        print(vif_df.head(10))
        vif_results[cluster_name] = vif_df
    
    # 결과 정리
    results = {
        'predictor': predictor,
        'cluster_models': cluster_models,
        'evaluation_results': evaluation_results,
        'importance_results': importance_results,
        'vif_results': vif_results
    }
    
    # 모델 저장
    if save_models:
        print("\n[모델 저장]")
        import joblib
        from datetime import datetime
        
        date_str = datetime.now().strftime('%m%d')
        
        for cluster_name, model_info in cluster_models.items():
            model_filename = f'./models/{cluster_name}_model_{date_str}.pkl'
            joblib.dump(model_info['model'], model_filename)
            print(f"✅ {cluster_name} 모델 저장: {model_filename}")
    
    print("\n✅ 모델링 파이프라인 완료!")
    return results