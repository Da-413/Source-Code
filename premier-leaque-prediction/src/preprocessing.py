"""
데이터 전처리 모듈

크롤링한 데이터를 분석에 적합한 형태로 가공하는 함수들
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
import os
from datetime import datetime


class DataPreprocessor:
    """데이터 전처리 클래스"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.team_names = [
            'Bournemouth', 'Arsenal', 'Aston Villa', 'Brentford', 'Brighton & Hove Albion',
            'Chelsea', 'Crystal Palace', 'Everton', 'Fulham', 'Leeds United',
            'Leicester City', 'Liverpool', 'Manchester City', 'Manchester United',
            'Newcastle United', 'Nottingham Forest', 'Southampton', 'Tottenham Hotspur',
            'West Ham United', 'Wolverhampton Wanderers'
        ]
    
    def clean_data(self, record):
        """
        데이터 정제
        
        Parameters:
        - record: 원본 데이터프레임
        
        Returns:
        - cleaned_record: 정제된 데이터프레임
        """
        # 복사본 생성
        cleaned_record = record.copy()
        
        # 공백 제거
        cleaned_record = cleaned_record.apply(lambda x: x.str.strip() if x.dtype == "object" else x, axis=1)
        
        # 쉼표 제거
        cleaned_record = cleaned_record.apply(lambda x: x.str.replace(',', '') if x.dtype == "object" else x, axis=1)
        
        # 결측치 처리
        cleaned_record = cleaned_record.fillna(0)
        
        # 숫자형 변환
        cleaned_record = cleaned_record.apply(pd.to_numeric, errors='coerce')
        
        print(f"✅ 데이터 정제 완료: {cleaned_record.shape}")
        return cleaned_record
    
    def remove_multicollinearity(self, record, cols_to_drop=None):
        """
        다중공선성 제거
        
        Parameters:
        - record: 데이터프레임
        - cols_to_drop: 제거할 컬럼 리스트
        
        Returns:
        - record_clean: 다중공선성이 제거된 데이터프레임
        """
        if cols_to_drop is None:
            cols_to_drop = ['goals', 'goals_from_inside_box', 'shots', 'headed_clearences']
        
        record_clean = record.drop(cols_to_drop, axis=1, errors='ignore')
        print(f"✅ 다중공선성 제거 완료: {len(cols_to_drop)}개 컬럼 제거")
        
        return record_clean
    
    def scale_data(self, data, method='standard'):
        """
        데이터 스케일링
        
        Parameters:
        - data: 스케일링할 데이터
        - method: 'standard' or 'minmax'
        
        Returns:
        - scaled_data: 스케일링된 데이터
        - scaler: 사용된 스케일러 객체
        """
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("method는 'standard' 또는 'minmax'여야 합니다")
        
        scaled_array = scaler.fit_transform(data)
        scaled_data = pd.DataFrame(scaled_array, 
                                   index=data.index, 
                                   columns=data.columns)
        
        return scaled_data, scaler
    
    def create_head_to_head_matrix(self, df_against, team_names=None):
        """
        상대 전적 매트릭스 생성
        
        Parameters:
        - df_against: 상대 전적 데이터프레임
        - team_names: 팀 이름 리스트
        
        Returns:
        - against_matrix: 상대 전적 매트릭스
        """
        if team_names is None:
            team_names = self.team_names
        
        # 20x20 매트릭스 초기화
        against_matrix = pd.DataFrame(np.zeros((20, 20)), 
                                     columns=team_names,
                                     index=team_names)
        
        wp = df_against['winning_percentage']
        
        # 매트릭스 채우기
        for i in range(19):
            for j in range(i, 20):
                if i == j:
                    against_matrix.iloc[i, j] = 1
                else:
                    idx = int((-1) + i*(37-i)/2 + j)
                    against_matrix.iloc[i, j] = wp[idx]
                    against_matrix.iloc[j, i] = 1 - against_matrix.iloc[i, j]
        
        against_matrix.iloc[19, 19] = 1
        
        print("✅ 상대 전적 매트릭스 생성 완료")
        return against_matrix
    
    def calculate_home_advantage(self, match_results):
        """
        홈 어드밴티지 계산
        
        Parameters:
        - match_results: 경기 결과 데이터프레임
        
        Returns:
        - home_weights: 팀별 홈 어드밴티지 가중치
        - processed_results: 처리된 경기 결과
        """
        # 팀명 정규화
        team_name_mapping = {
            'Man City': 'Manchester City',
            'Man Utd': 'Manchester United',
            'Spurs': 'Tottenham Hotspur',
            'Wolves': 'Wolverhampton Wanderers',
            'Leicester': 'Leicester City',
            'Leeds': 'Leeds United',
            'Brighton': 'Brighton & Hove Albion',
            'West Ham': 'West Ham United',
            'Newcastle': 'Newcastle United',
            "Nott'm Forest": 'Nottingham Forest'
        }
        
        match_results = match_results.copy()
        match_results['Home_Team'] = match_results['Home_Team'].replace(team_name_mapping)
        match_results['Away_Team'] = match_results['Away_Team'].replace(team_name_mapping)
        
        # 경기 결과 파싱
        match_results['Home_Goals'] = match_results['Score'].str.split('-').str[0].astype(int)
        match_results['Away_Goals'] = match_results['Score'].str.split('-').str[1].astype(int)
        
        # 결과 분류
        match_results['Result'] = match_results.apply(
            lambda x: 'H' if x['Home_Goals'] > x['Away_Goals'] else 
                      ('D' if x['Home_Goals'] == x['Away_Goals'] else 'A'), axis=1
        )
        
        # 팀별 홈/어웨이 승률 계산
        teams = match_results['Home_Team'].unique()
        home_weights = pd.DataFrame(index=teams, 
                                   columns=['home_win_rate', 'away_win_rate', 'weight'])
        
        for team in teams:
            # 홈 경기
            home_games = match_results[match_results['Home_Team'] == team]
            if len(home_games) > 0:
                home_win_rate = ((home_games['Result'] == 'H').sum() + 
                               0.5*(home_games['Result'] == 'D').sum())
                home_win_rate /= len(home_games)
                home_weights.loc[team, 'home_win_rate'] = home_win_rate
            
            # 어웨이 경기
            away_games = match_results[match_results['Away_Team'] == team]
            if len(away_games) > 0:
                away_win_rate = ((away_games['Result'] == 'A').sum() + 
                               0.5*(away_games['Result'] == 'D').sum())
                away_win_rate /= len(away_games)
                home_weights.loc[team, 'away_win_rate'] = away_win_rate
        
        # 홈 어드밴티지 가중치 계산
        home_weights['weight'] = home_weights['home_win_rate'] - home_weights['away_win_rate']
        home_weights = home_weights.dropna()
        
        print("✅ 홈 어드밴티지 계산 완료")
        return home_weights, match_results
    
    def perform_variable_clustering(self, record_clean, n_clusters=2):
        """
        변수 클러스터링
        
        Parameters:
        - record_clean: 정제된 데이터
        - n_clusters: 클러스터 수
        
        Returns:
        - variable_labels: 변수 클러스터 레이블
        """
        # 표준화
        scaled_data, _ = self.scale_data(record_clean)
        
        # 변수 클러스터링 (전치 행렬 사용)
        variable_cluster = KMeans(n_clusters=n_clusters, algorithm='auto', random_state=65)
        variable_cluster.fit(scaled_data.T)
        
        # 결과 정리
        variable_labels = pd.DataFrame({
            'variables': record_clean.columns,
            'cluster': variable_cluster.labels_
        })
        
        print(f"✅ 변수 클러스터링 완료: {n_clusters}개 클러스터")
        return variable_labels
    
    def perform_team_clustering(self, record_clean, n_clusters=3):
        """
        팀 클러스터링
        
        Parameters:
        - record_clean: 정제된 데이터
        - n_clusters: 클러스터 수
        
        Returns:
        - team_labels: 팀 클러스터 레이블
        - linkage_matrix: 계층적 클러스터링 결과
        """
        # 표준화
        scaled_data, _ = self.scale_data(record_clean)
        
        # 계층적 클러스터링
        linkage_matrix = sch.linkage(y=scaled_data, method='ward', metric='euclidean')
        
        # 클러스터 할당
        team_labels = pd.DataFrame({
            'team': record_clean.index,
            'cluster': sch.fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        })
        
        print(f"✅ 팀 클러스터링 완료: {n_clusters}개 클러스터")
        return team_labels, linkage_matrix
    
    def prepare_match_features(self, home_team, away_team, record_clean, 
                              variable_labels=None, home_weights=None):
        """
        경기 예측을 위한 특성 생성
        
        Parameters:
        - home_team: 홈팀 이름
        - away_team: 어웨이팀 이름
        - record_clean: 정제된 통계 데이터
        - variable_labels: 변수 클러스터 정보
        - home_weights: 홈 어드밴티지 가중치
        
        Returns:
        - features: 예측에 사용할 특성
        """
        if home_team not in record_clean.index or away_team not in record_clean.index:
            print(f"⚠️ 팀 데이터 없음: {home_team} 또는 {away_team}")
            return None
        
        # 홈팀과 어웨이팀의 통계
        home_stats = record_clean.loc[home_team]
        away_stats = record_clean.loc[away_team]
        
        # 표준화
        scaler = StandardScaler()
        scaler.fit(record_clean)
        home_stats_scaled = scaler.transform([home_stats])[0]
        away_stats_scaled = scaler.transform([away_stats])[0]
        
        # 통계 차이를 특성으로 사용
        features = pd.DataFrame([home_stats_scaled - away_stats_scaled], 
                               columns=record_clean.columns)
        
        # 홈 어드밴티지 적용
        if home_weights is not None and home_team in home_weights.index and variable_labels is not None:
            weight = home_weights.loc[home_team, 'weight']
            
            for col in features.columns:
                var_cluster = variable_labels[variable_labels['variables'] == col]['cluster'].values[0]
                if var_cluster == 0:  # 공격 변수
                    features[col] *= (1 + weight)
                else:  # 수비 변수
                    features[col] *= (1 - weight)
        
        return features
    
    def save_data(self, data, filename, directory='./data'):
        """
        데이터를 CSV 파일로 저장
        
        Parameters:
        - data: 저장할 데이터프레임
        - filename: 파일명
        - directory: 저장 디렉토리
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        filepath = os.path.join(directory, filename)
        data.to_csv(filepath, index=True, header=True)
        print(f"✅ 데이터 저장 완료: {filepath}")
    
    def load_data(self, filename, directory='./data'):
        """
        CSV 파일에서 데이터 로드
        
        Parameters:
        - filename: 파일명
        - directory: 디렉토리
        
        Returns:
        - data: 로드된 데이터프레임
        """
        filepath = os.path.join(directory, filename)
        
        if not os.path.exists(filepath):
            print(f"⚠️ 파일이 존재하지 않습니다: {filepath}")
            return None
        
        data = pd.read_csv(filepath, index_col=0)
        print(f"✅ 데이터 로드 완료: {filepath}")
        return data
    
    def create_ml_dataset(self, record_clean, match_results, team_clusters, 
                         variable_labels=None, home_weights=None):
        """
        머신러닝을 위한 데이터셋 생성
        
        Parameters:
        - record_clean: 정제된 팀 통계
        - match_results: 경기 결과
        - team_clusters: 팀 클러스터 정보
        - variable_labels: 변수 클러스터 정보
        - home_weights: 홈 어드밴티지 가중치
        
        Returns:
        - ml_datasets: 클러스터별 ML 데이터셋
        """
        ml_datasets = {}
        
        # 클러스터별 팀 리스트
        cluster_teams = {}
        for cluster in team_clusters['cluster'].unique():
            cluster_teams[f'cluster_{cluster}'] = team_clusters[
                team_clusters['cluster'] == cluster
            ]['team'].tolist()
        
        # 각 클러스터별 데이터셋 생성
        for cluster_name, teams in cluster_teams.items():
            print(f"\n{cluster_name} 데이터셋 생성 중...")
            
            features_list = []
            labels_list = []
            
            for _, match in match_results.iterrows():
                home_team = match['Home_Team']
                away_team = match['Away_Team']
                
                # 해당 클러스터 팀이 포함된 경기만 선택
                if home_team in teams or away_team in teams:
                    # 특성 생성
                    features = self.prepare_match_features(
                        home_team, away_team, record_clean,
                        variable_labels, home_weights
                    )
                    
                    if features is not None:
                        features_list.append(features.values[0])
                        
                        # 레이블 생성 (홈팀 관점)
                        if match['Result'] == 'H':
                            labels_list.append('win')
                        elif match['Result'] == 'D':
                            labels_list.append('draw')
                        else:
                            labels_list.append('lose')
            
            if features_list:
                X = pd.DataFrame(features_list, columns=record_clean.columns)
                y = pd.Series(labels_list)
                
                ml_datasets[cluster_name] = {
                    'X': X,
                    'y': y,
                    'teams': teams
                }
                
                print(f"✅ {cluster_name}: {len(X)}개 샘플 생성 완료")
            else:
                print(f"⚠️ {cluster_name}: 데이터 생성 실패")
        
        return ml_datasets


class FeatureEngineer:
    """특성 공학 클래스"""
    
    @staticmethod
    def create_interaction_features(data):
        """
        상호작용 특성 생성
        
        Parameters:
        - data: 원본 데이터
        
        Returns:
        - data_with_interactions: 상호작용 특성이 추가된 데이터
        """
        data_with_interactions = data.copy()
        
        # 공격 효율성
        if 'goals' in data.columns and 'shots' in data.columns:
            data_with_interactions['goal_efficiency'] = data['goals'] / (data['shots'] + 1)
        
        # 수비 효율성
        if 'clean_sheets' in data.columns and 'goals_conceded' in data.columns:
            data_with_interactions['defensive_efficiency'] = (
                data['clean_sheets'] / (data['clean_sheets'] + data['goals_conceded'] + 1)
            )
        
        # 패스 정확도
        if 'passes' in data.columns and 'dispossessed' in data.columns:
            data_with_interactions['pass_accuracy'] = (
                data['passes'] / (data['passes'] + data['dispossessed'] + 1)
            )
        
        # 득점 다양성
        if all(col in data.columns for col in ['goals', 'penalties_scored', 'headed_goals']):
            data_with_interactions['goal_diversity'] = (
                1 - (data['penalties_scored'] + data['headed_goals']) / (data['goals'] + 1)
            )
        
        return data_with_interactions
    
    @staticmethod
    def create_form_features(match_results, team_name, n_matches=5):
        """
        최근 폼 기반 특성 생성
        
        Parameters:
        - match_results: 경기 결과
        - team_name: 팀 이름
        - n_matches: 고려할 최근 경기 수
        
        Returns:
        - form_features: 폼 관련 특성
        """
        # 팀의 최근 경기 추출
        team_matches = match_results[
            (match_results['Home_Team'] == team_name) | 
            (match_results['Away_Team'] == team_name)
        ].tail(n_matches)
        
        if len(team_matches) == 0:
            return pd.Series({
                'recent_wins': 0,
                'recent_draws': 0,
                'recent_losses': 0,
                'recent_goals_scored': 0,
                'recent_goals_conceded': 0,
                'recent_points': 0
            })
        
        wins = 0
        draws = 0
        losses = 0
        goals_scored = 0
        goals_conceded = 0
        
        for _, match in team_matches.iterrows():
            if match['Home_Team'] == team_name:
                goals_scored += match['Home_Goals']
                goals_conceded += match['Away_Goals']
                
                if match['Result'] == 'H':
                    wins += 1
                elif match['Result'] == 'D':
                    draws += 1
                else:
                    losses += 1
            else:  # Away team
                goals_scored += match['Away_Goals']
                goals_conceded += match['Home_Goals']
                
                if match['Result'] == 'A':
                    wins += 1
                elif match['Result'] == 'D':
                    draws += 1
                else:
                    losses += 1
        
        form_features = pd.Series({
            'recent_wins': wins,
            'recent_draws': draws,
            'recent_losses': losses,
            'recent_goals_scored': goals_scored,
            'recent_goals_conceded': goals_conceded,
            'recent_points': wins * 3 + draws,
            'recent_win_rate': wins / len(team_matches),
            'recent_goals_per_match': goals_scored / len(team_matches),
            'recent_goals_conceded_per_match': goals_conceded / len(team_matches)
        })
        
        return form_features


def preprocess_pipeline(raw_data, save_processed=True):
    """
    전체 전처리 파이프라인
    
    Parameters:
    - raw_data: 크롤링한 원본 데이터
    - save_processed: 처리된 데이터 저장 여부
    
    Returns:
    - processed_data: 전처리된 데이터 딕셔너리
    """
    preprocessor = DataPreprocessor()
    
    # 1. 데이터 정제
    print("\n[1단계: 데이터 정제]")
    cleaned_data = preprocessor.clean_data(raw_data)
    
    # 2. 다중공선성 제거
    print("\n[2단계: 다중공선성 제거]")
    record_clean = preprocessor.remove_multicollinearity(cleaned_data)
    
    # 3. 특성 공학
    print("\n[3단계: 특성 공학]")
    feature_engineer = FeatureEngineer()
    record_with_features = feature_engineer.create_interaction_features(record_clean)
    
    # 4. 스케일링
    print("\n[4단계: 데이터 스케일링]")
    scaled_data, scaler = preprocessor.scale_data(record_with_features)
    
    # 5. 클러스터링
    print("\n[5단계: 클러스터링]")
    variable_labels = preprocessor.perform_variable_clustering(record_clean, n_clusters=2)
    team_labels, linkage_matrix = preprocessor.perform_team_clustering(record_clean, n_clusters=3)
    
    # 결과 정리
    processed_data = {
        'cleaned_data': cleaned_data,
        'record_clean': record_clean,
        'record_with_features': record_with_features,
        'scaled_data': scaled_data,
        'scaler': scaler,
        'variable_labels': variable_labels,
        'team_labels': team_labels,
        'linkage_matrix': linkage_matrix
    }
    
    # 데이터 저장
    if save_processed:
        print("\n[데이터 저장]")
        date_str = datetime.now().strftime('%m.%d')
        
        preprocessor.save_data(cleaned_data, f'record_{date_str}.csv')
        preprocessor.save_data(record_clean, f'record_clean_{date_str}.csv')
        preprocessor.save_data(variable_labels, f'variable_clusters_{date_str}.csv')
        preprocessor.save_data(team_labels, f'team_clusters_{date_str}.csv')
    
    print("\n✅ 전처리 파이프라인 완료!")
    return processed_data