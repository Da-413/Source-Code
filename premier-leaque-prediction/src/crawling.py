"""
Premier League 데이터 크롤링 모듈

이 모듈은 Premier League 공식 웹사이트에서 팀 통계 데이터를 크롤링하는 기능을 제공합니다.
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import numpy as np
import pandas as pd


class PremierLeagueCrawler:
    """Premier League 통계 데이터 크롤러"""
    
    def __init__(self):
        """Chrome 드라이버 초기화"""
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
        self.base_url = 'https://www.premierleague.com/stats/top/clubs/wins?se=489'
        
        # 통계 카테고리 정의
        self.stat_categories = {
            'basic': {
                'indices': [1, 2, 3, 4],
                'names': ['wins', 'loses', 'goals', 'yellow_cards'],
                'category_num': 1
            },
            'attack': {
                'indices': [1, 2, 4, 5, 7, 8, 9, 10],
                'names': ['shots', 'shots_on_target', 'headed_goals', 'penalties_scored', 
                         'goals_from_inside_box', 'goals_from_outside_box', 
                         'goals_from_counter_attack', 'offsides'],
                'category_num': 2
            },
            'defense': {
                'indices': [1, 2, 3, 4, 5, 6, 8, 9],
                'names': ['clean_sheets', 'goals_conceded', 'saves', 'blocks', 
                         'interceptions', 'tackles', 'clearences', 'headed_clearences'],
                'category_num': 3
            },
            'passing': {
                'indices': [1, 2, 3, 5, 6],
                'names': ['passes', 'through_balls', 'long_passes', 'crosses', 'corners'],
                'category_num': 4
            }
        }
        
        self.team_names = [
            'Bournemouth', 'Arsenal', 'Aston Villa', 'Brentford', 'Brighton & Hove Albion',
            'Chelsea', 'Crystal Palace', 'Everton', 'Fulham', 'Leeds United',
            'Leicester City', 'Liverpool', 'Manchester City', 'Manchester United',
            'Newcastle United', 'Nottingham Forest', 'Southampton', 'Tottenham Hotspur',
            'West Ham United', 'Wolverhampton Wanderers'
        ]
    
    def start(self):
        """크롤링 시작 - Premier League 통계 페이지 접속"""
        self.driver.get(self.base_url)
        time.sleep(3)
    
    def crawl_stat_category(self, category_name):
        """
        특정 카테고리의 통계 데이터를 크롤링
        
        Parameters:
        -----------
        category_name : str
            크롤링할 카테고리 이름 ('basic', 'attack', 'defense', 'passing')
            
        Returns:
        --------
        list : 크롤링된 DataFrame 리스트
        """
        category = self.stat_categories[category_name]
        indices = category['indices']
        stat_names = category['names']
        category_num = category['category_num']
        
        dataframes = []
        
        for n, i in enumerate(indices):
            # 통계 선택
            path = f'//*[@id="mainContent"]/div[2]/div/div[2]/div[2]/div[{category_num}]/nav/ul/li[{i}]'
            self.driver.find_element(by=By.XPATH, value=path).click()
            self.driver.implicitly_wait(3)
            
            # 전체 시즌 데이터 선택
            self._select_all_season_data()
            
            # 데이터 추출
            table_data = self.driver.find_elements(by=By.CSS_SELECTOR, value='.statsTableContainer')
            time.sleep(2)
            
            # 특수 케이스 처리
            if stat_names[n] == 'penalties_scored':
                arr = np.array(table_data[0].text.split('\n')).reshape(19, 3)
            else:
                arr = np.array(table_data[0].text.split('\n')).reshape(20, 3)
            
            # DataFrame 생성
            colnames = ['rank', 'team_name', stat_names[n]]
            df = pd.DataFrame(arr, columns=colnames).set_index('team_name')
            df = df.drop(['rank'], axis=1)
            
            dataframes.append(df)
            
            print(f"✓ {stat_names[n]} 데이터 수집 완료")
        
        return dataframes
    
    def crawl_additional_stats(self):
        """
        추가 통계 데이터 크롤링 (big_chances_missed, dispossessed)
        
        Returns:
        --------
        dict : 추가 통계 DataFrame 딕셔너리
        """
        additional_stats = {}
        
        # Big Chances Missed
        self._crawl_single_additional_stat('big_chances_missed', 12, additional_stats)
        
        # Dispossessed
        self._crawl_single_additional_stat('dispossessed', 16, additional_stats)
        
        return additional_stats
    
    def crawl_head_to_head_records(self):
        """
        팀 간 상대 전적 데이터 크롤링
        
        Returns:
        --------
        pd.DataFrame : 상대 전적 데이터프레임
        """
        # 상대 전적 페이지로 이동
        self.driver.find_element(By.XPATH, '//*[@id="mainContent"]/div[2]/nav/div/ul/li[6]/a').click()
        self.driver.implicitly_wait(3)
        
        list_against = []
        
        print("상대 전적 데이터 크롤링 중...")
        
        for i in range(20):
            for j in range(i+1, 20):
                # 팀 선택 및 전적 데이터 추출
                self._select_teams_and_get_record(i, j, list_against)
            
            print(f"✓ {self.team_names[i]} 팀 상대 전적 수집 완료 ({i+1}/20)")
        
        # DataFrame 생성
        df_against = self._create_head_to_head_dataframe(list_against)
        
        print("\n✅ 상대 전적 데이터 크롤링 완료!")
        
        return df_against
    
    def crawl_match_results(self, seasons=3):
        """
        최근 시즌의 경기 결과 데이터 크롤링
        
        Parameters:
        -----------
        seasons : int
            크롤링할 시즌 수 (최대 6)
            
        Returns:
        --------
        pd.DataFrame : 경기 결과 데이터프레임
        """
        self.driver.get('https://www.premierleague.com/results')
        
        all_results = []
        season_names = ['22/23', '21/22', '20/21', '19/20', '18/19', '17/18']
        
        for i in range(min(seasons, 6)):
            print(f"\n{season_names[i]} 시즌 데이터 크롤링 중...")
            
            # 시즌 선택
            self._select_season(i)
            
            # 모든 경기 로드
            self._scroll_to_load_all_matches()
            
            # 데이터 추출 및 정제
            df = self._extract_season_results(season_names[i])
            
            all_results.append(df)
            print(f"✓ {season_names[i]} 시즌: {len(df)}경기 수집 완료")
        
        # 전체 결과 통합
        match_results = pd.concat(all_results, ignore_index=True)
        
        return match_results
    
    def close(self):
        """드라이버 종료"""
        self.driver.quit()
    
    # Private 메서드들
    def _select_all_season_data(self):
        """전체 시즌 데이터 선택"""
        self.driver.execute_script("window.scrollTo(0, 250)")
        self.driver.implicitly_wait(3)
        
        button1 = self.driver.find_element(
            By.XPATH, 
            '//*[@id="mainContent"]/div[2]/div/div[2]/div[1]/section/div[1]/div[2]'
        )
        self.driver.execute_script("arguments[0].click();", button1)
        time.sleep(1)
        
        button2 = self.driver.find_element(
            By.XPATH, 
            '//*[@id="mainContent"]/div[2]/div/div[2]/div[1]/section/div[1]/ul/li[2]'
        )
        self.driver.execute_script("arguments[0].click();", button2)
        time.sleep(1)
    
    def _crawl_single_additional_stat(self, stat_name, li_index, stats_dict):
        """단일 추가 통계 크롤링"""
        # 드롭다운 열기
        self.driver.find_element(
            by=By.XPATH, 
            value='//*[@id="mainContent"]/div[2]/div/div[2]/div[1]/div[1]/div'
        ).click()
        
        if stat_name == 'dispossessed':
            self.driver.execute_script("window.scrollTo(0, 500)")
            time.sleep(2)
        else:
            time.sleep(1)
        
        # 통계 선택
        self.driver.find_element(
            by=By.XPATH, 
            value=f'//*[@id="mainContent"]/div[2]/div/div[2]/div[1]/div[1]/ul/li[{li_index}]'
        ).click()
        time.sleep(3)
        
        # 전체 시즌 데이터 선택
        self._select_all_season_data()
        
        # 데이터 추출
        crawl_data = self.driver.find_elements(by=By.CSS_SELECTOR, value='.statsTableContainer')
        arr = np.array(crawl_data[0].text.split('\n')).reshape(20, 3)
        colnames = ['rank', 'team_name', stat_name]
        
        df = pd.DataFrame(arr, columns=colnames).set_index('team_name')
        df = df.drop(['rank'], axis=1)
        
        stats_dict[stat_name] = df
        
        print(f"✓ {stat_name} 데이터 수집 완료")
    
    def _select_teams_and_get_record(self, i, j, list_against):
        """두 팀을 선택하고 상대 전적 데이터 추출"""
        # 왼쪽 팀 선택
        path_i = f'//*[@id="mainContent"]/div[2]/div/div[2]/div/section[2]/div/div[1]/div[{i+1}]'
        self.driver.find_element(
            By.XPATH, 
            '//*[@id="mainContent"]/div[2]/div/div[2]/div/section[1]/div[2]/div[1]/div[1]'
        ).click()
        self.driver.implicitly_wait(3)
        lteam = self.driver.find_element(By.XPATH, path_i)
        self.driver.execute_script("arguments[0].click();", lteam)
        self.driver.implicitly_wait(3)
        
        # 오른쪽 팀 선택
        path_j = f'//*[@id="mainContent"]/div[2]/div/div[2]/div/section[2]/div/div[1]/div[{j+1}]'
        self.driver.find_element(
            By.XPATH, 
            '//*[@id="mainContent"]/div[2]/div/div[2]/div/section[1]/div[2]/div[3]/div[1]'
        ).click()
        self.driver.implicitly_wait(3)
        rteam = self.driver.find_element(By.XPATH, path_j)
        self.driver.execute_script("arguments[0].click();", rteam)
        time.sleep(1)
        
        # 전적 데이터 추출
        w = self.driver.find_element(
            By.XPATH, 
            '//*[@id="mainContent"]/div[2]/div/div[2]/div/section[4]/div[1]/table/tbody/tr/td[1]'
        ).text
        d = self.driver.find_element(
            By.XPATH, 
            '//*[@id="mainContent"]/div[2]/div/div[2]/div/section[4]/div[1]/table/tbody/tr/td[2]'
        ).text.strip('Drawn: ')
        l = self.driver.find_element(
            By.XPATH, 
            '//*[@id="mainContent"]/div[2]/div/div[2]/div/section[4]/div[1]/table/tbody/tr/td[3]'
        ).text
        
        list_against.extend([w, d, l])
    
    def _create_head_to_head_dataframe(self, list_against):
        """상대 전적 리스트를 DataFrame으로 변환"""
        index = []
        for i in range(20):
            for j in range(i+1, 20):
                index.append(f'{self.team_names[i]}_vs_{self.team_names[j]}')
        
        arr_against = np.array(list_against).reshape(len(list_against)//3, 3)
        df_against = pd.DataFrame(arr_against, columns=['wins', 'draws', 'loses'])
        df_against.index = index
        df_against = df_against.astype('int')
        
        # 승률 계산
        df_against['winning_percentage'] = (
            (df_against['wins'] + 0.5*df_against['draws']) / 
            (df_against['wins'] + df_against['draws'] + df_against['loses'])
        )
        
        return df_against
    
    def _select_season(self, season_index):
        """시즌 선택"""
        button1 = self.driver.find_element(
            By.XPATH, 
            '//*[@id="mainContent"]/div[3]/div[1]/section/div[3]/div[2]'
        )
        self.driver.execute_script("arguments[0].click();", button1)
        self.driver.implicitly_wait(3)
        
        button2 = self.driver.find_element(
            By.XPATH, 
            f'//*[@id="mainContent"]/div[3]/div[1]/section/div[3]/ul/li[{season_index+1}]'
        )
        self.driver.execute_script("arguments[0].click();", button2)
        self.driver.implicitly_wait(3)
    
    def _scroll_to_load_all_matches(self):
        """모든 경기를 로드하기 위해 스크롤"""
        scroll_location = self.driver.execute_script("return document.body.scrollHeight")
        while True:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(2)
            scroll_height = self.driver.execute_script("return document.body.scrollHeight")
            if scroll_location == scroll_height:
                break
            else:
                scroll_location = scroll_height
    
    def _extract_season_results(self, season_name):
        """시즌 결과 추출 및 정제"""
        results_section = self.driver.find_element(
            By.XPATH, 
            '//*[@id="mainContent"]/div[3]/div[1]/div[2]/section'
        )
        season_data = results_section.text.replace('\n', ',').split(',')
        
        # 데이터 정제
        clean_results = []
        for item in season_data:
            if item.count(' ') < 3:  # 팀명과 점수만 추출
                clean_results.append(item)
        
        # DataFrame 생성
        arr = np.array(clean_results).reshape(-1, 3)
        df = pd.DataFrame(arr, columns=['Home_Team', 'Score', 'Away_Team'])
        df['Season'] = season_name
        
        return df