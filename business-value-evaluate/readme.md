# 📈 기업 가치 평가 모델 분석

본 프로젝트는 웹 크롤링을 활용해 국내 상장 기업의 재무 데이터를 수집하고, 다양한 재무 지표를 기반으로 기업의 가치를 분석하는 과정을 다룹니다.

## 🔍 주요 내용
- 웹 크롤링을 통한 재무 데이터 수집 (FnGuide)
- 재무 지표 정제 및 전처리
- 가치 평가 모델 계산 로직 구현
- 재무 건전성, 수익성, 성장성 지표 종합 분석

---

# 🧠 반도체 결함 이미지 탐지 프로젝트

본 프로젝트는 반도체 제조 공정에서 발생하는 이상 소자를 이미지 기반으로 탐지하기 위해 다양한 머신러닝/딥러닝 모델을 실험한 프로젝트입니다.

## 🔧 주요 내용
- 반도체 이미지 전처리 및 증강
- 이상 탐지 모델: Isolation Forest, One-Class SVM, AutoEncoder
- 성능 비교 및 최종 모델 하이퍼파라미터 튜닝 (Bayesian Optimization)
- 결과 시각화 및 평가

---

## 📌 기술 스택
- Python, Selenium, Pandas, Numpy
- Sklearn, TensorFlow, PyTorch
- Matplotlib, Seaborn
- Web Crawling, AutoML, Feature Engineering

---

# 📂 프로젝트 세부 코드

## [기업 가치 평가 모델](#기업-가치-평가-모델)
```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import numpy as np
import time, os
```


```python
driver = webdriver.Chrome(service = Service(ChromeDriverManager().install()))
```


```python
comp = pd.read_csv("C:/Users/gyoo4/OneDrive/바탕 화면/데이터마이닝/company.csv", encoding = 'cp949')
#comp2 = pd.read_csv("C:/Users/gyoo4/OneDrive/바탕 화면/데이터마이닝/데이터마이닝_기업.csv")
comp_name = comp['기업명']
comp_code = comp['기업코드']
comp = pd.concat([comp_name, comp_code], axis = 1)

#os.chdir('C:/Users/gyoo4/OneDrive/바탕 화면/데이터마이닝')
#comp.to_csv('company.csv', index = False, encoding = 'cp949')
```


```python
driver.get('http://comp.fnguide.com/SVO2/ASP/SVD_Main.asp?pGB=1&gicode=A005930&cID=&MenuYn=Y&ReportGB=D&NewMenuID=Y&stkGb=701')

value = [0] * len(comp_code) #기업가치
market_cap = [0] * len(comp_code) #시가총액
B = [0] * len(comp_code) #지배주주자본
ROE = [0] * len(comp_code) 
current_assets = [0] * len(comp_code) #유동자산
current_liabilities = [0] * len(comp_code) #유동부채
quick_assets = [0]  * len(comp_code) #비유동자산
owner_capital = [0] * len(comp_code) #자기자본 = 자본총계
total_capital = [0] * len(comp_code) #총자본 = 자산총계
liabilities = [0] * len(comp_code) #부채
interest_expense = [0] * len(comp_code) #이자비용
operating_profit = [0] * len(comp_code) #영업이익
gross_profit = [0] * len(comp_code) #매출총이익
revenue = [0] * len(comp_code) #매출액
revenue_B = [0] * len(comp_code) #전기매출액
net_income = [0] * len(comp_code) #당기순이익
net_income_B = [0] * len(comp_code) #전기순이익


K = 8.09 # 한국신용평가에서 제공하는 BBB-등급 회사채의 1년 수익률

for i in range(len(comp_code)):
    
    #기업 검색창 열기
    driver.find_element(By.XPATH, '/html/body/div[2]/div/div[1]/div[1]/div[2]/form/div/a').click()
    time.sleep(1)
    
    #검색창으로 이동하여 기업 종목코드 입력
    driver.switch_to.window(driver.window_handles[1])
    driver.find_element(By.XPATH, '//*[@id="txtSearchKey"]').send_keys(comp_code[i])
    driver.find_element(By.XPATH, '//*[@id="btnSearch"]').click()
    time.sleep(1)
    
    #검색된 기업 클릭
    driver.find_element(By.XPATH, '//*[@id="body_contents"]/tr').click()
    time.sleep(1)
    
    #원래 창으로 돌아와서 기업개요-Snapshot 클릭
    driver.switch_to.window(driver.window_handles[0])
    driver.find_element(By.XPATH, '//*[@id="compGnb"]/ul/li[1]/ul/li[1]/a[1]').click()
    time.sleep(1)
    #시가총액
    market_cap[i] = int(driver.find_element(By.XPATH, '//*[@id="svdMainGrid1"]/table/tbody/tr[5]/td[1]').text.replace(',','')) 
    
    #연결재무제표로 선택
    driver.find_element(By.XPATH, '//*[@id="divHighFis"]/a[1]').click()
    
    revenue[i] = int(driver.find_element(By.XPATH, '//*[@id="highlight_D_A"]/table/tbody/tr[1]/td[2]').text.replace(',','')) #매출액 또는 이자수익 또는 보험료수익
    revenue_B[i] = int(driver.find_element(By.XPATH, '//*[@id="highlight_D_A"]/table/tbody/tr[1]/td[1]').text.replace(',',''))
   
    #snapshot 재무제표의 행 길이(행 길이가 기업마다 다름)
    row_length = int((driver.find_element(By.XPATH, '//*[@id="highlight_D_A"]/table/tbody').text.count('\n') - driver.find_element(By.XPATH, '//*[@id="highlight_D_A"]/table/tbody').text.count('원') + 1) / 2)
    
    #해당 문자열이 있는 j에서 값 추출
    for j in range(row_length):
             
        if driver.find_element(By.XPATH, '//*[@id="highlight_D_A"]/table/tbody/tr[{}]'.format(j+1)).text.replace('\n', ' ').split(' ')[0] == '영업이익':
            operating_profit[i] = int(driver.find_element(By.XPATH, '//*[@id="highlight_D_A"]/table/tbody/tr[{}]/td[2]'.format(j+1)).text.replace(',',''))
            
        elif driver.find_element(By.XPATH, '//*[@id="highlight_D_A"]/table/tbody/tr[{}]'.format(j+1)).text.replace('\n', ' ').split(' ')[0] == '당기순이익':
            net_income[i] = int(driver.find_element(By.XPATH, '//*[@id="highlight_D_A"]/table/tbody/tr[{}]/td[2]'.format(j+1)).text.replace(',',''))
            net_income_B[i] = int(driver.find_element(By.XPATH, '//*[@id="highlight_D_A"]/table/tbody/tr[{}]/td[1]'.format(j+1)).text.replace(',',''))
            
        elif driver.find_element(By.XPATH, '//*[@id="highlight_D_A"]/table/tbody/tr[{}]'.format(j+1)).text.replace('\n', ' ').split(' ')[0] == '자산총계':
            total_capital[i] = int(driver.find_element(By.XPATH, '//*[@id="highlight_D_A"]/table/tbody/tr[{}]/td[2]'.format(j+1)).text.replace(',',''))
        
        elif driver.find_element(By.XPATH, '//*[@id="highlight_D_A"]/table/tbody/tr[{}]'.format(j+1)).text.replace('\n', ' ').split(' ')[0] == '부채총계':
            liabilities[i] = int(driver.find_element(By.XPATH, '//*[@id="highlight_D_A"]/table/tbody/tr[{}]/td[2]'.format(j+1)).text.replace(',',''))
            
        elif driver.find_element(By.XPATH, '//*[@id="highlight_D_A"]/table/tbody/tr[{}]'.format(j+1)).text.replace('\n', ' ').split(' ')[0] == '자본총계':
            owner_capital[i] = int(driver.find_element(By.XPATH, '//*[@id="highlight_D_A"]/table/tbody/tr[{}]/td[2]'.format(j+1)).text.replace(',',''))
            
        elif driver.find_element(By.XPATH, '//*[@id="highlight_D_A"]/table/tbody/tr[{}]'.format(j+1)).text.split('\n')[0].strip() == '지배주주지분':
            B[i] = int(driver.find_element(By.XPATH, '//*[@id="highlight_D_A"]/table/tbody/tr[{}]/td[2]'.format(j+1)).text.replace(',',''))
            
        elif driver.find_element(By.XPATH, '//*[@id="highlight_D_A"]/table/tbody/tr[{}]'.format(j+1)).text.replace('\n', ' ').split(' ')[0] == 'ROE':
            ROE[i] = int(driver.find_element(By.XPATH, '//*[@id="highlight_D_A"]/table/tbody/tr[{}]/td[2]'.format(j+1)).text.replace('.',''))/100
           
        else:
            continue
    
    #snapshot에서 재무제표 칸으로 이동
    driver.find_element(By.XPATH, '//*[@id="compGnb"]/ul/li[1]/ul/li[1]/a[3]').click()
    driver.implicitly_wait(3)
    
    #재무제표 칸에서 값 추출. 금융기업들은 손익계산서가 일반 기업들과 달라서 예외 처리
    try:
        current_assets[i] = int(driver.find_element(By.XPATH, '//*[@id="p_grid2_2"]/td[2]').text.replace(',','')) #유동자산 
    except:
        pass
    
    try:
        current_liabilities[i] = int(driver.find_element(By.XPATH, '//*[@id="p_grid2_6"]/td[2]').text.replace(',','')) #유동부채 
    except:
        pass
    
    try:
        quick_assets[i] = current_assets[i] - int(driver.find_element(By.XPATH, '//*[@id="divDaechaY"]/table/tbody/tr[3]/td[3]').text.replace(',','')) #당좌자산
    except:
        pass
     
    try:
        if driver.find_element(By.XPATH, '//*[@id="divSonikY"]/table/tbody/tr[3]').text.split('\n')[0] == '매출총이익':
            gross_profit[i] = int(driver.find_element(By.XPATH, '//*[@id="divSonikY"]/table/tbody/tr[3]/td[3]').text.replace(',','')) #매출총이익
        else:
            gross_profit[i] = 0
    except:
        pass   
    
    if gross_profit[i] == 0:
        try:
            gross_profit[i] = int(driver.find_element(By.XPATH, '//*[@id="divSonikY"]/table/tbody/tr[8]/td[3]').text.replace(',','')) #매출총이익이 없으면 순영업수익
        except:
            gross_profit[i] = int(driver.find_element(By.XPATH, '//*[@id="divSonikY"]/table/tbody/tr[35]/td[3]').text.replace(',','')) #순영업수익이 없으면 영업이익, 보험사들은 순영업수익이 없음
    else:
        continue
        
    
```


```python
current_ratio = [0] * len(comp_code) #유동비율
net_working_capital_ratio = [0] * len(comp_code) #순운전자본비율
liabilities_ratio = [0] * len(comp_code) #부채비율
owner_capital_ratio = [0] * len(comp_code) #자기자본비율
gross_profit_margin = [0] * len(comp_code) #매출액총이익률
ROA = [0] * len(comp_code) #총자본영업이익률
ROI = [0] * len(comp_code) #총자본순이익률
asset_turnover_ratio = [0] * len(comp_code) #총자산회전율
revenue_growth_rate = [0] * len(comp_code) #매출액증가율
net_income_growth_rate = [0] * len(comp_code) #순이익증가율

for i in range(len(comp_code)):
    
    if B[i] == 0:
        B[i] = owner_capital[i]
    
    if current_assets[i] == 0:
        current_assets[i] = total_capital[i] #자산을 유동자산으로 취급
   
    if current_liabilities[i] == 0:
        current_liabilities[i] = liabilities[i] #부채를 유동부채로 취급
    
for i in range(len(comp_code)):
    
    value[i] = B[i] + ( B[i] * (ROE[i] - K) / K )
    
    try:
        current_ratio[i] = current_assets[i] / current_liabilities[i]
    except ZeroDivisionError:
        current_ratio[i] = 0
        
    try:
        revenue_growth_rate[i] = (revenue[i] - revenue_B[i]) / revenue_B[i]
    except ZeroDivisionError:
        revenue_growth_rate[i] = 0
        
    try:
        net_income_growth_rate[i] = (net_income[i] - net_income_B[i]) / net_income_B[i]
    except ZeroDivisionError:
        net_income_growth_rate[i] = 0
        
    net_working_capital_ratio[i] = (current_assets[i] - current_liabilities[i]) / total_capital[i]
    
    liabilities_ratio[i] = liabilities[i] / owner_capital[i]
    
    owner_capital_ratio[i] = owner_capital[i] / total_capital[i]
    
    gross_profit_margin[i] = gross_profit[i] / revenue[i]
    
    ROA[i] = operating_profit[i] / total_capital[i]
    
    ROI[i] = net_income[i] / total_capital[i]
    
    asset_turnover_ratio[i] = revenue[i] / total_capital[i]
    
```


```python
driver.get('http://comp.fnguide.com/SVO2/ASP/SVD_Main.asp?pGB=1&gicode=A005930&cID=&MenuYn=Y&ReportGB=D&NewMenuID=Y&stkGb=701')
```


```python
stocks = [0] * len(comp_code)

for i in range(len(comp_code)):
    
    #기업 검색창 열기
    driver.find_element(By.XPATH, '/html/body/div[2]/div/div[1]/div[1]/div[2]/form/div/a').click()
    time.sleep(1)
    
    #검색창으로 이동하여 기업 종목코드 입력
    driver.switch_to.window(driver.window_handles[1])
    driver.find_element(By.XPATH, '//*[@id="txtSearchKey"]').send_keys(comp_code[i])
    driver.find_element(By.XPATH, '//*[@id="btnSearch"]').click()
    time.sleep(1)
    
    #검색된 기업 클릭
    driver.find_element(By.XPATH, '//*[@id="body_contents"]/tr').click()
    time.sleep(1)
    
    #원래 창으로 돌아와서 기업개요-Snapshot 클릭
    driver.switch_to.window(driver.window_handles[0])
    driver.find_element(By.XPATH, '//*[@id="compGnb"]/ul/li[1]/ul/li[1]/a[1]').click()
    time.sleep(1)

    stocks[i] = int(driver.find_element(By.XPATH, '//*[@id="svdMainGrid1"]/table/tbody/tr[7]/td[1]').text.split('/')[0].replace(',',''))
```


```python
comp_value = pd.DataFrame(zip(market_cap, comp_name))
comp_value.columns = ('시가총액', '기업명')
comp_value['RIM기업가치'] = value
comp_value = comp_value.assign(매출액증가율 = revenue_growth_rate, 순이익증가율 = net_income_growth_rate, 유동비율 = current_ratio, 순운전자본비율 = net_working_capital_ratio,
                               부채비율 = liabilities_ratio, 자기자본비율 = owner_capital_ratio, 매출총이익률 = gross_profit_margin, ROA = ROA, ROI = ROI, 총자산회전율 = asset_turnover_ratio)
comp_value = comp_value.set_index('기업명')
comp_value = comp_value.astype('float')

os.chdir('C:/Users/gyoo4/OneDrive/바탕 화면/데이터마이닝')
comp_value.to_csv('data_re.csv', index = True, encoding = 'cp949')
```


```python
import FinanceDataReader as fdr

comp_value = pd.read_csv('c:/Users/gyoo4/OneDrive/바탕 화면/데이터마이닝/data_rere.csv', index_col = '기업명', encoding = 'cp949')
comp_value = comp_value.drop(['시가총액'], axis = 1)
comp_value['시가총액'] = [0] * len(comp_value)

for i in range(len(comp_code)):
    df = fdr.DataReader(comp_code[i].lstrip('A'), '2023-02-01', '2023-05-26')
    cap = sum(df['Close']) / len(df['Close'])
    
    comp_value.loc[comp_name[i], '시가총액'] = cap
```


```python
for i in range(len(comp_code)):
    comp_value.loc[comp_name[i],'시가총액'] = stocks[i] * comp_value.loc[comp_name[i],'시가총액'] / 100000000
```


```python
os.chdir('C:/Users/gyoo4/OneDrive/바탕 화면/데이터마이닝')
comp_value.to_csv('data_re.csv', index = True, encoding = 'cp949')
```


```python
import statsmodels.formula.api as sm  
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('c:/Users/gyoo4/OneDrive/바탕 화면/데이터마이닝/data_re.csv', index_col = '기업명', encoding = 'cp949')
data['target'] = data['시가총액'] - data['RIM기업가치']
#data = data.drop(['시가총액', 'RIM기업가치'], axis = 1)

scaler = StandardScaler()

scaler = scaler.fit_transform(data)
data = pd.DataFrame(scaler).set_index(comp_name)
data.columns = ['시가총액', 'RIM기업가치', '매출액증가율', '순이익증가율', '유동비율', '순운전자본비율', '부채비율', '자기자본비율', '매출총이익률', 'ROA', 'ROI', '총자산회전율', 'target']
```


```python
model1 = sm.ols('시가총액 ~ RIM기업가치 + 매출액증가율 + 순이익증가율 + 유동비율 + 순운전자본비율 + 부채비율 + 자기자본비율 + 매출총이익률 + ROA + ROI + 총자산회전율', data = data).fit()
print(model1.summary(), '\n')

model2 = sm.ols('시가총액 ~ RIM기업가치', data = data).fit()
print(model2.summary(), '\n')

model3 = sm.ols('target ~ 매출액증가율 + 순이익증가율 + 유동비율 + 순운전자본비율 + 부채비율 + 자기자본비율 + 매출총이익률 + ROA + ROI + 총자산회전율', data = data).fit()
print(model3.summary(), '\n')

model4 = sm.ols('target ~ 매출액증가율 + 부채비율 + 매출총이익률 + ROA + ROI + 총자산회전율', data = data).fit()
print(model4.summary(), '\n')


```