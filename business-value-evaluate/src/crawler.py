# crawler.py
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time

def get_driver():
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()))

def crawl_financial_data(driver, company_codes):
    market_cap, revenue, revenue_B = [], [], []
    
    for code in company_codes:
        driver.get(f"http://comp.fnguide.com/SVO2/ASP/SVD_Main.asp?pGB=1&gicode=A{code}")
        time.sleep(1)
        
        try:
            # 시가총액
            market_cap_val = driver.find_element(By.XPATH, '//*[@id="svdMainGrid1"]/table/tbody/tr[5]/td[1]').text
            market_cap.append(int(market_cap_val.replace(',', '')))

            # 재무 데이터
            revenue_val = driver.find_element(By.XPATH, '//*[@id="highlight_D_A"]/table/tbody/tr[1]/td[2]').text
            revenue_B_val = driver.find_element(By.XPATH, '//*[@id="highlight_D_A"]/table/tbody/tr[1]/td[1]').text

            revenue.append(int(revenue_val.replace(',', '')))
            revenue_B.append(int(revenue_B_val.replace(',', '')))
        except Exception as e:
            print(f"Error with company {code}: {e}")
            market_cap.append(0)
            revenue.append(0)
            revenue_B.append(0)
    
    return market_cap, revenue, revenue_B
