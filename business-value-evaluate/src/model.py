# model.py

def calculate_company_value(revenue, revenue_B, market_cap, interest_rate=8.09):
    """
    간단한 EVA 평가 방식 (예시): 가치 = 현재매출 * 성장률 / 수익률
    """
    growth = (revenue - revenue_B) / revenue_B if revenue_B != 0 else 0
    value_estimate = revenue * (1 + growth) / (interest_rate / 100)
    return value_estimate

def evaluate_all_companies(revenues, revenues_B, market_caps, interest_rate=8.09):
    values = []
    for r, rB, m in zip(revenues, revenues_B, market_caps):
        val = calculate_company_value(r, rB, m, interest_rate)
        values.append(val)
    return values
