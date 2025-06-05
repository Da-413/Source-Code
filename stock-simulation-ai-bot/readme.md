# 주식 모의 투자 서비스 및 AI 자동매매 시스템 (코드 중심)

> **설명**: FastAPI 백엔드와 Kiwoom Open API를 연동해 주식 모의 투자 플랫폼을 구축하고, Envelope, Bollinger, Short-Term, DRL-UTrans 전략을 구현한 자동매매 봇을 개발했습니다. 아래 샘플 코드를 참고하여 프로젝트 구조와 주요 기능을 살펴보세요.

---

## 목차

1. [프로젝트 개요](#프로젝트-개요)  
2. [FastAPI 백엔드](#fastapi-백엔드)  
   - 2.1 [환경 설정 및 실행](#21-환경-설정-및-실행)  
   - 2.2 [핵심 엔드포인트 예제](#22-핵심-엔드포인트-예제)  
3. [Kiwoom API 연동](#kiwoom-api-연동)  
   - 3.1 [토큰 관리 (`TokenManager`)](#31-토큰-관리-tokenmanager)  
   - 3.2 [실시간 시세 구독](#32-실시간-시세-구독)  
4. [AI 자동매매 전략](#ai-자동매매-전략)  
   - 4.1 [EnvelopeTradingModel](#41-envelopetradingmodel)  
   - 4.2 [BollingerBandTradingModel](#42-bollingerbandtradingmodel)  
   - 4.3 [ShortTermTradingModel](#43-shorttermtradingmodel)  
   - 4.4 [DRLUTransTradingModel](#44-drlutrustradingmodel)  
5. [자동매매 흐름 예제](#자동매매-흐름-예제)  
6. [실행 방법](#실행-방법)  
7. [도전 과제 및 해결 방안](#도전-과제-및-해결-방안)  

---

## 프로젝트 개요

- **프레임워크**: FastAPI (RESTful API 서버)  
- **인증**: JWT 기반 토큰 관리 (`TokenManager`)  
- **증권사 API**: Kiwoom Open API (async 방식)  
- **자동매매 모듈**: Envelope, Bollinger, Short-Term, DRL-UTrans (각 전략은 `BaseTradingModel` 상속)  

---

## FastAPI 백엔드

### 2.1 환경 설정 및 실행

```bash
git clone https://github.com/Da-413/stock-simulation-ai-bot.git
cd stock-simulation-ai-bot
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2.2 핵심 엔드포인트 예제

```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any

from app.auth.token_manager import TokenManager
from app.auth.auth_client import AuthClient
from app.api.kiwoom_api import KiwoomAPI
from app.bot.bot_manager import BotManager

app = FastAPI(title="주식 자동매매 봇 API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class BotStatusResponse(BaseModel):
    email: str
    strategy: str
    is_running: bool
    start_time: str = None
    last_data_update: str = None
    account_info: Dict[str, Any] = None

token_manager = TokenManager()
auth_client = AuthClient()
kiwoom_api = None
bot_manager = BotManager()

service_status = {
    "is_running": False,
    "start_time": None,
    "last_data_update": None,
    "active_strategy": None,
    "current_user": None
}

@app.post("/initialize")
async def initialize_service(strategy: str):
    await token_manager.initialize()
    await auth_client.initialize()
    global kiwoom_api
    if not kiwoom_api:
        kiwoom_api = KiwoomAPI(token_manager)
        await kiwoom_api.connect()
    await bot_manager.initialize(kiwoom_api)
    service_status.update({
        "is_running": True,
        "start_time": "2025-06-05T09:00:00+09:00",
        "last_data_update": "2025-06-05T09:00:00+09:00",
        "active_strategy": strategy,
        "current_user": "user@example.com"
    })
    return {"status": "initialized", "strategy": strategy}

@app.get("/bot/status", response_model=BotStatusResponse)
async def get_bot_status():
    return BotStatusResponse(**service_status)

@app.post("/bot/stop")
async def stop_bot():
    if not service_status["is_running"]:
        raise HTTPException(status_code=400, detail="Service not running")
    await bot_manager.stop_all()
    service_status["is_running"] = False
    return {"status": "stopped"}
```

---

## Kiwoom API 연동

### 3.1 토큰 관리 (`TokenManager`)

```python
import asyncio

class TokenManager:
    def __init__(self):
        self.token = None

    async def initialize(self):
        self.token = await self._fetch_initial_token()

    async def _fetch_initial_token(self):
        return "initial_kiwoom_token"

    async def refresh_token(self):
        self.token = await self._fetch_initial_token()
        return self.token
```

### 3.2 실시간 시세 구독

```python
import asyncio
from typing import List, Dict

class KiwoomAPI:
    def __init__(self, token_manager):
        self.token_manager = token_manager
        self.stock_cache = {}
        self.subscribers = []

    async def connect(self) -> bool:
        token = await self.token_manager.refresh_token()
        if not token:
            return False
        return True

    async def initialize_stock_list(self, symbols: List[str]):
        self.stock_cache = {symbol: None for symbol in symbols}

    async def subscribe_realtime(self, symbols: List[str], callback):
        self.subscribers.append(callback)

    def update_price(self, symbol: str, price: float):
        self.stock_cache[symbol] = price
        for cb in self.subscribers:
            asyncio.create_task(cb(symbol, price))
```

---

## AI 자동매매 전략

각 전략은 `BaseTradingModel`을 상속하여 `start()`, `stop()`, `refresh_indicators()`, `handle_realtime_price()` 등의 메서드를 구현합니다.

### 4.1 EnvelopeTradingModel

```python
import logging
import asyncio
from app.strategies.base import BaseTradingModel
from datetime import datetime

logger = logging.getLogger(__name__)

class EnvelopeTradingModel(BaseTradingModel):
    def __init__(self, stock_cache=None):
        super().__init__(stock_cache)
        self.max_positions = 7
        self.trade_amount_per_stock = 14_000_000
        self.trading_signals = {}
        self.trade_history = {}

    async def start(self):
        if self.is_running:
            return
        self.is_running = True
        logger.info("Envelope 전략 시작")
        asyncio.create_task(self.monitor_signals())

    async def stop(self):
        self.is_running = False
        logger.info("Envelope 전략 중지")

    async def handle_realtime_price(self, symbol: str, price: float, indicators=None):
        if not self.is_running:
            return
        envelope = indicators.get("envelope") if indicators else self.stock_cache.get_envelope_indicators(symbol, price)
        if not envelope:
            return
        upper, middle, lower = envelope["upper"], envelope["middle"], envelope["lower"]
        now = datetime.now()
        holdings = self.positions.get(symbol, 0) > 0

        if price <= lower and not holdings:
            self.trading_signals[symbol] = {"signal": "buy", "price": price, "timestamp": now}
            logger.info(f"Envelope 매수 신호: {symbol} @ {price:.2f}")

        elif price >= middle and holdings:
            last_action = self.trade_history.get(symbol, {}).get("last_action")
            if last_action != "sell_half":
                self.trading_signals[symbol] = {"signal": "sell_half", "price": price, "timestamp": now}
                logger.info(f"Envelope 절반 매도 신호: {symbol} @ {price:.2f}")

        elif price >= upper and holdings:
            last_action = self.trade_history.get(symbol, {}).get("last_action")
            if last_action != "sell_all":
                self.trading_signals[symbol] = {"signal": "sell_all", "price": price, "timestamp": now}
                logger.info(f"Envelope 전량 매도 신호: {symbol} @ {price:.2f}")
```

### 4.2 BollingerBandTradingModel

```python
import logging
import asyncio
from typing import Dict
from datetime import datetime
from app.strategies.base import BaseTradingModel

logger = logging.getLogger(__name__)

class BollingerBandTradingModel(BaseTradingModel):
    def __init__(self, stock_cache=None):
        super().__init__(stock_cache)
        self.max_positions = 7
        self.trade_amount_per_stock = 14_000_000
        self.bb_period = 26
        self.bb_std_dev = 2.0
        self.buy_percentB_threshold = 0.05
        self.split_purchase_count = 3
        self.split_purchase_percentages = [0.4, 0.3, 0.3]
        self.trading_signals = {}
        self.trade_history = {}
        self.last_processed = {}

    async def start(self):
        if self.is_running:
            return
        self.is_running = True
        logger.info("Bollinger 전략 시작")
        asyncio.create_task(self.monitor_signals())

    async def stop(self):
        self.is_running = False
        logger.info("Bollinger 전략 중지")

    def _calculate_signal_confidence(self, symbol, price, bb):
        upper, middle, lower = bb["upper"], bb["middle"], bb["lower"]
        percentB = bb["percentB"]
        confidence = 0.5
        if percentB <= self.buy_percentB_threshold:
            pb_conf = 1.0 - (percentB / self.buy_percentB_threshold)
            band_pos = max(0.0, min(1.0, (lower - price) / lower))
            bw_factor = min(1.0, ((upper - lower) / middle) / 0.05)
            confidence = (0.2*confidence + 0.5*pb_conf + 0.2*band_pos + 0.1*bw_factor)
        elif percentB >= 0.9:
            pb_conf = (percentB - 0.9) / 0.1
            band_pos = max(0.0, min(1.0, (price - upper) / upper))
            bw_factor = min(1.0, ((upper - lower) / middle) / 0.05)
            confidence = (0.2*confidence + 0.5*pb_conf + 0.2*band_pos + 0.1*bw_factor)
        return max(0.0, min(1.0, confidence))

    async def handle_realtime_price(self, symbol: str, price: float, indicators=None):
        if not self.is_running:
            return
        now = datetime.now()
        bb = indicators.get("bollinger") if indicators else self.stock_cache.get_bollinger_indicators(symbol, price)
        if not bb:
            return
        upper, middle, lower = bb["upper"], bb["middle"], bb["lower"]
        holdings = self.positions.get(symbol, 0) > 0

        if price <= lower and not holdings:
            conf = self._calculate_signal_confidence(symbol, price, bb)
            if conf >= self.buy_percentB_threshold:
                self.trading_signals[symbol] = {"signal": "buy", "price": price, "conf": conf, "timestamp": now}
                logger.info(f"Bollinger 매수 신호: {symbol} @ {price:.2f}, 신뢰도 {conf:.2f}")

        elif price >= upper and holdings:
            self.trading_signals[symbol] = {"signal": "sell_all", "price": price, "timestamp": now}
            logger.info(f"Bollinger 전량 매도 신호: {symbol} @ {price:.2f}")
        elif price >= middle and holdings:
            last_action = self.trade_history.get(symbol, {}).get("last_action")
            if last_action != "sell_half":
                self.trading_signals[symbol] = {"signal": "sell_half", "price": price, "timestamp": now}
                logger.info(f"Bollinger 절반 매도 신호: {symbol} @ {price:.2f}")
```

### 4.3 ShortTermTradingModel

```python
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
import numpy as np
from app.auth.token_manager import TokenManager
from app.api.kiwoom_api import KiwoomAPI
from app.strategies.base import BaseTradingModel

logger = logging.getLogger(__name__)

class ShortTermTradingModel(BaseTradingModel):
    def __init__(self, stock_cache=None, backend_client=None):
        super().__init__(stock_cache)
        self.backend_client = backend_client
        self.max_positions = 7
        self.trade_amount_per_stock = 14_000_000
        self.buy_division_count = 2
        self.sell_division_count = 2
        self.division_interval = 600  # 초
        self.last_processed = {}
        self.minute_candle_cache = {}
        self.last_candle_update = {}
        self.top_amount_update_interval = 3600  # 초
        self.top_trading_amount = []
        self.potential_targets = []
        self.verified_targets = {}

    async def start(self):
        if self.is_running:
            return
        self.is_running = True
        logger.info("Short-Term 전략 시작")
        await self._sync_account_info()
        await self.initial_data_load()
        asyncio.create_task(self.monitor_signals())
        asyncio.create_task(self.monitor_top_volume_stocks())

    async def stop(self):
        self.is_running = False
        logger.info("Short-Term 전략 중지")

    async def _sync_account_info(self):
        if self.backend_client:
            acct = await self.backend_client.request_account_info()
            self.update_account_info(acct)

    async def initial_data_load(self):
        if hasattr(self.kiwoom_api, "get_all_top_trading_amount"):
            top_stocks = await self.kiwoom_api.get_all_top_trading_amount(limit=100)
            self.top_trading_amount = top_stocks
            self.last_top_trading_amount = datetime.now()
            await self.cross_verify_target_stocks()

    async def cross_verify_target_stocks(self) -> List[str]:
        filtered = self.stock_cache.get_filtered_stocks()
        verified = []
        for idx, stock in enumerate(self.top_trading_amount, 1):
            sym = stock.get("code", "").rstrip("_AL")
            if sym in filtered:
                verified.append(sym)
                self.verified_targets[sym] = {
                    "rank": idx,
                    "trading_amount": stock.get("trading_amount")
                }
        self.potential_targets = verified
        logger.info(f"교차 검증된 대상: {len(verified)}개 종목")
        return verified

    async def monitor_top_volume_stocks(self):
        while self.is_running:
            await asyncio.sleep(300)
            now = datetime.now()
            if (now - self.last_top_trading_amount).seconds >= self.top_amount_update_interval:
                await self.initial_data_load()
            targets = self.potential_targets[:30]
            for sym in targets:
                price = self._get_current_price(sym)
                if price:
                    await self.handle_realtime_price(sym, price)

    def _get_current_price(self, symbol: str) -> float:
        price = self.stock_cache.get_price(symbol) if self.stock_cache else 0
        if not price and symbol in self.verified_targets:
            price = self.verified_targets[symbol].get("price", 0)
        return price

    async def handle_realtime_price(self, symbol: str, price: float, indicators: Dict[str, Any]=None):
        if not self.is_running:
            return
        now = datetime.now()
        if not self._should_process(symbol, price):
            return
        candle_data = await self._get_or_update_candle_data(symbol)
        if not candle_data or len(candle_data) < 10:
            return
        latest = candle_data[-1]
        prev = candle_data[-2]
        if latest["volume"] > prev["volume"] * 1.5 and (latest["close"] - prev["close"]) / prev["close"] > 0.03:
            self.trading_signals[symbol] = {"signal": "buy", "price": price, "timestamp": now}
            logger.info(f"Short-Term 매수 신호: {symbol} @ {price:.2f}")

    def _should_process(self, symbol: str, price: float) -> bool:
        now = datetime.now()
        last_price = self.last_processed.get(symbol, {}).get("price", 0)
        last_time = self.last_processed.get(symbol, {}).get("time", datetime.min)
        if (now - last_time).seconds < 5:
            return False
        if last_price and abs(price - last_price) / last_price * 100 < 0.1:
            return False
        self.last_processed[symbol] = {"price": price, "time": now}
        return True

    async def _get_or_update_candle_data(self, symbol: str):
        last_update = self.last_candle_update.get(symbol, datetime.min)
        if (datetime.now() - last_update).seconds < 300 and symbol in self.minute_candle_cache:
            return self.minute_candle_cache[symbol]
        data = await self.kiwoom_api.get_minute_chart_data(symbol, 5)
        if data:
            self.minute_candle_cache[symbol] = data[:50]
            self.last_candle_update[symbol] = datetime.now()
        return self.minute_candle_cache.get(symbol, [])
```

### 4.4 DRLUTransTradingModel

```python
import os
import asyncio
import torch
import pickle
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from app.strategies.base import BaseTradingModel

class DRLUTransTradingModel(BaseTradingModel):
    def __init__(self, stock_cache=None):
        super().__init__(stock_cache)
        self.max_positions = 7
        self.trade_amount_per_stock = 14_000_000
        self.model_path = os.environ.get('DRL_UTRANS_MODEL_PATH', './models/drl_utrans')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scalers = {}
        self.seq_len = 20
        self.is_model_loaded = False
        self.trading_signals = {}
        self.predictions = {}
        self.trade_history = {}
        self.last_processed = {}
        self.prediction_interval = 4  # 시간 단위
        self.confidence_threshold = 0.98

    async def load_model(self) -> bool:
        model_file = os.path.join(self.model_path, "drl_utrans_model.pth")
        scalers_file = os.path.join(self.model_path, "drl_utrans_scalers.pkl")
        def _load():
            from app.models.drl_utrans_network import DRLUTransPPOnet
            input_dim, seq_len, action_dim = 26, 20, 3
            net = DRLUTransPPOnet(input_dim=input_dim, seq_len=seq_len, action_dim=action_dim).to(self.device)
            state = torch.load(model_file, map_location=self.device)
            if 'model_state_dict' in state:
                net.load_state_dict(state['model_state_dict'], strict=False)
            else:
                net.load_state_dict(state, strict=False)
            net.eval()
            if os.path.exists(scalers_file):
                with open(scalers_file, 'rb') as f:
                    scalers = pickle.load(f)
            else:
                scalers = {'default': StandardScaler()}
            return net, scalers

        if not os.path.exists(model_file):
            return False
        loop = asyncio.get_event_loop()
        self.model, self.scalers = await loop.run_in_executor(None, _load)
        self.is_model_loaded = True
        return True

    async def start(self):
        if self.is_running:
            return
        if not self.is_model_loaded:
            ok = await self.load_model()
            if not ok:
                return
        self.is_running = True
        asyncio.create_task(self.batch_prediction_scheduler())
        asyncio.create_task(self.monitor_signals())

    async def stop(self):
        self.is_running = False

    async def handle_realtime_price(self, symbol: str, price: float, indicators=None):
        if not self.is_running:
            return
        now = datetime.now()
        if not self._should_process(symbol, price):
            return
        # 실시간 예측 로직은 생략, 주로 배치 예측 처리

    async def batch_prediction_scheduler(self):
        while self.is_running:
            await asyncio.sleep(self.prediction_interval * 3600)
            for sym in self.stock_cache.filtered_stockcode_list:
                await self.process_single_prediction(sym)

    async def process_single_prediction(self, symbol: str):
        data = self.stock_cache.get_chart_data(symbol)
        if not data or len(data) < self.seq_len:
            return
        df = np.array(data[-self.seq_len:])
        scaler = self.scalers.get(symbol, self.scalers.get('default'))
        features = scaler.transform(df)
        x = torch.tensor(features.reshape(1, self.seq_len, -1), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            out = self.model(x).cpu().numpy()[0]
        action = np.argmax(out)
        if out[action] >= self.confidence_threshold:
            signal = "buy" if action == 1 else "sell" if action == 2 else "hold"
            self.trading_signals[symbol] = {"signal": signal, "timestamp": datetime.now()}
```

---

## 자동매매 흐름 예제

```python
# FastAPI 엔드포인트에서 특정 전략의 자동매매 시작 흐름 예시
from fastapi import APIRouter

router = APIRouter(prefix="/trade")

@router.post("/start/{strategy}")
async def start_strategy(strategy: str):
    """
    전략별 자동매매 시작
    - strategy: "envelope" | "bollinger" | "short_term" | "drl_utrans"
    """
    global kiwoom_api, backend_client, bot_manager
    if strategy == "envelope":
        model = EnvelopeTradingModel(stock_cache=kiwoom_api.stock_cache)
    elif strategy == "bollinger":
        model = BollingerBandTradingModel(stock_cache=kiwoom_api.stock_cache)
    elif strategy == "short_term":
        model = ShortTermTradingModel(stock_cache=kiwoom_api.stock_cache, backend_client=backend_client)
    elif strategy == "drl_utrans":
        model = DRLUTransTradingModel(stock_cache=kiwoom_api.stock_cache)
    else:
        return {"error": "Invalid strategy"}

    await model.start()
    bot_manager.register_bot(strategy, model)
    return {"status": "started", "strategy": strategy}

@router.post("/stop/{strategy}")
async def stop_strategy(strategy: str):
    """
    전략별 자동매매 중지
    """
    bot = bot_manager.get_bot(strategy)
    if not bot:
        return {"error": "Bot not found"}
    await bot.stop()
    bot_manager.unregister_bot(strategy)
    return {"status": "stopped", "strategy": strategy}
```

---

## 실 실행 방법

1. 리포지토리 클론 및 가상환경 설정
```bash
git clone https://github.com/Da-413/stock-simulation-ai-bot.git
cd stock-simulation-ai-bot
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. FastAPI 서버 실행
```bash
uvicorn main:app --reload
```

3. Jupyter Notebook 실행 (모델 학습/시험)
```bash
jupyter notebook
```

4. 프론트엔드 빌드 및 실행
```bash
cd frontend
npm install
npm run dev
```

---

## 도전 과제 및 해결 방안

1. **비동기 처리 병목**
   - 실시간 가격 처리 시 asyncio 태스크 관리 필요
   - **해결**: 캐시 구조와 큐를 도입하여 이벤트 기반으로 효율적 처리

2. **전략 간 충돌 및 우선순위 조정**
   - 동시에 여러 전략이 동일 종목에 신호 생성 시 혼선 발생
   - **해결**: 각 전략별 신호 큐를 분리하고, 포지션 이력에 따라 우선순위 판단

3. **실시간 시세 연동 안정성**
   - Kiwoom API 연결 불안정 및 토큰 만료 이슈
   - **해결**: TokenManager를 통해 주기적 토큰 갱신, 예외 처리 강화

4. **강화학습 모델 로드 시간 및 메모리**
   - DRLUTrans 모델 파일 크기 및 로딩 지연
   - **해결**: torch.serialization.safe_globals 사용하여 빠른 로딩, GPU 메모리 관리 최적화
