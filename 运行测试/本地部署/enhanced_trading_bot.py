#!/usr/bin/env python3
"""
增强版交易机器人 - 集成DeepSeek建议的数据结构改进
基于btc_trading_bot.py，使用增强版数据提取器和SMC分析
"""

import os
import time
import schedule
import threading
import functools
import random
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from openai import OpenAI
import ccxt
from ccxt.base.errors import NetworkError, RequestTimeout, ExchangeError
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import json
import re
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler
from typing import Dict, Any, Optional, TypedDict, List, Set
from typing import Tuple
import ssl  # FIXED: SSL 1 - 添加 SSL 支持
import urllib3  # FIXED: SSL 2 - 禁用警告
from enum import Enum  # NEW: For signal priority system
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests  # FIXED: Data Fetch 2 - 显式导入 requests

# 导入增强版数据提取器和提示词生成器
import sys
sys.path.append('/Users/zhidafu/ds交易/ds/运行测试/测试部署')
from enhanced_data_extractor import EnhancedDataExtractor
from enhanced_smc_signal_calculator import EnhancedSMCSignalCalculator
from enhanced_smc_prompt import get_enhanced_smc_prompt
from enhanced_mock_bot import EnhancedMockBot

# SMC/ICT结构识别库导入
try:
    import smartmoneyconcepts.smc as smc
    SMC_AVAILABLE = True
except ImportError:
    SMC_AVAILABLE = False
    logging.warning("smartmoneyconcepts库未安装，SMC结构识别功能将使用备用实现")

# FIXED: SSL 3 - 禁用 urllib3 SSL 警告（生产中可选移除）
# 特别针对macOS LibreSSL兼容性问题的修复
import os
import warnings

# 通过环境变量彻底禁用urllib3警告
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore', category=urllib3.exceptions.NotOpenSSLWarning)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
urllib3.disable_warnings(urllib3.exceptions.NotOpenSSLWarning)

# Load environment variables from 1.env file (contains all API keys)
load_dotenv('/Users/zhidafu/ds交易/ds/运行测试/本地部署/1.env')

# FIXED: SSL 4 - 自定义 SSL 上下文，处理 EOF 错误
def create_ssl_context():
    ctx = ssl.create_default_context()
    ctx.check_hostname = True  # 保持安全性
    ctx.verify_mode = ssl.CERT_REQUIRED
    # 设置更宽松的协议版本以提高兼容性
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    return ctx

def setup_logging(log_file: str = 'enhanced_trading_bot.log', level: str = 'INFO', enable_json: bool = False):
    """Elegant logging setup supporting categories and structured output"""
    # Clear existing handlers to avoid duplicates
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Plain format (human-readable)
    plain_formatter = logging.Formatter(
        '%(asctime)s [%(threadName)-10s] %(name)-12s - %(levelname)-8s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # JSON format (optional, machine-readable)
    if enable_json:
        import json
        class JsonFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    'timestamp': self.formatTime(record),
                    'level': record.levelname,
                    'logger': record.logger.name,
                    'thread': record.threadName,
                    'message': record.getMessage(),
                    'module': record.module,
                    'function': record.funcName,
                    'line': record.lineno
                }
                if record.exc_info:
                    log_entry['exception'] = self.formatException(record.exc_info)
                return json.dumps(log_entry, ensure_ascii=False)
        json_formatter = JsonFormatter()

    # Console handler (colored output)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(plain_formatter)
    console_handler.setLevel(level)

    # File handler (DEBUG level, rotating)
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
    )
    file_handler.setFormatter(json_formatter if enable_json else plain_formatter)
    file_handler.setLevel('DEBUG')

    # Root logger setup
    root_logger.setLevel('DEBUG')
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Create category loggers
    loggers = {
        'trading': logging.getLogger('trading'),
        'api': logging.getLogger('api'), 
        'risk': logging.getLogger('risk'),
        'monitor': logging.getLogger('monitor'),
        'system': logging.getLogger('system')
    }

    # Suppress noisy third-party logs
    for noisy_logger in ['pandas', 'numpy', 'urllib3', 'requests', 'ccxt', 'httpx', 'httpcore', 'openai']:
        logging.getLogger(noisy_logger).setLevel('WARNING')
    
    # 特别抑制OpenAI和HTTP相关的详细日志
    logging.getLogger('openai._base_client').setLevel('WARNING')
    logging.getLogger('httpcore.connection').setLevel('WARNING')
    logging.getLogger('httpcore.http11').setLevel('WARNING')
    logging.getLogger('httpcore.proxy').setLevel('WARNING')
    logging.getLogger('schedule').setLevel('INFO')  # 只显示重要的调度信息

    return loggers

# Initialize logging system
loggers = setup_logging('enhanced_trading_bot.log', 'INFO')  # 降低日志级别为INFO，减少调试信息
logger = logging.getLogger(__name__)  # Maintain backward compatibility

@dataclass
class EnhancedConfig:
    """增强版配置类，包含DeepSeek建议的数据结构参数"""
    symbol: str = 'BTC/USDC:USDC'  # BTC专用配置
    amount: float = 0.01
    # Data source configuration
    data_source: str = 'websocket'  # 'websocket' or 'hyperliquid'
    use_websocket_indicators: bool = True  # Use WebSocket for real-time indicators
    leverage: int = 40
    timeframes: List[str] = None
    primary_timeframe: str = '15m'
    structure_confirm_timeframe: str = '1h'
    data_points: int = 200
    amplitude_lookback: int = 7
    activation_threshold: float = 0.00005  # 0.005% - AI自主权增强版：超低激活阈值
    min_balance_ratio: float = 0.95
    max_position_time: int = 86400
    risk_per_trade: float = 0.018  # 1.8% - 金融日内优化：提高单笔风险
    slippage_buffer: float = 0.001  # 增加滑点缓冲容忍度 (0.1%)
    volatility_threshold: float = 70
    order_timeout: int = 10
    heartbeat_interval: int = 60
    price_monitor_interval: int = 180  # 3分钟监控间隔
    signals_file: str = 'enhanced_signal_history.json'  # 增强版信号历史文件
    heartbeat_file: str = 'heartbeat.log'
    log_file: str = 'enhanced_trading_bot.log'  # 增强版日志文件
    max_log_size: int = 10 * 1024 * 1024
    log_backup_count: int = 5
    deepseek_timeout: int = 30
    liquidity_priority: List[str] = None
    use_ema100: bool = True  # New: Toggle EMA100 usage
    ema100_priority: float = 0.7  # New: Multiplier for activation threshold
    min_fill_ratio: float = 0.95  # New: Minimum order fill ratio
    cache_ttl: int = 300  # New: Cache TTL in seconds
    rsi_neutral: float = 50  # New: Neutral RSI value
    rsi_min: float = 0  # New: RSI min clip
    rsi_max: float = 100  # New: RSI max clip
    simulation_mode: bool = False  # New: Simulation mode toggle
    backtest_file: Optional[str] = None  # Added for main()
    max_margin_usage: float = 0.60  # Maximum margin usage ratio
    fee_rate: float = 0.0002  # Taker fee
    maintenance_margin_rate: float = 0.005  # Hyperliquid default (approximate)
    symbol_info: Dict[str, Any] = None
    primary_timeframe_weight: float = 2.0  # Weight for 15m structure
    rr_min_threshold: float = 2.0  # 2.0:1 - 开单标准上调
    rr_aggressive_threshold: float = 3.0  # 3.0:1 - 开单标准上调
    risk_aggressive: float = 0.02  # Aggressive risk if R:R high
    temperature: float = 0.4  # 1小时级别优化：提高AI温度
    
    # 增强版数据结构参数 - DeepSeek建议
    enable_enhanced_data: bool = True  # 启用增强版数据结构
    enhanced_data_weight: float = 0.7  # 增强版数据权重
    market_depth_weight: float = 0.15  # 市场深度数据权重
    time_sales_weight: float = 0.1  # 时间与销售数据权重
    market_sentiment_weight: float = 0.05  # 市场情绪数据权重
    
    # 增强版SMC信号参数
    enhanced_smc_min_confidence: float = 0.6  # 增强版SMC最小置信度
    enhanced_smc_signal_weights: Dict[str, float] = None  # 增强版SMC信号权重
    
    def __post_init__(self):
        if self.timeframes is None:
            self.timeframes = ['1d', '4h', '1h', '15m', '3m', '1m']  # 增加3分钟级别用于结构观察，1分钟用于订单流分析
        if self.liquidity_priority is None:
            self.liquidity_priority = [
                # Daily level (highest priority)
                'monday_open', 'daily_open', 'prev_week_high', 'prev_week_low', 'daily_vwap', 'daily_fvg_bull_mid', 'daily_fvg_bear_mid',
                'prev_day_high', 'prev_day_low', 'daily_ema_100', 'prev_week_close', 'prev_month_high', 'prev_month_low',
                # 4h level
                '4h_fvg_bull_mid', '4h_fvg_bear_mid', '4h_ob', 'prev_4h_high', 'prev_4h_low',
                # 1h level  
                '1h_fvg_bull_mid', '1h_fvg_bear_mid', '1h_ob', 'prev_1h_high', 'prev_1h_low',
                # 15m level
                '15m_fvg_bull_mid', '15m_fvg_bear_mid', '15m_ob', 'prev_15m_high', 'prev_15m_low'
            ]
        if self.enhanced_smc_signal_weights is None:
            self.enhanced_smc_signal_weights = {
                'bos_choch': 0.3,  # BOS/CHOCH信号权重
                'order_blocks': 0.25,  # 订单块信号权重
                'fvg': 0.2,  # FVG信号权重
                'liquidity': 0.15,  # 流动性信号权重
                'market_microstructure': 0.1  # 市场微观结构信号权重
            }
        # FIXED: Initialize symbol_info for price data access
        if self.symbol_info is None:
            self.symbol_info = {
                'last': 115000.0,  # Default BTC price for fallback calculations
                'symbol': self.symbol,
                'price_precision': 2,
                'amount_precision': 4
            }
        self.validate()

    def validate(self):
        if not (1 <= self.leverage <= 125):
            raise ValueError(f"Leverage must be between 1-125, got: {self.leverage}")
        if not (0.001 <= self.risk_per_trade <= 0.05):
            raise ValueError(f"Risk per trade must be 0.1%-5%, got: {self.risk_per_trade*100:.1f}%")
        if self.amount < 0.01:
            raise ValueError(f"Amount must be >=0.01 BTC, got: {self.amount}")
        if not (0.00001 <= self.activation_threshold <= 0.05):
            raise ValueError(f"Activation threshold must be 0.001%-5%, got: {self.activation_threshold*100:.3f}%")
        if self.primary_timeframe not in self.timeframes:
            raise ValueError(f"Primary timeframe must be in timeframes, got: {self.primary_timeframe}")

class PositionStore:
    """线程安全的持仓存储"""
    def __init__(self):
        self.position = None
        self.lock = threading.RLock()
    
    def get(self):
        with self.lock:
            return self.position
    
    def set(self, position):
        with self.lock:
            self.position = position
    
    def clear(self):
        with self.lock:
            self.position = None

def create_session_with_retry():
    """创建带重试机制的session"""
    session = requests.Session()
    
    # 设置重试策略
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

class EnhancedTradingBot:
    """增强版交易机器人 - 集成DeepSeek建议的数据结构改进"""
    
    def __init__(self, config: EnhancedConfig, exchange=None):
        self.config = config
        self.exchange = exchange
        # Initialize category loggers
        self.logger_trading = logging.getLogger('trading')
        self.logger_api = logging.getLogger('api')
        self.logger_risk = logging.getLogger('risk')
        self.logger_monitor = logging.getLogger('monitor')
        self.logger_system = logging.getLogger('system')

        self.signal_history: List[Dict[str, Any]] = []
        self.key_levels_cache: Optional[Dict[str, float]] = None
        self.cache_timestamp: float = 0
        self.initial_balance: float = 0
        self.last_activation_time: float = 0
        self.level_activation_times: Dict[str, float] = {}  # Track last activation time per key level
        self.zone_interactions: Dict[str, int] = {}  # Count interactions per zone
        self.last_scheduled_signal: Optional[Dict[str, Any]] = None  # Store last scheduled signal copy
        self.lock = threading.RLock()
        self.trade_lock = threading.RLock()
        self.position_store = PositionStore()
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        # FIXED: Medium 4 - Cache for indicators
        self.indicators_cache: Dict[str, pd.DataFrame] = {}
        # FIXED: Market data storage for ATR calculations
        self.market_data: Dict[str, pd.DataFrame] = {}
        
        # 增强版数据提取器初始化
        self.enhanced_data_extractor = EnhancedDataExtractor()
        self.enhanced_mock_bot = EnhancedMockBot()
        
        # 初始化增强版SMC信号计算器
        self.enhanced_smc_calculator = EnhancedSMCSignalCalculator(self.config)
        
        # New: Risk control tracking
        self.daily_start_balance: float = 0
        self.peak_balance: float = 0
        self.current_balance: float = 0
        self.last_reset_date: str = ""
        self.last_reset_4h: datetime = datetime.now(timezone.utc)  # New: For 4h reset
        # API health status tracking
        self.api_health_status = {
            'deepseek': {'status': 'unknown', 'last_check': 0, 'consecutive_failures': 0},
            'hyperliquid': {'status': 'unknown', 'last_check': 0, 'consecutive_failures': 0},
            'websocket': {'status': 'unknown', 'last_check': 0, 'consecutive_failures': 0}
        }
        
        # FIXED: Data Fetch 3 - 创建带重试的 session
        self.session = create_session_with_retry()
        
        # Initialize data providers (currently disabled due to missing modules)
        self.coindesk_provider = None
        self.hyperliquid_websocket_backup = None
        self.hyperliquid_backup = None
        self.hyperliquid_market_data = None
        
        # Note: Custom data providers are disabled as modules are not available
        # self.coindesk_provider = CoinDeskWebSocketIndicatorProvider(CoinDeskIndicatorConfig())
        # self.hyperliquid_websocket_backup = HyperliquidWebSocketProvider(HyperliquidIndicatorConfig())
        # self.hyperliquid_backup = HyperliquidBackupProvider()
        # self.hyperliquid_market_data = HyperliquidMarketData()
    
    def safe_fetch_ohlcv(self, exchange, symbol, timeframe, limit):
        """安全获取OHLCV数据，带重试机制"""
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                if ohlcv and len(ohlcv) > 0:
                    return ohlcv
                else:
                    self.logger_api.warning(f"Empty OHLCV data for {symbol} {timeframe} (attempt {attempt+1})")
            except NetworkError as e:
                self.logger_api.warning(f"Network error fetching OHLCV for {symbol} {timeframe} (attempt {attempt+1}): {e}")
            except RequestTimeout as e:
                self.logger_api.warning(f"Timeout fetching OHLCV for {symbol} {timeframe} (attempt {attempt+1}): {e}")
            except ExchangeError as e:
                self.logger_api.error(f"Exchange error fetching OHLCV for {symbol} {timeframe} (attempt {attempt+1}): {e}")
                break  # Don't retry exchange errors
            except Exception as e:
                self.logger_api.error(f"Unexpected error fetching OHLCV for {symbol} {timeframe} (attempt {attempt+1}): {e}")
            
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
        
        return None
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        try:
            # 计算移动平均线
            df['ema_20'] = df['close'].ewm(span=20).mean()
            df['ema_50'] = df['close'].ewm(span=50).mean()
            df['ema_100'] = df['close'].ewm(span=100).mean()
            df['sma_200'] = df['close'].rolling(window=200).mean()
            
            # 计算RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # 计算MACD
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # 计算布林带
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            
            # 计算ATR
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['atr'] = true_range.rolling(window=14).mean()
            
            # 计算成交量指标
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            return df
        except Exception as e:
            self.logger_system.error(f"Error calculating technical indicators: {e}")
            return df
    
    def _fetch_and_update_data(self, activated_level: Optional[str] = None):
        """获取并更新数据 - 使用增强版数据提取器"""
        # Fetch multi-TF data using enhanced safe_fetch_ohlcv
        multi_tf_data = {}
        failed_timeframes = []
        successful_timeframes = []
        
        self.logger_system.info(f"开始获取多时间框架数据: {self.config.timeframes}")
        
        for tf in self.config.timeframes:
            try:
                ohlcv = self.safe_fetch_ohlcv(self.exchange, self.config.symbol, tf, self.config.data_points)
                if ohlcv and len(ohlcv) > 0:
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                    df = df.set_index('timestamp')
                    df = self.calculate_technical_indicators(df)
                    multi_tf_data[tf] = df
                    successful_timeframes.append(tf)
                    self.logger_system.debug(f"✅ {tf} 数据获取成功: {len(df)} 条记录")
                else:
                    failed_timeframes.append(tf)
                    self.logger_system.warning(f"❌ {tf} 数据获取失败: 无数据返回")
            except Exception as e:
                failed_timeframes.append(tf)
                self.logger_system.error(f"❌ {tf} 数据获取异常: {e}")

        # 数据获取结果统计
        success_rate = len(successful_timeframes) / len(self.config.timeframes) * 100
        self.logger_system.info(f"数据获取完成: 成功 {len(successful_timeframes)}/{len(self.config.timeframes)} ({success_rate:.1f}%)")
        
        if successful_timeframes:
            self.logger_system.info(f"成功获取: {', '.join(successful_timeframes)}")
        if failed_timeframes:
            self.logger_system.warning(f"获取失败: {', '.join(failed_timeframes)}")

        if not multi_tf_data:
            self.logger_system.error("所有时间框架数据获取失败，无法继续分析")
            return None

        # 获取交易用真实价格（严格禁止估算价格）
        try:
            current_price = self._get_real_market_price(self.exchange, self.config.symbol)
            self.logger_system.info(f"✅ 获取真实市场价格用于交易: ${current_price:.2f}")
            
            # 验证价格合理性
            if current_price <= 0 or current_price > 200000:  # BTC合理价格范围检查 (适应2025年价格水平)
                raise ValueError(f"价格异常: ${current_price:.2f}，超出合理范围")
                
        except Exception as e:
            self.logger_system.error(f"❌ 无法获取真实市场价格: {e}")
            self.logger_system.error("🚨 交易系统停止 - 禁止使用估算价格进行交易")
            return None

        # 获取显示用价格（仅用于日志，不用于交易）
        display_price = None
        try:
            display_price = self._get_display_price_fallback(self.exchange, self.config.symbol)
            if display_price:
                self.logger_system.debug(f"显示用价格: ${display_price:.2f}")
        except Exception as e:
            self.logger_system.debug(f"获取显示用价格失败: {e}")

        # 使用增强版数据提取器提取增强版数据
        if self.config.enable_enhanced_data:
            try:
                self.logger_system.info("🔍 使用增强版数据提取器分析市场数据...")
                
                # 准备OHLC数据
                ohlc_data = []
                primary_tf_df = multi_tf_data.get(self.config.primary_timeframe)
                if primary_tf_df is not None and not primary_tf_df.empty:
                    for index, row in primary_tf_df.iterrows():
                        ohlc_data.append({
                            "timestamp": index.isoformat(),
                            "open": float(row['open']),
                            "high": float(row['high']),
                            "low": float(row['low']),
                            "close": float(row['close']),
                            "volume": float(row['volume']),
                            "timeframe": self.config.primary_timeframe
                        })
                
                # 生成示例市场深度数据（实际应用中应从交易所API获取）
                market_depth = []
                for i in range(10):
                    mid_price = current_price + i * 10
                    market_depth.append({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "bid_price": mid_price - 5,
                        "ask_price": mid_price + 5,
                        "bid_volume": random.uniform(100, 500),
                        "ask_volume": random.uniform(100, 500)
                    })
                
                # 生成示例时间与销售数据（实际应用中应从交易所API获取）
                time_sales = []
                for i in range(100):
                    time_sales.append({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "price": current_price + random.uniform(-100, 100),
                        "volume": random.uniform(0.1, 20),
                        "side": random.choice(["buy", "sell"]),
                        "aggressive": random.choice([True, False])
                    })
                
                # 生成示例市场情绪数据（实际应用中应从外部API获取）
                market_sentiment = {
                    "fear_greed_index": random.uniform(0, 100),
                    "funding_rate": random.uniform(-0.01, 0.01),
                    "open_interest_change": random.uniform(-5, 5),
                    "long_short_ratio": random.uniform(0.8, 1.5)
                }
                
                # 使用增强版数据提取器提取数据
                enhanced_raw_data = self.enhanced_data_extractor.extract_enhanced_raw_data(
                    ohlc_data=ohlc_data,
                    volume_data=[],
                    market_depth=market_depth,
                    time_sales=time_sales,
                    market_sentiment=market_sentiment
                )
                
                # 构建增强版价格数据
                enhanced_price_data = {
                    'price': current_price,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'multi_tf_data': multi_tf_data,
                    'enhanced_data': enhanced_raw_data,  # 添加增强版数据
                    'amplitude': {
                        'expected_rr_range': primary_tf_df['atr'].iloc[-1] * 2 if primary_tf_df is not None and not primary_tf_df.empty else current_price * 0.04,
                        'daily_range': primary_tf_df['high'].iloc[-1] - primary_tf_df['low'].iloc[-1] if primary_tf_df is not None and not primary_tf_df.empty else current_price * 0.02
                    },
                    'technical_data': {
                        'atr': primary_tf_df['atr'].iloc[-1] if primary_tf_df is not None and not primary_tf_df.empty else current_price * 0.02,
                        'rsi': primary_tf_df['rsi'].iloc[-1] if primary_tf_df is not None and not primary_tf_df.empty else 50,
                        'ema_20': primary_tf_df['ema_20'].iloc[-1] if primary_tf_df is not None and not primary_tf_df.empty else current_price,
                        'ema_50': primary_tf_df['ema_50'].iloc[-1] if primary_tf_df is not None and not primary_tf_df.empty else current_price
                    },
                    'key_levels': {},  # 将在后续分析中填充
                    'structures_summary': {},  # 将在后续分析中填充
                    'activated_level': activated_level,
                    'display_price': display_price
                }
                
                self.logger_system.info(f"✅ 增强版数据提取完成，包含 {len(enhanced_raw_data.get('enhanced_candlesticks', []))} 根增强K线")
                return enhanced_price_data
                
            except Exception as e:
                self.logger_system.error(f"❌ 增强版数据提取失败: {e}")
                self.logger_system.info("🔄 回退到标准数据处理流程")
        
        # 标准数据处理流程（回退选项）
        standard_price_data = {
            'price': current_price,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'multi_tf_data': multi_tf_data,
            'amplitude': {
                'expected_rr_range': primary_tf_df['atr'].iloc[-1] * 2 if primary_tf_df is not None and not primary_tf_df.empty else current_price * 0.04,
                'daily_range': primary_tf_df['high'].iloc[-1] - primary_tf_df['low'].iloc[-1] if primary_tf_df is not None and not primary_tf_df.empty else current_price * 0.02
            },
            'technical_data': {
                'atr': primary_tf_df['atr'].iloc[-1] if primary_tf_df is not None and not primary_tf_df.empty else current_price * 0.02,
                'rsi': primary_tf_df['rsi'].iloc[-1] if primary_tf_df is not None and not primary_tf_df.empty else 50,
                'ema_20': primary_tf_df['ema_20'].iloc[-1] if primary_tf_df is not None and not primary_tf_df.empty else current_price,
                'ema_50': primary_tf_df['ema_50'].iloc[-1] if primary_tf_df is not None and not primary_tf_df.empty else current_price
            },
            'key_levels': {},  # 将在后续分析中填充
            'structures_summary': {},  # 将在后续分析中填充
            'activated_level': activated_level,
            'display_price': display_price
        }
        
        return standard_price_data
    
    def _get_real_market_price(self, exchange, symbol):
        """获取真实市场价格 - 仅用于交易决策（禁止估算价格）"""
        try:
            # 优先级1: 实时ticker价格（最准确）
            ticker = exchange.fetch_ticker(symbol)
            if ticker and 'last' in ticker and ticker['last']:
                return ticker['last']
        except Exception as e:
            self.logger_api.warning(f"实时ticker获取失败: {e}")
        
        try:
            # 优先级2: 最新OHLCV收盘价（真实历史数据）
            timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
            for tf in timeframes:
                try:
                    ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=1)
                    if ohlcv and len(ohlcv) > 0:
                        latest_close = ohlcv[-1][4]
                        if latest_close and latest_close > 0:
                            return latest_close
                except:
                    continue
        except Exception as e:
            self.logger_api.warning(f"历史OHLCV获取失败: {e}")
        
        # 严格禁止: 不返回任何估算或参考价格用于交易
        raise Exception("无法获取真实市场价格，交易系统停止")
    
    def _get_display_price_fallback(self, exchange, symbol):
        """获取显示用价格（仅用于日志，不用于交易）"""
        try:
            ticker = exchange.fetch_ticker(symbol)
            if ticker and 'last' in ticker and ticker['last']:
                return ticker['last']
        except Exception as e:
            self.logger_api.debug(f"获取显示用价格失败: {e}")
            return None
    
    def analyze_with_enhanced_smc(self, price_data: Dict[str, Any], activated_level: Optional[str]) -> Optional[Dict[str, Any]]:
        """使用增强版SMC分析生成交易信号"""
        try:
            # 检查是否有增强版数据
            if 'enhanced_data' not in price_data:
                self.logger_system.warning("未找到增强版数据，回退到标准SMC分析")
                return self.analyze_with_standard_smc(price_data, activated_level)
            
            # 使用增强版SMC信号计算器计算信号
            signal_result = self.enhanced_smc_calculator.calculate_enhanced_smc_signal(price_data['enhanced_data'])
            
            # 生成增强版SMC分析提示词
            enhanced_prompt = get_enhanced_smc_prompt(price_data['enhanced_data'])
            
            # 如果有API密钥，调用DeepSeek进行增强版分析
            if hasattr(self.config, 'deepseek_api_key') and self.config.deepseek_api_key and hasattr(self.config, 'enable_enhanced_ai_analysis') and self.config.enable_enhanced_ai_analysis:
                try:
                    deepseek_response = self._call_deepseek_enhanced(enhanced_prompt)
                    signal_result.update({
                        'deepseek_enhanced_analysis': deepseek_response,
                        'enhanced_prompt_length': len(enhanced_prompt),
                        'enhanced_data_size': {
                            'candlesticks': len(price_data['enhanced_data'].get('enhanced_candlesticks', [])),
                            'swing_points': len(price_data['enhanced_data'].get('swing_points', [])),
                            'market_depth': len(price_data['enhanced_data'].get('market_depth', [])),
                            'time_sales': len(price_data['enhanced_data'].get('time_sales', []))
                        }
                    })
                except Exception as e:
                    self.logger_system.error(f"增强版DeepSeek分析失败: {e}")
                    signal_result['deepseek_enhanced_analysis'] = None
            else:
                signal_result['deepseek_enhanced_analysis'] = None
            
            # 确保信号结果包含必要的字段
            if 'signal' not in signal_result:
                signal_result['signal'] = 'HOLD'
            if 'confidence' not in signal_result:
                signal_result['confidence'] = 0.5
            if 'reason' not in signal_result:
                signal_result['reason'] = '增强版SMC分析'
            if 'source' not in signal_result:
                signal_result['source'] = 'enhanced_smc'
            
            self.logger_system.info(f"✅ 增强版SMC分析完成: {signal_result['signal']} 信号，置信度 {signal_result['confidence']:.2f}")
            
            return signal_result
            
        except Exception as e:
            self.logger_system.error(f"增强版SMC分析失败: {e}")
            self.logger_system.info("🔄 回退到标准SMC分析")
            return self.analyze_with_standard_smc(price_data, activated_level)
    
    def analyze_with_standard_smc(self, price_data: Dict[str, Any], activated_level: Optional[str]) -> Optional[Dict[str, Any]]:
        """标准SMC分析（回退选项）"""
        try:
            # 简单的规则基础信号生成
            current_price = price_data['price']
            rsi = price_data['technical_data'].get('rsi', 50)
            
            # 基本RSI策略
            if rsi < 30:
                signal = "BUY"
                reason = f"RSI超卖 ({rsi:.1f})"
            elif rsi > 70:
                signal = "SELL"
                reason = f"RSI超买 ({rsi:.1f})"
            else:
                signal = "HOLD"
                reason = f"RSI中性 ({rsi:.1f})"
            
            if signal == "HOLD":
                return {
                    'signal': 'HOLD',
                    'confidence': 0.5,
                    'source': 'standard_smc',
                    'reason': reason,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
            
            # 计算止损和止盈
            atr = price_data['technical_data'].get('atr', current_price * 0.02)
            
            if signal == "BUY":
                stop_loss = current_price - (atr * 2)
                take_profit = current_price + (atr * 3)
            else:  # SELL
                stop_loss = current_price + (atr * 2)
                take_profit = current_price - (atr * 3)
            
            return {
                'signal': signal,
                'confidence': 0.6,
                'source': 'standard_smc',
                'reason': reason,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward_ratio': 1.5,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger_system.error(f"标准SMC分析失败: {e}")
            return None
    
    def _call_deepseek_enhanced(self, prompt: str) -> Dict[str, Any]:
        """
        调用DeepSeek API进行增强版分析
        
        参数:
            prompt: 增强版分析提示词
            
        返回:
            DeepSeek API响应
        """
        try:
            # 这里应该调用真实的DeepSeek API
            # 为了演示，我们返回模拟响应
            
            # 模拟API延迟
            import time
            time.sleep(0.5)
            
            # 模拟增强版AI响应
            return {
                "signal": "BUY" if random.random() > 0.4 else "SELL",  # 稍微偏向买入
                "confidence": random.uniform(0.7, 0.95),
                "reason": f"基于增强版数据和市场微观结构分析，检测到高质量{'看涨' if random.random() > 0.4 else '看跌'}结构",
                "stop_loss": None,  # 将在调用方计算
                "take_profit": None,  # 将在调用方计算
                "risk_reward_ratio": None,  # 将在调用方计算
                "strength": random.uniform(0.8, 0.98),
                "enhanced_data_score": random.uniform(0.8, 0.95),
                "market_microstructure_score": random.uniform(0.75, 0.9),
                "liquidity_analysis_score": random.uniform(0.8, 0.95),
                "order_flow_bias": "bullish" if random.random() > 0.4 else "bearish",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"调用增强版DeepSeek API失败: {e}")
            raise
    
    def save_signal_history(self):
        """保存信号历史"""
        try:
            with open(self.config.signals_file, 'w') as f:
                json.dump(self.signal_history, f, indent=2)
        except Exception as e:
            self.logger_system.error(f"保存信号历史失败: {e}")
    
    def trading_bot(self, activated_level: Optional[str] = None, is_scheduled: bool = False):
        """主要交易逻辑执行方法"""
        if not self.trade_lock.acquire(blocking=False):
            self.logger_system.warning("交易进行中，跳过本次执行")
            return
        
        try:
            start_time = time.time()
            self.logger_system.info("=== 开始增强版交易分析 ===")
            
            # 获取价格数据
            price_data = self._fetch_and_update_data(activated_level)
            if not price_data:
                self.logger_system.error("无法获取价格数据，跳过本次交易")
                return
            
            # 使用增强版SMC分析
            signal_data = self.analyze_with_enhanced_smc(price_data, activated_level)
            if not signal_data:
                self.logger_system.error("SMC分析失败，跳过本次交易")
                return
            
            # 记录信号
            signal_record = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'signal': signal_data,
                'price_data': {
                    'price': price_data['price'],
                    'rsi': price_data['technical_data'].get('rsi', 50),
                    'enhanced_data_available': 'enhanced_data' in price_data
                },
                'activated_level': activated_level,
                'is_scheduled': is_scheduled
            }
            
            self.signal_history.append(signal_record)
            self.save_signal_history()
            
            # 记录分析结果
            self.logger_trading.info(f"🎯 信号分析完成: {signal_data['signal']} (置信度: {signal_data['confidence']:.2f})")
            self.logger_trading.info(f"📝 信号原因: {signal_data['reason']}")
            
            if signal_data['signal'] != 'HOLD':
                self.logger_trading.info(f"🎯 止损: ${signal_data.get('stop_loss', 0):.2f}, 止盈: ${signal_data.get('take_profit', 0):.2f}")
                self.logger_trading.info(f"📊 风险回报比: {signal_data.get('risk_reward_ratio', 0):.2f}:1")
            
            # 如果是模拟模式，不执行实际交易
            if self.config.simulation_mode:
                self.logger_trading.info("🔍 模拟模式 - 不执行实际交易")
                return
            
            # 实际交易执行（这里简化处理，实际应用中需要完整的风险管理）
            if signal_data['signal'] in ['BUY', 'SELL'] and signal_data['confidence'] > self.config.enhanced_smc_min_confidence:
                self.logger_trading.info(f"🚀 准备执行{signal_data['signal']}交易")
                # 这里应该调用实际的交易执行方法
                # self.execute_trade(signal_data, price_data, activated_level)
            else:
                self.logger_trading.info("⏸️ 信号置信度不足或为HOLD，不执行交易")
            
            elapsed = time.time() - start_time
            self.logger_system.info(f"=== 增强版交易分析完成，耗时 {elapsed:.2f} 秒 ===")
            
        except Exception as e:
            self.logger_system.error(f"交易分析过程中发生错误: {e}")
        finally:
            self.trade_lock.release()

def main():
    """主函数"""
    # 初始化日志系统
    loggers = setup_logging('enhanced_trading_bot.log', 'INFO')
    logger_system = logging.getLogger('system')
    
    # 创建增强版配置
    config = EnhancedConfig()
    
    # 检查命令行参数决定运行模式
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--live':
        config.simulation_mode = False  # 实盘模式
        logger_system.info("🚀 启动实盘交易模式")
    else:
        config.simulation_mode = True  # 模拟模式
        logger_system.info("🔍 启动模拟交易模式")
    
    # 创建交易所实例
    exchange = None
    try:
        # 尝试初始化Hyperliquid交易所
        if os.getenv('HYPERLIQUID_WALLET_ADDRESS') and os.getenv('HYPERLIQUID_PRIVATE_KEY'):
            import hyperliquid
            import hyperliquid.ccxt_module as ccxt_module
            
            # 使用HyperliquidSync创建交易所实例
            exchange = hyperliquid.HyperliquidSync()
            logger_system.info("✅ Hyperliquid交易所初始化成功")
        else:
            logger_system.warning("⚠️ 未配置Hyperliquid API密钥，使用模拟模式")
    except Exception as e:
        logger_system.error(f"❌ 交易所初始化失败: {e}")
        logger_system.info("🔄 继续使用模拟模式")
    
    # 如果没有真实交易所，创建模拟交易所
    if exchange is None:
        try:
            # 创建模拟交易所实例
            class MockExchange:
                def __init__(self):
                    self.symbol = config.symbol
                    self.current_price = 115000.0  # 模拟BTC价格
                    
                def fetch_ohlcv(self, symbol, timeframe, limit=None):
                    """模拟OHLCV数据"""
                    import random
                    import time
                    
                    # 生成模拟数据
                    ohlcv = []
                    base_price = self.current_price
                    timestamp = int(time.time() * 1000) - (limit or 200) * 60 * 1000  # 根据时间框架调整
                    
                    for i in range(limit or 200):
                        # 根据时间框架调整时间间隔
                        if timeframe == '1m':
                            interval = 60 * 1000
                        elif timeframe == '3m':
                            interval = 3 * 60 * 1000
                        elif timeframe == '15m':
                            interval = 15 * 60 * 1000
                        elif timeframe == '1h':
                            interval = 60 * 60 * 1000
                        elif timeframe == '4h':
                            interval = 4 * 60 * 60 * 1000
                        elif timeframe == '1d':
                            interval = 24 * 60 * 60 * 1000
                        else:
                            interval = 60 * 1000
                            
                        # 生成随机OHLCV数据
                        open_price = base_price + random.uniform(-100, 100)
                        high_price = open_price + random.uniform(0, 200)
                        low_price = open_price - random.uniform(0, 200)
                        close_price = open_price + random.uniform(-100, 100)
                        volume = random.uniform(100, 1000)
                        
                        ohlcv.append([timestamp + i * interval, open_price, high_price, low_price, close_price, volume])
                        base_price = close_price  # 下一根K线从当前收盘价开始
                    
                    return ohlcv
                
                def fetch_ticker(self, symbol):
                    """模拟ticker数据"""
                    return {
                        'symbol': symbol,
                        'last': self.current_price,
                        'bid': self.current_price - 10,
                        'ask': self.current_price + 10,
                        'baseVolume': 1000,
                        'quoteVolume': self.current_price * 1000
                    }
            
            exchange = MockExchange()
            logger_system.info("✅ 模拟交易所初始化成功")
        except Exception as e:
            logger_system.error(f"❌ 模拟交易所初始化失败: {e}")
            logger_system.error("🚨 无法继续，退出")
            return
    
    # 创建增强版交易机器人
    bot = EnhancedTradingBot(config, exchange)
    
    if config.simulation_mode:
        # 模拟模式：执行一次交易分析
        bot.trading_bot()
        print("✅ 增强版交易机器人测试完成")
    else:
        # 实盘模式：持续运行
        logger_system.info("🚀 启动实盘交易机器人，持续运行...")
        
        # 设置定时任务
        schedule.every(15).minutes.do(bot.trading_bot, is_scheduled=True)
        
        # 立即执行一次分析
        bot.trading_bot()
        
        # 持续运行
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            logger_system.info("⏹️ 收到停止信号，正在关闭交易机器人...")
            logger_system.info("✅ 交易机器人已安全停止")

if __name__ == "__main__":
    main()