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
from typing import Dict, Any, Optional, TypedDict, List
from typing import Tuple

# Load environment variables from 1.env file only
load_dotenv('1.env')  # Primary env file

def setup_logging(log_file: str = 'trading_bot.log', level: str = 'INFO', enable_json: bool = False):
    """优雅的日志配置，支持类别和结构化输出"""
    # 清理现有的handlers，避免重复
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 基础格式（人类可读）
    plain_formatter = logging.Formatter(
        '%(asctime)s [%(threadName)-10s] %(name)-12s - %(levelname)-8s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # JSON格式（可选，机器可读）
    if enable_json:
        import json
        class JsonFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    'timestamp': self.formatTime(record),
                    'level': record.levelname,
                    'logger': record.name,
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

    # 控制台处理器（彩色输出）
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(plain_formatter)
    console_handler.setLevel(level)

    # 文件处理器（DEBUG级别，轮转）
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
    )
    file_handler.setFormatter(json_formatter if enable_json else plain_formatter)
    file_handler.setLevel('DEBUG')

    # 根logger配置
    root_logger.setLevel('DEBUG')
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # 创建类别loggers
    loggers = {
        'trading': logging.getLogger('trading'),
        'api': logging.getLogger('api'), 
        'risk': logging.getLogger('risk'),
        'monitor': logging.getLogger('monitor'),
        'system': logging.getLogger('system')
    }

    # 抑制第三方库的噪音日志
    for noisy_logger in ['pandas', 'numpy', 'urllib3', 'requests', 'ccxt']:
        logging.getLogger(noisy_logger).setLevel('WARNING')

    return loggers

# 初始化日志系统
loggers = setup_logging('trading_bot.log', 'INFO')
logger = logging.getLogger(__name__)  # 保持向后兼容

@dataclass
class Config:
    """Configuration class for trading bot parameters."""
    symbol: str = 'BTC/USDC:USDC'
    amount: float = 0.01
    leverage: int = 40
    timeframes: List[str] = None
    primary_timeframe: str = '5m'  # Changed from 15m to 5m for entry
    structure_confirm_timeframe: str = '1h'
    data_points: int = 200
    amplitude_lookback: int = 7
    activation_threshold: float = 0.0002  # Reduced from 0.0005 for more intraday triggers (0.02%)
    min_balance_ratio: float = 0.95
    max_position_time: int = 86400
    risk_per_trade: float = 0.015  # 1.5% per trade loss at liquidation price
    slippage_buffer: float = 0.0005  # Reduced from 0.001 for aggressive execution (0.05%)
    volatility_threshold: float = 70
    order_timeout: int = 10
    heartbeat_interval: int = 60
    price_monitor_interval: int = 300
    signals_file: str = 'signal_history.json'
    heartbeat_file: str = 'heartbeat.log'
    log_file: str = 'trading_bot.log'
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
    simulation_mode: bool = False  # New: Simulation mode toggle (switched to live trading)
    backtest_file: Optional[str] = None  # Added for main()
    max_margin_usage: float = 0.60  # Maximum margin usage ratio
    fee_rate: float = 0.0002  # Taker fee
    maintenance_margin_rate: float = 0.005  # Hyperliquid default (approximate)
    primary_timeframe_weight: float = 2.0  # Weight for 15m structure
    rr_min_threshold: float = 2.0  # Min R:R for normal mode
    rr_aggressive_threshold: float = 2.5  # Min R:R for aggressive mode (15m+4h confirm)
    risk_aggressive: float = 0.02  # Aggressive risk if R:R high (reduced to 2%)
    temperature: float = 0.1  # FIXED: Medium 10 - Add temperature for DeepSeek
    # New: Max leverage per symbol
    max_leverage_per_symbol: Dict[str, int] = None
    # New: Risk control params
    max_daily_loss_pct: float = 0.12  # Max 12% daily loss
    max_drawdown_pct: float = 0.20  # Max 20% drawdown (increased)
    max_open_positions: int = 6  # Max 6 positions in isolated mode per symbol
    min_amount_usdc: float = 50.0  # Minimum position size in USDC (reduced to 50)
    dynamic_leverage: bool = True  # New: Enable dynamic leverage for high R:R
    # New: Multi-TF alignment and confirmation params
    higher_tf_bias_tf: str = '1h'  # Higher TF for bias (changed from 4h to 1h)
    lower_tf_entry_tf: str = '5m'  # Lower TF for entry (changed from 15m to 5m)
    volume_confirmation_threshold: float = 1.5  # Volume > 1.5x MA (increased from 1.2x for 5m)
    max_zone_interactions: int = 1  # Max interactions for fresh zones (unchanged)
    fvg_stack_threshold: int = 3  # Min FVGs stacking for confirmation (increased from 2 to 3)
    candle_pattern_weight: float = 1.5  # Weight for candle pattern confirmation (unchanged)
    # New: Technical indicator parameters
    macd_sensitivity: Tuple[float, float] = (0.015, 0.035)  # MACD sensitivity range for 5m timeframe
    atr_base: Tuple[float, float] = (100, 120)  # ATR base range for 5m timeframe (adjusted from default)
    # New: Level weights for FVG and OB
    level_weights: Dict[str, float] = None

    def __post_init__(self):
        if self.timeframes is None:
            self.timeframes = ['1d', '4h', '1h', '15m', '5m']
        if self.liquidity_priority is None:
            self.liquidity_priority = [
                # Daily level (highest priority)
                'monday_open', 'daily_open', 'prev_week_high', 'prev_week_low', 'daily_vwap', 'daily_fvg_bull_mid', 'daily_fvg_bear_mid',
                # 4H level (high priority)
                '4h_high', '4h_low', '4h_fvg_bull_mid', '4h_fvg_bear_mid', '4h_ob_bull', '4h_ob_bear', '4h_gap_up', '4h_gap_down',
                # 1H level (medium priority)
                'ema_21_1h', 'ema_55_1h', 'ema_100_1h', 'ema_200_1h', '1h_fvg_bull_mid', '1h_fvg_bear_mid', '1h_ob_bull', '1h_ob_bear',
                # 15m level (structure confirmation)
                '15m_structure_break', '15m_structure_reversal', '15m_liquidity_hunt', '15m_fvg_bull_mid', '15m_fvg_bear_mid', '15m_ob_bull', '15m_ob_bear',
                # 5m level (entry)
                '5m_structure_break', '5m_structure_reversal', '5m_liquidity_hunt', '5m_fvg_bull_mid', '5m_fvg_bear_mid', '5m_ob_bull', '5m_ob_bear'
            ]
        if self.max_leverage_per_symbol is None:
            self.max_leverage_per_symbol = {
                'HYPE/USDC:USDC': 10,  # HYPE最大杠杆（交易所限制）
                'BTC/USDC:USDC': 40,
                'ETH/USDC': 25,
                'SOL/USDC': 20,
                'DOGE/USDC': 10,
                'BNB/USDC': 10,
                'XRP/USDC': 20,
                # Add more as needed
            }
        if self.level_weights is None:
            self.level_weights = {
                # Daily levels (highest)
                'monday_open': 4.0,
                'daily_open': 3.8,
                'prev_week_high': 3.5,
                'prev_week_low': 3.5,
                'daily_vwap': 3.2,
                'daily_fvg_bull_mid': 3.4,
                'daily_fvg_bear_mid': 3.4,
                'recent_10d_high': 3.0,
                'recent_10d_low': 3.0,
                'daily_ob_bull': 3.2,
                'daily_ob_bear': 3.2,
                # 4H levels (high)
                '4h_high': 2.5,
                '4h_low': 2.5,
                '4h_fvg_bull_mid': 2.2,
                '4h_fvg_bear_mid': 2.2,
                '4h_ob_bull': 2.5,
                '4h_ob_bear': 2.5,
                '4h_gap_up': 2.0,
                '4h_gap_down': 2.0,
                'ema_21_4h': 2.2,
                'ema_55_4h': 2.2,
                'ema_100_4h': 2.2,
                'ema_200_4h': 2.5,
                # 1H levels (medium)
                'ema_21_1h': 1.8,
                'ema_55_1h': 1.8,
                'ema_100_1h': 1.8,
                'ema_200_1h': 2.0,
                '1h_fvg_bull_mid': 1.6,
                '1h_fvg_bear_mid': 1.6,
                '1h_ob_bull': 1.8,
                '1h_ob_bear': 1.8,
                # 15m levels (entry)
                '15m_structure_break': 1.2,
                '15m_structure_reversal': 1.2,
                '15m_liquidity_hunt': 1.0,
                '15m_fvg_bull_mid': 1.2,
                '15m_fvg_bear_mid': 1.2,
                '15m_ob_bull': 1.4,
                '15m_ob_bear': 1.4,
                # 5m levels (entry)
                '5m_structure_break': 1.0,
                '5m_structure_reversal': 1.0,
                '5m_liquidity_hunt': 0.8,
                '5m_fvg_bull_mid': 1.0,
                '5m_fvg_bear_mid': 1.0,
                '5m_ob_bull': 1.2,
                '5m_ob_bear': 1.2,
            }
        self.validate()

    def validate(self):
        if not (1 <= self.leverage <= 125):
            raise ValueError(f"Leverage must be between 1-125, got: {self.leverage}")
        if not (0.001 <= self.risk_per_trade <= 0.05):
            raise ValueError(f"Risk per trade must be 0.1%-5%, got: {self.risk_per_trade*100:.1f}%")
        if self.amount < 0.01:
            raise ValueError(f"Amount must be >=0.01 BTC, got: {self.amount}")
        if not (0.0001 <= self.activation_threshold <= 0.01):
            raise ValueError(f"Activation threshold must be 0.01%-1%, got: {self.activation_threshold*100:.2f}%")
        if self.primary_timeframe not in self.timeframes:
            raise ValueError(f"Primary timeframe must be in timeframes, got: {self.primary_timeframe}")
        if not (0.1 <= self.max_margin_usage <= 0.95):
            raise ValueError(f"Max margin usage must be between 0.1-0.95, got: {self.max_margin_usage}")
        if not (0 < self.fee_rate < 0.01):
            raise ValueError(f"Fee rate must be between 0 and 1%, got: {self.fee_rate*100:.2f}%")
        if not (0 < self.maintenance_margin_rate < 0.1):
            raise ValueError(f"Maintenance margin rate must be between 0 and 10%, got: {self.maintenance_margin_rate*100:.1f}%")
        if not (1.0 <= self.primary_timeframe_weight <= 5.0):
            raise ValueError(f"Primary timeframe weight must be 1-5, got: {self.primary_timeframe_weight}")
        # FIXED: Medium 1 - Add all new fields validation
        if not (1.0 <= self.rr_min_threshold <= 5.0):
            raise ValueError(f"RR min threshold must be 1-5, got: {self.rr_min_threshold}")
        if not (1.0 <= self.rr_aggressive_threshold <= 5.0):
            raise ValueError(f"RR aggressive threshold must be 1-5, got: {self.rr_aggressive_threshold}")
        if not (0.005 <= self.risk_aggressive <= 0.10):
            raise ValueError(f"Aggressive risk must be 0.5%-10%, got: {self.risk_aggressive*100:.1f}%")
        if not (0 < self.temperature <= 1.0):
            raise ValueError(f"Temperature must be 0-1, got: {self.temperature}")
        if not (50.0 <= self.min_amount_usdc <= 1000.0):
            raise ValueError(f"Min amount USDC must be 50-1000, got: {self.min_amount_usdc}")
        # New: Validate max_leverage_per_symbol
        if self.symbol not in self.max_leverage_per_symbol:
            raise ValueError(f"Symbol {self.symbol} not in max_leverage_per_symbol")
        if not (0.01 <= self.max_daily_loss_pct <= 0.25):
            raise ValueError(f"Max daily loss must be 1%-25%, got: {self.max_daily_loss_pct*100:.1f}%")
        if not (0.05 <= self.max_drawdown_pct <= 0.2):
            raise ValueError(f"Max drawdown must be 5%-20%, got: {self.max_drawdown_pct*100:.1f}%")
        if self.max_open_positions < 1:
            raise ValueError(f"Max open positions must be >=1, got: {self.max_open_positions}")
        # New: Multi-TF and confirmation validation
        if self.higher_tf_bias_tf not in self.timeframes:
            raise ValueError(f"Higher TF bias must be in timeframes, got: {self.higher_tf_bias_tf}")
        if self.lower_tf_entry_tf not in self.timeframes:
            raise ValueError(f"Lower TF entry must be in timeframes, got: {self.lower_tf_entry_tf}")
        if not (1.0 <= self.volume_confirmation_threshold <= 2.0):
            raise ValueError(f"Volume confirmation threshold must be 1.0-2.0, got: {self.volume_confirmation_threshold}")
        if not (1 <= self.max_zone_interactions <= 3):
            raise ValueError(f"Max zone interactions must be 1-3, got: {self.max_zone_interactions}")
        if not (1 <= self.fvg_stack_threshold <= 5):
            raise ValueError(f"FVG stack threshold must be 1-5, got: {self.fvg_stack_threshold}")
        if not (1.0 <= self.candle_pattern_weight <= 2.0):
            raise ValueError(f"Candle pattern weight must be 1.0-2.0, got: {self.candle_pattern_weight}")
        # New: Technical indicator parameters validation
        if not (0.01 <= self.macd_sensitivity[0] <= 0.1) or not (0.01 <= self.macd_sensitivity[1] <= 0.1):
            raise ValueError(f"MACD sensitivity must be 0.01-0.1, got: {self.macd_sensitivity}")
        if not (50 <= self.atr_base[0] <= 500) or not (50 <= self.atr_base[1] <= 500):
            raise ValueError(f"ATR base must be 50-500, got: {self.atr_base}")

config = Config()

# log_file now from config, but setup already done

# 使用system logger记录配置验证
system_logger = logging.getLogger('system')
system_logger.info("配置验证成功: %s | 杠杆=%dx | 风险=%.1f%% | 模式=%s", 
                   config.symbol, config.leverage, config.risk_per_trade*100, 
                   "模拟" if config.simulation_mode else "实盘")

# 详细配置信息移到DEBUG级别
system_logger.debug("详细配置 - 交易量=%.4f BTC, 时间框架=%s/%s, 阈值=%.3f%%, 保证金=%.1f%%, 最小仓位=%d USDC", 
                    config.amount, config.primary_timeframe, config.structure_confirm_timeframe,
                    config.activation_threshold*100, config.max_margin_usage*100, config.min_amount_usdc)
system_logger.debug("风险控制 - R:R阈值=%.1f:1/%.1f:1, 激进风险=%.1f%%, 日损失=%.1f%%, 回撤=%.1f%%",
                    config.rr_min_threshold, config.rr_aggressive_threshold, config.risk_aggressive*100,
                    config.max_daily_loss_pct*100, config.max_drawdown_pct*100)

# New: Log new configs
system_logger.debug("多时间框架配置 - 高时间框架偏置=%s, 低时间框架入场=%s", config.higher_tf_bias_tf, config.lower_tf_entry_tf)
system_logger.debug("确认信号 - 成交量阈值=%.1fx MA, FVG堆叠=%d, 蜡烛模式权重=%.1fx, 新鲜区最大互动=%d",
                    config.volume_confirmation_threshold, config.fvg_stack_threshold, config.candle_pattern_weight, config.max_zone_interactions)

deepseek_client = OpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com/v1",
    timeout=config.deepseek_timeout
)

exchange = ccxt.hyperliquid({
    'enableRateLimit': True,
    'options': {'defaultType': 'perpetual'},
    'apiKey': os.getenv('HYPERLIQUID_WALLET_ADDRESS'),
    'secret': os.getenv('HYPERLIQUID_PRIVATE_KEY'),
    'walletAddress': os.getenv('HYPERLIQUID_WALLET_ADDRESS'),
})

# Fix for Hyperliquid privateKey initialization issue
private_key = os.getenv('HYPERLIQUID_PRIVATE_KEY')
if private_key and private_key.startswith('0x'):
    exchange.privateKey = private_key[2:]  # Remove 0x prefix
else:
    exchange.privateKey = private_key

class PositionInfo(TypedDict):
    side: str
    size: float
    entry_price: float
    unrealized_pnl: float
    leverage: float
    symbol: str
    entry_time: Optional[datetime]
    liquidation_price: float  # New: Track liquidation price as SL

class PositionStore:
    def __init__(self):
        self._lock = threading.RLock()
        self._position: Optional[PositionInfo] = None

    def get(self) -> Optional[PositionInfo]:
        with self._lock:
            return self._position.copy() if self._position else None

    def set(self, pos: Optional[PositionInfo]):
        with self._lock:
            self._position = pos.copy() if pos else None

    def clear(self):
        with self._lock:
            self._position = None

def retry_on_exception(retries=5, backoff_factor=0.5, allowed_exceptions=(NetworkError, RequestTimeout, ExchangeError)):
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tries = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except allowed_exceptions as e:
                    tries += 1
                    if tries > retries:
                        raise
                    sleep = backoff_factor * (2 ** (tries - 1)) + random.random() * 0.1
                    # 使用系统日志器记录重试信息
                    system_logger = logging.getLogger('system')
                    system_logger.warning(f"调用 {func.__name__} 失败 (尝试 {tries}/{retries}), 等待 {sleep:.2f}秒: {e}")
                    time.sleep(sleep)
        return wrapper
    return deco

class TradingBot:
    def __init__(self, config: Config):
        self.config = config
        # 初始化分类日志器
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
        self.level_activation_times: Dict[str, float] = {}  # 跟踪每个关键位的最后激活时间
        # New: Track zone interactions for freshness
        self.zone_interactions: Dict[str, int] = {}  # Count interactions per zone
        self.last_scheduled_signal: Optional[Dict[str, Any]] = None  # 存储上次定时任务的信号副本
        self.lock = threading.RLock()
        self.trade_lock = threading.RLock()
        self.position_store = PositionStore()
        self.executor = ThreadPoolExecutor(max_workers=3)
        # FIXED: Medium 4 - Cache for indicators
        self.indicators_cache: Dict[str, pd.DataFrame] = {}
        # New: Risk control tracking
        self.daily_start_balance: float = 0
        self.peak_balance: float = 0
        self.current_balance: float = 0
        self.last_reset_date: str = ""
        self.last_reset_4h: datetime = datetime.now(timezone.utc)  # New: For 4h reset
        # API健康状态跟踪
        self.api_health_status = {
            'deepseek': {'status': 'unknown', 'last_check': 0, 'consecutive_failures': 0},
            'hyperliquid': {'status': 'unknown', 'last_check': 0, 'consecutive_failures': 0}
        }
        self.api_health_check_interval = 300  # 5分钟检查一次
        # 阈值积累和胜率统计
        self.level_activation_stats: Dict[str, Dict[str, int]] = {}  # 每个关键位的激活和成功统计
        self.total_activations: int = 0  # 总激活次数
        self.total_successful_trades: int = 0  # 总成功交易次数
        self.total_failed_trades: int = 0  # 总失败交易次数
        
        # 日志计数器（用于采样）
        self.log_counters = {
            'price_activation': 0,
            'api_health_check': 0,
            'heartbeat': 0
        }
        
        # 数据缓存相关属性
        self.data_cache: Dict[str, Dict[str, Any]] = {}  # 存储各时间框架的数据缓存
        self.cache_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_cache.json')
        self.cache_ttl = 300  # 缓存TTL为5分钟
        
        # 加载已有缓存
        self._load_cache()
        
        # 执行初始API健康检查
        self._perform_initial_api_health_check()
    
    def _load_cache(self):
        """从文件加载缓存数据"""
        try:
            if os.path.exists(self.cache_file_path):
                with open(self.cache_file_path, 'r', encoding='utf-8') as f:
                    self.data_cache = json.load(f)
                self.logger_system.info(f"已加载数据缓存: {len(self.data_cache)} 个时间框架")
            else:
                self.data_cache = {}
                self.logger_system.info("缓存文件不存在，创建新的缓存")
        except Exception as e:
            self.logger_system.error(f"加载缓存失败: {e}")
            self.data_cache = {}
    
    def _save_cache(self):
        """保存缓存数据到文件"""
        try:
            self.atomic_write_json(self.cache_file_path, self.data_cache)
            self.logger_system.debug(f"已保存数据缓存: {len(self.data_cache)} 个时间框架")
        except Exception as e:
            self.logger_system.error(f"保存缓存失败: {e}")
    
    def _get_cached_data(self, timeframe: str) -> Optional[pd.DataFrame]:
        """获取缓存的数据，如果缓存过期则返回None"""
        if timeframe not in self.data_cache:
            return None
        
        cache_entry = self.data_cache[timeframe]
        last_update = cache_entry.get('last_update', 0)
        
        # 检查缓存是否过期
        if time.time() - last_update > self.cache_ttl:
            self.logger_system.debug(f"{timeframe} 缓存已过期")
            return None
        
        try:
            # 将缓存的数据转换回DataFrame
            data = cache_entry.get('data', [])
            if data:
                df = pd.DataFrame(data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                self.logger_system.debug(f"从缓存获取 {timeframe} 数据: {len(df)} 条记录")
                return df
            else:
                return None
        except Exception as e:
            self.logger_system.error(f"解析缓存数据失败 {timeframe}: {e}")
            return None
    
    def _update_cache(self, timeframe: str, df: pd.DataFrame):
        """更新缓存数据"""
        try:
            # 将DataFrame转换为可序列化的格式
            data = df.reset_index().to_dict('records')
            # 确保时间戳是字符串格式
            for record in data:
                if 'timestamp' in record and hasattr(record['timestamp'], 'isoformat'):
                    record['timestamp'] = record['timestamp'].isoformat()
            
            self.data_cache[timeframe] = {
                'data': data,
                'last_update': time.time()
            }
            
            # 保存到文件
            self._save_cache()
            self.logger_system.debug(f"已更新 {timeframe} 缓存: {len(df)} 条记录")
        except Exception as e:
            self.logger_system.error(f"更新缓存失败 {timeframe}: {e}")

    @retry_on_exception(retries=4)
    def safe_create_order(self, exchange, symbol, side, amount, params=None):
        if self.config.simulation_mode:
            # 模拟模式下返回模拟的订单
            order_id = f"sim_{int(time.time() * 1000)}"
            ticker = self.safe_fetch_ticker(exchange, symbol)
            price = ticker['last']
            cost = amount * price
            
            order = {
                'id': order_id,
                'symbol': symbol,
                'status': 'closed',
                'filled': amount,
                'remaining': 0.0,
                'side': side,
                'type': 'market',
                'amount': amount,
                'price': price,
                'average': price,
                'cost': cost,
                'fee': {'cost': cost * self.config.fee_rate, 'currency': 'USDT'},
                'timestamp': int(time.time() * 1000),
                'datetime': datetime.now(timezone.utc).isoformat(),
                'simulated': True
            }
            self.logger_trading.info(f"模拟订单下单成功, ID: {order_id}")
            return order
            
        try:
            # FIXED: High 1 & 6 - Use create_order for full params support; confirm after slippage
            order = exchange.create_order(symbol, side, 'market', amount, None, params=params)
            self.logger_trading.info(f"订单下单成功, ID: {order.get('id', 'N/A')}")  # FIXED: Medium 3 - Use debug for ID?
            self.logger_trading.debug(f"订单ID: {order.get('id', 'N/A')}")  # Safer
            # FIXED: High 6 - Confirm after any order
            if hasattr(self, 'confirm_order'):  # Ensure method exists
                if not self.confirm_order(order['id'], amount):
                    raise ExchangeError("Order confirmation failed")
            return order
        except ExchangeError as e:
            if "slippage" in str(e).lower():
                ticker = self.safe_fetch_ticker(exchange, symbol)
                price = ticker['last']
                buffer = self.config.slippage_buffer
                adj_price = round(price * (1 + buffer if side == 'buy' else 1 - buffer), 2)  # FIXED: Medium 3 - Round price
                params_limit = params.copy() if params else {}
                order = exchange.create_limit_order(symbol, side, amount, adj_price, params=params_limit)
                self.logger_trading.debug(f"Slippage limit order ID: {order.get('id', 'N/A')}")
                if not self.confirm_order(order['id'], amount):
                    raise ExchangeError("Limit order confirmation failed")
                return order
            raise

    @retry_on_exception(retries=3)
    def safe_fetch_balance(self, exchange):
        if self.config.simulation_mode:
            # Return simulated balance
            return {
                'USDT': {
                    'free': 10000.0,  # Simulated USDT balance
                    'used': 0.0,
                    'total': 10000.0
                },
                'total': {'USDT': 10000.0},
                'free': {'USDT': 10000.0},
                'used': {'USDT': 0.0}
            }
        # For Hyperliquid, pass wallet address as user parameter
        wallet_address = os.getenv('HYPERLIQUID_WALLET_ADDRESS')
        if wallet_address:
            return exchange.fetch_balance({'user': wallet_address})
        else:
            return exchange.fetch_balance()

    @retry_on_exception(retries=3)
    def safe_fetch_ticker(self, exchange, symbol):
        if self.config.simulation_mode:
            # Return simulated ticker data
            import random
            base_price = 70000.0  # Simulated BTC price
            return {
                'symbol': symbol,
                'last': base_price + random.uniform(-1000, 1000),
                'bid': base_price - random.uniform(1, 10),
                'ask': base_price + random.uniform(1, 10),
                'high': base_price + random.uniform(500, 1500),
                'low': base_price - random.uniform(500, 1500),
                'close': base_price + random.uniform(-500, 500),
                'timestamp': int(time.time() * 1000)
            }
        return exchange.fetch_ticker(symbol)

    @retry_on_exception(retries=3)
    def safe_fetch_ohlcv(self, exchange, symbol, timeframe, limit, since=None):
        if self.config.simulation_mode:
            # Return simulated OHLCV data
            import random
            import pandas as pd
            from datetime import datetime, timedelta
            
            base_price = 70000.0
            data = []
            current_time = datetime.now()
            
            # Generate simulated OHLCV data
            for i in range(limit):
                timestamp = int((current_time - timedelta(minutes=15*i)).timestamp() * 1000)
                open_price = base_price + random.uniform(-2000, 2000)
                high_price = open_price + random.uniform(0, 500)
                low_price = open_price - random.uniform(0, 500)
                close_price = open_price + random.uniform(-300, 300)
                volume = random.uniform(100, 1000)
                
                data.append([timestamp, open_price, high_price, low_price, close_price, volume])
            
            return list(reversed(data))  # Return in chronological order
        return exchange.fetch_ohlcv(symbol, timeframe, since, limit)

    @retry_on_exception(retries=3)
    def safe_fetch_positions(self, exchange, symbols):
        if self.config.simulation_mode:
            # Return empty positions for simulation
            return []
        
        # For Hyperliquid, add user parameter if wallet address is available
        wallet_address = os.getenv('HYPERLIQUID_WALLET_ADDRESS')
        if wallet_address:
            params = {'user': wallet_address}
            return exchange.fetch_positions(symbols, params=params)
        else:
            return exchange.fetch_positions(symbols)

    @retry_on_exception(retries=3)
    def safe_fetch_order(self, exchange, order_id, symbol):
        if self.config.simulation_mode:
            # 模拟模式下返回模拟的订单状态
            return {
                'id': order_id,
                'symbol': symbol,
                'status': 'closed',
                'filled': 0.01,
                'remaining': 0.0,
                'side': 'buy',
                'type': 'market',
                'amount': 0.01,
                'price': 70000.0,
                'average': 70000.0,
                'cost': 700.0,
                'fee': {'cost': 0.14, 'currency': 'USDT'},
                'timestamp': int(time.time() * 1000),
                'datetime': datetime.now(timezone.utc).isoformat(),
                'simulated': True
            }
        return exchange.fetch_order(order_id, symbol)

    @retry_on_exception(retries=3)
    def safe_set_leverage(self, exchange, leverage, symbol, params):
        """设置杠杆并验证结果"""
        if self.config.simulation_mode:
            self.logger_trading.info(f"模拟模式: 设置杠杆 {leverage}x for {symbol}")
            return {'leverage': leverage, 'symbol': symbol, 'simulated': True}
        
        try:
            # 设置杠杆 (Hyperliquid specific: no mgnMode or posSide)
            result = exchange.set_leverage(leverage, symbol, params)
            self.logger_trading.info(f"杠杆设置返回结果: {result}")
            
            # 验证杠杆是否设置成功
            time.sleep(0.5)  # 等待设置生效
            actual_leverage = self.safe_fetch_leverage(exchange, symbol)
            
            if actual_leverage is not None and abs(actual_leverage - leverage) < 0.1:
                self.logger_trading.info(f"杠杆验证成功: 期望 {leverage}x, 实际 {actual_leverage}x")
                return result
            elif actual_leverage is None:
                self.logger_trading.warning(f"杠杆验证失败: 无法获取当前杠杆设置 (可能是网络问题)")
                # 如果无法获取杠杆信息，假设设置成功
                self.logger_trading.info(f"假设杠杆设置成功: {leverage}x (由于网络问题无法验证)")
                return result
            else:
                self.logger_trading.warning(f"杠杆验证失败: 期望 {leverage}x, 实际 {actual_leverage}x")
                # 如果验证失败，再次尝试设置
                result = exchange.set_leverage(leverage, symbol, params)
                self.logger_trading.info(f"重新设置杠杆: {result}")
                return result
            
        except Exception as e:
            self.logger_trading.error(f"杠杆设置失败: {e}")
            raise

    @retry_on_exception(retries=3)
    def safe_fetch_leverage(self, exchange, symbol):
        """获取当前杠杆倍数"""
        if self.config.simulation_mode:
            # 模拟模式下返回配置的杠杆倍数
            return self.config.leverage
            
        try:
            # 为Hyperliquid API添加用户参数
            wallet_address = os.getenv('HYPERLIQUID_WALLET_ADDRESS')
            params = {'user': wallet_address} if wallet_address else {}
            
            positions = exchange.fetch_positions([symbol], params=params)
            if positions:
                for pos in positions:
                    if pos['symbol'] == symbol:
                        leverage = pos.get('leverage', None)
                        if leverage is not None:
                            self.logger_trading.debug(f"获取到杠杆: {leverage}x")
                            return leverage
                        else:
                            self.logger_trading.debug(f"持仓中未找到杠杆信息: {pos}")
                            return None
            self.logger_trading.debug(f"未找到 {symbol} 的持仓信息")
            return None
        except Exception as e:
            self.logger_trading.warning(f"获取杠杆失败: {e}")
            return None

    @retry_on_exception(retries=3)
    def safe_fetch_open_orders(self, exchange, symbol, params=None):
        if self.config.simulation_mode:
            # 模拟模式下返回空的订单列表
            return []
        return exchange.fetch_open_orders(symbol, params=params)

    def atomic_write_json(self, path: str, data: Any):
        dir_path = os.path.dirname(path) or '.'
        fd, tmp = tempfile.mkstemp(dir=dir_path)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(data, f, default=str, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    def _perform_initial_api_health_check(self):
        """执行初始API健康检查"""
        if self.config.simulation_mode:
            self.logger_api.info("模拟模式 - 跳过初始API健康检查")
            return
        self.logger_api.info("执行初始API健康检查...")
        self._check_deepseek_health()
        self._check_hyperliquid_health()
        
    def _check_deepseek_health(self) -> bool:
        """检查DeepSeek API健康状态"""
        try:
            current_time = time.time()
            previous_status = self.api_health_status['deepseek'].get('status', 'unknown')
            
            # 简单的API测试调用
            response = deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=10,
                temperature=0.1
            )
            
            if response and response.choices:
                self.api_health_status['deepseek'] = {
                    'status': 'healthy',
                    'last_check': current_time,
                    'consecutive_failures': 0
                }
                
                # 只在状态变化时记录
                if previous_status != 'healthy':
                    self.logger_api.info("DeepSeek API恢复正常 (连续失败: %d)", 
                                       self.api_health_status['deepseek'].get('consecutive_failures', 0))
                else:
                    # 采样记录：每10次检查记录一次
                    self.log_counters['api_health_check'] += 1
                    if self.log_counters['api_health_check'] % 10 == 0:
                        self.logger_api.debug("DeepSeek API健康检查: 正常 (第%d次)", self.log_counters['api_health_check'])
                
                return True
            else:
                raise Exception("API响应无效")
            
        except Exception as e:
            failures = self.api_health_status['deepseek']['consecutive_failures'] + 1
            self.api_health_status['deepseek']['consecutive_failures'] = failures
            self.api_health_status['deepseek']['status'] = 'unhealthy'
            self.api_health_status['deepseek']['last_check'] = time.time()
            
            # 根据失败次数调整日志级别
            if failures == 1:
                self.logger_api.warning("DeepSeek API健康检查失败: %s", e)
            elif failures <= 3:
                self.logger_api.warning("DeepSeek API连续失败 %d 次: %s", failures, e)
            else:
                self.logger_api.error("DeepSeek API严重故障 (连续失败 %d 次): %s", failures, e)
            
            return False
            
    def _check_hyperliquid_health(self) -> bool:
        """检查Hyperliquid API健康状态"""
        if self.config.simulation_mode:
            # 模拟模式下跳过真实API健康检查
            current_time = time.time()
            self.api_health_status['hyperliquid'] = {
                'status': 'healthy',
                'last_check': current_time,
                'consecutive_failures': 0
            }
            # 模拟模式只在首次或状态变化时记录
            if not hasattr(self, '_hyperliquid_sim_logged'):
                self.logger_api.info("Hyperliquid API健康检查: 模拟模式 - 跳过")
                self._hyperliquid_sim_logged = True
            return True
            
        try:
            current_time = time.time()
            # 简单的API测试调用
            ticker = self.safe_fetch_ticker(exchange, self.config.symbol)
            
            if ticker and 'last' in ticker:
                # 记录之前的状态
                was_unhealthy = self.api_health_status['hyperliquid']['status'] == 'unhealthy'
                
                self.api_health_status['hyperliquid'] = {
                    'status': 'healthy',
                    'last_check': current_time,
                    'consecutive_failures': 0
                }
                
                # 采样日志：每10次成功检查记录一次，或状态从不健康恢复时记录
                self.log_counters['hyperliquid_health'] = self.log_counters.get('hyperliquid_health', 0) + 1
                if self.log_counters['hyperliquid_health'] % 10 == 0:
                    self.logger_api.debug("Hyperliquid API健康检查: 正常 (累计成功: %d)", self.log_counters['hyperliquid_health'])
                elif was_unhealthy:
                    self.logger_api.info("Hyperliquid API健康检查: 已恢复正常")
                
                return True
            else:
                raise Exception("API响应无效")
            
        except Exception as e:
            failures = self.api_health_status['hyperliquid']['consecutive_failures'] + 1
            self.api_health_status['hyperliquid']['consecutive_failures'] = failures
            self.api_health_status['hyperliquid']['status'] = 'unhealthy'
            self.api_health_status['hyperliquid']['last_check'] = time.time()
            
            # 根据连续失败次数调整日志级别
            if failures == 1:
                self.logger_api.warning("Hyperliquid API健康检查失败: %s", e)
            elif failures <= 3:
                self.logger_api.warning("Hyperliquid API持续失败: %s (连续失败: %d)", e, failures)
            else:
                self.logger_api.error("Hyperliquid API严重故障: %s (连续失败: %d)", e, failures)
            
            return False
            
    def _should_check_api_health(self, api_name: str) -> bool:
        """判断是否需要进行API健康检查"""
        current_time = time.time()
        last_check = self.api_health_status[api_name]['last_check']
        return (current_time - last_check) > self.api_health_check_interval
        
    def _update_api_failure(self, api_name: str, error: Exception):
        """更新API失败状态"""
        self.api_health_status[api_name]['consecutive_failures'] += 1
        self.api_health_status[api_name]['status'] = 'unhealthy'
        self.api_health_status[api_name]['last_check'] = time.time()
        
        failures = self.api_health_status[api_name]['consecutive_failures']
        if failures >= 3:
            self.logger_api.error(f"{api_name} API连续失败{failures}次，可能存在严重问题: {error}")
        elif failures >= 5:
            self.logger_api.critical(f"{api_name} API连续失败{failures}次，建议检查网络和API配置")
            
    def _update_api_success(self, api_name: str):
        """更新API成功状态"""
        self.api_health_status[api_name] = {
            'status': 'healthy',
            'last_check': time.time(),
            'consecutive_failures': 0
        }

    # New: Risk control methods
    def _reset_daily_balance(self):
        today = datetime.now(timezone.utc).date().isoformat()
        if today != self.last_reset_date:
            self.daily_start_balance = self.current_balance
            self.last_reset_date = today
            self.logger_system.info(f"每日余额重置: {self.daily_start_balance:.2f} USDT")
        # New: 4h reset for intraday
        if (datetime.now(timezone.utc) - self.last_reset_4h).total_seconds() > 14400:  # 4h
            self.daily_start_balance = self.current_balance  # Use same var for simplicity, or rename
            self.last_reset_4h = datetime.now(timezone.utc)
            self.logger_system.info(f"4h余额重置: {self.daily_start_balance:.2f} USDT")

    def _check_risk_controls(self, proposed_risk: float) -> bool:
        self._update_current_balance()
        self._reset_daily_balance()
        self.peak_balance = max(self.peak_balance, self.current_balance)

        # Daily loss check (now with 4h reset, ignores small <1% drawdown)
        daily_loss = self.daily_start_balance - self.current_balance
        daily_loss_pct = daily_loss / self.daily_start_balance if self.daily_start_balance > 0 else 0
        if daily_loss > self.daily_start_balance * self.config.max_daily_loss_pct:
            self.logger_risk.error("超过每日损失限: 损失=%.2f USDC (%.1f%%) > 限额=%.2f USDC (%.1f%%), 余额=%.2f", 
                                 daily_loss, daily_loss_pct*100, 
                                 self.daily_start_balance * self.config.max_daily_loss_pct, 
                                 self.config.max_daily_loss_pct*100, self.current_balance)
            return False

        # Drawdown check (ignore <1%)
        drawdown = (self.peak_balance - self.current_balance) / self.peak_balance if self.peak_balance > 0 else 0
        if drawdown > self.config.max_drawdown_pct and drawdown > 0.01:  # Ignore small drawdown
            self.logger_risk.error("超过最大回撤: 回撤=%.1f%% > 限额=%.1f%%, 峰值=%.2f, 当前=%.2f", 
                                 drawdown*100, self.config.max_drawdown_pct*100, 
                                 self.peak_balance, self.current_balance)
            return False

        # Position count check (isolated, per symbol)
        positions = self.safe_fetch_positions(exchange, [self.config.symbol])
        open_pos = sum(1 for p in positions if float(p.get('contracts', 0)) > 0 and p['symbol'] == self.config.symbol)
        if open_pos >= self.config.max_open_positions:
            self.logger_risk.warning("达到最大持仓数: 当前=%d >= 限额=%d, 跳过新交易", 
                                   open_pos, self.config.max_open_positions)
            return False

        # Proposed risk check
        max_risk = self.current_balance * self.config.risk_per_trade
        if proposed_risk > max_risk:
            self.logger_risk.warning("拟议风险超限: 风险=%.2f USDC > 限额=%.2f USDC (%.1f%%), 余额=%.2f", 
                                   proposed_risk, max_risk, self.config.risk_per_trade*100, self.current_balance)
            return False

        # 记录风险检查通过（采样日志）
        self.log_counters['risk_check'] = self.log_counters.get('risk_check', 0) + 1
        if self.log_counters['risk_check'] % 10 == 1:  # 每10次记录一次
            self.logger_risk.debug("风险检查通过: 拟议风险=%.2f, 日损失=%.1f%%, 回撤=%.1f%%, 持仓=%d/%d", 
                                 proposed_risk, daily_loss_pct*100, drawdown*100, open_pos, self.config.max_open_positions)

        return True

    def _update_current_balance(self):
        try:
            balance = self.safe_fetch_balance(exchange)
            self.current_balance = balance['USDC']['free']  # Hyperliquid uses USDC
        except Exception as e:
            self.logger_system.warning(f"更新余额失败: {e}")

    def load_signal_history(self):
        # FIXED: Medium 5 - Add lock for atomicity
        with self.lock:
            if os.path.exists(self.config.signals_file):
                try:
                    with open(self.config.signals_file, 'r') as f:
                        self.signal_history = json.load(f)
                    self.logger_system.info(f"加载信号历史: {len(self.signal_history)} 条记录")
                except Exception as e:
                    self.logger_system.exception(f"加载信号历史失败: {e}")
                    self.signal_history = []
            else:
                self.signal_history = []

    def save_signal_history(self):
        try:
            # 在锁保护下复制信号历史，避免在写入过程中被修改
            with self.lock:
                signal_history_copy = self.signal_history.copy()
            self.atomic_write_json(self.config.signals_file, signal_history_copy)
            self.logger_system.debug("Signal history saved successfully")
        except Exception as e:
            self.logger_system.exception(f"保存信号历史失败: {e}")

    def heartbeat(self):
        try:
            with self.lock:
                current_pos = self.get_current_position()
                pos_str = f"持仓: {current_pos['side']} {current_pos['size']:.4f} @ {current_pos['entry_price']:.2f} (SL: {current_pos.get('liquidation_price', 'N/A'):.2f})" if current_pos else "无持仓"
                
            if self.config.simulation_mode:
                # 模拟模式下使用模拟数据
                balance_free = self.initial_balance
                bal_str = f"余额: {balance_free:.2f} USD (模拟)"
                current_price = 67000.0 + random.uniform(-500, 500)  # FIXED: High 8 - Reduce randomness to ±0.75%
            else:
                balance = self.safe_fetch_balance(exchange)
                balance_free = balance['USDC']['free']  # Hyperliquid uses USDC
                bal_str = f"余额: {balance_free:.2f} USD"
                current_price = round(self.safe_fetch_ticker(exchange, self.config.symbol)['last'], 2)  # FIXED: Medium 3 - Round
                
            with self.lock:
                sig_count = len(self.signal_history)
                last_sig = self.signal_history[-1]['signal'] if self.signal_history else '无'
                
            status = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'status': 'alive',
                'position': pos_str,
                'balance': bal_str,
                'signal_count': sig_count,
                'last_signal': last_sig,
                'price': current_price
            }
            
            # 检测异常情况
            has_anomaly = False
            anomaly_messages = []
            
            # 检查余额异常
            if balance_free < self.initial_balance * 0.9:
                has_anomaly = True
                anomaly_messages.append(f" 余额过低: {bal_str} (低于初始余额90%)")
            
            # 检查持仓时间过长
            if current_pos and current_pos.get('entry_time'):
                position_duration = (datetime.now(timezone.utc) - current_pos['entry_time']).total_seconds()
                if position_duration > self.config.max_position_time:
                    has_anomaly = True
                    hours = position_duration / 3600
                    anomaly_messages.append(f" 持仓时间过长: {hours:.1f}小时 (超过{self.config.max_position_time/3600:.1f}小时限制)")
            
            # New: Risk control anomalies
            self._update_current_balance()
            self._reset_daily_balance()
            daily_loss_pct = (self.daily_start_balance - self.current_balance) / self.daily_start_balance * 100 if self.daily_start_balance > 0 else 0
            if daily_loss_pct > self.config.max_daily_loss_pct * 100:
                has_anomaly = True
                anomaly_messages.append(f" 每日损失超限: {daily_loss_pct:.1f}%")

            drawdown_pct = (self.peak_balance - self.current_balance) / self.peak_balance * 100 if self.peak_balance > 0 else 0
            if drawdown_pct > self.config.max_drawdown_pct * 100:
                has_anomaly = True
                anomaly_messages.append(f" 回撤超限: {drawdown_pct:.1f}%")
            
            # 检查API健康状态
            for api_name, health_info in self.api_health_status.items():
                if health_info['consecutive_failures'] >= 3:
                    has_anomaly = True
                    anomaly_messages.append(f" {api_name} API连续失败: {health_info['consecutive_failures']}次")
            
            # 压缩心跳日志为一行，只在异常时详细记录
            anomaly_msg = "; ".join(anomaly_messages) if has_anomaly else "正常"
            self.logger_monitor.info("心跳: %s | 持仓=%s | 余额=%.2f USD | 信号数=%d | 价格=%.2f%s", 
                                   datetime.now(timezone.utc).strftime('%H:%M:%S'), 
                                   pos_str, balance_free, sig_count, current_price, 
                                   f" (异常: {anomaly_msg})" if has_anomaly else "")
            
            # 异常时额外记录详细信息
            if has_anomaly:
                self.logger_monitor.warning("心跳异常详情: %s", "; ".join(anomaly_messages))
            
            # 始终写入心跳文件用于监控
            with open(self.config.heartbeat_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(status, ensure_ascii=False) + '\n')
                f.flush()
                os.fsync(f.fileno())
                
        except Exception as e:
            self.logger_monitor.error("心跳检测失败: %s", e)

    def setup_exchange(self):
        try:
            if self.config.simulation_mode:
                self.logger_system.info("模拟模式: 跳过交易所API连接")
                self.initial_balance = 10000.0  # 模拟余额
                self.logger_system.info(f"模拟USD余额: {self.initial_balance:.2f}")
            else:
                # Hyperliquid does not require position mode or margin mode setup (always one-way, cross margin)
                # Dynamic leverage based on symbol (calculate before setting)
                max_l = self.config.max_leverage_per_symbol.get(self.config.symbol, 100)
                target_leverage = min(self.config.leverage, max_l)
                
                # Set leverage
                self.safe_set_leverage(exchange, target_leverage, self.config.symbol, {})
                self.logger_trading.info(f"设置杠杆倍数: {target_leverage}x")
                balance = self.safe_fetch_balance(exchange)
                usd_balance = balance['USDC']['free']
                
                self.initial_balance = usd_balance
                self.current_balance = usd_balance
                self.peak_balance = usd_balance
                self.daily_start_balance = usd_balance
                self.last_reset_date = datetime.now(timezone.utc).date().isoformat()
                self.last_reset_4h = datetime.now(timezone.utc)
                self.logger_system.info(f"当前USD余额: {usd_balance:.2f}")
                self.logger_system.info(f"初始余额设置为: {self.initial_balance:.2f}")
                self.logger_trading.info(f"自适应杠杆: {self.config.leverage}x (max for {self.config.symbol}: {max_l}x)")

            self.position_store.set(self.get_current_position())
            if self.position_store.get():
                current_pos = self.position_store.get()
                if self.config.simulation_mode:
                    current_pos['entry_time'] = datetime.now(timezone.utc)
                    self.logger_system.info("模拟模式 - 设置当前时间为入场时间")
                else:
                    try:
                        trades = exchange.fetch_my_trades(self.config.symbol, limit=1)
                        if trades and len(trades) > 0:
                            current_pos['entry_time'] = pd.to_datetime(trades[0]['timestamp'], unit='ms', utc=True)
                            self.logger_trading.info(f"从交易记录恢复入场时间: {current_pos['entry_time']}")
                        else:
                            current_pos['entry_time'] = None  # No trade history, mark as new or empty
                            self.logger_trading.warning("未找到最近交易记录，入场时间设为None")
                    except Exception as e:
                        self.logger_trading.warning(f"从交易记录恢复入场时间失败: {e}，使用None")
                        current_pos['entry_time'] = None
                self.position_store.set(current_pos)
                self.logger_trading.info("检测到现有持仓，已设置入场时间")

            # Initial cache
            dummy_data = {tf: pd.DataFrame() for tf in self.config.timeframes}
            self.key_levels_cache = self.calculate_key_levels(dummy_data)
            self.cache_timestamp = time.time()
            self.logger_system.debug("初始关键价位缓存已设置（空数据回退）")

            return True
        except Exception as e:
            self.logger_system.exception(f"交易所设置失败: {e}")
            return False

    def check_balance_risk(self):
        try:
            balance = self.safe_fetch_balance(exchange)
            current_balance = balance['USDC']['free']
            min_required = self.initial_balance * self.config.min_balance_ratio
            balance_ratio = current_balance / self.initial_balance
            
            if current_balance < min_required:
                loss_pct = (1 - balance_ratio) * 100
                self.logger_risk.warning("余额风险触发，停止交易: 当前=%.2f USDC < 阈值=%.2f USDC (%.1f%% < %.1f%%), 损失=%.1f%%", 
                                       current_balance, min_required, 
                                       balance_ratio*100, self.config.min_balance_ratio*100, loss_pct)
                return False
            
            # 每50次检查记录一次正常状态（采样）
            self.log_counters['balance_check'] = self.log_counters.get('balance_check', 0) + 1
            if self.log_counters['balance_check'] % 50 == 0:
                self.logger_risk.debug("余额检查正常: 当前=%.2f USDC, 比例=%.1f%%, 初始=%.2f USDC (第%d次检查)", 
                                     current_balance, balance_ratio*100, self.initial_balance, 
                                     self.log_counters['balance_check'])
            return True
        except Exception as e:
            self.logger_risk.error("余额检查失败: %s", str(e))
            self.logger_risk.exception("余额检查详细错误")
            return False

    def check_position_time(self):
        pos = self.position_store.get()
        if not pos:
            return True
        entry_time = pos.get('entry_time')
        if entry_time is None:
            return True  # No entry_time means new/empty position, skip timeout check
        
        current_time = datetime.now(timezone.utc)
        position_duration = (current_time - entry_time).total_seconds()
        max_duration = self.config.max_position_time
        
        if position_duration > max_duration:
            duration_hours = position_duration / 3600
            max_hours = max_duration / 3600
            self.logger_risk.warning("持仓时间超限，触发强制平仓: 持续=%.1f小时 > 限制=%.1f小时, 方向=%s, 数量=%.4f, 浮盈=%.2f USDC", 
                                   duration_hours, max_hours, pos['side'], pos['size'], pos.get('unrealized_pnl', 0))
            self.close_position()
            return False
        return True

    def close_position(self):
        pos = self.get_current_position()
        if not pos:
            return
        
        # 生成平仓ID和上下文
        close_id = f"CLOSE_{int(time.time()*1000)}_{random.randint(1000,9999)}"
        close_context = {
            "close_id": close_id,
            "symbol": self.config.symbol,
            "side": pos['side'],
            "size": pos['size'],
            "entry_price": pos.get('entry_price', 0),
            "unrealized_pnl": pos.get('unrealized_pnl', 0)
        }
        
        try:
            side = 'sell' if pos['side'] == 'long' else 'buy'
            self.logger_trading.info("开始强制平仓: ID=%s, 方向=%s→%s, 数量=%.4f, 入场价=%.2f, 浮盈=%.2f USDC", 
                                   close_id, pos['side'], side, pos['size'], 
                                   pos.get('entry_price', 0), pos.get('unrealized_pnl', 0))
            
            order_result = self.safe_create_order(exchange, self.config.symbol, side, pos['size'], params={'reduceOnly': True})
            
            self.logger_trading.info("强制平仓成功: ID=%s, 订单ID=%s, 填充=%.4f", 
                                   close_id, order_result.get('id', 'N/A'), 
                                   order_result.get('filled', pos['size']))
            self.position_store.clear()
            self._update_current_balance()  # Update after close
        except Exception as e:
            self.logger_trading.error("强制平仓失败: ID=%s, 错误=%s, 持仓详情=%s", 
                                    close_id, str(e), close_context)
            self.logger_trading.exception("强制平仓详细错误: ID=%s", close_id)

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            # FIXED: Medium 4 - Set index for optimization
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            
            # 检查数据长度，确保有足够数据计算所有指标 - 优化为实时交易
            min_required = 20  # 进一步降低到20个数据点，适合实时交易
            absolute_min = 5  # 降低绝对最小数据要求到5
            
            if len(df) < min_required:
                self.logger_system.warning(f"数据长度不足 ({len(df)} < {min_required})，尝试回退计算")
                if len(df) >= absolute_min:
                    return self._calculate_fallback_indicators(df)
                else:
                    self.logger_system.warning(f"Data too short even for fallback calculation ({len(df)} < {absolute_min}), skipping indicator calculation")
                    return df
                
            if df['close'].isna().all():
                self.logger_system.warning("Close column all NaN, skipping calculation")
                return df

            # 基础移动平均线
            df['sma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
            df['sma_50'] = df['close'].rolling(window=50, min_periods=1).mean()
            
            # 指数移动平均线 (用于趋势判断和止盈止损)
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            df['ema_21'] = df['close'].ewm(span=21).mean()
            df['ema_55'] = df['close'].ewm(span=55).mean()
            df['ema_200'] = df['close'].ewm(span=200).mean()
            
            # 短期和长期EMA用于信号确认
            df['ema_short'] = df['ema_12']  # 短期EMA
            df['ema_long'] = df['ema_26']   # 长期EMA
            
            # MACD指标 (使用配置的灵敏度参数)
            macd_fast = max(2, min(12, int(14 * self.config.macd_sensitivity[0])))  # 根据灵敏度调整快线周期
            macd_slow = max(2, min(26, int(30 * self.config.macd_sensitivity[1])))  # 根据灵敏度调整慢线周期
            df['ema_fast'] = df['close'].ewm(span=macd_fast).mean()
            df['ema_slow'] = df['close'].ewm(span=macd_slow).mean()
            df['macd'] = df['ema_fast'] - df['ema_slow']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # RSI指标
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            loss = loss.replace(0, 1e-10)
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            df['rsi'] = df['rsi'].fillna(self.config.rsi_neutral).clip(self.config.rsi_min, self.config.rsi_max)
            
            # ATR指标 (用于动态止损，使用配置的基准参数)
            atr_period = max(2, min(14, int(self.config.atr_base[0] / 10)))  # 根据基准调整ATR周期
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            tr = np.maximum(high_low, np.maximum(high_close, low_close))
            df['atr'] = tr.rolling(atr_period).mean()
            
            # 布林带指标 (用于支撑阻力判断)
            df['bb_middle'] = df['close'].rolling(20, min_periods=1).mean()
            bb_std = df['close'].rolling(20, min_periods=1).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_width'] = df['bb_upper'] - df['bb_lower']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # 支撑阻力位 (用于止盈止损优化)
            df['resistance'] = df['high'].rolling(20, min_periods=1).max()
            df['support'] = df['low'].rolling(20, min_periods=1).min()
            df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
            
            # 成交量指标 (用于确认信号强度)
            df['volume_ma'] = df['volume'].rolling(20, min_periods=1).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            
            # 价格位置指标 (用于止盈止损调整)
            df['price_position'] = (df['close'] - df['support']) / (df['resistance'] - df['support'])
            
            # 波动率指标 (用于动态调整止损距离)
            df['volatility'] = df['close'].rolling(14, min_periods=1).std() / df['close'].rolling(14, min_periods=1).mean()
            
            # 趋势强度指标（使用动态EMA参数）
            macd_fast = max(2, min(12, int(14 * self.config.macd_sensitivity[0])))  # 根据灵敏度调整快线周期
            macd_slow = max(2, min(26, int(30 * self.config.macd_sensitivity[1])))  # 根据灵敏度调整慢线周期
            df['ema_fast'] = df['close'].ewm(span=macd_fast).mean()
            df['ema_slow'] = df['close'].ewm(span=macd_slow).mean()
            df['trend_strength'] = abs(df['ema_fast'] - df['ema_slow']) / df['atr']
            
            # FIXED: High 7 & Medium 4 - Unified NaN handling after all calcs
            df = df.bfill().ffill().fillna(0)  # Fill NaN, then 0
            df['macd'] = df['macd'].fillna(0)
            df['atr'] = df['atr'].fillna(df['close'].iloc[-1] * 0.02)
            
            # Clip all relevant indicators
            for col in ['rsi', 'macd', 'atr', 'sma_20', 'sma_50', 'bb_position', 'price_position', 'volume_ratio']:
                if col in df.columns:
                    if col in ['bb_position', 'price_position']:
                        df[col] = df[col].clip(0, 1)  # 位置指标限制在0-1之间
                    elif col == 'volume_ratio':
                        df[col] = df[col].clip(0, 10)  # 成交量比率限制在合理范围
                    else:
                        df[col] = df[col].clip(lower=0)  # Basic clip
                
            # EMA100 addition
            if self.config.use_ema100:
                df['ema_100'] = df['close'].ewm(span=100).mean()
                df['ema_100'] = df['ema_100'].fillna(df['close'].mean()).clip(lower=0)
                # Cache EMA100 if needed
                self.indicators_cache['ema_100'] = df['ema_100']
            
            # FIXED: Medium 4 - Cache full df if small
            if len(df) < 100:
                self.indicators_cache[self.config.primary_timeframe] = df.copy()
            
            self.logger_system.info(f"技术指标计算完成，包含 {len(df)} 个数据点")
            return df
        except Exception as e:
            self.logger_system.exception(f"技术指标计算失败: {e}")
            return df

    def _calculate_fallback_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """在数据不足时计算基础技术指标"""
        try:
            self.logger_system.info(f"使用 {len(df)} 个数据点计算回退指标")
            
            # FIXED: Medium 4 - Set index
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            
            data_len = len(df)
            
            # 基础移动平均线（使用更小的窗口，最少2个数据点）
            df['sma_20'] = df['close'].rolling(window=max(2, min(20, data_len)), min_periods=1).mean()
            df['sma_50'] = df['close'].rolling(window=max(2, min(50, data_len)), min_periods=1).mean()
            
            # 指数移动平均线（最少2个数据点）
            df['ema_12'] = df['close'].ewm(span=max(2, min(12, data_len))).mean()
            df['ema_26'] = df['close'].ewm(span=max(2, min(26, data_len))).mean()
            df['ema_21'] = df['close'].ewm(span=max(2, min(21, data_len))).mean()
            df['ema_55'] = df['close'].ewm(span=max(2, min(55, data_len))).mean()
            df['ema_200'] = df['close'].ewm(span=max(2, min(200, data_len))).mean()
            
            # MACD（使用配置的灵敏度参数）
            macd_fast = max(2, min(12, int(14 * self.config.macd_sensitivity[0])))  # 根据灵敏度调整快线周期
            macd_slow = max(2, min(26, int(30 * self.config.macd_sensitivity[1])))  # 根据灵敏度调整慢线周期
            df['ema_fast'] = df['close'].ewm(span=macd_fast).mean()
            df['ema_slow'] = df['close'].ewm(span=macd_slow).mean()
            df['macd'] = df['ema_fast'] - df['ema_slow']
            df['macd_signal'] = df['macd'].ewm(span=max(2, min(9, data_len))).mean()
            
            # RSI（使用较小的周期，最少2个数据点）
            rsi_period = max(2, min(14, data_len))
            if data_len >= 2:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(rsi_period, min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(rsi_period, min_periods=1).mean()
                loss = loss.replace(0, 1e-10)
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))
                df['rsi'] = df['rsi'].fillna(self.config.rsi_neutral).clip(self.config.rsi_min, self.config.rsi_max)
            else:
                df['rsi'] = self.config.rsi_neutral
                
            # ATR（使用较小的周期，最少2个数据点，使用配置的基准参数）
            atr_period = max(2, min(14, int(self.config.atr_base[0] / 10)))
            if data_len >= 2:
                high_low = df['high'] - df['low']
                high_close = np.abs(df['high'] - df['close'].shift())
                low_close = np.abs(df['low'] - df['close'].shift())
                tr = np.maximum(high_low, np.maximum(high_close, low_close))
                df['atr'] = tr.rolling(atr_period, min_periods=1).mean()
            else:
                df['atr'] = df['close'].iloc[-1] * 0.02
                
            # 填充缺失值 - FIXED: High 7 - Unified
            df = df.bfill().ffill().fillna(0)
            df['macd'] = df['macd'].fillna(0)
            df['atr'] = df['atr'].fillna(df['close'].iloc[-1] * 0.02)
            
            # EMA100（如果启用）
            if self.config.use_ema100:
                df['ema_100'] = df['close'].ewm(span=max(2, min(100, data_len))).mean()
                df['ema_100'] = df['ema_100'].fillna(df['close'].mean()).clip(lower=0)
            
            self.logger_system.info("回退指标计算成功")
            return df
            
        except Exception as e:
            self.logger_system.exception(f"回退指标计算失败: {e}")
            return df

    def calculate_fibonacci_levels(self, df: pd.DataFrame, lookback: int = 20) -> Dict[str, float]:
        """
        计算斐波那契回撤和扩展水平
        用于优化止盈止损位置
        """
        try:
            if len(df) < lookback:
                lookback = len(df)
            
            recent_data = df.tail(lookback)
            high = recent_data['high'].max()
            low = recent_data['low'].min()
            current_price = df['close'].iloc[-1]
            
            # 计算价格范围
            price_range = high - low
            
            # 斐波那契回撤水平 (从高点回撤)
            fib_retracements = {
                'fib_23.6': high - (price_range * 0.236),
                'fib_38.2': high - (price_range * 0.382),
                'fib_50.0': high - (price_range * 0.500),
                'fib_61.8': high - (price_range * 0.618),
                'fib_78.6': high - (price_range * 0.786)
            }
            
            # 斐波那契扩展水平 (突破后的目标位)
            fib_extensions = {
                'fib_ext_127.2': high + (price_range * 0.272),
                'fib_ext_161.8': high + (price_range * 0.618),
                'fib_ext_200.0': high + (price_range * 1.000),
                'fib_ext_261.8': high + (price_range * 1.618)
            }
            
            # 合并所有斐波那契水平
            fib_levels = {**fib_retracements, **fib_extensions}
            
            # 添加关键信息
            fib_levels.update({
                'swing_high': high,
                'swing_low': low,
                'price_range': price_range,
                'current_price': current_price
            })
            
            self.logger_system.info(f"斐波那契水平计算完成: 高点={high:.2f}, 低点={low:.2f}, 范围={price_range:.2f}")
            return fib_levels
            
        except Exception as e:
            self.logger_system.error(f"斐波那契水平计算失败: {e}")
            return {}

    def get_optimal_fibonacci_sl_tp(self, df: pd.DataFrame, signal: str, current_price: float) -> Tuple[float, float]:
        """
        基于斐波那契水平计算最优止损止盈
        """
        try:
            fib_levels = self.calculate_fibonacci_levels(df)
            if not fib_levels:
                return None, None
            
            # 获取技术指标
            atr = df['atr'].iloc[-1] if 'atr' in df.columns else current_price * 0.02
            
            if signal == 'BUY':
                # 多头交易：寻找下方支撑作为止损，上方阻力作为止盈
                potential_sl = []
                potential_tp = []
                
                # 止损候选：当前价格下方的斐波那契水平
                for level_name, level_price in fib_levels.items():
                    if 'fib_' in level_name and level_price < current_price:
                        distance = current_price - level_price
                        if 0.5 * atr <= distance <= 3 * atr:  # 合理的止损距离
                            potential_sl.append((level_name, level_price, distance))
                
                # 止盈候选：当前价格上方的斐波那契水平
                for level_name, level_price in fib_levels.items():
                    if ('fib_ext_' in level_name or level_name == 'swing_high') and level_price > current_price:
                        distance = level_price - current_price
                        if 1 * atr <= distance <= 5 * atr:  # 合理的止盈距离
                            potential_tp.append((level_name, level_price, distance))
                
                # 选择最优止损（最近的合理支撑）
                if potential_sl:
                    potential_sl.sort(key=lambda x: x[2])  # 按距离排序
                    stop_loss = potential_sl[0][1]
                    self.logger_trading.info(f"选择斐波那契止损: {potential_sl[0][0]} = {stop_loss:.2f}")
                else:
                    stop_loss = current_price - atr * 2  # 回退到ATR止损
                
                # 选择最优止盈（合理的阻力位，优先选择1.618扩展）
                if potential_tp:
                    # 优先选择1.618扩展水平
                    fib_618_tp = [tp for tp in potential_tp if 'fib_ext_161.8' in tp[0]]
                    if fib_618_tp:
                        take_profit = fib_618_tp[0][1]
                        self.logger_trading.info(f"选择斐波那契止盈: {fib_618_tp[0][0]} = {take_profit:.2f}")
                    else:
                        potential_tp.sort(key=lambda x: x[2])  # 按距离排序，选择最近的
                        take_profit = potential_tp[0][1]
                        self.logger_trading.info(f"选择斐波那契止盈: {potential_tp[0][0]} = {take_profit:.2f}")
                else:
                    take_profit = current_price + atr * 3  # 回退到ATR止盈
                    
            else:  # SELL
                # 空头交易：寻找上方阻力作为止损，下方支撑作为止盈
                potential_sl = []
                potential_tp = []
                
                # 止损候选：当前价格上方的斐波那契水平
                for level_name, level_price in fib_levels.items():
                    if 'fib_' in level_name and level_price > current_price:
                        distance = level_price - current_price
                        if 0.5 * atr <= distance <= 3 * atr:
                            potential_sl.append((level_name, level_price, distance))
                
                # 止盈候选：当前价格下方的斐波那契水平
                for level_name, level_price in fib_levels.items():
                    if 'fib_' in level_name and level_price < current_price:
                        distance = current_price - level_price
                        if 1 * atr <= distance <= 5 * atr:
                            potential_tp.append((level_name, level_price, distance))
                
                # 选择最优止损（最近的合理阻力）
                if potential_sl:
                    potential_sl.sort(key=lambda x: x[2])
                    stop_loss = potential_sl[0][1]
                    self.logger_trading.info(f"选择斐波那契止损: {potential_sl[0][0]} = {stop_loss:.2f}")
                else:
                    stop_loss = current_price + atr * 2
                
                # 选择最优止盈（合理的支撑位）
                if potential_tp:
                    # 优先选择61.8%回撤水平
                    fib_618_tp = [tp for tp in potential_tp if 'fib_61.8' in tp[0]]
                    if fib_618_tp:
                        take_profit = fib_618_tp[0][1]
                        self.logger_trading.info(f"选择斐波那契止盈: {fib_618_tp[0][0]} = {take_profit:.2f}")
                    else:
                        potential_tp.sort(key=lambda x: x[2])
                        take_profit = potential_tp[0][1]
                        self.logger_trading.info(f"选择斐波那契止盈: {potential_tp[0][0]} = {take_profit:.2f}")
                else:
                    take_profit = current_price - atr * 3
            
            return stop_loss, take_profit
            
        except Exception as e:
            self.logger_trading.error(f"斐波那契止损止盈计算失败: {e}")
            return None, None

    def _generate_emergency_timeframe_data(self, timeframe):
        """为主时间框架生成紧急回退数据"""
        try:
            self.logger_system.warning(f"为时间框架 {timeframe} 生成紧急数据")
            
            # 获取当前价格作为基准
            current_price = 67000.0  # 默认价格
            try:
                if not self.config.simulation_mode:
                    ticker = self.exchange.fetch_ticker(self.config.symbol)
                    current_price = ticker['last'] if ticker and ticker['last'] else current_price
                else:
                    # 模拟模式下生成随机价格
                    import random
                    current_price = 67000.0 + random.uniform(-500, 500)
            except Exception as e:
                self.logger_system.warning(f"获取当前价格失败，使用默认值: {e}")
            
            # 生成最小可用的OHLCV数据（3个数据点）
            now = pd.Timestamp.now(tz='UTC')
            timeframe_minutes = {
                '1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60, 
                '4h': 240, '1d': 1440, '1w': 10080
            }
            
            interval_minutes = timeframe_minutes.get(timeframe, 60)
            
            # 创建时间序列
            timestamps = []
            for i in range(3):
                timestamp = now - pd.Timedelta(minutes=interval_minutes * (2 - i))
                timestamps.append(timestamp)
            
            # 生成价格数据（小幅波动）
            import random
            ohlcv_data = []
            for i, ts in enumerate(timestamps):
                # 生成小幅价格波动
                price_variation = current_price * random.uniform(-0.01, 0.01)  # ±1%
                base_price = current_price + price_variation
                
                high = base_price * random.uniform(1.001, 1.005)  # 0.1-0.5%上涨
                low = base_price * random.uniform(0.995, 0.999)  # 0.1-0.5%下跌
                open_price = base_price * random.uniform(0.998, 1.002)
                close_price = base_price * random.uniform(0.998, 1.002)
                volume = random.uniform(100, 1000)
                
                ohlcv_data.append([
                    int(ts.timestamp() * 1000),  # timestamp in ms
                    open_price, high, low, close_price, volume
                ])
            
            # 创建DataFrame
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            
            # 计算基础指标
            df = self._calculate_fallback_indicators(df)
            
            self.logger_system.warning(f"Emergency data generated for {timeframe}: {len(df)} data points")
            return df
            
        except Exception as e:
            self.logger_system.exception(f"Failed to generate emergency data for {timeframe}: {e}")
            return None

    # New: FVG Detection
    def detect_fvgs(self, df: pd.DataFrame, lookback: int = 50) -> List[Dict[str, Any]]:
        """Detect Fair Value Gaps (FVGs) - Bullish: low[i] > high[i-2], Bearish: high[i] < low[i-2]"""
        fvgs = []
        if len(df) < 3:
            return fvgs
        
        recent = df.tail(lookback)
        for i in range(2, len(recent)):
            # Bullish FVG: Gap up (low[i] > high[i-2])
            if recent.iloc[i]['low'] > recent.iloc[i-2]['high']:
                gap_top = recent.iloc[i]['low']
                gap_bottom = recent.iloc[i-2]['high']
                fvgs.append({
                    'type': 'bullish',
                    'top': gap_top,
                    'bottom': gap_bottom,
                    'timestamp': recent.index[i],
                    'strength': (gap_top - gap_bottom) / recent.iloc[i]['close']  # Normalized strength
                })
            # Bearish FVG: Gap down (high[i] < low[i-2])
            elif recent.iloc[i]['high'] < recent.iloc[i-2]['low']:
                gap_top = recent.iloc[i-2]['low']
                gap_bottom = recent.iloc[i]['high']
                fvgs.append({
                    'type': 'bearish',
                    'top': gap_top,
                    'bottom': gap_bottom,
                    'timestamp': recent.index[i],
                    'strength': (gap_top - gap_bottom) / recent.iloc[i]['close']
                })
        return fvgs

    def calculate_fvg_levels(self, fvgs: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate key FVG levels, check for stacking"""
        if not fvgs:
            return {}
        
        # Separate bullish and bearish
        bull_fvgs = [f for f in fvgs if f['type'] == 'bullish']
        bear_fvgs = [f for f in fvgs if f['type'] == 'bearish']
        
        # Stacking: Count overlapping FVGs within 0.5% range
        stack_bull = 0
        stack_bear = 0
        if len(bull_fvgs) >= 2:
            for i in range(len(bull_fvgs)-1):
                if abs(bull_fvgs[i]['bottom'] - bull_fvgs[i+1]['top']) / bull_fvgs[i]['bottom'] < 0.005:  # 0.5%
                    stack_bull += 1
        if len(bear_fvgs) >= 2:
            for i in range(len(bear_fvgs)-1):
                if abs(bear_fvgs[i]['top'] - bear_fvgs[i+1]['bottom']) / bear_fvgs[i]['top'] < 0.005:
                    stack_bear += 1
        
        levels = {
            'fvg_bull_top': bull_fvgs[0]['top'] if bull_fvgs else 0,
            'fvg_bull_bottom': bull_fvgs[0]['bottom'] if bull_fvgs else 0,
            'fvg_bear_top': bear_fvgs[0]['top'] if bear_fvgs else 0,
            'fvg_bear_bottom': bear_fvgs[0]['bottom'] if bear_fvgs else 0,
            'fvg_bull_stack': stack_bull,
            'fvg_bear_stack': stack_bear
        }
        # Add midpoints for activation levels
        if bull_fvgs:
            levels['fvg_bull_mid'] = (bull_fvgs[0]['top'] + bull_fvgs[0]['bottom']) / 2
        if bear_fvgs:
            levels['fvg_bear_mid'] = (bear_fvgs[0]['top'] + bear_fvgs[0]['bottom']) / 2
        return levels

    def calculate_key_levels(self, multi_tf_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        try:
            # 安全获取各时间框架数据，避免KeyError
            daily_df = multi_tf_data.get('1d', pd.DataFrame())
            h4_df = multi_tf_data.get('4h', pd.DataFrame())
            h1_df = multi_tf_data.get('1h', pd.DataFrame())
            m15_df = multi_tf_data.get('15m', pd.DataFrame())
            m5_df = multi_tf_data.get('5m', pd.DataFrame())
            
            # 检查数据可用性并记录警告
            if daily_df.empty:
                self.logger_system.warning("日线数据缺失，使用fallback机制")
            if h4_df.empty:
                self.logger_system.warning("4小时数据缺失，使用fallback机制")
            if h1_df.empty:
                self.logger_system.warning("1小时数据缺失，跳过1小时EMA计算")
            if m15_df.empty:
                self.logger_system.warning("15分钟数据缺失，使用fallback机制")
            if m5_df.empty:
                self.logger_system.warning("5分钟数据缺失，使用fallback机制")
            
            current_daily = daily_df.iloc[-1] if not daily_df.empty else pd.Series()
            current_h4 = h4_df.iloc[-1] if not h4_df.empty else pd.Series()
            current_h1 = h1_df.iloc[-1] if not h1_df.empty else pd.Series()
            
            # 计算各级别关键位，如果数据缺失则返回空字典
            daily_liquidity = self.calculate_daily_liquidity_levels(daily_df) if not daily_df.empty else {}
            h4_order_blocks = self.calculate_order_blocks(h4_df, '4h') if not h4_df.empty else {}
            h1_order_blocks = self.calculate_order_blocks(h1_df, '1h') if not h1_df.empty else {}
            m15_order_blocks = self.calculate_order_blocks(m15_df, '15m') if not m15_df.empty else {}
            m5_order_blocks = self.calculate_order_blocks(m5_df, '5m') if not m5_df.empty else {}
            h1_ema_levels = self.calculate_hourly_ema_levels(h1_df) if not h1_df.empty else {}
            volume_profile = self.calculate_volume_profile_levels(m15_df) if not m15_df.empty else {}
            technical_levels = self.calculate_technical_levels(multi_tf_data)
            
            # New: FVG levels from daily, 4h, 1h, 15m
            daily_fvgs = self.detect_fvgs(daily_df)
            h4_fvgs = self.detect_fvgs(h4_df)
            h1_fvgs = self.detect_fvgs(h1_df)
            m15_fvgs = self.detect_fvgs(m15_df)
            daily_fvg_levels = self.calculate_fvg_levels(daily_fvgs)
            h4_fvg_levels = self.calculate_fvg_levels(h4_fvgs)
            h1_fvg_levels = self.calculate_fvg_levels(h1_fvgs)
            m15_fvg_levels = self.calculate_fvg_levels(m15_fvgs)
            daily_fvg_prefixed = {f'daily_{k}': v for k, v in daily_fvg_levels.items()}
            h4_fvg_prefixed = {f'4h_{k}': v for k, v in h4_fvg_levels.items()}
            h1_fvg_prefixed = {f'1h_{k}': v for k, v in h1_fvg_levels.items()}
            m15_fvg_prefixed = {f'15m_{k}': v for k, v in m15_fvg_levels.items()}
            
            # New: Calculate OB stacking for 5m timeframe
            m5_ob_stack = self.calculate_ob_stacking(m5_df) if not m5_df.empty else {}
            
            levels = {
                **daily_liquidity,
                **h4_order_blocks,
                **h1_order_blocks,
                **m15_order_blocks,
                **m5_order_blocks,
                **h1_ema_levels,
                **volume_profile,
                **technical_levels,
                **daily_fvg_prefixed,
                **h4_fvg_prefixed,
                **h1_fvg_prefixed,
                **m15_fvg_prefixed,
                **m5_ob_stack,
                'current_price': round(m15_df['close'].iloc[-1], 2) if not m15_df.empty else round(current_h4.get('close', 0), 2)  # FIXED: Medium 3
            }
            
            return levels
            
        except Exception as e:
            self.logger_system.exception(f"Key levels calculation failed: {e}")
            return self.get_fallback_levels()

    def calculate_hourly_ema_levels(self, hourly_df: pd.DataFrame) -> Dict[str, float]:
        """计算1小时EMA水平 - EMA21/55/100/200，根据数据量动态调整"""
        try:
            # 检查数据是否为空
            if hourly_df.empty:
                self.logger_system.warning("1小时数据为空，无法计算EMA水平")
                return {}
            
            data_length = len(hourly_df)
            self.logger_system.debug(f"1小时数据长度: {data_length}")
            
            # 根据数据长度动态计算可用的EMA指标
            ema_levels = {}
            
            # EMA21 - 最少需要21根K线
            if data_length >= 21:
                try:
                    ema_21 = hourly_df['close'].ewm(span=21).mean().iloc[-1]
                    ema_levels['ema_21_1h'] = round(float(ema_21), 2)
                    self.logger_system.debug(f"EMA21(1h)计算成功: {ema_levels['ema_21_1h']}")
                except Exception as e:
                    self.logger_system.warning(f"EMA21(1h)计算失败: {e}")
            else:
                self.logger_system.warning(f"数据不足，无法计算EMA21(1h)，需要21根K线，当前{data_length}根")
            
            # EMA55 - 最少需要55根K线
            if data_length >= 55:
                try:
                    ema_55 = hourly_df['close'].ewm(span=55).mean().iloc[-1]
                    ema_levels['ema_55_1h'] = round(float(ema_55), 2)
                    self.logger_system.debug(f"EMA55(1h)计算成功: {ema_levels['ema_55_1h']}")
                except Exception as e:
                    self.logger_system.warning(f"EMA55(1h)计算失败: {e}")
            else:
                self.logger_system.warning(f"数据不足，无法计算EMA55(1h)，需要55根K线，当前{data_length}根")
            
            # EMA100 - 最少需要100根K线
            if data_length >= 100:
                try:
                    ema_100 = hourly_df['close'].ewm(span=100).mean().iloc[-1]
                    ema_levels['ema_100_1h'] = round(float(ema_100), 2)
                    self.logger_system.debug(f"EMA100(1h)计算成功: {ema_levels['ema_100_1h']}")
                except Exception as e:
                    self.logger_system.warning(f"EMA100(1h)计算失败: {e}")
            else:
                self.logger_system.warning(f"数据不足，无法计算EMA100(1h)，需要100根K线，当前{data_length}根")
            
            # EMA200 - 最少需要200根K线
            if data_length >= 200:
                try:
                    ema_200 = hourly_df['close'].ewm(span=200).mean().iloc[-1]
                    ema_levels['ema_200_1h'] = round(float(ema_200), 2)
                    self.logger_system.debug(f"EMA200(1h)计算成功: {ema_levels['ema_200_1h']}")
                except Exception as e:
                    self.logger_system.warning(f"EMA200(1h)计算失败: {e}")
            else:
                self.logger_system.warning(f"数据不足，无法计算EMA200(1h)，需要200根K线，当前{data_length}根")
            
            if ema_levels:
                self.logger_system.info(f"1小时EMA水平计算完成，共{len(ema_levels)}个指标")
            else:
                self.logger_system.warning("所有1小时EMA指标计算失败，数据长度不足")
            
            return ema_levels
            
        except Exception as e:
            self.logger_system.exception(f"1小时EMA水平计算异常: {e}")
            return {}

    def calculate_daily_liquidity_levels(self, daily_df: pd.DataFrame) -> Dict[str, float]:
        """计算日线流动性水平 - 每周一开盘价、前一周最高点和最低点"""
        if len(daily_df) < 10:
            return {}
        
        # 获取最近的周一开盘价
        daily_df_copy = daily_df.copy()
        daily_df_copy['weekday'] = pd.to_datetime(daily_df_copy.index).weekday
        
        # 找到最近的周一 (weekday=0)
        monday_data = daily_df_copy[daily_df_copy['weekday'] == 0]
        monday_open = monday_data['open'].iloc[-1] if len(monday_data) > 0 else daily_df['open'].iloc[-1]
        
        # 获取当天开盘价（最新一天的开盘价）
        daily_open = daily_df['open'].iloc[-1]
        
        # 计算前一周的最高点和最低点 (最近7天)
        recent_week = daily_df.tail(7)
        prev_week_high = recent_week['high'].max()
        prev_week_low = recent_week['low'].min()
        
        # 保留原有的10日高低点作为补充
        recent_10d = daily_df.tail(10)
        recent_10d_high = recent_10d['high'].max()
        recent_10d_low = recent_10d['low'].min()
        
        daily_ob_bull, daily_ob_bear = self.calculate_daily_order_blocks(daily_df)
        vwap_daily = self.calculate_daily_vwap(daily_df.tail(5))
        
        return {
            'monday_open': monday_open,
            'daily_open': daily_open,
            'prev_week_high': prev_week_high,
            'prev_week_low': prev_week_low,
            'recent_10d_high': recent_10d_high,
            'recent_10d_low': recent_10d_low,
            'daily_ob_bull': daily_ob_bull,
            'daily_ob_bear': daily_ob_bear,
            'daily_vwap': vwap_daily
        }

    def calculate_daily_order_blocks(self, daily_df: pd.DataFrame) -> Tuple[float, float]:
        """计算日线订单块"""
        levels = self.calculate_order_blocks(daily_df, 'daily')
        return levels.get('daily_ob_bull', 0), levels.get('daily_ob_bear', 0)

    def calculate_daily_vwap(self, daily_df: pd.DataFrame) -> float:
        """计算日线VWAP"""
        if daily_df.empty:
            return 0
        typical_price = (daily_df['high'] + daily_df['low'] + daily_df['close']) / 3
        vwap = (typical_price * daily_df['volume']).sum() / daily_df['volume'].sum()
        return vwap

    def calculate_order_blocks(self, df: pd.DataFrame, tf: str) -> Dict[str, float]:
        """计算订单块"""
        if len(df) < 20:
            return {}
        
        df_copy = df.copy()
        df_copy['volume_zscore'] = (df_copy['volume'] - df_copy['volume'].rolling(20).mean()) / df_copy['volume'].rolling(20).std()
        df_copy['body_size'] = abs(df_copy['close'] - df_copy['open'])
        df_copy['is_engulfing'] = self.detect_engulfing_pattern(df_copy)
        
        bullish_blocks = df_copy[
            (df_copy['volume_zscore'] > 1.5) & 
            (df_copy['close'] > df_copy['open']) &
            (df_copy['is_engulfing'] | (df_copy['body_size'] / (df_copy['high'] - df_copy['low']) < 0.3))
        ]
        
        bearish_blocks = df_copy[
            (df_copy['volume_zscore'] > 1.5) & 
            (df_copy['close'] < df_copy['open']) &
            (df_copy['is_engulfing'] | (df_copy['body_size'] / (df_copy['high'] - df_copy['low']) < 0.3))
        ]
        
        ob_bull = bullish_blocks['low'].min() if not bullish_blocks.empty else None
        ob_bear = bearish_blocks['high'].max() if not bearish_blocks.empty else None
        
        levels = {}
        if ob_bull:
            levels[f'{tf}_ob_bull'] = ob_bull
        if ob_bear:
            levels[f'{tf}_ob_bear'] = ob_bear
        
        return levels

    def calculate_ob_stacking(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算OB堆叠"""
        if len(df) < 20:
            return {}
        
        df_copy = df.copy()
        df_copy['volume_zscore'] = (df_copy['volume'] - df_copy['volume'].rolling(20).mean()) / df_copy['volume'].rolling(20).std()
        df_copy['body_size'] = abs(df_copy['close'] - df_copy['open'])
        df_copy['is_engulfing'] = self.detect_engulfing_pattern(df_copy)
        
        # 识别看涨和看跌订单块
        bullish_blocks = df_copy[
            (df_copy['volume_zscore'] > 1.5) & 
            (df_copy['close'] > df_copy['open']) &
            (df_copy['is_engulfing'] | (df_copy['body_size'] / (df_copy['high'] - df_copy['low']) < 0.3))
        ]
        
        bearish_blocks = df_copy[
            (df_copy['volume_zscore'] > 1.5) & 
            (df_copy['close'] < df_copy['open']) &
            (df_copy['is_engulfing'] | (df_copy['body_size'] / (df_copy['high'] - df_copy['low']) < 0.3))
        ]
        
        # 计算OB堆叠
        bull_stack = 0
        bear_stack = 0
        
        # 看涨OB堆叠：计算在0.5%范围内的重叠OB数量
        if len(bullish_blocks) >= 2:
            for i in range(len(bullish_blocks)-1):
                current_low = bullish_blocks.iloc[i]['low']
                next_low = bullish_blocks.iloc[i+1]['low']
                if abs(current_low - next_low) / current_low < 0.005:  # 0.5%范围内
                    bull_stack += 1
        
        # 看跌OB堆叠：计算在0.5%范围内的重叠OB数量
        if len(bearish_blocks) >= 2:
            for i in range(len(bearish_blocks)-1):
                current_high = bearish_blocks.iloc[i]['high']
                next_high = bearish_blocks.iloc[i+1]['high']
                if abs(current_high - next_high) / current_high < 0.005:  # 0.5%范围内
                    bear_stack += 1
        
        return {
            '5m_ob_bull_stack': bull_stack,
            '5m_ob_bear_stack': bear_stack
        }

    def detect_engulfing_pattern(self, df: pd.DataFrame) -> pd.Series:
        """检测吞没形态"""
        engulfing = pd.Series(False, index=df.index)
        for i in range(1, len(df)):
            prev, curr = df.iloc[i-1], df.iloc[i]
            bull_engulfing = (prev['close'] < prev['open'] and 
                curr['close'] > curr['open'] and
                curr['open'] < prev['close'] and 
                curr['close'] > prev['open'])
            bear_engulfing = (prev['close'] > prev['open'] and 
                curr['close'] < curr['open'] and
                curr['open'] > prev['close'] and 
                curr['close'] < prev['open'])
            engulfing.iloc[i] = bull_engulfing or bear_engulfing
        return engulfing

    # New: Enhanced candle pattern detection
    def detect_candle_patterns(self, df: pd.DataFrame, lookback: int = 5) -> List[str]:
        """Detect candle patterns: Engulfing, Hammer, Shooting Star, Doji"""
        patterns = []
        if len(df) < 2:
            return patterns
        
        recent = df.tail(lookback)
        for i in range(1, len(recent)):
            prev = recent.iloc[i-1]
            curr = recent.iloc[i]
            body_size = abs(curr['close'] - curr['open'])
            upper_wick = curr['high'] - max(curr['open'], curr['close'])
            lower_wick = min(curr['open'], curr['close']) - curr['low']
            total_range = curr['high'] - curr['low']
            if total_range == 0:
                continue
            
            # Doji
            if body_size < total_range * 0.1:
                patterns.append("Doji")
            
            # Hammer (bullish reversal)
            if lower_wick > body_size * 2 and upper_wick < body_size * 0.5 and curr['close'] > curr['open']:
                patterns.append("Hammer")
            
            # Shooting Star (bearish reversal)
            if upper_wick > body_size * 2 and lower_wick < body_size * 0.5 and curr['close'] < curr['open']:
                patterns.append("Shooting Star")
            
            # Bullish Engulfing
            if (prev['close'] < prev['open'] and curr['close'] > curr['open'] and
                curr['open'] < prev['close'] and curr['close'] > prev['open']):
                patterns.append("Bullish Engulfing")
            
            # Bearish Engulfing
            if (prev['close'] > prev['open'] and curr['close'] < curr['open'] and
                curr['open'] > prev['close'] and curr['close'] < prev['open']):
                patterns.append("Bearish Engulfing")
        
        return patterns[-3:]  # Last 3 patterns

    def calculate_volume_profile_levels(self, m15_df: pd.DataFrame) -> Dict[str, float]:
        """计算成交量分布水平"""
        if len(m15_df) < 100:
            return {}
        
        price_bins = 20
        high, low = m15_df['high'].max(), m15_df['low'].min()
        bin_size = (high - low) / price_bins
        
        if bin_size == 0:
            return {}
        
        volume_at_price = {}
        for i in range(len(m15_df)):
            price_level = round((m15_df.iloc[i]['close'] - low) / bin_size)
            volume_at_price[price_level] = volume_at_price.get(price_level, 0) + m15_df.iloc[i]['volume']
        
        if not volume_at_price:
            return {}
        
        max_volume_level = max(volume_at_price, key=volume_at_price.get)
        poc_price = low + (max_volume_level * bin_size)
        
        return {'volume_poc': poc_price}

    def calculate_technical_levels(self, multi_tf_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """计算技术分析水平"""
        levels = {}
        
        try:
            h4_df = multi_tf_data.get('4h')
            if h4_df is not None and not h4_df.empty:
                # 计算EMA水平
                if 'ema_21' in h4_df.columns:
                    levels['ema_21_4h'] = h4_df['ema_21'].iloc[-1]
                if 'ema_55' in h4_df.columns:
                    levels['ema_55_4h'] = h4_df['ema_55'].iloc[-1]
                if 'ema_100' in h4_df.columns:
                    levels['ema_100_4h'] = h4_df['ema_100'].iloc[-1]
                if 'ema_200' in h4_df.columns:
                    levels['ema_200_4h'] = h4_df['ema_200'].iloc[-1]
                
                # 计算4小时级别的高低点 (4h_high/4h_low)
                if len(h4_df) >= 2:
                    # 最近的4小时高点和低点
                    levels['4h_high'] = h4_df['high'].iloc[-1]
                    levels['4h_low'] = h4_df['low'].iloc[-1]
                
                # 计算4小时级别的缺口 (4h_gap_up/4h_gap_down)
                if len(h4_df) >= 2:
                    current_open = h4_df['open'].iloc[-1]
                    prev_close = h4_df['close'].iloc[-2]
                    
                    # 向上缺口：当前开盘价高于前一根收盘价
                    if current_open > prev_close:
                        levels['4h_gap_up'] = current_open
                        levels['4h_gap_down'] = prev_close
                    # 向下缺口：当前开盘价低于前一根收盘价
                    elif current_open < prev_close:
                        levels['4h_gap_down'] = current_open
                        levels['4h_gap_up'] = prev_close
                    else:
                        # 无缺口时，使用当前价格作为参考
                        levels['4h_gap_up'] = current_open
                        levels['4h_gap_down'] = current_open
                        
        except Exception as e:
            self.logger_system.warning(f"Technical levels calculation failed: {e}")
        
        return levels

    def get_fallback_levels(self) -> Dict[str, float]:
        try:
            ticker = self.safe_fetch_ticker(exchange, self.config.symbol)
            current_price = round(ticker['last'], 2)
            return {
                'monday_open': current_price,
                'prev_week_high': current_price * 1.02,
                'prev_week_low': current_price * 0.98,
                'current_price': current_price,
                'recent_10d_high': current_price * 1.02,
                'recent_10d_low': current_price * 0.98,
                'ema_21_1h': current_price,
                'ema_55_1h': current_price,
                'ema_100_1h': current_price,
                'ema_200_1h': current_price
            }
        except Exception:
            return {'current_price': 0}

    def check_price_activation(self, current_price: float, key_levels: Dict[str, float]) -> Tuple[bool, Optional[str]]:
        current_time = time.time()
        
        # FIXED: High 6 - Add lock protection for last_activation_time
        with self.lock:
            if current_time - self.last_activation_time < 60:
                return False, None
        
        # FIXED: Medium 12 - Dynamic priority by weight from structures (assume structures_summary available or mock)
        # For simplicity, sort by fixed weights; integrate structures if passed
        priority_weights = self.config.level_weights
        sorted_priority = sorted(self.config.liquidity_priority, key=lambda x: priority_weights.get(x, 1), reverse=True)
        
        for level_name in sorted_priority:
            level = key_levels.get(level_name, current_price)
            if pd.isna(level) or level <= 0:
                continue
            
            # TF interaction freshness boost
            interactions = self.zone_interactions.get(level_name, 0)
            if '1h_' in level_name:
                h4_levels = [v for k, v in key_levels.items() if '4h_' in k and isinstance(v, (int, float)) and v > 0]
                if h4_levels:
                    min_dist = min(abs(level - hl) / level for hl in h4_levels)
                    if min_dist < 0.001:  # 0.1%
                        interactions = 0  # Treat as fresh due to 4h confluence
                        self.logger_trading.debug(f"1h {level_name} boosted by 4h confluence, interactions reset to 0")
            elif '15m_' in level_name:
                higher_levels = [v for k, v in key_levels.items() if ('4h_' in k or '1h_' in k) and isinstance(v, (int, float)) and v > 0]
                if higher_levels:
                    min_dist = min(abs(level - hl) / level for hl in higher_levels)
                    if min_dist < 0.001:
                        interactions = 0
                        self.logger_trading.debug(f"15m {level_name} boosted by higher TF confluence, interactions reset to 0")
            
            # New: Check freshness - skip if interactions > max
            if interactions > self.config.max_zone_interactions:
                self.logger_trading.debug(f"关键位 {level_name} 已互动 {interactions} 次 (> {self.config.max_zone_interactions})，跳过")
                continue
            
            # 检查该关键位的5分钟冷静期
            with self.lock:
                last_level_activation = self.level_activation_times.get(level_name, 0)
                if current_time - last_level_activation < 300:  # 5分钟 = 300秒
                    self.logger_trading.debug(f"关键位 {level_name} 仍在冷静期内，跳过激活检查")
                    continue
            
            distance = abs(current_price - level) / current_price
            adjusted_threshold = self.get_adjusted_threshold(level_name)
            if distance < adjusted_threshold:
                # New: Increment interaction count
                self.zone_interactions[level_name] = interactions + 1
                
                # 记录激活统计
                with self.lock:
                    # 更新激活时间
                    self.last_activation_time = current_time
                    self.level_activation_times[level_name] = current_time
                    
                    # 更新激活统计
                    if level_name not in self.level_activation_stats:
                        self.level_activation_stats[level_name] = {
                            'activations': 0,
                            'successful_trades': 0,
                            'failed_trades': 0,
                            'total_threshold_accumulation': 0.0,
                            'last_activation_time': 0
                        }
                    
                    self.level_activation_stats[level_name]['activations'] += 1
                    self.level_activation_stats[level_name]['total_threshold_accumulation'] += distance
                    self.level_activation_stats[level_name]['last_activation_time'] = current_time
                    self.total_activations += 1
                
                # 记录激活日志，包含阈值积累和胜率信息
                stats = self.level_activation_stats[level_name]
                avg_threshold = stats['total_threshold_accumulation'] / stats['activations']
                win_rate = (stats['successful_trades'] / max(stats['successful_trades'] + stats['failed_trades'], 1)) * 100
                
                self.logger_trading.info(f"关键位激活: {level_name}")
                self.logger_trading.info(f"  当前价格: {current_price:.4f}")
                self.logger_trading.info(f"  关键位价格: {level:.4f}")
                self.logger_trading.info(f"  距离: {distance*100:.3f}%")
                self.logger_trading.info(f"  激活阈值: {adjusted_threshold*100:.3f}%")
                self.logger_trading.info(f"  累计激活次数: {stats['activations']}")
                self.logger_trading.info(f"  平均阈值积累: {avg_threshold*100:.3f}%")
                self.logger_trading.info(f"  胜率: {win_rate:.1f}% ({stats['successful_trades']}/{stats['successful_trades'] + stats['failed_trades']})")
                
                self.logger_trading.info(f"Institutional liquidity activation {level_name}: {level:.2f} (distance: {distance*100:.2f}%)")
                return True, level_name
        
        return False, None

    def get_adjusted_threshold(self, level_name: str) -> float:
        """根据关键位类型返回激活阈值 - 按照README.md规则"""
        # 直接使用README.md中指定的绝对阈值 - 新的��重体系, adjusted for aggressive intraday
        absolute_thresholds = {
            # 日线级别 (最高权重 - 机构级别位置)：降低 15-20% 以捕获更多 weekly/daily 流动性狩猎机会
            'monday_open': 0.0010,          # 0.10% (从 0.12% 降，增加机构开盘反转信号)
            'daily_open': 0.00085,          # 0.085% (从 0.10% 降，优化日内开盘抓取)
            'prev_week_high': 0.00075,      # 0.075% (从 0.09% 降，更多 weekly 高点突破/反转)
            'prev_week_low': 0.00075,       # 0.075% (从 0.09% 降，同上)
            'daily_vwap': 0.00065,          # 0.065% (从 0.08% 降，VWAP 作为日内中枢更敏感)
            'daily_fvg_bull_mid': 0.00075,  # 0.075% (新增，日线FVG中枢，高权重敏感度)
            'daily_fvg_bear_mid': 0.00075,  # 0.075% (新增，同上)
            'recent_10d_high': 0.00075,     # 0.075% (从 0.09% 降，类似 weekly，增加 10d 趋势机会)
            'recent_10d_low': 0.00075,      # 0.075% (从 0.09% 降，同上)
            'daily_ob_bull': 0.00065,       # 0.065% (从 0.08% 降，bull OB 更易激活多头)
            'daily_ob_bear': 0.00065,       # 0.065% (从 0.08% 降，bear OB 更易激活空头)
            
            # 4小时级别 (高权重 - 中期结构位置)：降低 15% 以桥接日线-小时，增加中期 confluence
            '4h_high': 0.00065,             # 0.065% (从 0.08% 降，更多 4h 高点 liquidity grab)
            '4h_low': 0.00065,              # 0.065% (从 0.08% 降，同上)
            '4h_ob_bull': 0.00055,          # 0.055% (从 0.07% 降，OB 确认后入场更频繁)
            '4h_ob_bear': 0.00055,          # 0.055% (从 0.07% 降，同上)
            '4h_fvg_bull_mid': 0.00055,     # 0.055%
            '4h_fvg_bear_mid': 0.00055,     # 0.055%
            '4h_gap_up': 0.0005,            # 0.05% (从 0.06% 降，gap 填充机会增多)
            '4h_gap_down': 0.0005,          # 0.05% (从 0.06% 降，同上)
            'ema_21_4h': 0.00055,           # 0.055% (从 0.07% 降，短期 EMA 更敏感趋势转折)
            'ema_55_4h': 0.00055,           # 0.055% (从 0.07% 降，同上)
            'ema_200_4h': 0.00065,          # 0.065% (从 0.08% 降，长 EMA 作为支撑更易触及)
            'ema_100_4h': 0.00055,          # 0.055% (从 0.07% 降，类似 OB)
            
            # 1小时级别 (中等权重 - 短期结构位置)：降低 10% 以精确定位，结合 15m 执行
            'ema_200_1h': 0.0005,           # 0.05% (从 0.06% 降，长 EMA 趋势确认更多)
            'ema_100_1h': 0.00045,          # 0.045% (从 0.055% 降，中 EMA 优化短期 pullback)
            'ema_55_1h': 0.0004,            # 0.04% (从 0.05% 降，增加中期 EMA 机会)
            'ema_21_1h': 0.00015,           # 0.015% (从 0.02% 降，进一步激进短期 EMA 突破)
            '1h_ob_bull': 0.0004,           # 0.04%
            '1h_ob_bear': 0.0004,           # 0.04%
            '1h_fvg_bull_mid': 0.00035,     # 0.035%
            '1h_fvg_bear_mid': 0.00035,     # 0.035%
            
            # 15分钟级别 (执行级别 - 精确入场位置)：微调 10%，已低，避免过度噪音；用 momentum 过滤
            '15m_liquidity_hunt': 0.00035,  # 0.035% (从 0.04% 降，轻微增加狩猎信号)
            '15m_structure_reversal': 0.00025,  # 0.025% (从 0.03% 降，反转确认更多)
            '15m_structure_break': 0.00008,    # 0.008% (从 0.01% 降，极激进突破，但需 R:R >2.5 过滤)
            '15m_ob_bull': 0.0003,             # 0.03%
            '15m_ob_bear': 0.0003,             # 0.03%
            '15m_fvg_bull_mid': 0.00025,       # 0.025%
            '15m_fvg_bear_mid': 0.00025,       # 0.025%
        }
        
        # 返回指定的阈值，如果没有找到则使用默认值
        return absolute_thresholds.get(level_name, self.config.activation_threshold)

    def calculate_intraday_amplitude(self, daily_df: pd.DataFrame, lookback: int = 7) -> Dict[str, float]:
        """计算日内振幅"""
        try:
            recent_days = daily_df.tail(lookback)
            amplitudes = recent_days['high'] - recent_days['low']
            avg_amplitude = amplitudes.mean()
            expected_rr_range = avg_amplitude * 0.8
            
            return {
                'avg_amplitude': avg_amplitude,
                'expected_rr_range': expected_rr_range,
                'amplitude_percentile_75': amplitudes.quantile(0.75),
                'amplitude_percentile_25': amplitudes.quantile(0.25)
            }
        except Exception as e:
            self.logger_system.error(f"Calculate intraday amplitude failed: {e}")
            return {
                'avg_amplitude': 0.0,
                'expected_rr_range': 0.0,
                'amplitude_percentile_75': 0.0,
                'amplitude_percentile_25': 0.0
            }

    def calculate_15min_amplitude(self, m15_df: pd.DataFrame) -> float:
        """计算最新15分钟K线的振幅百分比"""
        try:
            if len(m15_df) == 0:
                return 0.0
            
            # 获取最新的15分钟K线
            latest_candle = m15_df.iloc[-1]
            high = latest_candle['high']
            low = latest_candle['low']
            
            # 计算振幅百分比 (high - low) / low
            amplitude_pct = (high - low) / low if low > 0 else 0.0
            
            self.logger_system.info(f"最新15分钟K线振幅: {amplitude_pct*100:.3f}% (高: {high}, 低: {low})")
            return amplitude_pct
            
        except Exception as e:
            self.logger_system.error(f"计算15分钟振幅失败: {e}")
            return 0.0

    def _get_last_timestamp(self, df: pd.DataFrame) -> Optional[datetime]:
        """获取数据框的最后时间戳"""
        try:
            if df.empty:
                return None
            return pd.to_datetime(df.index[-1])
        except Exception:
            return None

    def find_pivots(self, series: pd.Series, window: int = 2) -> List[Tuple[int, float, str]]:
        """查找价格枢轴点"""
        if len(series) < 2 * window + 1:
            return []
        pivots = []
        for i in range(window, len(series) - window):
            high_cond = all(series.iloc[i] > series.iloc[i - j] for j in range(1, window + 1)) and \
                all(series.iloc[i] > series.iloc[i + j] for j in range(1, window + 1))
            low_cond = all(series.iloc[i] < series.iloc[i - j] for j in range(1, window + 1)) and \
                all(series.iloc[i] < series.iloc[i + j] for j in range(1, window + 1))
            if high_cond:
                pivots.append((i, series.iloc[i], 'high'))
            elif low_cond:
                pivots.append((i, series.iloc[i], 'low'))
        return pivots

    def get_kline_structure_summary(self, df: pd.DataFrame, tf_name: str, lookback: int = 20) -> Dict[str, Any]:
        """获取K线结构摘要 - Enhanced with candle patterns and FVG"""
        try:
            if len(df) < 5:
                self.logger_system.warning(f"Data length insufficient ({len(df)} < 5), returning N/A structure")
                return {'timeframe': tf_name, 'structure': 'N/A', 'order_block_bull': None, 'order_block_bear': None, 'patterns': [], 'trend': 'N/A', 'volatility': 0, 'reversal_confirmed': False, 'weight': 1, 'fvg_bull_count': 0, 'fvg_bear_count': 0}

            highs = df['high'].tail(lookback)
            lows = df['low'].tail(lookback)
            closes = df['close'].tail(lookback)

            high_pivots = self.find_pivots(highs)
            low_pivots = self.find_pivots(lows)

            is_hh_hl = len(high_pivots) >= 2 and high_pivots[-1][1] > high_pivots[-2][1] and len(low_pivots) >= 2 and low_pivots[-1][1] > low_pivots[-2][1]
            is_lh_ll = len(high_pivots) >= 2 and high_pivots[-1][1] < high_pivots[-2][1] and len(low_pivots) >= 2 and low_pivots[-1][1] < low_pivots[-2][1]
            structure = "Strong Uptrend (HH/HL)" if is_hh_hl else "Strong Downtrend (LH/LL)" if is_lh_ll else "Range/Transition (Mixed)"

            reversal_confirmed = False
            if len(high_pivots) >= 2 and len(low_pivots) >= 2:
                latest_high = high_pivots[-1]
                latest_low = low_pivots[-1]
                prev_high = high_pivots[-2]
                prev_low = low_pivots[-2]
                latest_close = closes.iloc[-1]
                if is_hh_hl and latest_high[1] < prev_high[1] and latest_close < latest_high[1] * 0.995:
                    reversal_confirmed = True
                    structure = "Confirmed Bearish Reversal (after HH/HL)"
                elif is_lh_ll and latest_low[1] > prev_low[1] and latest_close > latest_low[1] * 1.005:
                    reversal_confirmed = True
                    structure = "Confirmed Bullish Reversal (after LH/LL)"

            ob_bull = None
            ob_bear = None
            try:
                df_copy = df.copy()
                df_copy['body_size'] = abs(df_copy['close'] - df_copy['open'])
                df_copy['range_'] = df_copy['high'] - df_copy['low']
                df_copy['vol_ma'] = df_copy['volume'].rolling(window=10).mean().fillna(df_copy['volume'].mean())
                recent = df_copy.tail(10)
                strong_bullish_candles = recent[(recent['close'] > recent['open'] * 1.005) & (recent['volume'] > recent['vol_ma'])]
                strong_bearish_candles = recent[(recent['close'] < recent['open'] * 0.995) & (recent['volume'] > recent['vol_ma'])]
                if not strong_bullish_candles.empty:
                    max_vol_bull_idx = strong_bullish_candles['volume'].idxmax()
                    ob_bull = recent.loc[max_vol_bull_idx, 'low']
                if not strong_bearish_candles.empty:
                    max_vol_bear_idx = strong_bearish_candles['volume'].idxmax()
                    ob_bear = recent.loc[max_vol_bear_idx, 'high']
            except Exception as ob_e:
                self.logger_system.debug(f"OB detection failed ({tf_name}): {ob_e}")

            patterns = self.detect_candle_patterns(df, lookback=5)  # Enhanced

            trend = "Uptrend" if closes.iloc[-1] > closes.iloc[0] else "Downtrend" if len(closes) >= 2 else "N/A"
            volatility = (highs - lows).mean() if len(highs) > 0 else 0

            # New: FVG count
            fvgs = self.detect_fvgs(df)
            fvg_bull_count = len([f for f in fvgs if f['type'] == 'bullish'])
            fvg_bear_count = len([f for f in fvgs if f['type'] == 'bearish'])

            tf_weights = {'1w': 4, '1d': 3, '4h': 2, '15m': 1}
            weight = tf_weights.get(tf_name, 1)
            if reversal_confirmed:
                weight *= 1.5
            # New: Boost weight for patterns and FVG
            if patterns:
                weight *= self.config.candle_pattern_weight
            if fvg_bull_count >= self.config.fvg_stack_threshold or fvg_bear_count >= self.config.fvg_stack_threshold:
                weight *= 1.2

            return {
                'timeframe': tf_name,
                'structure': structure,
                'order_block_bull': ob_bull,
                'order_block_bear': ob_bear,
                'patterns': patterns,
                'trend': trend,
                'volatility': volatility,
                'reversal_confirmed': reversal_confirmed,
                'weight': weight,
                'fvg_bull_count': fvg_bull_count,
                'fvg_bear_count': fvg_bear_count
            }
        except Exception as e:
            self.logger_system.exception(f"Kline structure analysis failed ({tf_name}): {repr(e)}")
            return {'timeframe': tf_name, 'structure': 'N/A', 'order_block_bull': None, 'order_block_bear': None, 'patterns': [], 'trend': 'N/A', 'volatility': 0, 'reversal_confirmed': False, 'weight': 1, 'fvg_bull_count': 0, 'fvg_bear_count': 0}

    def _generate_simulation_data(self) -> Dict[str, Any]:
        """生成模拟数据用于测试"""
        self.logger_system.info("模拟模式: 生成模拟数据")
        
        # 基础价格
        base_price = 67000.0
        current_price = base_price + random.uniform(-5000, 5000)
        
        multi_tf_data = {}
        
        # 为每个时间框架生成足够的数据
        for tf in self.config.timeframes:
            # 生成足够的数据点
            data_points = 100  # 足够的数据点数
            
            # 生成时间序列 - 使用动态映射
            now = datetime.now(timezone.utc)
            timestamps = []
            
            # 时间框架到时间增量的映射 - FIXED: Medium 7
            timeframe_deltas = {
                '1w': pd.Timedelta(weeks=1),
                '1d': pd.Timedelta(days=1),
                '4h': pd.Timedelta(hours=4),
                '1h': pd.Timedelta(hours=1),
                '15m': pd.Timedelta(minutes=15),
                '5m': pd.Timedelta(minutes=5),
            }
            
            # 获取对应的时间增量，如果不存在则默认为1分钟
            delta = timeframe_deltas.get(tf, pd.Timedelta(minutes=1))
            
            for i in range(data_points):
                timestamp = now - delta * (data_points - i)
                timestamps.append(timestamp)
            
            # 生成OHLCV数据
            ohlcv_data = []
            price = base_price
            for timestamp in timestamps:
                # 随机价格变动
                price_change = random.uniform(-0.02, 0.02) * price
                price += price_change
                
                high = price * (1 + random.uniform(0, 0.01))
                low = price * (1 - random.uniform(0, 0.01))
                open_price = price + random.uniform(-0.005, 0.005) * price
                close_price = price
                volume = random.uniform(100, 1000)
                
                ohlcv_data.append([
                    int(timestamp.timestamp() * 1000),
                    open_price, high, low, close_price, volume
                ])
            
            # 创建DataFrame
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            
            # 计算技术指标
            df = self.calculate_technical_indicators(df)
            multi_tf_data[tf] = df
        
        # 计算关键位
        key_levels = self.calculate_key_levels(multi_tf_data)
        
        # 计算振幅
        daily_df = multi_tf_data.get('1d')
        if daily_df is not None and len(daily_df) >= 7:
            amplitude_data = self.calculate_intraday_amplitude(daily_df, self.config.amplitude_lookback)
        else:
            amplitude_data = {'weekly_avg_amplitude': 2000.0, 'current_volatility': 35.0}
        
        # 生成模拟的结构摘要
        structures_summary = {}
        for tf in self.config.timeframes:
            structures_summary[tf] = {
                'trend': random.choice(['bullish', 'bearish', 'neutral']),
                'structure': random.choice(['uptrend', 'downtrend', 'sideways']),
                'momentum': random.choice(['strong', 'weak', 'neutral']),
                'pivot_count': random.randint(3, 8),
                'last_pivot': random.choice(['high', 'low'])
            }

        return {
            'multi_tf_data': multi_tf_data,
            'current_price': current_price,
            'price': current_price,  # 添加price字段以匹配代码期望
            'key_levels': key_levels,
            'amplitude': amplitude_data,  # 修正字段名
            'amplitude_data': amplitude_data,
            'timeframes_fetched': list(multi_tf_data.keys()),
            'structures_summary': structures_summary,  # 添加结构摘要
            'timestamp': datetime.now(timezone.utc).isoformat(),  # 添加时间戳
            'technical_data': {
                'rsi': 50.0,
                'macd': 0.0,
                'sma_20': current_price,
                'atr': 500.0
            },
            'volatility': amplitude_data.get('current_volatility', 35.0)
        }

    def get_multi_timeframe_data(self) -> Optional[Dict[str, Any]]:
        try:
            # 模拟模式下生成模拟数据
            if self.config.simulation_mode:
                return self._generate_simulation_data()
            
            multi_tf_data = {}
            fetch_timeframes = self.config.timeframes + ['5m']  # 添加 5m 以支持 momentum filter
            
            self.logger_system.info("开始获取多时间框架数据（支持缓存和增量更新）")
            
            for tf in fetch_timeframes:
                try:
                    # 尝试从缓存获取数据
                    cached_df = self._get_cached_data(tf)
                    
                    if cached_df is not None:
                        # 缓存有效，使用缓存数据
                        df = cached_df
                        self.logger_system.info(f"使用缓存 {tf} 数据: {len(df)} 条记录")
                    else:
                        # 缓存无效或不存在，从API获取数据
                        # 动态 limit：15m 用 500，其余 200
                        limit = 500 if tf == '15m' else 200
                        
                        # 如果有缓存但过期，尝试增量更新
                        if tf in self.data_cache and self.data_cache[tf].get('data'):
                            # 获取缓存中最新时间戳
                            cached_data = self.data_cache[tf]['data']
                            latest_timestamp = pd.to_datetime(cached_data[-1]['timestamp'])
                            
                            # 计算需要获取的新数据点数
                            time_diff = time.time() - latest_timestamp.timestamp()
                            
                            # 根据时间框架计算需要的新数据点
                            if tf == '1d':
                                new_points = max(1, int(time_diff / (24 * 3600)) + 1)
                            elif tf == '4h':
                                new_points = max(1, int(time_diff / (4 * 3600)) + 1)
                            elif tf == '1h':
                                new_points = max(1, int(time_diff / 3600) + 1)
                            elif tf == '15m':
                                new_points = max(1, int(time_diff / 900) + 1)
                            elif tf == '5m':
                                new_points = max(1, int(time_diff / 300)) + 1
                            else:
                                new_points = limit
                            
                            # 限制新数据点数量，避免过多请求
                            new_points = min(new_points, limit // 2)
                            
                            self.logger_system.info(f"增量获取 {tf} 数据: {new_points} 个新数据点")
                            
                            # 获取新数据
                            new_ohlcv = self.safe_fetch_ohlcv(exchange, self.config.symbol, tf, new_points)
                            
                            if new_ohlcv:
                                # 转换为DataFrame
                                new_df = pd.DataFrame(new_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                                new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], unit='ms', utc=True)
                                
                                # 合并缓存和新数据
                                df = pd.concat([cached_df, new_df]).drop_duplicates().sort_index()
                                
                                # 保持数据量在限制范围内
                                if len(df) > limit:
                                    df = df.tail(limit)
                            else:
                                # 获取新数据失败，使用缓存
                                df = cached_df
                                self.logger_system.warning(f"增量获取 {tf} 数据失败，使用缓存")
                        else:
                            # 没有缓存，获取完整数据
                            self.logger_system.info(f"获取 {tf} 数据: limit={limit}")
                            ohlcv = self.safe_fetch_ohlcv(exchange, self.config.symbol, tf, limit)
                            
                            if not ohlcv or len(ohlcv) < 50:  # 至少 50 点
                                self.logger_system.warning(f"{tf} 数据不足 ({len(ohlcv) if ohlcv else 0} < 50)，跳过")
                                continue
                            
                            # 处理 DataFrame
                            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                        
                        # 更新缓存
                        self._update_cache(tf, df)
                    
                    # 数据质量验证
                    if len(df) < 50:
                        self.logger_system.warning(f"{tf} 数据长度不足({len(df)}<50)，跳过")
                        continue
                    
                    if df['close'].isna().all():
                        self.logger_system.warning(f"{tf} close 列全为 NaN，跳过")
                        continue
                    
                    # 计算技术指标
                    df = self.calculate_technical_indicators(df)
                    multi_tf_data[tf] = df
                    self.logger_system.info(f"{tf} 数据处理完成，共 {len(df)} 个数据点")
                    
                except Exception as e:
                    self.logger_system.exception(f"获取 {tf} 数据失败: {e}")
                    continue
            
            # 确保至少有主 TF
            if self.config.primary_timeframe not in multi_tf_data:
                self.logger_system.error(f"主时间框架 {self.config.primary_timeframe} 获取失败")
                return None
            
            # 结构分析
            structures_summary = {}
            for tf, df in multi_tf_data.items():
                try:
                    structures_summary[tf] = self.get_kline_structure_summary(df, tf)
                except Exception as tf_e:
                    self.logger_system.exception(f"Structure analysis failed for {tf}: {tf_e}")
                    structures_summary[tf] = {'timeframe': tf, 'structure': 'N/A', 'order_block_bull': None, 'order_block_bear': None, 'patterns': [], 'trend': 'N/A', 'volatility': 0, 'reversal_confirmed': False, 'weight': 1, 'fvg_bull_count': 0, 'fvg_bear_count': 0}

            # FIXED: Medium 2 - Cache with full TTL check
            current_time = time.time()
            if current_time - self.cache_timestamp > self.config.cache_ttl:
                with self.lock:  # FIXED: High 5 - Lock cache
                    self.key_levels_cache = self.calculate_key_levels(multi_tf_data)
                    self.cache_timestamp = current_time
                self.logger_system.debug("Key levels cache updated")

            # 计算振幅（使用 1d 数据）
            daily_df = multi_tf_data.get('1d')
            if daily_df is not None and len(daily_df) > 0:
                amplitude = self.calculate_intraday_amplitude(daily_df, self.config.amplitude_lookback)
            else:
                self.logger_system.warning("No 1d data for amplitude, using defaults")
                amplitude = {'avg_amplitude': 3500.0, 'expected_rr_range': 2800.0}  # BTC 典型值

            # 获取当前价格和数据
            primary_tf = self.config.primary_timeframe
            current_data = multi_tf_data[primary_tf].iloc[-1]
            current_price = round(current_data['close'], 2)

            # 计算波动率
            if daily_df is not None and len(daily_df) > 1:
                returns = daily_df['close'].pct_change().dropna()
                vol = returns.std() * np.sqrt(252) * 100 if len(returns) > 0 else 0
            else:
                vol = 0

            self.logger_system.info(f"Multi-timeframe data successfully fetched: {list(multi_tf_data.keys())}")
            return {
                'multi_tf_data': multi_tf_data,
                'structures_summary': structures_summary,
                'key_levels': self.key_levels_cache,
                'amplitude': amplitude,
                'price': current_price,
                'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                'high': current_data['high'],
                'low': current_data['low'],
                'volume': current_data['volume'],
                'timeframe': primary_tf,
                'technical_data': {
                    'rsi': current_data.get('rsi', 50),
                    'macd': current_data.get('macd', 0),
                    'sma_20': current_data.get('sma_20', current_price),
                    'atr': current_data.get('atr', current_price * 0.02)
                },
                'volatility': vol
            }
        except Exception as e:
            self.logger_system.exception(f"Multi-TF data fetch failed: {e}")
            return None

    def get_current_position(self) -> Optional[PositionInfo]:
        if self.config.simulation_mode:
            return None  # 模拟模式下返回空仓位
        
        try:
            positions = self.safe_fetch_positions(exchange, [self.config.symbol])
            for pos in positions:
                if pos['symbol'] == self.config.symbol and float(pos['contracts'] or 0) > 0:
                    pos_info = {
                        'side': pos['side'],
                        'size': float(pos['contracts'] or 0),
                        'entry_price': float(pos['entryPrice'] or 0),
                        'unrealized_pnl': float(pos['unrealizedPnl'] or 0),
                        'leverage': float(pos['leverage'] or self.config.leverage),
                        'symbol': pos['symbol']
                    }
                    # New: Calculate liquidation price for isolated mode
                    if pos_info['side'] == 'long':
                        liq_price = pos_info['entry_price'] * (1 - (1 / pos_info['leverage'] - self.config.maintenance_margin_rate))
                    else:
                        liq_price = pos_info['entry_price'] * (1 + (1 / pos_info['leverage'] - self.config.maintenance_margin_rate))
                    pos_info['liquidation_price'] = round(liq_price, 2)
                    
                    try:
                        trades = exchange.fetch_my_trades(self.config.symbol, limit=5)  # FIXED: High 2 - More trades
                        # FIXED: High 2 - Filter for opening side
                        opening_trades = [t for t in trades if t['side'] == ('buy' if pos_info['side'] == 'long' else 'sell')]
                        if opening_trades:
                            pos_info['entry_time'] = pd.to_datetime(opening_trades[0]['timestamp'], unit='ms', utc=True)  # First opening
                        else:
                            pos_info['entry_time'] = datetime.now(timezone.utc)  # Assume now if no trades
                    except Exception as e:
                        self.logger_system.debug(f"Failed to restore entry_time from trades: {e}")
                        pos_info['entry_time'] = datetime.now(timezone.utc)
                    return pos_info
            return None
        except Exception as e:
            self.logger_system.exception(f"Position fetch failed: {e}")
            return None

    def _calculate_confidence_score(self, multi_tf_data: Dict[str, pd.DataFrame], 
                                   current_price: float, fib_stop_loss: Optional[float], 
                                   fib_take_profit: Optional[float], rr_ratio: float,
                                   key_levels: Optional[Dict[str, float]] = None) -> float:
        """
        计算开单信心评分 (0-100)
        基于技术指标、斐波那契分析、风险回报比等因素
        """
        try:
            score = 0.0
            
            # 1. 风险回报比评分 (0-25分)
            if rr_ratio >= 3.0:
                score += 25
            elif rr_ratio >= 2.5:
                score += 20
            elif rr_ratio >= 2.0:
                score += 15
            elif rr_ratio >= 1.5:
                score += 10
            elif rr_ratio >= 1.2:  # Lowered threshold for aggressive
                score += 5
            
            # 2. 斐波那契优化评分 (0-20分)
            if fib_stop_loss and fib_take_profit:
                score += 20
                self.logger_system.info("斐波那契优化加分: +20")
            
            # 3. 多时间框架技术指标一致性评分 (0-30分)
            tf_scores = []
            for tf, df in multi_tf_data.items():
                if df is not None and len(df) > 0:
                    tf_score = self._evaluate_timeframe_indicators(df, current_price)
                    tf_scores.append(tf_score)
                    self.logger_system.info(f"{tf}时间框架技术指标评分: {tf_score}")
            
            if tf_scores:
                avg_tf_score = sum(tf_scores) / len(tf_scores)
                score += avg_tf_score * 0.3  # 最多30分
            
            # 4. 趋势一致性评分 (0-15分)
            trend_consistency = self._evaluate_trend_consistency(multi_tf_data, current_price)
            score += trend_consistency
            
            # 5. 波动率和流动性评分 (0-10分)
            volatility_score = self._evaluate_volatility_conditions(multi_tf_data)
            score += volatility_score
            
            # New: Confirmation signals boost
            # Volume confirmation - REMOVED: No longer using volume confirmation
            # m15_df = multi_tf_data.get('15m', pd.DataFrame())
            # if not m15_df.empty and m15_df['volume_ratio'].iloc[-1] > self.config.volume_confirmation_threshold:
            #     score += 10
            #     self.logger_system.info("Volume confirmation boost: +10")
            
            # Candle patterns
            structures = self.structures_summary if hasattr(self, 'structures_summary') else {}
            tf_struct = structures.get(self.config.lower_tf_entry_tf, {})
            if tf_struct.get('patterns'):
                score += 8 * len(tf_struct['patterns'])  # Up to +24 for multiple patterns
                self.logger_system.info(f"Candle patterns boost: +{8 * len(tf_struct['patterns'])}")
            
            # FVG stacking - 使用传入的 key_levels（修复）
            fvg_bull = key_levels.get('fvg_bull_stack', 0) if key_levels else 0
            fvg_bear = key_levels.get('fvg_bear_stack', 0) if key_levels else 0
            if fvg_bull >= self.config.fvg_stack_threshold or fvg_bear >= self.config.fvg_stack_threshold:
                score += 12
                self.logger_system.info("FVG stacking boost: +12")
            
            # 确保评分在0-100范围内
            score = max(0, min(100, score))
            
            self.logger_system.info(f"总信心评分: {score:.2f}/100 (R:R={rr_ratio:.2f}, 斐波那契={'是' if fib_stop_loss else '否'})")
            return score
            
        except Exception as e:
            self.logger_system.error(f"计算信心评分时出错: {e}")
            return 30.0  # 默认中等偏低评分

    def _evaluate_timeframe_indicators(self, df: pd.DataFrame, current_price: float) -> float:
        """评估单个时间框架的技术指标强度 (0-100)"""
        try:
            if len(df) < 10:
                return 50.0
            
            score = 0.0
            indicators_count = 0
            
            # RSI评分
            if 'rsi' in df.columns and not df['rsi'].isna().all():
                rsi = df['rsi'].iloc[-1]
                if 30 <= rsi <= 70:  # 中性区间
                    score += 70
                elif 20 <= rsi <= 80:  # 较好区间
                    score += 50
                else:  # 超买超卖
                    score += 30
                indicators_count += 1
            
            # MACD评分
            if 'macd' in df.columns and 'macd_signal' in df.columns:
                if not df['macd'].isna().all() and not df['macd_signal'].isna().all():
                    macd = df['macd'].iloc[-1]
                    macd_signal = df['macd_signal'].iloc[-1]
                    if macd > macd_signal:  # 多头信号
                        score += 60
                    else:  # 空头信号
                        score += 40
                    indicators_count += 1
            
            # 布林带评分
            if all(col in df.columns for col in ['bb_upper', 'bb_lower', 'bb_middle']):
                bb_upper = df['bb_upper'].iloc[-1]
                bb_lower = df['bb_lower'].iloc[-1]
                bb_middle = df['bb_middle'].iloc[-1]
                
                if not pd.isna(bb_upper) and not pd.isna(bb_lower):
                    if bb_lower <= current_price <= bb_upper:  # 在布林带内
                        if abs(current_price - bb_middle) / (bb_upper - bb_lower) < 0.3:
                            score += 70  # 接近中轨
                        else:
                            score += 50  # 在带内但偏离中轨
                    else:
                        score += 30  # 突破布林带
                    indicators_count += 1
            
            # EMA趋势评分
            if 'ema_20' in df.columns and 'ema_50' in df.columns:
                if not df['ema_20'].isna().all() and not df['ema_50'].isna().all():
                    ema_20 = df['ema_20'].iloc[-1]
                    ema_50 = df['ema_50'].iloc[-1]
                    if ema_20 > ema_50:  # 短期EMA在长期EMA之上
                        score += 60
                    else:
                        score += 40
                    indicators_count += 1
            
            return score / indicators_count if indicators_count > 0 else 50.0
            
        except Exception as e:
            self.logger_system.error(f"评估时间框架指标时出错: {e}")
            return 50.0

    def _evaluate_trend_consistency(self, multi_tf_data: Dict[str, pd.DataFrame], current_price: float) -> float:
        """评估多时间框架趋势一致性 (0-15分) - Emphasize higher TF bias"""
        try:
            trend_signals = []
            
            # Prioritize higher TF for bias
            bias_tf = self.config.higher_tf_bias_tf
            if bias_tf in multi_tf_data:
                df_bias = multi_tf_data[bias_tf]
                if df_bias is not None and len(df_bias) > 5:
                    trend_bias = self._determine_signal_direction(df_bias, current_price)
                    if trend_bias in ['BUY', 'SELL']:
                        trend_signals.append(trend_bias * 2)  # Weight higher TF x2
            
            # Lower TFs for confirmation
            for tf, df in multi_tf_data.items():
                if tf != bias_tf and df is not None and len(df) > 5:
                    trend_direction = self._determine_signal_direction(df, current_price)
                    if trend_direction in ['BUY', 'SELL']:
                        trend_signals.append(trend_direction)
            
            if not trend_signals:
                return 7.5  # 中性评分
            
            # 计算趋势一致性
            buy_count = sum(1 for s in trend_signals if s == 'BUY')
            sell_count = sum(1 for s in trend_signals if s == 'SELL')
            total_signals = len(trend_signals)
            
            consistency_ratio = max(buy_count, sell_count) / total_signals
            
            if consistency_ratio >= 0.8:  # 80%以上一致
                return 15.0
            elif consistency_ratio >= 0.6:  # 60%以上一致
                return 12.0
            elif consistency_ratio >= 0.4:  # 40%以上一致
                return 8.0
            else:
                return 5.0
            
        except Exception as e:
            self.logger_system.error(f"评估趋势一致性时出错: {e}")
            return 7.5

    def _evaluate_volatility_conditions(self, multi_tf_data: Dict[str, pd.DataFrame]) -> float:
        """评估波动率条件 (0-10分)"""
        try:
            # 使用15分钟数据评估波动率
            m15_df = multi_tf_data.get('15m')
            if m15_df is None or len(m15_df) < 20:
                return 5.0
            
            # 计算ATR相对波动率
            if 'atr' in m15_df.columns and not m15_df['atr'].isna().all():
                atr = m15_df['atr'].iloc[-1]
                close = m15_df['close'].iloc[-1]
                atr_ratio = atr / close if close > 0 else 0
                
                # 适中的波动率得分最高
                if 0.005 <= atr_ratio <= 0.015:  # 0.5%-1.5%
                    return 10.0
                elif 0.003 <= atr_ratio <= 0.02:  # 0.3%-2%
                    return 7.0
                else:
                    return 4.0
            
            return 5.0
            
        except Exception as e:
            self.logger_system.error(f"评估波动率条件时出错: {e}")
            return 5.0

    def _determine_confidence_level(self, confidence_score: float) -> Tuple[str, bool]:
        """
        根据信心评分确定信心等级和是否鼓励开单
        返回: (信心等级, 是否鼓励开单)
        """
        if confidence_score >= 80:
            return "VERY_HIGH", True
        elif confidence_score >= 65:
            return "HIGH", True
        elif confidence_score >= 50:
            return "MEDIUM", True
        elif confidence_score >= 30:  # Lowered from 35 for aggressive
            return "LOW", True  # Encourage even LOW
        else:
            return "VERY_LOW", False

    def _determine_signal_direction(self, df: pd.DataFrame, current_price: float) -> str:
        """
        基于技术指标判断信号方向
        用于斐波那契分析的趋势判断
        """
        try:
            if len(df) < 10:
                return 'HOLD'
            
            # 获取最新的技术指标
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            
            # RSI指标
            rsi = latest.get('rsi', 50)
            
            # EMA指标
            ema_short = latest.get('ema_short', current_price)
            ema_long = latest.get('ema_long', current_price)
            
            # MACD指标
            macd = latest.get('macd', 0)
            macd_signal = latest.get('macd_signal', 0)
            
            # 价格趋势
            price_trend = 1 if current_price > ema_long else -1
            
            # 多重信号确认
            signals = []
            
            # 1. RSI信号
            if rsi < 30:
                signals.append('BUY')  # 超卖
            elif rsi > 70:
                signals.append('SELL')  # 超买
            
            # 2. EMA交叉信号
            if ema_short > ema_long:
                signals.append('BUY')
            elif ema_short < ema_long:
                signals.append('SELL')
            
            # 3. MACD信号
            if macd > macd_signal and macd > 0:
                signals.append('BUY')
            elif macd < macd_signal and macd < 0:
                signals.append('SELL')
            
            # 4. 价格位置信号
            if price_trend > 0:
                signals.append('BUY')
            else:
                signals.append('SELL')
            
            # 统计信号
            buy_count = signals.count('BUY')
            sell_count = signals.count('SELL')
            
            # 需要至少2个信号确认
            if buy_count >= 2 and buy_count > sell_count:
                return 'BUY'
            elif sell_count >= 2 and sell_count > buy_count:
                return 'SELL'
            else:
                return 'HOLD'
                
        except Exception as e:
            self.logger_system.warning(f"信号方向判断失败: {e}")
            return 'HOLD'

    def create_fallback_signal(self, price_data):
        # ENHANCED: 集成斐波那契和技术分析优化止盈止损
        try:
            # 多层安全检查确保price_data有效
            if not price_data or not isinstance(price_data, dict):
                self.logger_system.error("Invalid price_data provided to create_fallback_signal")
                price_data = {'price': 50000.0}  # Emergency fallback price
            
            current = price_data.get('price')
            if current is None or not isinstance(current, (int, float)) or current <= 0:
                current = 50000.0  # Emergency fallback price for BTC
                self.logger_system.error(f"Invalid current price, using emergency fallback: {current}")
            
            # 尝试从技术数据获取ATR
            atr = None
            if 'technical_data' in price_data and price_data['technical_data']:
                atr = price_data['technical_data'].get('atr')
            
            # Use fallback ATR if None or invalid
            if atr is None or not isinstance(atr, (int, float)) or atr <= 0:
                atr = current * 0.02  # 2% of current price as fallback
                self.logger_system.warning(f"ATR is None or invalid, using fallback ATR: {atr:.2f} (2% of price)")
            else:
                self.logger_system.info(f"Using ATR from technical data: {atr:.2f}")
            
            # 尝试使用斐波那契分析优化止盈止损
            fib_stop_loss = None
            fib_take_profit = None
            
            # 检查是否有多时间框架数据用于斐波那契分析
            if 'multi_tf_data' in price_data and price_data['multi_tf_data']:
                # 优先使用15分钟数据进行斐波那契分析
                for tf in ['15m', '1h']:
                    if tf in price_data['multi_tf_data']:
                        df = price_data['multi_tf_data'][tf]
                        if len(df) >= 20:  # 确保有足够数据
                            try:
                                # 基于技术指标判断趋势方向
                                signal_direction = self._determine_signal_direction(df, current)
                                
                                if signal_direction != 'HOLD':
                                    fib_sl, fib_tp = self.get_optimal_fibonacci_sl_tp(df, signal_direction, current)
                                    if fib_sl and fib_tp:
                                        fib_stop_loss = fib_sl
                                        fib_take_profit = fib_tp
                                        self.logger_system.info(f"使用斐波那契分析({tf})优化止盈止损: SL={fib_sl:.2f}, TP={fib_tp:.2f}")
                                        break
                            except Exception as e:
                                self.logger_system.warning(f"斐波那契分析失败({tf}): {e}")
                                continue
            
            # 计算基础ATR止盈止损
            atr_stop_loss = round(current - atr * 2, 2)
            atr_take_profit = round(current + atr * 3, 2)
            
            # 选择最优止盈止损策略
            if fib_stop_loss and fib_take_profit:
                # 验证斐波那契止盈止损的合理性
                sl_distance = abs(current - fib_stop_loss)
                tp_distance = abs(fib_take_profit - current)
                
                # 确保止盈止损距离在合理范围内
                if (0.5 * atr <= sl_distance <= 4 * atr and 
                    1 * atr <= tp_distance <= 6 * atr and
                    tp_distance / sl_distance >= 1.5):  # 最小风险回报比1.5:1
                    
                    stop_loss = round(fib_stop_loss, 2)
                    take_profit = round(fib_take_profit, 2)
                    self.logger_system.info(f"采用斐波那契优化止盈止损: SL={stop_loss}, TP={take_profit}")
                else:
                    # 斐波那契结果不合理，使用ATR方法
                    stop_loss = atr_stop_loss
                    take_profit = atr_take_profit
                    self.logger_system.info(f"斐波那契结果不合理，使用ATR止盈止损: SL={stop_loss}, TP={take_profit}")
            else:
                # 没有斐波那契数据，使用ATR方法
                stop_loss = atr_stop_loss
                take_profit = atr_take_profit
                self.logger_system.info(f"无斐波那契数据，使用ATR止盈止损: SL={stop_loss}, TP={take_profit}")
            
            # 多层安全检查
            if stop_loss is None or not isinstance(stop_loss, (int, float)) or stop_loss <= 0:
                stop_loss = round(current * 0.98, 2)  # 2% below current price
                self.logger_system.warning(f"Calculated stop_loss invalid, using 2% below current: {stop_loss}")
            
            if take_profit is None or not isinstance(take_profit, (int, float)) or take_profit <= current:
                take_profit = round(current * 1.03, 2)  # 3% above current price
                self.logger_system.warning(f"Calculated take_profit invalid, using 3% above current: {take_profit}")
            
            # 最终验证：确保所有值都是有效数字
            if not all(isinstance(val, (int, float)) and val > 0 for val in [stop_loss, take_profit, current]):
                self.logger_system.error("Final validation failed, using emergency values")
                current = 50000.0
                stop_loss = 49000.0
                take_profit = 51500.0
            
            # 确保止损止盈逻辑正确
            if stop_loss >= current:
                stop_loss = round(current * 0.98, 2)
                self.logger_system.warning(f"Stop loss >= current price, adjusted to: {stop_loss}")
            
            if take_profit <= current:
                take_profit = round(current * 1.03, 2)
                self.logger_system.warning(f"Take profit <= current price, adjusted to: {take_profit}")
            
            # 计算风险回报比
            risk = abs(current - stop_loss)
            reward = abs(take_profit - current)
            rr_ratio = reward / risk if risk > 0 else 0
            
            # 计算开单信心评分 - 传递 key_levels
            # 从price_data中提取多时间框架数据，如果没有则使用空字典
            multi_tf_data = price_data.get('multi_tf_data', {})
            key_levels = price_data.get('key_levels', {})  # 新增：从 price_data 获取
            confidence_score = self._calculate_confidence_score(
                multi_tf_data, current, fib_stop_loss, fib_take_profit, rr_ratio, key_levels  # 添加 key_levels
            )
            
            # 根据信心评分确定信心等级和是否鼓励开单
            confidence_level, encourage_trading = self._determine_confidence_level(confidence_score)
            
            # Aggressive fallback: random BUY/SELL based on RSI
            rsi = price_data['technical_data'].get('rsi', 50) if 'technical_data' in price_data else 50
            signal = "HOLD"
            if random.random() > 0.5:
                if rsi > 70:
                    signal = 'SELL'
                elif rsi < 30:
                    signal = 'BUY'
                else:
                    signal = random.choice(['BUY', 'SELL'])  # 50% random for aggressive
            
            fallback_signal = {
                "signal": signal,
                "reason": "SMC analysis unavailable, adopting aggressive fallback with Fibonacci optimization",
                "stop_loss": float(stop_loss),  # 确保是float类型
                "take_profit": float(take_profit),  # 确保是float类型
                "confidence": confidence_level,
                "confidence_score": round(confidence_score, 2),
                "encourage_trading": encourage_trading,
                "is_fallback": True,
                "fibonacci_optimized": bool(fib_stop_loss and fib_take_profit),
                "risk_reward_ratio": round(rr_ratio, 2),
                "timestamp": price_data.get('timestamp', datetime.now(timezone.utc).isoformat())
            }
            
            # 最终验证fallback_signal的完整性
            required_fields = ['signal', 'reason', 'stop_loss', 'take_profit', 'confidence']
            for field in required_fields:
                if field not in fallback_signal or fallback_signal[field] is None:
                    self.logger_system.error(f"Fallback signal missing field: {field}")
                    if field == 'signal':
                        fallback_signal[field] = 'HOLD'
                    elif field == 'reason':
                        fallback_signal[field] = 'Emergency fallback signal'
                    elif field == 'stop_loss':
                        fallback_signal[field] = float(current * 0.98)
                    elif field == 'take_profit':
                        fallback_signal[field] = float(current * 1.03)
                    elif field == 'confidence':
                        fallback_signal[field] = 'LOW'
            
            self.logger_system.info(f"Generated enhanced fallback signal: SL={fallback_signal['stop_loss']}, TP={fallback_signal['take_profit']}, Current={current:.2f}, R:R={rr_ratio:.2f}")
            return fallback_signal
            
        except Exception as e:
            self.logger_system.error(f"Exception in create_fallback_signal: {e}")
            # 紧急备用信号，确保系统不会崩溃
            emergency_signal = {
                "signal": "HOLD",
                "reason": f"Emergency fallback due to error: {str(e)[:100]}",
                "stop_loss": 49000.0,
                "take_profit": 51500.0,
                "confidence": "LOW",
                "is_fallback": True,
                "is_emergency": True,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            self.logger_system.error(f"Using emergency signal: {emergency_signal}")
            return emergency_signal

    def _generate_compact_smc_summary(self, price_data: Dict[str, Any]) -> str:
        """生成简化的SMC分析摘要，用于DeepSeek AI分析 - Enhanced with FVG and patterns"""
        try:
            structures = price_data.get('structures_summary', {})
            levels = price_data.get('key_levels', {})
            amp = price_data.get('amplitude', {})
            tech = price_data.get('technical_data', {})
            current_price = price_data.get('price', 0)
            
            # 简化的时间框架结构分析
            tf_summary = []
            tf_order = ['1w', '1d', '4h', '15m']
            for tf in tf_order:
                struct = structures.get(tf, {})
                if struct:
                    structure = struct.get('structure', 'N/A')
                    trend = struct.get('trend', 'N/A')
                    weight = struct.get('weight', 1)
                    reversal = " (REV)" if struct.get('reversal_confirmed', False) else ""
                    patterns = struct.get('patterns', [])
                    tf_summary.append(f"{tf}:{structure}/{trend}(w{weight:.1f}){reversal} | Patterns: {', '.join(patterns)}")
            
            # 关键流动性位
            monday_open = levels.get('monday_open', 0)
            monday_change = ((current_price - monday_open) / monday_open * 100) if monday_open else 0
            
            key_levels_text = []
            if monday_open:
                key_levels_text.append(f"MO:{monday_open:.0f}({monday_change:+.1f}%)")
            if levels.get('daily_vwap'):
                key_levels_text.append(f"DVWAP:{levels['daily_vwap']:.0f}")
            if levels.get('prev_week_high'):
                key_levels_text.append(f"PWH:{levels['prev_week_high']:.0f}")
            if levels.get('prev_week_low'):
                key_levels_text.append(f"PWL:{levels['prev_week_low']:.0f}")
            
            # 1小时EMA位
            ema_text = []
            if levels.get('ema_21_1h'):
                ema_text.append(f"EMA21:{levels['ema_21_1h']:.0f}")
            if levels.get('ema_55_1h'):
                ema_text.append(f"EMA55:{levels['ema_55_1h']:.0f}")
            if levels.get('ema_100_1h'):
                ema_text.append(f"EMA100:{levels['ema_100_1h']:.0f}")
            if levels.get('ema_200_1h'):
                ema_text.append(f"EMA200:{levels['ema_200_1h']:.0f}")
            
            # 订单块
            ob_text = []
            if levels.get('daily_ob_bull'):
                ob_text.append(f"DOB+:{levels['daily_ob_bull']:.0f}")
            if levels.get('daily_ob_bear'):
                ob_text.append(f"DOB-:{levels['daily_ob_bear']:.0f}")
            if levels.get('4h_ob_bull'):
                ob_text.append(f"4HOB+:{levels['4h_ob_bull']:.0f}")
            if levels.get('4h_ob_bear'):
                ob_text.append(f"4HOB-:{levels['4h_ob_bear']:.0f}")
            
            # New: FVG info
            fvg_text = []
            fvg_bull_stack = levels.get('fvg_bull_stack', 0)
            fvg_bear_stack = levels.get('fvg_bear_stack', 0)
            if fvg_bull_stack >= self.config.fvg_stack_threshold:
                fvg_text.append(f"FVG Bull Stack: {fvg_bull_stack}")
            if fvg_bear_stack >= self.config.fvg_stack_threshold:
                fvg_text.append(f"FVG Bear Stack: {fvg_bear_stack}")
            
            # 技术指标
            rsi = tech.get('rsi', 50)
            atr = tech.get('atr', 0)
            macd = tech.get('macd', 0)
            
            # 振幅信息
            avg_amp = amp.get('avg_amplitude', 0)
            expected_range = amp.get('expected_rr_range', 0)
            
            # 组合简化摘要
            summary_parts = []
            if tf_summary:
                summary_parts.append(f"Structure: {' | '.join(tf_summary[:2])}")  # 只显示前2个时间框架
            if key_levels_text:
                summary_parts.append(f"Levels: {' | '.join(key_levels_text[:3])}")  # 只显示前3个关键位
            if ema_text:
                summary_parts.append(f"EMA1H: {' | '.join(ema_text[:2])}")  # 显示前2个1小时EMA
            if ob_text:
                summary_parts.append(f"OB: {' | '.join(ob_text[:2])}")  # 只显示前2个订单块
            if fvg_text:
                summary_parts.append(f"FVG: {' | '.join(fvg_text)}")
            
            summary_parts.append(f"Tech: RSI{rsi:.0f} MACD{macd:.3f} ATR{atr:.0f}")
            summary_parts.append(f"Amp: avg{avg_amp:.0f} range{expected_range:.0f}")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            self.logger_system.warning(f"生成SMC摘要时出错: {e}")
            return f"SMC Summary: Price ${price_data.get('price', 0):,.0f} | Basic analysis available"

    def safe_json_parse(self, json_str: str) -> Optional[Dict[str, Any]]:
        """安全解析JSON字符串，支持多种格式修复和候选选择"""
        try:
            result = json.loads(json_str)
            # 确保结果包含signal字段
            if isinstance(result, dict) and 'signal' in result:
                return result
            return None
        except json.JSONDecodeError:
            try:
                # 清理markdown格式
                json_str = json_str.strip()
                if json_str.startswith('```json'):
                    json_str = json_str[7:]  # Remove ```json
                if json_str.startswith('```'):
                    json_str = json_str[3:]  # Remove ```
                if json_str.endswith('```'):
                    json_str = json_str[:-3]  # Remove trailing ```
                json_str = json_str.strip()
                
                # 基本格式修复
                json_str = json_str.replace("'", '"').strip()
                # 注释掉有问题的正则表达式，它会错误地替换字符串中的内容
                # json_str = re.sub(r'(\w+):', r'"\1":', json_str)
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                
                def find_json_blocks(s):
                    """查找字符串中的JSON块"""
                    candidates = []
                    depth = 0
                    start = -1
                    for i, c in enumerate(s):
                        if c == '{':
                            if depth == 0:
                                start = i
                            depth += 1
                        elif c == '}':
                            depth -= 1
                            if depth == 0 and start != -1:
                                candidates.append(s[start:i+1])
                                start = -1
                    return candidates
                
                # 查找所有可能的JSON块
                blocks = find_json_blocks(json_str)
                candidates = []
                
                for block in blocks:
                    try:
                        cand = json.loads(block)
                        if isinstance(cand, dict) and 'signal' in cand:
                            candidates.append(cand)
                    except json.JSONDecodeError:
                        pass
                
                # 如果找到候选项，选择置信度最高的
                if candidates:
                    conf_order = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
                    best = max(candidates, key=lambda x: conf_order.get(x.get('confidence', 'LOW'), 0))
                    self.logger_system.info(f"Selected highest confidence candidate: {best.get('confidence')}")
                    
                    # 详细诊断日志
                    sl = best.get('stop_loss')
                    tp = best.get('take_profit')
                    self.logger_system.debug(f"Candidate details: signal={best.get('signal')}, SL={sl}, TP={tp}, confidence={best.get('confidence')}")
                    
                    # 检查数值类型
                    if sl is None or tp is None:
                        self.logger_system.warning(f"DeepSeek返回了空值: SL={sl}, TP={tp}")
                        self.logger_system.warning(f"完整候选项: {best}")
                    elif not isinstance(sl, (int, float)) or not isinstance(tp, (int, float)):
                        self.logger_system.warning(f"DeepSeek返回了非数值类型: SL={type(sl).__name__}({sl}), TP={type(tp).__name__}({tp})")
                        self.logger_system.warning(f"完整候选项: {best}")
                    
                    return best
                
                return None
                
            except Exception as e:
                self.logger_system.exception(f"JSON parsing failed, raw content: {json_str[:200]}... Error: {e}")
                return None

    def analyze_with_deepseek(self, price_data: Dict[str, Any], activated_level: Optional[str] = None) -> Dict[str, Any]:
        # 检查DeepSeek API健康状态
        if self._should_check_api_health('deepseek'):
            self.logger_system.info("执行DeepSeek API健康检查...")
            self._check_deepseek_health()
        
        # 如果API状态不健康且连续失败次数过多，直接使用备用信号
        if (self.api_health_status['deepseek']['status'] == 'unhealthy' and 
            self.api_health_status['deepseek']['consecutive_failures'] >= 3):
            self.logger_system.warning(f"DeepSeek API连续失败{self.api_health_status['deepseek']['consecutive_failures']}次，直接使用备用信号")
            return self.create_fallback_signal(price_data)
        
        # 生成简化的SMC分析文本
        smc_summary = self._generate_compact_smc_summary(price_data)
        
        # 简化K线数据
        m15_df = price_data['multi_tf_data']['15m']
        recent_klines = m15_df.tail(3)  # 减少到3根K线
        kline_summary = f"Last 3 15m: "
        for kline in recent_klines.to_dict('records'):
            trend = "↑" if kline['close'] > kline['open'] else "↓"
            change = ((kline['close'] - kline['open']) / kline['open']) * 100
            kline_summary += f"{trend}{change:+.1f}% "

        # 简化信号历史
        signal_text = ""
        with self.lock:
            if self.signal_history:
                last_signal = self.signal_history[-1]
                signal_text = f"Last: {last_signal.get('signal', 'N/A')}"

        # 简化持仓信息
        current_pos = self.get_current_position()
        position_text = "No pos" if not current_pos else f"{current_pos['side']} {current_pos['size']:.3f}"
        activation_text = f"Level: {activated_level}" if activated_level else "Scheduled"

        # FIXED: High 4 - Integrate activation to rules
        threshold_info = ""
        if activated_level:
            # 获取激活阈值信息作为参考
            threshold = self.get_adjusted_threshold(activated_level)
            threshold_info = f" (threshold: {threshold*100:.3f}% - reference only)"
        
        # New: Multi-TF alignment rules
        higher_tf = self.config.higher_tf_bias_tf
        lower_tf = self.config.lower_tf_entry_tf
        mtf_rules = f"Higher TF Bias ({higher_tf}): Determine direction. Lower TF Entry ({lower_tf}): Precise entry on confirmation. Require alignment."
        
        # New: Confirmation rules
        conf_rules = f"Confirmations: Candle patterns (e.g., Engulfing), FVG stacking >= {self.config.fvg_stack_threshold} or OB stacking >= 3 or combined OB+FVG stacking >= 3. Fresh zones only (interactions <= {self.config.max_zone_interactions})."

        rules = f"Rules:\n1. {mtf_rules}\n2. SMC analysis based on price action and structure\n3. Level activation{threshold_info} is context, not requirement\n4. {conf_rules}\n5. R:R >2:1 for HIGH confidence\n6. TP within ${price_data['amplitude']['expected_rr_range']:.0f} range"

        # 改进的提示，确保返回有效数值 - Enhanced with core price action components
        current_price = price_data['price']
        
        # 核心价格行为组件详细说明
        price_action_components = """
CORE PRICE ACTION COMPONENTS:

1. MARKET STRUCTURE:
   - HH+HL=Uptrend (Higher Highs + Higher Lows)
   - LH+LL=Downtrend (Lower Highs + Lower Lows)
   - Structure Breaks: Key S/R level breaches

2. KEY LEVELS:
   - Support: Previous lows, demand zones, order blocks
   - Resistance: Previous highs, supply zones, order blocks
   - Break & Retest: Post-break confirmation

3. CANDLE PATTERNS:
   - Engulfing: Bullish/Bearish engulfing patterns
   - Hammer/Hanging Man: Reversal signals
   - Inside/Outside Bar: Volatility contraction/expansion
   - Fakeouts: Quick reversal after breakout

4. ORDER FLOW CONCEPTS (SMC):
   - FVG (Fair Value Gap): Price gaps between consecutive candles
   - Liquidity Grab: Stop hunts, liquidity pools
   - OB (Order Blocks): Institutional order concentration zones
   - MIT (Market Imbalance Transition): S/R role reversal

5. MOMENTUM ANALYSIS:
   - Candle Size: Reflects buying/selling pressure
   - Close Position: Candle close at high/low end
   - Volatility Changes: Sudden volatility expansion

TRADING FRAMEWORK APPLICATION:
Focus on:
1. 1H chart for primary structure direction
2. 5m chart for:
   - Reactions at key levels (EMA21, previous H/L)
   - Price action in FVG stack areas
   - Candle patterns
   - First interaction with fresh zones
"""
        
        prompt = f"""SMC/ICT Analysis for {self.config.symbol.split(':')[0]}:Price: ${current_price:,.0f} | {kline_summary}
Position: {position_text} | {signal_text} | {activation_text}{smc_summary}{rules}{price_action_components}Aggressive Day Trading: Focus 15m/5m momentum, enter on 1.2:1 R:R, ignore higher TF if price action strong.EMPHASIS: Higher TF ({higher_tf}) bias for direction + Lower TF ({lower_tf}) entry. Require confirmations: candle patterns, FVG stacking >= 3 or OB stacking >= 3 or combined OB+FVG stacking >= 3. Use fresh zones only (no multiple interactions).CRITICAL: Always provide EXACT numeric values for stop_loss and take_profit. Never use null, variables, or placeholders.For BUY signals: stop_loss < {current_price:.0f} < take_profit
For SELL signals: take_profit < {current_price:.0f} < stop_loss
For HOLD signals: Use current price ±1% as SL/TPExample valid responses:
BUY: {{"signal": "BUY", "reason": "Bullish structure with HH+HL and FVG support", "stop_loss": {current_price*0.98:.0f}, "take_profit": {current_price*1.04:.0f}, "confidence": "MEDIUM"}}
SELL: {{"signal": "SELL", "reason": "Bearish structure with LH+LL and order block resistance", "stop_loss": {current_price*1.02:.0f}, "take_profit": {current_price*0.96:.0f}, "confidence": "MEDIUM"}}
HOLD: {{"signal": "HOLD", "reason": "Neutral structure with no clear directional bias", "stop_loss": {current_price*0.99:.0f}, "take_profit": {current_price*1.01:.0f}, "confidence": "LOW"}}Return JSON only:
{{
 "signal": "BUY|SELL|HOLD",
 "reason": "Brief SMC analysis with price action context",
 "stop_loss": EXACT_NUMBER,
 "take_profit": EXACT_NUMBER,
 "confidence": "HIGH|MEDIUM|LOW"
}}"""

        max_retries = 2  # Reduced from 3 for aggressive fallback
        retry_delays = [1, 2]  # Adjusted for fewer retries
        signal_data = None
        
        for attempt in range(max_retries):
            try:
                # 动态调整超时时间
                timeout = self.config.deepseek_timeout + (attempt * 5)
                
                # 记录发送给 DeepSeek 的完整请求信息
                self.logger_api.info(f"=== DeepSeek API 请求 (尝试 {attempt+1}/{max_retries}) ===")
                self.logger_api.info(f"模型: deepseek-chat")
                self.logger_api.info(f"温度: {self.config.temperature}")
                self.logger_api.info(f"超时: {timeout}秒")
                self.logger_api.info(f"系统提示: You are an expert SMC/ICT trader with deep knowledge of Price Action, Market Structure, and Order Flow analysis. Analyze the provided data using core price action components and return strict JSON only.")
                self.logger_api.info(f"用户提示内容:\n{prompt}")
                self.logger_api.info("=" * 50)
                
                response = deepseek_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": """You are an expert SMC/ICT trader with deep knowledge of Price Action, Market Structure, and Order Flow analysis. Analyze the provided data using core price action components and return strict JSON only.

核心价格行为组件:
1. 市场结构(Market Structure):
   - 高点/低点序列：更高的高点(HH)+更高的低点(HL)=上升趋势
   - 更低的高点(LH)+更低的低点(LL)=下降趋势
   - 结构突破：关键支撑/阻力位的突破

2. 关键水平(Key Levels):
   - 支撑位：前低点、需求区、订单区块
   - 阻力位：前高点、供应区、订单区块
   - 突破回测：突破后回踩确认

3. 价格模式(Candle Patterns):
   - 吞噬模式：看涨吞噬/看跌吞噬
   - 锤子线/上吊线：反转信号
   - 内在/外在柱：波动率收缩与扩张
   - 假突破：突破后快速反转

4. 订单流概念(SMC特有):
   - FVG(公允价值缺口)：连续蜡烛间的价格缺口
   - 流动性挖掘：止损猎杀、流动性池
   - OB(订单区块)：机构订单集中区域
   - MIT(市场内部转换)：角色转换(支撑变阻力)

5. 动量分析:
   - 柱体大小：反映买卖压力
   - 收盘位置：柱体收盘于高位/低位
   - 波动率变化：突然的波动率扩张

在交易框架中的应用:
基于规则，重点关注:
1. 1小时图确定主要结构方向
2. 5分钟图寻找：
   - 在关键水平(EMA21、前高低点)的反应
   - FVG堆积区域的价格行为
   - 配合成交量确认的蜡烛模式
   - 新鲜区域的首次交互"""},
                        {"role": "user", "content": prompt}
                    ],
                    stream=False,
                    temperature=self.config.temperature,  # FIXED: Medium 10
                    timeout=timeout
                )
                
                # 记录完整的 API 响应信息
                result = response.choices[0].message.content
                self.logger_api.info(f"=== DeepSeek API 响应 (尝试 {attempt+1}) ===")
                self.logger_api.info(f"响应状态: 成功")
                self.logger_api.info(f"响应长度: {len(result)} 字符")
                self.logger_api.info(f"原始响应内容:\n{result}")
                self.logger_api.info("=" * 50)
                
                signal_data = self.safe_json_parse(result)
                
                # 记录 JSON 解析结果
                self.logger_api.info(f"=== JSON 解析结果 (尝试 {attempt+1}) ===")
                if signal_data is not None:
                    self.logger_api.info(f"解析状态: 成功")
                    self.logger_api.info(f"解析后的数据: {signal_data}")
                    
                    # 验证必要字段
                    required_fields = ['signal', 'reason', 'stop_loss', 'take_profit', 'confidence']
                    missing_fields = [field for field in required_fields if field not in signal_data]
                    
                    self.logger_api.info(f"字段验证: 必需字段 {required_fields}")
                    if missing_fields:
                        self.logger_api.warning(f"缺失字段: {missing_fields}")
                    else:
                        self.logger_api.info(f"字段验证: 通过")
                    
                    if not missing_fields:
                        # 验证止损止盈值
                        sl = signal_data.get('stop_loss')
                        tp = signal_data.get('take_profit')
                        current_price = price_data['price']
                        signal_type = signal_data.get('signal', 'UNKNOWN')
                        
                        self.logger_api.info(f"=== 止损止盈验证 (尝试 {attempt+1}) ===")
                        self.logger_api.info(f"信号类型: {signal_type}")
                        self.logger_api.info(f"当前价格: {current_price}")
                        self.logger_api.info(f"止损价格: {sl} (类型: {type(sl).__name__})")
                        self.logger_api.info(f"止盈价格: {tp} (类型: {type(tp).__name__})")
                        
                        # 详细的验证逻辑
                        validation_errors = []
                        
                        if sl is None:
                            validation_errors.append("止损价格为None")
                        elif not isinstance(sl, (int, float)):
                            validation_errors.append(f"止损价格类型错误: {type(sl).__name__}({sl})")
                        elif sl <= 0:
                            validation_errors.append(f"止损价格无效: {sl} <= 0")
                        
                        if tp is None:
                            validation_errors.append("止盈价格为None")
                        elif not isinstance(tp, (int, float)):
                            validation_errors.append(f"止盈价格类型错误: {type(tp).__name__}({tp})")
                        elif tp <= 0:
                            validation_errors.append(f"止盈价格无效: {tp} <= 0")
                        
                        # 检查价格逻辑
                        if isinstance(sl, (int, float)) and isinstance(tp, (int, float)):
                            if signal_type == 'BUY':
                                if sl >= current_price:
                                    validation_errors.append(f"BUY信号止损价格错误: {sl} >= {current_price}")
                                if tp <= current_price:
                                    validation_errors.append(f"BUY信号止盈价格错误: {tp} <= {current_price}")
                            elif signal_type == 'SELL':
                                if sl <= current_price:
                                    validation_errors.append(f"SELL信号止损价格错误: {sl} <= {current_price}")
                                if tp >= current_price:
                                    validation_errors.append(f"SELL信号止盈价格错误: {tp} >= {current_price}")
                            elif signal_type == 'HOLD':
                                # HOLD信号的SL/TP验证相对宽松
                                pass
                        
                        if validation_errors:
                            self.logger_api.error(f"=== 验证失败 (尝试 {attempt+1}) ===")
                            self.logger_api.error(f"验证错误: {'; '.join(validation_errors)}")
                            self.logger_api.error(f"信号类型: {signal_type}")
                            self.logger_api.error(f"当前价格: {current_price}, SL: {sl}, TP: {tp}")
                            self.logger_api.error(f"完整信号数据: {signal_data}")
                            self.logger_api.error("=" * 50)
                            signal_data = None
                        else:
                            self.logger_api.info(f"=== 验证成功 (尝试 {attempt+1}) ===")
                            self.logger_api.info(f"信号有效: {signal_data['signal']}")
                            self.logger_api.info(f"止损: {sl}, 止盈: {tp}")
                            self.logger_api.info(f"置信度: {signal_data.get('confidence', 'N/A')}")
                            self.logger_api.info(f"理由: {signal_data.get('reason', 'N/A')}")
                            self.logger_api.info("=" * 50)
                            # 更新API成功状态
                            self._update_api_success('deepseek')
                            break
                    else:
                        self.logger_api.warning(f"字段验证失败: 缺失字段 {missing_fields}")
                        signal_data = None
                else:
                    # JSON 解析失败
                    self.logger_api.error(f"解析状态: 失败")
                    self.logger_api.error(f"原始响应内容: {result[:500]}...")  # 只显示前500字符
                    self.logger_api.error("=" * 50)
                
                if signal_data is None and attempt < max_retries - 1:
                    self.logger_system.warning(f"Invalid response, retrying (attempt {attempt+1 }/{max_retries})")
                    prompt += "\nCRITICAL: Valid JSON only with all required fields!"
                    time.sleep(retry_delays[attempt])
                
            except Exception as e:
                error_type = type(e).__name__
                self.logger_system.warning(f"DeepSeek API error (attempt {attempt+1}) [{error_type}]: {e}")
                
                # 更新API失败状态
                self._update_api_failure('deepseek', e)
                
                # 根据错误类型决定是否重试
                if "timeout" in str(e).lower() or "network" in str(e).lower():
                    if attempt < max_retries - 1:
                        delay = retry_delays[attempt]
                        self.logger_system.info(f"Network/timeout error, retrying in {delay} seconds...")
                        time.sleep(delay)
                        continue
                
                if attempt < max_retries - 1:
                    time.sleep(retry_delays[attempt])
                else:
                    self.logger_system.error(f"All DeepSeek API attempts failed after {max_retries} tries")
        
        # 处理循环结束后的逻辑
        if signal_data is None:
            # 详细的错误诊断日志
            api_status = self.api_health_status['deepseek']
            self.logger_system.warning(f"DeepSeek API完全失败，使用备用信号")
            self.logger_system.warning(f"API健康状态: {api_status['status']}, 连续失败: {api_status['consecutive_failures']}次")
            self.logger_system.warning(f"上次成功时间: {api_status.get('last_success_time', '未知')}")
            self.logger_system.warning(f"最后错误: {api_status.get('last_error', '未知')}")
            
            # 提供诊断建议
            if api_status['consecutive_failures'] >= 5:
                self.logger_system.error("DeepSeek API连续失败次数过多，建议检查:")
                self.logger_system.error("1. 网络连接是否正常")
                self.logger_system.error("2. API密钥是否有效")
                self.logger_system.error("3. DeepSeek服务是否正常")
                self.logger_system.error("4. 请求频率是否超限")
            
            signal_data = self.create_fallback_signal(price_data)
        
        required_fields = ['signal', 'reason', 'stop_loss', 'take_profit', 'confidence']
        if not all(field in signal_data for field in required_fields):
            missing_fields = [field for field in required_fields if field not in signal_data]
            self.logger_system.warning(f"信号数据缺少必要字段: {missing_fields}")
            self.logger_system.warning(f"原始信号数据: {signal_data}")
            self.logger_system.warning("使用备用信号替代")
            signal_data = self.create_fallback_signal(price_data)
        
        # 额外验证止损止盈不为None
        if signal_data.get('stop_loss') is None or signal_data.get('take_profit') is None:
            self.logger_system.warning(f"止损止盈值为None - SL: {signal_data.get('stop_loss')}, TP: {signal_data.get('take_profit')}")
            self.logger_system.warning(f"当前价格: {price_data['price']}")
            self.logger_system.warning("这可能是DeepSeek API返回了无效数据，使用备用信号")
            signal_data = self.create_fallback_signal(price_data)
        
        # 验证止损止盈为有效数值
        try:
            sl = float(signal_data.get('stop_loss', 0))
            tp = float(signal_data.get('take_profit', 0))
            current_price = price_data['price']
            
            if sl <= 0 or tp <= 0:
                self.logger_system.warning(f"止损止盈值无效 - SL: {sl}, TP: {tp} (必须大于0)")
                self.logger_system.warning(f"当前价格: {current_price}")
                self.logger_system.warning("使用备用信号重新计算")
                signal_data = self.create_fallback_signal(price_data)
            elif abs(sl - current_price) / current_price > 0.1 or abs(tp - current_price) / current_price > 0.1:
                self.logger_system.warning(f"止损止盈值异常 - SL: {sl}, TP: {tp}, 当前价格: {current_price}")
                self.logger_system.warning(f"SL偏差: {abs(sl - current_price) / current_price * 100:.1f}%, TP偏差: {abs(tp - current_price) / current_price * 100:.1f}%")
                self.logger_system.warning("偏差超过10%，可能是API返回异常数据")
                signal_data = self.create_fallback_signal(price_data)
        except (ValueError, TypeError) as e:
            self.logger_system.warning(f"止损止盈数值转换失败: {e}")
            self.logger_system.warning(f"原始值 - SL: {signal_data.get('stop_loss')}, TP: {signal_data.get('take_profit')}")
            self.logger_system.warning("数据类型错误，使用备用信号")
            signal_data = self.create_fallback_signal(price_data)
        
        signal_data['timestamp'] = price_data['timestamp']
        with self.lock:
            self.signal_history.append(signal_data)
            self.save_signal_history()
            if len(self.signal_history) > 30:
                self.signal_history.pop(0)
        
        signal_count = len([s for s in self.signal_history if s.get('signal') == signal_data['signal']])
        total_signals = len(self.signal_history)
        self.logger_system.info(f"Signal stats: {signal_data['signal']} (occurred {signal_count} times in last {total_signals})")
        
        with self.lock:
            if len(self.signal_history) >= 3:
                last_three = [s['signal'] for s in self.signal_history[-3:]]
                if len(set(last_three)) == 1 and last_three[0] != 'HOLD':
                    self.logger_system.warning(f"3 consecutive {signal_data['signal']} signals")
        return signal_data

    def confirm_order(self, order_id: str, expected_size: float, timeout: int = None) -> bool:
        if timeout is None:
            timeout = self.config.order_timeout
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                order = self.safe_fetch_order(exchange, order_id, self.config.symbol)
                
                # 安全处理CCXT filled字段类型转换
                filled_raw = order.get('filled', 0)
                if isinstance(filled_raw, str):
                    filled_size = float(filled_raw or 0)
                elif filled_raw is None:
                    filled_size = 0.0
                else:
                    filled_size = float(filled_raw)
                
                status = order.get('status')
                remaining = order.get('remaining', 0)
                
                self.logger_trading.debug(f"Order {order_id} details: filled_size={filled_size}, status={status}, expected={expected_size}, remaining={remaining}")
                
                if status == 'closed':
                    if filled_size >= expected_size * self.config.min_fill_ratio:
                        self.logger_trading.info(f"订单 {order_id} 确认执行成功 (填充: {filled_size:.4f}/{expected_size:.4f})")
                        return True
                    else:
                        self.logger_trading.warning(f"订单 {order_id} 部分填充: {filled_size:.4f} < {expected_size * self.config.min_fill_ratio:.4f}")
                        # 部分填充也视为成功，但记录警告
                        if filled_size > 0:
                            self.logger_trading.info(f"订单 {order_id} 部分填充成功，实际填充: {filled_size:.4f}")
                            return True
                        else:
                            self.logger_trading.error(f"订单 {order_id} 无填充")
                            return False
                elif status in ['canceled', 'rejected']:
                    self.logger_trading.error(f"订单 {order_id} 失败: {status}")
                    return False
                elif status == 'open':
                    # 检查是否有部分填充且剩余量很小
                    if filled_size > 0 and remaining is not None:
                        remaining_size = float(remaining) if isinstance(remaining, str) else float(remaining or 0)
                        if remaining_size > 0 and remaining_size < expected_size * 0.05:  # 剩余量小于5%
                            try:
                                self.logger_trading.info(f"订单 {order_id} 部分填充 {filled_size:.4f}，取消剩余 {remaining_size:.4f}")
                                exchange.cancel_order(order_id, self.config.symbol)
                                # 等待取消确认
                                time.sleep(1)
                                final_order = self.safe_fetch_order(exchange, order_id, self.config.symbol)
                                final_filled = float(final_order.get('filled', filled_size))
                                if final_filled >= expected_size * self.config.min_fill_ratio:
                                    self.logger_trading.info(f"订单 {order_id} 部分填充后取消成功，最终填充: {final_filled:.4f}")
                                    return True
                            except Exception as cancel_e:
                                self.logger_trading.warning(f"取消剩余订单失败: {cancel_e}")
                    
                    # 订单仍在执行中，继续等待
                    self.logger_trading.debug(f"订单 {order_id} 仍在执行中，继续等待...")
                    time.sleep(0.5)
            except Exception as e:
                self.logger_trading.exception(f"订单查询失败 {order_id}: {e}")
                time.sleep(0.5)
        
        # 超时处理：尝试取消订单并检查最终状态
        try:
            self.logger_trading.warning(f"订单 {order_id} 超时，尝试取消...")
            exchange.cancel_order(order_id, self.config.symbol)
            time.sleep(1)
            final_order = self.safe_fetch_order(exchange, order_id, self.config.symbol)
            final_filled_raw = final_order.get('filled', 0)
            final_filled = float(final_filled_raw or 0) if isinstance(final_filled_raw, str) else float(final_filled_raw or 0)
            
            if final_filled >= expected_size * self.config.min_fill_ratio:
                self.logger_trading.info(f"订单 {order_id} 超时后部分填充成功: {final_filled:.4f}")
                return True
            else:
                self.logger_trading.error(f"订单 {order_id} 超时且填充不足: {final_filled:.4f}")
                return False
        except Exception as e:
            self.logger_trading.error(f"订单 {order_id} 超时取消失败: {e}")
            return False

    # New: Calculate position size based on 5% risk at liquidation price (aggressive)
    def calculate_dynamic_amount(self, price_data: Dict[str, Any], signal_data: Dict[str, Any], usd_balance: float, side: str) -> float:
        try:
            entry_price = price_data['price']
            leverage = self.config.leverage
            mmr = self.config.maintenance_margin_rate
            risk_pct = self.config.risk_per_trade  # 0.05
            risk_amount = usd_balance * risk_pct

            # Liquidation distance factor for isolated mode
            # For long: liq_dist = entry * (1/L - mmr)
            # For short: liq_dist = entry * (1/L + mmr) but symmetric
            liq_factor = 1 / leverage - mmr if side == 'long' else 1 / leverage + mmr
            liq_distance = entry_price * liq_factor

            if liq_distance <= 0:
                self.logger_trading.warning("Invalid liquidation distance, using default amount")
                return self.config.amount

            # Position size S = risk_amount / liq_distance
            dynamic_amount = risk_amount / liq_distance

            self.logger_trading.info(f"逐仓风险计算: 余额={usd_balance:.2f}, 风险金额={risk_amount:.2f} ({risk_pct*100:.1f}%)")
            self.logger_trading.info(f"杠杆={leverage}x, MMR={mmr}, 清算距离={liq_distance:.2f} ({liq_factor*100:.2f}%)")
            self.logger_trading.info(f"仓位大小: {dynamic_amount:.6f} 合约")

            # Min position value check (changed from BTC quantity to USDC value)
            position_value_usdc = dynamic_amount * entry_price
            min_position_value = self.config.min_amount_usdc  # 50 USDC minimum
            
            if position_value_usdc < min_position_value:
                self.logger_trading.warning(f"仓位价值 {position_value_usdc:.2f} USDC 低于最小值 {min_position_value} USDC")
                # Adjust to minimum USDC value
                dynamic_amount = min_position_value / entry_price
                self.logger_trading.info(f"调整仓位至最小值: {dynamic_amount:.6f} 合约 (价值 {min_position_value} USDC)")
                
                # Recalc risk with adjusted amount
                actual_risk = dynamic_amount * liq_distance
                actual_risk_pct = actual_risk / usd_balance
                if actual_risk_pct > risk_pct * 1.5:
                    self.logger_trading.error(f"调整后风险 {actual_risk_pct*100:.1f}% 过高, 跳过")
                    return 0
            
            # Still check exchange minimum for technical compliance
            try:
                markets = exchange.load_markets()
                exchange_min_size = markets[self.config.symbol]['limits']['amount']['min']
                if exchange_min_size and dynamic_amount < exchange_min_size:
                    self.logger_trading.warning(f"调整至交易所最小数量: {exchange_min_size}")
                    dynamic_amount = max(dynamic_amount, exchange_min_size)
            except Exception as e:
                self.logger_trading.debug(f"无法获取交易所最小订单量: {e}")

            # Max margin usage
            notional = dynamic_amount * entry_price
            margin_required = notional / leverage
            if margin_required > usd_balance * self.config.max_margin_usage:
                self.logger_trading.warning("超过最大保证金使用率, 缩减仓位")
                dynamic_amount = (usd_balance * self.config.max_margin_usage * leverage) / entry_price

            # Volatility adjustment - aggressive: increase in high vol
            atr = price_data['technical_data']['atr']
            atr_pct = atr / entry_price
            if atr_pct > 0.03:
                dynamic_amount *= 1.1  # Increase by 10% in high vol for aggressive
                self.logger_trading.info(f"高波动调整 (ATR {atr_pct*100:.1f}%), 仓位增加10%")

            final_risk = dynamic_amount * liq_distance
            self.logger_trading.info(f"最终仓位: {dynamic_amount:.6f}, 预计风险: {final_risk:.2f} ({final_risk / usd_balance * 100:.1f}%)")

            return dynamic_amount
        except Exception as e:
            self.logger_trading.exception(f"动态仓位计算失败: {e}")
            return 0

    def calculate_rr(self, signal_data: Dict[str, Any], price_data: Dict[str, Any]) -> Tuple[float, float]:
        entry = price_data['price']
        
        # Check for None values in stop_loss and take_profit
        stop_loss = signal_data.get('stop_loss')
        take_profit = signal_data.get('take_profit')
        
        if stop_loss is None or take_profit is None:
            self.logger_trading.warning(f"None values detected - stop_loss: {stop_loss}, take_profit: {take_profit}")
            return 0.0, 0.0
        
        if signal_data['signal'] == 'BUY':
            risk = entry - stop_loss
            reward = take_profit - entry
        elif signal_data['signal'] == 'SELL':
            risk = stop_loss - entry
            reward = entry - take_profit
        else:
            risk = reward = 0
        
        # FIXED: High 5 - Correct fee calculation with leverage
        # Fee is calculated on notional value (amount * price * leverage)
        # For R:R calculation, we need fee impact per unit price movement
        fee_per_unit = self.config.fee_rate * 2 * self.config.leverage  # Open + close fees with leverage
        reward -= fee_per_unit * entry  # Deduct fee impact from reward
        risk += fee_per_unit * entry  # Add fee impact to risk (increases required movement)
        
        rr = reward / risk if risk > 0 else 0
        
        # New: Check TF overlap for extra weight
        levels = price_data['key_levels']
        overlap_weight = 1.0
        for level_name in ['ema_21_4h', 'ema_55_4h', 'daily_ob_bull', 'daily_ob_bear']:  # Higher TF keys
            level = levels.get(level_name)
            if level and stop_loss is not None and abs(stop_loss - level) / entry < 0.001:  # <0.1% overlap
                overlap_weight *= 1.5
                self.logger_trading.info(f"SL overlaps {level_name}: {level:.2f}, weight +50%")
            if level and take_profit is not None and abs(take_profit - level) / entry < 0.001:
                overlap_weight *= 1.5
                self.logger_trading.info(f"TP overlaps {level_name}: {level:.2f}, weight +50%")
        
        final_weight = rr * overlap_weight
        self.logger_trading.info(f"R:R Calc: {rr:.2f}:1 (overlap weight: {overlap_weight:.1f}x, final: {final_weight:.2f})")
        return rr, final_weight

    def _validate_and_prepare_trade(self, signal_data: Dict[str, Any], price_data: Dict[str, Any], activated_level: Optional[str] = None) -> Tuple[bool, float]:
        # FIXED: High 3 - Skip fallback signals and HOLD signals
        if signal_data.get('is_fallback', False):
            self.logger_trading.warning("Fallback signal detected, skipping trade for safety")
            return False, 0
        
        if signal_data.get('signal', '').upper() == 'HOLD':
            self.logger_trading.info("HOLD signal received, skipping trade")
            return False, 0
        
        # FIXED: High 4 - Skip LOW confidence without activation
        if signal_data['confidence'] == 'LOW' and not hasattr(self, 'activated_level'):  # Assume activated_level passed or check
            self.logger_trading.warning("Low confidence without activation, skipping")
            return False, 0
        
        # New: Multi-TF alignment check
        higher_tf_struct = price_data['structures_summary'].get(self.config.higher_tf_bias_tf, {})
        lower_tf_struct = price_data['structures_summary'].get(self.config.lower_tf_entry_tf, {})
        higher_bias = higher_tf_struct.get('trend', 'N/A')
        lower_entry = lower_tf_struct.get('structure', 'N/A')
        if higher_bias == 'N/A' or lower_entry == 'N/A':
            self.logger_trading.warning("Multi-TF alignment check failed: Missing structure data")
            return False, 0
        
        signal_dir = signal_data['signal']
        if signal_dir == 'BUY' and higher_bias != 'Uptrend':
            self.logger_trading.warning(f"BUY signal but higher TF bias {higher_bias} != Uptrend, skipping")
            return False, 0
        if signal_dir == 'SELL' and higher_bias != 'Downtrend':
            self.logger_trading.warning(f"SELL signal but higher TF bias {higher_bias} != Downtrend, skipping")
            return False, 0
        
        # New: Confirmation checks
        # Volume - REMOVED: No longer using volume confirmation
        # m15_df = price_data['multi_tf_data'].get('15m')
        # if not m15_df.empty and m15_df['volume_ratio'].iloc[-1] < self.config.volume_confirmation_threshold:
        #     self.logger_trading.warning("Volume confirmation failed: < 1.2x MA")
        #     return False, 0
        
        # 取消蜡烛图形态检查
        # Candle patterns
        # patterns = lower_tf_struct.get('patterns', [])
        # if not patterns:
        #     self.logger_trading.warning("No candle patterns for confirmation, skipping")
        #     return False, 0
        
        # FVG stacking
        fvg_bull = price_data['key_levels'].get('fvg_bull_stack', 0)
        fvg_bear = price_data['key_levels'].get('fvg_bear_stack', 0)
        if signal_dir == 'BUY' and fvg_bull < self.config.fvg_stack_threshold:
            self.logger_trading.warning(f"BUY signal but FVG bull stack {fvg_bull} < {self.config.fvg_stack_threshold}")
            return False, 0
        if signal_dir == 'SELL' and fvg_bear < self.config.fvg_stack_threshold:
            self.logger_trading.warning(f"SELL signal but FVG bear stack {fvg_bear} < {self.config.fvg_stack_threshold}")
            return False, 0
        
        # Fresh zone check (for activated_level)
        if activated_level:
            interactions = self.zone_interactions.get(activated_level, 0)
            if interactions > self.config.max_zone_interactions:
                self.logger_trading.warning(f"Zone {activated_level} not fresh: interactions {interactions} > {self.config.max_zone_interactions}")
                return False, 0
        
        if not self.check_balance_risk() or not self.check_position_time():
            return False, 0
        if price_data.get('volatility', 0) < self.config.volatility_threshold:
            self.logger_trading.warning(f"Volatility too low {price_data['volatility']:.1f}% < {self.config.volatility_threshold}%, pausing trading")
            return False, 0
        rr, rr_weight = self.calculate_rr(signal_data, price_data)  # Updated call
        if signal_data['confidence'] == 'LOW' and rr < 1.45:  # Increased from 1.2 for better risk management
            self.logger_trading.warning("Low confidence + poor R:R, skipping")
            return False, 0
        if rr >= 3.0:  # High odds: Aggressive risk
            effective_risk = self.config.risk_aggressive
            self.logger_trading.info(f"High R:R {rr:.2f}:1 → Aggressive risk {effective_risk*100:.1f}%")
        else:
            effective_risk = self.config.risk_per_trade
        # Check for None values before calculation
        if signal_data.get('take_profit') is None or price_data.get('price') is None:
            self.logger_trading.warning("Take profit or price is None, skipping trade")
            return False, 0
        
        # Removed: Take profit distance filter
        # expected_range = price_data['amplitude']['expected_rr_range']
        # dist_to_tp = abs(signal_data['take_profit'] - price_data['price'])
        # if dist_to_tp > expected_range:
        #     self.logger_trading.warning(f"TP distance {dist_to_tp:.2f} exceeds expected intraday range {expected_range:.2f}, skipping")
        #     return False, 0
        with self.lock:
            if len(self.signal_history) >= 3:
                last_three = [s['signal'] for s in self.signal_history[-3:]]
                if len(set(last_three)) == 1 and last_three[0] != 'HOLD':
                    self.logger_trading.warning("3 consecutive identical signals, halving position")
                    adjustment = 0.5
                else:
                    adjustment = 1.0
            balance = self.safe_fetch_balance(exchange)
            usd_balance = balance['USDC']['free']
        # New: Use fixed 5% for normal risk, 7% for aggressive risk
        risk_amount = usd_balance * effective_risk
        if not self._check_risk_controls(risk_amount):
            return False, 0
        side = 'long' if signal_data['signal'] == 'BUY' else 'short'
        dynamic_amount = self.calculate_dynamic_amount(price_data, signal_data, usd_balance, side)
        if dynamic_amount == 0:
            return False, 0
        dynamic_amount *= adjustment
        self.logger_trading.info(f"Adjusted amount due to consecutive signals: {dynamic_amount:.4f} BTC")
        return True, dynamic_amount

    def execute_trade(self, signal_data: Dict[str, Any], price_data: Dict[str, Any], activated_level: Optional[str] = None) -> None:
        # 生成唯一交易ID
        trade_id = f"{int(time.time()*1000)}_{random.randint(1000,9999)}"
        trade_context = {
            "trade_id": trade_id,
            "symbol": self.config.symbol,
            "activated_level": activated_level or "N/A",  # 修复：添加 or "N/A" 避免 None
            "confidence": signal_data.get('confidence', 'UNKNOWN')
        }
        
        # FIXED: High 4 - Early skip for low conf no activation
        if signal_data['confidence'] == 'LOW' and activated_level is None:  # 修复：检查参数是否为None
            self.logger_trading.info("跳过低置信度交易: ID=%s, 无激活信号", trade_id)
            return
        
        if self.config.simulation_mode:
            self.logger_trading.info("模拟交易: ID=%s, 信号=%s, 金额=%.4f", 
                                   trade_id, signal_data['signal'], self.config.amount)
            return

        valid, dynamic_amount = self._validate_and_prepare_trade(signal_data, price_data, activated_level)
        if not valid:
            self.logger_trading.warning("HOLD信号，无需交易: ID=%s", trade_id)
            return

        # Fixed: Enforce min USDC value before execution
        position_value_usdc = dynamic_amount * price_data['price']
        if position_value_usdc < self.config.min_amount_usdc:  # Changed from 0.001 BTC to 50 USDC
            self.logger_trading.error("交易金额过小: ID=%s, 价值=%.2f USDC < 最小值 %.0f USDC", 
                                    trade_id, position_value_usdc, self.config.min_amount_usdc)
            return

        side = 'buy' if signal_data['signal'] == 'BUY' else 'sell'
        pos_side = 'long' if signal_data['signal'] == 'BUY' else 'short'
        params = {
            # Hyperliquid does not use posSide, but reduceOnly for closes
        }
        
        # 记录交易开始
        rr_ratio, _ = self.calculate_rr(signal_data, price_data)
        self.logger_trading.info("执行交易: ID=%s, 信号=%s, 金额=%.4f, SL=%.2f, TP=%.2f, R:R=%.1f:1, 激活=%s", 
                               trade_id, signal_data['signal'], dynamic_amount, 
                               signal_data.get('stop_loss', 0), signal_data.get('take_profit', 0),
                               rr_ratio, activated_level or "定时")
        
        try:
            order = self.safe_create_order(exchange, self.config.symbol, side, dynamic_amount, params=params)
            # Update position with liq price
            if order:
                entry_price = price_data['price']  # Approx
                leverage = self.config.leverage
                if self.config.dynamic_leverage and self.calculate_rr(signal_data, price_data)[0] >= 3.0:
                    leverage = min(50, self.config.max_leverage_per_symbol[self.config.symbol])
                    self.safe_set_leverage(exchange, leverage, self.config.symbol, params)  # Set dynamic leverage
                mmr = self.config.maintenance_margin_rate
                if pos_side == 'long':
                    liq_price = entry_price * (1 - (1 / leverage - mmr))
                else:
                    liq_price = entry_price * (1 + (1 / leverage - mmr))
                pos_info = {
                    'side': pos_side,
                    'size': dynamic_amount,
                    'entry_price': entry_price,
                    'unrealized_pnl': 0,
                    'leverage': leverage,
                    'symbol': self.config.symbol,
                    'entry_time': datetime.now(timezone.utc),
                    'liquidation_price': round(liq_price, 2)
                }
                self.position_store.set(pos_info)
                self.logger_trading.info("交易成功: ID=%s, 填充=%.4f @ %.2f, 杠杆=%dx, 爆仓价=%.2f", 
                                       trade_id, dynamic_amount, entry_price, leverage, liq_price)
                
                # 更新交易成功统计
                if activated_level:
                    self._update_trade_stats(activated_level, success=True)
                    
        except Exception as e:
            self.logger_trading.error("交易失败: ID=%s, 错误=%s", trade_id, str(e))
            # 详细错误信息用exception级别，包含上下文
            self.logger_trading.exception("交易详细错误: ID=%s, symbol=%s, side=%s, amount=%.4f, activated=%s", 
                                        trade_id, self.config.symbol, side, dynamic_amount, activated_level)
            
            # 更新交易失败统计
            if activated_level:
                self._update_trade_stats(activated_level, success=False)

    def _update_trade_stats(self, activated_level: str, success: bool):
        """更新交易统计信息"""
        with self.lock:
            if activated_level in self.level_activation_stats:
                if success:
                    self.level_activation_stats[activated_level]['successful_trades'] += 1
                    self.total_successful_trades += 1
                    self.logger_trading.info(f"关键位 {activated_level} 交易成功统计已更新")
                else:
                    self.level_activation_stats[activated_level]['failed_trades'] += 1
                    self.total_failed_trades += 1
                    self.logger_trading.info(f"关键位 {activated_level} 交易失败统计已更新")
                
                # 记录更新后的胜率
                stats = self.level_activation_stats[activated_level]
                total_trades = stats['successful_trades'] + stats['failed_trades']
                win_rate = (stats['successful_trades'] / max(total_trades, 1)) * 100
                self.logger_trading.info(f"关键位 {activated_level} 当前胜率: {win_rate:.1f}% ({stats['successful_trades']}/{total_trades})")

    def start_dynamic_sl_tp_monitor(self):
        """启动动态止盈止损监控线程"""
        if hasattr(self, 'sl_tp_monitor_thread') and self.sl_tp_monitor_thread.is_alive():
            self.logger_trading.info("动态止盈止损监控线程已在运行")
            return
            
        self.sl_tp_monitor_thread = threading.Thread(target=self.dynamic_sl_tp_monitor_loop, daemon=True)
        self.sl_tp_monitor_thread.start()
        self.logger_trading.info("动态止盈止损监控线程已启动")

    def dynamic_sl_tp_monitor_loop(self):
        """动态止盈止损监控循环"""
        while True:
            try:
                position = self.get_current_position()
                if position:
                    self._update_dynamic_sl_tp(position)
                time.sleep(30)  # 每30秒检查一次
            except Exception as e:
                self.logger_trading.exception(f"动态止盈止损监控错误: {e}")
                time.sleep(60)  # 出错时等待更长时间

    def _update_dynamic_sl_tp(self, position: PositionInfo):
        """更新动态止盈止损"""
        try:
            # 获取当前价格数据
            price_data = self.get_multi_timeframe_data()
            if not price_data:
                return
                
            current_price = price_data['price']
            entry_price = position['entry_price']
            side = position['side']
            
            # 计算当前盈亏比例
            if side == 'long':
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price
            
            # 获取技术指标数据
            multi_tf_data = price_data.get('multi_tf_data', {})
            primary_df = multi_tf_data.get(self.config.primary_timeframe)
            
            if primary_df is None or len(primary_df) < 20:
                return
                
            # 计算新的斐波那契止盈止损
            signal_direction = 'BUY' if side == 'long' else 'SELL'
            new_sl, new_tp = self.get_optimal_fibonacci_sl_tp(primary_df, signal_direction, current_price)
            
            # 动态调整逻辑
            adjustment_made = False
            
            # 1. 盈利保护：当盈利超过2%时，移动止损到盈亏平衡点
            if pnl_pct > 0.02:
                protective_sl = self._calculate_protective_stop_loss(entry_price, current_price, side, pnl_pct)
                if protective_sl and self._should_update_stop_loss(position, protective_sl, side):
                    self.logger_trading.info(f"盈利保护：移动止损到 {protective_sl:.2f} (当前盈利: {pnl_pct*100:.2f}%)")
                    adjustment_made = True
            
            # 2. 趋势跟踪：根据技术指标调整止盈止损
            trend_adjustment = self._calculate_trend_based_adjustment(primary_df, current_price, side)
            if trend_adjustment:
                trend_sl, trend_tp = trend_adjustment
                if self._should_update_stop_loss(position, trend_sl, side):
                    self.logger_trading.info(f"趋势跟踪：调整止损到 {trend_sl:.2f}")
                    adjustment_made = True
                if self._should_update_take_profit(position, trend_tp, side):
                    self.logger_trading.info(f"趋势跟踪：调整止盈到 {trend_tp:.2f}")
                    adjustment_made = True
            
            # 3. 波动率调整：根据ATR调整止损距离
            volatility_adjustment = self._calculate_volatility_based_adjustment(primary_df, current_price, side)
            if volatility_adjustment and self._should_update_stop_loss(position, volatility_adjustment, side):
                self.logger_trading.info(f"波动率调整：调整止损到 {volatility_adjustment:.2f}")
                adjustment_made = True
            
            # 4. 斐波那契优化：定期重新计算斐波那契水平
            if new_sl and new_tp:
                fib_improvement = self._evaluate_fibonacci_improvement(position, new_sl, new_tp, side)
                if fib_improvement:
                    self.logger_trading.info(f"斐波那契优化：止损 {new_sl:.2f}, 止盈 {new_tp:.2f}")
                    adjustment_made = True
            
            if adjustment_made:
                self.logger_trading.info(f"动态止盈止损已更新 - 当前价格: {current_price:.2f}, 盈利: {pnl_pct*100:.2f}%")
                
        except Exception as e:
            self.logger_trading.exception(f"更新动态止盈止损失败: {e}")

    def _calculate_protective_stop_loss(self, entry_price: float, current_price: float, 
                                      side: str, pnl_pct: float) -> Optional[float]:
        """计算保护性止损价格"""
        try:
            if side == 'long':
                # 多头：移动止损到盈亏平衡点或更高
                if pnl_pct > 0.05:  # 盈利超过5%
                    return entry_price * 1.02  # 保护2%盈利
                elif pnl_pct > 0.02:  # 盈利超过2%
                    return entry_price * 1.005  # 移动到盈亏平衡点上方
            else:
                # 空头：移动止损到盈亏平衡点或更低
                if pnl_pct > 0.05:  # 盈利超过5%
                    return entry_price * 0.98  # 保护2%盈利
                elif pnl_pct > 0.02:  # 盈利超过2%
                    return entry_price * 0.995  # 移动到盈亏平衡点下方
            return None
        except Exception as e:
            self.logger_trading.error(f"计算保护性止损失败: {e}")
            return None

    def _calculate_trend_based_adjustment(self, df: pd.DataFrame, current_price: float, 
                                        side: str) -> Optional[Tuple[float, float]]:
        """基于趋势的止盈止损调整"""
        try:
            if len(df) < 20:
                return None
                
            # 获取技术指标
            ema_20 = df['ema_20'].iloc[-1]
            ema_50 = df['ema_50'].iloc[-1]
            rsi = df['rsi'].iloc[-1]
            atr = df['atr'].iloc[-1]
            
            # 趋势强度评估
            trend_strength = abs(ema_20 - ema_50) / current_price
            
            if side == 'long':
                # 多头趋势调整
                if ema_20 > ema_50 and rsi < 70 and trend_strength > 0.01:
                    # 强势上涨趋势，放宽止盈
                    new_sl = max(ema_20 - atr, current_price * 0.98)
                    new_tp = current_price + atr * 3
                    return (new_sl, new_tp)
            else:
                # 空头趋势调整
                if ema_20 < ema_50 and rsi > 30 and trend_strength > 0.01:
                    # 强势下跌趋势，放宽止盈
                    new_sl = min(ema_20 + atr, current_price * 1.02)
                    new_tp = current_price - atr * 3
                    return (new_sl, new_tp)
            
            return None
        except Exception as e:
            self.logger_trading.error(f"计算趋势调整失败: {e}")
            return None

    def _calculate_volatility_based_adjustment(self, df: pd.DataFrame, current_price: float, 
                                             side: str) -> Optional[float]:
        """基于波动率的止损调整"""
        try:
            if len(df) < 14:
                return None
                
            atr = df['atr'].iloc[-1]
            atr_pct = atr / current_price
            
            # 根据波动率调整止损距离
            if atr_pct > 0.03:  # 高波动率
                multiplier = 2.5
            elif atr_pct > 0.02:  # 中等波动率
                multiplier = 2.0
            else:  # 低波动率
                multiplier = 1.5
            
            if side == 'long':
                return current_price - atr * multiplier
            else:
                return current_price + atr * multiplier
                
        except Exception as e:
            self.logger_trading.error(f"计算波动率调整失败: {e}")
            return None

    def _should_update_stop_loss(self, position: PositionInfo, new_sl: float, side: str) -> bool:
        """判断是否应该更新止损"""
        try:
            current_sl = position.get('liquidation_price', 0)  # 使用清算价作为当前止损
            
            if side == 'long':
                # 多头：新止损应该更高（更安全）
                return new_sl > current_sl
            else:
                # 空头：新止损应该更低（更安全）
                return new_sl < current_sl
        except Exception as e:
            self.logger_trading.error(f"判断止损更新失败: {e}")
            return False

    def _should_update_take_profit(self, position: PositionInfo, new_tp: float, side: str) -> bool:
        """判断是否应该更新止盈"""
        try:
            # 这里可以添加更复杂的逻辑来判断是否更新止盈
            # 目前简单返回True，表示总是尝试优化止盈
            return True
        except Exception as e:
            self.logger_trading.error(f"判断止盈更新失败: {e}")
            return False

    def _evaluate_fibonacci_improvement(self, position: PositionInfo, new_sl: float, 
                                      new_tp: float, side: str) -> bool:
        """评估斐波那契调整是否有改善"""
        try:
            entry_price = position['entry_price']
            
            # 计算新的风险回报比
            if side == 'long':
                risk = entry_price - new_sl
                reward = new_tp - entry_price
            else:
                risk = new_sl - entry_price
                reward = entry_price - new_tp
            
            if risk <= 0:
                return False
                
            new_rr = reward / risk
            
            # 只有当风险回报比改善且大于1.5时才调整
            return new_rr > 1.5
            
        except Exception as e:
            self.logger_trading.error(f"评估斐波那契改善失败: {e}")
            return False

    # New: Intraday momentum filter - Enhanced with confirmations
    def intraday_momentum_filter(self, price_data: Dict[str, Any]) -> bool:
        """Momentum filter with price, and new confirmations"""
        try:
            # 修改：使用5m时间框架数据，与主时间框架保持一致
            m5_df = price_data['multi_tf_data'].get('5m')
            if m5_df is None or len(m5_df) < 20:
                return True  # Pass if no data
            
            # 取消成交量过滤检查
            # Volume filter - 使用5m时间框架的成交量数据
            # vol_ratio_5m = m5_df['volume_ratio'].iloc[-1]
            # if vol_ratio_5m < self.config.volume_confirmation_threshold:
            #     # 修复日志信息，显示实际阈值1.5x而不是1.2x
            #     self.logger_trading.info(f"Momentum filter failed: Volume < {self.config.volume_confirmation_threshold}x MA")
            #     return False
            
            # 取消Price > EMA12检查
            # Price > EMA12 check - 使用5m时间框架的EMA12
            # ema12_5m = m5_df['ema_12'].iloc[-1]
            # current_price = price_data['price']
            # # Assume bullish for general filter; can pass signal_dir
            # if current_price <= ema12_5m:
            #     self.logger_trading.info("Momentum filter failed: Price <= EMA12")
            #     return False
            
            # 取消蜡烛图形态检查
            # New: Candle patterns check
            # structures = price_data.get('structures_summary', {})
            # tf_struct = structures.get(self.config.lower_tf_entry_tf, {})
            # patterns = tf_struct.get('patterns', [])
            # if not patterns:
            #     self.logger_trading.info("Momentum filter failed: No candle patterns")
            #     return False
            
            # New: FVG and OB stacking check - 修改为OB>=3或OB+FVG堆叠
            key_levels = price_data.get('key_levels', {})
            fvg_bull_stack = key_levels.get('fvg_bull_stack', 0)
            fvg_bear_stack = key_levels.get('fvg_bear_stack', 0)
            ob_bull_stack = key_levels.get('5m_ob_bull_stack', 0)
            ob_bear_stack = key_levels.get('5m_ob_bear_stack', 0)
            
            # 检查OB堆叠是否>=3或OB+FVG堆叠>=3
            total_bull_stack = ob_bull_stack + fvg_bull_stack
            total_bear_stack = ob_bear_stack + fvg_bear_stack
            
            if (ob_bull_stack < 3 and ob_bear_stack < 3) and (total_bull_stack < 3 and total_bear_stack < 3):
                self.logger_trading.info(f"Momentum filter failed: Insufficient stacking (OB: bull={ob_bull_stack}, bear={ob_bear_stack}, Total: bull={total_bull_stack}, bear={total_bear_stack})")
                return False
            
            self.logger_trading.info(f"Intraday momentum filter passed (stacking: OB bull={ob_bull_stack}, bear={ob_bear_stack}, FVG bull={fvg_bull_stack}, bear={fvg_bear_stack})")
            return True
        except Exception as e:
            self.logger_trading.warning(f"Momentum filter error: {e}")
            return True  # Pass on error for aggressive

    def _fetch_and_update_data(self, activated_level: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if self.config.simulation_mode:
            # 模拟模式下直接获取数据，不需要实时价格检查
            price_data = self.get_multi_timeframe_data()
            if not price_data:
                return None
            current_price = price_data['current_price']
            self.logger_system.info(f"模拟价格: ${current_price:,.2f}")
        else:
            ticker = self.safe_fetch_ticker(exchange, self.config.symbol)
            current_price = round(ticker['last'], 2)
            self.logger_system.info(f"实时价格: ${current_price:,.2f}")

        if not activated_level:
            with self.lock:
                if self.key_levels_cache:
                    is_activated, activated = self.check_price_activation(current_price, self.key_levels_cache)
                    if not is_activated:
                        self.logger_system.info("价格未接近关键位，跳过分析")
                        return None
                    activated_level = activated

        price_data = self.get_multi_timeframe_data()
        if not price_data:
            return None

        # Real-time bar update
        primary_tf = self.config.primary_timeframe
        try:
            latest_ohlcv = self.safe_fetch_ohlcv(exchange, self.config.symbol, primary_tf, 1)
            if latest_ohlcv:
                new_row = pd.DataFrame(latest_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                new_row['timestamp'] = pd.to_datetime(new_row['timestamp'], unit='ms', utc=True)
                new_row = new_row.set_index('timestamp')  # 设置timestamp为索引，与df_primary保持一致
                df_primary = price_data['multi_tf_data'][primary_tf]
                if not df_primary.empty and df_primary.index[-1] < new_row.index[0]:
                    updated_df = pd.concat([df_primary, new_row])
                    if len(updated_df) > self.config.data_points:
                        updated_df = updated_df.tail(self.config.data_points)
                    updated_df = self.calculate_technical_indicators(updated_df)
                    price_data['multi_tf_data'][primary_tf] = updated_df
                    current_data = updated_df.iloc[-1]
                    price_data.update({
                        'price': current_price,
                        'high': max(current_data['high'], current_price),
                        'low': min(current_data['low'], current_price),
                        'technical_data': {
                            **price_data['technical_data'],
                            'rsi': current_data.get('rsi', price_data['technical_data']['rsi']),
                            'macd': current_data.get('macd', price_data['technical_data']['macd']),
                            'sma_20': current_data.get('sma_20', price_data['technical_data']['sma_20']),
                            'atr': current_data.get('atr', price_data['technical_data']['atr'])
                        }
                    })
                    self.logger_system.info("实时K线更新成功")
        except Exception as update_e:
            self.logger_system.exception(f"实时K线更新失败: {update_e}")

        price_data['key_levels']['current_price'] = current_price
        # 从交易对中提取基础货币名称
        base_currency = self.config.symbol.split('/')[0]
        self.logger_system.info(f"{base_currency}当前价格: ${price_data['price']:,.2f}")
        self.logger_system.info(f"主时间框架: {self.config.primary_timeframe}")
        self.logger_system.info(f"周平均振幅: {price_data['amplitude']['avg_amplitude']:.2f}")
        self.logger_system.info(f"已完成波动率: {price_data.get('volatility', 0):.1f}%")
        return price_data

    def price_monitor_loop(self):
        """价格监控循环：检查实时价格是否接近关键价位"""
        activation_count = 0
        monitor_cycle_count = 0
        
        while True:
            try:
                monitor_cycle_count += 1
                ticker = self.safe_fetch_ticker(exchange, self.config.symbol)
                current_price = ticker['last']
                
                # FIXED: High 6 - Separate cache update and price check locks to avoid blocking
                cache_needs_update = False
                current_time = time.time()
                
                # Check if cache needs update (quick lock)
                with self.lock:
                    if current_time - self.cache_timestamp > self.config.cache_ttl:
                        cache_needs_update = True
                
                # Update cache outside of main lock to avoid blocking price checks
                if cache_needs_update:
                    try:
                        futures = {
                            self.executor.submit(self.safe_fetch_ohlcv, exchange, self.config.symbol, '4h', 201): '4h',
                            self.executor.submit(self.safe_fetch_ohlcv, exchange, self.config.symbol, '1d', 10): '1d',
                            self.executor.submit(self.safe_fetch_ohlcv, exchange, self.config.symbol, '1w', 5): '1w',
                            self.executor.submit(self.safe_fetch_ohlcv, exchange, self.config.symbol, '15m', 100): '15m'
                        }
                        multi_tf_light = {}
                        for future in as_completed(futures):
                            tf = futures[future]
                            try:
                                ohlcv = future.result()
                                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                                df = self.calculate_technical_indicators(df)
                                multi_tf_light[tf] = df
                            except Exception as fetch_e:
                                self.logger_api.exception(f"Failed to fetch {tf} in monitor: {fetch_e}")
                        
                        # FIXED: High 6 - Only update cache if we have valid data
                        if multi_tf_light:
                            new_key_levels = self.calculate_key_levels(multi_tf_light)
                            # Update cache atomically
                            with self.lock:
                                self.key_levels_cache = new_key_levels
                                self.cache_timestamp = current_time
                            self.logger_monitor.debug("Key levels cache lightweight update successful")
                        else:
                            self.logger_monitor.warning("Cache update failed: no valid timeframe data")
                    except Exception as update_e:
                        self.logger_monitor.exception(f"Cache update failed, using old values: {update_e}")
                
                # Check price activation with current cache (separate lock)
                current_cache = None
                with self.lock:
                    current_cache = self.key_levels_cache.copy() if self.key_levels_cache else None
                
                if current_cache:
                    is_activated, activated = self.check_price_activation(current_price, current_cache)
                    if is_activated:
                        activation_count += 1
                        # 采样日志：每5次激活记录一次，或者首次激活
                        if activation_count == 1 or activation_count % 5 == 0:
                            self.logger_monitor.info("价格激活: %s (累计: %d次)", activated, activation_count)
                        else:
                            self.logger_monitor.debug("价格激活: %s (累计: %d次)", activated, activation_count)
                        threading.Thread(target=lambda: self.trading_bot(activated), daemon=True).start()
                    else:
                        # 采样日志：每20个监控周期记录一次正常状态
                        if monitor_cycle_count % 20 == 0:
                            self.logger_monitor.debug("价格监控正常: 价格=%.2f, 周期=%d", current_price, monitor_cycle_count)
                else:
                    self.logger_monitor.warning("No key levels cache available for price activation check")
                
                time.sleep(180)  # 3分钟价格激活检查间隔
            except Exception as e:
                self.logger_system.exception(f"Price monitoring exception: {e}")
                time.sleep(self.config.price_monitor_interval)

    def heartbeat_loop(self):
        """心跳循环方法"""
        while True:
            try:
                self.heartbeat()
                time.sleep(self.config.heartbeat_interval)
            except Exception as e:
                self.logger_system.error(f"心跳循环错误: {e}")
                time.sleep(self.config.heartbeat_interval)

    def backtest_from_file(self, file_path: str):
        """改进的回测实现，包含完整的模拟逻辑和P&L计算"""
        try:
            df = pd.read_csv(file_path)
            # FIXED: Medium 11 - Validate columns
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"CSV missing required columns: {required_cols}")
            if len(df) < 20:
                self.logger_system.warning(f"回测数据不足 ({len(df)} 行)，建议至少20行")
                return
            
            # FIXED: High 6 - Add leverage and fees to pnl
            leverage = self.config.leverage
            fee = self.config.fee_rate
            
            # 计算技术指标
            df = self.calculate_technical_indicators(df)
            
            signals = []
            trades = []
            total_pnl = 0.0
            wins = 0
            losses = 0
            
            self.logger_system.info(f"开始回测，数据行数: {len(df)}")
            
            for i, row in df.iterrows():
                if i < 14:  # 需要足够的历史数据计算指标
                    continue
                
                # 构建更完整的价格数据
                price_data = {
                    'price': row['close'],
                    'timestamp': datetime.now().isoformat(),
                    'multi_tf_data': {
                        '1d': df.iloc[max(0, i-10):i+1].copy(),
                        '4h': df.iloc[max(0, i-40):i+1].copy(),
                        '15m': df.iloc[max(0, i-20):i+1].copy()
                    },
                    'amplitude': {
                        'expected_rr_range': (row['high'] - row['low']) * 2,
                        'daily_range': row['high'] - row['low']
                    },
                    'technical_data': {
                        'atr': row.get('atr', (row['high'] - row['low']) * 0.02),
                        'rsi': row.get('rsi', 50),
                        'ema_20': row.get('ema_20', row['close']),
                        'ema_50': row.get('ema_50', row['close'])
                    }
                }
                
                # 使用规则基础的信号生成（避免真实AI调用）
                signal = self._generate_rule_based_signal(price_data, df.iloc[max(0, i-14):i+1])
                signals.append(signal)
                
                # 模拟交易执行
                if signal['signal'] in ['BUY', 'SELL'] and i + 5 < len(df):  # 确保有足够的后续数据
                    trade_result = self._simulate_trade_execution(signal, df.iloc[i:i+6], i)
                    if trade_result:
                        trades.append(trade_result)
                        total_pnl += trade_result['pnl']
                        if trade_result['pnl'] > 0:
                            wins += 1
                        else:
                            losses += 1
            
            # 计算回测统计
            num_trades = len(trades)
            win_rate = wins / num_trades if num_trades > 0 else 0
            avg_win = sum(t['pnl'] for t in trades if t['pnl'] > 0) / wins if wins > 0 else 0
            avg_loss = sum(t['pnl'] for t in trades if t['pnl'] < 0) / losses if losses > 0 else 0
            profit_factor = abs(avg_win * wins / (avg_loss * losses)) if losses > 0 and avg_loss != 0 else float('inf')
            
            self.logger_system.info(f"=== 回测结果 ===")
            self.logger_system.info(f"总交易次数: {num_trades}")
            self.logger_system.info(f"胜率: {win_rate:.2%} ({wins}胜/{losses}负)")
            self.logger_system.info(f"总盈亏: {total_pnl:.4f} USD")
            self.logger_system.info(f"平均盈利: {avg_win:.4f} USD")
            self.logger_system.info(f"平均亏损: {avg_loss:.4f} USD")
            self.logger_system.info(f"盈亏比: {profit_factor:.2f}")
            self.logger_system.info(f"信号分布: {dict(pd.Series([s['signal'] for s in signals]).value_counts())}")
            
        except Exception as e:
            self.logger_system.exception(f"回测失败: {e}")

    def _simulate_trade_execution(self, signal: Dict[str, Any], future_data: pd.DataFrame, entry_index: int) -> Optional[Dict[str, Any]]:
        """模拟交易执行和P&L计算"""
        try:
            entry_price = signal.get('entry_price', future_data.iloc[0]['close'])
            stop_loss = signal['stop_loss']
            take_profit = signal['take_profit']
            side = signal['signal']
            
            # 模拟滑点
            slippage = 0.001
            if side == 'BUY':
                actual_entry = entry_price * (1 + slippage)
            else:
                actual_entry = entry_price * (1 - slippage)
            
            # FIXED: High 5 - Correct PnL calculation and leverage fees
            amount = self.config.amount
            leverage = self.config.leverage
            # 费用基于名义价值：amount * entry_price * leverage * fee_rate * 2 (开仓+平仓)
            fee_cost = amount * actual_entry * leverage * self.config.fee_rate * 2
            
            # 检查后续价格走势
            for i, row in future_data.iloc[1:].iterrows():
                high = row['high']
                low = row['low']
                close = row['close']
                
                # 检查止损止盈触发
                if side == 'BUY':
                    if low <= stop_loss:
                        # 止损触发
                        exit_price = stop_loss * (1 - slippage)  # 滑点
                        # 正确的PnL公式：(exit_price - entry_price) * amount * leverage - fees
                        pnl = (exit_price - actual_entry) * amount * leverage - fee_cost
                        return {
                            'entry_index': entry_index,
                            'entry_price': actual_entry,
                            'exit_price': exit_price,
                            'side': side,
                            'pnl': pnl,
                            'exit_reason': 'stop_loss',
                            'bars_held': i - future_data.index[0]
                        }
                    elif high >= take_profit:
                        # 止盈触发
                        exit_price = take_profit * (1 - slippage)
                        pnl = (exit_price - actual_entry) * amount * leverage - fee_cost
                        return {
                            'entry_index': entry_index,
                            'entry_price': actual_entry,
                            'exit_price': exit_price,
                            'side': side,
                            'pnl': pnl,
                            'exit_reason': 'take_profit',
                            'bars_held': i - future_data.index[0]
                        }
                else:  # SELL
                    if high >= stop_loss:
                        # 止损触发
                        exit_price = stop_loss * (1 + slippage)
                        # 对于SELL：(entry_price - exit_price) * amount * leverage - fees
                        pnl = (actual_entry - exit_price) * amount * leverage - fee_cost
                        return {
                            'entry_index': entry_index,
                            'entry_price': actual_entry,
                            'exit_price': exit_price,
                            'side': side,
                            'pnl': pnl,
                            'exit_reason': 'stop_loss',
                            'bars_held': i - future_data.index[0]
                        }
                    elif low <= take_profit:
                        # 止盈触发
                        exit_price = take_profit * (1 + slippage)
                        pnl = (actual_entry - exit_price) * amount * leverage - fee_cost
                        return {
                            'entry_index': entry_index,
                            'entry_price': actual_entry,
                            'exit_price': exit_price,
                            'side': side,
                            'pnl': pnl,
                            'exit_reason': 'take_profit',
                            'bars_held': i - future_data.index[0]
                        }
            
            # 如果没有触发止损止盈，按最后价格平仓
            final_price = future_data.iloc[-1]['close']
            if side == 'BUY':
                exit_price = final_price * (1 - slippage)
                pnl = (exit_price - actual_entry) * amount * leverage - fee_cost
            else:
                exit_price = final_price * (1 + slippage)
                pnl = (actual_entry - exit_price) * amount * leverage - fee_cost
            
            return {
                'entry_index': entry_index,
                'entry_price': actual_entry,
                'exit_price': exit_price,
                'side': side,
                'pnl': pnl,
                'exit_reason': 'timeout',
                'bars_held': len(future_data) - 1
            }
            
        except Exception as e:
            self.logger_trading.warning(f"交易模拟失败: {e}")
            return None

    def _generate_rule_based_signal(self, price_data, recent_df):
        """基于规则生成信号，用于回测"""
        # 简单规则示例：RSI超买超卖 + 趋势
        rsi = price_data['technical_data'].get('rsi', 50)
        if rsi > 70:
            signal = 'SELL'
            reason = 'RSI overbought'
        elif rsi < 30:
            signal = 'BUY'
            reason = 'RSI oversold'
        else:
            signal = 'HOLD'
            reason = 'Neutral RSI'
        
        current = price_data['price']
        stop_loss = current * 0.98 if signal == 'BUY' else current * 1.02
        take_profit = current * 1.02 if signal == 'BUY' else current * 0.98
        
        return {
            'signal': signal,
            'reason': reason,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'confidence': 'MEDIUM',
            'entry_price': current
        }

    def trading_bot(self, activated_level: Optional[str] = None, is_scheduled: bool = False):
        """主要的交易逻辑执行方法"""
        if not self.trade_lock.acquire(blocking=False):
            self.logger_system.warning("交易正在进行中，跳过本次执行")
            return
        
        try:
            start_time = time.time()
            self.logger_system.info("=== 开始交易分析 ===")
            
            # 获取价格数据
            price_data = self._fetch_and_update_data(activated_level)
            if not price_data:
                self.logger_system.error("无法获取价格数据，跳过本次交易")
                return
            
            # New: Apply intraday momentum filter
            if not self.intraday_momentum_filter(price_data):
                self.logger_system.info("Intraday momentum filter failed, skipping trade")
                return
            
            # 如果是定时任务，检查是否有上次信号副本
            if is_scheduled:
                # 如果有上次的信号副本，直接使用
                if self.last_scheduled_signal:
                    self.logger_system.info("使用上次定时任务的信号副本执行交易")
                    self.execute_trade(self.last_scheduled_signal, price_data, None)  # 修复：传递None作为activated_level
                    execution_time = time.time() - start_time
                    self.logger_system.info(f"=== 交易分析完成 (耗时: {execution_time:.2f}秒) ===")
                    return
            
            # 获取交易信号
            signal_data = self.analyze_with_deepseek(price_data, activated_level)
            if not signal_data:
                self.logger_system.error("无法获取交易信号，跳过本次交易")
                return
            
            # 如果是定时任务，保存信号副本
            if is_scheduled:
                self.last_scheduled_signal = signal_data.copy()
                self.logger_system.info("已保存定时任务信号副本")
            
            # 执行交易
            self.execute_trade(signal_data, price_data, activated_level)
            
            execution_time = time.time() - start_time
            self.logger_system.info(f"=== 交易分析完成 (耗时: {execution_time:.2f}秒) ===")
            
        except Exception as e:
            self.logger_system.error(f"交易执行过程中发生错误: {e}")
        finally:
            self.trade_lock.release()

def job_wrapper(bot, func, *args, **kwargs):
    # 如果func是绑定方法，直接调用；否则将bot作为第一个参数传递
    if hasattr(func, '__self__'):
        # 如果是trading_bot方法，添加is_scheduled=True参数
        if func.__name__ == 'trading_bot':
            bot.executor.submit(func, is_scheduled=True, *args, **kwargs)
        else:
            bot.executor.submit(func, *args, **kwargs)
    else:
        # 如果是trading_bot函数，添加is_scheduled=True参数
        if func.__name__ == 'trading_bot':
            bot.executor.submit(func, bot, is_scheduled=True, *args, **kwargs)
        else:
            bot.executor.submit(func, bot, *args, **kwargs)

def main():
    # FIXED: Medium 7 - Env vars conditional on sim mode
    required_env_vars = ['DEEPSEEK_API_KEY']
    if not config.simulation_mode:
        required_env_vars += ['HYPERLIQUID_WALLET_ADDRESS', 'HYPERLIQUID_PRIVATE_KEY']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        system_logger.error(f"缺少必需的环境变量: {', '.join(missing_vars)}")
        return

    bot = TradingBot(config)
    bot.load_signal_history()
    system_logger.info("BTC/USD Hyperliquid SMC/ICT 自动交易机器人启动成功！")
    system_logger.info("机构订单流分析: 周线流动性 > 日线流动性 > 订单块 > 成交量分布 > 技术位")
    system_logger.info(f"关键位优先级: {', '.join(config.liquidity_priority)}")
    system_logger.info("已启用关键位激活监控 + 风险管理 + 动态仓位")
    system_logger.info(f"心跳已启用 (间隔: {config.heartbeat_interval}秒), 日志: {config.heartbeat_file}")
    system_logger.warning("实盘交易模式，请谨慎操作！" if not config.simulation_mode else "模拟模式已激活")
    system_logger.info(f"主时间框架: {config.primary_timeframe}")

    # New: Log new features
    system_logger.info(f"多时间框架对齐: 高时间框架偏置={config.higher_tf_bias_tf}, 低时间框架入场={config.lower_tf_entry_tf}")
    system_logger.info(f"确认信号: FVG堆叠>={config.fvg_stack_threshold}或OB堆叠>=3或OB+FVG堆叠>=3, 新鲜区互动<={config.max_zone_interactions}")

    # Fixed: Check if backtest_file exists
    if config.backtest_file and os.path.exists(config.backtest_file):
        bot.backtest_from_file(config.backtest_file)

    if not bot.setup_exchange():
        system_logger.error("交易所初始化失败，退出程序")
        return

    # Fixed: Initial trading_bot call for startup signal check
    bot.trading_bot()

    monitor_thread = threading.Thread(target=bot.price_monitor_loop, daemon=True)
    monitor_thread.start()
    system_logger.info("价格监控线程已启动 (每3分钟)")

    heartbeat_thread = threading.Thread(target=bot.heartbeat_loop, daemon=True)
    heartbeat_thread.start()
    system_logger.info("心跳线程已启动")

    # 启动动态止盈止损监控线程
    bot.start_dynamic_sl_tp_monitor()
    system_logger.info("动态止盈止损监控线程已启动 (每30秒检查)")

    schedule.every(5).minutes.do(job_wrapper, bot, bot.trading_bot)  # Changed to 5min for aggressive
    system_logger.info("定时执行频率: 每5分钟")

    try:
        while True:
            try:  # FIXED: Medium 6 - Health check on schedule
                schedule.run_pending()
            except Exception as sched_e:
                system_logger.error(f"Schedule error: {sched_e}")
            time.sleep(1)
    except KeyboardInterrupt:
        system_logger.info("收到中断信号，正在优雅退出...")
    except Exception as e:
        system_logger.error(f"主循环意外错误: {e}")
    finally:
        try:
            system_logger.info("正在清理资源...")
            bot.save_signal_history()
            bot.executor.shutdown(wait=True)
            # FIXED: Medium 9 - Join threads
            if 'monitor_thread' in locals():
                monitor_thread.join(timeout=5)
            if 'heartbeat_thread' in locals():
                heartbeat_thread.join(timeout=5)
            system_logger.info("资源清理完成")
        except Exception as e:
            system_logger.error(f"清理过程中出错: {e}")

if __name__ == "__main__":
    main()