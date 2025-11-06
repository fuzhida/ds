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

# SMC/ICT结构识别库导入
try:
    import smartmoneyconcepts.smc as smc
    SMC_AVAILABLE = True
except ImportError:
    SMC_AVAILABLE = False
    logging.warning("smartmoneyconcepts库未安装，SMC结构识别功能将使用备用实现")

# FIXED: SSL 3 - 禁用 urllib3 SSL 警告（生产中可选移除）
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load environment variables from 1.env file (contains all API keys)
load_dotenv('1.env')

# FIXED: SSL 4 - 自定义 SSL 上下文，处理 EOF 错误
def create_ssl_context():
    ctx = ssl.create_default_context()
    ctx.check_hostname = True  # 保持安全性
    ctx.verify_mode = ssl.CERT_REQUIRED
    # 设置更宽松的协议版本以提高兼容性
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    return ctx

# Note: Custom modules commented out as they are not available in current environment
# from coindesk_websocket_indicators import CoinDeskWebSocketIndicatorProvider, CoinDeskIndicatorConfig
# from hyperliquid_websocket_backup import WebSocketIndicatorProvider as HyperliquidWebSocketProvider, IndicatorConfig as HyperliquidIndicatorConfig, HyperliquidBackupProvider
# from hyperliquid_market_data import HyperliquidMarketData

def setup_logging(log_file: str = 'trading_bot.log', level: str = 'INFO', enable_json: bool = False):
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
loggers = setup_logging('trading_bot.log', 'INFO')
logger = logging.getLogger(__name__)  # Maintain backward compatibility

@dataclass
class Config:
    """Configuration class for trading bot parameters."""
    symbol: str = 'ETH/USDC:USDC'
    amount: float = 0.01
    # Data source configuration
    data_source: str = 'websocket'  # 'websocket' or 'hyperliquid'
    use_websocket_indicators: bool = True  # Use WebSocket for real-time indicators
    leverage: int = 25
    timeframes: List[str] = None
    primary_timeframe: str = '15m'
    structure_confirm_timeframe: str = '1h'
    data_points: int = 200
    amplitude_lookback: int = 7
    activation_threshold: float = 0.0002  # 0.02% - 高敏感度激活阈值，更精确的价格触发
    min_balance_ratio: float = 0.95
    max_position_time: int = 86400
    risk_per_trade: float = 0.015  # 1.5% per trade loss at liquidation price
    slippage_buffer: float = 0.0005  # Reduced from 0.001 for aggressive execution (0.05%)
    volatility_threshold: float = 70
    order_timeout: int = 10
    heartbeat_interval: int = 60
    price_monitor_interval: int = 180  # 3分钟监控间隔，更及时捕捉价格变动
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
    higher_tf_bias_tf: str = '4h'  # Higher TF for bias (e.g., 4h or 1d)
    lower_tf_entry_tf: str = '15m'  # Lower TF for entry
    volume_confirmation_threshold: float = 1.2  # Volume > 1.2x MA
    max_zone_interactions: int = 1  # Max interactions for fresh zones
    fvg_stack_threshold: int = 2  # Min FVGs stacking for confirmation
    candle_pattern_weight: float = 1.5  # Weight for candle pattern confirmation
    # FIXED: Kill Zone 1 - 添加 Kill Zone 配置（可选全天测试）
    kill_zone_start_utc: int = 8  # UTC 开始小时
    kill_zone_end_utc: int = 16   # UTC 结束小时
    enable_kill_zone: bool = False  # 暂时禁用Kill Zone
    # New: Level weights for FVG and OB
    level_weights: Dict[str, float] = None
    # New: SMC结构识别配置
    enable_smc_structures: bool = True  # 启用SMC结构识别
    smc_window: int = 5  # swing检测窗口大小
    smc_range_percent: float = 0.01  # BOS/CHOCH突破阈值
    structure_weights: Dict[str, float] = None  # 结构权重配置
    min_structure_score: float = 0.6  # 最小结构评分阈值
    mtf_consensus_threshold: float = 0.5  # 多时间框架一致性阈值
    
    # NEW: Signal optimization parameters
    signal_stabilizer_window: int = 300  # Signal stabilizer sampling window in seconds (5 minutes)
    trend_consistency_threshold: float = 0.7  # Minimum trend consistency threshold (0.0-1.0)
    enable_signal_fusion: bool = True  # Enable weighted signal fusion
    signal_fusion_weights: Dict[str, float] = None  # Weights for signal fusion components
    enable_duplicate_filtering: bool = True  # Enable duplicate entry prevention
    duplicate_signal_ttl: int = 600  # Duplicate signal TTL in seconds (10 minutes)
    enable_contextual_logging: bool = True  # Enable contextual rejection logging
    contextual_log_file: str = 'contextual_rejections.json'  # File for contextual rejection logs

    def __post_init__(self):
        if self.timeframes is None:
            self.timeframes = ['1d', '4h', '1h', '15m']
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
        if self.structure_weights is None:
            self.structure_weights = {
                'bos_choch': 0.4,      # BOS/CHOCH趋势确认权重
                'ob_fvg': 0.3,         # 订单块/公平价值缺口权重
                'swing_strength': 0.2, # swing点强度权重
                'liquidity': 0.1       # 流动性权重
            }
        self.liquidity_priority = [
            # Daily level (highest priority)
            'daily_fvg_bull_mid', 'daily_fvg_bear_mid', 'daily_ob_bull', 'daily_ob_bear',
            'prev_week_high', 'prev_week_low', 'daily_vwap', 'monday_open', 'daily_open',
            'recent_10d_high', 'recent_10d_low',
            # 4H level (high priority)
            '4h_high', '4h_low', '4h_fvg_bull_mid', '4h_fvg_bear_mid', '4h_ob_bull', '4h_ob_bear', '4h_gap_up', '4h_gap_down',
                # 1H level (medium priority)
                'ema_21_1h', 'ema_55_1h', 'ema_100_1h', 'ema_200_1h', '1h_fvg_bull_mid', '1h_fvg_bear_mid', '1h_ob_bull', '1h_ob_bear',
                # 15m level (structure confirmation)
                '15m_structure_break', '15m_structure_reversal', '15m_liquidity_hunt', '15m_fvg_bull_mid', '15m_fvg_bear_mid', '15m_ob_bull', '15m_ob_bear'
            ]
        if self.max_leverage_per_symbol is None:
            self.max_leverage_per_symbol = {
                'HYPE/USDC:USDC': 10,  # HYPE最大杠杆（交易所限制）
                'BTC/USDC:USDC': 40,
                'ETH/USDC:USDC': 25,
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
            }
        self.validate()

    def validate(self):
        if not (1 <= self.leverage <= 125):
            raise ValueError(f"Leverage must be between 1-125, got: {self.leverage}")
        if not (0.001 <= self.risk_per_trade <= 0.05):
            raise ValueError(f"Risk per trade must be 0.1%-5%, got: {self.risk_per_trade*100:.1f}%")
        if self.amount < 0.01:
            raise ValueError(f"Amount must be >=0.01 BTC, got: {self.amount}")
        if not (0.0001 <= self.activation_threshold <= 0.05):
            raise ValueError(f"Activation threshold must be 0.01%-5%, got: {self.activation_threshold*100:.2f}%")
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
        # FIXED: Kill Zone 2 - 验证 Kill Zone
        if not (0 <= self.kill_zone_start_utc < 24 and 0 <= self.kill_zone_end_utc < 24):
            raise ValueError(f"Kill Zone hours must be 0-23, got start={self.kill_zone_start_utc}, end={self.kill_zone_end_utc}")
        
        # NEW: Initialize signal fusion weights if not provided
        if self.signal_fusion_weights is None:
            self.signal_fusion_weights = {
                'ai_analysis': 0.4,      # DeepSeek AI analysis weight
                'smc_structure': 0.3,    # SMC structure analysis weight
                'momentum': 0.2,         # Momentum-based signals weight
                'fallback': 0.1          # Fallback signals weight
            }
        
        # NEW: Validate signal optimization parameters
        if not (60 <= self.signal_stabilizer_window <= 1800):  # 1-30 minutes
            raise ValueError(f"Signal stabilizer window must be 60-1800 seconds, got: {self.signal_stabilizer_window}")
        if not (0.1 <= self.trend_consistency_threshold <= 1.0):
            raise ValueError(f"Trend consistency threshold must be 0.1-1.0, got: {self.trend_consistency_threshold}")
        if not (60 <= self.duplicate_signal_ttl <= 3600):  # 1-60 minutes
            raise ValueError(f"Duplicate signal TTL must be 60-3600 seconds, got: {self.duplicate_signal_ttl}")
        # Validate signal fusion weights sum to 1.0
        total_weight = sum(self.signal_fusion_weights.values())
        if abs(total_weight - 1.0) > 0.01:  # Allow small rounding errors
            raise ValueError(f"Signal fusion weights must sum to 1.0, got: {total_weight:.3f}")

# NEW: Signal Priority Enum for opposite trigger handling
class SignalPriority(Enum):
    """Signal priority levels for handling opposite triggers"""
    AI_ANALYSIS = 4      # Highest priority: DeepSeek AI analysis
    SMC_STRUCTURE = 3    # SMC structure analysis
    MOMENTUM = 2         # Momentum-based signals
    FALLBACK = 1         # Fallback signals (RSI-based)
    HOLD = 0             # Lowest priority: Hold signals

# NEW: Signal Stabilizer for handling time desync and signal conflicts
class SignalStabilizer:
    """Stabilizes signals to handle time desync interference and opposite triggers"""
    
    def __init__(self, sampling_window_seconds: int = 300, trend_consistency_threshold: float = 0.7):
        self.sampling_window_seconds = sampling_window_seconds
        self.trend_consistency_threshold = trend_consistency_threshold
        self.signal_history: List[Dict[str, Any]] = []
        self.logger = logging.getLogger('system')
    
    def add_signal(self, signal_data: Dict[str, Any], priority: SignalPriority, source: str):
        """Add a new signal to the stabilizer"""
        signal_entry = {
            'timestamp': time.time(),
            'signal': signal_data['signal'],
            'priority': priority,
            'source': source,
            'confidence': signal_data.get('confidence', 'MEDIUM'),
            'entry_price': signal_data.get('entry_price', 0),
            'reason': signal_data.get('reason', ''),
            'data': signal_data
        }
        
        self.signal_history.append(signal_entry)
        
        # Clean old signals outside sampling window
        cutoff_time = time.time() - self.sampling_window_seconds
        self.signal_history = [
            sig for sig in self.signal_history 
            if sig['timestamp'] > cutoff_time
        ]
        
        self.logger.debug(f"Added signal: {signal_data['signal']} from {source} with priority {priority.name}")
    
    def get_consolidated_signal(self) -> Optional[Dict[str, Any]]:
        """Get the consolidated signal based on priority and consistency"""
        if not self.signal_history:
            return None
        
        # Group signals by type
        buy_signals = [sig for sig in self.signal_history if sig['signal'] == 'BUY']
        sell_signals = [sig for sig in self.signal_history if sig['signal'] == 'SELL']
        hold_signals = [sig for sig in self.signal_history if sig['signal'] == 'HOLD']
        
        # If no actionable signals, return None
        if not buy_signals and not sell_signals:
            return None
        
        # Determine dominant signal based on priority and recency
        if buy_signals and sell_signals:
            # Handle opposite triggers - use priority and recency
            return self._resolve_opposite_signals(buy_signals, sell_signals)
        elif buy_signals:
            return self._get_best_signal(buy_signals)
        elif sell_signals:
            return self._get_best_signal(sell_signals)
        else:
            return None
    
    def _resolve_opposite_signals(self, buy_signals: List[Dict], sell_signals: List[Dict]) -> Optional[Dict[str, Any]]:
        """Resolve opposite signals using priority and recency"""
        
        # Get highest priority signals
        best_buy = self._get_best_signal(buy_signals)
        best_sell = self._get_best_signal(sell_signals)
        
        if not best_buy or not best_sell:
            return best_buy or best_sell
        
        # Compare priorities
        buy_priority = best_buy['priority']
        sell_priority = best_sell['priority']
        
        if buy_priority.value > sell_priority.value:
            self.logger.info(f"Resolved opposite signals: BUY wins (priority {buy_priority.name} > {sell_priority.name})")
            return best_buy
        elif sell_priority.value > buy_priority.value:
            self.logger.info(f"Resolved opposite signals: SELL wins (priority {sell_priority.name} > {buy_priority.name})")
            return best_sell
        else:
            # Same priority - use recency (latest signal wins)
            buy_time = best_buy['timestamp']
            sell_time = best_sell['timestamp']
            
            if buy_time >= sell_time:
                self.logger.info(f"Resolved opposite signals: BUY wins (same priority, more recent)")
                return best_buy
            else:
                self.logger.info(f"Resolved opposite signals: SELL wins (same priority, more recent)")
                return best_sell
    
    def _get_best_signal(self, signals: List[Dict]) -> Optional[Dict[str, Any]]:
        """Get the best signal from a list based on priority and recency"""
        if not signals:
            return None
        
        # Sort by priority (descending) and timestamp (descending)
        signals.sort(key=lambda x: (x['priority'].value, x['timestamp']), reverse=True)
        
        return signals[0]
    
    def get_trend_consistency(self, signal_type: str) -> float:
        """Calculate trend consistency for a specific signal type"""
        relevant_signals = [
            sig for sig in self.signal_history 
            if sig['signal'] == signal_type
        ]
        
        if not relevant_signals:
            return 0.0
        
        total_signals = len(self.signal_history)
        consistent_signals = len(relevant_signals)
        
        return consistent_signals / total_signals if total_signals > 0 else 0.0
    
    def should_filter_signal(self, signal_data: Dict[str, Any], priority: SignalPriority) -> bool:
        """Determine if a signal should be filtered based on consistency"""
        signal_type = signal_data['signal']
        consistency = self.get_trend_consistency(signal_type)
        
        should_filter = consistency < self.trend_consistency_threshold
        
        if should_filter:
            self.logger.info(f"Filtering {signal_type} signal: consistency {consistency:.2f} < threshold {self.trend_consistency_threshold}")
        
        return should_filter

# 全局变量声明，但不在模块级别初始化
config = None
deepseek_client = None
exchange = None
system_logger = logging.getLogger('system')

def _display_startup_parameters(config):
    """显示启动时的关键参数和逻辑条件"""
    system_logger.info("=" * 80)
    system_logger.info("🚀 DeepSeek AI 交易机器人启动参数报告")
    system_logger.info("=" * 80)
    
    # 基础交易参数
    system_logger.info("📊 基础交易参数:")
    system_logger.info(f"   交易对: {config.symbol}")
    system_logger.info(f"   杠杆倍数: {config.leverage}x")
    system_logger.info(f"   基础交易量: {config.amount:.4f} ETH")
    system_logger.info(f"   运行模式: {'🔴 实盘模式' if not config.simulation_mode else '🟡 模拟模式'}")
    
    # 关键激活参数
    system_logger.info("🎯 价格激活参数:")
    system_logger.info(f"   激活阈值: {config.activation_threshold*100:.3f}% (价格接近关键水平的触发距离)")
    system_logger.info(f"   主要时间框架: {config.primary_timeframe}")
    system_logger.info(f"   确认时间框架: {config.structure_confirm_timeframe}")
    system_logger.info(f"   监控间隔: {config.price_monitor_interval}秒")
    
    # Kill Zone 设置
    system_logger.info("⏰ Kill Zone 设置:")
    if config.enable_kill_zone:
        system_logger.info(f"   状态: 🟢 启用")
        system_logger.info(f"   交易时间: UTC {config.kill_zone_start_utc}:00 - {config.kill_zone_end_utc}:00")
    else:
        system_logger.info(f"   状态: 🔴 禁用 (全天候交易)")
    
    # 风险管理参数
    system_logger.info("🛡️ 风险管理参数:")
    system_logger.info(f"   每笔交易风险: {config.risk_per_trade*100:.1f}%")
    system_logger.info(f"   最大保证金使用: {config.max_margin_usage*100:.0f}%")
    system_logger.info(f"   最大日亏损: {config.max_daily_loss_pct*100:.0f}%")
    system_logger.info(f"   最大回撤: {config.max_drawdown_pct*100:.0f}%")
    system_logger.info(f"   最小持仓金额: ${config.min_amount_usdc:.0f} USDC")
    
    # 风险回报比设置
    system_logger.info("📈 风险回报比设置:")
    system_logger.info(f"   最小R:R比例: {config.rr_min_threshold:.1f}:1")
    system_logger.info(f"   激进R:R比例: {config.rr_aggressive_threshold:.1f}:1")
    system_logger.info(f"   激进模式风险: {config.risk_aggressive*100:.1f}%")
    
    # 技术分析参数
    system_logger.info("📊 技术分析参数:")
    system_logger.info(f"   成交量确认阈值: {config.volume_confirmation_threshold:.1f}x MA")
    system_logger.info(f"   FVG堆叠要求: {config.fvg_stack_threshold}个")
    system_logger.info(f"   新鲜区域最大交互: {config.max_zone_interactions}次")
    system_logger.info(f"   蜡烛图形权重: {config.candle_pattern_weight:.1f}x")
    
    # AI 参数
    system_logger.info("🤖 AI 分析参数:")
    system_logger.info(f"   DeepSeek 温度: {config.temperature}")
    system_logger.info(f"   超时时间: {config.deepseek_timeout}秒")
    
    # 监控参数
    system_logger.info("📡 监控参数:")
    system_logger.info(f"   心跳间隔: {config.heartbeat_interval}秒")
    system_logger.info(f"   价格监控间隔: {config.price_monitor_interval}秒")
    
    # 逻辑条件总结
    system_logger.info("🧠 关键逻辑条件:")
    system_logger.info("   1. 价格必须接近关键水平 (激活阈值内)")
    if config.enable_kill_zone:
        system_logger.info(f"   2. 必须在Kill Zone时间内 (UTC {config.kill_zone_start_utc}-{config.kill_zone_end_utc})")
    else:
        system_logger.info("   2. Kill Zone已禁用，全天候交易")
    system_logger.info("   3. 风险回报比必须满足最小要求")
    system_logger.info("   4. 账户余额必须充足")
    system_logger.info("   5. 无现有持仓冲突")
    system_logger.info("   6. 技术指标确认信号")
    
    system_logger.info("=" * 80)

def initialize_globals():
    """初始化全局配置和客户端，避免重复初始化"""
    global config, deepseek_client, exchange
    
    if config is not None:
        return  # 已经初始化过了
    
    config = Config()
    
    # 显示详细的启动参数报告
    _display_startup_parameters(config)
    
    # Use system logger to record config validation
    system_logger.info("Config validation successful: %s | Leverage=%dx | Risk=%.1f%% | Mode=%s", 
                       config.symbol, config.leverage, config.risk_per_trade*100, 
                       "Simulation" if config.simulation_mode else "Live")

    # Initialize DeepSeek client with error handling
    try:
        deepseek_client = OpenAI(
            api_key=os.getenv('DEEPSEEK_API_KEY'),
            base_url="https://api.deepseek.com/v1",
            timeout=config.deepseek_timeout
        )
        system_logger.info("DeepSeek client initialized successfully")
    except Exception as e:
        system_logger.error(f"Failed to initialize DeepSeek client: {e}")
        deepseek_client = None

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
                    # Use system logger for retry info
                    system_logger = logging.getLogger('system')
                    system_logger.warning(f"Call to {func.__name__} failed (attempt {tries}/{retries}), waiting {sleep:.2f}s: {e}")
                    time.sleep(sleep)
        return wrapper
    return deco

# FIXED: Data Fetch 1 - 添加带重试的 session，用于 CoinDesk 请求
def create_session_with_retry():
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=5, pool_maxsize=5)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    # FIXED: SSL 5 - 应用自定义 SSL 上下文 (注意：requests.Session.verify 应该是布尔值或证书路径)
    session.verify = True  # 启用SSL验证
    return session

class TradingBot:
    def __init__(self, config: Config, exchange=None):
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
        # New: Track zone interactions for freshness
        self.zone_interactions: Dict[str, int] = {}  # Count interactions per zone
        self.last_scheduled_signal: Optional[Dict[str, Any]] = None  # Store last scheduled signal copy
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
        # API health status tracking
        self.api_health_status = {
            'deepseek': {'status': 'unknown', 'last_check': 0, 'consecutive_failures': 0},
            'hyperliquid': {'status': 'unknown', 'last_check': 0, 'consecutive_failures': 0},
            'websocket': {'status': 'unknown', 'last_check': 0, 'consecutive_failures': 0}
        }
        
        # FIXED: Data Fetch 3 - 创建带重试的 session
        self.session = create_session_with_retry()
        
        # Initialize CoinDesk WebSocket API (primary) and Hyperliquid backup providers
        self.coindesk_provider = None
        self.hyperliquid_websocket_backup = None
        self.hyperliquid_backup = None
        
        # Initialize data providers (currently disabled due to missing modules)
        self.coindesk_provider = None
        self.hyperliquid_websocket_backup = None
        self.hyperliquid_backup = None
        self.hyperliquid_market_data = None
        
        # Note: Custom data providers are disabled as modules are not available
        self.logger_system.warning("Custom data providers (CoinDesk, Hyperliquid WebSocket) are disabled - modules not available")
        self.logger_system.info("Using fallback to exchange API for data fetching")
        
        # Data source priority explanation
        self.logger_system.info("Data source: Exchange API (Hyperliquid) - primary and only available source")
        
        self.api_health_check_interval = 300  # Check every 5 minutes
        # Threshold accumulation and win rate stats
        self.level_activation_stats: Dict[str, Dict[str, int]] = {}  # Activation and success stats per key level
        self.total_activations: int = 0  # Total activations
        self.total_successful_trades: int = 0  # Total successful trades
        self.total_failed_trades: int = 0  # Total failed trades
        
        # Log counters (for sampling)
        self.log_counters = {
            'price_activation': 0,
            'api_health_check': 0,
            'heartbeat': 0
        }
        
        # Perform initial API health check
        self._perform_initial_api_health_check()
        # New: Dynamic SL/TP monitor thread
        self.sl_tp_monitor_thread = None
        
        # NEW: Signal stabilizer for handling time desync and opposite triggers
        self.signal_stabilizer = SignalStabilizer(
            sampling_window_seconds=config.signal_stabilizer_window,
            trend_consistency_threshold=config.trend_consistency_threshold
        )
        
        # NEW: Duplicate signal tracking for entry prevention
        self.signal_hashes: Set[str] = set()
        self.signal_hash_timestamps: Dict[str, float] = {}
        
        # NEW: Contextual rejection logging
        self.contextual_rejections: List[Dict[str, Any]] = []
        
        # NEW: Track last market data for contextual logging
        self.last_volatility: float = 0.0
        self.last_rsi: float = 50.0
        self.last_price: float = 0.0

    @retry_on_exception(retries=4)
    def safe_create_order(self, exchange, symbol, side, amount, params=None):
        if self.config.simulation_mode:
            # Simulation mode returns simulated order
            order_id = f"sim_{int(time.time() * 1000)}"
            ticker = self.safe_fetch_ticker(exchange, symbol)
            price = ticker['last'] if ticker else 67000.0  # Use default price as backup
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
            self.logger_trading.info(f"Simulated order placed successfully, ID: {order_id}")
            return order
            
        try:
            # FIXED: High 1 & 6 - Use create_order for full params support; confirm after slippage
            # 获取当前市场价格作为备用价格
            current_price = None
            try:
                ticker = self.safe_fetch_ticker(exchange, symbol)
                current_price = ticker.get('last') if ticker else None
            except Exception as price_e:
                self.logger_trading.warning(f"Failed to fetch current price: {price_e}")
            
            # 使用市场价格，如果price为None的话
            order_price = current_price if current_price is not None else 0.0
            self.logger_trading.debug(f"Using order price: {order_price}")
            
            # 确保order_price不为None且大于0，避免ccxt hyperliquid的price_to_precision方法出现NoneType错误
            if order_price is None or order_price <= 0:
                self.logger_trading.error(f"Invalid order price: {order_price}, using fallback price 4000.0")
                order_price = 4000.0  # ETH的合理默认价格
            
            order = exchange.create_order(symbol, side, 'market', amount, order_price, params=params)
            self.logger_trading.info(f"Order placed successfully, ID: {order.get('id', 'N/A')}")  # FIXED: Medium 3 - Use debug for ID?
            self.logger_trading.debug(f"Order ID: {order.get('id', 'N/A')}")  # Safer
            # FIXED: High 6 - Confirm after any order
            if hasattr(self, 'confirm_order'):  # Ensure method exists
                if not self.confirm_order(order['id'], amount):
                    raise ExchangeError("Order confirmation failed")
            return order
        except ExchangeError as e:
            if "slippage" in str(e).lower():
                self.logger_trading.warning(f"Slippage error in order: {e}")
            raise

    # FIXED: Data Fetch 4 - 增强 safe_fetch_ohlcv，添加 session 和 TF 填充
    @retry_on_exception(retries=4)
    def safe_fetch_ohlcv(self, exchange, symbol, timeframe, limit=100, session=None):
        """Enhanced OHLCV fetch with fallback data filling"""
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if ohlcv and len(ohlcv) >= min(20, limit):  # FIXED: Data 1 - 验证数据完整性
                return ohlcv
            else:
                self.logger_api.warning(f"Insufficient data for {timeframe} ({len(ohlcv) if ohlcv else 0} rows), no fallback available")
                return []  # 返回空数据
        except Exception as e:
            self.logger_api.error(f"Fetch OHLCV failed for {timeframe}: {e}")
            return []

    def safe_fetch_ticker(self, exchange, symbol):
        try:
            return exchange.fetch_ticker(symbol)
        except Exception as e:
            self.logger_api.warning(f"Ticker fetch failed for {symbol}: {e}")
            return {'last': 67000.0}  # Fallback price

    def _perform_initial_api_health_check(self):
        self.logger_api.info("Performing initial API health check...")
        # DeepSeek check
        try:
            if deepseek_client is None:
                raise Exception("DeepSeek client not initialized")
            
            test_prompt = "test"
            
            # 记录健康检查的提示词
            self.logger_api.info("🔍 API健康检查 - 发送测试提示词:")
            self.logger_api.info(f"   提示词: '{test_prompt}'")
            self.logger_api.info(f"   模型: deepseek-chat")
            self.logger_api.info(f"   最大tokens: 10")
            self.logger_api.info(f"   温度: {config.temperature}")
                
            response = deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": test_prompt}],
                max_tokens=10,
                temperature=config.temperature
            )
            
            # 记录健康检查的响应
            response_text = response.choices[0].message.content.strip()
            self.logger_api.info("✅ API健康检查 - 收到响应:")
            self.logger_api.info(f"   响应内容: '{response_text}'")
            
            self.api_health_status['deepseek']['status'] = 'healthy'
            self.api_health_status['deepseek']['last_check'] = time.time()
            self.api_health_status['deepseek']['consecutive_failures'] = 0
            self.logger_api.info("DeepSeek API healthy")
        except Exception as e:
            self.api_health_status['deepseek']['status'] = 'unhealthy'
            self.api_health_status['deepseek']['consecutive_failures'] += 1
            self.logger_api.error(f"DeepSeek API health check failed: {e}")

    def setup_exchange(self):
        try:
            self.logger_trading.info("Setting up exchange...")
            # Set leverage
            leverage_result = exchange.set_leverage(config.leverage, config.symbol)
            self.logger_trading.info(f"Leverage set result: {leverage_result}")
            # Verify leverage
            positions = exchange.fetch_positions([config.symbol])
            if positions:
                actual_leverage = positions[0].get('leverage', config.leverage)
                self.logger_trading.info(f"Leverage verification successful: Expected {config.leverage}x, actual {actual_leverage}x")
            else:
                self.logger_trading.debug("No active position found - leverage verification skipped")
            self.logger_trading.info(f"Set leverage: {config.leverage}x")
            # Fetch balance
            balance = exchange.fetch_balance()
            usdc_balance = balance.get('USDC', {}).get('free', 0)
            self.logger_system.info(f"Current USD balance: {usdc_balance:.2f}")
            self.initial_balance = usdc_balance
            return True
        except Exception as e:
            self.logger_trading.error(f"Exchange setup failed: {e}")
            return False

    def load_signal_history(self):
        try:
            if os.path.exists(config.signals_file):
                with open(config.signals_file, 'r') as f:
                    self.signal_history = json.load(f)
                self.logger_system.info(f"Loaded signal history: {len(self.signal_history)} records")
            else:
                self.signal_history = []
        except Exception as e:
            self.logger_system.error(f"Failed to load signal history: {e}")
            self.signal_history = []

    def save_signal_history(self):
        try:
            with open(config.signals_file, 'w') as f:
                json.dump(self.signal_history, f, indent=2)
            self.logger_system.debug("Signal history saved successfully")
        except Exception as e:
            self.logger_system.error(f"Failed to save signal history: {e}")

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return df
        # Basic indicators (expand as needed)
        df['sma_20'] = df['close'].rolling(20).mean()
        df['ema_20'] = df['close'].ewm(span=20).mean()
        df['rsi'] = self._rsi(df['close'], 14)
        df['atr'] = self._atr(df, 14)
        
        # 添加MACD指标
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        return df

    def _rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        return tr.rolling(period).mean()

    def calculate_key_levels(self, multi_tf_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        key_levels = {}
        # Simplified key level calculation (expand based on liquidity_priority)
        for tf, df in multi_tf_data.items():
            if not df.empty:
                key_levels[f'{tf}_high'] = df['high'].max()
                key_levels[f'{tf}_low'] = df['low'].min()
                key_levels[f'{tf}_open'] = df['open'].iloc[-1]
                key_levels[f'{tf}_close'] = df['close'].iloc[-1]
                
                # 添加移动平均线作为关键水平
                if 'ema_20' in df.columns:
                    key_levels[f'{tf}_ema_20'] = df['ema_20'].iloc[-1]
                if 'sma_20' in df.columns:
                    key_levels[f'{tf}_sma_20'] = df['sma_20'].iloc[-1]
                
                # 计算前一周期的最高最低值
                if len(df) > 1:
                    key_levels[f'prev_{tf}_high'] = df['high'].iloc[-2]
                    key_levels[f'prev_{tf}_low'] = df['low'].iloc[-2]
        
        return key_levels

    def check_price_activation(self, current_price: float, key_levels: Dict[str, float]) -> Tuple[bool, Optional[str]]:
        """智能价格激活检查，支持多层次激活阈值"""
        if not key_levels:
            return False, None
            
        closest_level = None
        closest_distance = float('inf')
        closest_level_name = None
        
        # 找到最接近的关键水平
        for level_name, level_price in key_levels.items():
            if level_price > 0:  # 确保价格有效
                distance = abs(current_price - level_price) / level_price
                if distance < closest_distance:
                    closest_distance = distance
                    closest_level = level_price
                    closest_level_name = level_name
        
        # 使用动态阈值：基础阈值 + 波动性调整
        base_threshold = self.config.activation_threshold
        
        # 如果距离在基础阈值内，直接激活
        if closest_distance <= base_threshold:
            self.logger_system.debug(f"价格激活: {closest_level_name} (距离: {closest_distance*100:.3f}%)")
            return True, closest_level_name
        
        # 如果距离在扩展阈值内（2倍基础阈值），且满足其他条件，也可以激活
        extended_threshold = base_threshold * 2
        if closest_distance <= extended_threshold:
            # 检查是否在交易时间内
            now_utc = datetime.now(timezone.utc).hour
            if self.config.enable_kill_zone and (self.config.kill_zone_start_utc <= now_utc <= self.config.kill_zone_end_utc):
                self.logger_system.debug(f"扩展激活: {closest_level_name} (距离: {closest_distance*100:.3f}%, 在交易时间内)")
                return True, closest_level_name
        
        # 记录最接近的水平（降低日志频率）
        if hasattr(self, '_last_closest_log_time'):
            if time.time() - self._last_closest_log_time > 300:  # 每5分钟记录一次
                self.logger_system.debug(f"最接近关键水平: {closest_level_name} @ ${closest_level:.2f} (距离: {closest_distance*100:.3f}%)")
                self._last_closest_log_time = time.time()
        else:
            self._last_closest_log_time = time.time()
        
        return False, None

    def _fetch_and_update_data(self, activated_level: Optional[str] = None):
        # Fetch multi-TF data using enhanced safe_fetch_ohlcv
        multi_tf_data = {}
        failed_timeframes = []
        successful_timeframes = []
        
        self.logger_system.info(f"开始获取多时间框架数据: {config.timeframes}")
        
        for tf in config.timeframes:
            try:
                ohlcv = self.safe_fetch_ohlcv(self.exchange, config.symbol, tf, config.data_points)
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
        success_rate = len(successful_timeframes) / len(config.timeframes) * 100
        self.logger_system.info(f"数据获取完成: 成功 {len(successful_timeframes)}/{len(config.timeframes)} ({success_rate:.1f}%)")
        
        if successful_timeframes:
            self.logger_system.info(f"成功获取: {', '.join(successful_timeframes)}")
        if failed_timeframes:
            self.logger_system.warning(f"获取失败: {', '.join(failed_timeframes)}")

        if not multi_tf_data:
            self.logger_system.error("所有时间框架数据获取失败，无法继续分析")
            return None

        # 获取当前价格（容错处理）
        try:
            ticker = self.safe_fetch_ticker(self.exchange, config.symbol)
            current_price = ticker['last'] if ticker and 'last' in ticker else None
            if current_price is None:
                # 如果无法获取ticker，尝试从最新的OHLCV数据获取
                if multi_tf_data is not None and isinstance(multi_tf_data, dict) and config.primary_timeframe in multi_tf_data:
                    current_price = multi_tf_data[config.primary_timeframe]['close'].iloc[-1]
                    self.logger_system.warning(f"从{config.primary_timeframe}数据获取当前价格: ${current_price:.2f}")
                else:
                    current_price = 4000.0  # ETH的合理默认价格
                    self.logger_system.error(f"无法获取当前价格，使用默认值: ${current_price:.2f}")
            else:
                self.logger_system.debug(f"成功获取当前价格: ${current_price:.2f}")
        except Exception as e:
            current_price = 4000.0  # ETH的合理默认价格
            self.logger_system.error(f"价格获取异常: {e}，使用默认值: ${current_price:.2f}")

        # Calculate amplitude (容错处理)
        try:
            if multi_tf_data is not None and isinstance(multi_tf_data, dict) and len(multi_tf_data) > 0:
                amplitudes = []
                for tf, df in multi_tf_data.items():
                    if not df.empty and len(df) > 0:
                        amp = df['high'].max() - df['low'].min()
                        amplitudes.append(amp)
                avg_amplitude = np.mean(amplitudes) if amplitudes else current_price * 0.05
            else:
                avg_amplitude = current_price * 0.05  # 默认5%振幅
            amplitude = {'avg_amplitude': avg_amplitude}
            self.logger_system.debug(f"计算振幅: {avg_amplitude:.2f}")
        except Exception as e:
            amplitude = {'avg_amplitude': current_price * 0.05}
            self.logger_system.warning(f"振幅计算异常: {e}，使用默认值")

        # Update cache (容错处理)
        try:
            with self.lock:
                self.key_levels_cache = self.calculate_key_levels(multi_tf_data)
                self.cache_timestamp = time.time()
            self.logger_system.debug(f"关键水平缓存更新成功: {len(self.key_levels_cache)} 个水平")
        except Exception as e:
            self.logger_system.error(f"关键水平计算异常: {e}")
            # 保持现有缓存或使用空缓存
            if not hasattr(self, 'key_levels_cache') or not self.key_levels_cache:
                self.key_levels_cache = {}

        # Volatility (容错处理)
        try:
            if multi_tf_data is not None and isinstance(multi_tf_data, dict) and len(multi_tf_data) > 0:
                volatilities = []
                for tf, df in multi_tf_data.items():
                    if not df.empty and len(df) > 1:
                        vol = df['close'].pct_change().std() * 100
                        if not np.isnan(vol):
                            volatilities.append(vol)
                volatility = np.std(volatilities) if volatilities else 2.0
            else:
                volatility = 2.0  # 默认波动率
            self.logger_system.debug(f"计算波动率: {volatility:.2f}%")
        except Exception as e:
            volatility = 2.0
            self.logger_system.warning(f"波动率计算异常: {e}，使用默认值")

        # 构建基础价格数据（无论是否激活都需要）
        price_data = {
            'price': current_price,
            'multi_tf_data': multi_tf_data,
            'amplitude': amplitude,
            'volatility': volatility,
            'key_levels': self.key_levels_cache,
            'technical_data': {
                'rsi': multi_tf_data.get(config.primary_timeframe, pd.DataFrame()).iloc[-1].get('rsi', 50) if not multi_tf_data.get(config.primary_timeframe, pd.DataFrame()).empty else 50,
                'macd': multi_tf_data.get(config.primary_timeframe, pd.DataFrame()).iloc[-1].get('macd', 0) if not multi_tf_data.get(config.primary_timeframe, pd.DataFrame()).empty else 0,
                'sma_20': multi_tf_data.get(config.primary_timeframe, pd.DataFrame()).iloc[-1].get('sma_20', current_price) if not multi_tf_data.get(config.primary_timeframe, pd.DataFrame()).empty else current_price,
                'atr': multi_tf_data.get(config.primary_timeframe, pd.DataFrame()).iloc[-1].get('atr', current_price * 0.02) if not multi_tf_data.get(config.primary_timeframe, pd.DataFrame()).empty else current_price * 0.02
            },
            'activated_level': None,  # 初始化激活水平
            'is_activated': False     # 初始化激活状态
        }

        # 价格激活检查（不影响数据返回）
        if not activated_level:
            with self.lock:
                if self.key_levels_cache:
                    is_activated, activated = self.check_price_activation(current_price, self.key_levels_cache)
                    price_data['is_activated'] = is_activated
                    if is_activated:
                        price_data['activated_level'] = activated
                        self.logger_system.info(f"价格激活成功: {activated} (距离: {abs(current_price - self.key_levels_cache.get(activated, current_price)) / current_price * 100:.3f}%)")
                    else:
                        # 降低日志级别，避免频繁的INFO日志
                        self.logger_system.debug("Price not close to key level, will use fallback signal if needed")
        else:
            # 如果提供了activated_level（如测试模式），跳过价格激活检查
            price_data['activated_level'] = activated_level
            price_data['is_activated'] = True
            self.logger_system.debug(f"Using provided activated level: {activated_level}")

        # Real-time bar update (simplified)
        primary_tf = self.config.primary_timeframe
        try:
            latest_ohlcv = self.safe_fetch_ohlcv(self.exchange, self.config.symbol, primary_tf, 1)
            if latest_ohlcv is not None and isinstance(latest_ohlcv, list) and len(latest_ohlcv) > 0:
                new_row = pd.DataFrame(latest_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                new_row['timestamp'] = pd.to_datetime(new_row['timestamp'], unit='ms', utc=True)
                new_row = new_row.set_index('timestamp')
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
                            'rsi': current_data.get('rsi', price_data['technical_data'].get('rsi', 50)),
                            'macd': current_data.get('macd', price_data['technical_data'].get('macd', 0)),
                            'sma_20': current_data.get('sma_20', price_data['technical_data'].get('sma_20', current_price)),
                            'atr': current_data.get('atr', price_data['technical_data'].get('atr', current_price * 0.02))
                        }
                    })
                    self.logger_system.info("Real-time K-line update successful")
        except Exception as update_e:
            self.logger_system.exception(f"Real-time K-line update failed: {update_e}")

        price_data['key_levels']['current_price'] = current_price
        
        # SMC结构分析集成
        if self.config.enable_smc_structures:
            smc_structures = {}
            mtf_analysis = {}
            
            # 检测各时间框架的SMC结构
            for tf, df in multi_tf_data.items():
                if not df.empty:
                    structures = self.detect_smc_structures(df, tf)
                    if structures is not None and isinstance(structures, dict) and len(structures) > 0:
                        smc_structures[tf] = structures
                        # 计算流动性评分
                        liquidity_score = self.calculate_structure_liquidity_score(structures, df)
                        self.logger_system.info(f"{tf} SMC结构分析完成，流动性评分: {liquidity_score:.3f}")
            
            # 多时间框架结构分析
            if smc_structures is not None and isinstance(smc_structures, dict) and len(smc_structures) > 0:
                mtf_analysis = self._mtf_structure_analysis(multi_tf_data)
                self.logger_system.info(f"MTF结构分析: {mtf_analysis.get('recommendation', 'neutral')} (一致性: {mtf_analysis.get('consistency', 0):.2f})")
            
            # 将SMC分析结果添加到价格数据中
            price_data.update({
                'smc_structures': smc_structures,
                'mtf_analysis': mtf_analysis
            })
            
            # 计算更高时间框架CHOCH-BOS失效点和最近关键水平
            if smc_structures and len(smc_structures) > 0:
                higher_tf = config.higher_tf_bias_tf
                primary_tf = config.primary_timeframe
                
                # 获取更高时间框架的结构数据
                higher_tf_structures = smc_structures.get(higher_tf, {})  # FIXED: 修复双花括号语法错误
                primary_tf_structures = smc_structures.get(primary_tf, {})
                
                # 计算更高时间框架CHOCH-BOS失效点
                higher_tf_invalidation = self._calculate_higher_tf_invalidation(
                    higher_tf_structures, 
                    primary_tf_structures, 
                    current_price,
                    multi_tf_data.get(higher_tf, pd.DataFrame()),
                    multi_tf_data.get(primary_tf, pd.DataFrame())
                )
                
                # 计算最近关键水平和距离
                nearest_key_level, key_level_distance = self._calculate_nearest_key_level(
                    current_price, 
                    price_data['key_levels']
                )
                
                # 更新smc_structures以包含新计算的数据
                smc_structures.update({
                    'higher_tf_choch_bos_invalidation': higher_tf_invalidation,
                    'nearest_key_level': nearest_key_level,
                    'key_level_distance': key_level_distance,
                    'structure_score': higher_tf_structures.get('strength_score', 0.5) if higher_tf_structures else 0.5,
                    'fresh_zones': higher_tf_structures.get('fresh_zones', 0) if higher_tf_structures else 0,
                    'bos_choch': higher_tf_structures.get('bos_choch', 'neutral') if higher_tf_structures else 'neutral',
                    'ob_fvg': higher_tf_structures.get('ob_fvg', 'neutral') if higher_tf_structures else 'neutral'
                })
                
                self.logger_system.info(f"更高时间框架失效点计算: {higher_tf_invalidation:.4f}, "
                                      f"最近关键水平: {nearest_key_level:.4f} (距离: {key_level_distance:.4f})")
            else:
                # 如果没有SMC结构，使用默认值
                smc_structures.update({
                    'higher_tf_choch_bos_invalidation': current_price * 0.98,
                    'nearest_key_level': current_price * 0.98,
                    'key_level_distance': 0.02,
                    'structure_score': 0.5,
                    'fresh_zones': 0,
                    'bos_choch': 'neutral',
                    'ob_fvg': 'neutral'
                })
        
        # Store volatility and RSI for contextual logging
        self.last_volatility = price_data.get('volatility', 0)
        self.last_rsi = price_data['technical_data'].get('rsi', 50)
        self.last_price = current_price
        
        # Extract base currency name from trading pair
        if self.config.symbol and '/' in self.config.symbol:
            base_currency = self.config.symbol.split('/')[0]
        else:
            base_currency = 'ETH'  # 默认货币
            self.logger_system.warning(f"Invalid symbol format: {self.config.symbol}, using default: {base_currency}")
        self.logger_system.info(f"{base_currency} current price: ${price_data['price']:,.2f}")
        self.logger_system.info(f"Primary timeframe: {self.config.primary_timeframe}")
        self.logger_system.info(f"Weekly average amplitude: {price_data['amplitude']['avg_amplitude']:.2f}")
        self.logger_system.info(f"Completed volatility: {price_data.get('volatility', 0):.1f}%")
        return price_data

    def detect_smc_structures(self, df: pd.DataFrame, tf: str) -> Dict[str, Any]:
        """自动检测SMC结构，返回量化数据和权重。"""
        if len(df) < 10:  # 最小数据要求
            return {}
        
        try:
            # SMC可用性检查 - 在函数开始处定义
            global SMC_AVAILABLE  # 使用全局变量避免本地变量引用错误
            smc_available = SMC_AVAILABLE
            
            # 初始化变量
            highs_lows = None
            bos_choch = None
            ob = None
            fvg = None
            liq = None
            
            if smc_available is True:
                # 使用smartmoneyconcepts库进行结构识别
                try:
                    # Swing high/low检测
                    highs_lows = smc.swing_highs_lows(df, swing_length=self.config.smc_window)
                    
                    # BOS/CHOCH计算 - 需要传入swing数据
                    if highs_lows is not None and hasattr(highs_lows, 'empty') and hasattr(highs_lows, '__len__') and len(highs_lows) > 0:
                        try:
                            bos_choch = smc.bos_choch(df, highs_lows, close_break=True)
                        except Exception as e:
                            self.logger_system.warning(f"BOS/CHOCH计算失败: {e}，使用空DataFrame")
                            bos_choch = pd.DataFrame()  # 空DataFrame
                    else:
                        bos_choch = pd.DataFrame()  # 空DataFrame
                    
                    # OB/FVG检测
                    try:
                        ob = smc.ob(df, swing_highs_lows=highs_lows)  # Order Blocks
                    except Exception as e:
                        self.logger_system.warning(f"Order Blocks检测失败: {e}，使用空DataFrame")
                        ob = pd.DataFrame()
                    
                    try:
                        fvg = smc.fvg(df)  # Fair Value Gaps (bull/bear)
                    except Exception as e:
                        self.logger_system.warning(f"FVG检测失败: {e}，使用空DataFrame")
                        fvg = pd.DataFrame()
                    
                    # 流动性（作为辅助）
                    try:
                        liq = smc.liquidity(df, swing_highs_lows=highs_lows, range_percent=self.config.smc_range_percent)
                    except Exception as e:
                        self.logger_system.warning(f"流动性检测失败: {e}，使用空DataFrame")
                        liq = pd.DataFrame()
                except Exception as smc_error:
                    self.logger_system.warning(f"smartmoneyconcepts库调用失败: {smc_error}，使用备用实现")
                    smc_available = False  # 更新本地变量
                    SMC_AVAILABLE = False  # 同时更新全局变量
                    # 继续使用备用实现
                    highs_lows = self._manual_highs_lows(df, window=self.config.smc_window)
                    bos_choch = self._manual_bos_choch(df, window=self.config.smc_window)
                    ob = self._manual_order_blocks(df)
                    fvg = self._manual_fvg(df)
                    liq = self._manual_liquidity(df)
                
                # 量化强度计算
                atr = self._atr(df, 14).iloc[-1] if len(df) >= 14 else df['close'].std()
                
                # BOS强度计算
                bos_strength = 0
                if bos_choch is not None and hasattr(bos_choch, 'empty') and not bos_choch.empty and hasattr(bos_choch, '__len__') and len(bos_choch) > 0:
                    last_bos = bos_choch.iloc[-1]
                    if isinstance(last_bos, pd.Series) and last_bos.get('type') == 'BOS':
                        price_change = abs(df['close'].iloc[-1] - last_bos.get('level', df['close'].iloc[-1]))
                        bos_strength = min(price_change / atr, 2.0) if atr > 0 else 0
                
                # FVG深度计算
                fvg_depth = len(fvg) / len(df) if fvg is not None and hasattr(fvg, '__len__') and len(fvg) > 0 else 0
                
                # 结构强度评分
                strength_score = (
                    self.config.structure_weights['bos_choch'] * bos_strength +
                    self.config.structure_weights['ob_fvg'] * fvg_depth +
                    self.config.structure_weights['swing_strength'] * (len(highs_lows) / len(df) if highs_lows is not None and hasattr(highs_lows, '__len__') and len(highs_lows) > 0 else 0)
                )
                
            else:
                # 备用实现：手动计算基础结构
                highs_lows = self._manual_highs_lows(df, window=self.config.smc_window)
                bos_choch = self._manual_bos_choch(df, window=self.config.smc_window)
                ob = self._manual_order_blocks(df)
                fvg = self._manual_fvg(df)
                liq = self._manual_liquidity(df)
                
                # 计算强度评分
                atr = self._atr(df, 14).iloc[-1] if len(df) >= 14 else df['close'].std()
                bos_strength = self._calculate_manual_bos_strength(df, bos_choch, atr)
                fvg_depth = len(fvg) / len(df) if fvg is not None and isinstance(fvg, list) and len(fvg) > 0 else 0
                
                strength_score = (
                    self.config.structure_weights['bos_choch'] * bos_strength +
                    self.config.structure_weights['ob_fvg'] * fvg_depth +
                    self.config.structure_weights['swing_strength'] * (len(highs_lows) / len(df) if highs_lows is not None and isinstance(highs_lows, pd.DataFrame) and len(highs_lows) > 0 else 0)
                )
            
            # 结构区块日志输出
            fvg_count = len(fvg) if fvg is not None and hasattr(fvg, '__len__') and len(fvg) > 0 else 0
            ob_count = len(ob) if ob is not None and hasattr(ob, '__len__') and len(ob) > 0 else 0
            self.logger_system.info(f"{tf} SMC结构: BOS强度={bos_strength:.2f}, FVG数量={fvg_count}, OB区域={ob_count}, 总强度={strength_score:.2f}")
            
            # 安全处理返回值，避免DataFrame布尔错误
            def safe_convert_to_records(data, limit=None):
                """安全转换数据到记录格式"""
                if data is None:
                    return []
                if isinstance(data, pd.DataFrame):
                    records = data.to_dict('records')
                    return records[-limit:] if limit and records else records
                elif isinstance(data, list):
                    return data[-limit:] if limit and data else data
                else:
                    return []
            
            return {
                'swings': safe_convert_to_records(highs_lows, 3),
                'bos_choch': safe_convert_to_records(bos_choch, 2),
                'ob_fvg': {
                    'ob': safe_convert_to_records(ob),
                    'fvg': safe_convert_to_records(fvg)
                },
                'strength_score': strength_score,
                'liq_sweeps': safe_convert_to_records(liq)
            }
        except Exception as e:
            self.logger_system.error(f"SMC结构检测失败 {tf}: {e}")
            return {}
    
    def _manual_highs_lows(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """手动实现swing high/low检测"""
        highs = []
        lows = []
        
        for i in range(window, len(df) - window):
            # 检测swing high
            if all(df['high'].iloc[i] >= df['high'].iloc[i-j] for j in range(1, window+1)) and \
               all(df['high'].iloc[i] >= df['high'].iloc[i+j] for j in range(1, window+1)):
                highs.append({'index': i, 'price': df['high'].iloc[i], 'type': 'swing_high'})
            
            # 检测swing low
            if all(df['low'].iloc[i] <= df['low'].iloc[i-j] for j in range(1, window+1)) and \
               all(df['low'].iloc[i] <= df['low'].iloc[i+j] for j in range(1, window+1)):
                lows.append({'index': i, 'price': df['low'].iloc[i], 'type': 'swing_low'})
        
        return pd.DataFrame(highs + lows)
    
    def _manual_bos_choch(self, df: pd.DataFrame, window: int = 5) -> list:
        """手动实现BOS/CHOCH检测"""
        structures = []
        swing_points = self._manual_highs_lows(df, window)
        
        if swing_points is None or not hasattr(swing_points, '__len__') or len(swing_points) == 0:
            return []
            
        for i in range(1, len(swing_points)):
            current = swing_points.iloc[i]
            previous = swing_points.iloc[i-1]
            
            current_price = df['close'].iloc[-1]
            
            # BOS (Break of Structure) 检测
            if current['type'] == 'swing_high' and previous['type'] == 'swing_high':
                if current['price'] > previous['price'] and current_price > current['price']:
                    structures.append({
                        'type': 'BOS',
                        'direction': 1,
                        'level': current['price'],
                        'strength': abs(current_price - current['price']) / df['close'].std()
                    })
            
            # CHOCH (Change of Character) 检测
            if current['type'] == 'swing_low' and previous['type'] == 'swing_high':
                if current['price'] < previous['price'] and current_price < current['price']:
                    structures.append({
                        'type': 'CHOCH',
                        'direction': -1,
                        'level': current['price'],
                        'strength': abs(current_price - current['price']) / df['close'].std()
                    })
        
        return structures
    
    def _manual_order_blocks(self, df: pd.DataFrame) -> list:
        """手动实现订单块检测"""
        order_blocks = []
        
        for i in range(3, len(df)):
            current_candle = df.iloc[i]
            prev_candle = df.iloc[i-1]
            
            # 看涨订单块：大阳线后出现小阴线
            if (current_candle['close'] > current_candle['open'] and  # 当前阳线
                prev_candle['close'] > prev_candle['open'] and        # 前一根阳线
                (current_candle['close'] - current_candle['open']) > (prev_candle['high'] - prev_candle['low']) * 0.7):  # 大阳线
                
                order_blocks.append({
                    'type': 'bullish_ob',
                    'high': min(current_candle['open'], prev_candle['close']),
                    'low': max(current_candle['open'], prev_candle['close']),
                    'strength': abs(current_candle['close'] - current_candle['open']) / current_candle['open']
                })
            
            # 看跌订单块：大阴线后出现小阳线
            if (current_candle['close'] < current_candle['open'] and  # 当前阴线
                prev_candle['close'] < prev_candle['open'] and        # 前一根阴线
                abs(current_candle['close'] - current_candle['open']) > (prev_candle['high'] - prev_candle['low']) * 0.7):  # 大阴线
                
                order_blocks.append({
                    'type': 'bearish_ob',
                    'high': min(current_candle['open'], prev_candle['close']),
                    'low': max(current_candle['open'], prev_candle['close']),
                    'strength': abs(current_candle['close'] - current_candle['open']) / current_candle['open']
                })
        
        return order_blocks
    
    def _manual_fvg(self, df: pd.DataFrame) -> list:
        """手动实现公平价值缺口检测"""
        fvgs = []
        
        for i in range(2, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            prev2 = df.iloc[i-2]
            
            # 看涨FVG：价格向上跳空
            if (prev['high'] < current['low'] and  # 缺口存在
                prev2['close'] > prev2['open']):     # 前两根是阳线
                
                fvgs.append({
                    'type': 'bullish_fvg',
                    'high': prev['high'],
                    'low': current['low'],
                    'gap_size': current['low'] - prev['high'],
                    'strength': (current['low'] - prev['high']) / prev['close']
                })
            
            # 看跌FVG：价格向下跳空
            if (prev['low'] > current['high'] and  # 缺口存在
                prev2['close'] < prev2['open']):    # 前两根是阴线
                
                fvgs.append({
                    'type': 'bearish_fvg',
                    'high': current['high'],
                    'low': prev['low'],
                    'gap_size': prev['low'] - current['high'],
                    'strength': (prev['low'] - current['high']) / prev['close']
                })
        
        return fvgs
    
    def _manual_liquidity(self, df: pd.DataFrame) -> list:
        """手动实现流动性检测"""
        liquidity_sweeps = []
        
        # 简单实现：检测价格突破前高/低的情况
        for i in range(10, len(df)):
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]
            
            # 检测是否突破前10根K线的最高点（流动性吸收）
            prev_high = df['high'].iloc[i-10:i].max()
            if current_high > prev_high:
                liquidity_sweeps.append({
                    'type': 'high_liquidity_sweep',
                    'swept_level': prev_high,
                    'sweep_price': current_high,
                    'sweep_size': current_high - prev_high,
                    'strength': (current_high - prev_high) / prev_high
                })
            
            # 检测是否突破前10根K线的最低点
            prev_low = df['low'].iloc[i-10:i].min()
            if current_low < prev_low:
                liquidity_sweeps.append({
                    'type': 'low_liquidity_sweep',
                    'swept_level': prev_low,
                    'sweep_price': current_low,
                    'sweep_size': prev_low - current_low,
                    'strength': (prev_low - current_low) / prev_low
                })
        
        return liquidity_sweeps
    
    def _calculate_manual_bos_strength(self, df: pd.DataFrame, bos_choch: list, atr: float) -> float:
        """计算手动BOS强度"""
        if (bos_choch is None or (isinstance(bos_choch, list) and len(bos_choch) == 0)) or atr <= 0:
            return 0
        
        current_price = df['close'].iloc[-1]
        last_bos = bos_choch[-1]
        
        if last_bos.get('type') == 'BOS':
            price_change = abs(current_price - last_bos.get('level', current_price))
            return min(price_change / atr, 2.0)
        
        return 0
    
    def _mtf_structure_analysis(self, multi_tf_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """多时间框架结构分析：大级别偏置 * 小级别确认"""
        if not self.config.enable_smc_structures:
            return {'bias': {}, 'consistency': 1.0, 'recommendation': 'neutral'}
        
        htf_bias = {}  # Higher Time Frame偏置
        consistency_score = 0
        
        # 分析高时间框架的趋势偏置
        for tf in ['1d', '4h', '1h']:  # HTF优先
            if tf not in multi_tf_data or not isinstance(multi_tf_data, dict) or multi_tf_data[tf].empty:
                continue
                
            structures = self.detect_smc_structures(multi_tf_data[tf], tf)
            if structures is not None and isinstance(structures, dict) and structures.get('bos_choch'):
                last_bos = structures['bos_choch'][-1] if structures['bos_choch'] else {}
                if last_bos.get('type') == 'BOS':
                    direction = last_bos.get('direction', 0)
                    if direction > 0:
                        htf_bias[tf] = 'bull'
                    elif direction < 0:
                        htf_bias[tf] = 'bear'
                    else:
                        htf_bias[tf] = 'neutral'
                else:
                    htf_bias[tf] = 'neutral'
            else:
                htf_bias[tf] = 'neutral'
        
        # 小级别（15m）信号验证
        ltf_struct = {}
        if '15m' in multi_tf_data and isinstance(multi_tf_data, dict) and not multi_tf_data['15m'].empty:
            ltf_struct = self.detect_smc_structures(multi_tf_data['15m'], '15m')
        
        # 一致性检查
        if ltf_struct is not None and isinstance(ltf_struct, dict) and ltf_struct.get('strength_score', 0) > 0:
            htf_trend = htf_bias.get('4h', 'neutral')  # 以H4为主
            ltf_signal = 'valid' if (htf_trend == 'bull' and ltf_struct['strength_score'] > self.config.min_structure_score) else 'invalid'
            
            if htf_trend == 'bull' and ltf_struct['strength_score'] > self.config.min_structure_score:
                consistency_score = 1.0
                recommendation = 'strong_buy'
            elif htf_trend == 'bear' and ltf_struct['strength_score'] > self.config.min_structure_score:
                consistency_score = 1.0
                recommendation = 'strong_sell'
            else:
                consistency_score = 0.3  # 权重惩罚
                recommendation = 'weak_signal'
        else:
            consistency_score = 0.5
            recommendation = 'neutral'
        
        self.logger_system.info(f"MTF偏置: D1={htf_bias.get('1d', 'neutral')}, H4={htf_bias.get('4h', 'neutral')}, H1={htf_bias.get('1h', 'neutral')}, 一致性={consistency_score:.2f}, 建议={recommendation}")
        
        return {
            'bias': htf_bias,
            'consistency': consistency_score,
            'recommendation': recommendation,
            'ltf_strength': ltf_struct.get('strength_score', 0) if ltf_struct else 0
        }
    
    def calculate_structure_liquidity_score(self, structures: Dict[str, Any], df: pd.DataFrame) -> float:
        """流动性评分：整合结构+深度+成交量"""
        if structures is None or not isinstance(structures, dict) or df.empty:
            return 0.0
        
        try:
            strength = structures.get('strength_score', 0)
            liq_sweeps = structures.get('liq_sweeps', [])
            
            # 流动性分数：成交量堆积 / ATR
            vol_ma = df['volume'].rolling(20).mean().iloc[-1] if len(df) >= 20 else df['volume'].mean()
            current_volume = df['volume'].iloc[-1]
            liq_score = current_volume / vol_ma if vol_ma > 0 else 1.0
            
            # 订单区深度计算
            ob_data = structures.get('ob_fvg', {}).get('ob', [])
            ob_depth = 0
            if ob_data is not None and isinstance(ob_data, list) and len(ob_data) > 0:
                for ob in ob_data:
                    if isinstance(ob, dict) and 'high' in ob and 'low' in ob:
                        ob_size = abs(ob['high'] - ob['low'])
                        ob_volume = ob.get('volume', current_volume * 0.1)  # 默认成交量
                        ob_depth += ob_size * ob_volume
                ob_depth = ob_depth / len(ob_data) if ob_data else 0
            
            # ATR标准化
            atr = self._atr(df, 14).iloc[-1] if len(df) >= 14 else df['close'].std()
            normalized_depth = (ob_depth / atr) if atr > 0 else 0
            
            # 加权总分
            total_score = (
                self.config.structure_weights['bos_choch'] * strength +
                self.config.structure_weights['ob_fvg'] * min(normalized_depth, 1.0) +
                self.config.structure_weights['liquidity'] * min(liq_score, 2.0)
            )
            
            self.logger_system.info(f"流动性评分: 结构强度={strength:.2f}, 流动性={liq_score:.2f}, OB深度={normalized_depth:.2f}, 总分={total_score:.2f}")
            return min(total_score, 1.0)  # 限制在[0,1]范围
            
        except Exception as e:
            self.logger_system.error(f"流动性评分计算失败: {e}")
            return 0.0
    
    def intraday_momentum_filter(self, price_data: Dict[str, Any]) -> bool:
        # 增强的动量过滤器，结合SMC结构分析
        rsi = price_data['technical_data'].get('rsi', 50)
        
        # 检查是否有SMC结构数据
        multi_tf_data = price_data.get('multi_tf_data', {})
        if multi_tf_data is not None and isinstance(multi_tf_data, dict) and self.config.enable_smc_structures:
            # 进行多时间框架分析
            mtf_analysis = self._mtf_structure_analysis(multi_tf_data)
            consistency = mtf_analysis.get('consistency', 0)
            
            # 如果一致性评分太低，过滤掉信号
            if consistency < self.config.mtf_consensus_threshold:
                self.logger_system.info(f"动量过滤器：MTF一致性评分过低 ({consistency:.2f} < {self.config.mtf_consensus_threshold})")
                return False
        
        # 基础RSI过滤
        base_filter = 30 < rsi < 70
        
        if base_filter is False:
            self.logger_system.info(f"动量过滤器：RSI超出范围 ({rsi})")
        
        return base_filter

    def analyze_with_deepseek(self, price_data: Dict[str, Any], activated_level: Optional[str]) -> Optional[Dict[str, Any]]:
        try:
            if deepseek_client is None:
                self.logger_system.error("DeepSeek client not available")
                return None
    
            # Extract data for the new optimized prompt
            current_price = price_data['price']
            technical_data = price_data.get('technical_data', {})
            smc_structures = price_data.get('smc_structures', {})  # FIXED: 修复双花括号语法错误
            mtf_analysis = price_data.get('mtf_analysis', {})  # FIXED: 修复双花括号语法错误
    
            # Get higher timeframe data
            higher_tf = config.higher_tf_bias_tf
            primary_tf = config.primary_timeframe
            lower_tf = config.lower_tf_entry_tf
    
            # Extract SMC structure data with fallback values
            higher_tf_invalidation = smc_structures.get('higher_tf_choch_bos_invalidation', current_price * 0.98)
            nearest_key_level = smc_structures.get('nearest_key_level', current_price * 0.98)
            key_level_distance = smc_structures.get('key_level_distance', 0.02)
            structure_score = smc_structures.get('structure_score', 0.5)
            fresh_zones = smc_structures.get('fresh_zones', 0)
    
            # Extract MTF analysis data
            higher_tf_trend = mtf_analysis.get(higher_tf, {}).get('trend', 'neutral')
            higher_tf_strength = mtf_analysis.get(higher_tf, {}).get('strength', 0.5)
            primary_tf_trend = mtf_analysis.get(primary_tf, {}).get('trend', 'neutral')
            primary_tf_strength = mtf_analysis.get(primary_tf, {}).get('strength', 0.5)
            lower_tf_trend = mtf_analysis.get(lower_tf, {}).get('trend', 'neutral')
            lower_tf_strength = mtf_analysis.get(lower_tf, {}).get('strength', 0.5)
            mtf_consistency = mtf_analysis.get('consistency', 0.5)
    
            # Extract technical indicators with fallbacks
            rsi = technical_data.get('rsi', 50)
            macd_line = technical_data.get('macd', 0)
            macd_signal = technical_data.get('macd_signal', 0)
            macd_histogram = macd_line - macd_signal
            atr = technical_data.get('atr', current_price * 0.02)
            ema_20 = technical_data.get('sma_20', current_price)
            ema_100 = technical_data.get('ema_100', current_price)
    
            # Calculate volume confirmation
            volume_confirmation = 1.0  # Default fallback
            if 'multi_tf_data' in price_data and primary_tf in price_data['multi_tf_data']:
                df = price_data['multi_tf_data'][primary_tf]
                if not df.empty and 'volume' in df.columns and len(df) > 20:
                    volume_ma = df['volume'].rolling(20).mean().iloc[-1]
                    current_volume = df['volume'].iloc[-1]
                    if volume_ma > 0:
                        volume_confirmation = current_volume / volume_ma
    
            # Risk context
            kill_zone_active = False
            if config.enable_kill_zone:
                now_utc = datetime.now(timezone.utc).hour
                kill_zone_active = config.kill_zone_start_utc <= now_utc <= config.kill_zone_end_utc

            prompt = f"""
        You are a professional crypto trader specializing in SMC/ICT strategies for {config.symbol}. Analyze the provided market data and generate a high-conviction trading signal. Consider multi-timeframe alignment (higher TF bias on {higher_tf}, entry on {lower_tf}), SMC structures (CHOCH for reversal on higher TF, then BOS for confirmation; SL at higher TF CHOCH-BOS invalidation for high RR, prioritize nearest key level if close). Only trade if volume > {config.volume_confirmation_threshold}x MA and fresh zones (interactions <= {config.max_zone_interactions}).

        Market Snapshot (JSON):
        {{
            "current_price": {current_price},
            "activated_level": "{activated_level or 'none'}",
            "timestamp": "{datetime.now(timezone.utc).isoformat()}",
            "primary_tf": "{primary_tf}",
            "technical_indicators": {{
                    "rsi": {rsi:.2f},
                    "macd": {{"line": {macd_line:.4f}, "signal": {macd_signal:.4f}, "histogram": {macd_histogram:.4f}}},
                    "atr": {atr:.2f},
                    "ema_20": {ema_20:.2f},
                    "ema_100": {ema_100:.2f},
                    "volume_confirmation": "{volume_confirmation:.2f}x MA"
            }},
            "mtf_summary": {{
                "{higher_tf}": "{higher_tf_trend} (strength: {higher_tf_strength:.2f})",
                "{primary_tf}": "{primary_tf_trend} (strength: {primary_tf_strength:.2f})",
                "{lower_tf}": "{lower_tf_trend} (strength: {lower_tf_strength:.2f})",
                "consistency": {mtf_consistency:.2f}
            }},
            "smc_structures": {{
                "bos_choch": "{smc_structures.get('bos_choch', 'neutral')}",
                "ob_fvg": "{smc_structures.get('ob_fvg', 'neutral')}",
                "structure_score": {structure_score:.2f},
                "fresh_zones": {fresh_zones},
                "higher_tf_choch_bos_invalidation": {higher_tf_invalidation:.4f},
                "nearest_key_level": {nearest_key_level:.4f},
                "key_level_distance": {key_level_distance:.4f}
            }},
            "risk_context": {{
                "volatility": "{price_data.get('volatility', 2.0):.1f}%",
                "kill_zone_active": {kill_zone_active},
                "suggested_leverage": {config.leverage},
                "min_rr": {config.rr_min_threshold},
                "activation_threshold": {config.activation_threshold}
            }}
        }}

        Rules:
        - BUY: Bullish MTF alignment + bullish SMC (e.g., bullish CHOCH first for reversal on {higher_tf}, then BOS up for confirmation + FVG support) + RSI <70 + positive MACD histogram. SL at higher TF CHOCH low or BOS invalidation below (structure break point); if nearest key level within {config.activation_threshold} distance, prioritize it for tighter risk. TP at next higher TF liquidity/OB (ensure RR >= {config.rr_min_threshold}:1, e.g., 3x risk).
        - SELL: Bearish MTF alignment + bearish SMC (e.g., bearish CHOCH first for reversal on {higher_tf}, then BOS down for confirmation + OB resistance) + RSI >30 + negative MACD histogram. SL at higher TF CHOCH high or BOS invalidation above (structure break point); if nearest key level within {config.activation_threshold} distance, prioritize it for tighter risk. TP at next higher TF liquidity/FVG (ensure RR >= {config.rr_min_threshold}:1, e.g., 3x risk).
        - HOLD: No clear alignment (e.g., missing higher TF CHOCH-BOS sequence) or adjusted RR < {config.rr_min_threshold}:1 (e.g., invalidation too far even after key level prioritization).
        - Set SL/TP realistically: Use higher_tf_choch_bos_invalidation for base SL (ATR-adjusted); override with nearest_key_level if key_level_distance < {config.activation_threshold}; TP based on next structure for high RR.
        """
        
            self.logger_system.info("=" * 80)
            self.logger_system.info("📤 发送给DeepSeek的提示词:")
            self.logger_system.info("-" * 40)
            self.logger_system.info(prompt.strip())
            self.logger_system.info("-" * 40)
            response = deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=config.temperature
            )
            signal_text = response.choices[0].message.content.strip()
            # 记录DeepSeek的完整响应
            self.logger_system.info("📥 DeepSeek的完整响应:")
            self.logger_system.info("-" * 40)
            self.logger_system.info(signal_text)
            self.logger_system.info("-" * 40)
            self.logger_system.info("=" * 80)
            # 尝试提取JSON部分
            # 查找JSON开始和结束位置
            start_idx = signal_text.find('{{')
            end_idx = signal_text.rfind('}}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = signal_text[start_idx:end_idx]
                signal_data = json.loads(json_str)
            else:
                raise ValueError("No valid JSON found in response")
            # 验证信号数据完整性
            required_fields = ['signal', 'entry_price', 'stop_loss', 'take_profit', 'confidence', 'reason']
            if not all(field in signal_data for field in required_fields):
                self.logger_system.warning("Incomplete signal data, using fallback")
                signal_data = self._generate_fallback_signal(price_data, activated_level)
            # 验证信号值的合理性
            if signal_data['signal'] not in ['BUY', 'SELL', 'HOLD']:
                signal_data['signal'] = 'HOLD'
            self.logger_system.info(f"Generated signal: {signal_data['signal']} at {signal_data['entry_price']:.2f}")
            return signal_data
        
        except (json.JSONDecodeError, ValueError, Exception) as e:
            self.logger_system.error(f"DeepSeek analysis failed: {e}")
            return self._generate_fallback_signal(price_data, activated_level)

    def _generate_optimized_signal(self, price_data: Dict[str, Any], activated_level: Optional[str]) -> Optional[Dict[str, Any]]:
        # Generate optimized signal using SignalStabilizer with priority-based conflict resolution
        try:
            # Generate signals from different sources with priorities
            signals = []
            
            # 1. DeepSeek AI Analysis (Highest Priority)
            if self.config.enable_signal_fusion:
                ai_signal = self.analyze_with_deepseek(price_data, activated_level)
                if ai_signal and ai_signal['signal'] != 'HOLD':
                    signals.append((ai_signal, SignalPriority.AI_ANALYSIS, 'ai_analysis'))
            
            # 2. SMC Structure Analysis (High Priority)
            if self.config.enable_smc_structures and price_data.get('smc_structures'):
                smc_signal = self._generate_smc_signal(price_data, activated_level)
                if smc_signal and smc_signal['signal'] != 'HOLD':
                    signals.append((smc_signal, SignalPriority.SMC_STRUCTURE, 'smc_structure'))
            
            # 3. Momentum-based signals (Medium Priority)
            momentum_signal = self._generate_momentum_signal(price_data, activated_level)
            if momentum_signal and momentum_signal['signal'] != 'HOLD':
                signals.append((momentum_signal, SignalPriority.MOMENTUM, 'momentum'))
            
            # 4. Fallback signals (Low Priority)
            fallback_signal = self._generate_fallback_signal(price_data, activated_level)
            if fallback_signal and fallback_signal['signal'] != 'HOLD':
                signals.append((fallback_signal, SignalPriority.FALLBACK, 'fallback'))
            
            # Add all signals to stabilizer
            for signal_data, priority, source in signals:
                # Check for duplicate signals
                if self._is_duplicate_signal(signal_data, source):
                    self.logger_system.info(f"Skipping duplicate {source} signal")
                    continue
                
                # Validate risk-reward ratio
                if not self._validate_risk_reward_ratio(signal_data):
                    self._log_contextual_rejection(signal_data, source, "risk_reward_validation")
                    continue
                
                # Check trend consistency filtering
                if self.signal_stabilizer.should_filter_signal(signal_data, priority):
                    self._log_contextual_rejection(signal_data, source, "trend_consistency_filter")
                    continue
                
                self.signal_stabilizer.add_signal(signal_data, priority, source)
            
            # Get consolidated signal from stabilizer
            consolidated_signal = self.signal_stabilizer.get_consolidated_signal()
            
            if consolidated_signal:
                self.logger_system.info(f"Consolidated signal: {consolidated_signal['signal']} "
                f"(priority: {consolidated_signal['priority'].name}, "
                f"source: {consolidated_signal['source']})")
                return consolidated_signal['data']
            else:
                self.logger_system.info("No actionable signals from stabilizer")
                return None
                
        except Exception as e:
            self.logger_system.error(f"Optimized signal generation failed: {e}")
            return None
    
    def _generate_smc_signal(self, price_data: Dict[str, Any], activated_level: Optional[str]) -> Optional[Dict[str, Any]]:
        # Generate signal based on SMC structure analysis with higher TF CHOCH-BOS invalidation and key level prioritization
        try:
            current_price = price_data['price']
            smc_structures = price_data.get('smc_structures', {})  # FIXED: 修复双花括号语法错误
            mtf_analysis = price_data.get('mtf_analysis', {})  # FIXED: 修复双花括号语法错误
            
            if not smc_structures or not mtf_analysis:
                return None
            
            # Get higher timeframe and primary timeframe structures
            higher_tf = self.config.higher_tf_bias_tf
            primary_tf = self.config.primary_timeframe
            higher_tf_structures = smc_structures.get(higher_tf, {})  # FIXED: 修复双花括号语法错误
            primary_structures = smc_structures.get(primary_tf, {})  # FIXED: 修复双花括号语法错误
            
            if not primary_structures:
                return None
            
            # Extract optimization data
            higher_tf_invalidation = smc_structures.get('higher_tf_choch_bos_invalidation', current_price * 0.98)
            nearest_key_level = smc_structures.get('nearest_key_level', current_price * 0.98)
            key_level_distance = smc_structures.get('key_level_distance', 0.02)
            structure_score = smc_structures.get('structure_score', 0)
            fresh_zones = smc_structures.get('fresh_zones', 0)
            
            # Check if nearest key level should be prioritized (within activation threshold)
            prioritize_key_level = key_level_distance < self.config.activation_threshold
            
            recommendation = mtf_analysis.get('recommendation', 'neutral')
            consistency = mtf_analysis.get('consistency', 0)
            
            # Generate signal based on SMC analysis with optimization
            if recommendation in ['strong_buy', 'buy'] and consistency > self.config.mtf_consensus_threshold:
                signal = 'BUY'
                
                # Determine stop loss with key level prioritization
                if prioritize_key_level:
                    # Use nearest key level for tighter risk if within threshold
                    base_stop_loss = nearest_key_level * 0.998  # Slightly below key level
                    reason_suffix = f", key level prioritized (distance: {key_level_distance * 100:.2f}%)"
                else:
                    # Use higher timeframe CHOCH-BOS invalidation
                    base_stop_loss = higher_tf_invalidation
                    reason_suffix = f", higher TF invalidation used"
                
                # Calculate take profit for minimum R:R ratio
                risk_amount = abs(current_price - base_stop_loss)
                min_reward = risk_amount * self.config.rr_min_threshold
                take_profit = current_price + min_reward
                
                # Validate R:R ratio
                actual_rr = abs(take_profit - current_price) / risk_amount if risk_amount > 0 else 0
                if actual_rr < self.config.rr_min_threshold:
                    self.logger_system.info(f"SMC BUY signal rejected: R:R {actual_rr:.2f} < minimum {self.config.rr_min_threshold}")
                    return None
                
                stop_loss = base_stop_loss
                reason = f"SMC bullish structure (score: {structure_score:.2f}, consistency: {consistency:.2f}, RR: {actual_rr:.1f}:1{reason_suffix})"
                
            elif recommendation in ['strong_sell', 'sell'] and consistency > self.config.mtf_consensus_threshold:
                signal = 'SELL'
                
                # Determine stop loss with key level prioritization
                if prioritize_key_level:
                    # Use nearest key level for tighter risk if within threshold
                    base_stop_loss = nearest_key_level * 1.002  # Slightly above key level
                    reason_suffix = f", key level prioritized (distance: {key_level_distance * 100:.2f}%)"
                else:
                    # Use higher timeframe CHOCH-BOS invalidation
                    base_stop_loss = higher_tf_invalidation
                    reason_suffix = f", higher TF invalidation used"
                
                # Calculate take profit for minimum R:R ratio
                risk_amount = abs(current_price - base_stop_loss)
                min_reward = risk_amount * self.config.rr_min_threshold
                take_profit = current_price - min_reward
                
                # Validate R:R ratio
                actual_rr = abs(take_profit - current_price) / risk_amount if risk_amount > 0 else 0
                if actual_rr < self.config.rr_min_threshold:
                    self.logger_system.info(f"SMC SELL signal rejected: R:R {actual_rr:.2f} < minimum {self.config.rr_min_threshold}")
                    return None
                
                stop_loss = base_stop_loss
                reason = f"SMC bearish structure (score: {structure_score:.2f}, consistency: {consistency:.2f}, RR: {actual_rr:.1f}:1{reason_suffix})"
                
            else:
                return None
            
            # Additional validation: check fresh zones and volume confirmation
            if fresh_zones < 1:
                self.logger_system.info(f"SMC signal rejected: insufficient fresh zones ({fresh_zones})")
                return None
            
            return {
                'signal': signal,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence': 'HIGH' if consistency > 0.8 and actual_rr >= self.config.rr_min_threshold * 1.5 else 'MEDIUM',
                'reason': reason
            }

        except Exception as e:
            self.logger_system.error(f"SMC signal generation failed: {e}")
            return None
    
    def _generate_momentum_signal(self, price_data: Dict[str, Any], activated_level: Optional[str]) -> Optional[Dict[str, Any]]:
        """Generate signal based on momentum indicators"""
        try:
            current_price = price_data['price']
            technical_data = price_data.get('technical_data', {})
            
            if not technical_data:
                return None
            
            rsi = technical_data.get('rsi', 50)
            volatility = price_data.get('volatility', 0)
            
            # Momentum-based signal logic
            if rsi < 30 and volatility > self.config.volatility_threshold:
                signal = 'BUY'
                stop_loss = current_price * 0.97
                take_profit = current_price * 1.05
                reason = f"Oversold momentum (RSI: {rsi:.1f}, volatility: {volatility:.1f}%)"
            elif rsi > 70 and volatility > self.config.volatility_threshold:
                signal = 'SELL'
                stop_loss = current_price * 1.03
                take_profit = current_price * 0.95
                reason = f"Overbought momentum (RSI: {rsi:.1f}, volatility: {volatility:.1f}%)"
            else:
                return None
            
            return {
                'signal': signal,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence': 'MEDIUM',
                'reason': reason
            }
            
        except Exception as e:
            self.logger_system.error(f"Momentum signal generation failed: {e}")
            return None
    
    def _is_duplicate_signal(self, signal_data: Dict[str, Any], source: str) -> bool:
        """Check if signal is duplicate based on hash"""
        if not self.config.enable_duplicate_filtering:
            return False
        
        try:
            # Create signal hash
            signal_hash = self._create_signal_hash(signal_data, source)
            current_time = time.time()
            
            # Clean old hashes
            cutoff_time = current_time - self.config.duplicate_signal_ttl
            self.signal_hashes = {
                hash_val for hash_val in self.signal_hashes
                if self.signal_hash_timestamps.get(hash_val, 0) > cutoff_time
            }
            
            # Check if hash exists
            if signal_hash in self.signal_hashes:
                return True
            
            # Add new hash
            self.signal_hashes.add(signal_hash)
            self.signal_hash_timestamps[signal_hash] = current_time
            
            return False
            
        except Exception as e:
            self.logger_system.error(f"Duplicate signal check failed: {e}")
            return False
    
    def _create_signal_hash(self, signal_data: Dict[str, Any], source: str) -> str:
        """Create hash for signal data"""
        import hashlib
        
        hash_input = f"{signal_data['signal']}:{signal_data['entry_price']:.2f}:{source}:{int(time.time() / 60)}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _log_contextual_rejection(self, signal_data: Dict[str, Any], source: str, reason: str):
        """Log contextual rejection for analysis"""
        if not self.config.enable_contextual_logging:
            return
        
        try:
            rejection_entry = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'signal': signal_data['signal'],
                'source': source,
                'reason': reason,
                'entry_price': signal_data['entry_price'],
                'confidence': signal_data.get('confidence', 'UNKNOWN'),
                'market_data': {
                    'price': signal_data.get('entry_price', 0),
                    'volatility': getattr(self, 'last_volatility', 0),
                    'rsi': getattr(self, 'last_rsi', 50)
                }
            }
            
            self.contextual_rejections.append(rejection_entry)
            
            # Save to file periodically (every 10 rejections)
            if len(self.contextual_rejections) >= 10:
                self._save_contextual_rejections()
                
        except Exception as e:
            self.logger_system.error(f"Contextual rejection logging failed: {e}")
    
    def _save_contextual_rejections(self):
        """Save contextual rejections to file"""
        try:
            with open(self.config.contextual_log_file, 'a') as f:
                for rejection in self.contextual_rejections:
                    f.write(json.dumps(rejection) + '\n')
            
            self.contextual_rejections.clear()
            self.logger_system.info(f"Saved {len(self.contextual_rejections)} contextual rejections to {self.config.contextual_log_file}")
            
        except Exception as e:
            self.logger_system.error(f"Failed to save contextual rejections: {e}")

    def _calculate_higher_tf_invalidation(self, higher_tf_structures: Dict, primary_tf_structures: Dict, 
                                          current_price: float, higher_tf_df: pd.DataFrame, 
                                          primary_tf_df: pd.DataFrame) -> float:
        """Calculate higher timeframe CHOCH-BOS invalidation point for stop loss placement"""
        try:
            # Default invalidation point (2% from current price)
            default_invalidation = current_price * 0.98 if current_price > 0 else 4000 * 0.98
            
            if not higher_tf_structures or not primary_tf_structures:
                return default_invalidation
            
            # Extract structure information
            higher_bos_choch = higher_tf_structures.get('bos_choch', 'neutral')
            primary_bos_choch = primary_tf_structures.get('bos_choch', 'neutral')
            
            # Get recent swing highs and lows from data
            if higher_tf_df.empty or primary_tf_df.empty:
                return default_invalidation
            
            # Calculate recent swing points
            higher_high = higher_tf_df['high'].tail(20).max() if len(higher_tf_df) >= 20 else higher_tf_df['high'].max()
            higher_low = higher_tf_df['low'].tail(20).min() if len(higher_tf_df) >= 20 else higher_tf_df['low'].min()
            
            primary_high = primary_tf_df['high'].tail(10).max() if len(primary_tf_df) >= 10 else primary_tf_df['high'].max()
            primary_low = primary_tf_df['low'].tail(10).min() if len(primary_tf_df) >= 10 else primary_tf_df['low'].min()
            
            # Determine invalidation based on structure bias
            if higher_bos_choch == 'bullish' and primary_bos_choch == 'bullish':
                # For bullish bias, invalidation is below the higher timeframe swing low
                invalidation = min(higher_low, primary_low) * 0.995
            elif higher_bos_choch == 'bearish' and primary_bos_choch == 'bearish':
                # For bearish bias, invalidation is above the higher timeframe swing high
                invalidation = max(higher_high, primary_high) * 1.005
            else:
                # For neutral or mixed bias, use the structure that provides better risk-reward
                if higher_bos_choch == 'bullish':
                    invalidation = higher_low * 0.995
                elif higher_bos_choch == 'bearish':
                    invalidation = higher_high * 1.005
                else:
                    # Use ATR-based invalidation as fallback
                    atr_multiplier = 1.5
                    if not primary_tf_df.empty and 'atr' in primary_tf_df.columns:
                        atr = primary_tf_df['atr'].iloc[-1] if not primary_tf_df['atr'].empty else current_price * 0.02
                    else:
                        atr = current_price * 0.02
                    
                    if primary_bos_choch == 'bullish':
                        invalidation = current_price - (atr * atr_multiplier)
                    elif primary_bos_choch == 'bearish':
                        invalidation = current_price + (atr * atr_multiplier)
                    else:
                        invalidation = default_invalidation
            
            # Ensure invalidation is reasonable (within 1-5% of current price)
            invalidation_pct = abs(invalidation - current_price) / current_price * 100
            if invalidation_pct < 0.5:  # Too tight
                invalidation = current_price * (0.995 if invalidation < current_price else 1.005)
            elif invalidation_pct > 8:  # Too wide
                invalidation = current_price * (0.92 if invalidation < current_price else 1.08)
            
            self.logger_system.info(f"Higher TF invalidation calculated: {invalidation:.4f} "
                                    f"({abs(invalidation - current_price) / current_price * 100:.2f}% from current price)")
            
            return invalidation
            
        except Exception as e:
            self.logger_system.error(f"Higher TF invalidation calculation failed: {e}")
            return current_price * 0.98 if current_price > 0 else 4000 * 0.98

    def _calculate_nearest_key_level(self, current_price: float, key_levels: Dict) -> tuple[float, float]:
        """Calculate the nearest key level and its distance from current price"""
        try:
            if not key_levels or len(key_levels) == 0:
                # Default fallback values
                default_level = current_price * 0.98 if current_price > 0 else 4000 * 0.98
                return default_level, abs(default_level - current_price) / current_price
            
            # Extract all key level values
            level_values = []
            for level_name, level_value in key_levels.items():
                if level_name != 'current_price' and isinstance(level_value, (int, float)) and level_value > 0:
                    level_values.append(level_value)
            
            if not level_values:
                default_level = current_price * 0.98 if current_price > 0 else 4000 * 0.98
                return default_level, abs(default_level - current_price) / current_price
            
            # Find the nearest level
            nearest_level = min(level_values, key=lambda x: abs(x - current_price))
            distance = abs(nearest_level - current_price) / current_price
            
            self.logger_system.info(f"Nearest key level: {nearest_level:.4f} "
                                    f"(distance: {distance * 100:.2f}% from current price {current_price:.4f})")
            
            return nearest_level, distance
            
        except Exception as e:
            self.logger_system.error(f"Nearest key level calculation failed: {e}")
            default_level = current_price * 0.98 if current_price > 0 else 4000 * 0.98
            return default_level, abs(default_level - current_price) / current_price
    
    def _validate_risk_reward_ratio(self, signal_data: Dict) -> bool:
        """Validate that the signal meets minimum risk-reward ratio requirements."""
        try:
            action = signal_data.get('signal', '').upper()  # FIXED: 修复字段名错误
            entry_price = signal_data.get('entry_price', 0)
            stop_loss = signal_data.get('stop_loss', 0)
            take_profit = signal_data.get('take_profit', 0)
            
            if not all([entry_price, stop_loss, take_profit]):
                self.logger_system.warning(f"Missing price data for R:R validation: entry={entry_price}, SL={stop_loss}, TP={take_profit}")
                return False
            
            # Calculate risk and reward
            if action == 'BUY':
                risk = abs(entry_price - stop_loss)
                reward = abs(take_profit - entry_price)
            elif action == 'SELL':
                risk = abs(stop_loss - entry_price)
                reward = abs(entry_price - take_profit)
            else:
                return True  # HOLD signals don't need R:R validation
            
            # Validate risk and reward are positive
            if risk <= 0 or reward <= 0:
                self.logger_system.warning(f"Invalid risk/reward values: risk={risk}, reward={reward}")
                return False
            
            # Calculate R:R ratio
            rr_ratio = reward / risk
            min_rr = self.config.rr_min_threshold  # FIXED: 修复属性访问错误
            
            if rr_ratio < min_rr:
                self.logger_system.info(f"Signal rejected: R:R ratio {rr_ratio:.2f} below minimum {min_rr}")
                return False
            
            self.logger_system.info(f"Signal validated: R:R ratio {rr_ratio:.2f} meets minimum {min_rr}")
            return True
            
        except Exception as e:
            self.logger_system.error(f"Error validating R:R ratio: {e}")
            return False

    def _generate_fallback_signal(self, price_data: Dict[str, Any], activated_level: Optional[str]) -> Dict[str, Any]:
        """生成备用交易信号，基于技术指标"""
        try:
            current_price = price_data['price']
            rsi = price_data['technical_data'].get('rsi', 50)
            
            # 基于RSI的简单策略
            if rsi < 30:
                signal = 'BUY'
                reason = f'RSI oversold ({rsi:.1f})'
                stop_loss = current_price * 0.98  # 2% 止损
                take_profit = current_price * 1.04  # 4% 止盈
            elif rsi > 70:
                signal = 'SELL'
                reason = f'RSI overbought ({rsi:.1f})'
                stop_loss = current_price * 1.02  # 2% 止损
                take_profit = current_price * 0.96  # 4% 止盈
            else:
                signal = 'HOLD'
                reason = f'RSI neutral ({rsi:.1f})'
                stop_loss = current_price * 0.99
                take_profit = current_price * 1.01
            
            return {
                'signal': signal,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence': 'MEDIUM',
                'reason': reason
            }
            
        except Exception as e:
            self.logger_system.error(f"Fallback signal generation failed: {e}")
            # 最后的保险信号
            return {
                'signal': 'HOLD',
                'entry_price': price_data.get('price', 4000),
                'stop_loss': price_data.get('price', 4000) * 0.99,
                'take_profit': price_data.get('price', 4000) * 1.01,
                'confidence': 'LOW',
                'reason': 'Fallback signal due to error'
            }

    def execute_trade(self, signal_data: Dict[str, Any], price_data: Dict[str, Any], activated_level: Optional[str]):
        """执行交易，包含完整的风险检查和错误处理"""
        try:
            signal = signal_data.get('signal', 'HOLD')
            
            # 如果信号是HOLD，不执行交易
            if signal == 'HOLD':
                self.logger_trading.info("Signal is HOLD, no trade executed")
                return
            
            # 检查是否已有持仓
            current_position = self.position_store.get()
            if current_position:
                self.logger_trading.warning("Already have open position, skipping new trade")
                return
            
            # 获取当前余额
            try:
                balance = self.exchange.fetch_balance()
                usdc_balance = balance.get('USDC', {}).get('free', 0)
                
                if usdc_balance < self.config.min_amount_usdc:
                    self.logger_trading.error(f"Insufficient balance: {usdc_balance:.2f} USDC < {self.config.min_amount_usdc}")
                    return
                    
            except Exception as e:
                self.logger_trading.error(f"Failed to fetch balance: {e}")
                return
            
            # 计算交易参数
            side = signal.lower()
            if side not in ['buy', 'sell']:
                self.logger_trading.error(f"Invalid signal: {signal}")
                return
            
            # 计算交易数量（基于USDC余额和杠杆）
            current_price = price_data['price']
            max_position_value = usdc_balance * self.config.max_margin_usage
            amount = min(self.config.amount, max_position_value / current_price / self.config.leverage)
            
            if amount < 0.001:  # 最小交易量检查
                self.logger_trading.error(f"Trade amount too small: {amount}")
                return
            
            self.logger_trading.info(f"Executing {side.upper()} order: {amount:.4f} ETH at ~${current_price:.2f}")
            
            # 执行订单
            params = {'reduce_only': False}
            order = self.safe_create_order(self.exchange, self.config.symbol, side, amount, params)
            
            if order:
                # 记录持仓信息
                position = {
                    'side': side,
                    'size': amount,
                    'entry_price': order.get('average', current_price),
                    'unrealized_pnl': 0,
                    'leverage': self.config.leverage,
                    'symbol': self.config.symbol,
                    'entry_time': datetime.now(timezone.utc),
                    'liquidation_price': signal_data.get('stop_loss', current_price * 0.95)
                }
                self.position_store.set(position)
                
                # 记录交易历史
                trade_record = {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'signal': signal_data,
                    'order': order,
                    'activated_level': activated_level,
                    'price_data': {
                        'price': current_price,
                        'rsi': price_data['technical_data'].get('rsi', 50)
                    }
                }
                self.signal_history.append(trade_record)
                self.save_signal_history()
                
                # 安全获取订单执行价格，避免NoneType格式错误
                execution_price = order.get('average') or order.get('price') or current_price or 0
                if execution_price and execution_price > 0:
                    self.logger_trading.info(f"✅ Trade executed successfully: {side.upper()} {amount:.4f} ETH at ${execution_price:.2f}")
                else:
                    self.logger_trading.info(f"✅ Trade executed successfully: {side.upper()} {amount:.4f} ETH (price data unavailable)")
                
                # 安全获取止损和止盈价格
                stop_loss = signal_data.get('stop_loss', 0) or 0
                take_profit = signal_data.get('take_profit', 0) or 0
                if stop_loss > 0 and take_profit > 0:
                    self.logger_trading.info(f"Stop Loss: ${stop_loss:.2f}, Take Profit: ${take_profit:.2f}")
                else:
                    self.logger_trading.info(f"Stop Loss: {stop_loss}, Take Profit: {take_profit}")
                
            else:
                self.logger_trading.error("Order execution failed")
                
        except Exception as e:
            self.logger_trading.error(f"Trade execution error: {e}")
            import traceback
            self.logger_trading.debug(f"Trade execution traceback: {traceback.format_exc()}")

    def price_monitor_loop(self):
        """Price monitoring loop: Check real-time price close to key levels"""
        activation_count = 0
        monitor_cycle_count = 0
        
        while True:
            try:
                monitor_cycle_count += 1
                ticker = self.safe_fetch_ticker(self.exchange, config.symbol)
                current_price = ticker['last'] if ticker else 67000.0
                
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
                            self.executor.submit(self.safe_fetch_ohlcv, self.exchange, config.symbol, '4h', 201): '4h',
                            self.executor.submit(self.safe_fetch_ohlcv, self.exchange, config.symbol, '1d', 10): '1d',
                            self.executor.submit(self.safe_fetch_ohlcv, self.exchange, config.symbol, '1w', 5): '1w',
                            self.executor.submit(self.safe_fetch_ohlcv, self.exchange, config.symbol, '15m', 100): '15m'
                        }
                        multi_tf_light = {}
                        for future in as_completed(futures):
                            tf = futures[future]
                            try:
                                ohlcv = future.result()
                                if not ohlcv:  # Check None return value
                                    continue
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
                        # Sample log: Log every 5 activations, or first activation
                        if activation_count == 1 or activation_count % 5 == 0:
                            self.logger_monitor.info("Price activation: %s (cumulative: %d times)", activated, activation_count)
                        else:
                            self.logger_monitor.debug("Price activation: %s (cumulative: %d times)", activated, activation_count)
                        threading.Thread(target=lambda: self.trading_bot(activated), daemon=True).start()
                    else:
                        # Sample log: Log normal status every 20 monitoring cycles
                        if monitor_cycle_count % 20 == 0:
                            self.logger_monitor.debug("Price monitoring normal: Price=%.2f, Cycle=%d", current_price, monitor_cycle_count)
                else:
                    self.logger_monitor.warning("No key levels cache available for price activation check")
                
                time.sleep(180)  # 3 minute price activation check interval
            except Exception as e:  # FIXED: 修复缩进错误
                self.logger_system.exception(f"Price monitoring exception: {e}")
                time.sleep(self.config.price_monitor_interval)

    def heartbeat(self):
        """系统心跳检查，监控关键指标"""
        try:
            # 缓存机制：每3次心跳才获取一次余额（减少API调用）
            if not hasattr(self, '_heartbeat_count'):
                self._heartbeat_count = 0
                self._cached_balance = 0
                self._cached_price = 0
            
            self._heartbeat_count += 1
            
            # 获取持仓信息（本地操作，快速）
            position = self.position_store.get()
            position_info = position['side'] if position else 'No position'
            
            # 每3次心跳获取一次价格和余额
            if self._heartbeat_count % 3 == 1:
                # 获取当前价格
                ticker = self.safe_fetch_ticker(self.exchange, self.config.symbol)
                self._cached_price = ticker.get('last', self._cached_price) if ticker else self._cached_price
                
                # 获取余额信息
                try:
                    balance_data = self.exchange.fetch_balance()
                    self._cached_balance = balance_data.get('USDC', {}).get('free', self._cached_balance)
                except Exception as e:
                    self.logger_monitor.debug(f"Failed to fetch balance in heartbeat: {e}")
            
            # 使用缓存的数据
            current_price = self._cached_price
            balance = self._cached_balance
            
            # 写入心跳日志文件
            try:
                with open(self.config.heartbeat_file, 'a', encoding='utf-8') as f:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"{timestamp}: Price=${current_price:.2f}, Position={position_info}, Balance={balance:.2f} USDC\n")
            except Exception as e:
                self.logger_monitor.debug(f"Failed to write heartbeat file: {e}")
            
            # 输出心跳信息
            self.logger_monitor.info("💓 Heartbeat: %s | Position=%s | Balance=%.2f USDC | Signals=%d | Price=$%.2f",
                                     datetime.now().strftime("%H:%M:%S"), 
                                     position_info, balance, len(self.signal_history), current_price)
            
            # 检查系统健康状态（仅在获取新数据时检查）
            if self._heartbeat_count % 3 == 1:
                if current_price == 0:
                    self.logger_monitor.warning("⚠️ Price data unavailable")
                
                if balance < self.config.min_amount_usdc and not position:
                    self.logger_monitor.warning(f"⚠️ Low balance: {balance:.2f} USDC")
            
        except Exception as e:  # FIXED: 修复缩进错误
            self.logger_monitor.error(f"Heartbeat error: {e}")
            # 确保心跳不会因为错误而停止
            try:
                with open(self.config.heartbeat_file, 'a', encoding='utf-8') as f:
                    f.write(f"{datetime.now()}: HEARTBEAT ERROR - {str(e)}\n")
            except:
                pass  # 忽略文件写入错误

    def heartbeat_loop(self):
        """Heartbeat loop method"""
        while True:
            try:
                self.heartbeat()
                time.sleep(self.config.heartbeat_interval)
            except Exception as e:
                self.logger_system.error(f"Heartbeat loop error: {e}")
                time.sleep(self.config.heartbeat_interval)

    def backtest_from_file(self, file_path: str):
        """Improved backtest implementation, include full simulation logic and P&L calculation - Optimization 6: Add PF and max DD"""
        try:
            df = pd.read_csv(file_path)
            # FIXED: Medium 11 - Validate columns
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"CSV missing required columns: {required_cols}")
            if len(df) < 20:
                self.logger_system.warning(f"Backtest data insufficient ({len(df)} rows), suggest at least 20 rows")
                return
            
            # FIXED: High 6 - Add leverage and fees to pnl
            leverage = self.config.leverage
            fee = self.config.fee_rate
            
            # Calculate technical indicators
            df = self.calculate_technical_indicators(df)
            
            signals = []
            trades = []
            total_pnl = 0.0
            wins = 0
            losses = 0
            peak_balance = 10000.0  # Initial balance
            max_drawdown = 0.0
            current_balance = peak_balance
            
            self.logger_system.info(f"Starting backtest, data rows: {len(df)}")
            
            for i, row in df.iterrows():
                if i < 14:  # Need sufficient historical data for indicator calculation
                    continue
                
                # Build more complete price data
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
                    },
                    'key_levels': {},  # Simplified for backtest
                    'structures_summary': {}  # Simplified
                }
                
                # Use rule-based signal generation (avoid real AI call)
                signal = self._generate_rule_based_signal(price_data, df.iloc[max(0, i-14):i+1])
                signals.append(signal)
                
                # Simulate trade execution
                if signal['signal'] in ['BUY', 'SELL'] and i + 5 < len(df):  # Ensure sufficient subsequent data
                    trade_result = self._simulate_trade_execution(signal, df.iloc[i:i+6], i)
                    if trade_result:
                        trades.append(trade_result)
                        total_pnl += trade_result['pnl']
                        current_balance += trade_result['pnl']
                        peak_balance = max(peak_balance, current_balance)
                        drawdown = (peak_balance - current_balance) / peak_balance
                        max_drawdown = max(max_drawdown, drawdown)
                        if trade_result['pnl'] > 0:
                            wins += 1
                        else:
                            losses += 1
            
            # Calculate backtest stats - Optimization 6
            num_trades = len(trades)
            win_rate = wins / num_trades if num_trades > 0 else 0
            avg_win = sum(t['pnl'] for t in trades if t['pnl'] > 0) / wins if wins > 0 else 0
            avg_loss = sum(t['pnl'] for t in trades if t['pnl'] < 0) / losses if losses > 0 else 0
            profit_factor = abs(avg_win * wins / (avg_loss * losses)) if losses > 0 and avg_loss != 0 else float('inf')
            
            self.logger_system.info(f"=== Backtest Results ===")
            self.logger_system.info(f"Total trades: {num_trades}")
            self.logger_system.info(f"Win rate: {win_rate:.2%} ({wins} wins/{losses} losses)")
            self.logger_system.info(f"Total PnL: {total_pnl:.4f} USD")
            self.logger_system.info(f"Average win: {avg_win:.4f} USD")
            self.logger_system.info(f"Average loss: {avg_loss:.4f} USD")
            self.logger_system.info(f"Profit factor (PF): {profit_factor:.2f}")
            self.logger_system.info(f"Max drawdown: {max_drawdown*100:.2f}%")
            self.logger_system.info(f"Signal distribution: {dict(pd.Series([s['signal'] for s in signals]).value_counts())}")
            
        except Exception as e:
            self.logger_system.exception(f"Backtest failed: {e}")

    def _simulate_trade_execution(self, signal: Dict[str, Any], future_data: pd.DataFrame, entry_index: int) -> Optional[Dict[str, Any]]:
        """Simulate trade execution and P&L calculation - Optimization 6: Add PF calculation post-trade"""
        try:
            entry_price = signal.get('entry_price', future_data.iloc[0]['close'])
            stop_loss = signal['stop_loss']
            take_profit = signal['take_profit']
            side = signal['signal']
            
            # Simulate slippage
            slippage = 0.001
            if side == 'BUY':
                actual_entry = entry_price * (1 + slippage)
            else:
                actual_entry = entry_price * (1 - slippage)
            
            # FIXED: High 5 - Correct PnL calculation and leverage fees
            amount = self.config.amount
            leverage = self.config.leverage
            # Fee based on notional value: amount * entry_price * leverage * fee_rate * 2 (open + close)
            fee_cost = amount * actual_entry * leverage * self.config.fee_rate * 2
            
            # Check subsequent price movement
            for i, row in future_data.iloc[1:].iterrows():
                high = row['high']
                low = row['low']
                close = row['close']
                
                # Check stop loss/take profit trigger
                if side == 'BUY':
                    if low <= stop_loss:
                        # Stop loss triggered
                        exit_price = stop_loss * (1 - slippage)  # Slippage
                        # Correct PnL formula: (exit_price - entry_price) * amount * leverage - fees
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
                        # Take profit triggered
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
                        # Stop loss triggered
                        exit_price = stop_loss * (1 + slippage)
                        # For SELL: (entry_price - exit_price) * amount * leverage - fees
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
                        # Take profit triggered
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
            
            # If no stop loss/take profit triggered, close at last price
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
            self.logger_trading.warning(f"Trade simulation failed: {e}")
            return None

    def _generate_rule_based_signal(self, price_data, recent_df):
        """Generate signal based on rules for backtest"""
        # Simple rule example: RSI overbought/oversold + trend
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

    def start_dynamic_sl_tp_monitor(self):
        def monitor_sl_tp():
            while True:
                position = self.position_store.get()
                if position:
                    # Fetch current price and check SL/TP (simplified)
                    ticker = self.safe_fetch_ticker(self.exchange, config.symbol)
                    current_price = ticker['last']
                    side = position['side']
                    if side == 'buy':
                        if current_price <= position['liquidation_price']:
                            # Close position on SL
                            self.safe_create_order(self.exchange, config.symbol, 'sell', position['size'], {'reduce_only': True})
                            self.position_store.clear()
                            self.logger_trading.info("Stop loss triggered and position closed")
                    # Similar for TP and sell side
                time.sleep(30)  # Check every 30s
        self.sl_tp_monitor_thread = threading.Thread(target=monitor_sl_tp, daemon=True)
        self.sl_tp_monitor_thread.start()

    def trading_bot(self, activated_level: Optional[str] = None, is_scheduled: bool = False):
        """Main trade logic execution method - FIXED: Kill Zone 4 - 可配置过滤 + 数据完整性检查"""
        if not self.trade_lock.acquire(blocking=False):
            self.logger_system.warning("Trade in progress, skip this execution")
            return
        
        try:
            start_time = time.time()
            self.logger_system.info("=== Start trade analysis ===")
            
            # FIXED: Kill Zone 5 - 配置化过滤，如果禁用则警告但继续
            if self.config.enable_kill_zone:
                now_utc = datetime.now(timezone.utc).hour
                if not (self.config.kill_zone_start_utc <= now_utc <= self.config.kill_zone_end_utc):
                    self.logger_system.info(f"Outside Kill Zone (UTC {now_utc}), skipping trade")
                    return
                else:
                    self.logger_system.debug(f"Inside Kill Zone (UTC {now_utc})")
            else:
                self.logger_system.warning("Kill Zone disabled - proceeding with analysis (test mode)")
            
            # Get price data
            price_data = self._fetch_and_update_data(activated_level)
            if not price_data:
                self.logger_system.error("Unable to get price data, skip this trade")
                return
            
            # FIXED: Data 3 - 检查多 TF 数据完整性（至少 70% TF 有数据）
            multi_tf_data = price_data.get('multi_tf_data', {})
            valid_tfs = sum(1 for df in multi_tf_data.values() if len(df) >= 20)
            if valid_tfs < len(self.config.timeframes) * 0.7:
                self.logger_system.warning(f"Insufficient multi-TF data (valid: {valid_tfs}/{len(self.config.timeframes)}), skipping")
                return
            
            # 检查价格激活状态
            is_activated = price_data.get('is_activated', False)
            activated_level_from_data = price_data.get('activated_level', activated_level)
            
            self.logger_system.info(f"价格激活状态: {'✅ 已激活' if is_activated else '❌ 未激活'}")
            if activated_level_from_data:
                self.logger_system.info(f"激活水平: {activated_level_from_data}")
            
            # New: Apply intraday momentum filter with SMC structure analysis
            if not self.intraday_momentum_filter(price_data):
                self.logger_system.info("Intraday momentum filter failed, skipping trade")
                return
            
            # SMC结构增强过滤：如果启用SMC结构分析，进行额外的信号过滤
            if self.config.enable_smc_structures:
                mtf_analysis = price_data.get('mtf_analysis', {})
                smc_structures = price_data.get('smc_structures', {})
                
                # 检查多时间框架一致性
                consistency = mtf_analysis.get('consistency', 0)
                recommendation = mtf_analysis.get('recommendation', 'neutral')
                
                if consistency < self.config.mtf_consensus_threshold:
                    self.logger_system.info(f"MTF一致性评分过低 ({consistency:.2f} < {self.config.mtf_consensus_threshold})，跳过交易")
                    return
                
                # 检查主要时间框架的结构强度
                primary_tf = self.config.primary_timeframe
                if primary_tf in smc_structures and smc_structures[primary_tf]:
                    primary_structures = smc_structures[primary_tf]
                    structure_score = primary_structures.get('strength_score', 0)
                    
                    if structure_score < self.config.min_structure_score:
                        self.logger_system.info(f"主要时间框架结构强度不足 ({structure_score:.2f} < {self.config.min_structure_score})，跳过交易")
                        return
                
                # 记录结构分析结果
                self.logger_system.info(f"SMC结构过滤通过: MTF一致性={consistency:.2f}, 建议={recommendation}, 主要TF结构强度={structure_score:.2f}")
            
            # If scheduled task, check if last signal copy exists
            if is_scheduled:
                # If last signal copy exists, directly use
                if self.last_scheduled_signal:
                    self.logger_system.info("Use last scheduled task signal copy for trade execution")
                    self.execute_trade(self.last_scheduled_signal, price_data, None)  # Fix: Pass None as activated_level
                    execution_time = time.time() - start_time
                    self.logger_system.info(f"=== Trade analysis completed (time: {execution_time:.2f}s) ===")
                    return
            
            # 使用优化的信号生成器（包含优先级冲突解决、去重、趋势一致性过滤）
            self.logger_system.info("🎯 使用优化信号生成器进行多源信号融合")
            signal_data = self._generate_optimized_signal(price_data, activated_level_from_data)
            
            if not signal_data:
                self.logger_system.error("无法生成交易信号，跳过本次交易")
                return
            
            # If scheduled task, save signal copy
            if is_scheduled:
                self.last_scheduled_signal = signal_data.copy()
                self.logger_system.info("Scheduled task signal copy saved")
            
            # Execute trade
            self.execute_trade(signal_data, price_data, activated_level)
            
            execution_time = time.time() - start_time
            self.logger_system.info(f"=== Trade analysis completed (time: {execution_time:.2f}s) ===")
            
        except Exception as e:
            self.logger_system.error(f"Trade execution error: {e}")
        finally:
            self.trade_lock.release()

def job_wrapper(bot, func, *args, **kwargs):
    # If func is bound method, call directly; otherwise pass bot as first parameter
    if hasattr(func, '__self__'):
        # If trading_bot method, add is_scheduled=True parameter
        if func.__name__ == 'trading_bot':
            bot.executor.submit(func, is_scheduled=True, *args, **kwargs)
        else:
            bot.executor.submit(func, *args, **kwargs)
    else:
        # If trading_bot function, add is_scheduled=True parameter
        if func.__name__ == 'trading_bot':
            bot.executor.submit(func, bot, is_scheduled=True, *args, **kwargs)
        else:
            bot.executor.submit(func, bot, *args, **kwargs)

def main():
    # 初始化全局配置和客户端
    initialize_globals()
    
    # FIXED: Medium 7 - Env vars conditional on sim mode
    required_env_vars = ['DEEPSEEK_API_KEY']
    if not config.simulation_mode:
        required_env_vars += ['HYPERLIQUID_WALLET_ADDRESS', 'HYPERLIQUID_PRIVATE_KEY']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        system_logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        return

    bot = TradingBot(config, exchange)
    bot.load_signal_history()
    system_logger.info("BTC/USD Hyperliquid SMC/ICT Auto Trading Bot Started Successfully!")
    system_logger.info("Institutional order flow analysis: Weekly liquidity > Daily liquidity > Order blocks > Volume distribution > Technical levels")
    system_logger.info(f"Key level priority: {', '.join(config.liquidity_priority)}")
    system_logger.info("Key level activation monitoring + Risk management + Dynamic position enabled")
    system_logger.info(f"Heartbeat enabled (interval: {config.heartbeat_interval}s), Log: {config.heartbeat_file}")
    system_logger.warning("Live trading mode, operate cautiously!" if not config.simulation_mode else "Simulation mode activated")
    system_logger.info(f"Primary timeframe: {config.primary_timeframe}")

    # New: Log new features
    system_logger.info(f"Multi-timeframe alignment: Higher TF bias={config.higher_tf_bias_tf}, Lower TF entry={config.lower_tf_entry_tf}")
    system_logger.info(f"Confirmation signals: Volume>{config.volume_confirmation_threshold}x MA, FVG stacking>={config.fvg_stack_threshold}, Fresh zone interactions<={config.max_zone_interactions}")
    
    # SMC结构分析功能状态
    system_logger.info(f"SMC结构分析: {'启用' if config.enable_smc_structures else '禁用'}")
    if config.enable_smc_structures:
        system_logger.info(f"SMC窗口: {config.smc_window}, 范围百分比: {config.smc_range_percent}%")
        system_logger.info(f"结构权重: BOS/CHOCH={config.structure_weights['bos_choch']}, OB/FVG={config.structure_weights['ob_fvg']}, 摆动强度={config.structure_weights['swing_strength']}, 流动性={config.structure_weights['liquidity']}")
        system_logger.info(f"最小结构评分: {config.min_structure_score}, MTF一致性阈值: {config.mtf_consensus_threshold}")

    # Fixed: Check if backtest_file exists
    if config.backtest_file and os.path.exists(config.backtest_file):
        bot.backtest_from_file(config.backtest_file)

    if not bot.setup_exchange():
        system_logger.error("Exchange initialization failed, exit program")
        return

    # Fixed: Initial trading_bot call for startup signal check
    bot.trading_bot()
    
    # FIXED: 启动持续监控线程
    system_logger.info("启动持续监控线程...")
    
    # 启动价格监控线程
    price_monitor_thread = threading.Thread(target=bot.price_monitor_loop, daemon=True)
    price_monitor_thread.start()
    system_logger.info("价格监控线程已启动")
    
    # 启动心跳线程
    heartbeat_thread = threading.Thread(target=bot.heartbeat_loop, daemon=True)
    heartbeat_thread.start()
    system_logger.info("心跳线程已启动")
    
    # 启动动态止损止盈监控线程
    bot.start_dynamic_sl_tp_monitor()
    system_logger.info("动态止损止盈监控线程已启动")
    
    system_logger.info("所有监控线程已启动，程序将持续运行...")
    system_logger.info("按 Ctrl+C 停止程序")
    
    try:
        # 主线程保持运行
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        system_logger.info("收到停止信号，正在关闭程序...")
    except Exception as e:
        system_logger.error(f"主循环异常: {e}")
    finally:
        system_logger.info("程序已安全关闭")
if __name__ == "__main__":
    main()
    