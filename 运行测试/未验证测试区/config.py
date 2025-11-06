"""
配置模块 - 包含交易系统的所有配置参数
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import os
import json
from datetime import time


@dataclass
class Config:
    """交易系统配置类"""
    
    # 基础交易参数
    symbol: str = "PAXG/USD:USD"
    amount: float = 0.01
    leverage: int = 10
    simulation_mode: bool = True
    min_amount_usdc: float = 100.0
    max_margin_usage: float = 0.8
    exchange_name: str = "binance"  # 交易所名称
    
    # 数据源配置
    data_source: str = "hyperliquid"
    timeframes: List[str] = field(default_factory=lambda: ['1m', '3m', '5m', '15m', '30m', '1h', '4h', '1d'])
    primary_timeframe: str = "3m"
    higher_tf_bias_tf: str = "1h"
    lower_tf_entry_tf: str = "3m"
    
    # 风险控制参数
    max_risk_per_trade: float = 0.02
    max_daily_loss: float = 0.05
    max_daily_risk: float = 0.05  # 每日最大风险
    max_drawdown: float = 0.1
    max_positions: int = 1
    max_open_positions: int = 1  # 最大开仓数量
    rr_ratio: float = 2.0
    risk_reward_ratio: float = 2.0  # 风险回报比
    atr_multiplier: float = 2.0
    stop_loss_atr_multiplier: float = 2.0  # 止损ATR倍数
    take_profit_atr_multiplier: float = 3.0  # 止盈ATR倍数
    trailing_stop_enabled: bool = True
    trailing_stop_atr_multiplier: float = 1.5
    trailing_stop_activation_atr: float = 1.5  # 跟踪止损激活ATR
    
    # 订单流分析参数
    volume_confirmation_threshold: float = 1.2
    fvg_stack_threshold: int = 2
    max_zone_interactions: int = 3
    liquidity_sweep_threshold: float = 0.5
    order_flow_window: int = 20
    
    # 多时间框架确认参数
    mtf_consensus_threshold: float = 0.6
    trend_alignment_threshold: float = 0.7
    strength_alignment_threshold: float = 0.6
    
    # Kill Zone配置
    enable_kill_zone: bool = True
    kill_zone_start_utc: int = 8
    kill_zone_end_utc: int = 16
    
    # SMC结构分析配置
    enable_smc_structures: bool = True
    smc_window: int = 50
    smc_range_percent: float = 0.5
    min_structure_score: float = 0.6
    
    # 结构权重配置
    structure_weights: Dict[str, float] = field(default_factory=lambda: {
        'bos_choch': 0.4,
        'ob_fvg': 0.3,
        'swing_strength': 0.2,
        'liquidity': 0.1
    })
    
    # 价格激活参数
    price_activation_threshold: float = 0.002
    price_activation_distance: float = 0.01
    
    # AI参数
    ai_temperature: float = 0.3
    ai_max_tokens: int = 1000
    ai_timeout: int = 30
    ai_retry_count: int = 3
    ai_confidence_threshold: float = 0.6
    
    # 监控参数
    heartbeat_interval: int = 60
    heartbeat_file: str = "heartbeat.log"
    cache_ttl: int = 300  # 5分钟
    price_monitor_interval: int = 60  # 价格监控与调度循环检查间隔(秒)，适配3m主时间框架
    
    # 日志配置
    log_level: str = "INFO"
    log_file: str = "paxg_trading.log"
    contextual_log_file: str = "logs/contextual_rejects.jsonl"
    max_log_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    # 信号持久化（按交易品种自动命名）
    signals_file: str = "logs/signals_history.jsonl"
    
    # 回测配置
    backtest_file: Optional[str] = None
    backtest_start_date: Optional[str] = None
    backtest_end_date: Optional[str] = None
    
    # API配置
    api_timeout: int = 30
    api_retry_count: int = 3
    api_retry_delay: float = 1.0
    api_key: str = ""  # API密钥
    api_secret: str = ""  # API密钥
    api_passphrase: str = ""  # API密码(某些交易所需要)
    
    # AI API配置
    deepseek_api_key: str = ""  # DeepSeek API密钥
    openai_api_key: str = ""  # OpenAI API密钥
    deepseek_base_url: str = "https://api.deepseek.com"  # DeepSeek API基础URL
    deepseek_model: str = "deepseek-chat"  # DeepSeek模型名称
    openai_base_url: str = "https://api.openai.com"  # OpenAI API基础URL
    openai_model: str = "gpt-4"  # OpenAI模型名称
    ai_timeout: int = 30  # AI API超时时间(秒)
    
    # 订单配置
    order_timeout: int = 60  # 订单超时时间(秒)
    order_retry_count: int = 3  # 订单重试次数
    max_order_retries: int = 3  # 最大重试次数（与order_retry_count相同）
    order_retry_delay: float = 1.0  # 订单重试延迟(秒)
    slippage_tolerance: float = 0.001  # 滑点容忍度
    spread_tolerance: float = 0.001  # 点差容忍度（买卖价差占中间价比例）
    # 别名配置（与建议保持一致）：如提供则优先使用
    max_slippage_pct_entry: Optional[float] = None  # 入场滑点容忍度别名
    spread_threshold: Optional[float] = None        # 点差容忍度别名

    # 执行相关的默认参数（供 TradingExecutor 使用）
    default_position_size: float = 0.01  # 默认仓位大小（单位数量）
    max_position_size: float = 1.0       # 最大仓位大小（单位数量上限）
    default_stop_loss_pct: float = 0.01  # 默认止损百分比（例如 0.01 = 1%）
    default_take_profit_pcts: List[float] = field(default_factory=lambda: [0.005, 0.01, 0.02])  # 默认止盈百分比列表
    
    # 执行器配置
    execution_interval: int = 30  # 执行间隔(秒) - 调整为30秒以适应3m主时间框架
    max_concurrent_orders: int = 3  # 最大并发订单数
    max_workers: int = 4  # 线程池最大工作线程数
    risk_check_interval: int = 1  # 风险检查间隔(分钟) - 调整为1分钟以适应3m主时间框架
    
    # 流动性优先级
    liquidity_priority: List[str] = field(default_factory=lambda: [
        "weekly_high_low", "daily_high_low", "vwap", "order_blocks", 
        "fvg", "volume_profile", "pivot_points", "ema_levels"
    ])
    
    # 温度系数（用于AI决策）
    temperature_coefficient: float = 1.0

    # 运行脚本与调度相关的新增字段
    config_path: Optional[str] = None
    analysis_interval: int = 60  # 分析任务间隔(秒)
    report_interval: int = 24    # 报告任务间隔(小时)
    min_signal_confidence: float = 0.6  # 最小执行置信度
    min_signal_interval: int = 300       # 最小信号间隔(秒)
    
    def __post_init__(self):
        """初始化后处理"""
        # 从配置文件加载（如果提供了路径）
        try:
            if getattr(self, "config_path", None):
                with open(self.config_path, "r") as f:
                    cfg = json.load(f)
                trade = cfg.get("trading", {})
                analysis = cfg.get("analysis", {})
                exchange = cfg.get("exchange", {})
                ai = cfg.get("ai", {})
                # 基础交易参数
                self.symbol = trade.get("symbol", self.symbol) or self.symbol
                self.max_risk_per_trade = trade.get("max_risk_per_trade", self.max_risk_per_trade)
                self.max_daily_risk = trade.get("max_daily_risk", self.max_daily_risk)
                # 兼容别名：max_open_positions / max_positions
                _mo = trade.get("max_open_positions")
                _mp = trade.get("max_positions")
                if _mo is not None:
                    try:
                        self.max_open_positions = int(_mo)
                        self.max_positions = int(_mo)
                    except Exception:
                        pass
                elif _mp is not None:
                    try:
                        self.max_open_positions = int(_mp)
                        self.max_positions = int(_mp)
                    except Exception:
                        pass
                self.min_signal_confidence = trade.get("min_signal_confidence", self.min_signal_confidence)
                self.min_signal_interval = trade.get("min_signal_interval", self.min_signal_interval)
                # 执行相关的默认参数（可从配置文件覆盖）
                self.default_position_size = trade.get("default_position_size", self.default_position_size)
                self.max_position_size = trade.get("max_position_size", self.max_position_size)
                self.default_stop_loss_pct = trade.get("default_stop_loss_pct", self.default_stop_loss_pct)
                self.default_take_profit_pcts = trade.get("default_take_profit_pcts", self.default_take_profit_pcts)
                self.spread_tolerance = trade.get("spread_tolerance", self.spread_tolerance)
                # 兼容别名：risk_reward_ratio / rr_ratio
                _rr = trade.get("risk_reward_ratio")
                _rr2 = trade.get("rr_ratio")
                if isinstance(_rr, (int, float)):
                    try:
                        self.risk_reward_ratio = float(_rr)
                        self.rr_ratio = float(_rr)
                    except Exception:
                        pass
                elif isinstance(_rr2, (int, float)):
                    try:
                        self.risk_reward_ratio = float(_rr2)
                        self.rr_ratio = float(_rr2)
                    except Exception:
                        pass
                # 分析与调度参数
                tf = analysis.get("timeframes")
                if tf:
                    self.timeframes = tf
                self.analysis_interval = analysis.get("analysis_interval", self.analysis_interval)
                self.risk_check_interval = analysis.get("risk_check_interval", self.risk_check_interval)
                self.report_interval = analysis.get("report_interval", self.report_interval)
                # 交易所名称
                if exchange.get("name"):
                    self.exchange_name = exchange.get("name")
                # AI 提供商与置信度
                if ai.get("consensus_threshold") is not None:
                    self.ai_confidence_threshold = ai.get("consensus_threshold")
                deepseek = ai.get("deepseek", {})
                openai = ai.get("openai", {})
                if deepseek:
                    self.deepseek_api_key = deepseek.get("api_key", self.deepseek_api_key)
                    self.deepseek_model = deepseek.get("model", self.deepseek_model)
                    self.deepseek_base_url = deepseek.get("base_url", self.deepseek_base_url)
                if openai:
                    self.openai_api_key = openai.get("api_key", self.openai_api_key)
                    self.openai_model = openai.get("model", self.openai_model)
                    self.openai_base_url = openai.get("base_url", self.openai_base_url)
                # 日志相关配置
                logging_cfg = cfg.get("logging", {})
                self.contextual_log_file = logging_cfg.get("contextual_log_file", self.contextual_log_file)
                # log_file 与 signals_file 支持从配置文件加载
                _lf = logging_cfg.get("log_file")
                if _lf:
                    self.log_file = _lf
                _sf = logging_cfg.get("signals_file")
                if _sf:
                    self.signals_file = _sf
        except Exception:
            # 保持静默以兼容无配置文件场景
            pass
        # 从环境变量覆盖配置
        if os.getenv("SIMULATION_MODE"):
            self.simulation_mode = os.getenv("SIMULATION_MODE").lower() == "true"
        
        if os.getenv("LEVERAGE"):
            self.leverage = int(os.getenv("LEVERAGE"))
        
        if os.getenv("AMOUNT"):
            self.amount = float(os.getenv("AMOUNT"))
        
        if os.getenv("SYMBOL"):
            self.symbol = os.getenv("SYMBOL")
        
        # 从环境变量读取API配置
        if os.getenv("BINANCE_API_KEY"):
            self.api_key = os.getenv("BINANCE_API_KEY")
        
        if os.getenv("BINANCE_SECRET"):
            self.api_secret = os.getenv("BINANCE_SECRET")
        
        if os.getenv("OKX_API_KEY"):
            self.api_key = os.getenv("OKX_API_KEY")
        
        if os.getenv("OKX_SECRET"):
            self.api_secret = os.getenv("OKX_SECRET")
        
        if os.getenv("OKX_PASSWORD"):
            self.api_passphrase = os.getenv("OKX_PASSWORD")
        
        # 从环境变量读取AI API配置
        if os.getenv("DEEPSEEK_API_KEY"):
            self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        
        if os.getenv("OPENAI_API_KEY"):
            self.openai_api_key = os.getenv("OPENAI_API_KEY")

        # 日志/信号文件环境覆盖
        if os.getenv("LOG_FILE"):
            self.log_file = os.getenv("LOG_FILE")
        if os.getenv("SIGNALS_FILE"):
            self.signals_file = os.getenv("SIGNALS_FILE")
        if os.getenv("CONTEXTUAL_LOG_FILE"):
            self.contextual_log_file = os.getenv("CONTEXTUAL_LOG_FILE")
        # 自动化：默认上下文日志文件按符号命名（未显式指定时）
        try:
            path = getattr(self, "contextual_log_file", None)
            # 当使用默认占位名或为空时，改为按符号自动命名
            if (not path or not str(path).strip() or str(path).endswith("contextual_rejects.jsonl")) and isinstance(self.symbol, str) and self.symbol:
                base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
                os.makedirs(base_dir, exist_ok=True)
                sanitized = self.symbol.replace("/", "").replace(":", "").replace("-", "")
                self.contextual_log_file = os.path.join(base_dir, f"contextual_{sanitized}.jsonl")
        except Exception:
            # 保持静默，避免影响后续校验
            pass
        # 自动化：log_file 与 signals_file 按符号命名（未显式指定或为占位名时）
        try:
            base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
            os.makedirs(base_dir, exist_ok=True)
            sanitized = self.symbol.replace("/", "").replace(":", "").replace("-", "") if isinstance(self.symbol, str) else ""
            # log_file：默认占位或空 -> <symbol>_trading_bot.log
            lf = getattr(self, "log_file", None)
            if (not lf or not str(lf).strip() or str(lf).endswith("paxg_trading.log")) and sanitized:
                self.log_file = os.path.join(base_dir, f"{sanitized.lower()}_trading_bot.log")
            # signals_file：默认占位或空 -> signals_<symbol>.jsonl
            sf = getattr(self, "signals_file", None)
            if (not sf or not str(sf).strip() or str(sf).endswith("signals_history.jsonl")) and sanitized:
                self.signals_file = os.path.join(base_dir, f"signals_{sanitized.lower()}.jsonl")
        except Exception:
            pass
        # 别名环境覆盖（风险回报与最大持仓）
        if os.getenv("RISK_REWARD_RATIO"):
            try:
                v = float(os.getenv("RISK_REWARD_RATIO"))
                self.risk_reward_ratio = v
                self.rr_ratio = v
            except Exception:
                pass
        if os.getenv("RR_RATIO"):
            try:
                v = float(os.getenv("RR_RATIO"))
                self.risk_reward_ratio = v
                self.rr_ratio = v
            except Exception:
                pass
        if os.getenv("MAX_OPEN_POSITIONS"):
            try:
                v = int(os.getenv("MAX_OPEN_POSITIONS"))
                self.max_open_positions = v
                self.max_positions = v
            except Exception:
                pass
        if os.getenv("MAX_POSITIONS"):
            try:
                v = int(os.getenv("MAX_POSITIONS"))
                self.max_open_positions = v
                self.max_positions = v
            except Exception:
                pass
        # 别名环境覆盖
        if os.getenv("MAX_SLIPPAGE_PCT_ENTRY"):
            try:
                self.max_slippage_pct_entry = float(os.getenv("MAX_SLIPPAGE_PCT_ENTRY"))
            except Exception:
                pass
        if os.getenv("SPREAD_THRESHOLD"):
            try:
                self.spread_threshold = float(os.getenv("SPREAD_THRESHOLD"))
            except Exception:
                pass
        
        # 验证配置参数
        self._validate_config()
    
    def _validate_config(self):
        """验证配置参数"""
        if self.amount <= 0:
            raise ValueError("Amount must be greater than 0")
        
        if self.leverage <= 0:
            raise ValueError("Leverage must be greater than 0")
        
        if self.max_risk_per_trade <= 0 or self.max_risk_per_trade > 1:
            raise ValueError("Max risk per trade must be between 0 and 1")
        
        if self.max_daily_loss <= 0 or self.max_daily_loss > 1:
            raise ValueError("Max daily loss must be between 0 and 1")
        
        if self.max_drawdown <= 0 or self.max_drawdown > 1:
            raise ValueError("Max drawdown must be between 0 and 1")
        
        if self.rr_ratio <= 0:
            raise ValueError("Risk-reward ratio must be greater than 0")
        if self.risk_reward_ratio <= 0:
            raise ValueError("Risk-reward ratio must be greater than 0")
        
        if self.atr_multiplier <= 0:
            raise ValueError("ATR multiplier must be greater than 0")
        
        if self.price_activation_threshold <= 0:
            raise ValueError("Price activation threshold must be greater than 0")
        
        if self.price_activation_distance <= 0:
            raise ValueError("Price activation distance must be greater than 0")
        
        if self.ai_temperature < 0 or self.ai_temperature > 2:
            raise ValueError("AI temperature must be between 0 and 2")
        
        if self.ai_max_tokens <= 0:
            raise ValueError("AI max tokens must be greater than 0")
        
        if self.ai_timeout <= 0:
            raise ValueError("AI timeout must be greater than 0")
        
        if self.ai_retry_count < 0:
            raise ValueError("AI retry count must be non-negative")
        
        if self.ai_confidence_threshold < 0 or self.ai_confidence_threshold > 1:
            raise ValueError("AI confidence threshold must be between 0 and 1")
        
        if self.heartbeat_interval <= 0:
            raise ValueError("Heartbeat interval must be greater than 0")
        
        if self.cache_ttl <= 0:
            raise ValueError("Cache TTL must be greater than 0")
        
        if self.api_timeout <= 0:
            raise ValueError("API timeout must be greater than 0")
        
        if self.api_retry_count < 0:
            raise ValueError("API retry count must be non-negative")
        
        if self.api_retry_delay < 0:
            raise ValueError("API retry delay must be non-negative")
        
        if self.order_timeout <= 0:
            raise ValueError("Order timeout must be greater than 0")
        
        if self.order_retry_count < 0:
            raise ValueError("Order retry count must be non-negative")
        
        if self.max_order_retries < 0:
            raise ValueError("Max order retries must be non-negative")
        
        if self.order_retry_delay < 0:
            raise ValueError("Order retry delay must be non-negative")
        
        if self.slippage_tolerance < 0:
            raise ValueError("Slippage tolerance must be non-negative")
        if self.spread_tolerance < 0:
            raise ValueError("Spread tolerance must be non-negative")
        if self.max_slippage_pct_entry is not None and self.max_slippage_pct_entry < 0:
            raise ValueError("max_slippage_pct_entry must be non-negative")
        if self.spread_threshold is not None and self.spread_threshold < 0:
            raise ValueError("spread_threshold must be non-negative")
        if not isinstance(self.contextual_log_file, str) or not self.contextual_log_file.strip():
            raise ValueError("Contextual log file must be a non-empty string")
        if not isinstance(self.log_file, str) or not self.log_file.strip():
            raise ValueError("Log file must be a non-empty string")
        if not isinstance(self.signals_file, str) or not self.signals_file.strip():
            raise ValueError("Signals file must be a non-empty string")

        # 执行相关默认参数校验
        if self.default_position_size <= 0:
            raise ValueError("Default position size must be greater than 0")
        if self.max_position_size <= 0:
            raise ValueError("Max position size must be greater than 0")
        if self.max_position_size < self.default_position_size:
            raise ValueError("Max position size must be >= default position size")
        if self.default_stop_loss_pct <= 0:
            raise ValueError("Default stop loss percent must be greater than 0")
        if not isinstance(self.default_take_profit_pcts, list) or not self.default_take_profit_pcts:
            raise ValueError("Default take profit percents must be a non-empty list")
        for pct in self.default_take_profit_pcts:
            if pct <= 0:
                raise ValueError("Take profit percents must be greater than 0")
        
        if self.execution_interval <= 0:
            raise ValueError("Execution interval must be greater than 0")
        
        if self.max_concurrent_orders <= 0:
            raise ValueError("Max concurrent orders must be greater than 0")
        
        if self.temperature_coefficient <= 0:
            raise ValueError("Temperature coefficient must be greater than 0")
        
        if self.risk_check_interval <= 0:
            raise ValueError("Risk check interval must be greater than 0")
        
        if self.analysis_interval <= 0:
            raise ValueError("Analysis interval must be greater than 0")
        if self.report_interval <= 0:
            raise ValueError("Report interval must be greater than 0")
        if self.min_signal_confidence < 0 or self.min_signal_confidence > 1:
            raise ValueError("Min signal confidence must be between 0 and 1")
        if self.min_signal_interval <= 0:
            raise ValueError("Min signal interval must be greater than 0")

        # 验证时间框架
        valid_timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '4h', '1d']
        for tf in self.timeframes:
            if tf not in valid_timeframes:
                raise ValueError(f"Invalid timeframe: {tf}")
        
        if self.primary_timeframe not in valid_timeframes:
            raise ValueError(f"Invalid primary timeframe: {self.primary_timeframe}")
        
        if self.higher_tf_bias_tf not in valid_timeframes:
            raise ValueError(f"Invalid higher TF bias timeframe: {self.higher_tf_bias_tf}")
        
        if self.lower_tf_entry_tf not in valid_timeframes:
            raise ValueError(f"Invalid lower TF entry timeframe: {self.lower_tf_entry_tf}")
        
        # 验证Kill Zone时间
        if self.kill_zone_start_utc < 0 or self.kill_zone_start_utc > 23:
            raise ValueError("Kill Zone start UTC must be between 0 and 23")
        
        if self.kill_zone_end_utc < 0 or self.kill_zone_end_utc > 23:
            raise ValueError("Kill Zone end UTC must be between 0 and 23")
        
        # 验证SMC参数
        if self.smc_window <= 0:
            raise ValueError("SMC window must be greater than 0")
        
        if self.smc_range_percent <= 0:
            raise ValueError("SMC range percent must be greater than 0")
        
        if self.min_structure_score < 0 or self.min_structure_score > 1:
            raise ValueError("Min structure score must be between 0 and 1")
        
        # 验证结构权重
        total_weight = sum(self.structure_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Structure weights must sum to 1.0, got {total_weight}")
        
        for weight in self.structure_weights.values():
            if weight < 0 or weight > 1:
                raise ValueError("Structure weights must be between 0 and 1")