#!/usr/bin/env python3
"""
测试AI信号生成器
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 加载环境变量
from dotenv import load_dotenv
load_dotenv('1.env')

from ai_signal_generator import AISignalGenerator
from config import Config
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AISignalTest")

def test_ai_signal_generator():
    """测试AI信号生成器"""
    try:
        # 初始化配置
        config = Config()
        
        # 初始化AI信号生成器
        ai_signal_generator = AISignalGenerator(config, logger)
        
        # 创建模拟市场数据
        market_data = {
            "current_price": 2000.0,
            "technical_indicators": {
                "ema": {"ema_20": 1995.0, "ema_50": 1980.0, "ema_200": 1950.0},
                "rsi": {"rsi_14": 55.0},
                "macd": {"macd": 5.0, "signal": 3.0, "histogram": 2.0},
                "bollinger": {"upper": 2020.0, "middle": 2000.0, "lower": 1980.0},
                "volume": {"volume_sma": 1000000, "volume_ratio": 1.2}
            },
            "smc_structures": {
                "bos_choch": {"bos": True, "choch": False},
                "order_blocks": [{"type": "bullish", "high": 1990.0, "low": 1985.0}],
                "fvg": [{"type": "bullish", "top": 1992.0, "bottom": 1988.0}],
                "swing_points": [{"type": "high", "price": 2010.0}, {"type": "low", "price": 1970.0}]
            },
            "key_levels": {
                "support": [1980.0, 1950.0],
                "resistance": [2020.0, 2050.0],
                "ema": {"ema_20": 1995.0, "ema_50": 1980.0, "ema_200": 1950.0}
            },
            "price_action": {
                "candlestick_patterns": {"hammer": True},
                "price_efficiency": 0.7,
                "volatility": 0.3,
                "momentum": 0.6
            }
        }
        
        # 生成AI信号
        logger.info("生成AI信号...")
        ai_signals = ai_signal_generator.generate_signals(market_data)
        
        # 打印结果
        logger.info(f"AI信号结果: {ai_signals}")
        
        # 检查主要信号
        primary_signal = ai_signals.get("primary", {})
        signal_data = primary_signal.get("signal", {})
        signal_type = signal_data.get("signal", "HOLD")
        confidence = signal_data.get("confidence", 0.0)
        
        logger.info(f"主要信号: {signal_type}, 置信度: {confidence}")
        
        # 检查共识信号
        consensus_signal = ai_signals.get("consensus", "HOLD")
        consensus_confidence = ai_signals.get("consensus_confidence", 0.0)
        
        logger.info(f"共识信号: {consensus_signal}, 置信度: {consensus_confidence}")
        
        # 判断测试结果
        if signal_type != "HOLD" and confidence > 0.3:
            logger.info("✅ AI信号生成器测试成功！")
            return True
        else:
            logger.warning(f"⚠️ AI信号生成器生成了HOLD信号或置信度过低: {signal_type}, {confidence}")
            return False
            
    except Exception as e:
        logger.error(f"AI信号生成器测试失败: {e}")
        return False

if __name__ == "__main__":
    success = test_ai_signal_generator()
    sys.exit(0 if success else 1)