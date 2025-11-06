#!/usr/bin/env python3
"""
增强版交易机器人测试脚本
测试增强版交易机器人的各项功能，包括数据获取、信号计算和交易决策
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, List

# 导入增强版交易机器人相关模块
sys.path.append('/Users/zhidafu/ds交易/ds/运行测试/测试部署')
from enhanced_data_extractor import EnhancedDataExtractor
from enhanced_smc_signal_calculator import EnhancedSMCSignalCalculator
from enhanced_trading_bot import EnhancedConfig, EnhancedTradingBot

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_config():
    """创建示例配置"""
    return EnhancedConfig(
        symbol='BTC/USDC:USDC',
        amount=0.01,
        leverage=40,
        primary_timeframe='15m',
        data_points=200,
        enable_enhanced_data=True,
        simulation_mode=True  # 启用模拟模式
    )

def create_sample_ohlcv_data(num_candles=50):
    """创建示例OHLCV数据"""
    # 生成随机价格数据
    base_price = 115000.0
    prices = [base_price]
    
    for _ in range(num_candles - 1):
        # 随机波动 -0.5% 到 +0.5%
        change = np.random.uniform(-0.005, 0.005)
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # 生成OHLCV数据
    ohlcv_data = []
    for i, price in enumerate(prices):
        high = price * np.random.uniform(1.0, 1.002)
        low = price * np.random.uniform(0.998, 1.0)
        
        # 确保开盘价在前一根K线的范围内
        if i > 0:
            prev_close = ohlcv_data[i-1][4]
            open_price = prev_close * np.random.uniform(0.999, 1.001)
            open_price = max(min(open_price, high), low)
        else:
            open_price = price
        
        close = price * np.random.uniform(0.999, 1.001)
        close = max(min(close, high), low)
        
        volume = np.random.uniform(100, 1000)
        
        ohlcv_data.append([
            int(datetime.now().timestamp() - (num_candles - i) * 60 * 15),  # 时间戳
            open_price,
            high,
            low,
            close,
            volume
        ])
    
    return ohlcv_data

def test_enhanced_data_extraction():
    """测试增强版数据提取"""
    logger.info("--- 测试1: 增强版数据提取 ---")
    
    # 创建数据提取器
    data_extractor = EnhancedDataExtractor()
    
    # 创建示例数据
    ohlcv_data = create_sample_ohlcv_data(50)
    volume_data = [candle[5] for candle in ohlcv_data]
    
    # 创建市场深度数据
    market_depth = {
        'bids': [[115000 - i*10, i+1] for i in range(10)],
        'asks': [[115000 + i*10, i+1] for i in range(10)]
    }
    
    # 创建时间销售数据
    time_sales = []
    for i in range(100):
        price = 115000 + np.random.uniform(-100, 100)
        amount = np.random.uniform(0.001, 0.01)
        side = 'buy' if np.random.random() > 0.5 else 'sell'
        time_sales.append({
            'timestamp': int(datetime.now().timestamp()) - i,
            'price': price,
            'amount': amount,
            'side': side
        })
    
    # 创建市场情绪数据
    market_sentiment = {
        'fear_greed_index': np.random.uniform(0, 100),
        'funding_rate': np.random.uniform(-0.01, 0.01),
        'long_short_ratio': np.random.uniform(0.5, 2.0)
    }
    
    # 转换OHLCV数据为字典格式
    ohlc_data = []
    for candle in ohlcv_data:
        ohlc_data.append({
            'timestamp': candle[0],
            'open': candle[1],
            'high': candle[2],
            'low': candle[3],
            'close': candle[4],
            'volume': candle[5]
        })
    
    # 转换成交量数据为字典格式
    volume_data = []
    for i, volume in enumerate([candle[5] for candle in ohlcv_data]):
        volume_data.append({
            'timestamp': ohlcv_data[i][0],
            'volume': volume
        })
    
    # 提取增强版数据
    enhanced_data = data_extractor.extract_enhanced_raw_data(
        ohlc_data=ohlc_data,
        volume_data=volume_data,
        market_depth=market_depth,
        time_sales=time_sales,
        market_sentiment=market_sentiment
    )
    
    logger.info(f"✅ 增强版数据提取完成")
    logger.info(f"K线数据: {len(enhanced_data.get('enhanced_candlesticks', []))}")
    logger.info(f"摆动点: {len(enhanced_data.get('swing_points', []))}")
    logger.info(f"市场深度: {len(enhanced_data.get('market_depth', {}).get('bids', []))}")
    logger.info(f"时间销售: {len(enhanced_data.get('time_sales', []))}")
    
    return enhanced_data

def test_enhanced_smc_signal_calculation(enhanced_data):
    """测试增强版SMC信号计算"""
    logger.info("--- 测试2: 增强版SMC信号计算 ---")
    
    # 创建配置
    config = create_sample_config()
    
    # 创建SMC信号计算器
    smc_calculator = EnhancedSMCSignalCalculator(config)
    
    # 计算增强版SMC信号
    signal_result = smc_calculator.calculate_enhanced_smc_signal(enhanced_data)
    
    logger.info(f"✅ 增强版SMC信号计算完成: {signal_result.get('signal', 'UNKNOWN')} (置信度: {signal_result.get('confidence', 0):.2f})")
    logger.info(f"原因: {signal_result.get('reason', '未知')}")
    
    # 输出细分信号
    if 'signal_breakdown' in signal_result:
        logger.info("信号细分:")
        for signal_type, signal_info in signal_result['signal_breakdown'].items():
            if isinstance(signal_info, dict):
                signal = signal_info.get('signal', 'UNKNOWN')
                confidence = signal_info.get('confidence', 0)
                logger.info(f"  {signal_type}: {signal} (置信度: {confidence:.2f})")
    
    return signal_result

def test_enhanced_trading_bot():
    """测试增强版交易机器人"""
    logger.info("--- 测试3: 增强版交易机器人 ---")
    
    # 创建配置
    config = create_sample_config()
    
    # 创建交易机器人（不传入exchange，使用模拟模式）
    bot = EnhancedTradingBot(config, exchange=None)
    
    # 创建示例数据
    ohlcv_data = create_sample_ohlcv_data(50)
    
    # 转换OHLCV数据为字典格式
    ohlc_data = []
    for candle in ohlcv_data:
        ohlc_data.append({
            'timestamp': candle[0],
            'open': candle[1],
            'high': candle[2],
            'low': candle[3],
            'close': candle[4],
            'volume': candle[5]
        })
    
    # 转换成交量数据为字典格式
    volume_data = []
    for i, volume in enumerate([candle[5] for candle in ohlcv_data]):
        volume_data.append({
            'timestamp': ohlcv_data[i][0],
            'volume': volume
        })
    
    # 创建模拟的原始数据字典
    raw_data = {
        'ohlcv_data': ohlc_data,
        'volume_data': volume_data,
        'market_depth': {
            'bids': [[115000 - i*10, i+1] for i in range(10)],
            'asks': [[115000 + i*10, i+1] for i in range(10)]
        },
        'time_sales': [
            {
                'timestamp': int(datetime.now().timestamp()) - i,
                'price': 115000 + np.random.uniform(-100, 100),
                'amount': np.random.uniform(0.001, 0.01),
                'side': 'buy' if np.random.random() > 0.5 else 'sell'
            } for i in range(100)
        ],
        'market_sentiment': {
            'fear_greed_index': np.random.uniform(0, 100),
            'funding_rate': np.random.uniform(-0.01, 0.01),
            'long_short_ratio': np.random.uniform(0.5, 2.0)
        }
    }
    
    # 测试数据提取
    enhanced_data = bot.enhanced_data_extractor.extract_enhanced_raw_data(
        ohlc_data=raw_data['ohlcv_data'],
        volume_data=raw_data['volume_data'],
        market_depth=raw_data['market_depth'],
        time_sales=raw_data['time_sales'],
        market_sentiment=raw_data['market_sentiment']
    )
    
    logger.info(f"✅ 增强版数据提取完成")
    
    # 测试SMC信号计算
    signal_result = bot.enhanced_smc_calculator.calculate_enhanced_smc_signal(enhanced_data)
    
    logger.info(f"✅ 增强版SMC信号计算完成: {signal_result.get('signal', 'UNKNOWN')} (置信度: {signal_result.get('confidence', 0):.2f})")
    
    # 测试增强版SMC分析
    analysis_result = bot.analyze_with_enhanced_smc(enhanced_data, None)
    
    if analysis_result is None:
        logger.warning("增强版SMC分析返回None")
        analysis_result = {'signal': 'UNKNOWN', 'confidence': 0.0, 'reason': '分析失败'}
    
    logger.info(f"✅ 增强版SMC分析完成: {analysis_result.get('signal', 'UNKNOWN')} (置信度: {analysis_result.get('confidence', 0):.2f})")
    logger.info(f"原因: {analysis_result.get('reason', '未知')}")
    
    return analysis_result

def main():
    """主函数"""
    logger.info("=== 增强版交易机器人测试开始 ===")
    
    # 测试1: 增强版数据提取
    enhanced_data = test_enhanced_data_extraction()
    
    # 测试2: 增强版SMC信号计算
    signal_result = test_enhanced_smc_signal_calculation(enhanced_data)
    
    # 测试3: 增强版交易机器人
    analysis_result = test_enhanced_trading_bot()
    
    logger.info("=== 测试完成 ===")
    
    # 保存测试结果
    test_results = {
        'timestamp': datetime.now().isoformat(),
        'enhanced_data_extraction': {
            'status': 'completed',
            'data_points': len(enhanced_data.get('enhanced_candlesticks', []))
        },
        'smc_signal_calculation': {
            'status': 'completed',
            'signal': signal_result.get('signal', 'UNKNOWN'),
            'confidence': signal_result.get('confidence', 0)
        },
        'trading_bot_analysis': {
            'status': 'completed',
            'signal': analysis_result.get('signal', 'UNKNOWN'),
            'confidence': analysis_result.get('confidence', 0)
        }
    }
    
    with open('enhanced_trading_bot_test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    logger.info("测试结果已保存到 enhanced_trading_bot_test_results.json")

if __name__ == "__main__":
    main()