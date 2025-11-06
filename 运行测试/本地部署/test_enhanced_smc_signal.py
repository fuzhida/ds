#!/usr/bin/env python3
"""
测试增强版SMC信号计算器
"""

import sys
import os
import json
import random
import logging
from datetime import datetime, timezone

# 添加测试部署路径
sys.path.append('/Users/zhidafu/ds交易/ds/运行测试/测试部署')
sys.path.append('/Users/zhidafu/ds交易/ds/运行测试/本地部署')

from enhanced_data_extractor import EnhancedDataExtractor
from enhanced_smc_signal_calculator import EnhancedSMCSignalCalculator
from enhanced_mock_bot import EnhancedMockBot

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_enhanced_data():
    """创建示例增强版数据"""
    # 创建模拟K线数据
    candlesticks = []
    base_price = 50000
    for i in range(50):
        price_change = random.uniform(-200, 200)
        open_price = base_price + price_change
        high_price = open_price + random.uniform(0, 100)
        low_price = open_price - random.uniform(0, 100)
        close_price = low_price + random.uniform(0, high_price - low_price)
        volume = random.uniform(100, 1000)
        
        candlesticks.append({
            'timestamp': (datetime.now(timezone.utc).timestamp() - (50-i) * 60) * 1000,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
        
        base_price = close_price
    
    # 创建摆动点数据
    swing_points = []
    for i in range(10, 40, 5):
        swing_points.append({
            'timestamp': candlesticks[i]['timestamp'],
            'price': candlesticks[i]['high'] if i % 2 == 0 else candlesticks[i]['low'],
            'type': 'high' if i % 2 == 0 else 'low',
            'strength': random.uniform(0.6, 0.9)
        })
    
    # 创建市场深度数据
    market_depth = []
    for i in range(10):
        bid_price = base_price - (i + 1) * 10
        ask_price = base_price + (i + 1) * 10
        market_depth.append({
            'bid_price': bid_price,
            'bid_volume': random.uniform(10, 100),
            'ask_price': ask_price,
            'ask_volume': random.uniform(10, 100)
        })
    
    # 创建时间与销售数据
    time_sales = []
    for i in range(50):
        price_change = random.uniform(-50, 50)
        price = base_price + price_change
        time_sales.append({
            'timestamp': (datetime.now(timezone.utc).timestamp() - (50-i) * 10) * 1000,
            'price': price,
            'volume': random.uniform(0.1, 10),
            'side': 'buy' if random.random() > 0.5 else 'sell'
        })
    
    # 计算ATR
    atr = sum([abs(c['high'] - c['low']) for c in candlesticks[-20:]]) / 20
    
    return {
        'enhanced_candlesticks': candlesticks,
        'swing_points': swing_points,
        'market_depth': market_depth,
        'time_sales': time_sales,
        'atr': atr
    }

def test_enhanced_smc_signal_calculator():
    """测试增强版SMC信号计算器"""
    logger.info("开始测试增强版SMC信号计算器...")
    
    # 创建配置
    config = {
        'enhanced_smc_signal_weights': {
            'bos_choch': 0.3,
            'order_blocks': 0.25,
            'fvg': 0.2,
            'liquidity': 0.15,
            'market_microstructure': 0.1
        },
        'enhanced_smc_min_confidence': 0.6,
        'enhanced_data_weight': 0.7,
        'market_depth_weight': 0.15,
        'time_sales_weight': 0.1,
        'market_sentiment_weight': 0.05
    }
    
    # 创建增强版SMC信号计算器
    calculator = EnhancedSMCSignalCalculator(config)
    
    # 创建示例数据
    enhanced_data = create_sample_enhanced_data()
    
    # 计算信号
    signal_result = calculator.calculate_enhanced_smc_signal(enhanced_data)
    
    # 输出结果
    logger.info("✅ 增强版SMC信号计算完成")
    logger.info(f"信号: {signal_result['signal']}")
    logger.info(f"置信度: {signal_result['confidence']:.2f}")
    logger.info(f"原因: {signal_result['reason']}")
    
    if signal_result['signal'] != 'HOLD':
        logger.info(f"止损: {signal_result['stop_loss']:.2f}")
        logger.info(f"止盈: {signal_result['take_profit']:.2f}")
        logger.info(f"风险回报比: {signal_result['risk_reward_ratio']:.2f}")
    
    logger.info("信号细分:")
    for component, data in signal_result['signal_breakdown'].items():
        logger.info(f"  {component}: {data['signal']} (置信度: {data['confidence']:.2f})")
    
    logger.info(f"强度评分: 买入={signal_result['strength_scores']['buy_strength']:.2f}, 卖出={signal_result['strength_scores']['sell_strength']:.2f}")
    
    return signal_result

def test_enhanced_data_integration():
    """测试增强版数据集成"""
    logger.info("开始测试增强版数据集成...")
    
    # 创建配置
    config = {
        'symbol': 'BTC/USDC:USDC',
        'timeframe': '1h',
        'enhanced_data_extraction': True,
        'enable_enhanced_ai_analysis': True
    }
    
    # 创建增强版数据提取器
    data_extractor = EnhancedDataExtractor()
    
    # 创建增强版MockBot
    mock_bot = EnhancedMockBot()
    
    # 生成模拟数据
    symbol = 'BTC/USDC:USDC'
    timeframe = '1h'
    limit = 50
    
    # 生成示例数据
    import random
    
    # 生成OHLC数据
    ohlc_data = []
    base_price = 42000
    for i in range(50):
        timestamp = f"2024-01-{(i%30)+1:02d}T{(i%24):02d}:00:00Z"
        open_price = base_price + random.uniform(-100, 100)
        close_price = open_price + random.uniform(-50, 50)
        high_price = max(open_price, close_price) + random.uniform(0, 50)
        low_price = min(open_price, close_price) - random.uniform(0, 50)
        volume = random.uniform(800, 1500)
        
        ohlc_data.append({
            "timestamp": timestamp,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume,
            "timeframe": "1h"
        })
        
        base_price = close_price
    
    # 生成市场深度数据
    market_depth = []
    for i in range(10):
        mid_price = 42000 + i * 10
        market_depth.append({
            "timestamp": f"2024-01-01T{(i%24):02d}:00:00Z",
            "bid_price": mid_price - 5,
            "ask_price": mid_price + 5,
            "bid_volume": random.uniform(100, 500),
            "ask_volume": random.uniform(100, 500)
        })
    
    # 生成时间与销售数据
    time_sales = []
    for i in range(100):
        time_sales.append({
            "timestamp": f"2024-01-01T{(i%24):02d}:{(i%60):02d}:{(i%60):02d}",
            "price": 42000 + random.uniform(-100, 100),
            "volume": random.uniform(0.1, 20),
            "side": random.choice(["buy", "sell"]),
            "aggressive": random.choice([True, False])
        })
    
    # 获取模拟原始数据
    raw_data = {
        'ohlc_data': ohlc_data,
        'volume_data': [],  # 简化处理
        'market_depth': market_depth,
        'time_sales': time_sales,
        'market_sentiment': {}  # 简化处理
    }
    
    # 提取增强版数据
    enhanced_data = data_extractor.extract_enhanced_raw_data(
        ohlc_data=raw_data['ohlc_data'],
        volume_data=raw_data['volume_data'],
        market_depth=raw_data['market_depth'],
        time_sales=raw_data['time_sales'],
        market_sentiment=raw_data['market_sentiment']
    )
    
    logger.info(f"✅ 增强版数据提取完成")
    logger.info(f"K线数据: {len(enhanced_data.get('enhanced_candlesticks', []))}")
    logger.info(f"摆动点: {len(enhanced_data.get('swing_points', []))}")
    logger.info(f"市场深度: {len(enhanced_data.get('market_depth', []))}")
    logger.info(f"时间销售: {len(enhanced_data.get('time_sales', []))}")
    
    # 创建增强版SMC信号计算器
    calculator = EnhancedSMCSignalCalculator(config)
    
    # 计算信号
    signal_result = calculator.calculate_enhanced_smc_signal(enhanced_data)
    
    logger.info(f"✅ 增强版SMC信号计算完成: {signal_result['signal']} (置信度: {signal_result['confidence']:.2f})")
    
    return enhanced_data, signal_result

def main():
    """主函数"""
    logger.info("=== 增强版SMC信号计算器测试 ===")
    
    # 测试1: 增强版SMC信号计算器
    logger.info("\n--- 测试1: 增强版SMC信号计算器 ---")
    signal_result = test_enhanced_smc_signal_calculator()
    
    # 测试2: 增强版数据集成
    logger.info("\n--- 测试2: 增强版数据集成 ---")
    enhanced_data, integrated_signal_result = test_enhanced_data_integration()
    
    # 保存测试结果
    test_results = {
        'signal_calculator_result': signal_result,
        'integrated_signal_result': integrated_signal_result,
        'enhanced_data_stats': {
            'candlesticks': len(enhanced_data.get('enhanced_candlesticks', [])),
            'swing_points': len(enhanced_data.get('swing_points', [])),
            'market_depth': len(enhanced_data.get('market_depth', [])),
            'time_sales': len(enhanced_data.get('time_sales', []))
        },
        'test_timestamp': datetime.now(timezone.utc).isoformat()
    }
    
    # 保存测试结果到文件
    with open('/Users/zhidafu/ds交易/ds/运行测试/本地部署/enhanced_smc_signal_test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    logger.info("\n=== 测试完成 ===")
    logger.info("测试结果已保存到 enhanced_smc_signal_test_results.json")
    
    # 比较两种信号计算方法的结果
    logger.info("\n--- 信号计算方法比较 ---")
    logger.info(f"独立信号计算: {signal_result['signal']} (置信度: {signal_result['confidence']:.2f})")
    logger.info(f"集成信号计算: {integrated_signal_result['signal']} (置信度: {integrated_signal_result['confidence']:.2f})")
    
    if signal_result['signal'] == integrated_signal_result['signal']:
        logger.info("✅ 两种方法计算结果一致")
    else:
        logger.info("⚠️ 两种方法计算结果不一致，这是正常的，因为使用了不同的随机数据")

if __name__ == "__main__":
    main()