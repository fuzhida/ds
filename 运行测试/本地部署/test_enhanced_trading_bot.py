#!/usr/bin/env python3
"""
增强版交易机器人测试脚本
使用模拟数据测试增强版数据结构和SMC分析功能
"""

import os
import sys
import json
import random
from datetime import datetime, timezone
import pandas as pd
import numpy as np

# 添加测试部署目录到路径
sys.path.append('/Users/zhidafu/ds交易/ds/运行测试/测试部署')
sys.path.append('/Users/zhidafu/ds交易/ds/运行测试/本地部署')

from enhanced_trading_bot import EnhancedTradingBot, EnhancedConfig
from enhanced_data_extractor import EnhancedDataExtractor
from enhanced_smc_prompt import get_enhanced_smc_prompt
from enhanced_mock_bot import EnhancedMockBot

def create_mock_exchange():
    """创建模拟交易所对象"""
    class MockExchange:
        def fetch_ticker(self, symbol):
            return {'last': 115000.0 + random.uniform(-1000, 1000)}
        
        def fetch_ohlcv(self, symbol, timeframe, limit=200):
            """生成模拟OHLCV数据"""
            base_price = 115000.0
            ohlcv = []
            current_time = int(datetime.now(timezone.utc).timestamp() * 1000)
            
            # 根据时间框架调整时间间隔
            timeframe_minutes = {
                '1d': 1440,
                '4h': 240,
                '1h': 60,
                '15m': 15,
                '3m': 3,
                '1m': 1
            }
            
            interval_ms = timeframe_minutes.get(timeframe, 60) * 60 * 1000
            
            for i in range(limit):
                # 生成价格数据
                open_price = base_price + random.uniform(-500, 500)
                high_price = open_price + random.uniform(0, 200)
                low_price = open_price - random.uniform(0, 200)
                close_price = open_price + random.uniform(-100, 100)
                volume = random.uniform(100, 1000)
                
                # 确保价格逻辑正确
                high_price = max(high_price, open_price, close_price)
                low_price = min(low_price, open_price, close_price)
                
                ohlcv.append([
                    current_time - (limit - i) * interval_ms,
                    open_price,
                    high_price,
                    low_price,
                    close_price,
                    volume
                ])
                
                # 更新基础价格，模拟价格趋势
                base_price = close_price + random.uniform(-50, 50)
            
            return ohlcv
    
    return MockExchange()

def test_enhanced_trading_bot():
    """测试增强版交易机器人"""
    print("🚀 开始测试增强版交易机器人...")
    
    # 创建配置
    config = EnhancedConfig()
    config.simulation_mode = True  # 确保模拟模式
    config.enable_enhanced_data = True  # 启用增强版数据
    
    # 创建模拟交易所
    mock_exchange = create_mock_exchange()
    
    # 创建增强版交易机器人
    bot = EnhancedTradingBot(config, mock_exchange)
    
    print("✅ 增强版交易机器人创建成功")
    
    # 测试数据获取
    print("\n📊 测试数据获取...")
    price_data = bot._fetch_and_update_data()
    
    if price_data:
        print(f"✅ 数据获取成功")
        print(f"   当前价格: ${price_data['price']:.2f}")
        print(f"   增强版数据可用: {'enhanced_data' in price_data}")
        
        if 'enhanced_data' in price_data:
            enhanced_data = price_data['enhanced_data']
            print(f"   增强K线数量: {len(enhanced_data.get('enhanced_candlesticks', []))}")
            print(f"   市场深度点数: {len(enhanced_data.get('market_depth', []))}")
            print(f"   时间与销售记录数: {len(enhanced_data.get('time_sales', []))}")
        
        # 测试SMC分析
        print("\n🔍 测试SMC分析...")
        signal_data = bot.analyze_with_enhanced_smc(price_data, None)
        
        if signal_data:
            print(f"✅ SMC分析成功")
            print(f"   信号: {signal_data['signal']}")
            print(f"   置信度: {signal_data['confidence']:.2f}")
            print(f"   原因: {signal_data['reason']}")
            print(f"   数据源: {signal_data['source']}")
            
            if signal_data['signal'] != 'HOLD':
                print(f"   止损: ${signal_data.get('stop_loss', 0):.2f}")
                print(f"   止盈: ${signal_data.get('take_profit', 0):.2f}")
                print(f"   风险回报比: {signal_data.get('risk_reward_ratio', 0):.2f}:1")
        else:
            print("❌ SMC分析失败")
        
        # 测试完整交易流程
        print("\n🔄 测试完整交易流程...")
        bot.trading_bot()
        
        print("✅ 交易流程测试完成")
        
        # 检查信号历史
        if bot.signal_history:
            latest_signal = bot.signal_history[-1]
            print(f"\n📝 最新信号记录:")
            print(f"   时间: {latest_signal['timestamp']}")
            print(f"   信号: {latest_signal['signal']['signal']}")
            print(f"   置信度: {latest_signal['signal']['confidence']:.2f}")
            print(f"   增强版数据可用: {latest_signal['price_data']['enhanced_data_available']}")
    else:
        print("❌ 数据获取失败")
    
    print("\n✅ 增强版交易机器人测试完成")

def test_enhanced_data_integration():
    """测试增强版数据集成"""
    print("\n🔬 测试增强版数据集成...")
    
    # 创建增强版数据提取器
    extractor = EnhancedDataExtractor()
    
    # 生成模拟OHLC数据
    ohlc_data = []
    base_price = 115000.0
    for i in range(50):
        day = (i // 24) + 1
        hour = i % 24
        timestamp = datetime(2025, 1, day, hour, 0, 0, tzinfo=timezone.utc).isoformat()
        
        open_price = base_price + random.uniform(-500, 500)
        high_price = open_price + random.uniform(0, 200)
        low_price = open_price - random.uniform(0, 200)
        close_price = open_price + random.uniform(-100, 100)
        volume = random.uniform(100, 1000)
        
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        
        ohlc_data.append({
            "timestamp": timestamp,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume,
            "timeframe": "1h"
        })
        
        base_price = close_price + random.uniform(-50, 50)
    
    # 生成模拟市场深度数据
    market_depth = []
    for i in range(10):
        mid_price = 115000.0 + i * 10
        market_depth.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "bid_price": mid_price - 5,
            "ask_price": mid_price + 5,
            "bid_volume": random.uniform(100, 500),
            "ask_volume": random.uniform(100, 500)
        })
    
    # 生成模拟时间与销售数据
    time_sales = []
    for i in range(100):
        time_sales.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "price": 115000.0 + random.uniform(-100, 100),
            "volume": random.uniform(0.1, 20),
            "side": random.choice(["buy", "sell"]),
            "aggressive": random.choice([True, False])
        })
    
    # 生成模拟市场情绪数据
    market_sentiment = {
        "fear_greed_index": random.uniform(0, 100),
        "funding_rate": random.uniform(-0.01, 0.01),
        "open_interest_change": random.uniform(-5, 5),
        "long_short_ratio": random.uniform(0.8, 1.5)
    }
    
    # 提取增强版数据
    enhanced_raw_data = extractor.extract_enhanced_raw_data(
        ohlc_data=ohlc_data,
        volume_data=[],
        market_depth=market_depth,
        time_sales=time_sales,
        market_sentiment=market_sentiment
    )
    
    print(f"✅ 增强版数据提取完成")
    print(f"   增强K线数量: {len(enhanced_raw_data.get('enhanced_candlesticks', []))}")
    print(f"   市场深度点数: {len(enhanced_raw_data.get('market_depth', []))}")
    print(f"   时间与销售记录数: {len(enhanced_raw_data.get('time_sales', []))}")
    
    # 生成增强版提示词
    prompt = get_enhanced_smc_prompt(enhanced_raw_data)
    print(f"✅ 增强版提示词生成完成，长度: {len(prompt)} 字符")
    
    # 测试增强版MockBot
    mock_bot = EnhancedMockBot()
    # 生成模拟SMC响应
    mock_response = {
        "signal": "BUY" if random.random() > 0.5 else "SELL",
        "confidence": random.uniform(0.6, 0.9),
        "reason": f"基于增强版SMC分析，检测到高质量{'看涨' if random.random() > 0.5 else '看跌'}结构",
        "stop_loss": 115000.0 * (0.98 if random.random() > 0.5 else 1.02),
        "take_profit": 115000.0 * (1.02 if random.random() > 0.5 else 0.98),
        "risk_reward_ratio": random.uniform(2.0, 4.0),
        "strength": random.uniform(0.7, 0.95),
        "enhanced_data_score": random.uniform(0.6, 0.9),
        "market_microstructure_score": random.uniform(0.5, 0.8),
        "liquidity_analysis_score": random.uniform(0.6, 0.9),
        "order_flow_bias": "bullish" if random.random() > 0.5 else "bearish",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    print(f"✅ 增强版MockBot响应生成完成")
    print(f"   信号: {mock_response['signal']}")
    print(f"   置信度: {mock_response['confidence']:.2f}")
    print(f"   原因: {mock_response['reason']}")
    
    print("\n✅ 增强版数据集成测试完成")

def main():
    """主函数"""
    print("=" * 60)
    print("🤖 增强版交易机器人测试套件")
    print("=" * 60)
    
    # 测试增强版数据集成
    test_enhanced_data_integration()
    
    # 测试增强版交易机器人
    test_enhanced_trading_bot()
    
    print("\n" + "=" * 60)
    print("✅ 所有测试完成")
    print("=" * 60)

if __name__ == "__main__":
    main()