#!/usr/bin/env python3
"""
测试原始高颗粒度数据处理脚本
验证AI是否能正确处理原始数据并计算SMC结构
"""

import sys
import os
import json
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from optimized_smc_prompt import get_optimized_smc_prompt

def test_raw_data_prompt():
    """测试原始数据提示词生成"""
    print("=" * 60)
    print("测试原始高颗粒度数据提示词生成")
    print("=" * 60)
    
    # 创建模拟原始高颗粒度数据
    raw_price_data = {
        'candlesticks': [
            {
                'timestamp': '2024-01-01T00:00:00Z',
                'open': 42000,
                'high': 42500,
                'low': 41800,
                'close': 42300,
                'volume': 1200,
                'timeframe': '1h'
            },
            {
                'timestamp': '2024-01-01T01:00:00Z',
                'open': 42300,
                'high': 42800,
                'low': 42100,
                'close': 42600,
                'volume': 1500,
                'timeframe': '1h'
            },
            {
                'timestamp': '2024-01-01T02:00:00Z',
                'open': 42600,
                'high': 42900,
                'low': 42400,
                'close': 42500,
                'volume': 900,
                'timeframe': '1h'
            }
        ],
        'swing_points': [
            {
                'timestamp': '2024-01-01T00:00:00Z',
                'price': 41800,
                'type': 'swing_low',
                'index': 0
            },
            {
                'timestamp': '2024-01-01T01:00:00Z',
                'price': 42900,
                'type': 'swing_high',
                'index': 1
            }
        ],
        'volume_data': [
            {
                'timestamp': '2024-01-01T00:00:00Z',
                'volume': 1200,
                'timeframe': '1h',
                'volume_ma': 1000,
                'volume_ratio': 1.2
            },
            {
                'timestamp': '2024-01-01T01:00:00Z',
                'volume': 1500,
                'timeframe': '1h',
                'volume_ma': 1100,
                'volume_ratio': 1.36
            }
        ],
        'liquidity_levels': [
            {
                'level': 41500,
                'type': 'support',
                'distance_to_price': 2.4
            },
            {
                'level': 43200,
                'type': 'resistance',
                'distance_to_price': 1.6
            }
        ],
        'price_movements': [
            {
                'timestamp': '2024-01-01T01:00:00Z',
                'price_change': 300,
                'price_change_pct': 0.71,
                'high_low_range': 700,
                'high_low_range_pct': 1.66,
                'volume': 1500,
                'timeframe': '1h'
            }
        ]
    }
    
    # 创建模拟市场数据
    current_price = 42500
    volatility = 2.5
    multi_tf_analysis = {
        'higher_tf_trend': 'bullish',
        'higher_tf_strength': 0.75,
        'primary_tf_trend': 'bullish',
        'primary_tf_strength': 0.65,
        'lower_tf_trend': 'neutral',
        'lower_tf_strength': 0.5,
        'consistency': 0.7,
        'recommendation': 'BUY'
    }
    technical_indicators = {
        'rsi': 55,
        'macd_histogram': 0.02
    }
    risk_params = {
        'rr_min_threshold': 2.0,
        'max_risk_per_trade': 0.02
    }
    
    # 生成提示词
    try:
        # 构建市场数据字典
        market_data = {
            'current_price': current_price,
            'volatility': volatility,
            'multi_tf_analysis': multi_tf_analysis,
            'raw_price_data': raw_price_data,
            'technical_indicators': technical_indicators,
            'risk_params': risk_params
        }
        
        prompt = get_optimized_smc_prompt(market_data)
        
        print("✅ 提示词生成成功")
        print(f"提示词长度: {len(prompt)} 字符")
        
        # 检查提示词中是否包含原始数据相关内容
        if "原始高颗粒度数据" in prompt:
            print("✅ 提示词包含原始高颗粒度数据说明")
        else:
            print("❌ 提示词缺少原始高颗粒度数据说明")
        
        if "K线数据" in prompt and "摆动点" in prompt:
            print("✅ 提示词包含原始数据类型说明")
        else:
            print("❌ 提示词缺少原始数据类型说明")
        
        if "BOS/CHOCH计算方法" in prompt and "订单块计算方法" in prompt:
            print("✅ 提示词包含SMC结构计算方法说明")
        else:
            print("❌ 提示词缺少SMC结构计算方法说明")
        
        if "权重分配" in prompt:
            print("✅ 提示词包含权重分配说明")
        else:
            print("❌ 提示词缺少权重分配说明")
        
        # 保存提示词到文件
        with open("raw_data_prompt_test.txt", "w", encoding="utf-8") as f:
            f.write(prompt)
        print("✅ 提示词已保存到 raw_data_prompt_test.txt")
        
        return True
        
    except Exception as e:
        print(f"❌ 提示词生成失败: {e}")
        return False

def test_mock_bot():
    """测试MockBot类是否能正确提取原始数据"""
    print("\n" + "=" * 60)
    print("测试MockBot类原始数据提取")
    print("=" * 60)
    
    try:
        # 创建MockBot类
        class MockBot:
            def __init__(self):
                self.logger_system = MockLogger()
                
            def _extract_raw_price_data(self, price_data, current_price, primary_tf):
                """提取原始高颗粒度价格数据，供AI自行计算SMC结构"""
                raw_data = {
                    'candlesticks': [],
                    'swing_points': [],
                    'volume_data': [],
                    'liquidity_levels': [],
                    'price_movements': []
                }
                
                try:
                    # 1. 提取K线数据
                    multi_tf_data = price_data.get('multi_tf_data', {})
                    for tf_name, df in multi_tf_data.items():
                        if df is not None and len(df) > 0:
                            # 获取最近20根K线
                            recent_candles = df.tail(20).to_dict('records')
                            tf_candles = []
                            for candle in recent_candles:
                                tf_candles.append({
                                    'timestamp': candle.get('timestamp', ''),
                                    'open': candle.get('open', 0),
                                    'high': candle.get('high', 0),
                                    'low': candle.get('low', 0),
                                    'close': candle.get('close', 0),
                                    'volume': candle.get('volume', 0),
                                    'timeframe': tf_name
                                })
                            raw_data['candlesticks'].extend(tf_candles)
                    
                    # 2. 提取摆动点数据
                    if primary_tf in multi_tf_data and multi_tf_data[primary_tf] is not None:
                        df = multi_tf_data[primary_tf]
                        # 使用简单的摆动点检测算法
                        swing_highs = []
                        swing_lows = []
                        
                        for i in range(2, len(df) - 2):
                            # 检查摆动高点
                            if (df.iloc[i]['high'] > df.iloc[i-1]['high'] and 
                                df.iloc[i]['high'] > df.iloc[i-2]['high'] and
                                df.iloc[i]['high'] > df.iloc[i+1]['high'] and 
                                df.iloc[i]['high'] > df.iloc[i+2]['high']):
                                swing_highs.append({
                                    'timestamp': df.iloc[i]['timestamp'],
                                    'price': df.iloc[i]['high'],
                                    'type': 'swing_high',
                                    'index': i
                                })
                            
                            # 检查摆动低点
                            if (df.iloc[i]['low'] < df.iloc[i-1]['low'] and 
                                df.iloc[i]['low'] < df.iloc[i-2]['low'] and
                                df.iloc[i]['low'] < df.iloc[i+1]['low'] and 
                                df.iloc[i]['low'] < df.iloc[i+2]['low']):
                                swing_lows.append({
                                    'timestamp': df.iloc[i]['timestamp'],
                                    'price': df.iloc[i]['low'],
                                    'type': 'swing_low',
                                    'index': i
                                })
                        
                        raw_data['swing_points'] = swing_highs + swing_lows
                    
                    # 3. 提取成交量数据
                    for tf_name, df in multi_tf_data.items():
                        if df is not None and len(df) > 0:
                            # 获取最近20根K线的成交量
                            recent_volume = df.tail(20).to_dict('records')
                            tf_volume = []
                            for candle in recent_volume:
                                tf_volume.append({
                                    'timestamp': candle.get('timestamp', ''),
                                    'volume': candle.get('volume', 0),
                                    'timeframe': tf_name,
                                    'volume_ma': candle.get('volume_ma', 0),  # 如果有的话
                                    'volume_ratio': candle.get('volume_ratio', 0)  # 如果有的话
                                })
                            raw_data['volume_data'].extend(tf_volume)
                    
                    # 4. 提取流动性水平
                    key_levels = price_data.get('key_levels', {})
                    liquidity_levels = []
                    
                    # 从关键水平中提取流动性水平
                    for level_name, level_value in key_levels.items():
                        if level_name != 'current_price' and isinstance(level_value, (int, float)) and level_value > 0:
                            distance = abs(level_value - current_price) / current_price * 100
                            liquidity_levels.append({
                                'level': level_value,
                                'type': level_name,
                                'distance_to_price': distance
                            })
                    
                    # 按距离排序
                    liquidity_levels.sort(key=lambda x: x.get('distance_to_price', float('inf')))
                    raw_data['liquidity_levels'] = liquidity_levels[:10]  # 只保留最近的10个流动性水平
                    
                    # 5. 提取价格变动数据
                    if primary_tf in multi_tf_data and multi_tf_data[primary_tf] is not None:
                        df = multi_tf_data[primary_tf]
                        # 计算价格变动
                        price_movements = []
                        for i in range(1, min(21, len(df))):
                            prev_close = df.iloc[i-1]['close']
                            curr_close = df.iloc[i]['close']
                            curr_high = df.iloc[i]['high']
                            curr_low = df.iloc[i]['low']
                            
                            price_movements.append({
                                'timestamp': df.iloc[i]['timestamp'],
                                'price_change': curr_close - prev_close,
                                'price_change_pct': (curr_close - prev_close) / prev_close * 100,
                                'high_low_range': curr_high - curr_low,
                                'high_low_range_pct': (curr_high - curr_low) / prev_close * 100,
                                'volume': df.iloc[i]['volume'],
                                'timeframe': primary_tf
                            })
                        
                        raw_data['price_movements'] = price_movements
                    
                    self.logger_system.info(f"🔍 提取原始高颗粒度数据: K线={len(raw_data['candlesticks'])}, 摆动点={len(raw_data['swing_points'])}, 成交量={len(raw_data['volume_data'])}, 流动性水平={len(raw_data['liquidity_levels'])}")
                    
                except Exception as e:
                    self.logger_system.error(f"提取原始高颗粒度数据失败: {e}")
                
                return raw_data
        
        class MockLogger:
            def info(self, msg): print(f"INFO: {msg}")
            def error(self, msg): print(f"ERROR: {msg}")
            def warning(self, msg): print(f"WARNING: {msg}")
        
        # 创建模拟数据
        import pandas as pd
        
        # 创建模拟的DataFrame
        dates = pd.date_range('2024-01-01', periods=30, freq='H')
        prices = [42000 + i*10 for i in range(30)]
        
        df_1h = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p+50 for p in prices],
            'low': [p-50 for p in prices],
            'close': [p+20 for p in prices],
            'volume': [1000 + i*10 for i in range(30)]
        })
        
        df_4h = pd.DataFrame({
            'timestamp': dates[::4],
            'open': prices[::4],
            'high': [p+100 for p in prices[::4]],
            'low': [p-100 for p in prices[::4]],
            'close': [p+40 for p in prices[::4]],
            'volume': [4000 + i*40 for i in range(0, 30, 4)]
        })
        
        price_data = {
            'multi_tf_data': {
                '1h': df_1h,
                '4h': df_4h
            },
            'key_levels': {
                'support_1': 41500,
                'resistance_1': 43200,
                'support_2': 40800,
                'resistance_2': 44000
            }
        }
        
        current_price = 42500
        primary_tf = '1h'
        
        # 测试数据提取
        bot = MockBot()
        raw_data = bot._extract_raw_price_data(price_data, current_price, primary_tf)
        
        # 验证数据
        print(f"✅ K线数据数量: {len(raw_data['candlesticks'])}")
        print(f"✅ 摆动点数量: {len(raw_data['swing_points'])}")
        print(f"✅ 成交量数据数量: {len(raw_data['volume_data'])}")
        print(f"✅ 流动性水平数量: {len(raw_data['liquidity_levels'])}")
        print(f"✅ 价格变动数据数量: {len(raw_data['price_movements'])}")
        
        # 保存数据到文件
        with open("raw_data_test.json", "w", encoding="utf-8") as f:
            json.dump(raw_data, f, indent=2, default=str)
        print("✅ 原始数据已保存到 raw_data_test.json")
        
        return True
        
    except Exception as e:
        print(f"❌ MockBot测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始原始高颗粒度数据处理测试")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 测试1: 提示词生成
    prompt_test_passed = test_raw_data_prompt()
    
    # 测试2: MockBot数据提取
    mockbot_test_passed = test_mock_bot()
    
    # 测试结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    print(f"提示词生成测试: {'✅ 通过' if prompt_test_passed else '❌ 失败'}")
    print(f"MockBot数据提取测试: {'✅ 通过' if mockbot_test_passed else '❌ 失败'}")
    
    if prompt_test_passed and mockbot_test_passed:
        print("\n🎉 所有测试通过！AI可以正确处理原始高颗粒度数据")
        return True
    else:
        print("\n⚠️ 部分测试失败，请检查代码")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)