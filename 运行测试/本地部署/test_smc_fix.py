#!/usr/bin/env python3
"""
SMC数据结构异常修复测试脚本
"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['PYTHONWARNINGS'] = 'ignore'

import sys
import pandas as pd
import numpy as np
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 创建测试数据
def create_test_data():
    """创建更真实的测试数据"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
    np.random.seed(42)
    
    # 生成更真实的测试数据
    base_price = 45000
    price_data = []
    current_price = base_price
    
    for i in range(100):
        # 模拟真实的价格波动
        volatility = np.random.normal(0, 200)
        trend = i * 10  # 轻微上升趋势
        noise = np.random.normal(0, 50)
        
        current_price = base_price + trend + volatility + noise
        
        # 确保价格为正
        current_price = max(1000, current_price)
        
        high = current_price + np.random.uniform(50, 200)
        low = current_price - np.random.uniform(50, 200)
        volume = np.random.uniform(1000, 5000)
        
        price_data.append({
            'timestamp': dates[i],
            'open': current_price,
            'high': high,
            'low': low,
            'close': current_price,
            'volume': volume
        })
    
    df = pd.DataFrame(price_data)
    df.set_index('timestamp', inplace=True)
    return df

# 测试智能计算函数
def test_intelligent_calculations():
    """测试智能计算函数"""
    print('🧪 开始测试智能计算函数...')
    
    # 创建测试数据
    test_df = create_test_data()
    
    print(f'📊 测试数据形状: {test_df.shape}')
    print(f'📈 价格范围: {test_df["close"].min():.0f} - {test_df["close"].max():.0f}')
    
    # 导入智能计算函数
    try:
        # 动态导入btc_trading_bot模块
        import importlib.util
        spec = importlib.util.spec_from_file_location('btc_trading_bot', 'btc_trading_bot.py')
        bot_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bot_module)
        
        # 创建完整的配置对象
        class SimpleConfig:
            def __init__(self):
                self.smc_window = 20
                self.structure_weights = {
                    'bos_choch': 0.4,
                    'ob_fvg': 0.4,
                    'swing_strength': 0.2
                }
                self.signal_stabilizer_window = 30
                self.trend_consistency_threshold = 0.7
                self.simulation_mode = True
                self.fee_rate = 0.001
        
        # 创建TradingBot实例
        config = SimpleConfig()
        bot = bot_module.TradingBot(config)
        
        # 测试不同时间框架
        timeframes = ['1h', '4h', '1d']
        
        for tf in timeframes:
            print(f'\n⏰ 测试时间框架: {tf}')
            
            # 测试智能BOS计算
            try:
                atr = test_df['close'].std()  # 简化ATR计算
                bos_strength = bot._calculate_intelligent_bos_strength(test_df, tf, atr)
                print(f'✅ 智能BOS计算成功: {bos_strength:.2f}')
                
                # 检查是否在合理范围内
                if 0.1 <= bos_strength <= 3.0:
                    print(f'✅ BOS强度在合理范围内')
                else:
                    print(f'⚠️  BOS强度超出范围: {bos_strength:.2f}')
                    
            except Exception as e:
                print(f'❌ 智能BOS计算失败: {e}')
            
            # 测试智能OB计算
            try:
                ob_count = bot._calculate_intelligent_ob_count(test_df, tf)
                print(f'✅ 智能OB计算成功: {ob_count}')
                
                # 检查是否在合理范围内
                if 1 <= ob_count <= len(test_df) // 8:
                    print(f'✅ OB数量在合理范围内')
                else:
                    print(f'⚠️  OB数量超出范围: {ob_count}')
                    
            except Exception as e:
                print(f'❌ 智能OB计算失败: {e}')
        
        print('\n🎯 智能函数测试完成')
        
    except Exception as e:
        print(f'❌ 模块导入失败: {e}')
        import traceback
        traceback.print_exc()

# 测试SMC检测函数
def test_smc_detection():
    """测试SMC检测函数"""
    print('\n🧪 开始测试SMC检测函数...')
    
    # 创建测试数据
    test_df = create_test_data()
    
    # 导入SMC检测函数
    try:
        # 动态导入btc_trading_bot模块
        import importlib.util
        spec = importlib.util.spec_from_file_location('btc_trading_bot', 'btc_trading_bot.py')
        bot_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bot_module)
        
        # 创建完整的配置对象
        class SimpleConfig:
            def __init__(self):
                self.smc_window = 20
                self.structure_weights = {
                    'bos_choch': 0.4,
                    'ob_fvg': 0.4,
                    'swing_strength': 0.2
                }
                self.signal_stabilizer_window = 30
                self.trend_consistency_threshold = 0.7
                self.simulation_mode = True
                self.fee_rate = 0.001
        
        # 创建TradingBot实例
        config = SimpleConfig()
        bot = bot_module.TradingBot(config)
        
        # 测试不同时间框架
        timeframes = ['1h', '4h', '1d']
        
        for tf in timeframes:
            print(f'\n⏰ 测试时间框架: {tf}')
            
            try:
                # 调用SMC检测
                result = bot.detect_smc_structures(test_df, tf)
                
                if result:
                    print(f'✅ SMC检测成功')
                    
                    # 提取关键指标
                    bos_strength = result.get('bos_strength', 0)
                    fvg_count = result.get('fvg_count', 0)
                    ob_count = result.get('ob_count', 0)
                    strength_score = result.get('strength_score', 0)
                    
                    print(f'   BOS强度: {bos_strength:.2f}')
                    print(f'   FVG数量: {fvg_count}')
                    print(f'   OB数量: {ob_count}')
                    print(f'   结构强度: {strength_score:.2f}')
                    
                    # 检查是否还有固定数值模式
                    if abs(bos_strength - 1.50) < 0.01:
                        print(f'⚠️  检测到固定BOS模式: {bos_strength}')
                    else:
                        print(f'✅  BOS数据正常')
                        
                    # 检查OB数据是否合理
                    if ob_count > len(test_df) // 4 or ob_count < 1:
                        print(f'⚠️  OB数量异常: {ob_count}')
                    else:
                        print(f'✅  OB数据正常')
                        
                    # 检查整体数据合理性
                    if 0.1 <= bos_strength <= 3.0 and 1 <= ob_count <= len(test_df) // 4:
                        print(f'✅  整体数据合理')
                    else:
                        print(f'⚠️  部分数据超出合理范围')
                        
                else:
                    print(f'❌ SMC检测返回空结果')
                    
            except Exception as e:
                print(f'❌ SMC检测失败: {e}')
                import traceback
                traceback.print_exc()
        
        print('\n🎯 SMC检测测试完成')
        
    except Exception as e:
        print(f'❌ 模块导入失败: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 SMC数据结构异常修复测试")
    print("=" * 50)
    
    # 测试智能计算函数
    test_intelligent_calculations()
    
    # 测试SMC检测函数
    test_smc_detection()
    
    print("\n" + "=" * 50)
    print("🎯 测试完成")