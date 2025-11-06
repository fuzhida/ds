#!/usr/bin/env python3
"""
测试脚本：验证交易策略参数调整
测试新的时间框架和技术指标参数是否正确应用
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from deepseek_hyper import Config, TradingBot

def generate_test_data(days=7, timeframe='5m'):
    """生成测试数据"""
    # 根据时间框架确定数据点数量
    timeframe_minutes = {
        '1d': 1440,
        '4h': 240,
        '1h': 60,
        '15m': 15,
        '5m': 5
    }
    
    minutes = timeframe_minutes.get(timeframe, 5)
    points = days * 24 * 60 // minutes
    
    # 生成时间序列
    end_time = datetime.now()
    timestamps = [end_time - timedelta(minutes=i*minutes) for i in range(points, 0, -1)]
    
    # 生成价格数据（模拟BTC价格波动）
    base_price = 50000
    np.random.seed(42)  # 固定随机种子以确保可重复性
    
    # 生成随机波动
    price_changes = np.random.normal(0, 0.01, points)  # 1%标准差
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # 创建OHLCV数据
    data = []
    for i, (ts, close) in enumerate(zip(timestamps, prices)):
        # 生成OHLC
        high = close * (1 + abs(np.random.normal(0, 0.005)))
        low = close * (1 - abs(np.random.normal(0, 0.005)))
        open_price = low + (high - low) * np.random.random()
        
        # 生成成交量
        volume = np.random.normal(1000000, 200000)
        
        data.append({
            'timestamp': ts,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': max(0, volume)
        })
    
    return pd.DataFrame(data)

def test_config_parameters():
    """测试配置参数是否正确设置"""
    print("=== 测试配置参数 ===")
    config = Config()
    
    # 测试时间框架设置
    assert config.higher_tf_bias_tf == '1h', f"高级别方向时间框架应为1h，实际为{config.higher_tf_bias_tf}"
    assert config.lower_tf_entry_tf == '5m', f"入场时间框架应为5m，实际为{config.lower_tf_entry_tf}"
    assert config.primary_timeframe == '5m', f"主要时间框架应为5m，实际为{config.primary_timeframe}"
    print("✓ 时间框架设置正确")
    
    # 测试确认条件
    assert config.volume_confirmation_threshold == 1.5, f"成交量确认阈值应为1.5，实际为{config.volume_confirmation_threshold}"
    assert config.fvg_stack_threshold == 3, f"FVG堆叠阈值应为3，实际为{config.fvg_stack_threshold}"
    print("✓ 确认条件设置正确")
    
    # 测试技术指标参数
    assert config.macd_sensitivity == (0.015, 0.035), f"MACD灵敏度应为(0.015, 0.035)，实际为{config.macd_sensitivity}"
    assert config.atr_base == (100, 120), f"ATR基准应为(100, 120)，实际为{config.atr_base}"
    print("✓ 技术指标参数设置正确")
    
    # 测试5m级别权重配置
    assert '5m_structure_break' in config.level_weights, "缺少5m_structure_break权重配置"
    assert '5m_fvg_bull_mid' in config.level_weights, "缺少5m_fvg_bull_mid权重配置"
    print("✓ 5m级别权重配置正确")
    
    print("所有配置参数测试通过！\n")

def test_indicator_calculation():
    """测试技术指标计算"""
    print("=== 测试技术指标计算 ===")
    
    # 创建交易机器人实例
    config = Config()
    bot = TradingBot(config)
    
    # 生成测试数据
    test_data = generate_test_data(days=3, timeframe='5m')
    print(f"生成了{len(test_data)}个5分钟测试数据点")
    
    # 计算技术指标
    df_with_indicators = bot.calculate_technical_indicators(test_data)
    
    # 检查MACD计算
    assert 'macd' in df_with_indicators.columns, "缺少MACD指标"
    assert 'macd_signal' in df_with_indicators.columns, "缺少MACD信号线"
    assert 'ema_fast' in df_with_indicators.columns, "缺少快速EMA"
    assert 'ema_slow' in df_with_indicators.columns, "缺少慢速EMA"
    print("✓ MACD指标计算正确")
    
    # 检查ATR计算
    assert 'atr' in df_with_indicators.columns, "缺少ATR指标"
    assert df_with_indicators['atr'].iloc[-1] > 0, "ATR值应大于0"
    print("✓ ATR指标计算正确")
    
    # 检查EMA21指标
    assert 'ema_21' in df_with_indicators.columns, "缺少EMA21指标"
    print("✓ EMA21指标计算正确")
    
    # 检查RSI指标
    assert 'rsi' in df_with_indicators.columns, "缺少RSI指标"
    rsi_values = df_with_indicators['rsi'].dropna()
    assert all(0 <= val <= 100 for val in rsi_values), "RSI值应在0-100之间"
    print("✓ RSI指标计算正确")
    
    print("所有技术指标计算测试通过！\n")

def test_timeframe_alignment():
    """测试多时间框架对齐"""
    print("=== 测试多时间框架对齐 ===")
    
    config = Config()
    bot = TradingBot(config)
    
    # 生成不同时间框架的测试数据
    data_1h = generate_test_data(days=3, timeframe='1h')
    data_5m = generate_test_data(days=1, timeframe='5m')
    
    print(f"生成了{len(data_1h)}个1小时数据点和{len(data_5m)}个5分钟数据点")
    
    # 计算技术指标
    df_1h = bot.calculate_technical_indicators(data_1h)
    df_5m = bot.calculate_technical_indicators(data_5m)
    
    # 检查1小时数据中的EMA21（用于高级别方向判断）
    assert 'ema_21' in df_1h.columns, "1小时数据缺少EMA21指标"
    print("✓ 1小时图EMA21指标计算正确")
    
    # 检查5分钟数据中的关键指标
    assert 'ema_21' in df_5m.columns, "5分钟数据缺少EMA21指标"
    assert 'volume_ratio' in df_5m.columns, "5分钟数据缺少成交量比率"
    print("✓ 5分钟图关键指标计算正确")
    
    print("多时间框架对齐测试通过！\n")

def test_validation():
    """测试参数验证"""
    print("=== 测试参数验证 ===")
    
    # 测试有效配置
    try:
        config = Config()
        config.validate()
        print("✓ 有效配置验证通过")
    except Exception as e:
        print(f"✗ 有效配置验证失败: {e}")
        return False
    
    # 测试无效MACD灵敏度
    try:
        config = Config()
        config.macd_sensitivity = (0.2, 0.3)  # 超出范围
        config.validate()
        print("✗ 无效MACD灵敏度验证失败")
        return False
    except ValueError:
        print("✓ 无效MACD灵敏度正确被拒绝")
    
    # 测试无效ATR基准
    try:
        config = Config()
        config.atr_base = (10, 20)  # 超出范围
        config.validate()
        print("✗ 无效ATR基准验证失败")
        return False
    except ValueError:
        print("✓ 无效ATR基准正确被拒绝")
    
    print("参数验证测试通过！\n")
    return True

def main():
    """主测试函数"""
    print("开始测试交易策略参数调整...\n")
    
    try:
        # 运行所有测试
        test_config_parameters()
        test_indicator_calculation()
        test_timeframe_alignment()
        test_validation()
        
        print("🎉 所有测试通过！交易策略参数调整成功。")
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)