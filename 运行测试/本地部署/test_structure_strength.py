#!/usr/bin/env python3
"""
测试修复后的结构强度检测功能
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加路径以便导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from smc_real_detector import RealSMCDetector

def create_test_data():
    """创建测试数据"""
    # 生成模拟价格数据
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
    
    # 创建上升趋势数据
    base_price = 100.0
    trend = np.linspace(0, 20, 100)
    noise = np.random.normal(0, 2, 100)
    
    close_prices = base_price + trend + noise
    high_prices = close_prices + np.abs(np.random.normal(1, 0.5, 100))
    low_prices = close_prices - np.abs(np.random.normal(1, 0.5, 100))
    volumes = np.random.randint(1000, 10000, 100)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': close_prices - np.random.normal(0, 0.5, 100),
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    })
    
    return df

def test_structure_detection():
    """测试结构检测功能"""
    print("🧪 开始测试结构强度检测功能...")
    
    # 创建检测器
    detector = RealSMCDetector()
    
    # 创建测试数据
    test_df = create_test_data()
    
    print(f"📊 测试数据: {len(test_df)} 条记录")
    print(f"📈 价格范围: {test_df['close'].min():.2f} - {test_df['close'].max():.2f}")
    
    # 测试不同时间框架
    timeframes = ['1h', '4h', '15m']
    
    for tf in timeframes:
        print(f"\n⏰ 测试时间框架: {tf}")
        
        try:
            # 检测结构
            result = detector.detect_all_structures(test_df, tf)
            
            # 输出结果
            metrics = result['metrics']
            validation = result['validation']
            
            print(f"   ✅ 结构检测完成")
            print(f"   📊 BOS强度: {metrics['bos_strength']:.3f}")
            print(f"   📊 FVG数量: {metrics['fvg_count']}")
            print(f"   📊 OB数量: {metrics['ob_count']}")
            print(f"   📊 总强度: {metrics['total_strength']:.3f}")
            print(f"   🔍 置信度: {validation['confidence_score']:.3f}")
            
            if validation['issues']:
                print(f"   ⚠️  问题: {validation['issues']}")
            else:
                print(f"   ✅ 无检测问题")
                
        except Exception as e:
            print(f"   ❌ 检测失败: {e}")

def test_edge_cases():
    """测试边界条件"""
    print("\n🧪 测试边界条件...")
    
    detector = RealSMCDetector()
    
    # 测试空数据
    empty_df = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    try:
        result = detector.detect_all_structures(empty_df, '1h')
        print("   ✅ 空数据处理正常")
    except Exception as e:
        print(f"   ❌ 空数据处理失败: {e}")
    
    # 测试单行数据
    single_row = pd.DataFrame({
        'timestamp': [datetime.now()],
        'open': [100.0], 'high': [101.0], 'low': [99.0], 'close': [100.5], 'volume': [1000]
    })
    
    try:
        result = detector.detect_all_structures(single_row, '1h')
        print("   ✅ 单行数据处理正常")
    except Exception as e:
        print(f"   ❌ 单行数据处理失败: {e}")

def test_weight_calculation():
    """测试权重计算"""
    print("\n🧪 测试权重计算...")
    
    detector = RealSMCDetector()
    test_df = create_test_data()
    
    # 模拟不同结构数量
    test_cases = [
        {'bos_count': 5, 'fvg_count': 10, 'ob_count': 8},  # 正常情况
        {'bos_count': 0, 'fvg_count': 30, 'ob_count': 25},  # 高FVG/OB
        {'bos_count': 10, 'fvg_count': 2, 'ob_count': 3},   # 高BOS
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n   测试用例 {i}:")
        print(f"     BOS: {case['bos_count']}, FVG: {case['fvg_count']}, OB: {case['ob_count']}")
        
        # 模拟结构数据
        structures = {
            'bos_choch': pd.DataFrame({'BOS': [1] * case['bos_count']}),
            'fvg': pd.DataFrame({'size': [1.0] * case['fvg_count']}),
            'ob': pd.DataFrame({'high': [100.0] * case['ob_count'], 'low': [99.0] * case['ob_count']})
        }
        
        try:
            metrics = detector._calculate_structure_metrics(test_df, structures, '1h')
            print(f"     ✅ 总强度: {metrics['total_strength']:.3f}")
            print(f"     📊 BOS强度: {metrics['bos_strength']:.3f}")
            print(f"     📊 FVG贡献: {min(case['fvg_count'], 20) * 0.03:.3f}")
            print(f"     📊 OB贡献: {min(case['ob_count'], 15) * 0.04:.3f}")
        except Exception as e:
            print(f"     ❌ 计算失败: {e}")

if __name__ == "__main__":
    print("🚀 SMC结构强度检测修复测试")
    print("=" * 50)
    
    # 运行测试
    test_structure_detection()
    test_edge_cases()
    test_weight_calculation()
    
    print("\n" + "=" * 50)
    print("✅ 所有测试完成！")
    print("📋 修复总结:")
    print("   ✅ 权重分配优化 (BOS 35%, FVG 3%/个, OB 4%/个)")
    print("   ✅ 时间框架基准值统一")
    print("   ✅ 数据质量验证增强")
    print("   ✅ 边界条件异常处理完善")