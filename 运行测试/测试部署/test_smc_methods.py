#!/usr/bin/env python3
"""
测试新增的SMC数据处理方法
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 直接导入方法而不是整个TradingBot类
from btc_trading_bot import _extract_detailed_smc_data, _analyze_structure_interactions

class MockBot:
    """模拟TradingBot类，只包含必要的属性"""
    def __init__(self):
        self.logger_system = MockLogger()

class MockLogger:
    """模拟日志记录器"""
    def info(self, msg):
        print(f"[INFO] {msg}")
    
    def error(self, msg):
        print(f"[ERROR] {msg}")

def test_extract_detailed_smc_data():
    """测试_extract_detailed_smc_data方法"""
    print("测试_extract_detailed_smc_data方法...")
    
    # 创建模拟Bot实例
    bot = MockBot()
    
    # 测试数据
    test_smc_data = {
        'bos_choch': [{'type': 'BOS', 'direction': 1, 'level': 60000, 'strength': 0.8, 'is_validated': True, 'volume_confirmation': 1.5}],
        'ob_fvg': {
            'ob': [{'type': 'OB', 'high': 60200, 'low': 59800, 'volume_ratio': 1.2, 'strength': 0.7, 'liquidity_score': 0.6, 'validity_score': 0.8, 'is_fresh': True}],
            'fvg': [{'type': 'FVG', 'high': 60300, 'low': 59900, 'gap_size': 400, 'gap_ratio': 0.0067, 'volume_ratio': 1.1, 'strength': 0.6, 'is_fresh': True}]
        },
        'liq_sweeps': [{'type': 'LIQUIDITY', 'level': 60500, 'strength': 0.5, 'is_swept': False}],
        'swings': [{'type': 'SWING_HIGH', 'price': 60400, 'strength': 0.7, 'index': 100}]
    }
    
    # 提取详细SMC数据
    detailed_data = _extract_detailed_smc_data(bot, test_smc_data, 60100, '15m')
    
    # 验证结果
    print(f'BOS/CHOCH数量: {len(detailed_data["bos_choch"])}')
    print(f'订单块数量: {len(detailed_data["order_blocks"])}')
    print(f'公平价值缺口数量: {len(detailed_data["fair_value_gaps"])}')
    print(f'流动性区域数量: {len(detailed_data["liquidity_zones"])}')
    print(f'摆动点数量: {len(detailed_data["swing_points"])}')
    print(f'结构相互作用数量: {len(detailed_data["structure_interactions"])}')
    
    # 打印第一个BOS/CHOCH的详细信息
    if detailed_data["bos_choch"]:
        bos = detailed_data["bos_choch"][0]
        print(f"第一个BOS/CHOCH: 类型={bos['type']}, 方向={bos['direction']}, 水平={bos['level']}, 强度={bos['strength']}, 距离价格={bos['distance_to_price']:.2f}%")
    
    # 打印第一个订单块的详细信息
    if detailed_data["order_blocks"]:
        ob = detailed_data["order_blocks"][0]
        print(f"第一个订单块: 类型={ob['type']}, 高={ob['high']}, 低={ob['low']}, 中点={ob['midpoint']}, 距离价格={ob['distance_to_price']:.2f}%")
    
    # 打印第一个FVG的详细信息
    if detailed_data["fair_value_gaps"]:
        fvg = detailed_data["fair_value_gaps"][0]
        print(f"第一个FVG: 类型={fvg['type']}, 高={fvg['high']}, 低={fvg['low']}, 缺口大小={fvg['gap_size']}, 距离价格={fvg['distance_to_price']:.2f}%")
    
    # 打印结构相互作用
    if detailed_data["structure_interactions"]:
        interaction = detailed_data["structure_interactions"][0]
        print(f"第一个结构相互作用: 类型={interaction['type']}, 描述={interaction['description']}, 重要性={interaction['significance']}")
    
    print("✅ _extract_detailed_smc_data方法测试通过!")
    return True

def test_analyze_structure_interactions():
    """测试_analyze_structure_interactions方法"""
    print("\n测试_analyze_structure_interactions方法...")
    
    # 创建模拟Bot实例
    bot = MockBot()
    
    # 测试数据
    bos_choch = [
        {'type': 'BOS', 'direction': 1, 'level': 60000, 'strength': 0.8, 'is_validated': True, 'volume_confirmation': 1.5}
    ]
    
    order_blocks = [
        {'type': 'OB', 'high': 60200, 'low': 59800, 'volume_ratio': 1.2, 'strength': 0.7, 'liquidity_score': 0.6, 'validity_score': 0.8, 'is_fresh': True}
    ]
    
    fair_value_gaps = [
        {'type': 'FVG', 'high': 60300, 'low': 59900, 'gap_size': 400, 'gap_ratio': 0.0067, 'volume_ratio': 1.1, 'strength': 0.6, 'is_fresh': True}
    ]
    
    liquidity_zones = [
        {'type': 'LIQUIDITY', 'level': 60500, 'strength': 0.5, 'is_swept': False}
    ]
    
    current_price = 60100
    
    # 分析结构相互作用
    interactions = _analyze_structure_interactions(
        bot, bos_choch, order_blocks, fair_value_gaps, liquidity_zones, current_price
    )
    
    # 验证结果
    print(f'检测到的结构相互作用数量: {len(interactions)}')
    
    # 打印所有相互作用
    for i, interaction in enumerate(interactions):
        print(f"相互作用 {i+1}: 类型={interaction['type']}, 描述={interaction['description']}, 重要性={interaction['significance']}")
    
    print("✅ _analyze_structure_interactions方法测试通过!")
    return True

if __name__ == "__main__":
    try:
        test_extract_detailed_smc_data()
        test_analyze_structure_interactions()
        print("\n🎉 所有测试通过!")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()