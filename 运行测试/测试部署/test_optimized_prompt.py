#!/usr/bin/env python3
"""
测试优化后的SMC/ICT策略分析提示词
"""

import sys
import json
from optimized_smc_prompt import get_optimized_smc_prompt

def test_optimized_prompt():
    """测试优化后的提示词生成"""
    
    # 模拟市场数据
    market_data = {
        'current_price': 67500.0,
        'symbol': 'BTC/USDT',
        'higher_tf': '4h',
        'higher_tf_trend': 'bullish',
        'higher_tf_strength': 0.8,
        'primary_tf': '15m',
        'primary_tf_trend': 'bullish',
        'primary_tf_strength': 0.7,
        'mtf_consistency': 0.75,
        'structure_score': 0.85,
        'structure_count': 3,
        'structure_quality': '高',
        'rsi': 55.0,
        'macd_histogram': 0.02,
        'volume_ratio': 1.5,
        'volatility': 2.5,
        'min_rr_ratio': 2.0,
        'invalidation_point': 66000.0,
        'nearest_key_level': 67000.0,
        'key_level_distance': 0.8
    }
    
    print("=" * 80)
    print("测试优化后的SMC/ICT策略分析提示词")
    print("=" * 80)
    
    # 生成提示词
    prompt = get_optimized_smc_prompt(market_data)
    
    print("\n📤 生成的提示词:")
    print("-" * 40)
    print(prompt)
    print("-" * 40)
    
    # 检查提示词内容
    print("\n📋 提示词检查:")
    print(f"- 包含市场状况: {'✓' if '当前市场状况' in prompt else '✗'}")
    print(f"- 包含多时间框架分析: {'✓' if '多时间框架分析' in prompt else '✗'}")
    print(f"- 包含SMC结构分析: {'✓' if 'SMC结构分析' in prompt else '✗'}")
    print(f"- 包含交易信号生成指南: {'✓' if '交易信号生成指南' in prompt else '✗'}")
    print(f"- 包含AI专业判断权限: {'✓' if 'AI专业判断权限' in prompt else '✗'}")
    print(f"- 包含输出要求: {'✓' if '输出要求' in prompt else '✗'}")
    print(f"- 包含分析重点: {'✓' if '分析重点' in prompt else '✗'}")
    
    # 检查关键变量是否替换
    print("\n🔍 变量替换检查:")
    current_price_str = f"{market_data['current_price']:.1f}"  # 使用.1f而不是.2f
    print(f"- 当前价格: {'✓' if current_price_str in prompt else '✗'}")
    print(f"- 高时间框架: {'✓' if market_data['higher_tf'] in prompt else '✗'}")
    print(f"- 结构质量: {'✓' if market_data['structure_quality'] in prompt else '✗'}")
    rsi_str = f"{market_data['rsi']:.1f}"
    print(f"- RSI: {'✓' if rsi_str in prompt else '✗'}")
    
    # 检查简化点
    print("\n🎯 简化点检查:")
    print(f"- 移除了复杂变量: {'✓' if 'fvg_ratio' not in prompt else '✗'}")
    print(f"- 简化了判断条件: {'✓' if '技术指标限制已放宽' in prompt else '✗'}")
    print(f"- 明确了AI权限: {'✓' if '专业判断权限' in prompt else '✗'}")
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)

if __name__ == "__main__":
    test_optimized_prompt()