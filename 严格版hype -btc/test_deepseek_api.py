#!/usr/bin/env python3
"""测试DeepSeek API调用是否正常工作"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from deepseek_hyper import TradingBot
import json

def test_deepseek_api():
    """测试DeepSeek API调用"""
    print("=== 测试DeepSeek API调用 ===")
    
    # 创建机器人实例
    bot = TradingBot()
    
    # 重置API健康状态
    bot.api_health_status['deepseek']['status'] = 'healthy'
    bot.api_health_status['deepseek']['consecutive_failures'] = 0
    
    # 创建测试数据
    price_data = {
        'price': 103921.0,
        'amplitude': {
            'avg_amplitude': 3542.00,
            'expected_rr_range': 2000.00
        },
        'multi_tf_data': {
            '15m': None  # 这里应该是DataFrame，但我们简化测试
        }
    }
    
    # 调用analyze_with_deepseek方法
    print("调用analyze_with_deepseek方法...")
    try:
        result = bot.analyze_with_deepseek(price_data)
        print(f"调用成功！结果: {json.dumps(result, indent=2)}")
        return True
    except Exception as e:
        print(f"调用失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_deepseek_api()
    if success:
        print("\n✅ DeepSeek API调用测试成功！")
    else:
        print("\n❌ DeepSeek API调用测试失败！")
    sys.exit(0 if success else 1)