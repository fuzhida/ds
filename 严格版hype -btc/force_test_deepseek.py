#!/usr/bin/env python3
"""
强制测试DeepSeek API调用脚本
用于验证更新后的提示词是否生效
"""

import os
import sys
import json
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from deepseek_hyper import TradingBot, Config

def main():
    """强制测试DeepSeek API调用"""
    print("=== 强制测试DeepSeek API调用 ===")
    
    # 创建配置
    config = Config()
    
    # 创建交易机器人实例
    bot = TradingBot(config)
    
    # 获取市场数据
    print("\n获取市场数据...")
    price_data = bot.get_multi_timeframe_data()
    
    if not price_data:
        print("❌ 无法获取市场数据")
        return
    
    print(f"✅ 成功获取市场数据，当前价格: ${price_data['price']:.2f}")
    
    # 强制重置API健康状态，确保调用DeepSeek API
    print("\n重置API健康状态...")
    bot.api_health_status['deepseek']['status'] = 'healthy'
    bot.api_health_status['deepseek']['consecutive_failures'] = 0
    
    # 强制跳过动量过滤器
    print("\n强制跳过动量过滤器...")
    
    # 手动触发DeepSeek分析
    print("\n触发DeepSeek API分析...")
    try:
        signal = bot.analyze_with_deepseek(price_data)
        
        print("\n=== 分析结果 ===")
        print(f"信号: {signal.get('signal', 'N/A')}")
        print(f"理由: {signal.get('reason', 'N/A')}")
        print(f"止损: ${signal.get('stop_loss', 0):,.2f}")
        print(f"止盈: ${signal.get('take_profit', 0):,.2f}")
        print(f"置信度: {signal.get('confidence', 'N/A')}")
        
        # 保存结果
        result = {
            "timestamp": datetime.now().isoformat(),
            "price": price_data['price'],
            "signal": signal
        }
        
        with open("force_test_result.json", "w") as f:
            json.dump(result, f, indent=2)
        
        print("\n结果已保存到 force_test_result.json")
        
        # 检查是否是备用信号
        if signal.get('is_fallback', False):
            print("\n⚠️ 警告: 使用了备用信号，DeepSeek API调用可能失败")
        else:
            print("\n✅ 成功调用DeepSeek API")
        
    except Exception as e:
        print(f"\n❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ 强制测试完成！")

if __name__ == "__main__":
    main()