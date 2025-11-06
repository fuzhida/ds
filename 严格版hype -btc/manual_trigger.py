#!/usr/bin/env python3
"""
手动触发DeepSeek API请求，验证更新后的提示词效果
"""

import os
import sys
import json
from datetime import datetime

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deepseek_hyper import TradingBot, Config

def manual_trigger_analysis():
    """手动触发一次DeepSeek分析"""
    print("手动触发DeepSeek API分析...")
    
    # 创建配置
    config = Config()
    
    # 创建交易机器人实例
    bot = TradingBot(config)
    
    # 获取当前价格数据
    print("获取市场数据...")
    price_data = bot.get_multi_timeframe_data()
    
    if not price_data:
        print("无法获取市场数据，退出")
        return False
    
    print(f"当前价格: ${price_data['price']:,.2f}")
    
    # 手动触发DeepSeek分析
    print("\n触发DeepSeek API分析...")
    try:
        # 重置API健康状态，确保调用DeepSeek API
        bot.api_health_status['deepseek']['status'] = 'healthy'
        bot.api_health_status['deepseek']['consecutive_failures'] = 0
        
        signal = bot.analyze_with_deepseek(price_data)
        
        if signal:
            print("\n=== 分析结果 ===")
            print(f"信号: {signal.get('signal')}")
            print(f"理由: {signal.get('reason')}")
            print(f"止损: ${signal.get('stop_loss'):,.2f}")
            print(f"止盈: ${signal.get('take_profit'):,.2f}")
            print(f"置信度: {signal.get('confidence')}")
            
            # 将结果写入日志文件
            with open('manual_analysis_result.json', 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'price': price_data['price'],
                    'signal': signal
                }, f, indent=2)
            
            print("\n结果已保存到 manual_analysis_result.json")
            return True
        else:
            print("分析失败，未返回有效信号")
            return False
            
    except Exception as e:
        print(f"分析过程中出错: {e}")
        return False

if __name__ == "__main__":
    success = manual_trigger_analysis()
    if success:
        print("\n✅ 手动分析成功完成！")
    else:
        print("\n❌ 手动分析失败！")