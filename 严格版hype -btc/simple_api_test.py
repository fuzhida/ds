#!/usr/bin/env python3
"""简单测试DeepSeek API是否正常工作"""

import sys
import os
import json
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(threadName)s ] %(name)-12s - %(levelname)-8s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from deepseek_hyper import TradingBot, Config
    print("✅ 成功导入TradingBot和Config")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

def test_deepseek_api():
    """测试DeepSeek API调用"""
    print("\n=== 测试DeepSeek API调用 ===")
    
    try:
        # 创建配置
        config = Config()
        print("✅ 成功创建配置")
        
        # 创建机器人实例
        bot = TradingBot(config)
        print("✅ 成功创建机器人实例")
        
        # 重置API健康状态
        bot.api_health_status['deepseek']['status'] = 'healthy'
        bot.api_health_status['deepseek']['consecutive_failures'] = 0
        print("✅ 重置API健康状态")
        
        # 创建测试数据
        import pandas as pd
        import numpy as np
        
        # 创建模拟的15分钟K线数据
        np.random.seed(42)  # 确保结果可重现
        dates = pd.date_range(end=datetime.now(), periods=100, freq='15min')
        base_price = 103921.0
        
        # 生成模拟OHLCV数据
        data = []
        for i, date in enumerate(dates):
            # 添加一些随机波动
            price_change = np.random.normal(0, 0.002)  # 0.2%的标准偏差
            open_price = base_price * (1 + price_change)
            high_price = open_price * (1 + abs(np.random.normal(0, 0.001)))
            low_price = open_price * (1 - abs(np.random.normal(0, 0.001)))
            close_price = open_price + np.random.normal(0, base_price * 0.001)
            volume = np.random.uniform(100, 1000)
            
            data.append([date, open_price, high_price, low_price, close_price, volume])
            base_price = close_price  # 下一根K线的基础价格
        
        # 创建DataFrame
        m15_df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        m15_df.set_index('timestamp', inplace=True)
        
        price_data = {
            'price': 103921.0,
            'timestamp': datetime.now(),  # 添加时间戳
            'amplitude': {
                'avg_amplitude': 3542.00,
                'expected_rr_range': 2000.00
            },
            'multi_tf_data': {
                '15m': m15_df  # 提供有效的DataFrame
            }
        }
        print("✅ 创建测试数据")
        
        # 调用analyze_with_deepseek方法
        print("\n调用analyze_with_deepseek方法...")
        result = bot.analyze_with_deepseek(price_data)
        
        print(f"\n✅ 调用成功！结果:")
        
        # 创建结果的可序列化副本
        serializable_result = {}
        for key, value in result.items():
            if key == 'timestamp' and hasattr(value, 'isoformat'):
                serializable_result[key] = value.isoformat()
            else:
                serializable_result[key] = value
        
        print(json.dumps(serializable_result, indent=2, ensure_ascii=False))
        
        # 保存结果到文件
        with open('api_test_result.json', 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'result': serializable_result
            }, f, indent=2, ensure_ascii=False)
        
        print("\n✅ 结果已保存到 api_test_result.json")
        return True
        
    except Exception as e:
        print(f"\n❌ 调用失败: {e}")
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