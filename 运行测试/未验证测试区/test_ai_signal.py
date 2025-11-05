#!/usr/bin/env python3
"""
测试AI信号生成器
"""

import os
import logging
from dotenv import load_dotenv
from ai_signal_generator import AISignalGenerator
from config import Config

def main():
    """主函数"""
    try:
        # 加载环境变量
        load_dotenv('1.env')
        
        # 创建配置
        config = Config()
        
        # 创建日志器
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("AI_Signal_Test")
        
        # 打印API密钥状态
        print(f'DeepSeek API Key: {"已设置" if config.deepseek_api_key else "未设置"}')
        print(f'OpenAI API Key: {"已设置" if config.openai_api_key else "未设置"}')
        
        # 创建AI信号生成器
        ai_generator = AISignalGenerator(config, logger)
        
        # 检查提供者状态
        print(f'AI信号提供者数量: {len(ai_generator.providers)}')
        for provider in ai_generator.providers:
            print(f'- {provider.__class__.__name__}')
        
        # 创建测试市场数据
        market_data = {
            'symbol': 'BTC/USDT',
            'price': 50000.0,
            'volume': 1000.0,
            'timeframe': '1h',
            'indicators': {
                'rsi': 55.0,
                'macd': {'macd': 100.0, 'signal': 90.0, 'histogram': 10.0},
                'bb': {'upper': 51000.0, 'middle': 50000.0, 'lower': 49000.0}
            }
        }
        
        # 生成信号
        print('\n生成AI信号...')
        signals = ai_generator.generate_signals(market_data)
        
        # 打印结果
        print(f'共识信号: {signals["consensus"]}')
        print(f'共识置信度: {signals["consensus_confidence"]:.2f}')
        print(f'主要信号提供者: {signals["primary"]["provider"]}')
        print(f'主要信号: {signals["primary"]["signal"]["signal"]}')
        print(f'主要信号置信度: {signals["primary"]["confidence"]:.2f}')
        print(f'主要信号原因: {signals["primary"]["signal"]["reasoning"]}')
        
        # 打印所有信号
        print('\n所有信号:')
        for provider, signal in signals["all"].items():
            print(f'- {provider}: {signal["signal"]["signal"]} (置信度: {signal["confidence"]:.2f})')
        
    except Exception as e:
        print(f'测试失败: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()