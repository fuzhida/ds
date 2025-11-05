#!/usr/bin/env python3
"""
测试仓位计算修复
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtest_engine import BacktestEngine
from config import Config
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_position_calculation():
    """测试仓位计算是否修复"""
    print("测试仓位计算修复...")
    
    # 初始化配置
    config = Config()
    
    # 创建回测引擎
    engine = BacktestEngine(config)
    
    # 模拟信号
    signal = {
        "signal": "BUY",
        "confidence": 0.7,
        "reasoning": "测试信号"
    }
    
    # 模拟当前价格和余额
    current_price = 70000.0
    current_balance = 10000.0
    
    # 测试仓位计算
    try:
        position_result = engine.risk_manager.calculate_position_size(
            signal, current_balance, current_price, current_price * 0.02
        )
        
        position_size = position_result.get("position_size", 0)
        
        print(f"仓位计算结果: {position_result}")
        print(f"仓位大小: {position_size}")
        
        # 验证结果
        if isinstance(position_size, (int, float)) and position_size > 0:
            print("✅ 仓位计算修复成功！")
            return True
        else:
            print("❌ 仓位计算仍有问题")
            return False
            
    except Exception as e:
        print(f"❌ 仓位计算出错: {e}")
        return False

if __name__ == "__main__":
    success = test_position_calculation()
    sys.exit(0 if success else 1)