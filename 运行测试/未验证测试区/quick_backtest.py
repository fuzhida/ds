#!/usr/bin/env python3
"""
快速回测测试 - 只运行一小段时间来验证修复
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 加载环境变量
from dotenv import load_dotenv
load_dotenv('1.env')

from backtest_engine import BacktestEngine
from config import Config
import logging
from datetime import datetime, timedelta

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def quick_backtest():
    """快速回测测试"""
    print("开始快速回测测试...")
    
    # 初始化配置
    config = Config()
    
    # 创建回测引擎
    engine = BacktestEngine(config)
    
    # 设置测试时间范围（只有1天）
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)
    
    # 转换为字符串格式
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    
    # 运行回测
    try:
        results = engine.run_backtest("BTC/USDT", ["1h"], start_date_str, end_date_str)
        
        # 打印结果
        print("\n=== 回测结果 ===")
        print(f"初始资金: ${results['initial_balance']:.2f}")
        print(f"最终资金: ${results['final_balance']:.2f}")
        
        # 获取性能指标
        metrics = results.get('performance_metrics', {})
        print(f"总交易次数: {metrics.get('total_trades', 0)}")
        print(f"胜率: {metrics.get('win_rate', 0):.2%}")
        print(f"总盈亏: ${metrics.get('total_pnl', 0):.2f}")
        print(f"最大回撤: {metrics.get('max_drawdown', 0):.2%}")
        
        # 检查是否有交易
        if metrics.get('total_trades', 0) > 0:
            print("\n✅ 回测成功！有交易记录")
            print("\n=== 交易记录 ===")
            for trade in results['trades']:
                print(f"{trade['entry_time']}: {trade['type']} {trade['symbol']} @ {trade['entry_price']}, "
                      f"数量: {trade['size']:.4f}, 止损: {trade['stop_loss']:.2f}, 止盈: {trade['take_profit']:.2f}")
                if 'exit_price' in trade:
                    print(f"  平仓 @ {trade['exit_price']}, 盈亏: {trade.get('pnl', 0):.2f}, 原因: {trade.get('exit_reason', 'N/A')}")
        else:
            print("\n❌ 回测无交易记录")
            
        return metrics.get('total_trades', 0) > 0
        
    except Exception as e:
        print(f"❌ 回测失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_backtest()
    sys.exit(0 if success else 1)