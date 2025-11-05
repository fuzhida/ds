#!/usr/bin/env python3
"""
快速验证回测脚本 - 使用更短的时间范围进行验证
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 加载环境变量
from dotenv import load_dotenv
load_dotenv('1.env')

from datetime import datetime, timedelta
from backtest_engine import BacktestEngine
from config import Config
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("QuickValidation")

def main():
    """主函数"""
    try:
        # 创建配置
        config = Config()
        
        # 创建回测引擎
        engine = BacktestEngine(config)
        
        # 设置回测时间范围 - 使用更短的时间范围进行快速验证
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3)  # 只使用3天数据
        
        logger.info(f"开始回测验证: {start_date.date()} 到 {end_date.date()}")
        
        # 运行回测
        results = engine.run_backtest(
            symbol="BTC/USDT",
            timeframes=["1h"],
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            initial_balance=10000.0
        )
        
        # 显示结果
        print("\n=== 回测验证结果 ===")
        print(f"初始资金: ${results['initial_balance']:.2f}")
        print(f"最终资金: ${results['final_balance']:.2f}")
        
        # 获取性能指标
        metrics = results.get('performance_metrics', {})
        print(f"总交易次数: {metrics.get('total_trades', 0)}")
        print(f"胜率: {metrics.get('win_rate', 0):.2f}%")
        print(f"总盈亏: ${metrics.get('total_profit_loss', 0):.2f}")
        print(f"最大回撤: {metrics.get('max_drawdown', 0):.2f}%")
        
        # 显示交易记录
        trades = results.get('trades', [])
        if trades:
            print("\n=== 交易记录 ===")
            for trade in trades:
                print(f"{trade['timestamp']}: {trade['action']} {trade['symbol']} @ {trade['price']:.2f}, "
                      f"数量: {trade['quantity']:.4f}, 止损: {trade['stop_loss']:.2f}, 止盈: {trade['take_profit']:.2f}")
                if 'exit_price' in trade:
                    print(f"  平仓 @ {trade['exit_price']:.2f}, 盈亏: {trade['profit_loss']:.2f}, 原因: {trade['exit_reason']}")
        
        # 检查是否有交易
        if metrics.get('total_trades', 0) > 0:
            print("\n✅ 回测验证成功！有交易记录")
        else:
            print("\n❌ 回测验证失败！无交易记录")
        
        return results
        
    except Exception as e:
        logger.error(f"回测验证失败: {str(e)}")
        return None

if __name__ == "__main__":
    main()