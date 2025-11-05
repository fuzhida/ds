#!/usr/bin/env python3
"""
使用真实数据格式运行回测
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
from dotenv import load_dotenv
from datetime import datetime, timedelta
from backtest_engine import BacktestEngine
from config import Config

def main():
    """主函数"""
    try:
        # 加载环境变量
        load_dotenv('1.env')
        
        # 创建日志器
        logging.basicConfig(level=logging.DEBUG)  # 设置为DEBUG级别以获取更多信息
        logger = logging.getLogger("Backtest")
        
        # 计算过去3天的日期（减少数据量）
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3)
        
        # 格式化为字符串
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        print(f'运行过去3天的回测: {start_str} 到 {end_str}')
        print(f'DeepSeek API Key: {"已设置" if os.getenv("DEEPSEEK_API_KEY") else "未设置"}')
        print(f'OpenAI API Key: {"已设置" if os.getenv("OPENAI_API_KEY") else "未设置"}')
        
        # 创建回测引擎
        backtest = BacktestEngine()
        
        # 运行回测
        report = backtest.run_backtest(
            symbol='BTC/USDT',
            timeframes=['1h'],  # 只使用1h时间框架以减少计算量
            start_date=start_str,
            end_date=end_str,
            initial_balance=10000.0
        )
        
        # 打印结果
        print('\n回测结果:')
        print(f'初始资金: ${report["initial_balance"]:.2f}')
        print(f'最终资金: ${report["final_balance"]:.2f}')
        print(f'总收益率: {report["performance_metrics"]["total_return"]:.2%}')
        print(f'总交易次数: {report["performance_metrics"]["total_trades"]}')
        print(f'胜率: {report["performance_metrics"]["win_rate"]:.2%}')
        print(f'最大回撤: {report["performance_metrics"]["max_drawdown"]:.2%}')
        print(f'夏普比率: {report["performance_metrics"]["sharpe_ratio"]:.2f}')
        
        # 如果有交易记录，打印前5笔交易
        if report["trades"]:
            print('\n前5笔交易:')
            for i, trade in enumerate(report["trades"][:5]):
                print(f'{i+1}. {trade["type"]} {trade["amount"]} @ {trade["price"]}, '
                      f'PnL: {trade.get("pnl", "N/A")}, 状态: {trade["status"]}')
        else:
            print('\n没有交易记录')
        
    except Exception as e:
        print(f'回测失败: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()