#!/usr/bin/env python3
"""
加载环境变量并运行回测
"""

import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import sys
from backtest_engine import BacktestEngine

def main():
    """主函数"""
    try:
        # 加载环境变量
        load_dotenv('1.env')
        
        # 计算过去一周的日期
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        # 格式化为字符串
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        print(f'运行过去一周的回测: {start_str} 到 {end_str}')
        print(f'DeepSeek API Key: {os.getenv("DEEPSEEK_API_KEY")[:10]}...' if os.getenv("DEEPSEEK_API_KEY") else 'DeepSeek API Key: 未设置')
        print(f'OpenAI API Key: {os.getenv("OPENAI_API_KEY")[:10]}...' if os.getenv("OPENAI_API_KEY") else 'OpenAI API Key: 未设置')
        
        # 创建回测引擎
        backtest = BacktestEngine()
        
        # 运行回测
        report = backtest.run_backtest(
            symbol='BTC/USDT',
            timeframes=['1h', '4h', '1d'],
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
        
    except Exception as e:
        print(f'回测失败: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()