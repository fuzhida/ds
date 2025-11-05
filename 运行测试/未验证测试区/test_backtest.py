#!/usr/bin/env python3
"""
简单回测测试脚本
"""

from backtest_engine import BacktestEngine
import sys

try:
    print('创建回测引擎...')
    backtest = BacktestEngine()
    
    # 获取一些历史数据来检查格式
    print("获取历史数据...")
    historical_data = backtest._get_historical_data("BTC/USDT", ["1h"], "2024-12-15", "2024-12-22")
    
    # 检查数据格式
    if historical_data and "1h" in historical_data:
        df = historical_data["1h"]
        print(f"数据形状: {df.shape}")
        print(f"数据列: {df.columns.tolist()}")
        print(f"前5行数据:\n{df.head()}")
    else:
        print("无法获取历史数据")
    
    print('\n运行回测...')
    report = backtest.run_backtest(
        symbol='BTC/USDT',
        timeframes=['1h'],
        start_date='2024-12-15',
        end_date='2024-12-22',
        initial_balance=10000.0
    )
    
    print('回测结果:')
    print('初始资金: ${:.2f}'.format(report['initial_balance']))
    print('最终资金: ${:.2f}'.format(report['final_balance']))
    print('总收益率: {:.2%}'.format(report['performance_metrics']['total_return']))
    print('总交易次数: {}'.format(report['performance_metrics']['total_trades']))
    print('胜率: {:.2%}'.format(report['performance_metrics']['win_rate']))
    
except Exception as e:
    print('回测失败: {}'.format(e))
    import traceback
    traceback.print_exc()
    sys.exit(1)