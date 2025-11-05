#!/usr/bin/env python3
"""
回测示例脚本 - 演示如何使用回测功能
"""

import os
import sys
from datetime import datetime, timedelta
from backtest_engine import BacktestEngine


def example_basic_backtest():
    """基本回测示例"""
    print("=" * 60)
    print("基本回测示例")
    print("=" * 60)
    
    # 创建回测引擎
    backtest = BacktestEngine()
    
    # 运行回测
    symbol = "BTC/USDT"
    timeframes = ["1h", "4h", "1d"]
    start_date = "2023-01-01"
    end_date = "2023-03-31"  # 使用较短时间范围作为示例
    
    print(f"回测 {symbol} 从 {start_date} 到 {end_date}")
    print(f"使用时间框架: {', '.join(timeframes)}")
    print("开始回测...")
    
    report = backtest.run_backtest(
        symbol=symbol,
        timeframes=timeframes,
        start_date=start_date,
        end_date=end_date,
        initial_balance=10000.0
    )
    
    # 打印结果
    print("\n回测结果:")
    print("-" * 40)
    print(f"初始资金: ${report['initial_balance']:.2f}")
    print(f"最终资金: ${report['final_balance']:.2f}")
    print(f"总收益: ${report['final_balance'] - report['initial_balance']:.2f}")
    print(f"总收益率: {report['performance_metrics']['total_return']:.2%}")
    print(f"总交易次数: {report['performance_metrics']['total_trades']}")
    print(f"胜率: {report['performance_metrics']['win_rate']:.2%}")
    print(f"最大回撤: {report['performance_metrics']['max_drawdown']:.2%}")
    print(f"夏普比率: {report['performance_metrics']['sharpe_ratio']:.2f}")
    print(f"盈利因子: {report['performance_metrics']['profit_factor']:.2f}")
    print("-" * 40)
    
    # 提示文件位置
    symbol_clean = symbol.replace('/', '_')
    report_file = f"backtest_reports/backtest_{symbol_clean}_{start_date}_{end_date}.json"
    chart_file = f"backtest_charts/backtest_{symbol_clean}_{start_date}_{end_date}.png"
    
    print(f"\n详细报告已保存到: {report_file}")
    print(f"回测图表已保存到: {chart_file}")
    
    return report


def example_multiple_symbols():
    """多交易对回测示例"""
    print("\n" + "=" * 60)
    print("多交易对回测示例")
    print("=" * 60)
    
    # 交易对列表
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
    timeframes = ["1h", "4h"]
    start_date = "2023-01-01"
    end_date = "2023-02-28"
    
    results = {}
    
    for symbol in symbols:
        print(f"\n回测 {symbol}...")
        
        # 创建回测引擎
        backtest = BacktestEngine()
        
        # 运行回测
        report = backtest.run_backtest(
            symbol=symbol,
            timeframes=timeframes,
            start_date=start_date,
            end_date=end_date,
            initial_balance=10000.0
        )
        
        results[symbol] = report['performance_metrics']
        
        # 打印简要结果
        print(f"  总收益率: {report['performance_metrics']['total_return']:.2%}")
        print(f"  胜率: {report['performance_metrics']['win_rate']:.2%}")
        print(f"  最大回撤: {report['performance_metrics']['max_drawdown']:.2%}")
    
    # 比较结果
    print("\n比较结果:")
    print("-" * 60)
    print(f"{'交易对':<12} {'总收益率':<10} {'胜率':<10} {'最大回撤':<10} {'夏普比率':<10}")
    print("-" * 60)
    
    for symbol, metrics in results.items():
        print(f"{symbol:<12} {metrics['total_return']:<10.2%} {metrics['win_rate']:<10.2%} "
              f"{metrics['max_drawdown']:<10.2%} {metrics['sharpe_ratio']:<10.2f}")
    
    return results


def example_timeframe_comparison():
    """时间框架比较示例"""
    print("\n" + "=" * 60)
    print("时间框架比较示例")
    print("=" * 60)
    
    # 不同时间框架组合
    timeframe_combinations = [
        ["15m", "1h", "4h"],
        ["1h", "4h", "1d"],
        ["4h", "1d"]
    ]
    
    symbol = "BTC/USDT"
    start_date = "2023-01-01"
    end_date = "2023-02-28"
    
    results = {}
    
    for i, timeframes in enumerate(timeframe_combinations):
        print(f"\n回测时间框架组合 {i+1}: {', '.join(timeframes)}")
        
        # 创建回测引擎
        backtest = BacktestEngine()
        
        # 运行回测
        report = backtest.run_backtest(
            symbol=symbol,
            timeframes=timeframes,
            start_date=start_date,
            end_date=end_date,
            initial_balance=10000.0
        )
        
        results[f"组合{i+1}"] = report['performance_metrics']
        
        # 打印简要结果
        print(f"  总收益率: {report['performance_metrics']['total_return']:.2%}")
        print(f"  胜率: {report['performance_metrics']['win_rate']:.2%}")
        print(f"  最大回撤: {report['performance_metrics']['max_drawdown']:.2%}")
    
    # 比较结果
    print("\n比较结果:")
    print("-" * 60)
    print(f"{'时间框架组合':<15} {'总收益率':<10} {'胜率':<10} {'最大回撤':<10} {'夏普比率':<10}")
    print("-" * 60)
    
    for combo, metrics in results.items():
        print(f"{combo:<15} {metrics['total_return']:<10.2%} {metrics['win_rate']:<10.2%} "
              f"{metrics['max_drawdown']:<10.2%} {metrics['sharpe_ratio']:<10.2f}")
    
    return results


def example_custom_analysis():
    """自定义分析示例"""
    print("\n" + "=" * 60)
    print("自定义分析示例")
    print("=" * 60)
    
    # 运行基本回测
    report = example_basic_backtest()
    
    # 获取交易记录
    trades = report['trades']
    equity_curve = report['equity_curve']
    
    # 分析交易记录
    print("\n交易分析:")
    print("-" * 40)
    
    # 按月份分组
    monthly_trades = {}
    for trade in trades:
        if trade['status'] == 'closed':
            month = datetime.fromisoformat(trade['entry_time']).strftime('%Y-%m')
            if month not in monthly_trades:
                monthly_trades[month] = {'count': 0, 'pnl': 0.0, 'wins': 0, 'losses': 0}
            
            monthly_trades[month]['count'] += 1
            monthly_trades[month]['pnl'] += trade.get('pnl', 0)
            
            if trade.get('pnl', 0) > 0:
                monthly_trades[month]['wins'] += 1
            else:
                monthly_trades[month]['losses'] += 1
    
    # 打印月度表现
    print("月度交易表现:")
    print(f"{'月份':<10} {'交易次数':<8} {'盈亏':<10} {'胜率':<10}")
    print("-" * 40)
    
    for month, data in sorted(monthly_trades.items()):
        win_rate = data['wins'] / data['count'] if data['count'] > 0 else 0
        print(f"{month:<10} {data['count']:<8} {data['pnl']:<10.2f} {win_rate:<10.2%}")
    
    # 分析连续亏损
    print("\n连续亏损分析:")
    print("-" * 40)
    
    closed_trades = [t for t in trades if t['status'] == 'closed']
    max_consecutive_losses = 0
    current_consecutive_losses = 0
    
    for trade in closed_trades:
        if trade.get('pnl', 0) < 0:
            current_consecutive_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, current_consecutive_losses)
        else:
            current_consecutive_losses = 0
    
    print(f"最大连续亏损次数: {max_consecutive_losses}")
    
    # 分析持仓时间
    print("\n持仓时间分析:")
    print("-" * 40)
    
    holding_times = []
    for trade in closed_trades:
        if 'entry_time' in trade and 'exit_time' in trade:
            entry_time = datetime.fromisoformat(trade['entry_time'])
            exit_time = datetime.fromisoformat(trade['exit_time'])
            holding_time = (exit_time - entry_time).total_seconds() / 3600  # 小时
            holding_times.append(holding_time)
    
    if holding_times:
        avg_holding_time = sum(holding_times) / len(holding_times)
        min_holding_time = min(holding_times)
        max_holding_time = max(holding_times)
        
        print(f"平均持仓时间: {avg_holding_time:.2f} 小时")
        print(f"最短持仓时间: {min_holding_time:.2f} 小时")
        print(f"最长持仓时间: {max_holding_time:.2f} 小时")
    
    return report


def main():
    """主函数"""
    try:
        # 基本回测示例
        example_basic_backtest()
        
        # 多交易对回测示例
        example_multiple_symbols()
        
        # 时间框架比较示例
        example_timeframe_comparison()
        
        # 自定义分析示例
        example_custom_analysis()
        
        print("\n" + "=" * 60)
        print("所有示例运行完成!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n示例被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"示例运行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()