#!/usr/bin/env python3
"""
回测运行脚本 - 简化回测流程
"""

import sys
import argparse
from datetime import datetime, timedelta
from backtest_engine import BacktestEngine


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='交易策略回测工具')
    
    parser.add_argument('--symbol', type=str, default='BTC/USDT',
                        help='交易对 (默认: BTC/USDT)')
    
    parser.add_argument('--timeframes', type=str, nargs='+', default=['1h', '4h', '1d'],
                        help='时间框架列表 (默认: 1h 4h 1d)')
    
    parser.add_argument('--start-date', type=str, default='2023-01-01',
                        help='开始日期 (格式: YYYY-MM-DD, 默认: 2023-01-01)')
    
    parser.add_argument('--end-date', type=str, default='2023-12-31',
                        help='结束日期 (格式: YYYY-MM-DD, 默认: 2023-12-31)')
    
    parser.add_argument('--initial-balance', type=float, default=10000.0,
                        help='初始资金 (默认: 10000.0)')
    
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径')
    
    return parser.parse_args()


def main():
    """主函数"""
    try:
        # 解析参数
        args = parse_arguments()
        
        # 验证日期格式
        try:
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
            end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
            
            if start_date >= end_date:
                print("错误: 开始日期必须早于结束日期")
                sys.exit(1)
                
            # 转换为字符串
            start_date_str = start_date.strftime("%Y-%m-%d")
            end_date_str = end_date.strftime("%Y-%m-%d")
            
        except ValueError:
            print("错误: 日期格式不正确，请使用 YYYY-MM-DD 格式")
            sys.exit(1)
        
        # 打印回测参数
        print("=" * 50)
        print("交易策略回测")
        print("=" * 50)
        print(f"交易对: {args.symbol}")
        print(f"时间框架: {', '.join(args.timeframes)}")
        print(f"回测期间: {start_date_str} 至 {end_date_str}")
        print(f"初始资金: {args.initial_balance:.2f}")
        print("=" * 50)
        
        # 创建回测引擎
        print("初始化回测引擎...")
        backtest = BacktestEngine(args.config)
        
        # 运行回测
        print("开始回测...")
        report = backtest.run_backtest(
            symbol=args.symbol,
            timeframes=args.timeframes,
            start_date=start_date_str,
            end_date=end_date_str,
            initial_balance=args.initial_balance
        )
        
        # 打印结果
        print("\n回测结果:")
        print("-" * 50)
        print(f"初始资金: {report['initial_balance']:.2f}")
        print(f"最终资金: {report['final_balance']:.2f}")
        print(f"总收益: {report['final_balance'] - report['initial_balance']:.2f}")
        print(f"总收益率: {report['performance_metrics']['total_return']:.2%}")
        print(f"总交易次数: {report['performance_metrics']['total_trades']}")
        print(f"盈利交易: {report['performance_metrics']['winning_trades']}")
        print(f"亏损交易: {report['performance_metrics']['losing_trades']}")
        print(f"胜率: {report['performance_metrics']['win_rate']:.2%}")
        print(f"最大回撤: {report['performance_metrics']['max_drawdown']:.2%}")
        print(f"夏普比率: {report['performance_metrics']['sharpe_ratio']:.2f}")
        print(f"盈利因子: {report['performance_metrics']['profit_factor']:.2f}")
        print("-" * 50)
        
        # 提示文件位置
        symbol_clean = args.symbol.replace('/', '_')
        report_file = f"backtest_reports/backtest_{symbol_clean}_{start_date_str}_{end_date_str}.json"
        chart_file = f"backtest_charts/backtest_{symbol_clean}_{start_date_str}_{end_date_str}.png"
        
        print(f"\n详细报告已保存到: {report_file}")
        print(f"回测图表已保存到: {chart_file}")
        
    except KeyboardInterrupt:
        print("\n回测被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"回测失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()