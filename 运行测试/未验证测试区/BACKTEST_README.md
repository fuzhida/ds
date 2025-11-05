# 交易策略回测指南

本指南介绍如何使用回测功能测试交易策略的历史表现。

## 功能概述

回测系统支持以下功能：

- 使用历史数据测试交易策略
- 支持多时间框架分析
- 模拟真实交易环境
- 计算关键性能指标
- 生成可视化图表
- 导出详细报告

## 快速开始

### 1. 使用默认参数运行回测

```bash
python run_backtest.py
```

这将使用默认参数对BTC/USDT在2023年全年进行回测。

### 2. 自定义参数运行回测

```bash
python run_backtest.py \
    --symbol ETH/USDT \
    --timeframes 15m 1h 4h \
    --start-date 2023-06-01 \
    --end-date 2023-12-31 \
    --initial-balance 5000
```

### 3. 使用自定义配置文件

```bash
python run_backtest.py --config config.json
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--symbol` | 交易对 | BTC/USDT |
| `--timeframes` | 时间框架列表 | 1h 4h 1d |
| `--start-date` | 开始日期 (YYYY-MM-DD) | 2023-01-01 |
| `--end-date` | 结束日期 (YYYY-MM-DD) | 2023-12-31 |
| `--initial-balance` | 初始资金 | 10000.0 |
| `--config` | 配置文件路径 | None |

## 回测结果

回测完成后，系统会生成以下输出：

### 1. 控制台输出

```
回测结果:
--------------------------------------------------
初始资金: 10000.00
最终资金: 11547.32
总收益: 1547.32
总收益率: 15.47%
总交易次数: 42
盈利交易: 25
亏损交易: 17
胜率: 59.52%
最大回撤: 5.23%
夏普比率: 1.24
盈利因子: 1.87
--------------------------------------------------
```

### 2. 详细报告 (JSON格式)

包含完整的交易记录、资金曲线和性能指标，保存在 `backtest_reports/` 目录。

### 3. 可视化图表

包含权益曲线和价格曲线，保存在 `backtest_charts/` 目录。

## 性能指标说明

| 指标 | 说明 |
|------|------|
| 总收益率 | 回测期间的总收益百分比 |
| 胜率 | 盈利交易占总交易次数的比例 |
| 最大回撤 | 从峰值到谷值的最大跌幅 |
| 夏普比率 | 风险调整后收益，越高越好 |
| 盈利因子 | 总盈利/总亏损，大于1表示盈利 |

## 高级用法

### 1. 自定义回测脚本

```python
from backtest_engine import BacktestEngine

# 创建回测引擎
backtest = BacktestEngine()

# 运行回测
report = backtest.run_backtest(
    symbol="BTC/USDT",
    timeframes=["1h", "4h", "1d"],
    start_date="2023-01-01",
    end_date="2023-12-31",
    initial_balance=10000.0
)

# 访问结果
print(f"总收益率: {report['performance_metrics']['total_return']:.2%}")
```

### 2. 分析交易记录

```python
# 获取已平仓交易
closed_trades = [t for t in report['trades'] if t['status'] == 'closed']

# 按盈亏排序
sorted_trades = sorted(closed_trades, key=lambda x: x.get('pnl', 0), reverse=True)

# 打印最大盈利和最大亏损交易
print(f"最大盈利: {sorted_trades[0]['pnl']:.2f}")
print(f"最大亏损: {sorted_trades[-1]['pnl']:.2f}")
```

### 3. 自定义分析

```python
# 分析月度表现
import pandas as pd

# 转换为DataFrame
df = pd.DataFrame(report['equity_curve'])
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# 计算月度收益
monthly_returns = df['equity'].resample('M').last().pct_change().dropna()
print("月度收益率:")
print(monthly_returns)
```

## 注意事项

1. **数据质量**: 回测结果依赖于历史数据的质量，确保数据完整准确。

2. **滑点与手续费**: 当前回测系统未考虑滑点和手续费，实际交易可能会有差异。

3. **市场条件**: 过去的表现在未来不一定能够复制，市场条件可能发生变化。

4. **过拟合风险**: 避免过度优化参数以适应历史数据，这可能导致实盘表现不佳。

## 常见问题

### Q: 如何获取真实历史数据？

A: 当前版本使用模拟数据，您可以通过修改 `backtest_engine.py` 中的 `_get_historical_data` 方法来接入真实数据源。

### Q: 如何调整交易策略参数？

A: 策略参数在配置文件中定义，您可以创建自定义配置文件并通过 `--config` 参数指定。

### Q: 回测速度很慢怎么办？

A: 回测速度取决于数据量和策略复杂度，可以尝试减少时间框架数量或缩短回测期间。

## 更新日志

- v1.0.0: 初始版本，支持基本回测功能
- 支持多时间框架分析
- 支持OB叠加结果优化开单价格和止损
- 生成详细报告和可视化图表