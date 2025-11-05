from backtest_engine import BacktestEngine

# 创建回测引擎
engine = BacktestEngine()

# 获取一些历史数据来检查格式
print("获取历史数据...")
historical_data = engine._get_historical_data("BTC/USDT", ["1h"], "2024-12-15", "2024-12-22")

# 检查数据格式
if historical_data and "1h" in historical_data:
    df = historical_data["1h"]
    print(f"数据形状: {df.shape}")
    print(f"数据列: {df.columns.tolist()}")
    print(f"前5行数据:\n{df.head()}")
else:
    print("无法获取历史数据")