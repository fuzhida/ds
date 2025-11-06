import json
from types import SimpleNamespace


def test_spread_gate_writes_contextual_log(tmp_path, monkeypatch):
    # 构造配置，指定上下文日志文件路径与价差容忍度
    log_path = tmp_path / "contextual.log"
    config = SimpleNamespace(
        symbol="BTC/USDC",
        spread_tolerance=0.001,  # 0.1%
        contextual_log_file=str(log_path),
        execution_interval=1,
        max_concurrent_orders=2,
    )

    # 构造必要的依赖与模拟
    class DummyLogger:
        def info(self, *args, **kwargs):
            pass
        def warning(self, *args, **kwargs):
            pass
        def error(self, *args, **kwargs):
            pass
        def debug(self, *args, **kwargs):
            pass

    logger = DummyLogger()

    class DummyExchange:
        def get_current_price(self, symbol):
            return 100.0
        def safe_fetch_order_book(self, symbol):
            # 制造宽价差，触发门控
            return {
                'bids': [[99.0, 1]],
                'asks': [[101.0, 1]]
            }

    class DummyRiskManager:
        def check_risk_limits(self, signal):
            return {"allowed": True}

    class DummyPositionManager:
        def create_position(self, *args, **kwargs):
            return {"id": "pos_1"}

    from trading_executor import TradingExecutor, OrderManager

    exchange = DummyExchange()
    risk = DummyRiskManager()
    pos = DummyPositionManager()
    order_manager = OrderManager(config, logger, exchange)

    executor = TradingExecutor(config, logger, exchange, risk, pos, order_manager=order_manager)

    # 执行信号（应被点差门控拒绝）
    signal = {"signal": "BUY", "symbol": "BTC/USDC", "confidence": 0.9}
    allowed = executor.execute_signal(signal)
    assert allowed is False

    # 校验日志文件内容
    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1
    record = json.loads(lines[-1])
    assert record.get("reason") == "spread_too_wide"
    assert record.get("module") == "trading_executor"
    assert record.get("symbol") == "BTC/USDC"