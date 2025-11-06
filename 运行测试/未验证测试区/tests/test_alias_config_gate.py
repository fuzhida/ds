import json
from types import SimpleNamespace


class DummyLogger:
    def info(self, *args, **kwargs): pass
    def warning(self, *args, **kwargs): pass
    def error(self, *args, **kwargs): pass
    def debug(self, *args, **kwargs): pass


def test_spread_alias_overrides_tolerance(tmp_path):
    # 别名 spread_threshold 生效，触发更严格的门控
    log_path = tmp_path / "contextual.log"
    config = SimpleNamespace(
        symbol="BTC/USDC",
        spread_tolerance=0.01,        # 1%（宽松）
        spread_threshold=0.0004,      # 0.04%（严格）
        contextual_log_file=str(log_path),
        execution_interval=1,
        max_concurrent_orders=2,
    )

    class DummyExchange:
        def get_current_price(self, symbol):
            return 100.0
        def safe_fetch_order_book(self, symbol):
            # 价差约 0.2%（> 0.04%，应触发拒单）
            return {
                'bids': [[99.8, 1]],
                'asks': [[100.2, 1]]
            }

    class DummyRisk:
        def check_risk_limits(self, signal):
            return {"allowed": True}

    class DummyPos:
        def create_position(self, *args, **kwargs):
            return {"id": "pos_1"}

    from trading_executor import TradingExecutor, OrderManager

    exchange = DummyExchange()
    logger = DummyLogger()
    risk = DummyRisk()
    pos = DummyPos()
    order_manager = OrderManager(config, logger, exchange)
    executor = TradingExecutor(config, logger, exchange, risk, pos, order_manager=order_manager)

    allowed = executor.execute_signal({"signal": "BUY", "symbol": "BTC/USDC", "confidence": 0.9})
    assert allowed is False
    # 检查使用了别名阈值记录
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    record = json.loads(lines[-1])
    assert record.get("reason") == "spread_too_wide"
    assert abs(record.get("extras", {}).get("tolerance") - 0.0004) < 1e-9


def test_slippage_alias_overrides_tolerance(tmp_path):
    # 别名 max_slippage_pct_entry 生效，用于 OrderManager
    log_path = tmp_path / "contextual.log"
    config = SimpleNamespace(
        symbol="BTC/USDC",
        slippage_tolerance=0.01,
        max_slippage_pct_entry=0.0005,  # 0.05%（严格）
        contextual_log_file=str(log_path),
    )

    class DummyExchange:
        def safe_create_order(self, symbol, type, side, amount, price, params):
            return {"id": f"{side}_order_1"}

    from trading_executor import OrderManager
    om = OrderManager(config, DummyLogger(), DummyExchange())
    order = {"position_size": 1.0, "entry_price": 100.0, "stop_loss": 0.0, "take_profits": []}
    om._wait_for_order_execution = lambda order_id, timeout=None: {"id": order_id, "price": 101.0, "filled": 1.0}
    result = om._execute_buy_order(order)
    assert result.get("success") is True
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    record = json.loads(lines[-1])
    assert record.get("reason") == "slippage_exceeds_tolerance"
    assert abs(record.get("extras", {}).get("tolerance") - 0.0005) < 1e-9