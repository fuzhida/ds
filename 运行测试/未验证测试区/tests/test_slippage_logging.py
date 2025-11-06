import json
from types import SimpleNamespace


class DummyLogger:
    def info(self, *args, **kwargs):
        pass
    def warning(self, *args, **kwargs):
        pass
    def error(self, *args, **kwargs):
        pass
    def debug(self, *args, **kwargs):
        pass


class DummyExchange:
    def safe_create_order(self, symbol, type, side, amount, price, params):
        # 返回一个伪订单ID，供 _wait_for_order_execution 使用
        return {"id": f"{side}_order_1"}


def test_slippage_logging_buy_side(tmp_path):
    # 配置：设置极低滑点容忍度以确保触发
    log_path = tmp_path / "contextual.log"
    config = SimpleNamespace(
        symbol="BTC/USDC",
        slippage_tolerance=0.001,  # 0.1%
        contextual_log_file=str(log_path),
    )

    from trading_executor import OrderManager

    order_manager = OrderManager(config, DummyLogger(), DummyExchange())

    # 伪造订单与执行结果：滑点 1%
    order = {
        "position_size": 1.0,
        "entry_price": 100.0,
        "stop_loss": 0.0,
        "take_profits": []
    }

    # 打桩 _wait_for_order_execution 返回实际价格 101，触发滑点门控日志
    order_manager._wait_for_order_execution = lambda order_id, timeout=None: {
        "id": order_id,
        "price": 101.0,
        "filled": 1.0
    }

    result = order_manager._execute_buy_order(order)
    assert result.get("success") is True

    # 验证日志写入
    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1
    record = json.loads(lines[-1])
    assert record.get("reason") == "slippage_exceeds_tolerance"
    assert record.get("module") == "order_manager"
    assert record.get("symbol") == "BTC/USDC"
    assert record.get("extras", {}).get("side") == "buy"


def test_slippage_logging_sell_side(tmp_path):
    # 配置：设置极低滑点容忍度以确保触发
    log_path = tmp_path / "contextual.log"
    config = SimpleNamespace(
        symbol="BTC/USDC",
        slippage_tolerance=0.001,  # 0.1%
        contextual_log_file=str(log_path),
    )

    from trading_executor import OrderManager

    order_manager = OrderManager(config, DummyLogger(), DummyExchange())

    # 伪造订单与执行结果：滑点 1%
    order = {
        "position_size": 1.0,
        "entry_price": 100.0,
        "stop_loss": 0.0,
        "take_profits": []
    }

    # 打桩 _wait_for_order_execution 返回实际价格 99，触发滑点门控日志
    order_manager._wait_for_order_execution = lambda order_id, timeout=None: {
        "id": order_id,
        "price": 99.0,
        "filled": 1.0
    }

    result = order_manager._execute_sell_order(order)
    assert result.get("success") is True

    # 验证日志写入
    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1
    record = json.loads(lines[-1])
    assert record.get("reason") == "slippage_exceeds_tolerance"
    assert record.get("module") == "order_manager"
    assert record.get("symbol") == "BTC/USDC"
    assert record.get("extras", {}).get("side") == "sell"