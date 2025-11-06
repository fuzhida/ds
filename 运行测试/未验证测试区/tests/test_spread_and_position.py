"""
点差门控与仓位大小计算测试
"""

import unittest
import os
import sys
from unittest.mock import Mock

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from trading_executor import TradingExecutor, OrderManager


class TestSpreadGatingAndPositionSize(unittest.TestCase):
    def setUp(self):
        # 真实 Config，用以验证字段存在与默认值
        self.real_config = Config()
        # 使用 Mock 包装，以便根据需要覆盖字段
        self.config = Mock(spec=Config)
        # 执行相关字段
        self.config.symbol = "PAXGUSDT"
        self.config.default_position_size = 0.2
        self.config.max_position_size = 0.5
        self.config.default_stop_loss_pct = 0.01
        self.config.default_take_profit_pcts = [0.005, 0.01]
        self.config.spread_tolerance = 0.001  # 0.1%
        # 其它必需字段
        self.config.execution_interval = 30
        self.config.max_concurrent_orders = 1

        self.logger = Mock()
        self.exchange_manager = Mock()
        self.order_manager = Mock(spec=OrderManager)
        self.position_manager = Mock()
        self.risk_manager = Mock()

        self.executor = TradingExecutor(
            self.config,
            self.logger,
            self.exchange_manager,
            self.risk_manager,
            self.position_manager,
            self.order_manager
        )

        self.buy_signal = {
            "symbol": "PAXGUSDT",
            "signal": "BUY",
            "confidence": 0.5
        }

    def test_spread_gating_blocks_on_wide_spread(self):
        # 风险允许
        self.risk_manager.check_risk_limits.return_value = {"allowed": True}
        # 当前价格
        self.exchange_manager.get_current_price.return_value = 100.0
        # 订单簿：点差为 0.5%，超过 0.1% 容忍度
        self.exchange_manager.safe_fetch_order_book.return_value = {
            "bids": [[99.75, 1]],
            "asks": [[100.25, 1]]
        }

        result = self.executor.execute_signal(self.buy_signal)
        self.assertFalse(result)
        self.order_manager.create_order.assert_not_called()

    def test_spread_gating_allows_on_narrow_spread_and_checks_size(self):
        # 风险允许
        self.risk_manager.check_risk_limits.return_value = {"allowed": True}
        # 当前价格
        self.exchange_manager.get_current_price.return_value = 100.0
        # 订单簿：点差为 0.06%，低于 0.1% 容忍度
        self.exchange_manager.safe_fetch_order_book.return_value = {
            "bids": [[99.95, 1]],
            "asks": [[100.01, 1]]
        }

        # 模拟下游创建与执行成功
        self.order_manager.create_order.return_value = {
            "id": "ord_1",
            "symbol": "PAXGUSDT",
            "side": "BUY",
            "type": "MARKET",
            "size": 0.1,
            "price": 100.0,
            "status": "filled",
            "filled": 0.1,
            "remaining": 0.0
        }
        self.order_manager.execute_order.return_value = {"success": True, "order_id": "ord_1", "status": "filled"}
        self.position_manager.create_position.return_value = {"id": "pos_1", "status": "open"}

        result = self.executor.execute_signal(self.buy_signal)
        self.assertTrue(result)

        # 校验仓位大小：默认 0.2 * 置信度 0.5 = 0.1，且未超过 max 0.5
        args, kwargs = self.order_manager.create_order.call_args
        self.assertEqual(kwargs.get("size"), 0.1)


if __name__ == "__main__":
    unittest.main()