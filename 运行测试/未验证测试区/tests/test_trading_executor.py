"""
交易执行模块测试
"""

import unittest
import os
import sys
from unittest.mock import Mock, patch
import asyncio

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from trading_executor import OrderManager, TradingExecutor


class TestOrderManager(unittest.TestCase):
    """订单管理器测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建模拟配置
        self.config = Mock(spec=Config)
        self.config.order_timeout = 30
        self.config.max_retries = 3
        self.config.retry_delay = 1
        
        # 创建模拟日志器
        self.logger = Mock()
        
        # 创建模拟交易所管理器
        self.exchange_manager = Mock()
        
        # 创建订单管理器实例
        self.order_manager = OrderManager(self.config, self.logger, self.exchange_manager)
        
        # 创建测试数据
        self.create_test_data()
    
    def create_test_data(self):
        """创建测试数据"""
        self.test_order = {
            "id": "12345",
            "symbol": "PAXGUSDT",
            "side": "BUY",
            "type": "MARKET",
            "size": 0.1,
            "price": None,  # 市价单不需要价格
            "status": "pending",
            "timestamp": "2023-01-01T10:00:00Z"
        }
    
    def test_create_order(self):
        """测试创建订单"""
        # 创建订单信息
        order_info = {
            "symbol": "PAXGUSDT",
            "side": "BUY",
            "type": "MARKET",
            "size": 0.1,
            "price": None
        }
        
        # 模拟交易所返回
        self.exchange_manager.create_order.return_value = {
            "id": "12345",
            "symbol": "PAXGUSDT",
            "side": "BUY",
            "type": "MARKET",
            "size": 0.1,
            "price": 1850.5,
            "status": "filled",
            "filled": 0.1,
            "remaining": 0.0
        }
        
        # 调用方法
        result = self.order_manager.create_order(order_info)
        
        # 验证结果
        self.assertIsInstance(result, dict)
        self.assertEqual(result["symbol"], "PAXGUSDT")
        self.assertEqual(result["side"], "BUY")
        self.assertEqual(result["type"], "MARKET")
        self.assertEqual(result["size"], 0.1)
        self.assertEqual(result["status"], "filled")
        
        # 验证交易所调用
        self.exchange_manager.create_order.assert_called_once_with(order_info)
    
    def test_create_limit_order(self):
        """测试创建限价单"""
        # 创建订单信息
        order_info = {
            "symbol": "PAXGUSDT",
            "side": "BUY",
            "type": "LIMIT",
            "size": 0.1,
            "price": 1845.0
        }
        
        # 模拟交易所返回
        self.exchange_manager.create_order.return_value = {
            "id": "12345",
            "symbol": "PAXGUSDT",
            "side": "BUY",
            "type": "LIMIT",
            "size": 0.1,
            "price": 1845.0,
            "status": "open",
            "filled": 0.0,
            "remaining": 0.1
        }
        
        # 调用方法
        result = self.order_manager.create_order(order_info)
        
        # 验证结果
        self.assertIsInstance(result, dict)
        self.assertEqual(result["symbol"], "PAXGUSDT")
        self.assertEqual(result["side"], "BUY")
        self.assertEqual(result["type"], "LIMIT")
        self.assertEqual(result["size"], 0.1)
        self.assertEqual(result["price"], 1845.0)
        self.assertEqual(result["status"], "open")
        
        # 验证交易所调用
        self.exchange_manager.create_order.assert_called_once_with(order_info)
    
    def test_cancel_order(self):
        """测试取消订单"""
        # 添加订单到管理器
        self.order_manager.orders[self.test_order["id"]] = self.test_order
        
        # 模拟交易所返回
        self.exchange_manager.cancel_order.return_value = {
            "id": "12345",
            "status": "cancelled"
        }
        
        # 调用方法
        result = self.order_manager.cancel_order(self.test_order["id"])
        
        # 验证结果
        self.assertTrue(result)
        
        # 验证订单状态已更新
        updated_order = self.order_manager.get_order(self.test_order["id"])
        self.assertEqual(updated_order["status"], "cancelled")
        
        # 验证交易所调用
        self.exchange_manager.cancel_order.assert_called_once_with(self.test_order["id"])
    
    def test_get_order(self):
        """测试获取订单"""
        # 添加订单到管理器
        self.order_manager.orders[self.test_order["id"]] = self.test_order
        
        # 调用方法
        result = self.order_manager.get_order(self.test_order["id"])
        
        # 验证结果
        self.assertEqual(result, self.test_order)
    
    def test_get_all_orders(self):
        """测试获取所有订单"""
        # 添加多个订单到管理器
        order1 = self.test_order.copy()
        order1["id"] = "12345"
        
        order2 = self.test_order.copy()
        order2["id"] = "67890"
        order2["symbol"] = "BTCUSDT"
        
        self.order_manager.orders[order1["id"]] = order1
        self.order_manager.orders[order2["id"]] = order2
        
        # 调用方法
        result = self.order_manager.get_all_orders()
        
        # 验证结果
        self.assertEqual(len(result), 2)
        self.assertIn(order1, result)
        self.assertIn(order2, result)
    
    def test_update_order_status(self):
        """测试更新订单状态"""
        # 添加订单到管理器
        self.order_manager.orders[self.test_order["id"]] = self.test_order
        
        # 模拟交易所返回
        self.exchange_manager.get_order.return_value = {
            "id": "12345",
            "status": "filled",
            "filled": 0.1,
            "remaining": 0.0
        }
        
        # 调用方法
        result = self.order_manager.update_order_status(self.test_order["id"])
        
        # 验证结果
        self.assertTrue(result)
        
        # 验证订单状态已更新
        updated_order = self.order_manager.get_order(self.test_order["id"])
        self.assertEqual(updated_order["status"], "filled")
        self.assertEqual(updated_order["filled"], 0.1)
        self.assertEqual(updated_order["remaining"], 0.0)
        
        # 验证交易所调用
        self.exchange_manager.get_order.assert_called_once_with(self.test_order["id"])
    
    def test_retry_failed_order(self):
        """测试重试失败订单"""
        # 添加失败订单到管理器
        failed_order = self.test_order.copy()
        failed_order["status"] = "failed"
        failed_order["retry_count"] = 0
        self.order_manager.orders[failed_order["id"]] = failed_order
        
        # 模拟交易所返回
        self.exchange_manager.create_order.return_value = {
            "id": "67890",
            "symbol": "PAXGUSDT",
            "side": "BUY",
            "type": "MARKET",
            "size": 0.1,
            "price": 1850.5,
            "status": "filled",
            "filled": 0.1,
            "remaining": 0.0
        }
        
        # 调用方法
        result = self.order_manager.retry_failed_order(failed_order["id"])
        
        # 验证结果
        self.assertTrue(result)
        
        # 验证原订单状态已更新
        updated_order = self.order_manager.get_order(failed_order["id"])
        self.assertEqual(updated_order["status"], "replaced")
        
        # 验证新订单已创建
        self.assertIn("67890", self.order_manager.orders)
        
        # 验证交易所调用
        self.exchange_manager.create_order.assert_called_once()


class TestTradingExecutor(unittest.TestCase):
    """交易执行器测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建模拟配置
        self.config = Mock(spec=Config)
        self.config.execution_interval = 60
        self.config.order_check_interval = 10
        self.config.position_check_interval = 30
        self.config.symbol = "PAXGUSDT"  # 添加symbol属性
        self.config.default_position_size = 0.1  # 添加默认仓位大小
        self.config.max_position_size = 1.0  # 添加最大仓位大小
        self.config.default_stop_loss_pct = 0.02  # 添加默认止损百分比
        self.config.default_take_profit_pcts = [0.01, 0.02, 0.03]  # 添加默认止盈百分比列表
        
        # 创建模拟日志器
        self.logger = Mock()
        
        # 创建模拟组件
        self.exchange_manager = Mock()
        self.order_manager = Mock(spec=OrderManager)
        self.position_manager = Mock()
        self.risk_manager = Mock()
        
        # 创建交易执行器实例，注入模拟的order_manager
        self.trading_executor = TradingExecutor(
            self.config, 
            self.logger, 
            self.exchange_manager,
            self.risk_manager,
            self.position_manager,
            self.order_manager  # 注入模拟的order_manager
        )
        
        # 创建测试数据
        self.create_test_data()
    
    def create_test_data(self):
        """创建测试数据"""
        self.test_signal = {
            "symbol": "PAXGUSDT",
            "signal": "BUY",
            "confidence": 0.8,
            "entry_price": 1850.5,
            "stop_loss": 1845.0,
            "take_profit": 1860.0,
            "size": 0.1,
            "reasoning": "Strong bullish momentum"
        }
    
    def test_execute_signal(self):
        """测试执行信号"""
        # 模拟风险管理器返回
        self.risk_manager.check_risk_limits.return_value = {
            "allowed": True,
            "reason": "Risk limits OK"
        }
        
        # 模拟交易所管理器返回价格
        self.exchange_manager.get_current_price.return_value = 1850.5
        
        # 模拟订单管理器返回
        self.order_manager.create_order.return_value = {
            "id": "12345",
            "symbol": "PAXGUSDT",
            "side": "BUY",
            "type": "MARKET",
            "size": 0.1,
            "price": 1850.5,
            "status": "filled",
            "filled": 0.1,
            "remaining": 0.0
        }
        
        # 模拟订单执行结果
        self.order_manager.execute_order.return_value = {
            "success": True,
            "order_id": "12345",
            "status": "filled"
        }
        
        # 模拟持仓管理器返回
        self.position_manager.create_position.return_value = {
            "id": "pos_12345",
            "symbol": "PAXGUSDT",
            "side": "BUY",
            "size": 0.1,
            "entry_price": 1850.5,
            "stop_loss": 1845.0,
            "take_profit": 1860.0,
            "status": "open"
        }
        
        # 调用方法
        result = self.trading_executor.execute_signal(self.test_signal)
        
        # 验证结果
        self.assertTrue(result)
        
        # 验证风险管理器调用
        self.risk_manager.check_risk_limits.assert_called_once()
        
        # 验证订单管理器调用
        self.order_manager.create_order.assert_called_once()
        self.order_manager.execute_order.assert_called_once()
        
        # 验证持仓管理器调用
        self.position_manager.create_position.assert_called_once()
    
    def test_execute_signal_risk_rejected(self):
        """测试执行信号被风险管理拒绝"""
        # 模拟风险管理器返回
        self.risk_manager.check_risk_limits.return_value = {
            "allowed": False,
            "reason": "Daily risk limit exceeded"
        }
        
        # 调用方法
        result = self.trading_executor.execute_signal(self.test_signal)
        
        # 验证结果
        self.assertFalse(result)
        
        # 验证风险管理器调用
        self.risk_manager.check_risk_limits.assert_called_once()
        
        # 验证订单管理器未被调用
        self.order_manager.create_order.assert_not_called()
        
        # 验证持仓管理器未被调用
        self.position_manager.create_position.assert_not_called()
    
    def test_close_position(self):
        """测试平仓"""
        # 创建测试持仓
        position = {
            "id": "pos_12345",
            "symbol": "PAXGUSDT",
            "side": "BUY",
            "size": 0.1,
            "entry_price": 1850.5,
            "status": "open"
        }
        
        # 模拟交易所管理器返回价格
        self.trading_executor.exchange_manager.safe_fetch_ticker.return_value = {
            "symbol": "PAXGUSDT",
            "last": 1855.0,
            "bid": 1854.5,
            "ask": 1855.5
        }
        
        # 模拟订单管理器返回
        self.order_manager.create_order.return_value = {
            "id": "67890",
            "symbol": "PAXGUSDT",
            "side": "SELL",
            "type": "MARKET",
            "size": 0.1,
            "price": 1855.0,
            "status": "filled",
            "filled": 0.1,
            "remaining": 0.0
        }
        
        # 模拟持仓管理器返回
        self.position_manager.close_position.return_value = True
        
        # 调用方法
        result = self.trading_executor.close_position(position, "take_profit")
        
        # 验证结果
        self.assertTrue(result)
        
        # 验证订单管理器调用
        self.order_manager.create_order.assert_called_once()
        
        # 验证持仓管理器调用
        self.position_manager.close_position.assert_called_once()
    
    def test_check_orders(self):
        """测试检查订单"""
        # 模拟订单管理器返回
        self.order_manager.get_all_orders.return_value = [
            {
                "id": "12345",
                "symbol": "PAXGUSDT",
                "status": "open"
            },
            {
                "id": "67890",
                "symbol": "BTCUSDT",
                "status": "open"
            }
        ]
        
        # 模拟订单管理器更新状态
        self.order_manager.update_order_status.return_value = True
        
        # 调用方法
        self.trading_executor.check_orders()
        
        # 验证订单管理器调用
        self.order_manager.get_all_orders.assert_called_once()
        self.assertEqual(self.order_manager.update_order_status.call_count, 2)
    
    def test_check_positions(self):
        """测试检查持仓"""
        # 获取当前价格
        current_price = 1855.0
        
        # 模拟持仓管理器返回
        self.position_manager.get_open_positions.return_value = [
            {
                "id": "pos_12345",
                "symbol": "PAXGUSDT",
                "side": "BUY",
                "size": 0.1,
                "entry_price": 1850.5,
                "stop_loss": 1845.0,
                "take_profit": 1860.0,
                "status": "open"
            }
        ]
        
        # 模拟持仓管理器检查
        self.position_manager.check_stop_loss.return_value = False
        self.position_manager.check_take_profit.return_value = False
        self.position_manager.check_trailing_stop.return_value = False
        
        # 调用方法
        self.trading_executor.check_positions(current_price)
        
        # 验证持仓管理器调用
        self.position_manager.get_open_positions.assert_called_once()
        self.position_manager.check_stop_loss.assert_called_once()
        self.position_manager.check_take_profit.assert_called_once()
        self.position_manager.check_trailing_stop.assert_called_once()
    
    def test_check_positions_stop_loss_triggered(self):
        """测试检查持仓触发止损"""
        # 获取当前价格
        current_price = 1844.0
        
        # 模拟持仓管理器返回
        self.position_manager.get_open_positions.return_value = [
            {
                "id": "pos_12345",
                "symbol": "PAXGUSDT",
                "side": "BUY",
                "size": 0.1,
                "entry_price": 1850.5,
                "stop_loss": 1845.0,
                "take_profit": 1860.0,
                "status": "open"
            }
        ]
        
        # 模拟持仓管理器检查
        self.position_manager.check_stop_loss.return_value = True
        self.position_manager.check_take_profit.return_value = False
        self.position_manager.check_trailing_stop.return_value = False
        
        # 模拟订单管理器返回
        self.order_manager.create_order.return_value = {
            "id": "67890",
            "symbol": "PAXGUSDT",
            "side": "SELL",
            "type": "MARKET",
            "size": 0.1,
            "price": 1844.0,
            "status": "filled",
            "filled": 0.1,
            "remaining": 0.0
        }
        
        # 模拟持仓管理器平仓
        self.position_manager.close_position.return_value = True
        
        # 调用方法
        self.trading_executor.check_positions(current_price)
        
        # 验证持仓管理器调用
        self.position_manager.get_open_positions.assert_called_once()
        self.position_manager.check_stop_loss.assert_called_once()
        self.position_manager.check_take_profit.assert_called_once()
        self.position_manager.check_trailing_stop.assert_called_once()
        
        # 验证订单管理器调用
        self.order_manager.create_order.assert_called_once()
        
        # 验证持仓管理器平仓调用
        self.position_manager.close_position.assert_called_once()
    
    def test_start(self):
        """测试启动交易执行器"""
        # 模拟异步循环
        with patch('asyncio.create_task') as mock_create_task:
            # 调用方法
            self.trading_executor.start()
            
            # 验证结果
            self.assertTrue(self.trading_executor.running)
            
            # 验证异步任务创建
            mock_create_task.assert_called()
    
    def test_stop(self):
        """测试停止交易执行器"""
        # 设置运行状态
        self.trading_executor.running = True
        
        # 调用方法
        self.trading_executor.stop()
        
        # 验证结果
        self.assertFalse(self.trading_executor.running)
    
    def test_get_status(self):
        """测试获取状态"""
        # 模拟订单管理器返回
        self.order_manager.get_all_orders.return_value = [
            {"id": "12345", "status": "filled"},
            {"id": "67890", "status": "open"}
        ]
        
        # 模拟持仓管理器返回
        self.position_manager.get_open_positions.return_value = [
            {"id": "pos_12345", "symbol": "PAXGUSDT", "status": "open"}
        ]
        
        # 调用方法
        result = self.trading_executor.get_status()
        
        # 验证结果
        self.assertIsInstance(result, dict)
        self.assertIn("running", result)
        self.assertIn("orders", result)
        self.assertIn("positions", result)
        self.assertEqual(result["running"], False)
        self.assertEqual(result["orders"]["total"], 2)
        self.assertEqual(result["orders"]["open"], 1)
        self.assertEqual(result["orders"]["filled"], 1)
        self.assertEqual(result["positions"]["total"], 1)
        self.assertEqual(result["positions"]["open"], 1)


if __name__ == "__main__":
    unittest.main()