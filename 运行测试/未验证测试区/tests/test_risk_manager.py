"""
风险管理模块测试
"""

import unittest
import os
import sys
from unittest.mock import Mock
import pandas as pd
import numpy as np

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from risk_manager import RiskManager, PositionManager


class TestRiskManager(unittest.TestCase):
    """风险管理器测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建模拟配置
        self.config = Mock(spec=Config)
        self.config.risk_per_trade = 0.02  # 2%
        self.config.max_risk_per_day = 0.05  # 5%
        self.config.max_positions = 3
        self.config.max_correlation = 0.7
        self.config.stop_loss_atr_multiplier = 2.0
        self.config.take_profit_atr_multiplier = 3.0
        self.config.trailing_stop_atr_multiplier = 1.5
        self.config.trailing_stop_activation_atr = 2.0
        
        # 创建模拟日志器
        self.logger = Mock()
        
        # 创建风险管理器实例
        self.risk_manager = RiskManager(self.config, self.logger)
        
        # 创建测试数据
        self.create_test_data()
    
    def create_test_data(self):
        """创建测试数据"""
        # 创建测试数据
        timestamps = pd.date_range(start="2023-01-01", periods=100, freq="1H")
        
        # 创建价格数据
        closes = 100 + np.cumsum(np.random.randn(100) * 0.5)
        highs = closes + np.random.rand(100) * 2
        lows = closes - np.random.rand(100) * 2
        opens = closes + np.random.randn(100) * 0.5
        volumes = np.random.randint(1000, 5000, size=100)
        
        self.test_data = pd.DataFrame({
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes
        })
        
        # 计算ATR
        self.atr = self.risk_manager.calculate_atr(self.test_data, 14)
    
    def test_calculate_position_size(self):
        """测试仓位大小计算"""
        account_balance = 10000.0
        entry_price = 1850.5
        stop_loss_price = 1845.0
        
        result = self.risk_manager.calculate_position_size(account_balance, entry_price, stop_loss_price)
        
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0.0)
        
        # 验证风险金额不超过账户余额的2%
        risk_amount = abs(entry_price - stop_loss_price) * result
        self.assertLessEqual(risk_amount, account_balance * self.config.risk_per_trade)
    
    def test_calculate_stop_loss(self):
        """测试止损计算"""
        entry_price = 1850.5
        signal = "BUY"
        atr = 2.0
        
        result = self.risk_manager.calculate_stop_loss(entry_price, signal, atr)
        
        self.assertIsInstance(result, float)
        
        if signal == "BUY":
            self.assertLess(result, entry_price)  # 买单止损应低于入场价
        else:
            self.assertGreater(result, entry_price)  # 卖单止损应高于入场价
    
    def test_calculate_take_profit(self):
        """测试止盈计算"""
        entry_price = 1850.5
        signal = "BUY"
        atr = 2.0
        
        result = self.risk_manager.calculate_take_profit(entry_price, signal, atr)
        
        self.assertIsInstance(result, float)
        
        if signal == "BUY":
            self.assertGreater(result, entry_price)  # 买单止盈应高于入场价
        else:
            self.assertLess(result, entry_price)  # 卖单止盈应低于入场价
    
    def test_calculate_trailing_stop(self):
        """测试移动止损计算"""
        current_price = 1860.0
        entry_price = 1850.5
        signal = "BUY"
        current_stop_loss = 1848.0
        atr = 2.0
        
        result = self.risk_manager.calculate_trailing_stop(
            current_price, entry_price, signal, current_stop_loss, atr
        )
        
        self.assertIsInstance(result, float)
        
        if signal == "BUY":
            self.assertGreaterEqual(result, current_stop_loss)  # 买单移动止损应不低于当前止损
        else:
            self.assertLessEqual(result, current_stop_loss)  # 卖单移动止损应不高于当前止损
    
    def test_check_risk_limits(self):
        """测试风险限制检查"""
        # 创建模拟持仓
        positions = [
            {"symbol": "PAXGUSDT", "size": 0.1, "entry_price": 1850.5, "side": "BUY"},
            {"symbol": "BTCUSDT", "size": 0.05, "entry_price": 45000.0, "side": "SELL"}
        ]
        
        # 创建模拟账户信息
        account_info = {
            "balance": 10000.0,
            "unrealized_pnl": 50.0,
            "daily_pnl": 100.0
        }
        
        # 创建新交易信息
        new_trade = {
            "symbol": "ETHUSDT",
            "size": 0.1,
            "entry_price": 3000.0,
            "side": "BUY"
        }
        
        result = self.risk_manager.check_risk_limits(positions, account_info, new_trade)
        
        self.assertIsInstance(result, dict)
        self.assertIn("allowed", result)
        self.assertIn("reason", result)
    
    def test_check_position_correlation(self):
        """测试持仓相关性检查"""
        # 创建模拟持仓
        positions = [
            {"symbol": "PAXGUSDT", "size": 0.1, "entry_price": 1850.5, "side": "BUY"},
            {"symbol": "BTCUSDT", "size": 0.05, "entry_price": 45000.0, "side": "BUY"}
        ]
        
        # 创建新交易信息
        new_trade = {
            "symbol": "ETHUSDT",
            "size": 0.1,
            "entry_price": 3000.0,
            "side": "BUY"
        }
        
        result = self.risk_manager.check_position_correlation(positions, new_trade)
        
        self.assertIsInstance(result, dict)
        self.assertIn("correlation", result)
        self.assertIn("allowed", result)
        self.assertIn("reason", result)
    
    def test_check_daily_risk(self):
        """测试每日风险检查"""
        # 创建模拟账户信息
        account_info = {
            "balance": 10000.0,
            "unrealized_pnl": 50.0,
            "daily_pnl": -600.0  # 超过每日风险限制
        }
        
        # 创建新交易信息
        new_trade = {
            "symbol": "ETHUSDT",
            "size": 0.1,
            "entry_price": 3000.0,
            "side": "BUY",
            "risk_amount": 200.0
        }
        
        result = self.risk_manager.check_daily_risk(account_info, new_trade)
        
        self.assertIsInstance(result, dict)
        self.assertIn("allowed", result)
        self.assertIn("reason", result)
        self.assertFalse(result["allowed"])  # 应该不允许交易
    
    def test_update_risk_metrics(self):
        """测试风险指标更新"""
        # 创建模拟持仓
        positions = [
            {"symbol": "PAXGUSDT", "size": 0.1, "entry_price": 1850.5, "side": "BUY"},
            {"symbol": "BTCUSDT", "size": 0.05, "entry_price": 45000.0, "side": "SELL"}
        ]
        
        # 创建模拟账户信息
        account_info = {
            "balance": 10000.0,
            "unrealized_pnl": 50.0,
            "daily_pnl": 100.0
        }
        
        # 调用方法
        self.risk_manager.update_risk_metrics(positions, account_info)
        
        # 验证风险指标已更新
        self.assertIsNotNone(self.risk_manager.total_exposure)
        self.assertIsNotNone(self.risk_manager.total_risk)
        self.assertIsNotNone(self.risk_manager.daily_risk)
        self.assertIsNotNone(self.risk_manager.position_count)
    
    def test_calculate_atr(self):
        """测试ATR计算"""
        df = self.test_data
        period = 14
        
        result = self.risk_manager.calculate_atr(df, period)
        
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0.0)


class TestPositionManager(unittest.TestCase):
    """持仓管理器测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建模拟配置
        self.config = Mock(spec=Config)
        
        # 创建模拟日志器
        self.logger = Mock()
        
        # 创建模拟风险管理器
        self.risk_manager = Mock(spec=RiskManager)
        
        # 创建持仓管理器实例
        self.position_manager = PositionManager(self.config, self.logger, self.risk_manager)
        
        # 创建测试数据
        self.create_test_data()
    
    def create_test_data(self):
        """创建测试数据"""
        self.test_position = {
            "id": "12345",
            "symbol": "PAXGUSDT",
            "side": "BUY",
            "size": 0.1,
            "entry_price": 1850.5,
            "stop_loss": 1845.0,
            "take_profit": 1860.0,
            "timestamp": "2023-01-01T10:00:00Z",
            "status": "open"
        }
    
    def test_create_position(self):
        """测试创建持仓"""
        # 创建交易信号
        signal = {
            "signal": "BUY",
            "confidence": 0.8,
            "reasoning": "测试信号"
        }
        
        # 定义参数
        entry_price = 1850.5
        position_size = 0.1
        stop_loss = 1845.0
        take_profits = [1860.0, 1870.0]
        account_balance = 10000.0
        
        # 调用方法
        result = self.position_manager.create_position(
            signal, entry_price, position_size, stop_loss, take_profits, account_balance
        )
        
        # 验证结果
        self.assertIsInstance(result, dict)
        self.assertEqual(result["signal_type"], "BUY")
        self.assertEqual(result["entry_price"], entry_price)
        self.assertEqual(result["position_size"], position_size)
        self.assertEqual(result["stop_loss"], stop_loss)
        self.assertEqual(result["take_profits"], take_profits)
        self.assertEqual(result["status"], "open")
        self.assertIn("id", result)
        self.assertIn("created_at", result)
        
        # 验证风险管理器调用
        self.risk_manager.add_position.assert_called_once_with(result)
    
    def test_update_position(self):
        """测试更新持仓"""
        # 创建测试仓位
        position = {
            "id": "test_position",
            "signal_type": "BUY",
            "entry_price": 1850.5,
            "position_size": 0.1,
            "stop_loss": 1845.0,
            "take_profits": [1860.0, 1870.0],
            "current_take_profit_index": 0,
            "status": "open"
        }
        
        # 添加持仓到管理器
        self.position_manager.positions[position["id"]] = position
        
        # 定义参数
        current_price = 1855.0
        atr = 5.0
        
        # 调用方法
        result = self.position_manager.update_position(position["id"], current_price, atr)
        
        # 验证结果
        self.assertIsInstance(result, dict)
        self.assertEqual(result["current_price"], current_price)
        self.assertIn("profit_loss", result)
        self.assertIn("profit_loss_percentage", result)
        self.assertIn("updated_at", result)
    
    def test_close_position(self):
        """测试关闭持仓"""
        # 添加持仓到管理器
        self.position_manager.positions[self.test_position["id"]] = self.test_position
        
        # 调用方法 - 使用实际的方法签名
        result = self.position_manager.close_position(
            self.test_position["id"], 
            close_price=1855.0, 
            close_reason="take_profit"
        )
        
        # 验证结果
        self.assertTrue(result)
        
        # 验证持仓已关闭
        closed_position = self.position_manager.get_position(self.test_position["id"])
        self.assertEqual(closed_position["status"], "closed")
        self.assertEqual(closed_position["close_price"], 1855.0)
        self.assertEqual(closed_position["close_reason"], "take_profit")
        self.assertIn("closed_at", closed_position)
    
    def test_get_position(self):
        """测试获取持仓"""
        # 添加持仓到管理器
        self.position_manager.positions[self.test_position["id"]] = self.test_position
        
        # 调用方法
        result = self.position_manager.get_position(self.test_position["id"])
        
        # 验证结果
        self.assertEqual(result, self.test_position)
    
    def test_get_all_positions(self):
        """测试获取所有持仓"""
        # 添加多个持仓到管理器
        position1 = self.test_position.copy()
        position1["id"] = "12345"
        
        position2 = self.test_position.copy()
        position2["id"] = "67890"
        position2["symbol"] = "BTCUSDT"
        
        self.position_manager.positions[position1["id"]] = position1
        self.position_manager.positions[position2["id"]] = position2
        
        # 调用方法
        result = self.position_manager.get_all_positions()
        
        # 验证结果
        self.assertEqual(len(result), 2)
        self.assertIn(position1["id"], result)
        self.assertIn(position2["id"], result)
        self.assertEqual(result[position1["id"]], position1)
        self.assertEqual(result[position2["id"]], position2)
    
    def test_get_open_positions(self):
        """测试获取开仓持仓"""
        # 添加多个持仓到管理器
        position1 = self.test_position.copy()
        position1["id"] = "12345"
        position1["status"] = "open"
        
        position2 = self.test_position.copy()
        position2["id"] = "67890"
        position2["symbol"] = "BTCUSDT"
        position2["status"] = "closed"
        
        self.position_manager.positions[position1["id"]] = position1
        self.position_manager.positions[position2["id"]] = position2
        
        # 调用方法
        result = self.position_manager.get_open_positions()
        
        # 验证结果
        self.assertEqual(len(result), 1)
        self.assertIn(position1["id"], result)
        self.assertEqual(result[position1["id"]], position1)
    
    def test_position_pnl_calculation(self):
        """测试仓位盈亏计算（在update_position中）"""
        # 创建测试仓位
        position = {
            "id": "test_position",
            "signal_type": "BUY",
            "entry_price": 1850.5,
            "position_size": 0.1,
            "stop_loss": 1845.0,
            "take_profits": [1860.0, 1870.0],
            "current_take_profit_index": 0,
            "status": "open"
        }
        
        # 添加持仓到管理器
        self.position_manager.positions[position["id"]] = position
        
        # 定义参数
        current_price = 1855.0
        atr = 5.0
        
        # 调用方法
        result = self.position_manager.update_position(position["id"], current_price, atr)
        
        # 验证盈亏计算
        expected_pnl = (current_price - position["entry_price"]) * position["position_size"]
        self.assertAlmostEqual(result["profit_loss"], expected_pnl, places=5)
        
        expected_pnl_pct = (current_price - position["entry_price"]) / position["entry_price"]
        self.assertAlmostEqual(result["profit_loss_percentage"], expected_pnl_pct, places=5)
    
    def test_stop_loss_and_take_profit_in_update(self):
        """测试止损和止盈检查（在update_position中）"""
        # 创建测试仓位
        position = {
            "id": "test_position",
            "signal_type": "BUY",
            "entry_price": 1850.5,
            "position_size": 0.1,
            "stop_loss": 1845.0,
            "take_profits": [1860.0, 1870.0],
            "current_take_profit_index": 0,
            "status": "open"
        }
        
        # 测试止损触发
        position_stop = position.copy()
        position_stop["id"] = "test_stop"
        self.position_manager.positions[position_stop["id"]] = position_stop
        
        # 价格低于止损价
        result = self.position_manager.update_position(position_stop["id"], 1844.0, 5.0)
        self.assertEqual(result["status"], "closed")
        self.assertEqual(result["close_reason"], "止损")
        
        # 测试止盈触发
        position_profit = position.copy()
        position_profit["id"] = "test_profit"
        self.position_manager.positions[position_profit["id"]] = position_profit
        
        # 价格高于止盈价
        result = self.position_manager.update_position(position_profit["id"], 1861.0, 5.0)
        self.assertEqual(result["status"], "closed")
        self.assertEqual(result["close_reason"], "止盈1")
        
        # 测试移动止损
        position_trail = position.copy()
        position_trail["id"] = "test_trail"
        self.position_manager.positions[position_trail["id"]] = position_trail
        
        # 模拟风险管理器返回新的止损价
        self.risk_manager.calculate_trailing_stop.return_value = 1848.0
        
        # 价格上涨，触发移动止损
        result = self.position_manager.update_position(position_trail["id"], 1860.0, 5.0)
        self.assertEqual(result["stop_loss"], 1848.0)


if __name__ == "__main__":
    unittest.main()