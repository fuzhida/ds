"""
交易所管理模块测试
"""

import unittest
import os
import sys
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from exchange_manager import ExchangeManager


class TestExchangeManager(unittest.TestCase):
    """交易所管理器测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建模拟配置
        self.config = Mock(spec=Config)
        self.config.exchange_name = "binance"
        self.config.sandbox = True
        self.config.api_key = "test_key"
        self.config.secret = "test_secret"
        self.config.timeout = 30000
        self.config.retries = 3
        self.config.rate_limit = 10
        
        # 创建模拟日志器
        self.logger = Mock()
        
        # 创建交易所管理器实例
        with patch('ccxt.binance'):
            self.exchange_manager = ExchangeManager(self.config, self.logger)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.exchange_manager.exchange_name, "binance")
        self.assertTrue(self.exchange_manager.sandbox)
        self.assertEqual(self.exchange_manager.api_key, "test_key")
        self.assertEqual(self.exchange_manager.secret, "test_secret")
    
    def test_connect(self):
        """测试连接交易所"""
        # 模拟交易所连接成功
        self.exchange_manager.exchange.load_markets = Mock(return_value=True)
        
        result = self.exchange_manager.connect()
        
        self.assertTrue(result)
        self.exchange_manager.exchange.load_markets.assert_called_once()
    
    def test_connect_failure(self):
        """测试连接交易所失败"""
        # 模拟交易所连接失败
        self.exchange_manager.exchange.load_markets = Mock(side_effect=Exception("连接失败"))
        
        result = self.exchange_manager.connect()
        
        self.assertFalse(result)
    
    def test_safe_fetch_ohlcv(self):
        """测试安全获取OHLCV数据"""
        # 模拟OHLCV数据
        mock_ohlcv = [
            [1609459200000, 100.0, 110.0, 95.0, 105.0, 1000],
            [1609459260000, 105.0, 115.0, 100.0, 110.0, 1200]
        ]
        
        self.exchange_manager.exchange.fetch_ohlcv = Mock(return_value=mock_ohlcv)
        
        result = self.exchange_manager.safe_fetch_ohlcv("BTC/USDT", "1m", limit=100)
        
        self.assertEqual(result, mock_ohlcv)
        self.exchange_manager.exchange.fetch_ohlcv.assert_called_once_with("BTC/USDT", "1m", limit=100)
    
    def test_safe_fetch_ohlcv_retry(self):
        """测试安全获取OHLCV数据重试机制"""
        # 模拟第一次失败，第二次成功
        mock_ohlcv = [
            [1609459200000, 100.0, 110.0, 95.0, 105.0, 1000]
        ]
        
        self.exchange_manager.exchange.fetch_ohlcv = Mock(
            side_effect=[Exception("网络错误"), mock_ohlcv]
        )
        
        result = self.exchange_manager.safe_fetch_ohlcv("BTC/USDT", "1m", limit=100)
        
        self.assertEqual(result, mock_ohlcv)
        self.assertEqual(self.exchange_manager.exchange.fetch_ohlcv.call_count, 2)
    
    def test_safe_fetch_ticker(self):
        """测试安全获取行情数据"""
        # 模拟行情数据
        mock_ticker = {
            "symbol": "BTC/USDT",
            "last": 50000.0,
            "bid": 49999.0,
            "ask": 50001.0,
            "baseVolume": 1000.0,
            "quoteVolume": 50000000.0
        }
        
        self.exchange_manager.exchange.fetch_ticker = Mock(return_value=mock_ticker)
        
        result = self.exchange_manager.safe_fetch_ticker("BTC/USDT")
        
        self.assertEqual(result, mock_ticker)
        self.exchange_manager.exchange.fetch_ticker.assert_called_once_with("BTC/USDT")
    
    def test_safe_create_order(self):
        """测试安全创建订单"""
        # 模拟订单数据
        mock_order = {
            "id": "123456",
            "symbol": "BTC/USDT",
            "type": "market",
            "side": "buy",
            "amount": 0.001,
            "price": 50000.0,
            "status": "open"
        }
        
        self.exchange_manager.exchange.create_order = Mock(return_value=mock_order)
        
        result = self.exchange_manager.safe_create_order("BTC/USDT", "market", "buy", 0.001)
        
        self.assertEqual(result, mock_order)
        self.exchange_manager.exchange.create_order.assert_called_once_with(
            "BTC/USDT", "market", "buy", 0.001
        )
    
    def test_get_multiple_timeframes_data(self):
        """测试获取多时间框架数据"""
        # 模拟OHLCV数据
        mock_ohlcv_1m = [
            [1609459200000, 100.0, 110.0, 95.0, 105.0, 1000]
        ]
        mock_ohlcv_5m = [
            [1609459200000, 100.0, 110.0, 95.0, 105.0, 5000]
        ]
        
        self.exchange_manager.safe_fetch_ohlcv = Mock(
            side_effect=[mock_ohlcv_1m, mock_ohlcv_5m]
        )
        
        result = self.exchange_manager.get_multiple_timeframes_data("BTC/USDT", ["1m", "5m"])
        
        self.assertIn("1m", result)
        self.assertIn("5m", result)
        self.assertEqual(result["1m"], mock_ohlcv_1m)
        self.assertEqual(result["5m"], mock_ohlcv_5m)
    
    def test_get_real_market_price(self):
        """测试获取真实市场价格"""
        # 模拟订单簿数据
        mock_orderbook = {
            "bids": [[49999.0, 0.1], [49998.0, 0.2]],
            "asks": [[50001.0, 0.1], [50002.0, 0.2]],
            "timestamp": 1609459200000
        }
        
        self.exchange_manager.safe_fetch_order_book = Mock(return_value=mock_orderbook)
        
        result = self.exchange_manager.get_real_market_price("BTC/USDT")
        
        # 应该返回中间价
        self.assertEqual(result, 50000.0)
    
    def test_get_balance(self):
        """测试获取账户余额"""
        # 模拟余额数据
        mock_balance = {
            "USDT": {"free": 1000.0, "used": 0.0, "total": 1000.0},
            "BTC": {"free": 0.01, "used": 0.0, "total": 0.01}
        }
        
        self.exchange_manager.exchange.fetch_balance = Mock(return_value=mock_balance)
        
        result = self.exchange_manager.get_balance()
        
        self.assertEqual(result, mock_balance)
        self.exchange_manager.exchange.fetch_balance.assert_called_once()


if __name__ == "__main__":
    unittest.main()