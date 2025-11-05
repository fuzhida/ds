"""
技术指标分析模块测试
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
from technical_analyzer import TechnicalIndicatorCalculator, KeyLevelsCalculator, PriceActionAnalyzer


class TestTechnicalIndicatorCalculator(unittest.TestCase):
    """技术指标计算器测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建模拟配置
        self.config = Mock(spec=Config)
        
        # 创建模拟日志器
        self.logger = Mock()
        
        # 创建技术指标计算器实例
        self.technical_calculator = TechnicalIndicatorCalculator(self.config, self.logger)
        
        # 创建测试数据
        self.create_test_data()
    
    def create_test_data(self):
        """创建测试数据"""
        # 创建测试数据
        timestamps = pd.date_range(start="2023-01-01", periods=200, freq="1H")
        
        # 创建价格数据，包含趋势
        closes = 100 + np.cumsum(np.random.randn(200) * 0.5)
        highs = closes + np.random.rand(200) * 2
        lows = closes - np.random.rand(200) * 2
        opens = closes + np.random.randn(200) * 0.5
        volumes = np.random.randint(1000, 5000, size=200)
        
        self.test_data = {
            "1h": pd.DataFrame({
                "timestamp": timestamps,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes
            })
        }
    
    def test_calculate_ema(self):
        """测试EMA计算"""
        df = self.test_data["1h"]
        period = 20
        
        result = self.technical_calculator.calculate_ema(df, period)
        
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(df))
        self.assertTrue(result.notna().sum() > 0)  # 确保有非NaN值
    
    def test_calculate_rsi(self):
        """测试RSI计算"""
        df = self.test_data["1h"]
        period = 14
        
        result = self.technical_calculator.calculate_rsi(df, period)
        
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(df))
        self.assertTrue(result.notna().sum() > 0)  # 确保有非NaN值
        self.assertTrue((result >= 0).all() and (result <= 100).all())  # RSI应在0-100之间
    
    def test_calculate_atr(self):
        """测试ATR计算"""
        df = self.test_data["1h"]
        period = 14
        
        result = self.technical_calculator.calculate_atr(df, period)
        
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(df))
        self.assertTrue(result.notna().sum() > 0)  # 确保有非NaN值
        self.assertTrue((result >= 0).all())  # ATR应为非负数
    
    def test_calculate_macd(self):
        """测试MACD计算"""
        df = self.test_data["1h"]
        fast_period = 12
        slow_period = 26
        signal_period = 9
        
        result = self.technical_calculator.calculate_macd(df, fast_period, slow_period, signal_period)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("macd", result)
        self.assertIn("signal", result)
        self.assertIn("histogram", result)
        self.assertEqual(len(result), len(df))
    
    def test_calculate_bollinger_bands(self):
        """测试布林带计算"""
        df = self.test_data["1h"]
        period = 20
        std_dev = 2
        
        result = self.technical_calculator.calculate_bollinger_bands(df, period, std_dev)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("upper", result)
        self.assertIn("middle", result)
        self.assertIn("lower", result)
        self.assertEqual(len(result), len(df))
    
    def test_calculate_all_indicators(self):
        """测试计算所有指标"""
        result = self.technical_calculator.calculate_all_indicators(self.test_data)
        
        self.assertIsInstance(result, dict)
        self.assertIn("1h", result)
        
        # 检查各种指标是否存在
        indicators_1h = result["1h"]
        self.assertIn("ema", indicators_1h)
        self.assertIn("rsi", indicators_1h)
        self.assertIn("atr", indicators_1h)
        self.assertIn("macd", indicators_1h)
        self.assertIn("bollinger_bands", indicators_1h)
        self.assertIn("overall_score", indicators_1h)
    
    def test_calculate_technical_score(self):
        """测试技术评分计算"""
        # 创建测试指标
        indicators = {
            "ema": {"signal": "BUY", "strength": 0.7},
            "rsi": {"signal": "BUY", "strength": 0.6},
            "macd": {"signal": "BUY", "strength": 0.8},
            "bollinger_bands": {"signal": "HOLD", "strength": 0.5}
        }
        
        result = self.technical_calculator.calculate_technical_score(indicators)
        
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)


class TestKeyLevelsCalculator(unittest.TestCase):
    """关键水平计算器测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建模拟配置
        self.config = Mock(spec=Config)
        
        # 创建模拟日志器
        self.logger = Mock()
        
        # 创建关键水平计算器实例
        self.key_levels_calculator = KeyLevelsCalculator(self.config, self.logger)
        
        # 创建测试数据
        self.create_test_data()
    
    def create_test_data(self):
        """创建测试数据"""
        self.test_data = {}
        
        # 创建不同时间框架的测试数据
        for tf in ["15m", "1h", "4h", "1d"]:
            periods = 100 if tf != "1d" else 30
            timestamps = pd.date_range(start="2023-01-01", periods=periods, freq="1H")
            
            closes = 100 + np.cumsum(np.random.randn(periods) * 0.5)
            highs = closes + np.random.rand(periods) * 2
            lows = closes - np.random.rand(periods) * 2
            opens = closes + np.random.randn(periods) * 0.5
            volumes = np.random.randint(1000, 5000, size=periods)
            
            self.test_data[tf] = pd.DataFrame({
                "timestamp": timestamps,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes
            })
    
    def test_calculate_key_levels(self):
        """测试关键水平计算"""
        current_price = 105.0
        
        result = self.key_levels_calculator.calculate_key_levels(self.test_data, current_price)
        
        self.assertIn("support", result)
        self.assertIn("resistance", result)
        self.assertIn("pivot_points", result)
        self.assertIn("vwap", result)
        
        self.assertIsInstance(result["support"], list)
        self.assertIsInstance(result["resistance"], list)
    
    def test_calculate_support_resistance(self):
        """测试支撑阻力计算"""
        df = self.test_data["1h"]
        
        result = self.key_levels_calculator.calculate_support_resistance(df)
        
        self.assertIn("support", result)
        self.assertIn("resistance", result)
        self.assertIsInstance(result["support"], list)
        self.assertIsInstance(result["resistance"], list)
    
    def test_calculate_pivot_points(self):
        """测试枢轴点计算"""
        df = self.test_data["1h"]
        
        result = self.key_levels_calculator.calculate_pivot_points(df)
        
        self.assertIn("pivot", result)
        self.assertIn("support1", result)
        self.assertIn("support2", result)
        self.assertIn("resistance1", result)
        self.assertIn("resistance2", result)
    
    def test_calculate_vwap(self):
        """测试VWAP计算"""
        df = self.test_data["1h"]
        
        result = self.key_levels_calculator.calculate_vwap(df)
        
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(df))
    
    def test_calculate_fibonacci_levels(self):
        """测试斐波那契水平计算"""
        df = self.test_data["1h"]
        
        # 获取最高价和最低价
        high_price = df["high"].max()
        low_price = df["low"].min()
        
        result = self.key_levels_calculator.calculate_fibonacci_levels(high_price, low_price)
        
        self.assertIsInstance(result, dict)
        self.assertIn("0.0%", result)
        self.assertIn("23.6%", result)
        self.assertIn("38.2%", result)
        self.assertIn("50.0%", result)
        self.assertIn("61.8%", result)
        self.assertIn("78.6%", result)
        self.assertIn("100.0%", result)


class TestPriceActionAnalyzer(unittest.TestCase):
    """价格行为分析器测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建模拟配置
        self.config = Mock(spec=Config)
        
        # 创建模拟日志器
        self.logger = Mock()
        
        # 创建价格行为分析器实例
        self.price_action_analyzer = PriceActionAnalyzer(self.config, self.logger)
        
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
        
        self.test_data = {
            "1h": pd.DataFrame({
                "timestamp": timestamps,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes
            })
        }
    
    def test_analyze_price_action(self):
        """测试价格行为分析"""
        current_price = 105.0
        
        result = self.price_action_analyzer.analyze_price_action(self.test_data, current_price)
        
        self.assertIn("candlestick_patterns", result)
        self.assertIn("trend", result)
        self.assertIn("volatility", result)
        self.assertIn("momentum", result)
        self.assertIn("overall_score", result)
    
    def test_detect_candlestick_patterns(self):
        """测试蜡烛图模式检测"""
        df = self.test_data["1h"]
        
        result = self.price_action_analyzer.detect_candlestick_patterns(df)
        
        self.assertIsInstance(result, dict)
        self.assertIn("doji", result)
        self.assertIn("hammer", result)
        self.assertIn("engulfing", result)
        self.assertIn("morning_star", result)
        self.assertIn("evening_star", result)
    
    def test_analyze_trend(self):
        """测试趋势分析"""
        df = self.test_data["1h"]
        
        result = self.price_action_analyzer.analyze_trend(df)
        
        self.assertIn("direction", result)
        self.assertIn("strength", result)
        self.assertIn("duration", result)
        self.assertIn("direction", result["direction"])
        self.assertIn("strength", result["strength"])
    
    def test_calculate_volatility(self):
        """测试波动率计算"""
        df = self.test_data["1h"]
        
        result = self.price_action_analyzer.calculate_volatility(df)
        
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
    
    def test_calculate_momentum(self):
        """测试动量计算"""
        df = self.test_data["1h"]
        
        result = self.price_action_analyzer.calculate_momentum(df)
        
        self.assertIsInstance(result, dict)
        self.assertIn("roc", result)
        self.assertIn("rsi", result)
        self.assertIn("stochastic", result)
    
    def test_calculate_price_action_score(self):
        """测试价格行为评分计算"""
        # 创建测试价格行为数据
        price_action = {
            "candlestick_patterns": {
                "doji": 0.2,
                "hammer": 0.3,
                "engulfing": 0.7
            },
            "trend": {
                "direction": "bullish",
                "strength": 0.8
            },
            "volatility": 0.4,
            "momentum": {
                "roc": 0.6,
                "rsi": 0.7,
                "stochastic": 0.5
            }
        }
        
        result = self.price_action_analyzer.calculate_price_action_score(price_action)
        
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)


if __name__ == "__main__":
    unittest.main()