"""
SMC分析模块测试
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
from smc_analyzer import SMCDetector, MTFAnalyzer


class TestSMCDetector(unittest.TestCase):
    """SMC检测器测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建模拟配置
        self.config = Mock(spec=Config)
        self.config.bos_choch_sensitivity = 0.7
        self.config.order_block_min_strength = 0.6
        self.config.fvg_min_strength = 0.5
        
        # 创建模拟日志器
        self.logger = Mock()
        
        # 创建SMC检测器实例
        self.smc_detector = SMCDetector(self.config, self.logger)
        
        # 创建测试数据
        self.create_test_data()
    
    def create_test_data(self):
        """创建测试数据"""
        # 创建看涨BOS模式的测试数据
        timestamps = pd.date_range(start="2023-01-01", periods=100, freq="1H")
        
        # 创建价格数据，包含看涨BOS模式
        closes = np.ones(100) * 100
        highs = closes + np.random.rand(100) * 2
        lows = closes - np.random.rand(100) * 2
        opens = closes + np.random.randn(100) * 0.5
        volumes = np.random.randint(1000, 5000, size=100)
        
        # 在特定位置创建看涨BOS模式
        # 第30根K线突破前高
        closes[25:30] = [100, 101, 102, 103, 104]  # 上涨趋势
        highs[25:30] = [101, 102, 103, 104, 105]
        closes[30:35] = [104, 103, 102, 105, 106]  # 回调后突破
        
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
    
    def test_detect_bos_choch(self):
        """测试BOS/CHOCH检测"""
        current_price = 105.0
        
        result = self.smc_detector.detect_bos_choch(self.test_data, current_price)
        
        self.assertIn("bos_choch", result)
        self.assertIn("signal", result["bos_choch"])
        self.assertIn("confidence", result["bos_choch"])
        self.assertIn("strength", result["bos_choch"])
    
    def test_detect_order_blocks(self):
        """测试订单块检测"""
        current_price = 105.0
        
        result = self.smc_detector.detect_order_blocks(self.test_data, current_price)
        
        self.assertIn("order_blocks", result)
        self.assertIn("bullish", result["order_blocks"])
        self.assertIn("bearish", result["order_blocks"])
        self.assertIsInstance(result["order_blocks"]["bullish"], list)
        self.assertIsInstance(result["order_blocks"]["bearish"], list)
    
    def test_detect_fvg(self):
        """测试公平价值缺口(FVG)检测"""
        current_price = 105.0
        
        result = self.smc_detector.detect_fvg(self.test_data, current_price)
        
        self.assertIn("fvg", result)
        self.assertIn("bullish", result["fvg"])
        self.assertIn("bearish", result["fvg"])
        self.assertIsInstance(result["fvg"]["bullish"], list)
        self.assertIsInstance(result["fvg"]["bearish"], list)
    
    def test_detect_swing_points(self):
        """测试摆动点检测"""
        current_price = 105.0
        
        result = self.smc_detector.detect_swing_points(self.test_data, current_price)
        
        self.assertIn("swing_points", result)
        self.assertIn("highs", result["swing_points"])
        self.assertIn("lows", result["swing_points"])
        self.assertIsInstance(result["swing_points"]["highs"], list)
        self.assertIsInstance(result["swing_points"]["lows"], list)
    
    def test_calculate_structure_strength(self):
        """测试结构强度计算"""
        # 创建测试结构
        structure = {
            "type": "bullish_bos",
            "strength": 0.8,
            "volume_confirmation": True,
            "price_rejection": False
        }
        
        result = self.smc_detector.calculate_structure_strength(structure)
        
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)
    
    def test_detect_all_structures(self):
        """测试检测所有结构"""
        current_price = 105.0
        
        result = self.smc_detector.detect_all_structures(self.test_data, current_price)
        
        self.assertIn("bos_choch", result)
        self.assertIn("order_blocks", result)
        self.assertIn("fvg", result)
        self.assertIn("swing_points", result)
        self.assertIn("overall_score", result)


class TestMTFAnalyzer(unittest.TestCase):
    """多时间框架分析器测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建模拟配置
        self.config = Mock(spec=Config)
        self.config.mtf_weight = {
            "15m": 0.1,
            "1h": 0.3,
            "4h": 0.3,
            "1d": 0.3
        }
        
        # 创建模拟日志器
        self.logger = Mock()
        
        # 创建MTF分析器实例
        self.mtf_analyzer = MTFAnalyzer(self.config, self.logger)
        
        # 创建测试数据
        self.create_test_data()
    
    def create_test_data(self):
        """创建测试数据"""
        self.test_data = {}
        
        # 创建不同时间框架的测试数据
        for tf in ["15m", "1h", "4h", "1d"]:
            periods = 100 if tf != "1d" else 30
            timestamps = pd.date_range(start="2023-01-01", periods=periods, freq="1H")
            
            closes = np.ones(periods) * 100
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
    
    def test_analyze_multiple_timeframes(self):
        """测试多时间框架分析"""
        current_price = 105.0
        
        result = self.mtf_analyzer.analyze_multiple_timeframes(self.test_data, current_price)
        
        self.assertIn("mtf_signals", result)
        self.assertIn("consensus", result)
        self.assertIn("overall_score", result)
        
        # 检查每个时间框架的信号
        for tf in ["15m", "1h", "4h", "1d"]:
            if tf in result["mtf_signals"]:
                self.assertIn("signal", result["mtf_signals"][tf])
                self.assertIn("strength", result["mtf_signals"][tf])
    
    def test_calculate_mtf_consensus(self):
        """测试多时间框架共识计算"""
        # 创建测试信号
        mtf_signals = {
            "15m": {"signal": "BUY", "strength": 0.6},
            "1h": {"signal": "BUY", "strength": 0.7},
            "4h": {"signal": "HOLD", "strength": 0.5},
            "1d": {"signal": "BUY", "strength": 0.8}
        }
        
        result = self.mtf_analyzer.calculate_mtf_consensus(mtf_signals)
        
        self.assertIn("consensus", result)
        self.assertIn("confidence", result)
        self.assertIn("weighted_score", result)
    
    def test_calculate_timeframe_weight(self):
        """测试时间框架权重计算"""
        # 测试不同时间框架的权重
        tf_15m_weight = self.mtf_analyzer.calculate_timeframe_weight("15m")
        tf_1h_weight = self.mtf_analyzer.calculate_timeframe_weight("1h")
        tf_4h_weight = self.mtf_analyzer.calculate_timeframe_weight("4h")
        tf_1d_weight = self.mtf_analyzer.calculate_timeframe_weight("1d")
        
        self.assertEqual(tf_15m_weight, 0.1)
        self.assertEqual(tf_1h_weight, 0.3)
        self.assertEqual(tf_4h_weight, 0.3)
        self.assertEqual(tf_1d_weight, 0.3)
    
    def test_validate_mtf_signal(self):
        """测试多时间框架信号验证"""
        # 创建测试信号
        mtf_signals = {
            "15m": {"signal": "BUY", "strength": 0.6},
            "1h": {"signal": "BUY", "strength": 0.7},
            "4h": {"signal": "HOLD", "strength": 0.5},
            "1d": {"signal": "BUY", "strength": 0.8}
        }
        
        result = self.mtf_analyzer.validate_mtf_signal(mtf_signals)
        
        self.assertIn("valid", result)
        self.assertIn("reason", result)
        self.assertIn("confidence", result)


if __name__ == "__main__":
    unittest.main()