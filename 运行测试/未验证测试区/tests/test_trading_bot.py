"""
交易机器人模块测试
"""

import unittest
import os
import sys
from unittest.mock import Mock, patch, MagicMock
import asyncio

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from trading_bot import TradingBot


class TestTradingBot(unittest.TestCase):
    """交易机器人测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建模拟配置
        self.config = Mock(spec=Config)
        self.config.symbol = "PAXGUSDT"
        self.config.timeframes = ["1h", "4h", "1d"]
        self.config.analysis_interval = 60
        self.config.monitoring_interval = 30
        self.config.trading_enabled = True
        self.config.paper_trading = True
        
        # 创建模拟日志器
        self.logger = Mock()
        
        # 创建模拟组件
        self.exchange_manager = Mock()
        self.smc_analyzer = Mock()
        self.technical_analyzer = Mock()
        self.key_levels_calculator = Mock()
        self.price_action_analyzer = Mock()
        self.ai_signal_generator = Mock()
        self.risk_manager = Mock()
        self.position_manager = Mock()
        self.trading_executor = Mock()
        
        # 创建交易机器人实例
        self.trading_bot = TradingBot(self.config, self.logger)
        
        # 替换模拟组件
        self.trading_bot.exchange_manager = self.exchange_manager
        self.trading_bot.smc_analyzer = self.smc_analyzer
        self.trading_bot.technical_analyzer = self.technical_analyzer
        self.trading_bot.key_levels_calculator = self.key_levels_calculator
        self.trading_bot.price_action_analyzer = self.price_action_analyzer
        self.trading_bot.ai_signal_generator = self.ai_signal_generator
        self.trading_bot.risk_manager = self.risk_manager
        self.trading_bot.position_manager = self.position_manager
        self.trading_bot.trading_executor = self.trading_executor
        
        # 创建测试数据
        self.create_test_data()
    
    def create_test_data(self):
        """创建测试数据"""
        self.test_market_data = {
            "1h": {
                "timestamp": ["2023-01-01T10:00:00Z"],
                "open": [1850.0],
                "high": [1852.0],
                "low": [1848.0],
                "close": [1851.0],
                "volume": [1000]
            },
            "4h": {
                "timestamp": ["2023-01-01T08:00:00Z"],
                "open": [1845.0],
                "high": [1855.0],
                "low": [1840.0],
                "close": [1850.0],
                "volume": [4000]
            },
            "1d": {
                "timestamp": ["2023-01-01T00:00:00Z"],
                "open": [1840.0],
                "high": [1860.0],
                "low": [1830.0],
                "close": [1850.0],
                "volume": [10000]
            }
        }
        
        self.test_smc_analysis = {
            "structure": "Bullish",
            "bos_choch": [{"type": "BOS", "direction": "bullish", "price": 1850.0}],
            "order_blocks": [{"price": 1845.0, "type": "support", "strength": 0.8}],
            "fvg": [{"top": 1848.0, "bottom": 1852.0, "type": "bullish", "strength": 0.7}],
            "swing_points": [{"price": 1840.0, "type": "low", "strength": 0.9}],
            "structure_strength": 0.75
        }
        
        self.test_technical_analysis = {
            "ema": {"signal": "BUY", "strength": 0.7},
            "rsi": {"signal": "BUY", "strength": 0.6},
            "macd": {"signal": "BUY", "strength": 0.8},
            "bollinger_bands": {"signal": "HOLD", "strength": 0.5},
            "overall_score": 0.7
        }
        
        self.test_key_levels = {
            "support": [1845.0, 1830.0],
            "resistance": [1860.0, 1875.0],
            "pivot_points": {
                "pivot": 1850.0,
                "support1": 1845.0,
                "support2": 1830.0,
                "resistance1": 1860.0,
                "resistance2": 1875.0
            },
            "vwap": 1848.0
        }
        
        self.test_price_action = {
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
            },
            "overall_score": 0.7
        }
        
        self.test_ai_signal = {
            "signal": "BUY",
            "confidence": 0.8,
            "reasoning": "Strong bullish momentum with support at key level",
            "provider": "Consensus",
            "consensus": {
                "agreement": 2,
                "total_providers": 2,
                "consensus_ratio": 1.0
            }
        }
        
        self.test_combined_signal = {
            "signal": "BUY",
            "confidence": 0.75,
            "components": {
                "smc": {"signal": "BUY", "weight": 0.3, "score": 0.75},
                "technical": {"signal": "BUY", "weight": 0.3, "score": 0.7},
                "price_action": {"signal": "BUY", "weight": 0.2, "score": 0.7},
                "ai": {"signal": "BUY", "weight": 0.2, "score": 0.8}
            },
            "reasoning": "Combined bullish signal from all components"
        }
    
    def test_initialize(self):
        """测试初始化"""
        # 调用方法
        result = self.trading_bot.initialize()
        
        # 验证结果
        self.assertTrue(result)
        self.assertTrue(self.trading_bot.initialized)
    
    def test_start(self):
        """测试启动"""
        # 模拟异步循环
        with patch('asyncio.create_task') as mock_create_task:
            # 设置已初始化状态
            self.trading_bot.initialized = True
            
            # 调用方法
            result = self.trading_bot.start()
            
            # 验证结果
            self.assertTrue(result)
            self.assertTrue(self.trading_bot.running)
            
            # 验证异步任务创建
            mock_create_task.assert_called()
    
    def test_start_not_initialized(self):
        """测试未初始化时启动"""
        # 设置未初始化状态
        self.trading_bot.initialized = False
        
        # 调用方法
        result = self.trading_bot.start()
        
        # 验证结果
        self.assertFalse(result)
        self.assertFalse(self.trading_bot.running)
    
    def test_stop(self):
        """测试停止"""
        # 设置运行状态
        self.trading_bot.running = True
        
        # 调用方法
        result = self.trading_bot.stop()
        
        # 验证结果
        self.assertTrue(result)
        self.assertFalse(self.trading_bot.running)
    
    def test_pause(self):
        """测试暂停"""
        # 设置运行状态
        self.trading_bot.running = True
        self.trading_bot.paused = False
        
        # 调用方法
        result = self.trading_bot.pause()
        
        # 验证结果
        self.assertTrue(result)
        self.assertTrue(self.trading_bot.paused)
    
    def test_resume(self):
        """测试恢复"""
        # 设置暂停状态
        self.trading_bot.running = True
        self.trading_bot.paused = True
        
        # 调用方法
        result = self.trading_bot.resume()
        
        # 验证结果
        self.assertTrue(result)
        self.assertFalse(self.trading_bot.paused)
    
    def test_get_market_data(self):
        """测试获取市场数据"""
        # 模拟交易所管理器返回
        self.exchange_manager.get_market_data.return_value = self.test_market_data
        
        # 调用方法
        result = self.trading_bot.get_market_data()
        
        # 验证结果
        self.assertEqual(result, self.test_market_data)
        
        # 验证交易所管理器调用
        self.exchange_manager.get_market_data.assert_called_once()
    
    def test_analyze_market(self):
        """测试市场分析"""
        # 模拟各分析器返回
        self.smc_analyzer.analyze_structure.return_value = self.test_smc_analysis
        self.technical_analyzer.calculate_all_indicators.return_value = self.test_technical_analysis
        self.key_levels_calculator.calculate_key_levels.return_value = self.test_key_levels
        self.price_action_analyzer.analyze_price_action.return_value = self.test_price_action
        
        # 调用方法
        result = self.trading_bot.analyze_market(self.test_market_data)
        
        # 验证结果
        self.assertIsInstance(result, dict)
        self.assertIn("smc", result)
        self.assertIn("technical", result)
        self.assertIn("key_levels", result)
        self.assertIn("price_action", result)
        
        # 验证各分析器调用
        self.smc_analyzer.analyze_structure.assert_called_once()
        self.technical_analyzer.calculate_all_indicators.assert_called_once()
        self.key_levels_calculator.calculate_key_levels.assert_called_once()
        self.price_action_analyzer.analyze_price_action.assert_called_once()
    
    def test_generate_signal(self):
        """测试生成信号"""
        # 创建市场分析结果
        market_analysis = {
            "smc": self.test_smc_analysis,
            "technical": self.test_technical_analysis,
            "key_levels": self.test_key_levels,
            "price_action": self.test_price_action
        }
        
        # 模拟AI信号生成器返回
        self.ai_signal_generator.generate_signal.return_value = self.test_ai_signal
        
        # 调用方法
        result = self.trading_bot.generate_signal(market_analysis)
        
        # 验证结果
        self.assertIsInstance(result, dict)
        self.assertIn("signal", result)
        self.assertIn("confidence", result)
        self.assertIn("components", result)
        self.assertIn("reasoning", result)
        
        # 验证AI信号生成器调用
        self.ai_signal_generator.generate_signal.assert_called_once()
    
    def test_execute_trade(self):
        """测试执行交易"""
        # 模拟交易执行器返回
        self.trading_executor.execute_signal.return_value = True
        
        # 调用方法
        result = self.trading_bot.execute_trade(self.test_combined_signal)
        
        # 验证结果
        self.assertTrue(result)
        
        # 验证交易执行器调用
        self.trading_executor.execute_signal.assert_called_once_with(self.test_combined_signal)
    
    def test_monitor_positions(self):
        """测试监控持仓"""
        # 模拟持仓管理器返回
        self.position_manager.get_open_positions.return_value = [
            {
                "id": "pos_12345",
                "symbol": "PAXGUSDT",
                "side": "BUY",
                "size": 0.1,
                "entry_price": 1850.5,
                "status": "open"
            }
        ]
        
        # 模拟交易执行器返回
        self.trading_executor.check_positions.return_value = None
        
        # 调用方法
        self.trading_bot.monitor_positions()
        
        # 验证持仓管理器调用
        self.position_manager.get_open_positions.assert_called_once()
        
        # 验证交易执行器调用
        self.trading_executor.check_positions.assert_called_once()
    
    def test_get_status(self):
        """测试获取状态"""
        # 模拟各组件返回
        self.exchange_manager.get_status.return_value = {"connected": True}
        self.position_manager.get_open_positions.return_value = [
            {"id": "pos_12345", "symbol": "PAXGUSDT", "status": "open"}
        ]
        self.trading_executor.get_status.return_value = {
            "running": True,
            "orders": {"total": 1, "open": 0, "filled": 1},
            "positions": {"total": 1, "open": 1}
        }
        
        # 调用方法
        result = self.trading_bot.get_status()
        
        # 验证结果
        self.assertIsInstance(result, dict)
        self.assertIn("running", result)
        self.assertIn("paused", result)
        self.assertIn("initialized", result)
        self.assertIn("exchange", result)
        self.assertIn("positions", result)
        self.assertIn("executor", result)
        
        # 验证各组件调用
        self.exchange_manager.get_status.assert_called_once()
        self.position_manager.get_open_positions.assert_called_once()
        self.trading_executor.get_status.assert_called_once()
    
    def test_run_analysis_cycle(self):
        """测试运行分析周期"""
        # 模拟交易所管理器返回
        self.exchange_manager.get_market_data.return_value = self.test_market_data
        
        # 模拟各分析器返回
        self.smc_analyzer.analyze_structure.return_value = self.test_smc_analysis
        self.technical_analyzer.calculate_all_indicators.return_value = self.test_technical_analysis
        self.key_levels_calculator.calculate_key_levels.return_value = self.test_key_levels
        self.price_action_analyzer.analyze_price_action.return_value = self.test_price_action
        
        # 模拟AI信号生成器返回
        self.ai_signal_generator.generate_signal.return_value = self.test_ai_signal
        
        # 模拟交易执行器返回
        self.trading_executor.execute_signal.return_value = True
        
        # 调用方法
        self.trading_bot.run_analysis_cycle()
        
        # 验证交易所管理器调用
        self.exchange_manager.get_market_data.assert_called_once()
        
        # 验证各分析器调用
        self.smc_analyzer.analyze_structure.assert_called_once()
        self.technical_analyzer.calculate_all_indicators.assert_called_once()
        self.key_levels_calculator.calculate_key_levels.assert_called_once()
        self.price_action_analyzer.analyze_price_action.assert_called_once()
        
        # 验证AI信号生成器调用
        self.ai_signal_generator.generate_signal.assert_called_once()
        
        # 验证交易执行器调用
        self.trading_executor.execute_signal.assert_called_once()
    
    def test_run_analysis_cycle_paused(self):
        """测试暂停状态下运行分析周期"""
        # 设置暂停状态
        self.trading_bot.paused = True
        
        # 调用方法
        self.trading_bot.run_analysis_cycle()
        
        # 验证交易所管理器未被调用
        self.exchange_manager.get_market_data.assert_not_called()
        
        # 验证各分析器未被调用
        self.smc_analyzer.analyze_structure.assert_not_called()
        self.technical_analyzer.calculate_all_indicators.assert_not_called()
        self.key_levels_calculator.calculate_key_levels.assert_not_called()
        self.price_action_analyzer.analyze_price_action.assert_not_called()
        
        # 验证AI信号生成器未被调用
        self.ai_signal_generator.generate_signal.assert_not_called()
        
        # 验证交易执行器未被调用
        self.trading_executor.execute_signal.assert_not_called()
    
    def test_run_monitoring_cycle(self):
        """测试运行监控周期"""
        # 模拟持仓管理器返回
        self.position_manager.get_open_positions.return_value = [
            {
                "id": "pos_12345",
                "symbol": "PAXGUSDT",
                "side": "BUY",
                "size": 0.1,
                "entry_price": 1850.5,
                "status": "open"
            }
        ]
        
        # 模拟交易执行器返回
        self.trading_executor.check_positions.return_value = None
        
        # 调用方法
        self.trading_bot.run_monitoring_cycle()
        
        # 验证持仓管理器调用
        self.position_manager.get_open_positions.assert_called_once()
        
        # 验证交易执行器调用
        self.trading_executor.check_positions.assert_called_once()
    
    def test_run_monitoring_cycle_paused(self):
        """测试暂停状态下运行监控周期"""
        # 设置暂停状态
        self.trading_bot.paused = True
        
        # 调用方法
        self.trading_bot.run_monitoring_cycle()
        
        # 验证持仓管理器未被调用
        self.position_manager.get_open_positions.assert_not_called()
        
        # 验证交易执行器未被调用
        self.trading_executor.check_positions.assert_not_called()


if __name__ == "__main__":
    unittest.main()