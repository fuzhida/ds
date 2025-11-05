"""
AI信号生成模块测试
"""

import unittest
import os
import sys
from unittest.mock import Mock, patch, MagicMock
import json

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from ai_signal_generator import (
    AISignalProvider, 
    DeepSeekSignalProvider, 
    OpenAISignalProvider,
    AISignalGenerator
)


class TestDeepSeekSignalProvider(unittest.TestCase):
    """DeepSeek信号提供者测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建模拟配置
        self.config = Mock(spec=Config)
        self.config.deepseek_api_key = "test_deepseek_key"
        self.config.deepseek_model = "deepseek-chat"
        
        # 创建模拟日志器
        self.logger = Mock()
        
        # 创建DeepSeek信号提供者实例
        self.signal_provider = DeepSeekSignalProvider(self.config, self.logger)
    
    @patch('requests.post')
    def test_generate_signal_success(self, mock_post):
        """测试成功生成信号"""
        # 模拟API响应
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "signal": "BUY",
                        "confidence": 0.85,
                        "reasoning": "Strong bullish momentum with support at key level"
                    })
                }
            }]
        }
        mock_post.return_value = mock_response
        
        # 创建测试数据
        market_data = {
            "symbol": "PAXGUSDT",
            "price": 1850.5,
            "timeframe": "1h",
            "indicators": {
                "rsi": 35,
                "ema": {
                    "signal": "BUY",
                    "strength": 0.7
                },
                "macd": {
                    "signal": "BUY",
                    "strength": 0.8
                }
            },
            "smc": {
                "structure": "Bullish",
                "order_blocks": [{"price": 1845.0, "type": "support"}],
                "fvg": [{"top": 1848.0, "bottom": 1852.0}]
            },
            "key_levels": {
                "support": [1845.0, 1830.0],
                "resistance": [1860.0, 1875.0]
            }
        }
        
        # 调用方法
        result = self.signal_provider.generate_signal(market_data)
        
        # 验证结果
        self.assertEqual(result["signal"], "BUY")
        self.assertEqual(result["confidence"], 0.85)
        self.assertIn("reasoning", result)
        self.assertEqual(result["provider"], "DeepSeek")
        
        # 验证API调用
        mock_post.assert_called_once()
    
    @patch('requests.post')
    def test_generate_signal_api_error(self, mock_post):
        """测试API错误情况"""
        # 模拟API错误响应
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response
        
        # 创建测试数据
        market_data = {
            "symbol": "PAXGUSDT",
            "price": 1850.5,
            "timeframe": "1h"
        }
        
        # 调用方法
        result = self.signal_provider.generate_signal(market_data)
        
        # 验证结果
        self.assertEqual(result["signal"], "HOLD")
        self.assertEqual(result["confidence"], 0.0)
        self.assertIn("API Error", result["reasoning"])
        self.assertEqual(result["provider"], "DeepSeek")
    
    def test_generate_signal_missing_api_key(self):
        """测试缺少API密钥情况"""
        # 创建没有API密钥的配置
        config_no_key = Mock(spec=Config)
        config_no_key.deepseek_api_key = None
        
        # 创建信号提供者
        provider_no_key = DeepSeekSignalProvider(config_no_key, self.logger)
        
        # 创建测试数据
        market_data = {
            "symbol": "PAXGUSDT",
            "price": 1850.5,
            "timeframe": "1h"
        }
        
        # 调用方法
        result = provider_no_key.generate_signal(market_data)
        
        # 验证结果
        self.assertEqual(result["signal"], "HOLD")
        self.assertEqual(result["confidence"], 0.0)
        self.assertIn("API key not configured", result["reasoning"])
        self.assertEqual(result["provider"], "DeepSeek")


class TestOpenAISignalProvider(unittest.TestCase):
    """OpenAI信号提供者测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建模拟配置
        self.config = Mock(spec=Config)
        self.config.openai_api_key = "test_openai_key"
        self.config.openai_model = "gpt-4"
        
        # 创建模拟日志器
        self.logger = Mock()
        
        # 创建OpenAI信号提供者实例
        self.signal_provider = OpenAISignalProvider(self.config, self.logger)
    
    @patch('openai.ChatCompletion.create')
    def test_generate_signal_success(self, mock_create):
        """测试成功生成信号"""
        # 模拟API响应
        mock_response = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "signal": "SELL",
                        "confidence": 0.75,
                        "reasoning": "Resistance level tested with bearish divergence"
                    })
                }
            }]
        }
        mock_create.return_value = mock_response
        
        # 创建测试数据
        market_data = {
            "symbol": "PAXGUSDT",
            "price": 1850.5,
            "timeframe": "1h",
            "indicators": {
                "rsi": 65,
                "ema": {
                    "signal": "SELL",
                    "strength": 0.6
                },
                "macd": {
                    "signal": "SELL",
                    "strength": 0.7
                }
            },
            "smc": {
                "structure": "Bearish",
                "order_blocks": [{"price": 1855.0, "type": "resistance"}],
                "fvg": [{"top": 1853.0, "bottom": 1857.0}]
            },
            "key_levels": {
                "support": [1845.0, 1830.0],
                "resistance": [1860.0, 1875.0]
            }
        }
        
        # 调用方法
        result = self.signal_provider.generate_signal(market_data)
        
        # 验证结果
        self.assertEqual(result["signal"], "SELL")
        self.assertEqual(result["confidence"], 0.75)
        self.assertIn("reasoning", result)
        self.assertEqual(result["provider"], "OpenAI")
        
        # 验证API调用
        mock_create.assert_called_once()
    
    @patch('openai.ChatCompletion.create')
    def test_generate_signal_api_error(self, mock_create):
        """测试API错误情况"""
        # 模拟API错误
        mock_create.side_effect = Exception("API Error")
        
        # 创建测试数据
        market_data = {
            "symbol": "PAXGUSDT",
            "price": 1850.5,
            "timeframe": "1h"
        }
        
        # 调用方法
        result = self.signal_provider.generate_signal(market_data)
        
        # 验证结果
        self.assertEqual(result["signal"], "HOLD")
        self.assertEqual(result["confidence"], 0.0)
        self.assertIn("API Error", result["reasoning"])
        self.assertEqual(result["provider"], "OpenAI")
    
    def test_generate_signal_missing_api_key(self):
        """测试缺少API密钥情况"""
        # 创建没有API密钥的配置
        config_no_key = Mock(spec=Config)
        config_no_key.openai_api_key = None
        
        # 创建信号提供者
        provider_no_key = OpenAISignalProvider(config_no_key, self.logger)
        
        # 创建测试数据
        market_data = {
            "symbol": "PAXGUSDT",
            "price": 1850.5,
            "timeframe": "1h"
        }
        
        # 调用方法
        result = provider_no_key.generate_signal(market_data)
        
        # 验证结果
        self.assertEqual(result["signal"], "HOLD")
        self.assertEqual(result["confidence"], 0.0)
        self.assertIn("API key not configured", result["reasoning"])
        self.assertEqual(result["provider"], "OpenAI")


class TestAISignalGenerator(unittest.TestCase):
    """AI信号生成器测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建模拟配置
        self.config = Mock(spec=Config)
        self.config.ai_providers = ["deepseek", "openai"]
        self.config.ai_consensus_threshold = 0.7
        self.config.ai_signal_timeout = 30
        
        # 创建模拟日志器
        self.logger = Mock()
        
        # 创建AI信号生成器实例
        self.signal_generator = AISignalGenerator(self.config, self.logger)
        
        # 模拟信号提供者
        self.mock_deepseek_provider = Mock(spec=DeepSeekSignalProvider)
        self.mock_openai_provider = Mock(spec=OpenAISignalProvider)
        
        # 替换信号提供者
        self.signal_generator.providers = {
            "deepseek": self.mock_deepseek_provider,
            "openai": self.mock_openai_provider
        }
    
    def test_generate_signal_consensus(self):
        """测试信号共识计算"""
        # 模拟信号提供者返回
        self.mock_deepseek_provider.generate_signal.return_value = {
            "signal": "BUY",
            "confidence": 0.8,
            "reasoning": "Strong bullish momentum",
            "provider": "DeepSeek"
        }
        
        self.mock_openai_provider.generate_signal.return_value = {
            "signal": "BUY",
            "confidence": 0.75,
            "reasoning": "Support level tested",
            "provider": "OpenAI"
        }
        
        # 创建测试数据
        market_data = {
            "symbol": "PAXGUSDT",
            "price": 1850.5,
            "timeframe": "1h"
        }
        
        # 调用方法
        result = self.signal_generator.generate_signal(market_data)
        
        # 验证结果
        self.assertEqual(result["signal"], "BUY")
        self.assertGreater(result["confidence"], 0.7)  # 应该高于共识阈值
        self.assertIn("reasoning", result)
        self.assertIn("consensus", result)
        self.assertEqual(result["consensus"]["agreement"], 2)  # 两个提供者都同意
        self.assertEqual(result["consensus"]["total_providers"], 2)
        
        # 验证API调用
        self.mock_deepseek_provider.generate_signal.assert_called_once_with(market_data)
        self.mock_openai_provider.generate_signal.assert_called_once_with(market_data)
    
    def test_generate_signal_no_consensus(self):
        """测试没有达成共识的情况"""
        # 模拟信号提供者返回
        self.mock_deepseek_provider.generate_signal.return_value = {
            "signal": "BUY",
            "confidence": 0.8,
            "reasoning": "Strong bullish momentum",
            "provider": "DeepSeek"
        }
        
        self.mock_openai_provider.generate_signal.return_value = {
            "signal": "SELL",
            "confidence": 0.75,
            "reasoning": "Resistance level tested",
            "provider": "OpenAI"
        }
        
        # 创建测试数据
        market_data = {
            "symbol": "PAXGUSDT",
            "price": 1850.5,
            "timeframe": "1h"
        }
        
        # 调用方法
        result = self.signal_generator.generate_signal(market_data)
        
        # 验证结果
        self.assertEqual(result["signal"], "HOLD")  # 没有共识，返回HOLD
        self.assertLess(result["confidence"], 0.7)  # 应该低于共识阈值
        self.assertIn("reasoning", result)
        self.assertIn("consensus", result)
        self.assertEqual(result["consensus"]["agreement"], 0)  # 没有提供者同意
        self.assertEqual(result["consensus"]["total_providers"], 2)
    
    def test_generate_signal_partial_consensus(self):
        """测试部分共识情况"""
        # 模拟信号提供者返回
        self.mock_deepseek_provider.generate_signal.return_value = {
            "signal": "BUY",
            "confidence": 0.8,
            "reasoning": "Strong bullish momentum",
            "provider": "DeepSeek"
        }
        
        self.mock_openai_provider.generate_signal.return_value = {
            "signal": "HOLD",
            "confidence": 0.5,
            "reasoning": "Unclear market direction",
            "provider": "OpenAI"
        }
        
        # 创建测试数据
        market_data = {
            "symbol": "PAXGUSDT",
            "price": 1850.5,
            "timeframe": "1h"
        }
        
        # 调用方法
        result = self.signal_generator.generate_signal(market_data)
        
        # 验证结果
        self.assertEqual(result["signal"], "BUY")  # 部分共识，选择最高置信度的信号
        self.assertGreater(result["confidence"], 0.5)  # 应该高于HOLD的置信度
        self.assertIn("reasoning", result)
        self.assertIn("consensus", result)
        self.assertEqual(result["consensus"]["agreement"], 1)  # 一个提供者同意
        self.assertEqual(result["consensus"]["total_providers"], 2)
    
    def test_calculate_consensus(self):
        """测试共识计算"""
        # 创建测试信号
        signals = [
            {"signal": "BUY", "confidence": 0.8, "provider": "DeepSeek"},
            {"signal": "BUY", "confidence": 0.75, "provider": "OpenAI"},
            {"signal": "SELL", "confidence": 0.6, "provider": "AnotherAI"}
        ]
        
        # 调用方法
        result = self.signal_generator.calculate_consensus(signals)
        
        # 验证结果
        self.assertEqual(result["signal"], "BUY")
        self.assertEqual(result["confidence"], 0.775)  # (0.8 + 0.75) / 2
        self.assertEqual(result["agreement"], 2)  # 两个提供者同意
        self.assertEqual(result["total_providers"], 3)
        self.assertEqual(result["consensus_ratio"], 2/3)
    
    def test_select_primary_signal(self):
        """测试主信号选择"""
        # 创建测试信号
        signals = [
            {"signal": "BUY", "confidence": 0.8, "provider": "DeepSeek"},
            {"signal": "BUY", "confidence": 0.75, "provider": "OpenAI"},
            {"signal": "SELL", "confidence": 0.9, "provider": "AnotherAI"}
        ]
        
        # 调用方法
        result = self.signal_generator.select_primary_signal(signals)
        
        # 验证结果
        self.assertEqual(result["signal"], "BUY")  # 应该选择共识信号
        self.assertEqual(result["confidence"], 0.775)  # (0.8 + 0.75) / 2
        self.assertEqual(result["provider"], "Consensus")
    
    def test_get_provider_status(self):
        """测试提供者状态获取"""
        # 调用方法
        result = self.signal_generator.get_provider_status()
        
        # 验证结果
        self.assertIsInstance(result, dict)
        self.assertIn("deepseek", result)
        self.assertIn("openai", result)
        self.assertTrue(result["deepseek"])
        self.assertTrue(result["openai"])


if __name__ == "__main__":
    unittest.main()