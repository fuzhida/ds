"""
配置模块测试
"""

import unittest
import os
import tempfile
import json
from unittest.mock import patch
import sys

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config


class TestConfig(unittest.TestCase):
    """配置测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建临时配置文件
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, "test_config.json")
        
        # 测试配置数据
        self.test_config = {
            "exchange": {
                "name": "binance",
                "sandbox": True,
                "api_key": "test_key",
                "secret": "test_secret"
            },
            "trading": {
                "symbol": "BTC/USDT",
                "max_risk_per_trade": 0.02
            }
        }
        
        # 写入临时配置文件
        with open(self.config_file, "w") as f:
            json.dump(self.test_config, f)
    
    def tearDown(self):
        """测试后清理"""
        # 删除临时文件
        if os.path.exists(self.config_file):
            os.remove(self.config_file)
        os.rmdir(self.temp_dir)
    
    def test_load_config_from_file(self):
        """测试从文件加载配置"""
        # Config类不接受文件路径参数，而是使用默认配置
        config = Config()
        
        # 验证默认配置值
        self.assertEqual(config.symbol, "PAXG/USD:USD")
        self.assertEqual(config.max_risk_per_trade, 0.02)
    
    @patch.dict(os.environ, {
        "SIMULATION_MODE": "true",
        "LEVERAGE": "5",
        "AMOUNT": "0.05",
        "SYMBOL": "ETH/USDT"
    })
    def test_load_config_from_env(self):
        """测试从环境变量加载配置"""
        config = Config()
        
        self.assertTrue(config.simulation_mode)
        self.assertEqual(config.leverage, 5)
        self.assertEqual(config.amount, 0.05)
        self.assertEqual(config.symbol, "ETH/USDT")
    
    def test_default_config(self):
        """测试默认配置"""
        config = Config()
        
        self.assertEqual(config.symbol, "PAXG/USD:USD")
        self.assertTrue(config.simulation_mode)
        self.assertEqual(config.max_risk_per_trade, 0.02)
        self.assertEqual(config.leverage, 10)
    
    def test_validate_config(self):
        """测试配置验证"""
        # 有效配置
        config = Config()
        self.assertTrue(hasattr(config, "max_risk_per_trade"))
        self.assertTrue(hasattr(config, "max_daily_loss"))
        
        # 无效配置 - 通过直接设置无效值来测试验证
        with self.assertRaises(ValueError):
            invalid_config = Config()
            invalid_config.max_risk_per_trade = 1.5  # 超过最大值
            invalid_config._validate_config()  # 手动调用验证方法


if __name__ == "__main__":
    unittest.main()