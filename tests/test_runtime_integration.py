import os
import json
import pytest

from config import Config
from trading_bot import TradingBot


def _example_config_path():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "config.json.example")


def test_config_file_loading():
    cfg_path = _example_config_path()
    assert os.path.exists(cfg_path), "config.json.example 应该存在于项目根目录"

    cfg = Config(config_path=cfg_path)

    # 验证分析调度相关参数加载正确
    assert isinstance(cfg.timeframes, list) and len(cfg.timeframes) > 0
    assert cfg.analysis_interval == 60
    assert cfg.risk_check_interval == 5
    assert cfg.report_interval == 24


def test_bot_status_initialization():
    cfg_path = _example_config_path()

    # 使用文件路径初始化交易机器人（模拟模式）
    bot = TradingBot(cfg_path)

    status = bot.get_status()
    # 机器人初始状态不在运行，不暂停，无开仓
    assert status.get("is_running", False) is False
    assert status.get("is_paused", False) is False
    assert status.get("open_positions_count", 0) == 0