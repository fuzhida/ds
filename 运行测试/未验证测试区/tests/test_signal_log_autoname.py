import os
import json
import logging

from config import Config
from trading_bot import TradingBot


def test_signal_log_autoname_and_write():
    # 使用默认配置，符号应自动生成路径
    c = Config()
    c.symbol = "ETH/USDT:USDT"

    # 注入轻量logger以跳过重型初始化
    test_logger = logging.getLogger("TestLogger")
    bot = TradingBot(c, logger=test_logger)

    # 写入一条测试信号
    sig = {"signal": "BUY", "confidence": 0.9, "reasoning": "unit-test"}
    bot.write_signal_log(sig)

    # 验证文件存在且包含内容
    path = c.signals_file
    assert isinstance(path, str) and len(path) > 0
    assert os.path.exists(os.path.dirname(path))
    assert os.path.isfile(path)
    with open(path, "r") as f:
        lines = f.readlines()
        assert len(lines) >= 1
        last = json.loads(lines[-1])
        assert last.get("symbol") == c.symbol
        assert last.get("signal") == "BUY"