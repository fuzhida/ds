"""
Root-level trading_bot shim to reuse the TradingBot implementation
from 运行测试/未验证测试区 for integration tests.
"""

import os
import sys

try:
    # Ensure 未验证测试区 is on sys.path so its sibling imports resolve
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '运行测试', '未验证测试区'))
    from 运行测试.未验证测试区.trading_bot import TradingBot  # type: ignore
except Exception as e:
    raise e

__all__ = ["TradingBot"]