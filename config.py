"""
Root-level config shim to reuse the validated Config implementation
from 运行测试/未验证测试区 without duplicating logic.
"""

import os
import sys

try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '运行测试', '未验证测试区'))
    from 运行测试.未验证测试区.config import Config  # type: ignore
except Exception as e:
    # Surface the import error to callers for easier diagnosis
    raise e

__all__ = ["Config"]