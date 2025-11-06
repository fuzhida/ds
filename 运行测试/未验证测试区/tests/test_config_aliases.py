import os
import json
import tempfile

from config import Config


def test_config_alias_mapping_and_autoname():
    # 构造带有别名字段的临时配置文件
    cfg = {
        "trading": {
            "symbol": "BTC/USDT:USDT",
            "rr_ratio": 1.8,
            "max_positions": 3
        },
        "logging": {
            # 不显式提供，以便走自动命名
        }
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "config.json")
        with open(path, "w") as f:
            json.dump(cfg, f)

        c = Config(config_path=path)

        # 别名统一映射
        assert c.risk_reward_ratio == 1.8
        assert c.rr_ratio == 1.8
        assert c.max_open_positions == 3
        assert c.max_positions == 3

        # 按符号自动命名（小写且去掉分隔符）
        assert isinstance(c.log_file, str) and len(c.log_file) > 0
        assert isinstance(c.signals_file, str) and len(c.signals_file) > 0
        basename_log = os.path.basename(c.log_file)
        basename_sig = os.path.basename(c.signals_file)
        assert basename_log.endswith("_trading_bot.log")
        assert basename_sig.startswith("signals_") and basename_sig.endswith(".jsonl")