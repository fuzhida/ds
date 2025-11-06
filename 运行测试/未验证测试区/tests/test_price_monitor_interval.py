import json
import tempfile
import os

from config import Config


def test_default_price_monitor_interval_is_60():
    cfg = Config()
    assert getattr(cfg, "price_monitor_interval", None) == 60


def test_price_monitor_interval_loaded_from_config_file():
    data = {
        "analysis": {
            "price_monitor_interval": 5
        },
        "trading": {
            "symbol": "BTC/USDC"
        }
    }
    fd, path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        cfg = Config(config_path=path)
        assert cfg.price_monitor_interval == 5
    finally:
        os.remove(path)