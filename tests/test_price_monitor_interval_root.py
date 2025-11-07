import os
from config import Config


def _example_config_path():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "config.json.example")


def test_price_monitor_interval_loaded_from_example():
    cfg_path = _example_config_path()
    cfg = Config(config_path=cfg_path)
    assert getattr(cfg, "price_monitor_interval", None) == 60