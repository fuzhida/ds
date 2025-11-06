import os
from config import Config


def test_timeframes_default_list_present():
    cfg = Config()
    # 默认工厂应提供非空时间框架列表
    assert isinstance(cfg.timeframes, list)
    assert len(cfg.timeframes) >= 5
    # 包含关键短周期与长周期（顺序可不同）
    assert '1m' in cfg.timeframes
    assert '3m' in cfg.timeframes
    assert '1h' in cfg.timeframes
    assert '4h' in cfg.timeframes
    assert '1d' in cfg.timeframes


def test_contextual_log_auto_path_by_symbol():
    # 当使用默认占位日志文件名时，应自动改为按符号命名
    cfg = Config()
    # 默认符号来自配置，生成的路径应包含符号清洗后的前缀
    assert isinstance(cfg.contextual_log_file, str)
    basename = os.path.basename(cfg.contextual_log_file)
    assert basename.startswith("contextual_")
    assert basename.endswith(".jsonl")