import logging
import pandas as pd

from 运行测试.未验证测试区.ai_signal_generator import DeepSeekSignalProvider


class DummyConfig:
    deepseek_api_key = ""
    deepseek_base_url = "https://api.deepseek.com"
    deepseek_model = "deepseek-reasoner"
    symbol = "BTCUSDT"
    ai_timeout = 30


class DummyLogger:
    def debug(self, *args, **kwargs):
        pass
    def info(self, *args, **kwargs):
        pass
    def warning(self, *args, **kwargs):
        pass
    def error(self, *args, **kwargs):
        pass


def _make_market_data_minimal():
    tech_1h = {
        'ema': pd.Series([], dtype=float),
        'rsi': pd.Series([], dtype=float),
        'atr': pd.Series([], dtype=float),
        'macd': pd.DataFrame(columns=['macd', 'signal', 'histogram']),
        'bollinger_bands': pd.DataFrame(columns=['upper', 'middle', 'lower']),
        'overall_score': 0.0,
    }
    market_data = {
        'current_price': 12345.67,
        'technical_indicators': {
            '1h': tech_1h
        },
        'key_levels': {
            'support': [],
            'resistance': [],
            'pivot_points': {},
            'vwap': pd.Series([], dtype=float)
        },
        'price_action': {
            'candlestick_patterns': {},
            'price_efficiency': 0.0,
            'volatility': {},
            'momentum': {},
        },
        'smc_structures': {
            'bos_choch': {},
            'order_blocks': {'bullish': [], 'bearish': []},
            'fvg': {'bullish': [], 'bearish': []},
            'swing_points': {'highs': [], 'lows': []},
            'overall_score': 0.0,
        },
    }
    return market_data


def test_prompt_no_hard_150_char_limit_in_system_prompt():
    provider = DeepSeekSignalProvider(DummyConfig(), DummyLogger())
    prompt = provider._build_prompt(_make_market_data_minimal())
    assert isinstance(prompt, str)
    assert "≤150字" not in prompt
    assert "不超过150字" not in prompt


def test_fallback_prompt_no_hard_150_char_limit():
    provider = DeepSeekSignalProvider(DummyConfig(), DummyLogger())
    prompt = provider._build_prompt([])  # type: ignore
    assert isinstance(prompt, str)
    assert "≤150字" not in prompt
    assert "不超过150字" not in prompt