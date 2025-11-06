import types
from unittest.mock import Mock

from 运行测试.未验证测试区.btcusdc_trading_bot import create_btcusdc_config, main
from 运行测试.未验证测试区.trading_bot import TradingBot


def test_btcusdc_config_symbol():
    cfg = create_btcusdc_config()
    assert cfg.symbol == "BTC/USDC"


def test_btcusdc_bot_initialization_skips_heavy_init_when_logger_provided():
    cfg = create_btcusdc_config()
    logger = Mock()
    bot = TradingBot(cfg, logger=logger)
    # 提供 logger 时跳过 _initialize_components，相关组件应为 None
    assert bot.exchange_manager is None
    assert bot.trading_executor is None


def test_btcusdc_main_allows_injection_logger_to_avoid_real_run():
    # 传入 logger 时仅构建并返回，不实际启动运行循环
    logger = Mock()
    # main 不返回对象，但应当在传入 logger 情况下不启动长循环
    # 只需验证函数可正常调用
    main(logger=logger)