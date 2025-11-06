import warnings
import time
import logging

from config import Config
from exchange_manager import ExchangeManager
from risk_manager import RiskManager, PositionManager
from trading_executor import TradingExecutor
from trading_bot import TradingBot


def make_logger(name: str = "TestLogger"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def test_executor_async_loop_runs_without_never_awaited_warning():
    logger = make_logger("ExecutorTest")
    cfg = Config(symbol="BTC/USDC", simulation_mode=True, execution_interval=1)

    exm = ExchangeManager(cfg, logger)
    rm = RiskManager(cfg, logger)
    pm = PositionManager(cfg, logger, rm)
    executor = TradingExecutor(cfg, logger, exm, rm, pm)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        executor.start()
        time.sleep(0.2)
        executor.stop()

        msgs = [str(x.message) for x in w]
        assert not any("was never awaited" in m for m in msgs), msgs
        assert not any("_async_execution_loop" in m and "never awaited" in m for m in msgs), msgs


def test_trading_bot_start_without_sleep_never_awaited_warning():
    logger = make_logger("BotTest")
    # 使用真实初始化（logger=None）以覆盖 start 内的异步任务创建路径
    cfg = Config(symbol="BTC/USDC", simulation_mode=True)
    bot = TradingBot(cfg, logger=None)

    assert bot.initialize() is True

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        started = bot.start()
        assert started is True
        time.sleep(0.2)
        bot.stop()

        msgs = [str(x.message) for x in w]
        assert not any("sleep" in m and "was never awaited" in m for m in msgs), msgs