"""
BTC/USDC 交易对快速启动脚本

提供一个最小可用的配置与入口，便于基于现有 TradingBot
在 BTC/USDC 交易对上运行或测试。
"""

from typing import Optional
import time

from config import Config
from trading_bot import TradingBot


def create_btcusdc_config() -> Config:
    """创建 BTC/USDC 配置（默认使用模拟模式与 hyperliquid 数据源）。"""
    return Config(
        symbol="BTC/USDC",
        data_source="hyperliquid",
        simulation_mode=True,
        amount=0.01,
        leverage=10,
    )


def main(logger: Optional[object] = None):
    """启动交易机器人（若传入自定义 logger，将跳过重型初始化，用于测试）。"""
    cfg = create_btcusdc_config()
    bot = TradingBot(cfg, logger=logger)

    # 当未传入外部 logger 时，执行真实启动流程
    if logger is None:
        if bot.start():
            try:
                while True:
                    time.sleep(10)
            except KeyboardInterrupt:
                pass
            finally:
                bot.stop()


if __name__ == "__main__":
    main()