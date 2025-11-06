"""
主程序入口 - 简化版的交易机器人启动器
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from trading_bot import TradingBot


def setup_logging(log_level="INFO", symbol: str = None):
    """设置日志"""
    # 创建日志目录
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # 配置日志格式
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # 配置日志器
    logger = logging.getLogger("Main")
    logger.setLevel(getattr(logging, log_level))
    
    # 清除现有处理器
    logger.handlers.clear()
    
    # 文件处理器：若提供符号，则按符号命名
    if symbol and isinstance(symbol, str):
        sanitized = symbol.replace('/', '').replace(':', '').replace('-', '').lower()
        log_file = os.path.join(log_dir, f"{sanitized}_main.log")
    else:
        log_file = os.path.join(log_dir, f"main_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)
    
    return logger


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="PAXG交易机器人")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--log-level", type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日志级别")
    parser.add_argument("--mode", type=str, default="run", 
                       choices=["run", "status", "manual-trade", "close-all"], help="运行模式")
    parser.add_argument("--signal", type=str, default=None, 
                       choices=["BUY", "SELL", "HOLD"], help="手动交易信号")
    
    args = parser.parse_args()
    
    # 创建交易机器人实例
    try:
        bot = TradingBot(args.config)
        # 交易机器人创建后，使用其符号设置主日志
        logger = setup_logging(args.log_level, symbol=getattr(bot.config, 'symbol', None))
        
        # 根据模式执行不同操作
        if args.mode == "run":
            # 运行模式
            logger.info("启动交易机器人")
            bot.initialize()
            bot.start()
            
            # 保持运行
            try:
                while True:
                    import time
                    time.sleep(10)
            except KeyboardInterrupt:
                logger.info("接收到停止信号")
            
            # 停止机器人
            bot.stop()
            
        elif args.mode == "status":
            # 状态模式
            status = bot.get_status()
            print("交易机器人状态:")
            print(f"运行中: {status.get('is_running', False)}")
            print(f"暂停中: {status.get('is_paused', False)}")
            print(f"开仓数量: {status.get('open_positions_count', 0)}")
            print(f"最后信号: {status.get('last_signal', {}).get('signal', 'None')}")
            print(f"最后分析时间: {status.get('last_analysis_time', 'None')}")
            
        elif args.mode == "manual-trade":
            # 手动交易模式
            if not args.signal:
                logger.error("手动交易模式需要指定信号类型")
                sys.exit(1)
            
            logger.info(f"执行手动交易: {args.signal}")
            result = bot.manual_trade(args.signal)
            
            if result["success"]:
                logger.info(f"手动交易成功: {result}")
            else:
                logger.error(f"手动交易失败: {result}")
                sys.exit(1)
            
        elif args.mode == "close-all":
            # 关闭所有持仓模式
            logger.info("关闭所有持仓")
            result = bot.close_all_positions()
            
            if result["success"]:
                logger.info(f"关闭持仓成功: {result}")
            else:
                logger.error(f"关闭持仓失败: {result}")
                sys.exit(1)
    
    except Exception as e:
        logger.error(f"运行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()