"""
交易所管理模块 - 包含交易所连接、数据获取和订单管理功能
"""

import ccxt
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
import time
from datetime import datetime, timezone, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed


class ExchangeManager:
    """交易所管理器"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.exchange = None
        self.max_retries = config.api_retry_count
        self.retry_delay = config.api_retry_delay
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        self._setup_exchange()
    
    def _setup_exchange(self):
        """设置交易所连接"""
        try:
            if self.config.exchange_name.lower() == 'binance':
                self.exchange = ccxt.binance({
                    'apiKey': self.config.api_key,
                    'secret': self.config.api_secret,
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'future',
                    },
                })
            elif self.config.exchange_name.lower() == 'okx':
                self.exchange = ccxt.okx({
                    'apiKey': self.config.api_key,
                    'secret': self.config.api_secret,
                    'password': self.config.api_passphrase,
                    'enableRateLimit': True,
                })
            else:
                raise ValueError(f"不支持的交易所: {self.config.exchange_name}")
            
            self.logger.info(f"交易所 {self.config.exchange_name} 初始化成功")
            
        except Exception as e:
            self.logger.error(f"交易所初始化失败: {e}")
            raise
    
    def test_connection(self) -> bool:
        """测试交易所连接"""
        # 如果是模拟模式，直接返回True
        if self.config.simulation_mode:
            self.logger.info("模拟模式：跳过交易所连接测试")
            return True
            
        try:
            balance = self.exchange.fetch_balance()
            self.logger.info("交易所连接测试成功")
            return True
        except Exception as e:
            self.logger.error(f"交易所连接测试失败: {e}")
            return False
    
    def connect(self) -> bool:
        """连接交易所"""
        return self.test_connection()
    
    def safe_fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100, since: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        安全获取OHLCV数据
        :param symbol: 交易对
        :param timeframe: 时间框架
        :param limit: 数据条数
        :param since: 起始时间戳
        :return: DataFrame或None
        """
        for attempt in range(self.max_retries):
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
                
                if not ohlcv:
                    self.logger.warning(f"获取 {symbol} {timeframe} OHLCV数据为空")
                    return None
                
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                self.logger.debug(f"成功获取 {symbol} {timeframe} OHLCV数据: {len(df)} 条")
                return df
                
            except Exception as e:
                self.logger.warning(f"获取 {symbol} {timeframe} OHLCV数据失败 (尝试 {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error(f"获取 {symbol} {timeframe} OHLCV数据最终失败")
                    return None
    
    def safe_fetch_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        安全获取行情数据
        :param symbol: 交易对
        :return: 行情数据字典或None
        """
        for attempt in range(self.max_retries):
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                self.logger.debug(f"成功获取 {symbol} 行情数据")
                return ticker
                
            except Exception as e:
                self.logger.warning(f"获取 {symbol} 行情数据失败 (尝试 {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error(f"获取 {symbol} 行情数据最终失败")
                    return None
    
    def safe_create_order(self, symbol: str, order_type: str, side: str, amount: float, 
                         price: Optional[float] = None, params: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """
        安全创建订单
        :param symbol: 交易对
        :param order_type: 订单类型 (market, limit)
        :param side: 订单方向 (buy, sell)
        :param amount: 订单数量
        :param price: 订单价格 (限价单必需)
        :param params: 额外参数
        :return: 订单信息或None
        """
        for attempt in range(self.max_retries):
            try:
                if order_type == 'market':
                    order = self.exchange.create_market_order(symbol, side, amount, params)
                elif order_type == 'limit':
                    if price is None:
                        raise ValueError("限价单必须指定价格")
                    order = self.exchange.create_limit_order(symbol, side, amount, price, params)
                else:
                    raise ValueError(f"不支持的订单类型: {order_type}")
                
                self.logger.info(f"成功创建 {side} {order_type} 订单: {symbol} 数量={amount} 价格={price}")
                return order
                
            except Exception as e:
                self.logger.warning(f"创建订单失败 (尝试 {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error(f"创建订单最终失败")
                    return None
    
    def safe_cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        安全取消订单
        :param order_id: 订单ID
        :param symbol: 交易对
        :return: 是否成功
        """
        for attempt in range(self.max_retries):
            try:
                self.exchange.cancel_order(order_id, symbol)
                self.logger.info(f"成功取消订单: {order_id}")
                return True
                
            except Exception as e:
                self.logger.warning(f"取消订单失败 (尝试 {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error(f"取消订单最终失败")
                    return False
    
    def safe_fetch_order(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        """
        安全获取订单信息
        :param order_id: 订单ID
        :param symbol: 交易对
        :return: 订单信息或None
        """
        for attempt in range(self.max_retries):
            try:
                order = self.exchange.fetch_order(order_id, symbol)
                self.logger.debug(f"成功获取订单信息: {order_id}")
                return order
                
            except Exception as e:
                self.logger.warning(f"获取订单信息失败 (尝试 {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error(f"获取订单信息最终失败")
                    return None
    
    def safe_fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        安全获取未成交订单
        :param symbol: 交易对 (可选)
        :return: 未成交订单列表
        """
        for attempt in range(self.max_retries):
            try:
                orders = self.exchange.fetch_open_orders(symbol)
                self.logger.debug(f"成功获取未成交订单: {len(orders)} 条")
                return orders
                
            except Exception as e:
                self.logger.warning(f"获取未成交订单失败 (尝试 {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error(f"获取未成交订单最终失败")
                    return []
    
    def safe_fetch_balance(self) -> Optional[Dict[str, Any]]:
        """
        安全获取账户余额
        :return: 余额信息或None
        """
        for attempt in range(self.max_retries):
            try:
                balance = self.exchange.fetch_balance()
                self.logger.debug("成功获取账户余额")
                return balance
                
            except Exception as e:
                self.logger.warning(f"获取账户余额失败 (尝试 {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error(f"获取账户余额最终失败")
                    return None
    
    def safe_fetch_positions(self) -> List[Dict[str, Any]]:
        """
        安全获取持仓信息
        :return: 持仓信息列表
        """
        for attempt in range(self.max_retries):
            try:
                positions = self.exchange.fetch_positions()
                # 过滤出有持仓的合约
                active_positions = [p for p in positions if float(p.get('contracts', 0)) > 0]
                self.logger.debug(f"成功获取持仓信息: {len(active_positions)} 个")
                return active_positions
                
            except Exception as e:
                self.logger.warning(f"获取持仓信息失败 (尝试 {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error(f"获取持仓信息最终失败")
                    return []
    
    def fetch_multi_timeframe_data(self, symbol: str, timeframes: List[str], limit: int = 100) -> Dict[str, pd.DataFrame]:
        """
        并行获取多时间框架数据
        :param symbol: 交易对
        :param timeframes: 时间框架列表
        :param limit: 数据条数
        :return: 多时间框架数据字典
        """
        results = {}
        
        # 使用线程池并行获取数据
        futures = {
            self.executor.submit(self.safe_fetch_ohlcv, symbol, tf, limit): tf 
            for tf in timeframes
        }
        
        for future in as_completed(futures):
            tf = futures[future]
            try:
                df = future.result(timeout=30)  # 设置超时
                if df is not None:
                    results[tf] = df
                    self.logger.debug(f"成功获取 {symbol} {tf} 数据: {len(df)} 条")
                else:
                    self.logger.warning(f"获取 {symbol} {tf} 数据失败")
            except Exception as e:
                self.logger.error(f"获取 {symbol} {tf} 数据异常: {e}")
        
        return results
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        获取当前价格
        :param symbol: 交易对
        :return: 当前价格或None
        """
        ticker = self.safe_fetch_ticker(symbol)
        if ticker:
            return ticker.get('last')
        return None
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取交易对信息
        :param symbol: 交易对
        :return: 交易对信息或None
        """
        try:
            markets = self.exchange.load_markets()
            if symbol in markets:
                return markets[symbol]
            else:
                self.logger.error(f"未找到交易对信息: {symbol}")
                return None
        except Exception as e:
            self.logger.error(f"获取交易对信息失败: {e}")
            return None
    
    def calculate_position_size(self, symbol: str, risk_percent: float, stop_loss_pct: float) -> Optional[float]:
        """
        计算仓位大小
        :param symbol: 交易对
        :param risk_percent: 风险百分比 (0.01 = 1%)
        :param stop_loss_pct: 止损百分比 (0.02 = 2%)
        :return: 仓位大小或None
        """
        try:
            # 获取账户余额
            balance = self.safe_fetch_balance()
            if not balance:
                return None
            
            # 获取可用余额 (USDT)
            if 'USDT' in balance['free']:
                available_balance = float(balance['free']['USDT'])
            else:
                self.logger.error("账户中没有USDT余额")
                return None
            
            # 获取当前价格
            current_price = self.get_current_price(symbol)
            if not current_price:
                return None
            
            # 计算风险金额
            risk_amount = available_balance * risk_percent
            
            # 计算仓位大小
            position_size = risk_amount / (current_price * stop_loss_pct)
            
            # 获取交易对信息
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                return None
            
            # 获取最小交易量
            min_amount = symbol_info.get('limits', {}).get('amount', {}).get('min')
            if min_amount and position_size < min_amount:
                position_size = min_amount
            
            # 获取精度
            precision = symbol_info.get('precision', {}).get('amount', 8)
            
            # 应用精度
            position_size = round(position_size, precision)
            
            self.logger.debug(f"计算仓位大小: {symbol} 风险={risk_percent*100}% 止损={stop_loss_pct*100}% 仓位={position_size}")
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"计算仓位大小失败: {e}")
            return None
    
    def close_all_positions(self) -> bool:
        """
        关闭所有持仓
        :return: 是否成功
        """
        try:
            positions = self.safe_fetch_positions()
            if not positions:
                self.logger.info("没有需要关闭的持仓")
                return True
            
            success = True
            for position in positions:
                symbol = position['symbol']
                size = float(position['contracts'])
                side = 'buy' if position['side'] == 'short' else 'sell'
                
                # 创建市价单平仓
                order = self.safe_create_order(symbol, 'market', side, size)
                if not order:
                    success = False
                    self.logger.error(f"平仓失败: {symbol}")
                else:
                    self.logger.info(f"平仓成功: {symbol} {side} {size}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"关闭所有持仓失败: {e}")
            return False
    
    def cleanup(self):
        """清理资源"""
        try:
            if self.executor:
                self.executor.shutdown(wait=True)
            self.logger.info("交易所管理器资源清理完成")
        except Exception as e:
            self.logger.error(f"交易所管理器资源清理失败: {e}")