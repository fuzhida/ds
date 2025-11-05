"""
数据管理模块 - 负责数据获取、缓存和管理
"""

import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timezone, timedelta
import time
import threading


class DataManager:
    """数据管理器"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        
        # 数据缓存
        self.data_cache = {}
        self.cache_lock = threading.Lock()
        
        # 缓存配置
        self.cache_ttl = getattr(config, 'cache_ttl', 300)  # 默认5分钟
        self.cache_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache.json")
        
        # 加载缓存
        self._load_cache()
    
    def _load_cache(self):
        """加载缓存数据"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                current_time = time.time()
                
                # 检查缓存是否过期
                for symbol, data in cache_data.items():
                    if current_time - data.get('timestamp', 0) < self.cache_ttl:
                        self.data_cache[symbol] = data
                    else:
                        self.logger.debug(f"缓存数据过期: {symbol}")
                
                self.logger.info(f"加载了 {len(self.data_cache)} 个缓存数据")
        except Exception as e:
            self.logger.error(f"加载缓存失败: {e}")
    
    def _save_cache(self):
        """保存缓存数据"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.data_cache, f, indent=2)
        except Exception as e:
            self.logger.error(f"保存缓存失败: {e}")
    
    def get_market_data(self, symbol: str, timeframes: List[str], limit: int = 100) -> Dict[str, Any]:
        """获取市场数据"""
        try:
            with self.cache_lock:
                # 检查缓存
                cache_key = f"{symbol}_{'_'.join(sorted(timeframes))}_{limit}"
                if cache_key in self.data_cache:
                    cache_data = self.data_cache[cache_key]
                    current_time = time.time()
                    
                    # 检查缓存是否过期
                    if current_time - cache_data.get('timestamp', 0) < self.cache_ttl:
                        self.logger.debug(f"使用缓存数据: {cache_key}")
                        return cache_data['data']
                
                # 生成模拟数据（实际应用中应该从交易所API获取）
                data = {}
                for tf in timeframes:
                    data[tf] = self._generate_ohlcv_data(tf, limit)
                
                # 更新缓存
                self.data_cache[cache_key] = {
                    'timestamp': time.time(),
                    'data': data
                }
                
                # 定期保存缓存
                if len(self.data_cache) % 10 == 0:
                    self._save_cache()
                
                return data
                
        except Exception as e:
            self.logger.error(f"获取市场数据失败: {e}")
            return {}
    
    def _generate_ohlcv_data(self, timeframe: str, limit: int) -> pd.DataFrame:
        """生成模拟OHLCV数据"""
        try:
            # 根据时间框架确定时间间隔
            tf_minutes = {
                '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
                '1h': 60, '4h': 240, '1d': 1440
            }.get(timeframe, 15)
            
            # 生成时间序列
            end_time = datetime.now(timezone.utc)
            times = [end_time - timedelta(minutes=i * tf_minutes) for i in range(limit)]
            times.reverse()
            
            # 生成价格数据
            base_price = 2000.0  # 基础价格
            prices = [base_price]
            
            for i in range(1, limit):
                # 随机波动
                change = np.random.normal(0, 0.002)  # 0.2%标准差
                new_price = prices[-1] * (1 + change)
                prices.append(new_price)
            
            # 生成OHLCV数据
            data = []
            for i, (time, price) in enumerate(zip(times, prices)):
                # 随机生成高低价
                high = price * (1 + abs(np.random.normal(0, 0.001)))
                low = price * (1 - abs(np.random.normal(0, 0.001)))
                
                # 确保高低价关系正确
                high = max(high, price)
                low = min(low, price)
                
                # 生成开盘价（第一根K线的开盘价等于收盘价）
                if i == 0:
                    open_price = price
                else:
                    # 上一根K线的收盘价
                    open_price = prices[i-1]
                
                # 生成成交量
                volume = np.random.normal(1000000, 200000)
                volume = max(volume, 100000)  # 最小成交量
                
                data.append({
                    'time': time,
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': price,
                    'volume': volume
                })
            
            # 转换为DataFrame
            df = pd.DataFrame(data)
            df.set_index('time', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"生成模拟数据失败 {timeframe}: {e}")
            return pd.DataFrame()
    
    def clear_cache(self):
        """清空缓存"""
        try:
            with self.cache_lock:
                self.data_cache.clear()
                if os.path.exists(self.cache_file):
                    os.remove(self.cache_file)
                self.logger.info("缓存已清空")
        except Exception as e:
            self.logger.error(f"清空缓存失败: {e}")
    
    def get_cache_status(self) -> Dict[str, Any]:
        """获取缓存状态"""
        try:
            with self.cache_lock:
                return {
                    'cache_size': len(self.data_cache),
                    'cache_file': self.cache_file,
                    'cache_ttl': self.cache_ttl
                }
        except Exception as e:
            self.logger.error(f"获取缓存状态失败: {e}")
            return {}