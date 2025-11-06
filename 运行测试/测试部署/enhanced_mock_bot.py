#!/usr/bin/env python3
"""
增强版MockBot类 - 集成增强版数据提取器和提示词生成器
用于测试增强版原始数据处理功能
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from enhanced_data_extractor import EnhancedDataExtractor
from enhanced_smc_prompt import get_enhanced_smc_prompt

class EnhancedMockBot:
    """增强版MockBot类，提供更丰富的原始数据供AI计算SMC结构"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data_extractor = EnhancedDataExtractor()
    
    def extract_enhanced_raw_data(self, 
                                 ohlc_data: List[Dict], 
                                 volume_data: List[Dict],
                                 market_depth: Optional[List[Dict]] = None,
                                 time_sales: Optional[List[Dict]] = None,
                                 market_sentiment: Optional[Dict] = None) -> Dict[str, Any]:
        """
        提取增强版原始数据，包含DeepSeek建议的高优先级改进
        
        参数:
            ohlc_data: OHLC数据列表
            volume_data: 成交量数据列表
            market_depth: 市场深度数据 (可选)
            time_sales: 时间与销售数据 (可选)
            market_sentiment: 市场情绪数据 (可选)
            
        返回:
            增强版原始数据字典
        """
        return self.data_extractor.extract_enhanced_raw_data(
            ohlc_data=ohlc_data,
            volume_data=volume_data,
            market_depth=market_depth,
            time_sales=time_sales,
            market_sentiment=market_sentiment
        )
    
    def generate_enhanced_smc_prompt(self, market_data: Dict[str, Any]) -> str:
        """
        生成增强版SMC分析提示词
        
        参数:
            market_data: 包含增强版原始数据的字典
            
        返回:
            增强版SMC分析提示词
        """
        return get_enhanced_smc_prompt(market_data)

# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    import random
    
    # 生成OHLC数据
    ohlc_data = []
    base_price = 42000
    for i in range(50):
        timestamp = f"2024-01-{(i%30)+1:02d}T{(i%24):02d}:00:00Z"
        open_price = base_price + random.uniform(-100, 100)
        close_price = open_price + random.uniform(-50, 50)
        high_price = max(open_price, close_price) + random.uniform(0, 50)
        low_price = min(open_price, close_price) - random.uniform(0, 50)
        volume = random.uniform(800, 1500)
        
        ohlc_data.append({
            "timestamp": timestamp,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume,
            "timeframe": "1h"
        })
        
        base_price = close_price
    
    # 生成市场深度数据
    market_depth = []
    for i in range(10):
        mid_price = 42000 + i * 10
        market_depth.append({
            "timestamp": f"2024-01-01T{(i%24):02d}:00:00Z",
            "bid_price": mid_price - 5,
            "ask_price": mid_price + 5,
            "bid_volume": random.uniform(100, 500),
            "ask_volume": random.uniform(100, 500)
        })
    
    # 生成时间与销售数据
    time_sales = []
    for i in range(100):
        time_sales.append({
            "timestamp": f"2024-01-01T{(i%24):02d}:{(i%60):02d}:{(i%60):02d}",
            "price": 42000 + random.uniform(-100, 100),
            "volume": random.uniform(0.1, 20),
            "side": random.choice(["buy", "sell"]),
            "aggressive": random.choice([True, False])
        })
    
    # 生成市场情绪数据
    market_sentiment = {
        "fear_greed_index": random.uniform(0, 100),
        "funding_rate": random.uniform(-0.01, 0.01),
        "open_interest_change": random.uniform(-5, 5),
        "long_short_ratio": random.uniform(0.8, 1.5)
    }
    
    # 创建增强版MockBot实例
    enhanced_bot = EnhancedMockBot()
    
    # 提取增强版原始数据
    enhanced_raw_data = enhanced_bot.extract_enhanced_raw_data(
        ohlc_data=ohlc_data,
        volume_data=[],
        market_depth=market_depth,
        time_sales=time_sales,
        market_sentiment=market_sentiment
    )
    
    # 保存增强版原始数据
    with open("enhanced_raw_data_test.json", "w") as f:
        # 转换numpy类型为Python原生类型
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            elif hasattr(obj, 'dtype'):  # numpy类型检查
                if obj.dtype == 'bool':
                    return bool(obj)
                elif 'int' in str(obj.dtype):
                    return int(obj)
                elif 'float' in str(obj.dtype):
                    return float(obj)
            return obj
        
        json.dump(convert_numpy_types(enhanced_raw_data), f, indent=2)
    
    # 生成增强版提示词
    enhanced_prompt = enhanced_bot.generate_enhanced_smc_prompt(enhanced_raw_data)
    
    # 保存增强版提示词
    with open("enhanced_smc_prompt_test.txt", "w") as f:
        f.write(enhanced_prompt)
    
    print("✅ 增强版MockBot测试完成")
    print(f"📊 增强版原始数据已保存到 enhanced_raw_data_test.json")
    print(f"📝 增强版提示词已保存到 enhanced_smc_prompt_test.txt")
    print(f"📈 数据包含: {len(enhanced_raw_data['enhanced_candlesticks'])}根增强K线, "
          f"{len(enhanced_raw_data['market_depth'])}个市场深度点, "
          f"{len(enhanced_raw_data['time_sales'])}笔交易记录")
    print(f"📋 提示词长度: {len(enhanced_prompt)} 字符")