#!/usr/bin/env python3
"""
直接询问DeepSeek API，评估数据质量和需求
"""

import json
import requests
from datetime import datetime
from optimized_smc_prompt import get_optimized_smc_prompt

def query_deepseek_about_data():
    """询问DeepSeek关于数据质量和需求"""
    
    # 1. 创建示例数据
    current_price = 42500.0
    
    # 生成模拟原始数据
    candlesticks = []
    for i in range(28):
        timestamp = f"2024-01-{(i%28)+1:02d}T{(i%24):02d}:00:00Z"
        base_price = 42000 + (i * 50)
        candlesticks.append({
            "timestamp": timestamp,
            "open": base_price,
            "high": base_price + 100,
            "low": base_price - 80,
            "close": base_price + 20,
            "volume": 1000 + (i * 50),
            "timeframe": "1h"
        })
    
    # 摆动点数据
    swing_points = [
        {
            "timestamp": "2024-01-05T12:00:00Z",
            "price": 41800.0,
            "type": "swing_low",
            "strength": 0.8
        },
        {
            "timestamp": "2024-01-15T14:00:00Z",
            "price": 43200.0,
            "type": "swing_high",
            "strength": 0.9
        }
    ]
    
    # 成交量数据
    volume_data = []
    for i in range(28):
        timestamp = f"2024-01-{(i%28)+1:02d}T{(i%24):02d}:00:00Z"
        volume = 1000 + (i * 50)
        volume_data.append({
            "timestamp": timestamp,
            "volume": volume,
            "volume_avg": 1200,
            "volume_ratio": volume / 1200
        })
    
    # 流动性水平数据
    liquidity_levels = [
        {
            "price": 41800.0,
            "strength": 0.8,
            "type": "support"
        },
        {
            "price": 43200.0,
            "strength": 0.7,
            "type": "resistance"
        },
        {
            "price": 41500.0,
            "strength": 0.9,
            "type": "support"
        },
        {
            "price": 43500.0,
            "strength": 0.6,
            "type": "resistance"
        }
    ]
    
    # 价格变动数据
    price_movements = []
    for i in range(20):
        timestamp = f"2024-01-{(i%20)+1:02d}T{(i%24):02d}:00:00Z"
        direction = "up" if i % 3 != 0 else "down"
        magnitude = 50 + (i * 5)
        duration = 30 + (i * 5)
        price_movements.append({
            "timestamp": timestamp,
            "direction": direction,
            "magnitude": magnitude,
            "duration": duration
        })
    
    raw_data = {
        "candlesticks": candlesticks,
        "swing_points": swing_points,
        "volume_data": volume_data,
        "liquidity_levels": liquidity_levels,
        "price_movements": price_movements
    }
    
    # 构建市场数据字典
    market_data = {
        'current_price': current_price,
        'symbol': 'BTC/USDT',
        'higher_tf': '4h',
        'higher_tf_trend': 'bullish',
        'higher_tf_strength': 0.7,
        'primary_tf': '15m',
        'primary_tf_trend': 'bullish',
        'primary_tf_strength': 0.6,
        'mtf_consistency': 0.8,
        'rsi': 65.5,
        'macd_histogram': 0.02,
        'volume_ratio': 1.3,
        'volatility': 2.5,
        'min_rr_ratio': 2.5,
        'invalidation_point': current_price * 0.98,
        'nearest_key_level': current_price * 0.985,
        'key_level_distance': 1.5,
        'raw_price_data': raw_data
    }
    
    # 2. 生成提示词
    prompt = get_optimized_smc_prompt(market_data)
    
    # 3. 创建询问数据质量的提示词
    data_quality_prompt = f"""
我是一名交易系统开发者，正在开发一个基于SMC/ICT策略的AI交易助手。

我向AI提供以下类型的原始高颗粒度数据，让AI自行计算SMC结构并生成交易信号：

## 提供的数据类型：

1. **K线数据** (candlesticks):
   - 时间戳、开盘价、最高价、最低价、收盘价、成交量、时间框架
   - 示例: {json.dumps(candlesticks[0], indent=2)}

2. **摆动点数据** (swing_points):
   - 时间戳、价格、类型(swing_high/swing_low)、强度
   - 示例: {json.dumps(swing_points[0], indent=2)}

3. **成交量数据** (volume_data):
   - 时间戳、成交量、平均成交量、成交量比率
   - 示例: {json.dumps(volume_data[0], indent=2)}

4. **流动性水平数据** (liquidity_levels):
   - 价格、强度、类型(support/resistance)
   - 示例: {json.dumps(liquidity_levels[0], indent=2)}

5. **价格变动数据** (price_movements):
   - 时间戳、方向(up/down)、幅度、持续时间
   - 示例: {json.dumps(price_movements[0], indent=2)}

## 问题：

1. 基于这些原始数据，AI是否能够准确计算所有必要的SMC结构(BOS/CHOCH、订单块、FVG、流动性分析)？

2. 这些数据中缺少哪些关键信息，可能会影响SMC结构计算的准确性？

3. 还需要提供什么类型的额外数据，才能帮助AI更好地完成SMC分析和交易信号生成？

4. 对于每种SMC结构计算，哪些数据字段是最关键的？

5. 在当前数据基础上，如何改进数据结构或添加什么新字段来提高AI的分析质量？

请以专业SMC分析师的角度，详细分析数据充分性并提供改进建议。
"""

    # 4. 调用DeepSeek API
    try:
        # 直接使用API密钥
        api_key = "sk-250514fff2f6467a8c0aa2c9c17d2a54"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        data = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "你是一名专业的SMC/ICT交易分析师和数据科学家，擅长评估交易数据质量和完整性。"},
                {"role": "user", "content": data_quality_prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 2000
        }
        
        print("🔍 询问DeepSeek关于数据质量和需求...")
        
        # 添加重试机制
        max_retries = 3
        retry_delay = 2  # 秒
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=60  # 增加超时时间到60秒
                )
                
                if response.status_code == 200:
                    break  # 成功，跳出重试循环
                else:
                    print(f"❌ API请求失败 (尝试 {attempt + 1}/{max_retries}): {response.status_code} - {response.text}")
                    if attempt < max_retries - 1:
                        print(f"⏳ {retry_delay}秒后重试...")
                        import time
                        time.sleep(retry_delay)
            except requests.exceptions.RequestException as e:
                print(f"❌ 请求异常 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    print(f"⏳ {retry_delay}秒后重试...")
                    import time
                    time.sleep(retry_delay)
                else:
                    raise e
        
        if response.status_code == 200:
            result = response.json()
            analysis = result["choices"][0]["message"]["content"]
            
            # 保存分析结果
            with open("deepseek_data_analysis.txt", "w", encoding="utf-8") as f:
                f.write(f"DeepSeek数据质量分析\n")
                f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*50 + "\n\n")
                f.write(analysis)
            
            print("✅ DeepSeek分析完成，结果已保存到 deepseek_data_analysis.txt")
            print("\n" + "="*50)
            print("DeepSeek分析摘要:")
            print("="*50)
            print(analysis)
            
        else:
            print(f"❌ API请求失败: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ 查询DeepSeek失败: {str(e)}")

if __name__ == "__main__":
    query_deepseek_about_data()