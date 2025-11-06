"""
优化版SMC/ICT策略分析提示词
解决原提示词过于复杂、专业判断权限模糊、技术指标限制过严等问题
增强版：提供原始高颗粒度数据，让AI自行计算SMC结构，并提供计算方法和权重说明
"""

def get_optimized_smc_prompt(market_data):
    """
    获取优化后的SMC/ICT策略分析提示词
    
    参数:
        market_data (dict): 包含市场数据的字典
        
    返回:
        str: 优化后的提示词
    """
    
    # 提取关键市场数据
    current_price = market_data.get('current_price', 0)
    symbol = market_data.get('symbol', 'BTC/USD')
    
    # 多时间框架分析
    higher_tf = market_data.get('higher_tf', '4h')
    higher_tf_trend = market_data.get('higher_tf_trend', 'neutral')
    higher_tf_strength = market_data.get('higher_tf_strength', 0.5)
    
    primary_tf = market_data.get('primary_tf', '15m')
    primary_tf_trend = market_data.get('primary_tf_trend', 'neutral')
    primary_tf_strength = market_data.get('primary_tf_strength', 0.5)
    
    mtf_consistency = market_data.get('mtf_consistency', 0.5)
    
    # 技术指标
    rsi = market_data.get('rsi', 50)
    macd_histogram = market_data.get('macd_histogram', 0)
    volume_ratio = market_data.get('volume_ratio', 1.0)
    
    # 风险参数
    volatility = market_data.get('volatility', 2.0)
    min_rr_ratio = market_data.get('min_rr_ratio', 2.0)
    invalidation_point = market_data.get('invalidation_point', current_price * 0.98)
    
    # 关键水平
    nearest_key_level = market_data.get('nearest_key_level', current_price * 0.98)
    key_level_distance = market_data.get('key_level_distance', 0.02)
    
    # 原始高颗粒度数据 - 让AI自行计算SMC结构
    raw_price_data = market_data.get('raw_price_data', {})
    
    # K线数据
    candlesticks = raw_price_data.get('candlesticks', [])
    
    # 高低点数据
    swing_points = raw_price_data.get('swing_points', [])
    
    # 成交量数据
    volume_data = raw_price_data.get('volume_data', [])
    
    # 流动性水平数据
    liquidity_levels = raw_price_data.get('liquidity_levels', [])
    
    # 原始价格变动数据
    price_movements = raw_price_data.get('price_movements', [])
    
    # 构建K线数据摘要
    candlestick_summary = []
    for candle in candlesticks[-10:]:  # 最近10根K线
        timestamp = candle.get('timestamp', '')
        open_price = candle.get('open', 0)
        high_price = candle.get('high', 0)
        low_price = candle.get('low', 0)
        close_price = candle.get('close', 0)
        volume = candle.get('volume', 0)
        candlestick_summary.append(f"{timestamp}: O:{open_price:.2f} H:{high_price:.2f} L:{low_price:.2f} C:{close_price:.2f} V:{volume:.2f}")
    
    # 构建高低点数据摘要
    swing_points_summary = []
    for point in swing_points[-8:]:  # 最近8个高低点
        timestamp = point.get('timestamp', '')
        price = point.get('price', 0)
        point_type = point.get('type', 'unknown')  # high或low
        strength = point.get('strength', 0)
        swing_points_summary.append(f"{timestamp}: {point_type.upper()} {price:.2f} (强度:{strength:.2f})")
    
    # 构建成交量数据摘要
    volume_summary = []
    for vol in volume_data[-10:]:  # 最近10个成交量数据点
        timestamp = vol.get('timestamp', '')
        volume = vol.get('volume', 0)
        volume_avg = vol.get('volume_avg', 0)
        volume_ratio = volume / volume_avg if volume_avg > 0 else 1
        volume_summary.append(f"{timestamp}: {volume:.2f} (比率:{volume_ratio:.2f}x)")
    
    # 构建流动性水平数据摘要
    liquidity_summary = []
    for level in liquidity_levels[:10]:  # 最近10个流动性水平
        price = level.get('price', 0)
        strength = level.get('strength', 0)
        level_type = level.get('type', 'unknown')  # buy_side或sell_side
        liquidity_summary.append(f"{level_type.upper()} {price:.2f} (强度:{strength:.2f})")
    
    # 构建价格变动数据摘要
    price_movement_summary = []
    for movement in price_movements[-10:]:  # 最近10个价格变动
        timestamp = movement.get('timestamp', '')
        direction = movement.get('direction', 'unknown')  # up或down
        magnitude = movement.get('magnitude', 0)
        duration = movement.get('duration', 0)
        price_movement_summary.append(f"{timestamp}: {direction.upper()} {magnitude:.2f} ({duration}分钟)")
    
    prompt = f"""
# SMC/ICT策略分析任务 - 原始数据计算版

你是一名专业的SMC/ICT交易分析师，负责分析{symbol}市场并生成高质量交易信号。
你将获得原始高颗粒度市场数据，需要自行计算SMC结构并基于这些计算结果进行判断。

## 当前市场状况

**价格**: {current_price}
**波动率**: {volatility}%
**最小风险回报比**: {min_rr_ratio}:1

## 多时间框架分析

- {higher_tf}趋势: {higher_tf_trend} (强度: {higher_tf_strength:.2f})
- {primary_tf}趋势: {primary_tf_trend} (强度: {primary_tf_strength:.2f})
- 多时间框架一致性: {mtf_consistency:.2f}

## 原始高颗粒度市场数据

### 最近K线数据 (最近10根)
{chr(10).join(candlestick_summary) if candlestick_summary else "无K线数据"}

### 高低点数据 (最近8个)
{chr(10).join(swing_points_summary) if swing_points_summary else "无高低点数据"}

### 成交量数据 (最近10个)
{chr(10).join(volume_summary) if volume_summary else "无成交量数据"}

### 流动性水平数据 (最近10个)
{chr(10).join(liquidity_summary) if liquidity_summary else "无流动性水平数据"}

### 价格变动数据 (最近10个)
{chr(10).join(price_movement_summary) if price_movement_summary else "无价格变动数据"}

## 技术指标

- RSI: {rsi:.1f}
- MACD柱状图: {macd_histogram:.4f}
- 成交量比率: {volume_ratio:.2f}x

## 关键水平

- 最近关键水平: {nearest_key_level}
- 距离当前价格: {key_level_distance:.2f}%
- 无效点: {invalidation_point}

## SMC结构计算方法与权重说明

### 1. BOS (Break of Structure) 计算方法
**计算规则**:
- 看涨BOS: 价格突破前一个高点，且后续回调不跌破该高点
- 看跌BOS:价格跌破前一个低点，且后续反弹不突破该低点

**权重分配**:
- 高点/低点突破强度: 40%
- 突破后价格稳定性: 30%
- 成交量确认: 20%
- 多时间框架一致性: 10%

### 2. CHOCH (Change of Character) 计算方法
**计算规则**:
- 看涨CHOCH: 在下降趋势中，价格突破前一个重要高点并形成更高高点
- 看跌CHOCH: 在上升趋势中，价格跌破前一个重要低点并形成更低低点

**权重分配**:
- 趋势反转确认: 50%
- 突破幅度: 25%
- 成交量放大: 15%
- 多时间框架一致性: 10%

### 3. 订单块 (Order Block) 计算方法
**计算规则**:
- 看涨订单块: 形成看涨趋势前的最后一根下跌K线区域
- 看跌订单块: 形成看跌趋势前的最后一根上涨K线区域

**权重分配**:
- K线实体大小: 30%
- 成交量异常: 25%
- 价格反应速度: 25%
- 后续测试次数: 20%

### 4. 公允价值缺口 (Fair Value Gap) 计算方法
**计算规则**:
- 识别三根K线间形成的价格缺口
- 缺口大小应超过平均ATR的0.5倍

**权重分配**:
- 缺口大小: 40%
- 形成速度: 30%
- 成交量确认: 20%
- 回补概率: 10%

### 5. 流动性分析计算方法
**计算规则**:
- 识别历史高低点附近的流动性聚集区
- 分析流动性扫荡模式

**权重分配**:
- 流动性集中度: 35%
- 扫荡完成度: 35%
- 价格反应: 20%
- 多时间框架确认: 10%

## 综合信号生成权重

### 买入信号权重分配:
1. 看涨BOS/CHOCH确认: 30%
2. 订单块/FVG回撤区域: 25%
3. 流动性扫荡完成: 20%
4. 多时间框架一致性: 15%
5. 技术指标确认: 10%

### 卖出信号权重分配:
1. 看跌BOS/CHOCH确认: 30%
2. 订单块/FVG反弹区域: 25%
3. 流动性扫荡完成: 20%
4. 多时间框架一致性: 15%
5. 技术指标确认: 10%

## AI专业判断权限

作为专业分析师，你拥有以下判断权限：

1. **原始数据优先分析**:
   - 基于提供的原始数据自行计算所有SMC结构
   - 可以根据市场特性调整计算参数和阈值

2. **动态权重调整**:
   - 在高波动市场中，可提高流动性分析权重
   - 在趋势市场中，可提高BOS/CHOCH权重
   - 在震荡市场中，可提高订单块/FVG权重

3. **结构新鲜度评估**:
   - 根据数据时间戳评估结构新鲜度
   - 新鲜结构(形成<10根K线)权重可提高20%
   - 老旧结构(形成>30根K线)权重可降低30%

4. **多结构相互确认**:
   - 当多个SMC结构相互确认时，信号置信度可提高
   - 单一结构信号需要更严格的技术指标确认

## 输出要求

请以JSON格式返回你的分析结果，包含以下字段：

```json
{{
    "signal": "BUY|SELL|HOLD",
    "entry_price": 具体入场价格,
    "stop_loss": 具体止损价格,
    "take_profit": 具体止盈价格,
    "confidence": "HIGH|MEDIUM|LOW",
    "reason": "详细交易理由，说明基于哪些原始数据计算出的SMC结构",
    "smc_analysis": {{
        "bos": "BOS分析结果",
        "choch": "CHOCH分析结果",
        "order_blocks": "订单块分析结果",
        "fvg": "公允价值缺口分析结果",
        "liquidity": "流动性分析结果"
    }}
}}
```

## 分析重点

1. 基于原始数据准确计算所有SMC结构
2. 根据市场状况动态调整各结构权重
3. 识别多个结构之间的相互确认关系
4. 评估结构新鲜度与价格距离的综合关系
5. 提供基于原始数据计算的清晰、详细的分析理由

基于以上原始高颗粒度数据和SMC结构计算方法，请生成你的交易分析。
"""
    
    return prompt


# 使用示例
if __name__ == "__main__":
    # 示例市场数据 - 包含原始高颗粒度数据
    example_market_data = {
        'current_price': 108500,
        'symbol': 'BTC/USD',
        'higher_tf': '4h',
        'higher_tf_trend': 'bullish',
        'higher_tf_strength': 0.75,
        'primary_tf': '15m',
        'primary_tf_trend': 'bullish',
        'primary_tf_strength': 0.65,
        'mtf_consistency': 0.8,
        'rsi': 65,
        'macd_histogram': 0.002,
        'volume_ratio': 1.3,
        'volatility': 2.5,
        'min_rr_ratio': 2.0,
        'invalidation_point': 107800,
        'nearest_key_level': 108200,
        'key_level_distance': 0.03,
        # 新增：原始高颗粒度数据
        'raw_price_data': {
            'candlesticks': [
                {'timestamp': '2024-01-01T10:00:00', 'open': 108200, 'high': 108600, 'low': 108100, 'close': 108500, 'volume': 1200},
                {'timestamp': '2024-01-01T09:45:00', 'open': 108000, 'high': 108300, 'low': 107900, 'close': 108200, 'volume': 1100},
                {'timestamp': '2024-01-01T09:30:00', 'open': 107800, 'high': 108100, 'low': 107700, 'close': 108000, 'volume': 1300},
                {'timestamp': '2024-01-01T09:15:00', 'open': 107600, 'high': 107900, 'low': 107500, 'close': 107800, 'volume': 1000},
                {'timestamp': '2024-01-01T09:00:00', 'open': 107400, 'high': 107700, 'low': 107300, 'close': 107600, 'volume': 900},
                {'timestamp': '2024-01-01T08:45:00', 'open': 107200, 'high': 107500, 'low': 107100, 'close': 107400, 'volume': 800},
                {'timestamp': '2024-01-01T08:30:00', 'open': 107000, 'high': 107300, 'low': 106900, 'close': 107200, 'volume': 850},
                {'timestamp': '2024-01-01T08:15:00', 'open': 106800, 'high': 107100, 'low': 106700, 'close': 107000, 'volume': 750},
                {'timestamp': '2024-01-01T08:00:00', 'open': 106600, 'high': 106900, 'low': 106500, 'close': 106800, 'volume': 700},
                {'timestamp': '2024-01-01T07:45:00', 'open': 106400, 'high': 106700, 'low': 106300, 'close': 106600, 'volume': 650}
            ],
            'swing_points': [
                {'timestamp': '2024-01-01T10:00:00', 'price': 108600, 'type': 'high', 'strength': 0.8},
                {'timestamp': '2024-01-01T09:30:00', 'price': 107700, 'type': 'low', 'strength': 0.7},
                {'timestamp': '2024-01-01T09:00:00', 'price': 107700, 'type': 'high', 'strength': 0.75},
                {'timestamp': '2024-01-01T08:30:00', 'price': 106900, 'type': 'low', 'strength': 0.65},
                {'timestamp': '2024-01-01T08:00:00', 'price': 106900, 'type': 'high', 'strength': 0.7},
                {'timestamp': '2024-01-01T07:30:00', 'price': 106300, 'type': 'low', 'strength': 0.6},
                {'timestamp': '2024-01-01T07:00:00', 'price': 106300, 'type': 'high', 'strength': 0.65},
                {'timestamp': '2024-01-01T06:30:00', 'price': 105700, 'type': 'low', 'strength': 0.55}
            ],
            'volume_data': [
                {'timestamp': '2024-01-01T10:00:00', 'volume': 1200, 'volume_avg': 1000},
                {'timestamp': '2024-01-01T09:45:00', 'volume': 1100, 'volume_avg': 950},
                {'timestamp': '2024-01-01T09:30:00', 'volume': 1300, 'volume_avg': 900},
                {'timestamp': '2024-01-01T09:15:00', 'volume': 1000, 'volume_avg': 850},
                {'timestamp': '2024-01-01T09:00:00', 'volume': 900, 'volume_avg': 800},
                {'timestamp': '2024-01-01T08:45:00', 'volume': 800, 'volume_avg': 750},
                {'timestamp': '2024-01-01T08:30:00', 'volume': 850, 'volume_avg': 700},
                {'timestamp': '2024-01-01T08:15:00', 'volume': 750, 'volume_avg': 650},
                {'timestamp': '2024-01-01T08:00:00', 'volume': 700, 'volume_avg': 600},
                {'timestamp': '2024-01-01T07:45:00', 'volume': 650, 'volume_avg': 550}
            ],
            'liquidity_levels': [
                {'price': 109500, 'strength': 0.9, 'type': 'sell_side'},
                {'price': 109200, 'strength': 0.8, 'type': 'sell_side'},
                {'price': 108800, 'strength': 0.7, 'type': 'sell_side'},
                {'price': 108600, 'strength': 0.8, 'type': 'sell_side'},
                {'price': 108300, 'strength': 0.6, 'type': 'sell_side'},
                {'price': 108000, 'strength': 0.7, 'type': 'buy_side'},
                {'price': 107700, 'strength': 0.8, 'type': 'buy_side'},
                {'price': 107400, 'strength': 0.6, 'type': 'buy_side'},
                {'price': 107000, 'strength': 0.7, 'type': 'buy_side'},
                {'price': 106700, 'strength': 0.8, 'type': 'buy_side'}
            ],
            'price_movements': [
                {'timestamp': '2024-01-01T10:00:00', 'direction': 'up', 'magnitude': 300, 'duration': 15},
                {'timestamp': '2024-01-01T09:45:00', 'direction': 'up', 'magnitude': 200, 'duration': 15},
                {'timestamp': '2024-01-01T09:30:00', 'direction': 'up', 'magnitude': 200, 'duration': 15},
                {'timestamp': '2024-01-01T09:15:00', 'direction': 'up', 'magnitude': 200, 'duration': 15},
                {'timestamp': '2024-01-01T09:00:00', 'direction': 'up', 'magnitude': 200, 'duration': 15},
                {'timestamp': '2024-01-01T08:45:00', 'direction': 'up', 'magnitude': 200, 'duration': 15},
                {'timestamp': '2024-01-01T08:30:00', 'direction': 'up', 'magnitude': 200, 'duration': 15},
                {'timestamp': '2024-01-01T08:15:00', 'direction': 'up', 'magnitude': 200, 'duration': 15},
                {'timestamp': '2024-01-01T08:00:00', 'direction': 'up', 'magnitude': 200, 'duration': 15},
                {'timestamp': '2024-01-01T07:45:00', 'direction': 'up', 'magnitude': 200, 'duration': 15}
            ]
        }
    }
    
    # 生成优化后的提示词
    optimized_prompt = get_optimized_smc_prompt(example_market_data)
    print(optimized_prompt)