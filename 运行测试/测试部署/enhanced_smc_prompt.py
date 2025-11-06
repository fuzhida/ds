#!/usr/bin/env python3
"""
增强版提示词生成器 - 基于DeepSeek建议的改进
提供更详细的SMC计算方法和数据使用说明
"""

import json
from datetime import datetime
from typing import Dict, Any, List

def get_enhanced_smc_prompt(market_data: Dict[str, Any]) -> str:
    """
    生成增强版SMC分析提示词，包含详细的计算方法和数据使用说明
    
    参数:
        market_data: 包含增强版原始数据的字典
        
    返回:
        增强版SMC分析提示词
    """
    
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
    
    # 转换所有数据
    market_data = convert_numpy_types(market_data)
    
    # 提取关键数据
    enhanced_candlesticks = market_data.get('enhanced_candlesticks', [])
    swing_points = market_data.get('swing_points', [])
    volume_analysis = market_data.get('volume_analysis', [])
    market_depth = market_data.get('market_depth', [])
    time_sales = market_data.get('time_sales', [])
    market_sentiment = market_data.get('market_sentiment', {})
    multi_timeframe_context = market_data.get('multi_timeframe_context', {})
    liquidity_levels = market_data.get('liquidity_levels', [])
    price_movements = market_data.get('price_movements', [])
    order_flow_imbalance = market_data.get('order_flow_imbalance', {})
    market_microstructure = market_data.get('market_microstructure', {})
    
    # 获取当前价格
    current_price = enhanced_candlesticks[-1]['close'] if enhanced_candlesticks else 0
    
    prompt = f"""# SMC/ICT策略分析 - 增强版数据驱动分析

## 任务概述
基于提供的增强版原始市场数据，进行专业SMC/ICT（Smart Money Concepts/Inner Circle Trader）结构分析，识别关键交易机会。你拥有完全自主权，不受任何预设偏见影响，必须基于数据做出独立判断。

## 增强版原始市场数据

### 1. 增强版K线数据 (Enhanced Candlesticks)
```json
{json.dumps(enhanced_candlesticks[-20:], indent=2)}
```

**数据字段说明:**
- `body_size`: K线实体大小，表示开盘价与收盘价之间的绝对距离
- `upper_wick`: 上影线长度，表示最高价与实体顶部的距离
- `lower_wick`: 下影线长度，表示实体底部与最低价的距离
- `body_position`: 实体位置（upper/middle/lower/doji），反映收盘价在K线中的相对位置
- `body_ratio`: 实体比例，实体大小占总波动的比例（0-1）
- `gap_size`: 跳空大小，与前一收盘价的绝对距离
- `gap_direction`: 跳空方向（up/down/none）
- `volume_profile`: 成交量分布，包含POC（价格控制点）和价值区域
- `engulfing`: 吞没形态（bullish_engulfing/bearish_engulfing/none）
- `rejection`: 拒绝形态，当影线长度是实体长度的2倍以上时为True
- `inside_bar`: 内包线，当前K线完全包含在前一根K线内

### 2. 摆动点数据 (Swing Points)
```json
{json.dumps(swing_points[-10:], indent=2)}
```

**数据字段说明:**
- `strength`: 摆动点强度（0-10），基于相对高度计算
- `confirmed`: 是否已确认，True表示已形成完整结构

### 3. 增强版成交量分析 (Enhanced Volume Analysis)
```json
{json.dumps(volume_analysis[-20:], indent=2)}
```

**数据字段说明:**
- `volume_ratio`: 当前成交量与平均成交量的比率
- `volume_spike`: 成交量异常标记，当比率>2时为True
- `volume_price_efficiency`: 成交量价格效率，价格变动与成交量的关系
- `buying_pressure`: 买压估算（0-1），基于收盘位置和影线长度
- `volume_trend`: 成交量趋势（increasing/decreasing/stable）

### 4. 市场深度数据 (Market Depth)
```json
{json.dumps(market_depth[-10:], indent=2)}
```

**数据字段说明:**
- `imbalance_ratio`: 买卖不平衡比率，买量/卖量
- `spread_percentage`: 价差百分比
- `dominant_side`: 主导方（bid/ask）
- `liquidity_score`: 流动性得分，总成交量/价差

### 5. 时间与销售数据 (Time & Sales)
```json
{json.dumps(time_sales[-20:], indent=2)}
```

**数据字段说明:**
- `side`: 交易方向（buy/sell）
- `liquidity_removed`: 是否移除流动性
- `aggressive`: 是否主动成交
- `large_order`: 是否大单（成交量>10）

### 6. 市场情绪数据 (Market Sentiment)
```json
{json.dumps(market_sentiment, indent=2)}
```

### 7. 多时间框架上下文 (Multi-Timeframe Context)
```json
{json.dumps(multi_timeframe_context, indent=2)}
```

**数据字段说明:**
- `alignment_score`: 趋势一致性得分（0-1）
- `key_levels_aligned`: 关键水平是否对齐
- `trend_strength`: 趋势强度（0-1）

### 8. 增强版流动性水平 (Enhanced Liquidity Levels)
```json
{json.dumps(liquidity_levels[-10:], indent=2)}
```

**数据字段说明:**
- `strength`: 流动性强度（0-10），结合摆动点强度和成交量
- `tested`: 是否已被测试
- `volume_confirmation`: 成交量确认

### 9. 增强版价格变动数据 (Enhanced Price Movements)
```json
{json.dumps(price_movements[-20:], indent=2)}
```

**数据字段说明:**
- `strength`: 变动强度，结合价格变动和波动率
- `volume_confirmation`: 成交量确认
- `gap_pct`: 跳空百分比
- `volatility`: 波动率

### 10. 订单流不平衡 (Order Flow Imbalance)
```json
{json.dumps(order_flow_imbalance, indent=2)}
```

**数据字段说明:**
- `imbalance_ratio`: 买卖不平衡比率
- `dominant_side`: 主导方（buy/sell）
- `large_order_ratio`: 大单比例

### 11. 市场微观结构 (Market Microstructure)
```json
{json.dumps(market_microstructure, indent=2)}
```

**数据字段说明:**
- `price_efficiency`: 价格效率（0-1），实际位移与总路径的比率
- `liquidity_distribution`: 流动性分布
- `market_pressure`: 市场压力（0-1）
- `microstructure_score`: 微观结构得分

## SMC结构计算方法详解

### 1. BOS (Break of Structure) 计算方法
**定义**: 价格突破前一波段高点或低点，标志着市场结构改变

**计算步骤**:
1. 识别最近的摆动高点(SH)和摆动低点(SL)
2. 检查当前价格是否突破SH或SL
3. 确认突破的有效性：
   - 突破K线收盘价必须超过SH/SL
   - 突破时成交量应高于平均水平(volume_ratio > 1.2)
   - 突破K线body_ratio应 > 0.5，表示强劲突破
4. 评估突破强度：
   - 强突破: volume_ratio > 2.0 且 body_ratio > 0.7
   - 中等突破: 1.5 < volume_ratio ≤ 2.0 且 0.4 < body_ratio ≤ 0.7
   - 弱突破: 1.2 < volume_ratio ≤ 1.5 且 0.3 < body_ratio ≤ 0.4

**数据使用**:
- 使用enhanced_candlesticks中的body_size和body_ratio评估突破强度
- 使用volume_analysis中的volume_ratio确认成交量支持
- 使用swing_points确定需要突破的关键水平

### 2. CHOCH (Change of Character) 计算方法
**定义**: 市场从趋势状态转为盘整状态，或从盘整转为趋势状态

**计算步骤**:
1. 识别当前市场状态（趋势/盘整）
2. 寻找状态改变的信号：
   - 趋势转盘整: 连续3根K线无法创新高/新低，且body_ratio < 0.3
   - 盘整转趋势: 价格突破盘整区间，且volume_spike为True
3. 确认状态改变：
   - 盘整区间由至少5根K线形成
   - 突破时buying_pressure > 0.6（向上突破）或 < 0.4（向下突破）
4. 评估CHOCH强度：
   - 强信号: volume_spike为True且alignment_score > 0.8
   - 中等信号: volume_spike为True或alignment_score > 0.6
   - 弱信号: alignment_score > 0.4

**数据使用**:
- 使用enhanced_candlesticks中的body_position和body_ratio判断市场状态
- 使用volume_analysis中的volume_spike确认突破动力
- 使用multi_timeframe_context中的alignment_score评估多时间框架一致性

### 3. 订单块 (Order Block) 计算方法
**定义**: 机构订单集中的区域，通常在强烈价格变动前的最后一根K线

**计算步骤**:
1. 识别强烈价格变动（price_movements中strength > 0.7）
2. 定位强烈变动前的K线（通常是1-3根前）
3. 确认订单块特征：
   - K线body_ratio > 0.6，表示强烈意图
   - 成交量高于平均水平(volume_ratio > 1.5)
   - K线engulfing为bullish_engulfing或bearish_engulfing
4. 评估订单块强度：
   - 强订单块: body_ratio > 0.8 且 volume_ratio > 2.0
   - 中等订单块: 0.6 < body_ratio ≤ 0.8 且 1.5 < volume_ratio ≤ 2.0
   - 弱订单块: 0.4 < body_ratio ≤ 0.6 且 1.2 < volume_ratio ≤ 1.5

**数据使用**:
- 使用enhanced_candlesticks中的body_ratio和engulfing识别潜在订单块
- 使用volume_analysis中的volume_ratio确认订单强度
- 使用price_movements定位强烈价格变动

### 4. FVG (Fair Value Gap) 计算方法
**定义**: 价格快速移动时留下的不平衡区域，通常由三根K线形成

**计算步骤**:
1. 寻找三根连续K线，其中中间K线高点低于第一根K线低点，或中间K线低点高于第一根K线高点
2. 计算FVG范围：
   - 看涨FVG: 第二根K线高点到第一根K线低点之间的区域
   - 看跌FVG: 第二根K线低点到第一根K线高点之间的区域
3. 确认FVG有效性：
   - 形成FVG的K线中至少有一根volume_spike为True
   - FVG大小应大于平均ATR的0.5倍
4. 评估FVG强度：
   - 强FVG: 形成时volume_spike为True且gap_size > 平均ATR
   - 中等FVG: 形成时volume_ratio > 1.5且gap_size > 0.5*平均ATR
   - 弱FVG: volume_ratio > 1.2且gap_size > 0.3*平均ATR

**数据使用**:
- 使用enhanced_candlesticks中的gap_size和gap_direction识别潜在FVG
- 使用volume_analysis中的volume_spike确认FVG强度
- 使用price_movements中的volatility计算平均ATR

### 5. 流动性分析计算方法
**定义**: 识别市场中的流动性集中区域，这些区域可能成为价格目标

**计算步骤**:
1. 识别流动性水平（liquidity_levels中strength > 7）
2. 分析流动性特征：
   - 未测试的流动性水平(tested为False)更具吸引力
   - 高成交量确认的流动性水平(volume_confirmation > 平均值)更可靠
3. 评估流动性梯度：
   - 流动性梯度计算: 相邻流动性水平之间的strength差异
   - 陡峭梯度表明强阻力/支撑区域
4. 确认流动性清除：
   - 使用time_sales数据检查是否有大单在流动性水平附近成交
   - 使用market_depth数据检查流动性水平附近的订单不平衡

**数据使用**:
- 使用liquidity_levels中的strength和tested识别关键流动性区域
- 使用market_depth中的imbalance_ratio确认流动性分布
- 使用time_sales中的large_order检查流动性清除情况

## 增强版数据使用指南

### 1. 市场深度数据使用
- **不平衡分析**: 当imbalance_ratio > 2或 < 0.5时，表明强烈单边压力
- **流动性评估**: liquidity_score > 1000表示高流动性区域
- **价差分析**: spread_percentage < 0.05表示低延迟市场环境

### 2. 时间与销售数据使用
- **大单监控**: large_order为True的交易值得关注
- **主动成交**: aggressive为True表明市场参与者急切
- **流动性清除**: liquidity_removed为True表明关键水平被测试

### 3. 市场情绪数据使用
- **恐惧贪婪指数**: > 75表示贪婪，< 25表示恐惧
- **资金费率**: 正值表示多头主导，负值表示空头主导
- **多空比例**: > 1.2表示多头过多，< 0.8表示空头过多

### 4. 多时间框架上下文使用
- **趋势一致性**: alignment_score > 0.8表示多时间框架趋势一致
- **关键水平对齐**: key_levels_aligned为True时，支撑/阻力更可靠
- **趋势强度**: trend_strength > 0.7表示强趋势

## 综合信号生成与权重分配

### 1. 信号权重分配（总计100%）
- **BOS/CHOCH结构**: 30%（基础结构确认）
- **订单块确认**: 25%（机构活动区域）
- **FVG确认**: 15%（价格不平衡区域）
- **流动性分析**: 15%（流动性梯度与清除）
- **市场微观结构**: 10%（价格效率与压力）
- **市场情绪**: 5%（情绪极端指标）

### 2. 信号强度计算
每个信号类型根据以下标准评分（0-10分）：

#### BOS/CHOCH评分标准:
- 10分: 强突破(volume_ratio > 2.0, body_ratio > 0.7)且多时间框架一致
- 8分: 中等突破(1.5 < volume_ratio ≤ 2.0, 0.4 < body_ratio ≤ 0.7)且多时间框架部分一致
- 6分: 弱突破(1.2 < volume_ratio ≤ 1.5, 0.3 < body_ratio ≤ 0.4)
- 4分: 仅突破但无成交量支持
- 2分: 潜在突破但未确认

#### 订单块评分标准:
- 10分: 强订单块(body_ratio > 0.8, volume_ratio > 2.0)且未被测试
- 8分: 中等订单块(0.6 < body_ratio ≤ 0.8, 1.5 < volume_ratio ≤ 2.0)且部分测试
- 6分: 弱订单块(0.4 < body_ratio ≤ 0.6, 1.2 < volume_ratio ≤ 2.0)且已被测试
- 4分: 潜在订单块但特征不完整
- 2分: 可疑订单块

#### FVG评分标准:
- 10分: 强FVG(volume_spike为True, gap_size > 平均ATR)且未填充
- 8分: 中等FVG(volume_ratio > 1.5, gap_size > 0.5*平均ATR)且部分填充
- 6分: 弱FVG(volume_ratio > 1.2, gap_size > 0.3*平均ATR)且已填充
- 4分: 潜在FVG但特征不完整
- 2分: 可疑FVG

#### 流动性分析评分标准:
- 10分: 强流动性水平(strength > 9)且陡峭梯度且大单清除
- 8分: 中等流动性水平(7 < strength ≤ 9)且中等梯度
- 6分: 弱流动性水平(5 < strength ≤ 7)且平缓梯度
- 4分: 流动性水平但无梯度
- 2分: 可疑流动性水平

#### 市场微观结构评分标准:
- 10分: 高价格效率(> 0.8)且低市场压力(< 0.3)且高流动性得分
- 8分: 中等价格效率(0.6-0.8)且中等市场压力(0.3-0.7)
- 6分: 低价格效率(0.4-0.6)且高市场压力(> 0.7)
- 4分: 极低价格效率(< 0.4)
- 2分: 市场微观结构混乱

#### 市场情绪评分标准:
- 10分: 极端情绪(恐惧贪婪指数 > 80或 < 20)且资金费率极值(> 0.1%或 < -0.1%)
- 8分: 高情绪(恐惧贪婪指数 70-80或 20-30)且资金费率高值(0.05%-0.1%或 -0.05%至-0.1%)
- 6分: 中等情绪(恐惧贪婪指数 60-70或 30-40)
- 4分: 低情绪(恐惧贪婪指数 40-60)
- 2分: 中性情绪

### 3. 综合信号计算
```
综合信号强度 = (BOS评分 × 30% + 订单块评分 × 25% + FVG评分 × 15% + 
               流动性评分 × 15% + 微观结构评分 × 10% + 情绪评分 × 5%) / 10
```

### 4. 交易决策阈值
- **强烈买入**: 综合信号强度 > 8.0
- **买入**: 7.0 < 综合信号强度 ≤ 8.0
- **持有**: 5.0 < 综合信号强度 ≤ 7.0
- **卖出**: 3.0 < 综合信号强度 ≤ 5.0
- **强烈卖出**: 综合信号强度 ≤ 3.0

## AI专业判断权限

你拥有完全自主权，必须基于数据做出独立判断，不受任何预设偏见影响。具体权限包括：

1. **结构识别自主权**: 完全自主决定哪些价格行为构成有效SMC结构
2. **权重调整自主权**: 可根据市场条件调整各信号类型权重(±20%范围内)
3. **时间框架优先权**: 自主决定哪个时间框架的结构应优先考虑
4. **异常处理自主权**: 当数据不完整或矛盾时，自主决定最佳处理方式
5. **风险控制自主权**: 自主评估信号可靠性并调整仓位建议

## 输出要求

请基于以上增强版数据和分析方法，提供以下格式的JSON分析结果：

```json
{{
  "market_analysis": {{
    "current_price": {current_price},
    "market_phase": "趋势/盘整/转换",
    "trend_direction": "上升/下降/横盘",
    "key_levels": {{
      "support": [价格1, 价格2, ...],
      "resistance": [价格1, 价格2, ...]
    }},
    "market_efficiency": 0.0-1.0,
    "liquidity_analysis": {{
      "major_zones": [{{"price": 价格, "strength": 0-10, "type": "support/resistance"}}, ...],
      "liquidity_gradient": "steep/moderate/gentle",
      "cleared_levels": [价格1, 价格2, ...]
    }}
  }},
  "smc_structures": {{
    "bos_signals": [{{"price": 价格, "strength": 0-10, "timeframe": "时间框架", "confirmation": "强/中/弱"}}, ...],
    "choch_signals": [{{"price": 价格, "strength": 0-10, "timeframe": "时间框架", "confirmation": "强/中/弱"}}, ...],
    "order_blocks": [{{"price": 价格, "strength": 0-10, "type": "bullish/bearish", "tested": true/false, "volume_confirmation": true/false}}, ...],
    "fvg_zones": [{{"top": 价格, "bottom": 价格, "strength": 0-10, "type": "bullish/bearish", "filled_percentage": 0-100}}, ...]
  }},
  "signal_strength": {{
    "overall_score": 0.0-10.0,
    "component_scores": {{
      "bos_choch": 0.0-10.0,
      "order_blocks": 0.0-10.0,
      "fvg": 0.0-10.0,
      "liquidity": 0.0-10.0,
      "microstructure": 0.0-10.0,
      "sentiment": 0.0-10.0
    }},
    "weight_adjustments": {{
      "bos_choch_weight": "默认30%或调整值",
      "order_blocks_weight": "默认25%或调整值",
      "fvg_weight": "默认15%或调整值",
      "liquidity_weight": "默认15%或调整值",
      "microstructure_weight": "默认10%或调整值",
      "sentiment_weight": "默认5%或调整值"
    }}
  }},
  "trading_recommendation": {{
    "action": "强烈买入/买入/持有/卖出/强烈卖出",
    "confidence": 0.0-1.0,
    "entry_price": 价格,
    "stop_loss": 价格,
    "targets": [价格1, 价格2, ...],
    "position_size": "大/中/小",
    "risk_reward_ratio": 数值,
    "time_horizon": "短期/中期/长期",
    "key_reasons": ["原因1", "原因2", ...],
    "risk_factors": ["风险1", "风险2", ...]
  }},
  "data_quality_assessment": {{
    "completeness": 0.0-1.0,
    "reliability": 0.0-1.0,
    "timeliness": 0.0-1.0,
    "missing_elements": ["缺失元素1", "缺失元素2", ...],
    "confidence_adjustment": "建议调整幅度"
  }},
  "professional_judgment": {{
    "primary_thesis": "主要分析论点",
    "alternative_scenarios": ["备选情景1", "备选情景2"],
    "critical_levels": [{{"price": 价格, "reason": "原因", "importance": "高/中/低"}}, ...],
    "market_biases": "识别的市场偏见",
    "confidence_level": 0.0-1.0
  }}
}}
```

## 重要提醒

1. **数据驱动**: 所有结论必须基于提供的数据，不得凭空猜测
2. **结构优先**: SMC结构识别是分析基础，必须首先确认
3. **多维度验证**: 至少3个不同维度的信号确认才可形成强交易建议
4. **风险意识**: 明确指出所有潜在风险和不确定性
5. **专业判断**: 运用你的专业知识评估数据质量和信号可靠性
6. **透明度**: 清晰说明每个结论的数据依据和推理过程
7. **权重灵活性**: 根据市场条件调整权重，但需说明调整理由

请基于以上增强版数据和分析方法，提供专业、客观、数据驱动的SMC/ICT分析结果。"""

    return prompt

# 使用示例
if __name__ == "__main__":
    # 生成示例市场数据
    import random
    
    # 生成增强版K线数据
    enhanced_candlesticks = []
    base_price = 42000
    for i in range(20):
        timestamp = f"2024-01-{(i%30)+1:02d}T{(i%24):02d}:00:00Z"
        open_price = base_price + random.uniform(-100, 100)
        close_price = open_price + random.uniform(-50, 50)
        high_price = max(open_price, close_price) + random.uniform(0, 50)
        low_price = min(open_price, close_price) - random.uniform(0, 50)
        volume = random.uniform(800, 1500)
        
        # 计算增强字段
        body_size = abs(close_price - open_price)
        upper_wick = high_price - max(open_price, close_price)
        lower_wick = min(open_price, close_price) - low_price
        total_range = high_price - low_price
        
        enhanced_candlesticks.append({
            "timestamp": timestamp,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume,
            "timeframe": "1h",
            "body_size": body_size,
            "upper_wick": upper_wick,
            "lower_wick": lower_wick,
            "body_position": random.choice(["upper", "middle", "lower"]),
            "body_ratio": body_size / total_range if total_range > 0 else 0,
            "previous_close": base_price,
            "gap_size": abs(open_price - base_price),
            "gap_direction": "up" if open_price > base_price else "down" if open_price < base_price else "none",
            "volume_profile": {
                "poc_price": close_price,
                "value_area_high": high_price * 0.99,
                "value_area_low": low_price * 1.01,
                "value_area_volume_pct": 0.7
            },
            "engulfing": random.choice(["bullish_engulfing", "bearish_engulfing", "none"]),
            "rejection": random.choice([True, False]),
            "inside_bar": random.choice([True, False])
        })
        
        base_price = close_price
    
    # 生成摆动点数据
    swing_points = []
    for i in range(5):
        swing_points.append({
            "timestamp": f"2024-01-{(i*5+1):02d}T12:00:00Z",
            "price": 42000 + random.uniform(-500, 500),
            "type": random.choice(["swing_high", "swing_low"]),
            "strength": random.uniform(5, 10),
            "confirmed": True
        })
    
    # 生成成交量分析数据
    volume_analysis = []
    for candle in enhanced_candlesticks:
        volume_analysis.append({
            "timestamp": candle["timestamp"],
            "volume": candle["volume"],
            "volume_avg": random.uniform(900, 1100),
            "volume_ratio": random.uniform(0.8, 2.5),
            "volume_spike": random.choice([True, False]),
            "volume_spike_magnitude": random.uniform(1.5, 3.0),
            "volume_price_efficiency": random.uniform(0.01, 0.1),
            "buying_pressure": random.uniform(0.3, 0.8),
            "volume_trend": random.choice(["increasing", "decreasing", "stable"])
        })
    
    # 生成市场深度数据
    market_depth = []
    for i in range(5):
        mid_price = 42000 + i * 10
        bid_volume = random.uniform(100, 500)
        ask_volume = random.uniform(100, 500)
        market_depth.append({
            "timestamp": f"2024-01-01T{(i%24):02d}:00:00Z",
            "bid_price": mid_price - 5,
            "ask_price": mid_price + 5,
            "bid_volume": bid_volume,
            "ask_volume": ask_volume,
            "total_volume": bid_volume + ask_volume,
            "imbalance_ratio": bid_volume / ask_volume if ask_volume > 0 else float('inf'),
            "spread": 10,
            "spread_percentage": 0.02,
            "dominant_side": "bid" if bid_volume > ask_volume else "ask",
            "liquidity_score": (bid_volume + ask_volume) / 10
        })
    
    # 生成时间与销售数据
    time_sales = []
    for i in range(10):
        time_sales.append({
            "timestamp": f"2024-01-01T{(i%24):02d}:{(i%60):02d}:{(i%60):02d}",
            "price": 42000 + random.uniform(-100, 100),
            "volume": random.uniform(0.1, 20),
            "side": random.choice(["buy", "sell"]),
            "liquidity_removed": random.choice([True, False]),
            "aggressive": random.choice([True, False]),
            "large_order": random.choice([True, False])
        })
    
    # 生成市场情绪数据
    market_sentiment = {
        "fear_greed_index": random.uniform(0, 100),
        "funding_rate": random.uniform(-0.01, 0.01),
        "open_interest_change": random.uniform(-5, 5),
        "long_short_ratio": random.uniform(0.8, 1.5)
    }
    
    # 生成多时间框架上下文
    multi_timeframe_context = {
        "short_term_trend": random.choice(["bullish", "bearish", "neutral"]),
        "long_term_trend": random.choice(["bullish", "bearish", "neutral"]),
        "alignment_score": random.uniform(0.3, 1.0),
        "key_levels_aligned": random.choice([True, False]),
        "trend_strength": random.uniform(0.3, 1.0)
    }
    
    # 生成流动性水平数据
    liquidity_levels = []
    for i in range(5):
        liquidity_levels.append({
            "price": 42000 + random.uniform(-500, 500),
            "strength": random.uniform(5, 10),
            "type": random.choice(["support", "resistance"]),
            "timestamp": f"2024-01-{(i*5+1):02d}T12:00:00Z",
            "tested": random.choice([True, False]),
            "volume_confirmation": random.uniform(800, 1200)
        })
    
    # 生成价格变动数据
    price_movements = []
    for i in range(1, len(enhanced_candlesticks)):
        prev_candle = enhanced_candlesticks[i-1]
        curr_candle = enhanced_candlesticks[i]
        
        price_change = curr_candle['close'] - prev_candle['close']
        price_change_pct = (price_change / prev_candle['close']) * 100
        high_low_range = curr_candle['high'] - curr_candle['low']
        
        price_movements.append({
            "timestamp": curr_candle['timestamp'],
            "direction": "up" if price_change > 0 else "down" if price_change < 0 else "sideways",
            "magnitude": abs(price_change_pct),
            "duration": 60,
            "strength": abs(price_change_pct) / (high_low_range / prev_candle['close'] * 100) if high_low_range > 0 else 0,
            "volume_confirmation": curr_candle['volume'] / prev_candle['volume'] if prev_candle['volume'] > 0 else 1,
            "gap": curr_candle['open'] - prev_candle['close'],
            "gap_pct": ((curr_candle['open'] - prev_candle['close']) / prev_candle['close']) * 100,
            "volatility": high_low_range / prev_candle['close'] * 100
        })
    
    # 生成订单流不平衡数据
    order_flow_imbalance = {
        "imbalance_ratio": random.uniform(0.5, 2.0),
        "dominant_side": random.choice(["buy", "sell"]),
        "buy_volume": random.uniform(5000, 10000),
        "sell_volume": random.uniform(5000, 10000),
        "total_volume": random.uniform(10000, 20000),
        "large_order_ratio": random.uniform(0.1, 0.3),
        "aggressive_buy_ratio": random.uniform(0.3, 0.7)
    }
    
    # 生成市场微观结构数据
    market_microstructure = {
        "price_efficiency": random.uniform(0.3, 0.9),
        "liquidity_distribution": {
            "bid_liquidity": random.uniform(5000, 10000),
            "ask_liquidity": random.uniform(5000, 10000),
            "liquidity_ratio": random.uniform(0.8, 1.2),
            "liquidity_imbalance": random.uniform(0.1, 0.3)
        },
        "market_pressure": random.uniform(0.2, 0.8),
        "microstructure_score": random.uniform(0.3, 0.8)
    }
    
    # 构建市场数据字典
    market_data = {
        "enhanced_candlesticks": enhanced_candlesticks,
        "swing_points": swing_points,
        "volume_analysis": volume_analysis,
        "market_depth": market_depth,
        "time_sales": time_sales,
        "market_sentiment": market_sentiment,
        "multi_timeframe_context": multi_timeframe_context,
        "liquidity_levels": liquidity_levels,
        "price_movements": price_movements,
        "order_flow_imbalance": order_flow_imbalance,
        "market_microstructure": market_microstructure
    }
    
    # 生成增强版提示词
    enhanced_prompt = get_enhanced_smc_prompt(market_data)
    
    # 保存提示词
    with open("enhanced_smc_prompt.txt", "w") as f:
        f.write(enhanced_prompt)
    
    print("✅ 增强版SMC提示词已生成并保存到 enhanced_smc_prompt.txt")
    print(f"📝 提示词长度: {len(enhanced_prompt)} 字符")
    print(f"📊 数据包含: {len(enhanced_candlesticks)}根增强K线, {len(swing_points)}个摆动点, "
          f"{len(market_depth)}个市场深度点, {len(time_sales)}笔交易记录")