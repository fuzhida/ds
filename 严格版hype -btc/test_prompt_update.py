#!/usr/bin/env python3
"""
测试DeepSeek API请求，验证更新后的提示词是否正常工作
"""

import os
import sys
import json
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv('1.env')

# 初始化DeepSeek客户端
deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
if not deepseek_api_key:
    print("错误: DEEPSEEK_API_KEY 环境变量未设置")
    sys.exit(1)

deepseek_client = OpenAI(
    api_key=deepseek_api_key,
    base_url="https://api.deepseek.com"
)

def test_updated_prompt():
    """测试更新后的提示词"""
    print("测试更新后的DeepSeek API提示词...")
    
    # 模拟交易数据
    current_price = 103956.00
    kline_summary = "Last 3 15m: ↑+0.2% ↓-0.1% ↑+0.3%"
    position_text = "No pos"
    signal_text = ""
    activation_text = "Scheduled"
    smc_summary = """
SMC Summary:
- EMA21: 103,850 | EMA55: 103,700 | EMA100: 103,600
- Key Levels: Support at 103,800, Resistance at 104,200
- FVG: 103,900-103,950 (5m), 103,700-103,750 (15m)
- Order Blocks: Bullish OB at 103,750, Bearish OB at 104,100
- Structure: Higher highs and higher lows (uptrend)
"""
    rules = """Rules:
1. Higher TF Bias (1h): Determine direction. Lower TF Entry (5m): Precise entry on confirmation. Require alignment.
2. SMC analysis based on price action and structure
3. Level activation (threshold: 0.020% - reference only) is context, not requirement
4. Confirmations: Volume >1.5x MA, Candle patterns (e.g., Engulfing), FVG stacking >= 3. Fresh zones only (interactions <= 1).
5. R:R >2:1 for HIGH confidence
6. TP within $3542 range"""
    
    # 核心价格行为组件详细说明
    price_action_components = """
CORE PRICE ACTION COMPONENTS:

1. MARKET STRUCTURE:
   - HH+HL=Uptrend (Higher Highs + Higher Lows)
   - LH+LL=Downtrend (Lower Highs + Lower Lows)
   - Structure Breaks: Key S/R level breaches

2. KEY LEVELS:
   - Support: Previous lows, demand zones, order blocks
   - Resistance: Previous highs, supply zones, order blocks
   - Break & Retest: Post-break confirmation

3. CANDLE PATTERNS:
   - Engulfing: Bullish/Bearish engulfing patterns
   - Hammer/Hanging Man: Reversal signals
   - Inside/Outside Bar: Volatility contraction/expansion
   - Fakeouts: Quick reversal after breakout

4. ORDER FLOW CONCEPTS (SMC):
   - FVG (Fair Value Gap): Price gaps between consecutive candles
   - Liquidity Grab: Stop hunts, liquidity pools
   - OB (Order Blocks): Institutional order concentration zones
   - MIT (Market Imbalance Transition): S/R role reversal

5. MOMENTUM ANALYSIS:
   - Candle Size: Reflects buying/selling pressure
   - Close Position: Candle close at high/low end
   - Volatility Changes: Sudden volatility expansion

TRADING FRAMEWORK APPLICATION:
Focus on:
1. 1H chart for primary structure direction
2. 5m chart for:
   - Reactions at key levels (EMA21, previous H/L)
   - Price action in FVG stack areas
   - Candle patterns with volume confirmation
   - First interaction with fresh zones
"""
    
    prompt = f"""SMC/ICT Analysis for BTC/USDC:USDC:Price: ${current_price:,.0f} | {kline_summary}
Position: {position_text} | {signal_text} | {activation_text}{smc_summary}{rules}{price_action_components}Aggressive Day Trading: Focus 15m/5m momentum, enter on 1.2:1 R:R, ignore higher TF if price action strong.EMPHASIS: Higher TF (1h) bias for direction + Lower TF (5m) entry. Require confirmations: Volume >1.5x MA, candle patterns, FVG stacking. Use fresh zones only (no multiple interactions).CRITICAL: Always provide EXACT numeric values for stop_loss and take_profit. Never use null, variables, or placeholders.For BUY signals: stop_loss < {current_price:.0f} < take_profit
For SELL signals: take_profit < {current_price:.0f} < stop_loss
For HOLD signals: Use current price ±1% as SL/TPExample valid responses:
BUY: {{"signal": "BUY", "reason": "Bullish structure with HH+HL and FVG support", "stop_loss": {current_price*0.98:.0f}, "take_profit": {current_price*1.04:.0f}, "confidence": "MEDIUM"}}
SELL: {{"signal": "SELL", "reason": "Bearish structure with LH+LL and order block resistance", "stop_loss": {current_price*1.02:.0f}, "take_profit": {current_price*0.96:.0f}, "confidence": "MEDIUM"}}
HOLD: {{"signal": "HOLD", "reason": "Neutral structure with no clear directional bias", "stop_loss": {current_price*0.99:.0f}, "take_profit": {current_price*1.01:.0f}, "confidence": "LOW"}}Return JSON only:
{{
 "signal": "BUY|SELL|HOLD",
 "reason": "Brief SMC analysis with price action context",
 "stop_loss": EXACT_NUMBER,
 "take_profit": EXACT_NUMBER,
 "confidence": "HIGH|MEDIUM|LOW"
}}"""

    try:
        print("发送请求到DeepSeek API...")
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": """You are an expert SMC/ICT trader with deep knowledge of Price Action, Market Structure, and Order Flow analysis. Analyze the provided data using core price action components and return strict JSON only.

核心价格行为组件:
1. 市场结构(Market Structure):
   - 高点/低点序列：更高的高点(HH)+更高的低点(HL)=上升趋势
   - 更低的高点(LH)+更低的低点(LL)=下降趋势
   - 结构突破：关键支撑/阻力位的突破

2. 关键水平(Key Levels):
   - 支撑位：前低点、需求区、订单区块
   - 阻力位：前高点、供应区、订单区块
   - 突破回测：突破后回踩确认

3. 价格模式(Candle Patterns):
   - 吞噬模式：看涨吞噬/看跌吞噬
   - 锤子线/上吊线：反转信号
   - 内在/外在柱：波动率收缩与扩张
   - 假突破：突破后快速反转

4. 订单流概念(SMC特有):
   - FVG(公允价值缺口)：连续蜡烛间的价格缺口
   - 流动性挖掘：止损猎杀、流动性池
   - OB(订单区块)：机构订单集中区域
   - MIT(市场内部转换)：角色转换(支撑变阻力)

5. 动量分析:
   - 柱体大小：反映买卖压力
   - 收盘位置：柱体收盘于高位/低位
   - 波动率变化：突然的波动率扩张

在交易框架中的应用:
基于规则，重点关注:
1. 1小时图确定主要结构方向
2. 5分钟图寻找：
   - 在关键水平(EMA21、前高低点)的反应
   - FVG堆积区域的价格行为
   - 配合成交量确认的蜡烛模式
   - 新鲜区域的首次交互"""},
                {"role": "user", "content": prompt}
            ],
            stream=False,
            temperature=0.1,
            timeout=30
        )
        
        result = response.choices[0].message.content
        print("\n=== DeepSeek API 响应 ===")
        print(result)
        
        # 尝试解析JSON
        try:
            # 处理可能的```json标记
            if "```json" in result:
                # 提取JSON部分
                start = result.find("```json") + 7
                end = result.find("```", start)
                if end != -1:
                    result = result[start:end].strip()
            elif "```" in result:
                # 提取代码块部分
                start = result.find("```") + 3
                end = result.find("```", start)
                if end != -1:
                    result = result[start:end].strip()
            
            signal_data = json.loads(result)
            print("\n=== JSON 解析成功 ===")
            print(f"信号: {signal_data.get('signal')}")
            print(f"理由: {signal_data.get('reason')}")
            print(f"止损: {signal_data.get('stop_loss')}")
            print(f"止盈: {signal_data.get('take_profit')}")
            print(f"置信度: {signal_data.get('confidence')}")
            return True
        except json.JSONDecodeError as e:
            print(f"\n=== JSON 解析失败 ===")
            print(f"错误: {e}")
            print(f"原始响应: {result}")
            return False
            
    except Exception as e:
        print(f"请求失败: {e}")
        return False

if __name__ == "__main__":
    success = test_updated_prompt()
    if success:
        print("\n✅ 测试成功！更新后的提示词正常工作。")
    else:
        print("\n❌ 测试失败！请检查提示词或API配置。")