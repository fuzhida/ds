"""
SMC交易原则AI交易系统
专注于OB/FVG/SMC交易原则，只提供原始数据，减少本地运算
"""

import json
import random
from datetime import datetime
from typing import Dict, Any, List


class SMCOBFVGAITrader:
    """SMC交易原则AI交易系统"""
    
    def __init__(self):
        self.symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]
        self.analysis_results = []
        
        # 今日交易统计（简化版）
        self.today_stats = {
            "initial_capital": 10000.0,
            "today_pnl": 0.0,
            "today_trades": 0,
            "today_wins": 0,
            "today_losses": 0,
            "positions": [],
            "risk_per_trade": 0.005,  # 每单风险为余额的0.5%
            "fee_rate": 0.0005,  # 手续费率：0.05%（开仓+平仓各0.025%）
            "leverage": 10  # 杠杆倍数
        }
        
        print("🎯 SMC交易原则AI交易系统初始化完成")
    
    def get_today_trading_stats(self) -> Dict[str, Any]:
        """获取今日交易统计"""
        total_trades = self.today_stats["today_trades"]
        win_rate = (self.today_stats["today_wins"] / total_trades * 100) if total_trades > 0 else 0
        current_capital = self.today_stats["initial_capital"] + self.today_stats["today_pnl"]
        risk_amount = current_capital * self.today_stats["risk_per_trade"]
        
        return {
            "initial_capital": self.today_stats["initial_capital"],
            "current_capital": current_capital,
            "today_pnl": self.today_stats["today_pnl"],
            "today_trades": self.today_stats["today_trades"],
            "today_wins": self.today_stats["today_wins"],
            "today_losses": self.today_stats["today_losses"],
            "today_win_rate": win_rate,
            "positions": self.today_stats["positions"].copy(),
            "risk_per_trade": self.today_stats["risk_per_trade"],
            "risk_amount": risk_amount,  # 每单风险金额
            "fee_rate": self.today_stats["fee_rate"],  # 手续费率
            "leverage": self.today_stats["leverage"]  # 杠杆倍数
        }
    
    def extract_raw_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        提取原始市场数据
        只提供最基础的原始数据，不进行任何计算
        """
        print(f"📊 开始提取 {symbol} 原始市场数据...")
        
        # 模拟基础价格数据
        base_prices = {
            "BTC/USD": {"current": 108363.01, "open": 107500.00, "high": 109200.00, "low": 106800.00},
            "ETH/USD": {"current": 3250.75, "open": 3200.00, "high": 3300.00, "low": 3180.00},
            "SOL/USD": {"current": 145.20, "open": 142.00, "high": 148.00, "low": 140.00}
        }
        
        # 原始市场数据（不进行任何计算）
        raw_data = {
            "symbol": symbol,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "price_data": base_prices.get(symbol, {"current": 100.0, "open": 100.0, "high": 105.0, "low": 95.0}),
            "timeframes": {
                "15m": {"open": base_prices[symbol]["open"] * 0.998, "high": base_prices[symbol]["high"] * 0.999, 
                         "low": base_prices[symbol]["low"] * 1.001, "close": base_prices[symbol]["current"] * 0.997},
                "1h": {"open": base_prices[symbol]["open"] * 0.995, "high": base_prices[symbol]["high"] * 0.998,
                       "low": base_prices[symbol]["low"] * 1.002, "close": base_prices[symbol]["current"] * 0.996},
                "4h": {"open": base_prices[symbol]["open"] * 0.992, "high": base_prices[symbol]["high"] * 0.996,
                       "low": base_prices[symbol]["low"] * 1.004, "close": base_prices[symbol]["current"] * 0.994}
            },
            "volume_data": {
                "24h_volume": random.uniform(1000000, 5000000),
                "current_volume": random.uniform(50000, 200000)
            }
        }
        
        print(f"✅ {symbol} 原始市场数据提取完成")
        return raw_data
    
    def detect_smc_obfvg_patterns(self, symbol: str, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        检测SMC交易原则中的OB/FVG模式
        只提供原始模式数据，不进行强度计算
        """
        print(f"📍 开始检测 {symbol} SMC/OB/FVG模式...")
        
        current_price = raw_data["price_data"]["current"]
        
        # SMC交易原则中的关键模式（原始数据）
        smc_patterns = {
            "symbol": symbol,
            "order_blocks": {
                "bullish_ob": {
                    "support": current_price * 0.98,  # 看涨OB支撑
                    "resistance": current_price * 1.02,  # 看涨OB阻力
                    "mid_point": current_price * 1.00   # 看涨OB中点
                },
                "bearish_ob": {
                    "support": current_price * 0.96,  # 看跌OB支撑
                    "resistance": current_price * 0.98,  # 看跌OB阻力
                    "mid_point": current_price * 0.97   # 看跌OB中点
                },
                "recent_ob_count": random.randint(2, 8)  # 近期OB数量
            },
            "fair_value_gaps": {
                "bullish_fvg": {
                    "gap_top": current_price * 1.03,    # 看涨FVG顶部
                    "gap_bottom": current_price * 1.01,  # 看涨FVG底部
                    "gap_size": current_price * 0.02    # 看涨FVG大小
                },
                "bearish_fvg": {
                    "gap_top": current_price * 0.99,    # 看跌FVG顶部
                    "gap_bottom": current_price * 0.97,  # 看跌FVG底部
                    "gap_size": current_price * 0.02    # 看跌FVG大小
                },
                "recent_fvg_count": random.randint(1, 5)  # 近期FVG数量
            },
            "structure_breaks": {
                "bos_levels": {
                    "breakout_level": current_price * 1.01,      # BOS突破水平
                    "invalidation_level": current_price * 0.99   # BOS失效水平
                },
                "choch_levels": {
                    "reversal_level": current_price * 0.98,     # CHOCH反转水平
                    "confirmation_level": current_price * 0.97    # CHOCH确认水平
                }
            },
            "key_levels": {
                "daily_open": raw_data["price_data"]["open"],     # 日开盘价
                "4h_open": raw_data["timeframes"]["4h"]["open"],  # 4h开盘价
                "4h_high": raw_data["timeframes"]["4h"]["high"],  # 4h高点
                "4h_low": raw_data["timeframes"]["4h"]["low"],    # 4h低点
                "weekly_open": raw_data["price_data"]["open"] * 0.995,  # 本周开盘价
                "prev_week_high": raw_data["price_data"]["high"] * 1.02,  # 上周高点
                "prev_week_low": raw_data["price_data"]["low"] * 0.98,    # 上周低点
                "monthly_open": raw_data["price_data"]["open"] * 0.99,     # 本月开盘价
                "monthly_high": raw_data["price_data"]["high"] * 1.05,     # 本月高点
                "monthly_low": raw_data["price_data"]["low"] * 0.95,      # 本月低点
                "prev_month_high": raw_data["price_data"]["high"] * 1.08,  # 上月高点
                "prev_month_low": raw_data["price_data"]["low"] * 0.92     # 上月低点
            }
        }
        
        print(f"✅ {symbol} SMC/OB/FVG模式检测完成")
        return smc_patterns
    
    def generate_smc_ai_prompt(self, symbol: str, raw_data: Dict[str, Any], 
                              smc_patterns: Dict[str, Any]) -> str:
        """
        生成SMC交易原则AI提示词
        专注于OB/FVG/SMC交易原则，只提供原始数据
        """
        
        # 获取今日交易统计
        today_stats = self.get_today_trading_stats()
        
        # 确定4小时和1小时级别方向
        direction_info = self.determine_timeframe_direction(raw_data)
        
        prompt = f"""
你是一个专业的SMC交易原则AI，专注于订单块(OB)、公允价值缺口(FVG)和聪明钱概念(SMC)交易策略。

## 📊 今日交易统计
- 初始金额: ${today_stats['initial_capital']:,.2f}
- 当前资金: ${today_stats['current_capital']:,.2f}
- 今日盈亏: ${today_stats['today_pnl']:,.2f}
- 今日胜率: {today_stats['today_win_rate']:.1f}%
- 交易次数: {today_stats['today_trades']} (胜/败: {today_stats['today_wins']}/{today_stats['today_losses']})
- 持仓数量: {len(today_stats['positions'])}
- 每单风险比例: {today_stats['risk_per_trade']*100:.1f}%
- 每单风险金额: ${today_stats['risk_amount']:,.2f}
- 手续费率: {today_stats['fee_rate']*100:.3f}% (开仓+平仓各{today_stats['fee_rate']*50:.3f}%)
- 杠杆倍数: {today_stats['leverage']}x

## 📈 原始市场数据 - {symbol}
**当前价格**: ${raw_data['price_data']['current']:,.2f}
**日开盘价**: ${raw_data['price_data']['open']:,.2f}
**日高点**: ${raw_data['price_data']['high']:,.2f}
**日低点**: ${raw_data['price_data']['low']:,.2f}

**时间框架数据**:
- 15m: 开${raw_data['timeframes']['15m']['open']:,.2f} 高${raw_data['timeframes']['15m']['high']:,.2f} 低${raw_data['timeframes']['15m']['low']:,.2f} 收${raw_data['timeframes']['15m']['close']:,.2f}
- 1h: 开${raw_data['timeframes']['1h']['open']:,.2f} 高${raw_data['timeframes']['1h']['high']:,.2f} 低${raw_data['timeframes']['1h']['low']:,.2f} 收${raw_data['timeframes']['1h']['close']:,.2f}
- 4h: 开${raw_data['timeframes']['4h']['open']:,.2f} 高${raw_data['timeframes']['4h']['high']:,.2f} 低${raw_data['timeframes']['4h']['low']:,.2f} 收${raw_data['timeframes']['4h']['close']:,.2f}

**成交量数据**:
- 24h成交量: {raw_data['volume_data']['24h_volume']:,.0f}
- 当前成交量: {raw_data['volume_data']['current_volume']:,.0f}

## 🔥 SMC交易原则 - OB/FVG模式分析

### 📍 订单块(Order Blocks)分析
**看涨订单块**:
- 支撑: ${smc_patterns['order_blocks']['bullish_ob']['support']:,.2f}
- 阻力: ${smc_patterns['order_blocks']['bullish_ob']['resistance']:,.2f}
- 中点: ${smc_patterns['order_blocks']['bullish_ob']['mid_point']:,.2f}

**看跌订单块**:
- 支撑: ${smc_patterns['order_blocks']['bearish_ob']['support']:,.2f}
- 阻力: ${smc_patterns['order_blocks']['bearish_ob']['resistance']:,.2f}
- 中点: ${smc_patterns['order_blocks']['bearish_ob']['mid_point']:,.2f}

**近期OB数量**: {smc_patterns['order_blocks']['recent_ob_count']}个

### 📈 公允价值缺口(Fair Value Gaps)分析
**看涨FVG**:
- 缺口顶部: ${smc_patterns['fair_value_gaps']['bullish_fvg']['gap_top']:,.2f}
- 缺口底部: ${smc_patterns['fair_value_gaps']['bullish_fvg']['gap_bottom']:,.2f}
- 缺口大小: ${smc_patterns['fair_value_gaps']['bullish_fvg']['gap_size']:,.2f}

**看跌FVG**:
- 缺口顶部: ${smc_patterns['fair_value_gaps']['bearish_fvg']['gap_top']:,.2f}
- 缺口底部: ${smc_patterns['fair_value_gaps']['bearish_fvg']['gap_bottom']:,.2f}
- 缺口大小: ${smc_patterns['fair_value_gaps']['bearish_fvg']['gap_size']:,.2f}

**近期FVG数量**: {smc_patterns['fair_value_gaps']['recent_fvg_count']}个

### 🏗️ 结构破坏分析
**BOS (结构突破)**:
- 突破水平: ${smc_patterns['structure_breaks']['bos_levels']['breakout_level']:,.2f}
- 失效水平: ${smc_patterns['structure_breaks']['bos_levels']['invalidation_level']:,.2f}

**CHOCH (特征变化)**:
- 反转水平: ${smc_patterns['structure_breaks']['choch_levels']['reversal_level']:,.2f}
- 确认水平: ${smc_patterns['structure_breaks']['choch_levels']['confirmation_level']:,.2f}

### 📅 关键水平参考
- **日开盘价**: ${smc_patterns['key_levels']['daily_open']:,.2f}
- **4h开盘价**: ${smc_patterns['key_levels']['4h_open']:,.2f}
- **4h高点**: ${smc_patterns['key_levels']['4h_high']:,.2f}
- **4h低点**: ${smc_patterns['key_levels']['4h_low']:,.2f}
- **本周开盘价**: ${smc_patterns['key_levels']['weekly_open']:,.2f}
- **上周高点**: ${smc_patterns['key_levels']['prev_week_high']:,.2f}
- **上周低点**: ${smc_patterns['key_levels']['prev_week_low']:,.2f}
- **本月开盘价**: ${smc_patterns['key_levels']['monthly_open']:,.2f}
- **本月高点**: ${smc_patterns['key_levels']['monthly_high']:,.2f}
- **本月低点**: ${smc_patterns['key_levels']['monthly_low']:,.2f}
- **上月高点**: ${smc_patterns['key_levels']['prev_month_high']:,.2f}
- **上月低点**: ${smc_patterns['key_levels']['prev_month_low']:,.2f}

## 🎯 SMC交易原则决策要求

**核心原则**: 只在OB/FVG/SMC交易原则相同时才行动

**个人交易偏好规则**:
- **4小时级别方向判断**: 4小时开盘后第一个5分钟收盘价在4小时开盘上方则看多，下方则看空
- **1小时级别方向判断**: 1小时开盘后第一个5分钟收盘价在1小时开盘上方则看多，下方则看空
- **方向一致性要求**: 短线单需4小时、1小时与当前做单方向一致才开（高确定性情况除外）

**交易条件**:
1. **订单块确认**: 价格在有效的订单块区域内
2. **FVG确认**: 存在明显的公允价值缺口
3. **结构确认**: BOS/CHOCH结构得到确认
4. **关键水平**: 价格在重要的支撑/阻力水平附近
5. **方向一致性**: 4小时、1小时和当前做单方向一致（高确定性可忽略）

**决策输出格式**:
```json
{{
    "decision": "BUY/SELL/WAIT",
    "confidence": 0.0-1.0,
    "entry_price": 具体入场价格,
    "stop_loss": 基于OB/FVG的止损价,
    "take_profit": 目标价位,
    "analysis": "详细的SMC交易原则分析"
}}
```

**重点分析**:
- OB的有效性和强度
- FVG的缺口大小和位置
- 结构破坏的确认程度
- 关键水平的支撑/阻力作用
- 成交量与价格行为的配合

请基于纯粹的SMC交易原则进行分析，只在所有条件都符合时才给出交易信号。
"""
        
        return prompt
    
    def calculate_fibonacci_levels(self, current_price: float, stop_loss: float) -> Dict[str, float]:
        """计算斐波那契回撤和扩展水平"""
        # 斐波那契回撤水平 (从入场到止损)
        entry_to_stop = current_price - stop_loss
        
        fib_levels = {
            # 回撤水平 (从入场价向下)
            "fib_0_236": current_price - entry_to_stop * 0.236,
            "fib_0_382": current_price - entry_to_stop * 0.382,
            "fib_0_5": current_price - entry_to_stop * 0.5,
            "fib_0_618": current_price - entry_to_stop * 0.618,
            "fib_0_786": current_price - entry_to_stop * 0.786,
            
            # 扩展水平 (从入场价向上)
            "fib_1_0": current_price + entry_to_stop * 1.0,  # 1:1风险回报
            "fib_1_272": current_price + entry_to_stop * 1.272,
            "fib_1_414": current_price + entry_to_stop * 1.414,
            "fib_1_618": current_price + entry_to_stop * 1.618,  # 黄金比例
            "fib_2_0": current_price + entry_to_stop * 2.0,  # 2:1风险回报
            "fib_2_618": current_price + entry_to_stop * 2.618,
            "fib_3_0": current_price + entry_to_stop * 3.0,  # 3:1风险回报
            "fib_4_236": current_price + entry_to_stop * 4.236,
            
            # 特殊水平
            "fib_0_97": current_price + entry_to_stop * 0.97,  # 97%标准指标
        }
        
        return fib_levels
    
    def determine_take_profit_strategy(self, current_price: float, stop_loss: float, 
                                     smc_patterns: Dict[str, Any]) -> Dict[str, Any]:
        """确定动态止盈策略"""
        fib_levels = self.calculate_fibonacci_levels(current_price, stop_loss)
        
        # 基于SMC模式和斐波那契确定多段止盈
        ob_strength = smc_patterns["order_blocks"]["recent_ob_count"]
        fvg_strength = smc_patterns["fair_value_gaps"]["recent_fvg_count"]
        
        # 动态调整止盈策略
        if ob_strength >= 4 and fvg_strength >= 3:
            # 强势信号：多段止盈
            take_profit_strategy = {
                "strategy_type": "multi_stage_fibonacci",
                "stages": [
                    {"level": "fib_1_0", "percentage": 0.3, "description": "第一目标：1:1风险回报"},
                    {"level": "fib_1_618", "percentage": 0.4, "description": "第二目标：黄金比例扩展"},
                    {"level": "fib_2_618", "percentage": 0.3, "description": "第三目标：强势扩展"}
                ],
                "fib_levels": fib_levels
            }
        elif ob_strength >= 3 and fvg_strength >= 2:
            # 中等信号：两段止盈
            take_profit_strategy = {
                "strategy_type": "two_stage_fibonacci",
                "stages": [
                    {"level": "fib_1_0", "percentage": 0.5, "description": "第一目标：1:1风险回报"},
                    {"level": "fib_1_618", "percentage": 0.5, "description": "第二目标：黄金比例扩展"}
                ],
                "fib_levels": fib_levels
            }
        else:
            # 弱势信号：单段止盈
            take_profit_strategy = {
                "strategy_type": "single_stage_fibonacci",
                "stages": [
                    {"level": "fib_1_0", "percentage": 1.0, "description": "单目标：1:1风险回报"}
                ],
                "fib_levels": fib_levels
            }
        
        return take_profit_strategy
    
    def determine_timeframe_direction(self, raw_data: Dict[str, Any]) -> Dict[str, str]:
        """
        确定4小时和1小时级别方向
        基于开盘后第一个5分钟收盘价相对于开盘价的位置判断方向
        """
        # 模拟开盘后第一个5分钟收盘价（使用15分钟数据作为代理）
        first_5m_close_4h = raw_data["timeframes"]["15m"]["close"]
        first_5m_close_1h = raw_data["timeframes"]["15m"]["close"]
        
        # 获取4小时和1小时开盘价
        open_4h = raw_data["timeframes"]["4h"]["open"]
        open_1h = raw_data["timeframes"]["1h"]["open"]
        
        # 判断4小时级别方向
        if first_5m_close_4h > open_4h:
            direction_4h = "BULLISH"  # 看多
        elif first_5m_close_4h < open_4h:
            direction_4h = "BEARISH"  # 看空
        else:
            direction_4h = "NEUTRAL"  # 中性
        
        # 判断1小时级别方向
        if first_5m_close_1h > open_1h:
            direction_1h = "BULLISH"  # 看多
        elif first_5m_close_1h < open_1h:
            direction_1h = "BEARISH"  # 看空
        else:
            direction_1h = "NEUTRAL"  # 中性
        
        direction_info = {
            "4h_direction": direction_4h,
            "1h_direction": direction_1h,
            "4h_open": open_4h,
            "1h_open": open_1h,
            "first_5m_close_4h": first_5m_close_4h,
            "first_5m_close_1h": first_5m_close_1h,
            "4h_analysis": f"4小时开盘价: ${open_4h:,.2f}, 第一个5分钟收盘价: ${first_5m_close_4h:,.2f}, 方向: {direction_4h}",
            "1h_analysis": f"1小时开盘价: ${open_1h:,.2f}, 第一个5分钟收盘价: ${first_5m_close_1h:,.2f}, 方向: {direction_1h}"
        }
        
        return direction_info
    
    def simulate_smc_ai_analysis(self, prompt: str, raw_data: Dict[str, Any], 
                                smc_patterns: Dict[str, Any]) -> Dict[str, Any]:
        """模拟SMC交易原则AI分析"""
        print("🤖 开始SMC交易原则AI分析...")
        
        current_price = raw_data["price_data"]["current"]
        
        # 获取今日交易统计
        today_stats = self.get_today_trading_stats()
        risk_amount = today_stats["risk_amount"]
        
        # 基于SMC原则的简单决策逻辑
        ob_count = smc_patterns["order_blocks"]["recent_ob_count"]
        fvg_count = smc_patterns["fair_value_gaps"]["recent_fvg_count"]
        
        # 确定4小时和1小时级别方向
        direction_info = self.determine_timeframe_direction(raw_data)
        direction_4h = direction_info["4h_direction"]
        direction_1h = direction_info["1h_direction"]
        
        # 判断当前做单方向（基于价格相对于开盘价的位置）
        current_direction = "BULLISH" if current_price > raw_data["timeframes"]["15m"]["open"] else "BEARISH"
        
        # 检查方向一致性
        directions_consistent = (direction_4h == direction_1h == current_direction)
        
        # SMC交易原则：OB和FVG都有效时才交易
        # 增加方向一致性检查：除非高确定性，否则需要4小时、1小时和当前做单方向一致
        if ob_count >= 3 and fvg_count >= 2:
            # 高确定性情况：OB和FVG都很强，可以忽略方向一致性
            if ob_count >= 5 and fvg_count >= 4:
                decision = "BUY"
                confidence = 0.90
                direction_check = "高确定性，忽略方向一致性检查"
            # 正常情况：需要方向一致性
            elif directions_consistent:
                decision = "BUY"
                confidence = 0.85
                direction_check = "方向一致性检查通过"
            else:
                decision = "WAIT"
                confidence = 0.60
                direction_check = "方向不一致，等待更好时机"
        else:
            decision = "WAIT"
            confidence = 0.60
            direction_check = "SMC信号不足"
        
        # 基于每单风险金额的止损设置
        stop_loss_distance = 0.02  # 2%止损距离
        stop_loss = current_price * (1 - stop_loss_distance)
        
        # 计算仓位大小
        position_size = risk_amount / (current_price - stop_loss)
        
        # 计算仓位价值
        position_value = position_size * current_price
        
        # 计算精确风险金额
        exact_risk_amount = position_size * (current_price - stop_loss)
        
        # 计算手续费（开仓+平仓）
        fee_rate = today_stats["fee_rate"]
        leverage = today_stats["leverage"]
        
        # 开仓手续费 = 仓位价值 * 杠杆 * 开仓费率
        open_fee = position_value * leverage * (fee_rate / 2)
        
        # 平仓手续费 = 仓位价值 * 杠杆 * 平仓费率
        close_fee = position_value * leverage * (fee_rate / 2)
        
        # 总手续费
        total_fee = open_fee + close_fee
        
        # 手续费对风险回报的影响
        fee_impact_risk = total_fee / position_size  # 手续费对每单位价格的影响
        
        # 动态斐波那契止盈策略
        take_profit_strategy = self.determine_take_profit_strategy(current_price, stop_loss, smc_patterns)
        
        # 计算平均止盈价格（考虑手续费影响）
        avg_take_profit = sum(
            take_profit_strategy["fib_levels"][stage["level"]] * stage["percentage"] 
            for stage in take_profit_strategy["stages"]
        )
        
        analysis_result = {
            "symbol": raw_data["symbol"],
            "decision": decision,
            "confidence": confidence,
            "entry_price": current_price,
            "stop_loss": stop_loss,
            "take_profit": avg_take_profit,
            "take_profit_strategy": take_profit_strategy,
            "position_size": position_size,
            "position_value": position_value,
            "risk_amount": exact_risk_amount,
            "stop_loss_distance": stop_loss_distance,
            "fee_calculation": {
                "fee_rate": fee_rate,
                "leverage": leverage,
                "open_fee": open_fee,
                "close_fee": close_fee,
                "total_fee": total_fee,
                "fee_impact_risk": fee_impact_risk
            },
            "direction_analysis": {
                "4h_direction": direction_4h,
                "1h_direction": direction_1h,
                "current_direction": current_direction,
                "directions_consistent": directions_consistent,
                "direction_check": direction_check,
                "4h_open": direction_info["4h_open"],
                "1h_open": direction_info["1h_open"],
                "first_5m_close_4h": direction_info["first_5m_close_4h"],
                "first_5m_close_1h": direction_info["first_5m_close_1h"]
            },
            "analysis": f"基于SMC交易原则分析：OB数量{ob_count}个，FVG数量{fvg_count}个，{direction_check}。每单风险${exact_risk_amount:.2f}，手续费${total_fee:.2f}"
        }
        
        print("✅ SMC交易原则AI分析完成")
        return analysis_result
    
    def analyze_single_symbol(self, symbol: str) -> Dict[str, Any]:
        """分析单个品种"""
        print(f"🎯 开始分析 {symbol}...")
        
        # 1. 提取原始市场数据
        raw_data = self.extract_raw_market_data(symbol)
        
        # 2. 检测SMC/OB/FVG模式
        smc_patterns = self.detect_smc_obfvg_patterns(symbol, raw_data)
        
        # 3. 生成SMC交易原则AI提示词
        prompt = self.generate_smc_ai_prompt(symbol, raw_data, smc_patterns)
        
        # 4. 模拟AI分析
        analysis_result = self.simulate_smc_ai_analysis(prompt, raw_data, smc_patterns)
        
        # 组合完整结果
        result = {
            "symbol": symbol,
            "timestamp": raw_data["timestamp"],
            "raw_data": raw_data,
            "smc_patterns": smc_patterns,
            "ai_prompt": prompt,
            "analysis_result": analysis_result
        }
        
        print(f"✅ {symbol} 分析完成")
        return result
    
    def analyze_multiple_symbols(self) -> List[Dict[str, Any]]:
        """分析多个品种"""
        print("🎯 开始多品种SMC交易原则分析...")
        
        results = []
        for symbol in self.symbols:
            result = self.analyze_single_symbol(symbol)
            results.append(result)
        
        # 显示详细分析结果
        print(f"\n📊 详细分析结果:")
        for result in results:
            symbol = result["symbol"]
            analysis = result["analysis_result"]
            tp_strategy = analysis["take_profit_strategy"]
            
            print(f"\n🎯 {symbol} 分析结果:")
            print(f"   决策: {analysis['decision']} (置信度: {analysis['confidence']:.2f})")
            print(f"   入场价: ${analysis['entry_price']:,.2f}")
            print(f"   止损价: ${analysis['stop_loss']:,.2f} (距离: {analysis['stop_loss_distance']*100:.1f}%)")
            print(f"   平均目标价: ${analysis['take_profit']:,.2f}")
            print(f"   止盈策略: {tp_strategy['strategy_type']}")
            
            # 显示方向判断信息
            direction_analysis = analysis['direction_analysis']
            print(f"   📊 方向判断分析:")
            print(f"     4小时方向: {direction_analysis['4h_direction']}")
            print(f"     1小时方向: {direction_analysis['1h_direction']}")
            print(f"     当前方向: {direction_analysis['current_direction']}")
            print(f"     方向一致性: {'✅ 一致' if direction_analysis['directions_consistent'] else '❌ 不一致'}")
            print(f"     方向检查结果: {direction_analysis['direction_check']}")
            print(f"     4小时开盘价: ${direction_analysis['4h_open']:,.2f}")
            print(f"     1小时开盘价: ${direction_analysis['1h_open']:,.2f}")
            print(f"     4小时第一个5分钟收盘价: ${direction_analysis['first_5m_close_4h']:,.2f}")
            print(f"     1小时第一个5分钟收盘价: ${direction_analysis['first_5m_close_1h']:,.2f}")
            
            # 显示手续费信息
            fee_calc = analysis['fee_calculation']
            print(f"   💰 手续费计算:")
            print(f"     手续费率: {fee_calc['fee_rate']*100:.3f}% (开仓+平仓各{fee_calc['fee_rate']*50:.3f}%)")
            print(f"     杠杆倍数: {fee_calc['leverage']}x")
            print(f"     开仓手续费: ${fee_calc['open_fee']:.2f}")
            print(f"     平仓手续费: ${fee_calc['close_fee']:.2f}")
            print(f"     总手续费: ${fee_calc['total_fee']:.2f}")
            print(f"     手续费影响: ${fee_calc['fee_impact_risk']:.4f}/单位")
            
            # 显示斐波那契止盈水平
            print(f"   📈 斐波那契止盈水平:")
            fib_levels = tp_strategy["fib_levels"]
            print(f"     97%标准指标: ${fib_levels['fib_0_97']:,.2f}")
            print(f"     1:1风险回报: ${fib_levels['fib_1_0']:,.2f}")
            print(f"     黄金比例: ${fib_levels['fib_1_618']:,.2f}")
            print(f"     强势扩展: ${fib_levels['fib_2_618']:,.2f}")
            
            # 显示多段止盈分配
            print(f"   🎯 多段止盈分配:")
            for i, stage in enumerate(tp_strategy["stages"], 1):
                level_price = fib_levels[stage["level"]]
                print(f"     第{i}段: {stage['percentage']*100:.0f}% -> ${level_price:,.2f} ({stage['description']})")
            
            # 显示关键水平参考
            print(f"   📊 关键水平参考:")
            print(f"     日开盘价: ${result['smc_patterns']['key_levels']['daily_open']:,.2f}")
            print(f"     4h开盘价: ${result['smc_patterns']['key_levels']['4h_open']:,.2f}")
            print(f"     4h高点: ${result['smc_patterns']['key_levels']['4h_high']:,.2f}")
            print(f"     4h低点: ${result['smc_patterns']['key_levels']['4h_low']:,.2f}")
            print(f"     本周开盘价: ${result['smc_patterns']['key_levels']['weekly_open']:,.2f}")
            print(f"     上周高点: ${result['smc_patterns']['key_levels']['prev_week_high']:,.2f}")
            print(f"     上周低点: ${result['smc_patterns']['key_levels']['prev_week_low']:,.2f}")
            print(f"     本月开盘价: ${result['smc_patterns']['key_levels']['monthly_open']:,.2f}")
            print(f"     本月高点: ${result['smc_patterns']['key_levels']['monthly_high']:,.2f}")
            print(f"     本月低点: ${result['smc_patterns']['key_levels']['monthly_low']:,.2f}")
            print(f"     上月高点: ${result['smc_patterns']['key_levels']['prev_month_high']:,.2f}")
            print(f"     上月低点: ${result['smc_patterns']['key_levels']['prev_month_low']:,.2f}")
            
            print(f"   仓位大小: {analysis['position_size']:.4f} 单位")
            print(f"   仓位价值: ${analysis['position_value']:,.2f}")
            print(f"   每单风险: ${analysis['risk_amount']:,.2f}")
            print(f"   分析: {analysis['analysis']}")
        
        print(f"\n✅ 多品种分析完成")
        print(f"   分析品种数: {len(results)}")
        print(f"   🔥 专注于SMC交易原则 - OB/FVG/SMC")
        print(f"   💰 每单风险: 余额的{self.today_stats['risk_per_trade']*100:.1f}%")
        
        return results
    
    def save_analysis_results(self, results: List[Dict[str, Any]]) -> str:
        """保存分析结果"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"smc_obfvg_analysis_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"📁 分析结果已保存到: {filename}")
        return filename
    
    def run_analysis(self):
        """运行完整分析流程"""
        print("🚀 启动SMC交易原则AI交易系统...")
        
        # 分析多个品种
        results = self.analyze_multiple_symbols()
        
        # 保存结果
        filename = self.save_analysis_results(results)
        
        print("🎉 SMC交易原则AI交易系统运行完成")
        print(f"   结果文件: {filename}")
        print(f"   🔥 专注于纯粹的SMC交易原则")
        print(f"   📍 核心: OB/FVG/SMC模式分析")


def main():
    """主函数"""
    trader = SMCOBFVGAITrader()
    trader.run_analysis()


if __name__ == "__main__":
    main()