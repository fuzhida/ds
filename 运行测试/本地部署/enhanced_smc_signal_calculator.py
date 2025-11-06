#!/usr/bin/env python3
"""
增强版SMC信号计算器
实现基于DeepSeek建议的增强版数据结构的SMC信号计算
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
import logging

class EnhancedSMCSignalCalculator:
    """增强版SMC信号计算器，使用增强版数据结构进行更准确的信号计算"""
    
    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # 信号权重配置
        self.signal_weights = getattr(config, 'enhanced_smc_signal_weights', {
            'bos_choch': 0.3,  # BOS/CHOCH信号权重
            'order_blocks': 0.25,  # 订单块信号权重
            'fvg': 0.2,  # FVG信号权重
            'liquidity': 0.15,  # 流动性信号权重
            'market_microstructure': 0.1  # 市场微观结构信号权重
        })
        
        # 最小置信度阈值
        self.min_confidence = getattr(config, 'enhanced_smc_min_confidence', 0.6)
        
        # 增强版数据权重
        self.enhanced_data_weight = getattr(config, 'enhanced_data_weight', 0.7)
        self.market_depth_weight = getattr(config, 'market_depth_weight', 0.15)
        self.time_sales_weight = getattr(config, 'time_sales_weight', 0.1)
        self.market_sentiment_weight = getattr(config, 'market_sentiment_weight', 0.05)
    
    def calculate_bos_choch_signal(self, enhanced_candlesticks: List[Dict]) -> Dict[str, Any]:
        """
        计算BOS/CHOCH信号
        
        参数:
            enhanced_candlesticks: 增强版K线数据
            
        返回:
            BOS/CHOCH信号分析结果
        """
        if len(enhanced_candlesticks) < 10:
            return {'signal': 'HOLD', 'confidence': 0.0, 'reason': '数据不足'}
        
        # 获取最近的K线数据
        recent_candles = enhanced_candlesticks[-10:]
        
        # 转换为DataFrame便于分析
        df = pd.DataFrame(recent_candles)
        
        # 计算高低点
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        
        # 寻找最近的高点和低点
        recent_high = np.max(highs[-5:])
        recent_low = np.min(lows[-5:])
        
        # 检查是否有BOS（Break of Structure）
        prev_high = np.max(highs[-10:-5])
        prev_low = np.min(lows[-10:-5])
        
        # 当前价格
        current_price = closes[-1]
        
        # BOS/CHOCH判断逻辑
        signal = 'HOLD'
        confidence = 0.0
        reason = '无明确结构'
        
        # 看涨BOS/CHOCH
        if current_price > prev_high and recent_high > prev_high:
            signal = 'BUY'
            confidence = min(0.8, 0.5 + (current_price - prev_high) / prev_high * 10)
            reason = f'看涨BOS: 价格突破前高 {prev_high:.2f}'
        
        # 看跌BOS/CHOCH
        elif current_price < prev_low and recent_low < prev_low:
            signal = 'SELL'
            confidence = min(0.8, 0.5 + (prev_low - current_price) / prev_low * 10)
            reason = f'看跌BOS: 价格跌破前低 {prev_low:.2f}'
        
        # 检查CHOCH（Change of Character）
        # 看涨CHOCH：在下跌趋势中，价格突破最近的高点
        elif len(closes) >= 10:
            # 简单的趋势判断：如果前10根K线整体下跌，现在突破最近高点
            if closes[-1] > np.max(closes[-10:-1]) and closes[-10] > closes[-1]:
                signal = 'BUY'
                confidence = 0.7
                reason = '看涨CHOCH: 趋势反转信号'
            
            # 看跌CHOCH：在上涨趋势中，价格跌破最近的低点
            elif closes[-1] < np.min(closes[-10:-1]) and closes[-10] < closes[-1]:
                signal = 'SELL'
                confidence = 0.7
                reason = '看跌CHOCH: 趋势反转信号'
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reason': reason,
            'current_price': current_price,
            'prev_high': prev_high,
            'prev_low': prev_low
        }
    
    def calculate_order_blocks_signal(self, enhanced_candlesticks: List[Dict]) -> Dict[str, Any]:
        """
        计算订单块信号
        
        参数:
            enhanced_candlesticks: 增强版K线数据
            
        返回:
            订单块信号分析结果
        """
        if len(enhanced_candlesticks) < 20:
            return {'signal': 'HOLD', 'confidence': 0.0, 'reason': '数据不足'}
        
        # 获取最近的K线数据
        recent_candles = enhanced_candlesticks[-20:]
        
        # 转换为DataFrame便于分析
        df = pd.DataFrame(recent_candles)
        
        # 寻找订单块（强阳线或强阴线后的区域）
        bullish_candles = df[df['close'] > df['open'] * 1.002]  # 强阳线
        bearish_candles = df[df['close'] < df['open'] * 0.998]  # 强阴线
        
        current_price = df['close'].iloc[-1]
        
        signal = 'HOLD'
        confidence = 0.0
        reason = '无明确订单块'
        order_block_level = None
        
        # 检查看涨订单块
        if not bullish_candles.empty:
            # 找到最近的强阳线
            last_bullish = bullish_candles.iloc[-1]
            order_block_high = last_bullish['high']
            order_block_low = last_bullish['low']
            
            # 如果价格回调到订单块区域
            if order_block_low <= current_price <= order_block_high:
                signal = 'BUY'
                # 根据距离订单块中心的位置计算置信度
                distance_from_center = abs(current_price - (order_block_high + order_block_low) / 2)
                max_distance = (order_block_high - order_block_low) / 2
                confidence = max(0.3, 0.8 - (distance_from_center / max_distance) * 0.5)
                reason = f'看涨订单块: 价格回调至 {order_block_low:.2f}-{order_block_high:.2f}'
                order_block_level = (order_block_high + order_block_low) / 2
        
        # 检查看跌订单块
        if not bearish_candles.empty and signal == 'HOLD':
            # 找到最近的强阴线
            last_bearish = bearish_candles.iloc[-1]
            order_block_high = last_bearish['high']
            order_block_low = last_bearish['low']
            
            # 如果价格反弹到订单块区域
            if order_block_low <= current_price <= order_block_high:
                signal = 'SELL'
                # 根据距离订单块中心的位置计算置信度
                distance_from_center = abs(current_price - (order_block_high + order_block_low) / 2)
                max_distance = (order_block_high - order_block_low) / 2
                confidence = max(0.3, 0.8 - (distance_from_center / max_distance) * 0.5)
                reason = f'看跌订单块: 价格反弹至 {order_block_low:.2f}-{order_block_high:.2f}'
                order_block_level = (order_block_high + order_block_low) / 2
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reason': reason,
            'order_block_level': order_block_level,
            'current_price': current_price
        }
    
    def calculate_fvg_signal(self, enhanced_candlesticks: List[Dict]) -> Dict[str, Any]:
        """
        计算FVG（Fair Value Gap）信号
        
        参数:
            enhanced_candlesticks: 增强版K线数据
            
        返回:
            FVG信号分析结果
        """
        if len(enhanced_candlesticks) < 3:
            return {'signal': 'HOLD', 'confidence': 0.0, 'reason': '数据不足'}
        
        # 获取最近的K线数据
        recent_candles = enhanced_candlesticks[-3:]
        
        # 转换为DataFrame便于分析
        df = pd.DataFrame(recent_candles)
        
        # 当前价格
        current_price = df['close'].iloc[-1]
        
        signal = 'HOLD'
        confidence = 0.0
        reason = '无明确FVG'
        fvg_top = None
        fvg_bottom = None
        
        # 检查看涨FVG（Fair Value Gap）
        # 看涨FVG：第三根K线的最高价低于第一根K线的最低价
        if df['high'].iloc[2] < df['low'].iloc[0]:
            fvg_top = df['low'].iloc[0]
            fvg_bottom = df['high'].iloc[2]
            
            # 如果价格在FVG区域内或接近FVG
            if fvg_bottom <= current_price <= fvg_top:
                signal = 'BUY'
                # 根据在FVG中的位置计算置信度
                fvg_center = (fvg_top + fvg_bottom) / 2
                distance_from_center = abs(current_price - fvg_center)
                fvg_size = fvg_top - fvg_bottom
                confidence = max(0.4, 0.9 - (distance_from_center / fvg_size) * 0.4)
                reason = f'看涨FVG: 价格在缺口区域 {fvg_bottom:.2f}-{fvg_top:.2f}'
            
            # 如果价格接近FVG上方
            elif current_price > fvg_top and current_price - fvg_top < fvg_size * 0.5:
                signal = 'BUY'
                confidence = 0.6
                reason = f'看涨FVG: 价格接近缺口上方 {fvg_top:.2f}'
        
        # 检查看跌FVG
        elif df['low'].iloc[2] > df['high'].iloc[0]:
            fvg_top = df['high'].iloc[2]
            fvg_bottom = df['low'].iloc[0]
            
            # 如果价格在FVG区域内或接近FVG
            if fvg_bottom <= current_price <= fvg_top:
                signal = 'SELL'
                # 根据在FVG中的位置计算置信度
                fvg_center = (fvg_top + fvg_bottom) / 2
                distance_from_center = abs(current_price - fvg_center)
                fvg_size = fvg_top - fvg_bottom
                confidence = max(0.4, 0.9 - (distance_from_center / fvg_size) * 0.4)
                reason = f'看跌FVG: 价格在缺口区域 {fvg_bottom:.2f}-{fvg_top:.2f}'
            
            # 如果价格接近FVG下方
            elif current_price < fvg_bottom and fvg_bottom - current_price < fvg_size * 0.5:
                signal = 'SELL'
                confidence = 0.6
                reason = f'看跌FVG: 价格接近缺口下方 {fvg_bottom:.2f}'
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reason': reason,
            'fvg_top': fvg_top,
            'fvg_bottom': fvg_bottom,
            'current_price': current_price
        }
    
    def calculate_liquidity_signal(self, enhanced_candlesticks: List[Dict]) -> Dict[str, Any]:
        """
        计算流动性信号
        
        参数:
            enhanced_candlesticks: 增强版K线数据
            
        返回:
            流动性信号分析结果
        """
        if len(enhanced_candlesticks) < 20:
            return {'signal': 'HOLD', 'confidence': 0.0, 'reason': '数据不足'}
        
        # 获取最近的K线数据
        recent_candles = enhanced_candlesticks[-20:]
        
        # 转换为DataFrame便于分析
        df = pd.DataFrame(recent_candles)
        
        # 寻找流动性区域（高点和低点）
        highs = df['high'].values
        lows = df['low'].values
        current_price = df['close'].iloc[-1]
        
        # 找到近期的高点和低点
        recent_highs = []
        recent_lows = []
        
        # 简单的高点低点识别
        for i in range(1, len(highs) - 1):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                recent_highs.append(highs[i])
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                recent_lows.append(lows[i])
        
        signal = 'HOLD'
        confidence = 0.0
        reason = '无明确流动性信号'
        liquidity_level = None
        
        # 检查看涨流动性信号（价格接近流动性低点后反弹）
        if recent_lows:
            nearest_low = max(recent_lows)  # 最近的最高低点
            distance_to_low = abs(current_price - nearest_low) / current_price
            
            # 如果价格接近流动性低点
            if distance_to_low < 0.005:  # 0.5%范围内
                signal = 'BUY'
                confidence = max(0.3, 0.7 - distance_to_low * 100)
                reason = f'看涨流动性: 接近流动性低点 {nearest_low:.2f}'
                liquidity_level = nearest_low
        
        # 检查看跌流动性信号（价格接近流动性高点后回落）
        if signal == 'HOLD' and recent_highs:
            nearest_high = min(recent_highs)  # 最近的最低高点
            distance_to_high = abs(current_price - nearest_high) / current_price
            
            # 如果价格接近流动性高点
            if distance_to_high < 0.005:  # 0.5%范围内
                signal = 'SELL'
                confidence = max(0.3, 0.7 - distance_to_high * 100)
                reason = f'看跌流动性: 接近流动性高点 {nearest_high:.2f}'
                liquidity_level = nearest_high
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reason': reason,
            'liquidity_level': liquidity_level,
            'current_price': current_price
        }
    
    def calculate_market_microstructure_signal(self, market_depth: List[Dict], time_sales: List[Dict]) -> Dict[str, Any]:
        """
        计算市场微观结构信号
        
        参数:
            market_depth: 市场深度数据
            time_sales: 时间与销售数据
            
        返回:
            市场微观结构信号分析结果
        """
        if not market_depth or not time_sales:
            return {'signal': 'HOLD', 'confidence': 0.0, 'reason': '数据不足'}
        
        # 分析市场深度
        bid_volume = sum([item['bid_volume'] for item in market_depth])
        ask_volume = sum([item['ask_volume'] for item in market_depth])
        
        # 分析时间与销售数据
        buy_volume = sum([item['volume'] for item in time_sales if item['side'] == 'buy'])
        sell_volume = sum([item['volume'] for item in time_sales if item['side'] == 'sell'])
        
        # 计算买卖压力
        depth_imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        sales_imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume)
        
        # 综合信号
        combined_imbalance = (depth_imbalance + sales_imbalance) / 2
        
        signal = 'HOLD'
        confidence = 0.0
        reason = '无明显买卖压力'
        
        # 根据买卖压力判断信号
        if combined_imbalance > 0.2:  # 买压明显
            signal = 'BUY'
            confidence = min(0.8, 0.5 + combined_imbalance)
            reason = f'买压优势: 深度失衡 {depth_imbalance:.2f}, 成交失衡 {sales_imbalance:.2f}'
        elif combined_imbalance < -0.2:  # 卖压明显
            signal = 'SELL'
            confidence = min(0.8, 0.5 - combined_imbalance)
            reason = f'卖压优势: 深度失衡 {depth_imbalance:.2f}, 成交失衡 {sales_imbalance:.2f}'
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reason': reason,
            'depth_imbalance': depth_imbalance,
            'sales_imbalance': sales_imbalance,
            'combined_imbalance': combined_imbalance
        }
    
    def calculate_enhanced_smc_signal(self, enhanced_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        计算综合增强版SMC信号
        
        参数:
            enhanced_data: 增强版数据
            
        返回:
            综合SMC信号分析结果
        """
        # 提取各类数据
        enhanced_candlesticks = enhanced_data.get('enhanced_candlesticks', [])
        market_depth = enhanced_data.get('market_depth', [])
        time_sales = enhanced_data.get('time_sales', [])
        
        # 计算各类信号
        bos_choch_signal = self.calculate_bos_choch_signal(enhanced_candlesticks)
        order_blocks_signal = self.calculate_order_blocks_signal(enhanced_candlesticks)
        fvg_signal = self.calculate_fvg_signal(enhanced_candlesticks)
        liquidity_signal = self.calculate_liquidity_signal(enhanced_candlesticks)
        market_microstructure_signal = self.calculate_market_microstructure_signal(market_depth, time_sales)
        
        # 加权计算综合信号
        signals = [
            (bos_choch_signal, self.signal_weights['bos_choch']),
            (order_blocks_signal, self.signal_weights['order_blocks']),
            (fvg_signal, self.signal_weights['fvg']),
            (liquidity_signal, self.signal_weights['liquidity']),
            (market_microstructure_signal, self.signal_weights['market_microstructure'])
        ]
        
        # 计算加权信号强度
        buy_strength = 0.0
        sell_strength = 0.0
        
        for signal, weight in signals:
            if signal['signal'] == 'BUY':
                buy_strength += signal['confidence'] * weight
            elif signal['signal'] == 'SELL':
                sell_strength += signal['confidence'] * weight
        
        # 确定最终信号
        if buy_strength > sell_strength and buy_strength > self.min_confidence:
            final_signal = 'BUY'
            confidence = buy_strength
            reason = f"综合看涨信号: BOS/CHOCH({bos_choch_signal['confidence']:.2f}) 订单块({order_blocks_signal['confidence']:.2f}) FVG({fvg_signal['confidence']:.2f}) 流动性({liquidity_signal['confidence']:.2f}) 微观结构({market_microstructure_signal['confidence']:.2f})"
        elif sell_strength > buy_strength and sell_strength > self.min_confidence:
            final_signal = 'SELL'
            confidence = sell_strength
            reason = f"综合看跌信号: BOS/CHOCH({bos_choch_signal['confidence']:.2f}) 订单块({order_blocks_signal['confidence']:.2f}) FVG({fvg_signal['confidence']:.2f}) 流动性({liquidity_signal['confidence']:.2f}) 微观结构({market_microstructure_signal['confidence']:.2f})"
        else:
            final_signal = 'HOLD'
            confidence = max(buy_strength, sell_strength)
            reason = "信号不明确或置信度不足"
        
        # 获取当前价格
        current_price = enhanced_candlesticks[-1]['close'] if enhanced_candlesticks else 0
        
        # 计算止损和止盈
        atr = enhanced_data.get('atr', current_price * 0.02)  # 默认2%
        
        if final_signal == 'BUY':
            stop_loss = current_price - (atr * 2)
            take_profit = current_price + (atr * 3)
            risk_reward_ratio = 1.5
        elif final_signal == 'SELL':
            stop_loss = current_price + (atr * 2)
            take_profit = current_price - (atr * 3)
            risk_reward_ratio = 1.5
        else:
            stop_loss = None
            take_profit = None
            risk_reward_ratio = 0
        
        return {
            'signal': final_signal,
            'confidence': confidence,
            'reason': reason,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward_ratio': risk_reward_ratio,
            'current_price': current_price,
            'signal_breakdown': {
                'bos_choch': bos_choch_signal,
                'order_blocks': order_blocks_signal,
                'fvg': fvg_signal,
                'liquidity': liquidity_signal,
                'market_microstructure': market_microstructure_signal
            },
            'strength_scores': {
                'buy_strength': buy_strength,
                'sell_strength': sell_strength
            },
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

# 使用示例
if __name__ == "__main__":
    # 配置
    config = {
        'enhanced_smc_signal_weights': {
            'bos_choch': 0.3,
            'order_blocks': 0.25,
            'fvg': 0.2,
            'liquidity': 0.15,
            'market_microstructure': 0.1
        },
        'enhanced_smc_min_confidence': 0.6,
        'enhanced_data_weight': 0.7,
        'market_depth_weight': 0.15,
        'time_sales_weight': 0.1,
        'market_sentiment_weight': 0.05
    }
    
    # 创建增强版SMC信号计算器
    calculator = EnhancedSMCSignalCalculator(config)
    
    # 这里应该使用真实的增强版数据
    # 为了演示，我们使用空字典
    enhanced_data = {}
    
    # 计算信号
    signal = calculator.calculate_enhanced_smc_signal(enhanced_data)
    
    print("增强版SMC信号计算结果:")
    print(f"信号: {signal['signal']}")
    print(f"置信度: {signal['confidence']:.2f}")
    print(f"原因: {signal['reason']}")
    
    if signal['signal'] != 'HOLD':
        print(f"止损: {signal['stop_loss']}")
        print(f"止盈: {signal['take_profit']}")
        print(f"风险回报比: {signal['risk_reward_ratio']:.2f}")