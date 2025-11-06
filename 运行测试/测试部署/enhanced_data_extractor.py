#!/usr/bin/env python3
"""
增强版原始数据提取器 - 基于DeepSeek建议的改进数据结构
实现高优先级和中优先级的数据增强功能
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

class EnhancedDataExtractor:
    """增强版数据提取器，提供更丰富的原始数据供AI计算SMC结构"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
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
        try:
            enhanced_data = {
                "timestamp": datetime.now().isoformat(),
                "enhanced_candlesticks": self._extract_enhanced_candlesticks(ohlc_data),
                "swing_points": self._extract_swing_points(ohlc_data),
                "volume_analysis": self._extract_enhanced_volume_analysis(ohlc_data, volume_data),
                "market_depth": self._extract_market_depth(market_depth) if market_depth else [],
                "time_sales": self._extract_time_sales(time_sales) if time_sales else [],
                "market_sentiment": market_sentiment or {},
                "multi_timeframe_context": self._extract_multi_timeframe_context(ohlc_data),
                "liquidity_levels": self._extract_enhanced_liquidity_levels(ohlc_data),
                "price_movements": self._extract_enhanced_price_movements(ohlc_data),
                "order_flow_imbalance": self._calculate_order_flow_imbalance(time_sales) if time_sales else {},
                "market_microstructure": self._analyze_market_microstructure(ohlc_data, market_depth)
            }
            
            self.logger.info(f"🔍 提取增强版原始数据: K线={len(enhanced_data['enhanced_candlesticks'])}, "
                            f"摆动点={len(enhanced_data['swing_points'])}, "
                            f"市场深度={len(enhanced_data['market_depth'])}, "
                            f"时间销售={len(enhanced_data['time_sales'])}")
            
            return enhanced_data
            
        except Exception as e:
            self.logger.error(f"提取增强版原始数据失败: {str(e)}")
            return {}
    
    def _extract_enhanced_candlesticks(self, ohlc_data: List[Dict]) -> List[Dict]:
        """提取增强版K线数据，包含body_size、wick_size等新字段"""
        enhanced_candles = []
        
        for i, candle in enumerate(ohlc_data):
            if i == 0:
                prev_close = candle['close']
            else:
                prev_close = ohlc_data[i-1]['close']
            
            # 计算K线内部结构
            open_price = candle['open']
            high_price = candle['high']
            low_price = candle['low']
            close_price = candle['close']
            volume = candle['volume']
            
            body_size = abs(close_price - open_price)
            upper_wick = high_price - max(open_price, close_price)
            lower_wick = min(open_price, close_price) - low_price
            total_range = high_price - low_price
            
            # 计算实体位置
            if body_size == 0:
                body_position = "doji"
            else:
                body_midpoint = (open_price + close_price) / 2
                if body_midpoint > (high_price + low_price) / 2:
                    body_position = "upper"
                elif body_midpoint < (high_price + low_price) / 2:
                    body_position = "lower"
                else:
                    body_position = "middle"
            
            # 计算跳空
            gap_size = open_price - prev_close
            gap_direction = "up" if gap_size > 0 else "down" if gap_size < 0 else "none"
            
            # 计算成交量分布 (简化版)
            volume_profile = self._calculate_volume_profile(candle)
            
            enhanced_candle = {
                "timestamp": candle['timestamp'],
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
                "timeframe": candle.get('timeframe', '1h'),
                
                # 新增字段 - 高优先级改进
                "body_size": body_size,
                "upper_wick": upper_wick,
                "lower_wick": lower_wick,
                "body_position": body_position,
                "body_ratio": body_size / total_range if total_range > 0 else 0,
                "previous_close": prev_close,
                "gap_size": abs(gap_size),
                "gap_direction": gap_direction,
                
                # 成交量分析
                "volume_profile": volume_profile,
                
                # 价格行为标记
                "engulfing": self._detect_engulfing_pattern(candle, ohlc_data, i),
                "rejection": upper_wick > body_size * 2 or lower_wick > body_size * 2,
                "inside_bar": self._detect_inside_bar(candle, ohlc_data, i)
            }
            
            enhanced_candles.append(enhanced_candle)
        
        return enhanced_candles
    
    def _calculate_volume_profile(self, candle: Dict) -> Dict:
        """计算简化的成交量分布"""
        high = candle['high']
        low = candle['low']
        close = candle['close']
        volume = candle['volume']
        
        # 简化版POC (Point of Control) - 假设成交量在收盘价附近最高
        poc_price = close
        value_area_range = (high - low) * 0.7  # 价值区域占70%的范围
        
        return {
            "poc_price": poc_price,
            "value_area_high": min(high, poc_price + value_area_range/2),
            "value_area_low": max(low, poc_price - value_area_range/2),
            "value_area_volume_pct": 0.7  # 70%的成交量在价值区域内
        }
    
    def _detect_engulfing_pattern(self, candle: Dict, ohlc_data: List[Dict], index: int) -> str:
        """检测吞没形态"""
        if index == 0:
            return "none"
        
        prev_candle = ohlc_data[index-1]
        curr_open, curr_close = candle['open'], candle['close']
        prev_open, prev_close = prev_candle['open'], prev_candle['close']
        
        # 看涨吞没
        if (curr_open < prev_close and curr_close > prev_open and 
            abs(curr_close - curr_open) > abs(prev_close - prev_open)):
            return "bullish_engulfing"
        
        # 看跌吞没
        if (curr_open > prev_close and curr_close < prev_open and 
            abs(curr_close - curr_open) > abs(prev_close - prev_open)):
            return "bearish_engulfing"
        
        return "none"
    
    def _detect_inside_bar(self, candle: Dict, ohlc_data: List[Dict], index: int) -> bool:
        """检测内包线"""
        if index == 0:
            return False
        
        prev_candle = ohlc_data[index-1]
        return (candle['high'] < prev_candle['high'] and 
                candle['low'] > prev_candle['low'])
    
    def _extract_swing_points(self, ohlc_data: List[Dict]) -> List[Dict]:
        """提取摆动点数据"""
        swing_points = []
        
        # 简化版摆动点检测 - 使用局部极值
        for i in range(2, len(ohlc_data) - 2):
            prev_high = ohlc_data[i-1]['high']
            curr_high = ohlc_data[i]['high']
            next_high = ohlc_data[i+1]['high']
            
            prev_low = ohlc_data[i-1]['low']
            curr_low = ohlc_data[i]['low']
            next_low = ohlc_data[i+1]['low']
            
            # 摆动高点
            if curr_high > prev_high and curr_high > next_high:
                strength = min(
                    (curr_high - prev_low) / curr_high,
                    (curr_high - next_low) / curr_high
                ) * 10  # 转换为0-10范围
                
                swing_points.append({
                    "timestamp": ohlc_data[i]['timestamp'],
                    "price": curr_high,
                    "type": "swing_high",
                    "strength": min(strength, 10),
                    "confirmed": True
                })
            
            # 摆动低点
            if curr_low < prev_low and curr_low < next_low:
                strength = min(
                    (prev_high - curr_low) / curr_low,
                    (next_high - curr_low) / curr_low
                ) * 10  # 转换为0-10范围
                
                swing_points.append({
                    "timestamp": ohlc_data[i]['timestamp'],
                    "price": curr_low,
                    "type": "swing_low",
                    "strength": min(strength, 10),
                    "confirmed": True
                })
        
        return swing_points
    
    def _extract_enhanced_volume_analysis(self, ohlc_data: List[Dict], volume_data: List[Dict]) -> List[Dict]:
        """提取增强版成交量分析"""
        enhanced_volume = []
        
        # 计算成交量移动平均
        volumes = [candle['volume'] for candle in ohlc_data]
        volume_ma = self._calculate_sma(volumes, 20)  # 20期移动平均
        
        for i, candle in enumerate(ohlc_data):
            current_volume = candle['volume']
            avg_volume = volume_ma[i] if i < len(volume_ma) else current_volume
            
            # 成交量异常检测
            volume_spike = current_volume / avg_volume if avg_volume > 0 else 1
            is_volume_spike = volume_spike > 2.0  # 超过平均2倍认为是异常
            
            # 成交量与价格变动关系
            price_change = abs(candle['close'] - candle['open']) / candle['open'] * 100
            volume_price_efficiency = price_change / (current_volume / 1000) if current_volume > 0 else 0
            
            enhanced_volume.append({
                "timestamp": candle['timestamp'],
                "volume": current_volume,
                "volume_avg": avg_volume,
                "volume_ratio": current_volume / avg_volume if avg_volume > 0 else 1,
                "volume_spike": is_volume_spike,
                "volume_spike_magnitude": volume_spike,
                "volume_price_efficiency": volume_price_efficiency,
                "buying_pressure": self._estimate_buying_pressure(candle),
                "volume_trend": self._analyze_volume_trend(volumes, i)
            })
        
        return enhanced_volume
    
    def _estimate_buying_pressure(self, candle: Dict) -> float:
        """估算买压 (0-1范围)"""
        open_price = candle['open']
        close_price = candle['close']
        high_price = candle['high']
        low_price = candle['low']
        
        # 基于K线收盘位置和影线长度估算买压
        total_range = high_price - low_price
        if total_range == 0:
            return 0.5
        
        close_position = (close_price - low_price) / total_range
        upper_wick_ratio = (high_price - max(open_price, close_price)) / total_range
        lower_wick_ratio = (min(open_price, close_price) - low_price) / total_range
        
        # 收盘位置越高，下影线越长，买压越大
        buying_pressure = close_position * 0.6 + lower_wick_ratio * 0.4
        
        # 上影线过长会减少买压
        if upper_wick_ratio > 0.3:
            buying_pressure *= (1 - upper_wick_ratio)
        
        return max(0, min(1, buying_pressure))
    
    def _analyze_volume_trend(self, volumes: List[float], index: int) -> str:
        """分析成交量趋势"""
        if index < 5:
            return "insufficient_data"
        
        recent_volumes = volumes[max(0, index-5):index+1]
        
        # 简单线性回归判断趋势
        x = np.arange(len(recent_volumes))
        y = np.array(recent_volumes)
        
        if len(x) > 1 and np.var(y) > 0:
            slope = np.polyfit(x, y, 1)[0]
            if slope > 0.1:
                return "increasing"
            elif slope < -0.1:
                return "decreasing"
        
        return "stable"
    
    def _extract_market_depth(self, market_depth: List[Dict]) -> List[Dict]:
        """提取市场深度数据"""
        if not market_depth:
            return []
        
        enhanced_depth = []
        for depth in market_depth:
            bid_price = depth.get('bid_price', 0)
            ask_price = depth.get('ask_price', 0)
            bid_volume = depth.get('bid_volume', 0)
            ask_volume = depth.get('ask_volume', 0)
            
            # 计算不平衡比率
            total_volume = bid_volume + ask_volume
            imbalance_ratio = bid_volume / ask_volume if ask_volume > 0 else float('inf')
            
            # 计算价差百分比
            spread = ask_price - bid_price
            spread_pct = (spread / bid_price * 100) if bid_price > 0 else 0
            
            enhanced_depth.append({
                "timestamp": depth.get('timestamp'),
                "bid_price": bid_price,
                "ask_price": ask_price,
                "bid_volume": bid_volume,
                "ask_volume": ask_volume,
                "total_volume": total_volume,
                "imbalance_ratio": imbalance_ratio,
                "spread": spread,
                "spread_percentage": spread_pct,
                "dominant_side": "bid" if bid_volume > ask_volume else "ask",
                "liquidity_score": total_volume / spread if spread > 0 else 0
            })
        
        return enhanced_depth
    
    def _extract_time_sales(self, time_sales: List[Dict]) -> List[Dict]:
        """提取时间与销售数据"""
        if not time_sales:
            return []
        
        enhanced_sales = []
        for sale in time_sales:
            enhanced_sales.append({
                "timestamp": sale.get('timestamp'),
                "price": sale.get('price'),
                "volume": sale.get('volume'),
                "side": sale.get('side', 'unknown'),
                "liquidity_removed": sale.get('liquidity_removed', False),
                "aggressive": sale.get('aggressive', False),  # 是否主动成交
                "large_order": sale.get('volume', 0) > 10  # 大单标记
            })
        
        return enhanced_sales
    
    def _extract_multi_timeframe_context(self, ohlc_data: List[Dict]) -> Dict:
        """提取多时间框架上下文"""
        if len(ohlc_data) < 50:
            return {"alignment_score": 0, "key_levels_aligned": False}
        
        # 简化版多时间框架分析
        recent_candles = ohlc_data[-20:]
        older_candles = ohlc_data[-50:-20]
        
        # 计算短期和长期趋势
        short_trend = self._calculate_trend(recent_candles)
        long_trend = self._calculate_trend(older_candles)
        
        # 计算趋势一致性
        alignment_score = 1.0 if short_trend == long_trend else 0.5
        
        # 识别关键水平
        recent_highs = [c['high'] for c in recent_candles]
        recent_lows = [c['low'] for c in recent_candles]
        older_highs = [c['high'] for c in older_candles]
        older_lows = [c['low'] for c in older_candles]
        
        key_levels_aligned = (
            max(recent_highs) < max(older_highs) * 1.02 and
            min(recent_lows) > min(older_lows) * 0.98
        )
        
        return {
            "short_term_trend": short_trend,
            "long_term_trend": long_trend,
            "alignment_score": alignment_score,
            "key_levels_aligned": key_levels_aligned,
            "trend_strength": abs(alignment_score - 0.5) * 2
        }
    
    def _calculate_trend(self, candles: List[Dict]) -> str:
        """计算价格趋势"""
        if len(candles) < 3:
            return "neutral"
        
        # 使用线性回归计算趋势
        closes = [c['close'] for c in candles]
        x = np.arange(len(closes))
        
        if len(x) > 1 and np.var(closes) > 0:
            slope = np.polyfit(x, closes, 1)[0]
            avg_price = np.mean(closes)
            
            # 将斜率转换为百分比
            slope_pct = slope / avg_price * 100
            
            if slope_pct > 0.1:
                return "bullish"
            elif slope_pct < -0.1:
                return "bearish"
        
        return "neutral"
    
    def _extract_enhanced_liquidity_levels(self, ohlc_data: List[Dict]) -> List[Dict]:
        """提取增强版流动性水平"""
        # 基于摆动点和成交量识别流动性水平
        swing_points = self._extract_swing_points(ohlc_data)
        liquidity_levels = []
        
        for point in swing_points:
            # 计算流动性强度 - 基于摆动点强度和附近成交量
            point_price = point['price']
            point_strength = point['strength']
            
            # 查找附近的成交量
            nearby_volumes = []
            for candle in ohlc_data:
                if abs(candle['high'] - point_price) < (point_price * 0.005):  # 0.5%范围内
                    nearby_volumes.append(candle['volume'])
            
            avg_nearby_volume = np.mean(nearby_volumes) if nearby_volumes else 0
            
            # 计算流动性强度
            liquidity_strength = point_strength * (1 + np.log10(avg_nearby_volume + 1) / 10)
            liquidity_strength = min(10, liquidity_strength)  # 限制最大值
            
            liquidity_levels.append({
                "price": point_price,
                "strength": liquidity_strength,
                "type": "support" if point['type'] == 'swing_low' else "resistance",
                "timestamp": point['timestamp'],
                "tested": self._check_level_tested(ohlc_data, point_price, point['timestamp']),
                "volume_confirmation": avg_nearby_volume
            })
        
        # 按强度排序
        liquidity_levels.sort(key=lambda x: x['strength'], reverse=True)
        
        return liquidity_levels[:10]  # 返回前10个最强的流动性水平
    
    def _check_level_tested(self, ohlc_data: List[Dict], level_price: float, level_time: str) -> bool:
        """检查水平是否被测试过"""
        level_index = next((i for i, c in enumerate(ohlc_data) if c['timestamp'] == level_time), -1)
        if level_index == -1 or level_index >= len(ohlc_data) - 1:
            return False
        
        # 检查水平形成后是否被测试
        for candle in ohlc_data[level_index + 1:]:
            if candle['low'] <= level_price <= candle['high']:
                return True
        
        return False
    
    def _extract_enhanced_price_movements(self, ohlc_data: List[Dict]) -> List[Dict]:
        """提取增强版价格变动数据"""
        price_movements = []
        
        for i in range(1, len(ohlc_data)):
            prev_candle = ohlc_data[i-1]
            curr_candle = ohlc_data[i]
            
            # 计算价格变动
            price_change = curr_candle['close'] - prev_candle['close']
            price_change_pct = (price_change / prev_candle['close']) * 100
            
            # 计算变动持续时间 (简化为1根K线)
            duration = 60  # 假设1小时K线，转换为分钟
            
            # 计算变动强度
            high_low_range = curr_candle['high'] - curr_candle['low']
            movement_strength = abs(price_change_pct) / (high_low_range / prev_candle['close'] * 100) if high_low_range > 0 else 0
            
            # 确定方向
            direction = "up" if price_change > 0 else "down" if price_change < 0 else "sideways"
            
            # 计算成交量确认
            volume_confirmation = curr_candle['volume'] / prev_candle['volume'] if prev_candle['volume'] > 0 else 1
            
            price_movements.append({
                "timestamp": curr_candle['timestamp'],
                "direction": direction,
                "magnitude": abs(price_change_pct),
                "duration": duration,
                "strength": movement_strength,
                "volume_confirmation": volume_confirmation,
                "gap": curr_candle['open'] - prev_candle['close'],
                "gap_pct": ((curr_candle['open'] - prev_candle['close']) / prev_candle['close']) * 100,
                "volatility": high_low_range / prev_candle['close'] * 100
            })
        
        return price_movements
    
    def _calculate_order_flow_imbalance(self, time_sales: List[Dict]) -> Dict:
        """计算订单流不平衡"""
        if not time_sales:
            return {}
        
        # 统计买卖成交量
        buy_volume = sum(sale['volume'] for sale in time_sales if sale['side'] == 'buy')
        sell_volume = sum(sale['volume'] for sale in time_sales if sale['side'] == 'sell')
        total_volume = buy_volume + sell_volume
        
        if total_volume == 0:
            return {"imbalance_ratio": 1.0, "dominant_side": "neutral"}
        
        # 计算不平衡比率
        imbalance_ratio = buy_volume / sell_volume if sell_volume > 0 else float('inf')
        dominant_side = "buy" if buy_volume > sell_volume else "sell"
        
        # 计算大单比例
        large_order_threshold = 10  # 定义大单阈值
        large_orders = [sale for sale in time_sales if sale['volume'] > large_order_threshold]
        large_order_ratio = len(large_orders) / len(time_sales) if time_sales else 0
        
        return {
            "imbalance_ratio": imbalance_ratio,
            "dominant_side": dominant_side,
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "total_volume": total_volume,
            "large_order_ratio": large_order_ratio,
            "aggressive_buy_ratio": sum(1 for sale in time_sales if sale['side'] == 'buy' and sale.get('aggressive', False)) / len(time_sales) if time_sales else 0
        }
    
    def _analyze_market_microstructure(self, ohlc_data: List[Dict], market_depth: Optional[List[Dict]]) -> Dict:
        """分析市场微观结构"""
        if not ohlc_data:
            return {}
        
        recent_candles = ohlc_data[-10:]  # 最近10根K线
        
        # 计算价格效率
        price_efficiency = self._calculate_price_efficiency(recent_candles)
        
        # 计算流动性分布
        liquidity_distribution = self._calculate_liquidity_distribution(market_depth) if market_depth else {}
        
        # 计算市场压力
        market_pressure = self._calculate_market_pressure(recent_candles)
        
        return {
            "price_efficiency": price_efficiency,
            "liquidity_distribution": liquidity_distribution,
            "market_pressure": market_pressure,
            "microstructure_score": (price_efficiency + market_pressure) / 2
        }
    
    def _calculate_price_efficiency(self, candles: List[Dict]) -> float:
        """计算价格效率 (0-1范围)"""
        if len(candles) < 3:
            return 0.5
        
        # 计算价格路径效率 - 实际位移与总路径的比率
        total_path = 0
        for i in range(1, len(candles)):
            total_path += abs(candles[i]['close'] - candles[i-1]['close'])
        
        net_displacement = abs(candles[-1]['close'] - candles[0]['close'])
        
        if total_path == 0:
            return 1.0
        
        return net_displacement / total_path
    
    def _calculate_liquidity_distribution(self, market_depth: List[Dict]) -> Dict:
        """计算流动性分布"""
        if not market_depth:
            return {}
        
        # 计算买卖盘流动性分布
        bid_volumes = [depth['bid_volume'] for depth in market_depth]
        ask_volumes = [depth['ask_volume'] for depth in market_depth]
        
        return {
            "bid_liquidity": sum(bid_volumes),
            "ask_liquidity": sum(ask_volumes),
            "liquidity_ratio": sum(bid_volumes) / sum(ask_volumes) if sum(ask_volumes) > 0 else float('inf'),
            "liquidity_imbalance": abs(sum(bid_volumes) - sum(ask_volumes)) / (sum(bid_volumes) + sum(ask_volumes)) if (sum(bid_volumes) + sum(ask_volumes)) > 0 else 0
        }
    
    def _calculate_market_pressure(self, candles: List[Dict]) -> float:
        """计算市场压力 (0-1范围)"""
        if len(candles) < 3:
            return 0.5
        
        # 基于价格变动和成交量计算市场压力
        price_changes = []
        volume_weighted_changes = []
        
        for i in range(1, len(candles)):
            price_change = abs(candles[i]['close'] - candles[i-1]['close']) / candles[i-1]['close']
            volume = candles[i]['volume']
            
            price_changes.append(price_change)
            volume_weighted_changes.append(price_change * volume)
        
        if not price_changes:
            return 0.5
        
        # 计算成交量加权的平均价格变动
        avg_price_change = np.mean(price_changes)
        volume_weighted_avg_change = np.sum(volume_weighted_changes) / np.sum([c['volume'] for c in candles[1:]])
        
        # 结合简单平均和成交量加权平均
        combined_pressure = (avg_price_change + volume_weighted_avg_change) / 2
        
        # 转换为0-1范围，使用sigmoid函数
        return 1 / (1 + np.exp(-10 * (combined_pressure - 0.01)))
    
    def _calculate_sma(self, data: List[float], period: int) -> List[float]:
        """计算简单移动平均"""
        if len(data) < period:
            return [np.mean(data)] * len(data)
        
        sma = []
        for i in range(len(data)):
            if i < period - 1:
                sma.append(np.mean(data[:i+1]))
            else:
                sma.append(np.mean(data[i-period+1:i+1]))
        
        return sma

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
    
    # 提取增强版数据
    extractor = EnhancedDataExtractor()
    enhanced_data = extractor.extract_enhanced_raw_data(
        ohlc_data=ohlc_data,
        volume_data=[],
        market_depth=market_depth,
        time_sales=time_sales,
        market_sentiment=market_sentiment
    )
    
    # 保存结果
    with open("enhanced_raw_data_example.json", "w") as f:
        # 转换numpy类型为Python原生类型
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            else:
                return obj
        
        json.dump(convert_numpy_types(enhanced_data), f, indent=2)
    
    print("✅ 增强版原始数据示例已生成并保存到 enhanced_raw_data_example.json")
    print(f"📊 数据包含: {len(enhanced_data['enhanced_candlesticks'])}根增强K线, "
          f"{len(enhanced_data['market_depth'])}个市场深度点, "
          f"{len(enhanced_data['time_sales'])}笔交易记录")