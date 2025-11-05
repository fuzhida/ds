"""
TradingView SMC检测逻辑的Python实现
从Pine Script转换为Python，用于替换paxg_trading_bot.py中的SMC检测功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum


class MarketStructureType(Enum):
    """市场结构类型枚举"""
    BOS = "BOS"  # Break of Structure
    CHOCH = "CHOCH"  # Change of Character
    EQH = "EQH"  # Equal High
    EQL = "EQL"  # Equal Low


class TrendDirection(Enum):
    """趋势方向枚举"""
    UP = "UP"
    DOWN = "DOWN"
    SIDEWAYS = "SIDEWAYS"


@dataclass
class SwingPoint:
    """摆动点数据结构"""
    index: int
    price: float
    is_high: bool
    timestamp: pd.Timestamp


@dataclass
class OrderBlock:
    """订单块数据结构"""
    index: int
    top: float
    bottom: float
    is_bullish: bool
    timestamp: pd.Timestamp
    strength: float = 1.0


@dataclass
class FairValueGap:
    """公允价值缺口数据结构"""
    start_index: int
    end_index: int
    top: float
    bottom: float
    is_bullish: bool
    timestamp: pd.Timestamp
    filled: bool = False


@dataclass
class LiquidityLevel:
    """流动性水平数据结构"""
    index: int
    price: float
    is_sweep: bool
    timestamp: pd.Timestamp
    strength: float = 1.0


class SMCDetector:
    """
    TradingView SMC检测逻辑的Python实现
    
    该类实现了从TradingView Pine Script转换而来的SMC（Smart Money Concepts）检测逻辑，
    包括摆动点检测、市场结构分析、订单块识别、公允价值缺口检测等功能。
    """
    
    def __init__(self, 
                 swing_length: int = 5,
                 structure_lookback: int = 50,
                 fvg_threshold: float = 0.5,
                 ob_threshold: float = 0.3,
                 liquidity_threshold: float = 0.2):
        """
        初始化SMC检测器
        
        参数:
            swing_length: 摆动点检测的回看长度
            structure_lookback: 市场结构分析的回看长度
            fvg_threshold: 公允价值缺口检测阈值（ATR倍数）
            ob_threshold: 订单块检测阈值（ATR倍数）
            liquidity_threshold: 流动性检测阈值（ATR倍数）
        """
        self.swing_length = swing_length
        self.structure_lookback = structure_lookback
        self.fvg_threshold = fvg_threshold
        self.ob_threshold = ob_threshold
        self.liquidity_threshold = liquidity_threshold
        
        # 存储检测结果
        self.swing_points: List[SwingPoint] = []
        self.market_structure: List[Dict] = []
        self.order_blocks: List[OrderBlock] = []
        self.fair_value_gaps: List[FairValueGap] = []
        self.liquidity_levels: List[LiquidityLevel] = []
        
        # 趋势状态
        self.current_trend = TrendDirection.SIDEWAYS
        self.last_bos_price = None
        self.last_choch_price = None
    
    def detect_swing_points(self, df: pd.DataFrame) -> List[SwingPoint]:
        """
        检测摆动点（高点和低点）
        
        参数:
            df: 包含OHLC数据的DataFrame
            
        返回:
            摆动点列表
        """
        self.swing_points = []
        
        if len(df) < self.swing_length * 2 + 1:
            return self.swing_points
            
        highs = df['high'].values
        lows = df['low'].values
        timestamps = df.index
        
        # 检测摆动高点
        for i in range(self.swing_length, len(highs) - self.swing_length):
            is_swing_high = True
            current_high = highs[i]
            
            # 检查当前点是否是最高点
            for j in range(i - self.swing_length, i + self.swing_length + 1):
                if j != i and highs[j] >= current_high:
                    is_swing_high = False
                    break
                    
            if is_swing_high:
                self.swing_points.append(SwingPoint(
                    index=i,
                    price=current_high,
                    is_high=True,
                    timestamp=timestamps[i]
                ))
        
        # 检测摆动低点
        for i in range(self.swing_length, len(lows) - self.swing_length):
            is_swing_low = True
            current_low = lows[i]
            
            # 检查当前点是否是最低点
            for j in range(i - self.swing_length, i + self.swing_length + 1):
                if j != i and lows[j] <= current_low:
                    is_swing_low = False
                    break
                    
            if is_swing_low:
                self.swing_points.append(SwingPoint(
                    index=i,
                    price=current_low,
                    is_high=False,
                    timestamp=timestamps[i]
                ))
        
        # 按索引排序
        self.swing_points.sort(key=lambda x: x.index)
        
        return self.swing_points
    
    def detect_market_structure(self, df: pd.DataFrame) -> List[Dict]:
        """
        检测市场结构（BOS/CHOCH/EQH/EQL）
        
        参数:
            df: 包含OHLC数据的DataFrame
            
        返回:
            市场结构事件列表
        """
        self.market_structure = []
        
        if len(self.swing_points) < 3:
            return self.market_structure
            
        closes = df['close'].values
        timestamps = df.index
        
        # 分析最近的市场结构
        lookback_points = [p for p in self.swing_points if p.index >= len(df) - self.structure_lookback]
        
        if len(lookback_points) < 3:
            return self.market_structure
        
        # 检测BOS/CHOCH
        for i in range(2, len(lookback_points)):
            prev_point = lookback_points[i-2]
            middle_point = lookback_points[i-1]
            current_point = lookback_points[i]
            
            # 检测上升趋势中的BOS
            if (prev_point.is_high == False and middle_point.is_high == True and 
                current_point.is_high == False and 
                current_point.price > prev_point.price):
                
                # 检查是否是BOS或CHOCH
                if self.current_trend == TrendDirection.UP:
                    structure_type = MarketStructureType.BOS
                else:
                    structure_type = MarketStructureType.CHOCH
                    self.current_trend = TrendDirection.UP
                
                self.market_structure.append({
                    'type': structure_type.value,
                    'index': current_point.index,
                    'price': current_point.price,
                    'timestamp': timestamps[current_point.index],
                    'trend_direction': TrendDirection.UP.value,
                    'break_point': middle_point.price
                })
                
                self.last_bos_price = current_point.price if structure_type == MarketStructureType.BOS else self.last_bos_price
                self.last_choch_price = current_point.price if structure_type == MarketStructureType.CHOCH else self.last_choch_price
            
            # 检测下降趋势中的BOS
            elif (prev_point.is_high == True and middle_point.is_high == False and 
                  current_point.is_high == True and 
                  current_point.price < prev_point.price):
                
                # 检查是否是BOS或CHOCH
                if self.current_trend == TrendDirection.DOWN:
                    structure_type = MarketStructureType.BOS
                else:
                    structure_type = MarketStructureType.CHOCH
                    self.current_trend = TrendDirection.DOWN
                
                self.market_structure.append({
                    'type': structure_type.value,
                    'index': current_point.index,
                    'price': current_point.price,
                    'timestamp': timestamps[current_point.index],
                    'trend_direction': TrendDirection.DOWN.value,
                    'break_point': middle_point.price
                })
                
                self.last_bos_price = current_point.price if structure_type == MarketStructureType.BOS else self.last_bos_price
                self.last_choch_price = current_point.price if structure_type == MarketStructureType.CHOCH else self.last_choch_price
        
        # 检测EQH/EQL
        self._detect_equal_highs_lows(df)
        
        return self.market_structure
    
    def _detect_equal_highs_lows(self, df: pd.DataFrame):
        """检测等高点和等低点"""
        if len(self.swing_points) < 4:
            return
            
        # 检测等高点
        swing_highs = [p for p in self.swing_points if p.is_high]
        for i in range(len(swing_highs) - 1):
            for j in range(i + 1, min(i + 4, len(swing_highs))):  # 检查后面3个点
                price_diff = abs(swing_highs[i].price - swing_highs[j].price) / swing_highs[i].price
                if price_diff < 0.005:  # 0.5%以内视为等高
                    self.market_structure.append({
                        'type': MarketStructureType.EQH.value,
                        'index': swing_highs[j].index,
                        'price': swing_highs[j].price,
                        'timestamp': df.index[swing_highs[j].index],
                        'reference_index': swing_highs[i].index,
                        'reference_price': swing_highs[i].price
                    })
        
        # 检测等低点
        swing_lows = [p for p in self.swing_points if not p.is_high]
        for i in range(len(swing_lows) - 1):
            for j in range(i + 1, min(i + 4, len(swing_lows))):  # 检查后面3个点
                price_diff = abs(swing_lows[i].price - swing_lows[j].price) / swing_lows[i].price
                if price_diff < 0.005:  # 0.5%以内视为等低
                    self.market_structure.append({
                        'type': MarketStructureType.EQL.value,
                        'index': swing_lows[j].index,
                        'price': swing_lows[j].price,
                        'timestamp': df.index[swing_lows[j].index],
                        'reference_index': swing_lows[i].index,
                        'reference_price': swing_lows[i].price
                    })
    
    def detect_order_blocks(self, df: pd.DataFrame) -> List[OrderBlock]:
        """
        检测订单块
        
        参数:
            df: 包含OHLC数据的DataFrame
            
        返回:
            订单块列表
        """
        self.order_blocks = []
        
        if len(self.market_structure) == 0:
            return self.order_blocks
            
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        opens = df['open'].values
        timestamps = df.index
        
        # 计算ATR用于确定订单块阈值
        atr = self._calculate_atr(df, 14)
        
        # 检测BOS/CHOCH前的订单块
        for structure in self.market_structure:
            if structure['type'] not in [MarketStructureType.BOS.value, MarketStructureType.CHOCH.value]:
                continue
                
            structure_index = structure['index']
            if structure_index < 2:
                continue
                
            # 检测看涨订单块（下降趋势中的CHOCH或上升趋势中的BOS）
            if structure['trend_direction'] == TrendDirection.UP.value:
                # 查找形成结构前的最后一个下跌K线
                for i in range(structure_index - 2, max(0, structure_index - 6), -1):
                    if closes[i] < opens[i]:  # 阴线
                        # 检查这个K线后的价格是否继续下跌
                        if i + 1 < len(df) and closes[i + 1] < closes[i]:
                            # 这是一个潜在的看涨订单块
                            ob_strength = min(1.0, (highs[i] - lows[i]) / (atr * self.ob_threshold))
                            
                            self.order_blocks.append(OrderBlock(
                                index=i,
                                top=highs[i],
                                bottom=lows[i],
                                is_bullish=True,
                                timestamp=timestamps[i],
                                strength=ob_strength
                            ))
                            break
            
            # 检测看跌订单块（上升趋势中的CHOCH或下降趋势中的BOS）
            elif structure['trend_direction'] == TrendDirection.DOWN.value:
                # 查找形成结构前的最后一个上涨K线
                for i in range(structure_index - 2, max(0, structure_index - 6), -1):
                    if closes[i] > opens[i]:  # 阳线
                        # 检查这个K线后的价格是否继续上涨
                        if i + 1 < len(df) and closes[i + 1] > closes[i]:
                            # 这是一个潜在的看跌订单块
                            ob_strength = min(1.0, (highs[i] - lows[i]) / (atr * self.ob_threshold))
                            
                            self.order_blocks.append(OrderBlock(
                                index=i,
                                top=highs[i],
                                bottom=lows[i],
                                is_bullish=False,
                                timestamp=timestamps[i],
                                strength=ob_strength
                            ))
                            break
        
        return self.order_blocks
    
    def detect_fair_value_gaps(self, df: pd.DataFrame) -> List[FairValueGap]:
        """
        检测公允价值缺口（FVG）
        
        参数:
            df: 包含OHLC数据的DataFrame
            
        返回:
            公允价值缺口列表
        """
        self.fair_value_gaps = []
        
        if len(df) < 3:
            return self.fair_value_gaps
            
        highs = df['high'].values
        lows = df['low'].values
        timestamps = df.index
        
        # 计算ATR用于确定FVG阈值
        atr = self._calculate_atr(df, 14)
        
        # 检测FVG
        for i in range(2, len(df)):
            # 检测看涨FVG（三根K线中，中间K线的低点高于前一根K线的高点）
            if lows[i-1] > highs[i-2]:
                gap_size = lows[i-1] - highs[i-2]
                if gap_size > atr * self.fvg_threshold:
                    self.fair_value_gaps.append(FairValueGap(
                        start_index=i-2,
                        end_index=i-1,
                        top=lows[i-1],
                        bottom=highs[i-2],
                        is_bullish=True,
                        timestamp=timestamps[i-1]
                    ))
            
            # 检测看跌FVG（三根K线中，中间K线的高点低于前一根K线的低点）
            if highs[i-1] < lows[i-2]:
                gap_size = lows[i-2] - highs[i-1]
                if gap_size > atr * self.fvg_threshold:
                    self.fair_value_gaps.append(FairValueGap(
                        start_index=i-2,
                        end_index=i-1,
                        top=lows[i-2],
                        bottom=highs[i-1],
                        is_bullish=False,
                        timestamp=timestamps[i-1]
                    ))
        
        # 检查FVG是否被填充
        self._check_fvg_filled(df)
        
        return self.fair_value_gaps
    
    def _check_fvg_filled(self, df: pd.DataFrame):
        """检查FVG是否被填充"""
        highs = df['high'].values
        lows = df['low'].values
        
        for fvg in self.fair_value_gaps:
            if fvg.filled:
                continue
                
            # 检查FVG后的K线是否填充了缺口
            for i in range(fvg.end_index + 1, len(df)):
                if fvg.is_bullish and lows[i] <= fvg.bottom:
                    fvg.filled = True
                    break
                elif not fvg.is_bullish and highs[i] >= fvg.top:
                    fvg.filled = True
                    break
    
    def detect_liquidity(self, df: pd.DataFrame) -> List[LiquidityLevel]:
        """
        检测流动性水平
        
        参数:
            df: 包含OHLC数据的DataFrame
            
        返回:
            流动性水平列表
        """
        self.liquidity_levels = []
        
        if len(self.swing_points) < 2:
            return self.liquidity_levels
            
        highs = df['high'].values
        lows = df['low'].values
        timestamps = df.index
        
        # 计算ATR用于确定流动性阈值
        atr = self._calculate_atr(df, 14)
        
        # 检测流动性扫荡
        swing_highs = [p for p in self.swing_points if p.is_high]
        swing_lows = [p for p in self.swing_points if not p.is_high]
        
        # 检测高点流动性扫荡
        for i in range(1, len(swing_highs)):
            prev_high = swing_highs[i-1]
            current_high = swing_highs[i]
            
            # 检查当前高点是否略微超过前一个高点然后回落
            if (current_high.price > prev_high.price and 
                abs(current_high.price - prev_high.price) / prev_high.price < 0.01):  # 1%以内
                
                # 检查后续价格是否回落
                if current_high.index + 5 < len(df):
                    subsequent_low = min(lows[current_high.index+1:current_high.index+6])
                    if subsequent_low < prev_high.price:
                        # 这是一个流动性扫荡
                        liq_strength = min(1.0, (current_high.price - prev_high.price) / (atr * self.liquidity_threshold))
                        
                        self.liquidity_levels.append(LiquidityLevel(
                            index=current_high.index,
                            price=current_high.price,
                            is_sweep=True,
                            timestamp=timestamps[current_high.index],
                            strength=liq_strength
                        ))
        
        # 检测低点流动性扫荡
        for i in range(1, len(swing_lows)):
            prev_low = swing_lows[i-1]
            current_low = swing_lows[i]
            
            # 检查当前低点是否略微低于前一个低点然后回升
            if (current_low.price < prev_low.price and 
                abs(current_low.price - prev_low.price) / prev_low.price < 0.01):  # 1%以内
                
                # 检查后续价格是否回升
                if current_low.index + 5 < len(df):
                    subsequent_high = max(highs[current_low.index+1:current_low.index+6])
                    if subsequent_high > prev_low.price:
                        # 这是一个流动性扫荡
                        liq_strength = min(1.0, (prev_low.price - current_low.price) / (atr * self.liquidity_threshold))
                        
                        self.liquidity_levels.append(LiquidityLevel(
                            index=current_low.index,
                            price=current_low.price,
                            is_sweep=True,
                            timestamp=timestamps[current_low.index],
                            strength=liq_strength
                        ))
        
        return self.liquidity_levels
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """计算平均真实范围（ATR）"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]
        
        return atr if not pd.isna(atr) else 0.0
    
    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        执行完整的SMC分析
        
        参数:
            df: 包含OHLC数据的DataFrame
            
        返回:
            包含所有SMC分析结果的字典
        """
        # 执行所有检测
        swing_points = self.detect_swing_points(df)
        market_structure = self.detect_market_structure(df)
        order_blocks = self.detect_order_blocks(df)
        fair_value_gaps = self.detect_fair_value_gaps(df)
        liquidity_levels = self.detect_liquidity(df)
        
        # 计算强度评分
        bos_strength = self._calculate_bos_strength(df)
        fvg_count = len([fvg for fvg in fair_value_gaps if not fvg.filled])
        ob_count = len(order_blocks)
        strength_score = self._calculate_strength_score(df)
        
        return {
            'swing_points': [{'index': sp.index, 'price': sp.price, 'is_high': sp.is_high, 'timestamp': sp.timestamp} 
                            for sp in swing_points],
            'market_structure': market_structure,
            'order_blocks': [{'index': ob.index, 'top': ob.top, 'bottom': ob.bottom, 
                            'is_bullish': ob.is_bullish, 'timestamp': ob.timestamp, 'strength': ob.strength} 
                           for ob in order_blocks],
            'fair_value_gaps': [{'start_index': fvg.start_index, 'end_index': fvg.end_index, 
                               'top': fvg.top, 'bottom': fvg.bottom, 'is_bullish': fvg.is_bullish, 
                               'timestamp': fvg.timestamp, 'filled': fvg.filled} 
                              for fvg in fair_value_gaps],
            'liquidity_levels': [{'index': liq.index, 'price': liq.price, 'is_sweep': liq.is_sweep, 
                                'timestamp': liq.timestamp, 'strength': liq.strength} 
                               for liq in liquidity_levels],
            'trend_direction': self.current_trend.value,
            'bos_strength': bos_strength,
            'fvg_count': fvg_count,
            'ob_count': ob_count,
            'strength_score': strength_score,
            'last_bos_price': self.last_bos_price,
            'last_choch_price': self.last_choch_price,
            'is_fixed_pattern': False  # TV实现不会产生固定模式
        }
    
    def _calculate_bos_strength(self, df: pd.DataFrame) -> float:
        """计算BOS强度"""
        if not self.market_structure:
            return 0.0
            
        # 获取最近的BOS/CHOCH事件
        recent_events = [e for e in self.market_structure 
                        if e['type'] in [MarketStructureType.BOS.value, MarketStructureType.CHOCH.value]]
        
        if not recent_events:
            return 0.0
            
        last_event = recent_events[-1]
        current_price = df['close'].iloc[-1]
        
        # 计算价格变化相对于ATR的比率
        atr = self._calculate_atr(df, 14)
        if atr == 0:
            return 0.0
            
        price_change = abs(current_price - last_event['price'])
        strength = min(2.0, price_change / atr)
        
        return max(0.1, strength)
    
    def _calculate_strength_score(self, df: pd.DataFrame) -> float:
        """计算整体强度评分"""
        # 基于多个因素计算综合强度评分
        bos_strength = self._calculate_bos_strength(df)
        fvg_count = len([fvg for fvg in self.fair_value_gaps if not fvg.filled])
        ob_count = len(self.order_blocks)
        
        # 归一化评分
        normalized_bos = min(1.0, bos_strength / 2.0)
        normalized_fvg = min(1.0, fvg_count / 5.0)
        normalized_ob = min(1.0, ob_count / 3.0)
        
        # 加权平均
        strength_score = (normalized_bos * 0.5 + normalized_fvg * 0.3 + normalized_ob * 0.2)
        
        return strength_score


def detect_smc_structures_tv(df: pd.DataFrame, 
                            swing_length: int = 5,
                            structure_lookback: int = 50,
                            fvg_threshold: float = 0.5,
                            ob_threshold: float = 0.3,
                            liquidity_threshold: float = 0.2) -> Dict[str, Any]:
    """
    便捷函数：使用TradingView逻辑检测SMC结构
    
    参数:
        df: 包含OHLC数据的DataFrame
        swing_length: 摆动点检测的回看长度
        structure_lookback: 市场结构分析的回看长度
        fvg_threshold: 公允价值缺口检测阈值（ATR倍数）
        ob_threshold: 订单块检测阈值（ATR倍数）
        liquidity_threshold: 流动性检测阈值（ATR倍数）
        
    返回:
        包含所有SMC分析结果的字典
    """
    detector = SMCDetector(
        swing_length=swing_length,
        structure_lookback=structure_lookback,
        fvg_threshold=fvg_threshold,
        ob_threshold=ob_threshold,
        liquidity_threshold=liquidity_threshold
    )
    
    return detector.analyze(df)