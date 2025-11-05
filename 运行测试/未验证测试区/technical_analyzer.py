"""
技术指标分析模块 - 包含技术指标计算和分析功能
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timezone


class TechnicalIndicatorCalculator:
    """技术指标计算器"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        计算技术指标
        :param df: 价格数据DataFrame
        :return: 技术指标字典
        """
        try:
            if len(df) < 50:  # 最小数据要求
                return {}
            
            # 基础指标
            indicators = {}
            
            # EMA
            indicators['ema'] = self._calculate_ema(df)
            
            # RSI
            indicators['rsi'] = self._calculate_rsi(df)
            
            # ATR
            indicators['atr'] = self._calculate_atr(df)
            
            # MACD
            indicators['macd'] = self._calculate_macd(df)
            
            # 布林带
            indicators['bollinger'] = self._calculate_bollinger_bands(df)
            
            # 随机指标
            indicators['stochastic'] = self._calculate_stochastic(df)
            
            # 成交量指标
            indicators['volume'] = self._calculate_volume_indicators(df)
            
            # 综合技术评分
            indicators['technical_score'] = self._calculate_technical_score(df, indicators)
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"技术指标计算失败: {e}")
            return {}
    
    def _calculate_ema(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算EMA"""
        try:
            close = df['close'].values
            
            ema_21 = talib.EMA(close, timeperiod=21)[-1] if len(close) >= 21 else None
            ema_50 = talib.EMA(close, timeperiod=50)[-1] if len(close) >= 50 else None
            ema_100 = talib.EMA(close, timeperiod=100)[-1] if len(close) >= 100 else None
            ema_200 = talib.EMA(close, timeperiod=200)[-1] if len(close) >= 200 else None
            
            return {
                'ema_21': ema_21,
                'ema_50': ema_50,
                'ema_100': ema_100,
                'ema_200': ema_200
            }
            
        except Exception as e:
            self.logger.error(f"EMA计算失败: {e}")
            return {}
    
    def _calculate_rsi(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算RSI"""
        try:
            close = df['close'].values
            
            rsi_14 = talib.RSI(close, timeperiod=14)[-1] if len(close) >= 15 else None
            
            return {
                'rsi_14': rsi_14
            }
            
        except Exception as e:
            self.logger.error(f"RSI计算失败: {e}")
            return {}
    
    def _calculate_atr(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算ATR"""
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            atr_14 = talib.ATR(high, low, close, timeperiod=14)[-1] if len(close) >= 15 else None
            
            return {
                'atr_14': atr_14
            }
            
        except Exception as e:
            self.logger.error(f"ATR计算失败: {e}")
            return {}
    
    def _calculate_macd(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算MACD"""
        try:
            close = df['close'].values
            
            macd, macd_signal, macd_hist = talib.MACD(close)
            
            return {
                'macd': macd[-1] if len(macd) > 0 else None,
                'signal': macd_signal[-1] if len(macd_signal) > 0 else None,
                'histogram': macd_hist[-1] if len(macd_hist) > 0 else None
            }
            
        except Exception as e:
            self.logger.error(f"MACD计算失败: {e}")
            return {}
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算布林带"""
        try:
            close = df['close'].values
            
            upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            
            return {
                'upper': upper[-1] if len(upper) > 0 else None,
                'middle': middle[-1] if len(middle) > 0 else None,
                'lower': lower[-1] if len(lower) > 0 else None
            }
            
        except Exception as e:
            self.logger.error(f"布林带计算失败: {e}")
            return {}
    
    def _calculate_stochastic(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算随机指标"""
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            slowk, slowd = talib.STOCH(high, low, close)
            
            return {
                'slowk': slowk[-1] if len(slowk) > 0 else None,
                'slowd': slowd[-1] if len(slowd) > 0 else None
            }
            
        except Exception as e:
            self.logger.error(f"随机指标计算失败: {e}")
            return {}
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算成交量指标"""
        try:
            close = df['close'].values
            volume = df['volume'].values
            
            # 成交量移动平均
            volume_sma = talib.SMA(volume, timeperiod=20)[-1] if len(volume) >= 20 else None
            
            # OBV
            obv = talib.OBV(close, volume)[-1] if len(close) >= 1 else None
            
            # ADX
            high = df['high'].values
            low = df['low'].values
            
            adx = talib.ADX(high, low, close, timeperiod=14)[-1] if len(close) >= 15 else None
            
            return {
                'volume_sma': volume_sma,
                'obv': obv,
                'adx': adx
            }
            
        except Exception as e:
            self.logger.error(f"成交量指标计算失败: {e}")
            return {}
    
    def _calculate_technical_score(self, df: pd.DataFrame, indicators: Dict) -> float:
        """计算综合技术评分"""
        try:
            score = 0.0
            count = 0
            
            # EMA趋势评分
            if 'ema' in indicators and indicators['ema']:
                ema = indicators['ema']
                current_price = df['close'].iloc[-1]
                
                if (ema.get('ema_21') and ema.get('ema_50') and ema.get('ema_200')):
                    if current_price > ema['ema_21'] > ema['ema_50'] > ema['ema_200']:
                        score += 1.0
                    elif current_price < ema['ema_21'] < ema['ema_50'] < ema['ema_200']:
                        score -= 1.0
                    count += 1
            
            # RSI评分
            if 'rsi' in indicators and indicators['rsi']:
                rsi = indicators['rsi']
                if rsi.get('rsi_14'):
                    if 30 < rsi['rsi_14'] < 70:
                        score += 0.5
                    elif rsi['rsi_14'] > 70:
                        score -= 0.3
                    elif rsi['rsi_14'] < 30:
                        score += 0.3
                    count += 1
            
            # MACD评分
            if 'macd' in indicators and indicators['macd']:
                macd = indicators['macd']
                if (macd.get('macd') and macd.get('signal') and macd.get('histogram')):
                    if macd['macd'] > macd['signal'] and macd['histogram'] > 0:
                        score += 0.5
                    elif macd['macd'] < macd['signal'] and macd['histogram'] < 0:
                        score -= 0.5
                    count += 1
            
            # ADX评分
            if 'volume' in indicators and indicators['volume']:
                volume = indicators['volume']
                if volume.get('adx'):
                    if volume['adx'] > 25:
                        score += 0.3
                    count += 1
            
            # 归一化评分
            if count > 0:
                score = score / count
            
            return min(max(score, -1.0), 1.0)
            
        except Exception as e:
            self.logger.error(f"技术评分计算失败: {e}")
            return 0.0


class KeyLevelsCalculator:
    """关键水平计算器"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
    
    def calculate_key_levels(self, multi_tf_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        计算关键价格水平
        :param multi_tf_data: 多时间框架数据字典
        :return: 关键水平字典
        """
        try:
            key_levels = {}
            
            for tf, df in multi_tf_data.items():
                if len(df) < 20:
                    continue
                
                # 计算各时间框架的关键水平
                tf_levels = self._calculate_tf_key_levels(df, tf)
                key_levels[tf] = tf_levels
            
            # 整合多时间框架关键水平
            consolidated_levels = self._consolidate_key_levels(key_levels)
            
            return {
                'timeframe_levels': key_levels,
                'consolidated_levels': consolidated_levels
            }
            
        except Exception as e:
            self.logger.error(f"关键水平计算失败: {e}")
            return {}
    
    def _calculate_tf_key_levels(self, df: pd.DataFrame, tf: str) -> Dict[str, Any]:
        """计算单个时间框架的关键水平"""
        try:
            levels = {}
            
            # 计算支撑阻力
            support_resistance = self._calculate_support_resistance(df)
            levels['support_resistance'] = support_resistance
            
            # 计算斐波那契水平
            fibonacci = self._calculate_fibonacci_levels(df)
            levels['fibonacci'] = fibonacci
            
            # 计算枢轴点
            pivots = self._calculate_pivot_points(df)
            levels['pivots'] = pivots
            
            # 计算VWAP
            vwap = self._calculate_vwap(df)
            levels['vwap'] = vwap
            
            # 计算EMA水平
            ema_levels = self._calculate_ema_levels(df)
            levels['ema'] = ema_levels
            
            return levels
            
        except Exception as e:
            self.logger.error(f"时间框架关键水平计算失败 {tf}: {e}")
            return {}
    
    def _calculate_support_resistance(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """计算支撑阻力水平"""
        try:
            # 使用局部极值点
            highs = df['high']
            lows = df['low']
            
            # 寻找局部高点（阻力）
            resistance_levels = []
            for i in range(2, len(highs) - 2):
                if (highs.iloc[i] > highs.iloc[i-1] and 
                    highs.iloc[i] > highs.iloc[i-2] and
                    highs.iloc[i] > highs.iloc[i+1] and
                    highs.iloc[i] > highs.iloc[i+2]):
                    resistance_levels.append(highs.iloc[i])
            
            # 寻找局部低点（支撑）
            support_levels = []
            for i in range(2, len(lows) - 2):
                if (lows.iloc[i] < lows.iloc[i-1] and 
                    lows.iloc[i] < lows.iloc[i-2] and
                    lows.iloc[i] < lows.iloc[i+1] and
                    lows.iloc[i] < lows.iloc[i+2]):
                    support_levels.append(lows.iloc[i])
            
            # 聚类相似水平
            resistance_levels = self._cluster_levels(resistance_levels)
            support_levels = self._cluster_levels(support_levels)
            
            # 按重要性排序
            resistance_levels.sort(reverse=True)
            support_levels.sort()
            
            return {
                'resistance': resistance_levels[:5],  # 前5个重要阻力
                'support': support_levels[:5]  # 前5个重要支撑
            }
            
        except Exception as e:
            self.logger.error(f"支撑阻力计算失败: {e}")
            return {'resistance': [], 'support': []}
    
    def _calculate_fibonacci_levels(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """计算斐波那契回撤水平"""
        try:
            # 获取最近的高低点
            recent_high = df['high'].max()
            recent_low = df['low'].min()
            
            # 计算斐波那契水平
            diff = recent_high - recent_low
            
            fib_levels = {
                '0.0%': recent_low,
                '23.6%': recent_low + 0.236 * diff,
                '38.2%': recent_low + 0.382 * diff,
                '50.0%': recent_low + 0.5 * diff,
                '61.8%': recent_low + 0.618 * diff,
                '78.6%': recent_low + 0.786 * diff,
                '100.0%': recent_high
            }
            
            return fib_levels
            
        except Exception as e:
            self.logger.error(f"斐波那契水平计算失败: {e}")
            return {}
    
    def _calculate_pivot_points(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算枢轴点"""
        try:
            # 获取最新数据
            latest = df.iloc[-1]
            
            high = latest['high']
            low = latest['low']
            close = latest['close']
            
            # 计算枢轴点
            pivot = (high + low + close) / 3
            
            # 计算支撑和阻力
            r1 = 2 * pivot - low
            r2 = pivot + (high - low)
            r3 = high + 2 * (pivot - low)
            
            s1 = 2 * pivot - high
            s2 = pivot - (high - low)
            s3 = low - 2 * (high - pivot)
            
            return {
                'pivot': pivot,
                'r1': r1,
                'r2': r2,
                'r3': r3,
                's1': s1,
                's2': s2,
                's3': s3
            }
            
        except Exception as e:
            self.logger.error(f"枢轴点计算失败: {e}")
            return {}
    
    def _calculate_vwap(self, df: pd.DataFrame) -> float:
        """计算VWAP"""
        try:
            # 计算VWAP
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            vwap = (typical_price * df['volume']).sum() / df['volume'].sum()
            
            return vwap
            
        except Exception as e:
            self.logger.error(f"VWAP计算失败: {e}")
            return 0.0
    
    def _calculate_ema_levels(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算EMA水平"""
        try:
            close = df['close'].values
            
            ema_21 = talib.EMA(close, timeperiod=21)[-1] if len(close) >= 21 else None
            ema_50 = talib.EMA(close, timeperiod=50)[-1] if len(close) >= 50 else None
            ema_100 = talib.EMA(close, timeperiod=100)[-1] if len(close) >= 100 else None
            ema_200 = talib.EMA(close, timeperiod=200)[-1] if len(close) >= 200 else None
            
            return {
                'ema_21': ema_21,
                'ema_50': ema_50,
                'ema_100': ema_100,
                'ema_200': ema_200
            }
            
        except Exception as e:
            self.logger.error(f"EMA水平计算失败: {e}")
            return {}
    
    def _cluster_levels(self, levels: List[float], threshold: float = 0.01) -> List[float]:
        """聚类相似水平"""
        try:
            if not levels:
                return []
            
            # 排序水平
            levels.sort()
            
            # 聚类
            clusters = []
            current_cluster = [levels[0]]
            
            for level in levels[1:]:
                # 检查是否与当前聚类足够接近
                if (level - current_cluster[-1]) / level <= threshold:
                    current_cluster.append(level)
                else:
                    # 保存当前聚类并开始新聚类
                    clusters.append(sum(current_cluster) / len(current_cluster))
                    current_cluster = [level]
            
            # 添加最后一个聚类
            clusters.append(sum(current_cluster) / len(current_cluster))
            
            return clusters
            
        except Exception as e:
            self.logger.error(f"水平聚类失败: {e}")
            return levels
    
    def _consolidate_key_levels(self, key_levels: Dict) -> Dict[str, Any]:
        """整合多时间框架关键水平"""
        try:
            # 收集所有时间框架的水平
            all_support = []
            all_resistance = []
            all_ema = {}
            
            for tf, levels in key_levels.items():
                # 支撑阻力
                if 'support_resistance' in levels:
                    sr = levels['support_resistance']
                    all_support.extend(sr.get('support', []))
                    all_resistance.extend(sr.get('resistance', []))
                
                # EMA
                if 'ema' in levels:
                    ema = levels['ema']
                    for key, value in ema.items():
                        if value is not None:
                            if key not in all_ema:
                                all_ema[key] = []
                            all_ema[key].append(value)
            
            # 聚类支撑阻力
            all_support = self._cluster_levels(all_support)
            all_resistance = self._cluster_levels(all_resistance)
            
            # 按重要性排序
            all_support.sort()
            all_resistance.sort(reverse=True)
            
            # 计算EMA平均值
            ema_avg = {}
            for key, values in all_ema.items():
                if values:
                    ema_avg[key] = sum(values) / len(values)
            
            return {
                'support': all_support[:10],  # 前10个重要支撑
                'resistance': all_resistance[:10],  # 前10个重要阻力
                'ema': ema_avg
            }
            
        except Exception as e:
            self.logger.error(f"关键水平整合失败: {e}")
            return {}


class PriceActionAnalyzer:
    """价格行为分析器"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
    
    def analyze_price_action(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        分析价格行为
        :param df: 价格数据DataFrame
        :return: 价格行为分析结果
        """
        try:
            if len(df) < 20:
                return {}
            
            # 计算价格行为指标
            price_action = {}
            
            # 蜡烛图模式
            candlestick_patterns = self._analyze_candlestick_patterns(df)
            price_action['candlestick_patterns'] = candlestick_patterns
            
            # 价格效率
            price_efficiency = self._calculate_price_efficiency(df)
            price_action['price_efficiency'] = price_efficiency
            
            # 波动性
            volatility = self._calculate_volatility(df)
            price_action['volatility'] = volatility
            
            # 动量
            momentum = self._calculate_momentum(df)
            price_action['momentum'] = momentum
            
            # 综合价格行为评分
            price_action_score = self._calculate_price_action_score(price_action)
            price_action['score'] = price_action_score
            
            return price_action
            
        except Exception as e:
            self.logger.error(f"价格行为分析失败: {e}")
            return {}
    
    def _analyze_candlestick_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析蜡烛图模式"""
        try:
            open_prices = df['open'].values
            high_prices = df['high'].values
            low_prices = df['low'].values
            close_prices = df['close'].values
            
            patterns = {}
            
            # 锤子线
            hammer = talib.CDLHAMMER(open_prices, high_prices, low_prices, close_prices)
            patterns['hammer'] = hammer[-1] if len(hammer) > 0 else 0
            
            # 流星线
            shooting_star = talib.CDLSHOOTINGSTAR(open_prices, high_prices, low_prices, close_prices)
            patterns['shooting_star'] = shooting_star[-1] if len(shooting_star) > 0 else 0
            
            # 吞没形态
            engulfing = talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices)
            patterns['engulfing'] = engulfing[-1] if len(engulfing) > 0 else 0
            
            # 十字星
            doji = talib.CDLDOJI(open_prices, high_prices, low_prices, close_prices)
            patterns['doji'] = doji[-1] if len(doji) > 0 else 0
            
            # 晨星/晚星
            morning_star = talib.CDLMORNINGSTAR(open_prices, high_prices, low_prices, close_prices)
            patterns['morning_star'] = morning_star[-1] if len(morning_star) > 0 else 0
            
            evening_star = talib.CDLEVENINGSTAR(open_prices, high_prices, low_prices, close_prices)
            patterns['evening_star'] = evening_star[-1] if len(evening_star) > 0 else 0
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"蜡烛图模式分析失败: {e}")
            return {}
    
    def _calculate_price_efficiency(self, df: pd.DataFrame) -> float:
        """计算价格效率"""
        try:
            # 计算价格变化
            price_changes = df['close'].pct_change().dropna()
            
            # 计算效率比率
            if len(price_changes) < 2:
                return 0.0
            
            # 方向性移动
            net_move = abs(df['close'].iloc[-1] - df['close'].iloc[0])
            
            # 总移动
            total_move = sum(abs(price_changes))
            
            # 效率比率
            efficiency = net_move / total_move if total_move > 0 else 0.0
            
            return min(max(efficiency, 0.0), 1.0)
            
        except Exception as e:
            self.logger.error(f"价格效率计算失败: {e}")
            return 0.0
    
    def _calculate_volatility(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算波动性"""
        try:
            # 计算收益率
            returns = df['close'].pct_change().dropna()
            
            # 计算波动性指标
            volatility = {
                'std': returns.std(),
                'range': (df['high'] - df['low']).mean(),
                'atr': self._calculate_atr(df)
            }
            
            return volatility
            
        except Exception as e:
            self.logger.error(f"波动性计算失败: {e}")
            return {}
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """计算ATR"""
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            atr = talib.ATR(high, low, close, timeperiod=period)
            
            return atr[-1] if len(atr) > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"ATR计算失败: {e}")
            return 0.0
    
    def _calculate_momentum(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算动量"""
        try:
            # 计算动量指标
            momentum = {}
            
            # ROC
            roc = talib.ROC(df['close'].values, timeperiod=10)
            momentum['roc'] = roc[-1] if len(roc) > 0 else 0.0
            
            # RSI
            rsi = talib.RSI(df['close'].values, timeperiod=14)
            momentum['rsi'] = rsi[-1] if len(rsi) > 0 else 50.0
            
            # 动量
            mom = talib.MOM(df['close'].values, timeperiod=10)
            momentum['mom'] = mom[-1] if len(mom) > 0 else 0.0
            
            return momentum
            
        except Exception as e:
            self.logger.error(f"动量计算失败: {e}")
            return {}
    
    def _calculate_price_action_score(self, price_action: Dict) -> float:
        """计算综合价格行为评分"""
        try:
            score = 0.0
            
            # 蜡烛图模式评分
            patterns = price_action.get('candlestick_patterns', {})
            pattern_score = 0.0
            
            # 看涨模式
            bullish_patterns = ['hammer', 'morning_star']
            for pattern in bullish_patterns:
                if patterns.get(pattern, 0) > 0:
                    pattern_score += 0.2
            
            # 看跌模式
            bearish_patterns = ['shooting_star', 'evening_star']
            for pattern in bearish_patterns:
                if patterns.get(pattern, 0) > 0:
                    pattern_score -= 0.2
            
            # 中性模式
            neutral_patterns = ['doji']
            for pattern in neutral_patterns:
                if patterns.get(pattern, 0) > 0:
                    pattern_score += 0.1
            
            # 价格效率评分
            efficiency = price_action.get('price_efficiency', 0.0)
            efficiency_score = efficiency * 0.3
            
            # 动量评分
            momentum = price_action.get('momentum', {})
            rsi = momentum.get('rsi', 50.0)
            
            if rsi > 70:
                momentum_score = -0.2  # 超买
            elif rsi < 30:
                momentum_score = 0.2   # 超卖
            else:
                momentum_score = 0.0   # 中性
            
            # 综合评分
            score = pattern_score + efficiency_score + momentum_score
            
            return min(max(score, -1.0), 1.0)
            
        except Exception as e:
            self.logger.error(f"价格行为评分计算失败: {e}")
            return 0.0