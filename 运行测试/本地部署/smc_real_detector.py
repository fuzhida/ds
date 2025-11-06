"""
真实SMC结构检测模块
提供基于价格行为的真实市场结构识别，避免智能备选计算的虚假性问题
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class RealSMCDetector:
    """真实SMC结构检测器 - 基于价格行为识别市场结构"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def detect_swing_points(self, df: pd.DataFrame, swing_length: int = 5) -> pd.DataFrame:
        """
        检测波段高低点（Swing Highs/Lows）
        
        Args:
            df: 包含OHLCV数据的DataFrame
            swing_length: 波段检测长度（默认5）
            
        Returns:
            包含波段高低点的DataFrame
        """
        if len(df) < swing_length * 2 + 1:
            return pd.DataFrame(columns=['HighLow', 'Level'])
            
        highs = []
        lows = []
        levels = []
        
        for i in range(swing_length, len(df) - swing_length):
            # 检测波段高点
            is_swing_high = True
            current_high = df.iloc[i]['high']
            
            for j in range(1, swing_length + 1):
                if (df.iloc[i]['high'] <= df.iloc[i-j]['high'] or 
                    df.iloc[i]['high'] <= df.iloc[i+j]['high']):
                    is_swing_high = False
                    break
                    
            if is_swing_high:
                highs.append(i)
                levels.append(current_high)
                
            # 检测波段低点
            is_swing_low = True
            current_low = df.iloc[i]['low']
            
            for j in range(1, swing_length + 1):
                if (df.iloc[i]['low'] >= df.iloc[i-j]['low'] or 
                    df.iloc[i]['low'] >= df.iloc[i+j]['low']):
                    is_swing_low = False
                    break
                    
            if is_swing_low:
                lows.append(i)
                levels.append(current_low)
        
        # 合并高低点并按时间排序
        all_points = []
        for idx in highs:
            all_points.append({'HighLow': 'High', 'Level': df.iloc[idx]['high'], 'index': idx})
        for idx in lows:
            all_points.append({'HighLow': 'Low', 'Level': df.iloc[idx]['low'], 'index': idx})
            
        # 按索引排序
        all_points.sort(key=lambda x: x['index'])
        
        result_df = pd.DataFrame([
            {'HighLow': point['HighLow'], 'Level': point['Level']} 
            for point in all_points
        ])
        
        self.logger.debug(f"检测到 {len(result_df)} 个波段点")
        return result_df
    
    def detect_bos_choch(self, df: pd.DataFrame, swing_points: pd.DataFrame, 
                         close_break: bool = True) -> pd.DataFrame:
        """
        检测结构突破（BOS）和特征改变（CHOCH）
        
        Args:
            df: 价格数据
            swing_points: 波段点数据
            close_break: 是否使用收盘价突破
            
        Returns:
            包含BOS和CHOCH事件的DataFrame
        """
        if len(swing_points) < 3:
            return pd.DataFrame(columns=['BOS', 'CHOCH', 'Level', 'BrokenIndex', 'type'])
            
        bos_events = []
        choch_events = []
        
        # 获取波段高低点的序列
        swing_sequence = []
        for _, point in swing_points.iterrows():
            swing_sequence.append({
                'type': point['HighLow'],
                'level': point['Level'],
                'index': swing_points.index[swing_points['Level'] == point['Level']].tolist()[0]
            })
        
        # 检测市场结构
        for i in range(2, len(swing_sequence)):
            current = swing_sequence[i]
            prev1 = swing_sequence[i-1]
            prev2 = swing_sequence[i-2]
            
            # 检测上升趋势中的BOS
            if (prev2['type'] == 'Low' and prev1['type'] == 'High' and 
                current['type'] == 'Low' and current['level'] > prev2['level']):
                
                # 检查是否有价格突破前高
                breakout_price = prev1['level']
                breakout_data = df.iloc[prev1['index']+1:current['index']+1]
                
                if len(breakout_data) > 0:
                    if close_break:
                        breakout_occurred = (breakout_data['close'] > breakout_price).any()
                    else:
                        breakout_occurred = (breakout_data['high'] > breakout_price).any()
                    
                    if breakout_occurred:
                        bos_events.append({
                            'BOS': True,
                            'CHOCH': False,
                            'Level': breakout_price,
                            'BrokenIndex': breakout_data[breakout_data['close'] > breakout_price].index[0] if close_break else 
                                          breakout_data[breakout_data['high'] > breakout_price].index[0],
                            'type': 'BOS'
                        })
            
            # 检测下降趋势中的BOS
            elif (prev2['type'] == 'High' and prev1['type'] == 'Low' and 
                  current['type'] == 'High' and current['level'] < prev2['level']):
                
                breakout_price = prev1['level']
                breakout_data = df.iloc[prev1['index']+1:current['index']+1]
                
                if len(breakout_data) > 0:
                    if close_break:
                        breakout_occurred = (breakout_data['close'] < breakout_price).any()
                    else:
                        breakout_occurred = (breakout_data['low'] < breakout_price).any()
                    
                    if breakout_occurred:
                        bos_events.append({
                            'BOS': True,
                            'CHOCH': False,
                            'Level': breakout_price,
                            'BrokenIndex': breakout_data[breakout_data['close'] < breakout_price].index[0] if close_break else 
                                          breakout_data[breakout_data['low'] < breakout_price].index[0],
                            'type': 'BOS'
                        })
        
        # 创建结果DataFrame
        result_df = pd.DataFrame(bos_events + choch_events)
        if len(result_df) == 0:
            result_df = pd.DataFrame(columns=['BOS', 'CHOCH', 'Level', 'BrokenIndex', 'type'])
        else:
            # 填充缺失值
            result_df['BOS'] = result_df['BOS'].fillna(False)
            result_df['CHOCH'] = result_df['CHOCH'].fillna(False)
            
        self.logger.debug(f"检测到 {len(result_df)} 个BOS/CHOCH事件")
        return result_df
    
    def detect_fvg(self, df: pd.DataFrame, min_gap_ratio: float = 0.001) -> pd.DataFrame:
        """
        检测公允价值缺口（FVG）
        
        Args:
            df: 价格数据
            min_gap_ratio: 最小缺口比例
            
        Returns:
            包含FVG事件的DataFrame
        """
        if len(df) < 3:
            return pd.DataFrame(columns=['type', 'top', 'bottom', 'bull', 'bear'])
            
        fvg_events = []
        
        for i in range(2, len(df)):
            current_candle = df.iloc[i]
            prev_candle = df.iloc[i-1]
            prev2_candle = df.iloc[i-2]
            
            # 检测看涨FVG（价格向上跳空）
            if (current_candle['low'] > prev_candle['high'] and 
                prev2_candle['high'] > prev_candle['low']):
                
                gap_size = current_candle['low'] - prev_candle['high']
                gap_ratio = gap_size / prev_candle['close']
                
                if gap_ratio >= min_gap_ratio:
                    fvg_events.append({
                        'type': 'FVG',
                        'top': current_candle['low'],
                        'bottom': prev_candle['high'],
                        'bull': True,
                        'bear': False
                    })
            
            # 检测看跌FVG（价格向下跳空）
            elif (current_candle['high'] < prev_candle['low'] and 
                  prev2_candle['low'] < prev_candle['high']):
                
                gap_size = prev_candle['low'] - current_candle['high']
                gap_ratio = gap_size / prev_candle['close']
                
                if gap_ratio >= min_gap_ratio:
                    fvg_events.append({
                        'type': 'FVG',
                        'top': prev_candle['low'],
                        'bottom': current_candle['high'],
                        'bull': False,
                        'bear': True
                    })
        
        result_df = pd.DataFrame(fvg_events)
        if len(result_df) == 0:
            result_df = pd.DataFrame(columns=['type', 'top', 'bottom', 'bull', 'bear'])
            
        self.logger.debug(f"检测到 {len(result_df)} 个FVG事件")
        return result_df
    
    def detect_order_blocks(self, df: pd.DataFrame, swing_points: pd.DataFrame, 
                           volume_threshold: float = 1.5) -> pd.DataFrame:
        """
        检测订单块（Order Blocks）
        
        Args:
            df: 价格数据
            swing_points: 波段点数据
            volume_threshold: 成交量阈值倍数
            
        Returns:
            包含OB事件的DataFrame
        """
        if len(df) < 20 or len(swing_points) < 2:
            return pd.DataFrame(columns=['type', 'high', 'low', 'bullish', 'bearish'])
            
        ob_events = []
        volume_avg = df['volume'].mean()
        
        # 寻找强势K线作为潜在的订单块
        for i in range(1, len(df)-1):
            current = df.iloc[i]
            
            # 成交量异常放大
            if current['volume'] > volume_avg * volume_threshold:
                
                # 看涨订单块：大阳线后价格不再跌破低点
                if (current['close'] > current['open'] and 
                    (current['close'] - current['open']) / (current['high'] - current['low']) > 0.7):
                    
                    # 检查后续价格是否守住低点
                    future_data = df.iloc[i+1:min(i+10, len(df))]
                    if len(future_data) > 0:
                        lowest_future = future_data['low'].min()
                        if lowest_future >= current['low'] * 0.99:  # 允许小幅跌破
                            ob_events.append({
                                'type': 'OB',
                                'high': current['high'],
                                'low': current['low'],
                                'bullish': True,
                                'bearish': False
                            })
                
                # 看跌订单块：大阴线后价格不再涨破高点
                elif (current['close'] < current['open'] and 
                      (current['open'] - current['close']) / (current['high'] - current['low']) > 0.7):
                    
                    future_data = df.iloc[i+1:min(i+10, len(df))]
                    if len(future_data) > 0:
                        highest_future = future_data['high'].max()
                        if highest_future <= current['high'] * 1.01:  # 允许小幅突破
                            ob_events.append({
                                'type': 'OB',
                                'high': current['high'],
                                'low': current['low'],
                                'bullish': False,
                                'bearish': True
                            })
        
        result_df = pd.DataFrame(ob_events)
        if len(result_df) == 0:
            result_df = pd.DataFrame(columns=['type', 'high', 'low', 'bullish', 'bearish'])
            
        self.logger.debug(f"检测到 {len(result_df)} 个OB事件")
        return result_df
    
    def validate_structures(self, structures: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        验证检测到的结构是否真实有效
        
        Args:
            structures: 包含各种SMC结构的字典
            
        Returns:
            验证结果和置信度评分
        """
        validation_result = {
            'is_valid': True,
            'confidence_score': 0.0,
            'issues': [],
            'recommendations': []
        }
        
        # 验证BOS/CHOCH
        if 'bos_choch' in structures and not structures['bos_choch'].empty:
            bos_choch = structures['bos_choch']
            
            # 检查是否有重复或固定的数值
            if 'Level' in bos_choch.columns:
                unique_levels = bos_choch['Level'].nunique()
                total_events = len(bos_choch)
                
                if unique_levels == 1 and total_events > 1:
                    validation_result['issues'].append("BOS/CHOCH检测到固定数值模式")
                    validation_result['is_valid'] = False
                elif unique_levels / total_events < 0.5:
                    validation_result['issues'].append("BOS/CHOCH数值重复率过高")
                    validation_result['confidence_score'] *= 0.7
        
        # 验证FVG
        if 'fvg' in structures and not structures['fvg'].empty:
            fvg = structures['fvg']
            
            # 检查FVG的合理性
            if len(fvg) > 50:  # 过多的FVG可能表示检测过于敏感
                validation_result['issues'].append("FVG数量异常多，可能检测过于敏感")
                validation_result['confidence_score'] *= 0.8
            
            # 深度检测FVG模式
            if len(fvg) > 3:
                # 检查连续微小FVG模式
                if 'size' in fvg.columns:
                    small_fvg_count = (fvg['size'] < fvg['size'].mean() * 0.3).sum()
                    if small_fvg_count > len(fvg) * 0.7:
                        validation_result['issues'].append("检测到过多微小FVG，可能为噪声")
                        validation_result['confidence_score'] *= 0.7
                
                # 检查FVG分布均匀性
                if 'timestamp' in fvg.columns:
                    time_diff = fvg['timestamp'].diff().dropna()
                    if len(time_diff) > 1:
                        time_std = time_diff.std()
                        if time_std < time_diff.mean() * 0.2:
                            validation_result['issues'].append("FVG时间分布异常均匀，可能为算法错误")
                            validation_result['confidence_score'] *= 0.6
        
        # 验证订单块
        if 'ob' in structures and not structures['ob'].empty:
            ob = structures['ob']
            
            # 检查订单块的分布
            if 'high' in ob.columns and 'low' in ob.columns:
                avg_size = (ob['high'] - ob['low']).mean()
                if avg_size == 0:
                    validation_result['issues'].append("订单块大小为0，数据异常")
                    validation_result['is_valid'] = False
                
                # 深度检测订单块分布
                ob_sizes = ob['high'] - ob['low']
                size_std = ob_sizes.std()
                
                # 检查订单块大小异常
                if size_std == 0 and len(ob) > 1:
                    validation_result['issues'].append("订单块大小完全一致，可能为算法错误")
                    validation_result['confidence_score'] *= 0.5
                
                # 检查订单块重叠情况
                if len(ob) > 2:
                    ob_sorted = ob.sort_values('high')
                    overlap_count = 0
                    for i in range(len(ob_sorted) - 1):
                        if ob_sorted.iloc[i]['low'] < ob_sorted.iloc[i+1]['high']:
                            overlap_count += 1
                    
                    if overlap_count > len(ob) * 0.5:
                        validation_result['issues'].append("订单块重叠率过高，检测可能不准确")
                        validation_result['confidence_score'] *= 0.8
        
        # 计算总体置信度
        base_confidence = 0.8
        issue_penalty = 0.1 * len(validation_result['issues'])
        validation_result['confidence_score'] = max(0.1, base_confidence - issue_penalty)
        
        return validation_result
    
    def detect_all_structures(self, df: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
        """
        完整的SMC结构检测流程
        
        Args:
            df: 价格数据
            timeframe: 时间框架
            
        Returns:
            包含所有结构和验证结果的字典
        """
        self.logger.info(f"开始真实SMC结构检测 - {timeframe}")
        
        # 1. 检测波段点
        swing_points = self.detect_swing_points(df)
        
        # 2. 检测BOS/CHOCH
        bos_choch = self.detect_bos_choch(df, swing_points)
        
        # 3. 检测FVG
        fvg = self.detect_fvg(df)
        
        # 4. 检测订单块
        ob = self.detect_order_blocks(df, swing_points)
        
        # 5. 验证结构
        structures = {
            'swing_points': swing_points,
            'bos_choch': bos_choch,
            'fvg': fvg,
            'ob': ob
        }
        
        validation = self.validate_structures(structures)
        
        # 计算结构强度
        structure_metrics = self._calculate_structure_metrics(df, structures, timeframe)
        
        result = {
            'structures': structures,
            'validation': validation,
            'metrics': structure_metrics,
            'timeframe': timeframe,
            'data_quality': self._assess_data_quality(df)
        }
        
        self.logger.info(f"完成真实SMC结构检测 - {timeframe}, 置信度: {validation['confidence_score']:.2f}")
        return result
    
    def _calculate_structure_metrics(self, df: pd.DataFrame, structures: Dict[str, pd.DataFrame], 
                                   timeframe: str) -> Dict[str, float]:
        """计算结构强度指标"""
        
        # 统一时间框架权重和基准强度
        # 时间框架权重：增强4h和1h权重
        timeframe_weights = {
            '1d': 1.0, '4h': 1.8, '1h': 2.0, '15m': 1.8, '3m': 1.5, '1m': 0.5
        }
        base_weight = timeframe_weights.get(timeframe, 1.0)
        
        # BOS基准强度：与btc_trading_bot.py保持一致
        timeframe_base = {
            '1d': 0.8, '4h': 1.2, '1h': 1.5, '15m': 1.8, '3m': 2.0, '1m': 0.5
        }.get(timeframe, 1.5)
        
        # 计算BOS强度
        bos_strength = 0.0
        if not structures['bos_choch'].empty and 'BOS' in structures['bos_choch'].columns:
            bos_count = structures['bos_choch']['BOS'].sum()
            bos_strength = min(bos_count * timeframe_base * base_weight, 3.0)
        
        # 计算FVG数量
        fvg_count = len(structures['fvg']) if not structures['fvg'].empty else 0
        
        # 计算OB数量
        ob_count = len(structures['ob']) if not structures['ob'].empty else 0
        
        # 计算综合强度 - 优化权重分配
        # BOS权重35%，FVG权重3%/个（上限20个），OB权重4%/个（上限15个）
        total_strength = (
            bos_strength * 0.35 +                    # BOS权重35%
            min(fvg_count, 20) * 0.03 +              # FVG权重3%/个，上限20个
            min(ob_count, 15) * 0.04                 # OB权重4%/个，上限15个
        )
        
        # 限制在合理范围内
        total_strength = max(0.1, min(total_strength, 1.0))
        
        return {
            'bos_strength': bos_strength,
            'fvg_count': fvg_count,
            'ob_count': ob_count,
            'total_strength': total_strength
        }
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """评估数据质量"""
        if len(df) == 0:
            return {'quality_score': 0.0, 'issues': ['无数据']}
            
        issues = []
        
        # 检查缺失值
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if missing_ratio > 0.05:
            issues.append(f"缺失值比例过高: {missing_ratio:.2%}")
        
        # 检查价格异常
        price_diff = df['high'] - df['low']
        zero_range = (price_diff == 0).sum()
        if zero_range > len(df) * 0.01:
            issues.append(f"零价格区间: {zero_range}个")
            
        # 检查成交量异常
        if 'volume' in df.columns:
            zero_volume = (df['volume'] == 0).sum()
            if zero_volume > len(df) * 0.1:
                issues.append(f"零成交量: {zero_volume}个")
        
        # 计算质量评分
        quality_score = max(0.0, 1.0 - len(issues) * 0.2 - missing_ratio)
        
        return {
            'quality_score': quality_score,
            'issues': issues,
            'data_points': len(df),
            'missing_ratio': missing_ratio
        }