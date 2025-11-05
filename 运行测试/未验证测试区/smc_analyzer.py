"""
SMC分析模块 - 包含Smart Money Concepts相关的分析功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timezone


class SMCDetector:
    """SMC结构检测器"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
    
    def detect_smc_structures(self, df: pd.DataFrame, tf: str) -> Dict[str, Any]:
        """
        检测SMC结构
        :param df: 价格数据DataFrame
        :param tf: 时间框架
        :return: SMC结构字典
        """
        if len(df) < 10:  # 最小数据要求
            return {}
        
        try:
            # 检测BOS/CHOCH
            bos_choch = self._detect_bos_choch(df, tf)
            
            # 检测订单块
            order_blocks = self._detect_order_blocks(df, tf)
            
            # 检测公平价值缺口
            fvg = self._detect_fvg(df, tf)
            
            # 检测摆动点
            swing_points = self._detect_swing_points(df, tf)
            
            # 计算结构强度
            structure_score = self._calculate_structure_score(bos_choch, order_blocks, fvg, swing_points)
            
            result = {
                'bos_choch': bos_choch,
                'order_blocks': order_blocks,
                'fvg': fvg,
                'swing_points': swing_points,
                'structure_score': structure_score,
                'overall_score': structure_score,
                'tf': tf
            }

            # 统一注入OB/FVG优化结构，提供overlay_result供后续流程使用
            try:
                current_price = float(df['close'].iloc[-1]) if isinstance(df, pd.DataFrame) and len(df) > 0 else 0.0
                result['ob_fvg_optimized'] = self._build_ob_fvg_optimized(df, order_blocks, fvg, current_price)
            except Exception:
                # 安全降级：提供可用的默认结构
                result['ob_fvg_optimized'] = {
                    'ob_fvg_summary': 'weak_or_invalid',
                    'meaningful_ob_count': 0,
                    'meaningful_fvg_count': 0,
                    'strongest_structure': None,
                    'price_relevance': 0.0,
                    'freshness_score': 0.0,
                    'overlay_result': {
                        'has_overlay': False,
                        'overlay_confidence_boost': 0.0,
                        'overlay_details': [],
                        'narrow_ob_for_entry': None,
                        'wide_ob_for_stop_loss': None
                    }
                }

            return result
            
        except Exception as e:
            self.logger.error(f"SMC结构检测失败 {tf}: {e}")
            return {}

    def detect_bos_choch(self, market_data: Dict[str, pd.DataFrame], current_price: float) -> Dict[str, Any]:
        """公共方法：检测BOS/CHOCH结构（适配测试期望的返回结构）"""
        try:
            tf = "1h" if isinstance(market_data, dict) and "1h" in market_data else (next(iter(market_data.keys())) if isinstance(market_data, dict) and market_data else None)
            df = market_data.get(tf) if isinstance(market_data, dict) else None
            if df is None or not isinstance(df, pd.DataFrame):
                return {"bos_choch": {"signal": "HOLD", "confidence": 0.0, "strength": 0.0}}

            result = self._detect_bos_choch(df, tf)
            structures = result.get('structures', [])
            best = max(structures, key=lambda s: s.get('strength', 0.0), default=None)
            if best:
                t = str(best.get('type', '')).lower()
                signal = "BUY" if 'bullish' in t else ("SELL" if 'bearish' in t else "HOLD")
                strength = float(best.get('strength', 0.0))
                confidence = min(max(strength / 3.0, 0.0), 1.0)
            else:
                signal = "HOLD"
                strength = 0.0
                confidence = 0.0
            return {"bos_choch": {"signal": signal, "confidence": confidence, "strength": strength}}
        except Exception:
            return {"bos_choch": {"signal": "HOLD", "confidence": 0.0, "strength": 0.0}}

    def detect_order_blocks(self, market_data: Dict[str, pd.DataFrame], current_price: float) -> Dict[str, Any]:
        """公共方法：检测订单块（适配测试期望的返回结构）"""
        try:
            tf = "1h" if isinstance(market_data, dict) and "1h" in market_data else (next(iter(market_data.keys())) if isinstance(market_data, dict) and market_data else None)
            df = market_data.get(tf) if isinstance(market_data, dict) else None
            if df is None or not isinstance(df, pd.DataFrame):
                return {"order_blocks": {"bullish": [], "bearish": []}}

            obs = self._detect_order_blocks(df, tf)
            bullish = [ob for ob in obs if 'bullish' in str(ob.get('type', '')).lower()]
            bearish = [ob for ob in obs if 'bearish' in str(ob.get('type', '')).lower()]
            return {"order_blocks": {"bullish": bullish, "bearish": bearish}}
        except Exception:
            return {"order_blocks": {"bullish": [], "bearish": []}}

    def detect_fvg(self, market_data: Dict[str, pd.DataFrame], current_price: float) -> Dict[str, Any]:
        """公共方法：检测公平价值缺口（适配测试期望的返回结构）"""
        try:
            tf = "1h" if isinstance(market_data, dict) and "1h" in market_data else (next(iter(market_data.keys())) if isinstance(market_data, dict) and market_data else None)
            df = market_data.get(tf) if isinstance(market_data, dict) else None
            if df is None or not isinstance(df, pd.DataFrame):
                return {"fvg": {"bullish": [], "bearish": []}}

            fvgs = self._detect_fvg(df, tf)
            bullish = [f for f in fvgs if 'bullish' in str(f.get('type', '')).lower()]
            bearish = [f for f in fvgs if 'bearish' in str(f.get('type', '')).lower()]
            return {"fvg": {"bullish": bullish, "bearish": bearish}}
        except Exception:
            return {"fvg": {"bullish": [], "bearish": []}}

    def detect_swing_points(self, market_data: Dict[str, pd.DataFrame], current_price: float) -> Dict[str, Any]:
        """公共方法：检测摆动点（适配测试期望的返回结构）"""
        try:
            tf = "1h" if isinstance(market_data, dict) and "1h" in market_data else (next(iter(market_data.keys())) if isinstance(market_data, dict) and market_data else None)
            df = market_data.get(tf) if isinstance(market_data, dict) else None
            if df is None or not isinstance(df, pd.DataFrame):
                return {"swing_points": {"highs": [], "lows": []}}

            sps = self._detect_swing_points(df, tf)
            highs = [s for s in sps if str(s.get('type', '')).lower() == 'swing_high']
            lows = [s for s in sps if str(s.get('type', '')).lower() == 'swing_low']
            return {"swing_points": {"highs": highs, "lows": lows}}
        except Exception:
            return {"swing_points": {"highs": [], "lows": []}}

    def detect_all_structures(self, market_data: Dict[str, pd.DataFrame], current_price: float) -> Dict[str, Any]:
        """公共方法：一次性检测所有SMC结构，包含overall_score别名"""
        try:
            res = {}
            res.update(self.detect_bos_choch(market_data, current_price))
            res.update(self.detect_order_blocks(market_data, current_price))
            res.update(self.detect_fvg(market_data, current_price))
            res.update(self.detect_swing_points(market_data, current_price))

            # 计算总体结构评分
            tf = "1h" if isinstance(market_data, dict) and "1h" in market_data else (next(iter(market_data.keys())) if isinstance(market_data, dict) and market_data else None)
            df = market_data.get(tf) if isinstance(market_data, dict) else None
            smc = self.detect_smc_structures(df, tf) if isinstance(df, pd.DataFrame) else {}
            overall = smc.get('overall_score', smc.get('structure_score', 0.0))
            res['overall_score'] = overall

            # 构建并注入OB/FVG优化结果（overlay_result）
            try:
                order_blocks = res.get('order_blocks', {})
                fvg = res.get('fvg', {})
                cp = float(current_price) if isinstance(current_price, (int, float)) else (float(df['close'].iloc[-1]) if isinstance(df, pd.DataFrame) and len(df) > 0 else 0.0)
                res['ob_fvg_optimized'] = self._build_ob_fvg_optimized(df, order_blocks, fvg, cp)
            except Exception:
                res['ob_fvg_optimized'] = {
                    'ob_fvg_summary': 'weak_or_invalid',
                    'meaningful_ob_count': 0,
                    'meaningful_fvg_count': 0,
                    'strongest_structure': None,
                    'price_relevance': 0.0,
                    'freshness_score': 0.0,
                    'overlay_result': {
                        'has_overlay': False,
                        'overlay_confidence_boost': 0.0,
                        'overlay_details': [],
                        'narrow_ob_for_entry': None,
                        'wide_ob_for_stop_loss': None
                    }
                }
            return res
        except Exception:
            return {
                'bos_choch': {"signal": "HOLD", "confidence": 0.0, "strength": 0.0},
                'order_blocks': {"bullish": [], "bearish": []},
                'fvg': {"bullish": [], "bearish": []},
                'swing_points': {"highs": [], "lows": []},
                'overall_score': 0.0
            }

    def _build_ob_fvg_optimized(self, df: Optional[pd.DataFrame], order_blocks: Any, fvg: Any, current_price: float) -> Dict[str, Any]:
        """构建轻量级的OB/FVG优化结果，始终提供overlay_result键以保证下游安全使用"""
        # 规范化输入为列表
        try:
            if isinstance(order_blocks, dict):
                ob_list = list(order_blocks.get('bullish', [])) + list(order_blocks.get('bearish', []))
            else:
                ob_list = list(order_blocks) if isinstance(order_blocks, list) else []

            if isinstance(fvg, dict):
                fvg_list = list(fvg.get('bullish', [])) + list(fvg.get('bearish', []))
            else:
                fvg_list = list(fvg) if isinstance(fvg, list) else []

            meaningful_ob_count = len(ob_list)
            meaningful_fvg_count = len(fvg_list)

            summary = 'weak_or_invalid'
            if meaningful_ob_count > 0 and meaningful_fvg_count > 0:
                summary = 'strong_structure'
            elif meaningful_ob_count > 0:
                summary = 'ob_dominant'
            elif meaningful_fvg_count > 0:
                summary = 'fvg_dominant'

            optimized = {
                'ob_fvg_summary': summary,
                'meaningful_ob_count': meaningful_ob_count,
                'meaningful_fvg_count': meaningful_fvg_count,
                'strongest_structure': None,
                'price_relevance': 0.0,
                'freshness_score': 0.0,
                'overlay_result': {
                    'has_overlay': False,
                    'overlay_confidence_boost': 0.0,
                    'overlay_details': [],
                    'narrow_ob_for_entry': None,
                    'wide_ob_for_stop_loss': None
                }
            }
            return optimized
        except Exception:
            return {
                'ob_fvg_summary': 'error',
                'meaningful_ob_count': 0,
                'meaningful_fvg_count': 0,
                'strongest_structure': None,
                'price_relevance': 0.0,
                'freshness_score': 0.0,
                'overlay_result': {
                    'has_overlay': False,
                    'overlay_confidence_boost': 0.0,
                    'overlay_details': [],
                    'narrow_ob_for_entry': None,
                    'wide_ob_for_stop_loss': None
                }
            }

    def calculate_structure_strength(self, structure: Dict[str, Any]) -> float:
        """公共方法：计算单个结构强度为0-1范围"""
        try:
            base = structure.get('strength', 0.0)
            base = base if isinstance(base, (int, float)) else 0.0
            score = float(base)
            if structure.get('volume_confirmation'):
                score += 0.1
            if structure.get('price_rejection'):
                score += 0.1
            return min(max(score, 0.0), 1.0)
        except Exception:
            return 0.0
    
    def _detect_bos_choch(self, df: pd.DataFrame, tf: str) -> Dict[str, Any]:
        """检测BOS/CHOCH结构"""
        try:
            # 计算ATR
            atr = self._atr(df, 14).iloc[-1] if len(df) >= 14 else df['close'].std()
            
            # 智能BOS强度计算
            bos_strength = self._calculate_intelligent_bos_strength(df, tf, atr)
            
            # 智能FVG数量计算
            fvg_count = self._calculate_intelligent_fvg_count(df, tf)
            
            # 智能OB数量计算
            ob_count = self._calculate_intelligent_ob_count(df, tf)
            
            # 检测BOS/CHOCH
            bos_choch = []
            for i in range(5, len(df)):
                current = df.iloc[i]
                prev = df.iloc[i-1]
                prev2 = df.iloc[i-2]
                
                # 看涨BOS/CHOCH
                if (current['high'] > prev['high'] and 
                    prev['high'] > prev2['high'] and
                    current['close'] > prev['close']):
                    
                    strength = self._calculate_bos_strength(current, prev, prev2, atr)
                    if strength > bos_strength:
                        bos_choch.append({
                            'type': 'bullish_bos',
                            'high': current['high'],
                            'low': current['low'],
                            'time': current.name,
                            'strength': strength,
                            'validity_score': min(strength, 3.0)
                        })
                
                # 看跌BOS/CHOCH
                elif (current['low'] < prev['low'] and 
                      prev['low'] < prev2['low'] and
                      current['close'] < prev['close']):
                    
                    strength = self._calculate_bos_strength(current, prev, prev2, atr)
                    if strength > bos_strength:
                        bos_choch.append({
                            'type': 'bearish_bos',
                            'high': current['high'],
                            'low': current['low'],
                            'time': current.name,
                            'strength': strength,
                            'validity_score': min(strength, 3.0)
                        })
            
            return {
                'structures': bos_choch,
                'count': len(bos_choch),
                'fvg_count': fvg_count,
                'ob_count': ob_count
            }
            
        except Exception as e:
            self.logger.error(f"BOS/CHOCH检测失败 {tf}: {e}")
            return {'structures': [], 'count': 0, 'fvg_count': 0, 'ob_count': 0}
    
    def _detect_order_blocks(self, df: pd.DataFrame, tf: str) -> List[Dict[str, Any]]:
        """检测订单块"""
        try:
            order_blocks = []
            
            for i in range(2, len(df)):
                current_candle = df.iloc[i]
                prev_candle = df.iloc[i-1]
                prev2_candle = df.iloc[i-2]
                
                # 计算ATR和成交量指标
                atr = self._atr(df, 14).iloc[-1] if len(df) >= 14 else df['close'].std()
                volume_ma = df['volume'].rolling(20).mean().iloc[-1] if len(df) >= 20 else df['volume'].mean()
                current_volume = current_candle['volume']
                volume_ratio = current_volume / volume_ma if volume_ma > 0 else 1.0
                
                # 看涨订单块：大阳线后出现小阴线 + 成交量确认
                if (current_candle['close'] > current_candle['open'] and  # 当前阳线
                    prev_candle['close'] > prev_candle['open'] and        # 前一根阳线
                    prev2_candle['close'] < prev2_candle['open'] and      # 前两根是阴线（整理）
                    (current_candle['close'] - current_candle['open']) > (prev_candle['high'] - prev_candle['low']) * 0.7):  # 大阳线
                    
                    body_size = current_candle['close'] - current_candle['open']
                    ob_size = abs(current_candle['open'] - prev_candle['close'])
                    body_ratio = body_size / atr if atr > 0 else 0
                    depth_ratio = ob_size / atr if atr > 0 else 0
                    
                    # 有效性验证：实体大小和深度要求
                    if body_ratio > 0.5 and depth_ratio > 0.1 and volume_ratio > 0.8:  # 实体>0.5ATR，深度>0.1ATR，成交量放大
                        order_blocks.append({
                            'type': 'bullish_ob',
                            'high': min(current_candle['open'], prev_candle['close']),
                            'low': max(current_candle['open'], prev_candle['close']),
                            'body_size': body_size,
                            'depth_size': ob_size,
                            'body_ratio': body_ratio,
                            'depth_ratio': depth_ratio,
                            'volume_ratio': volume_ratio,
                            'strength': body_ratio * volume_ratio,  # 综合强度
                            'liquidity_score': min(volume_ratio, 2.0),
                            'depth_score': min(depth_ratio, 1.0),
                            'validity_score': min(body_ratio * depth_ratio * volume_ratio, 5.0)  # 有效性评分
                        })
                
                # 看跌订单块：大阴线后出现小阳线 + 成交量确认
                if (current_candle['close'] < current_candle['open'] and  # 当前阴线
                    prev_candle['close'] < prev_candle['open'] and        # 前一根阴线
                    prev2_candle['close'] > prev2_candle['open'] and      # 前两根是阳线（整理）
                    abs(current_candle['close'] - current_candle['open']) > (prev_candle['high'] - prev_candle['low']) * 0.7):  # 大阴线
                    
                    body_size = abs(current_candle['close'] - current_candle['open'])
                    ob_size = abs(current_candle['open'] - prev_candle['close'])
                    body_ratio = body_size / atr if atr > 0 else 0
                    depth_ratio = ob_size / atr if atr > 0 else 0
                    
                    # 有效性验证：实体大小和深度要求
                    if body_ratio > 0.5 and depth_ratio > 0.1 and volume_ratio > 0.8:  # 实体>0.5ATR，深度>0.1ATR，成交量放大
                        order_blocks.append({
                            'type': 'bearish_ob',
                            'high': min(current_candle['open'], prev_candle['close']),
                            'low': max(current_candle['open'], prev_candle['close']),
                            'body_size': body_size,
                            'depth_size': ob_size,
                            'body_ratio': body_ratio,
                            'depth_ratio': depth_ratio,
                            'volume_ratio': volume_ratio,
                            'strength': body_ratio * volume_ratio,  # 综合强度
                            'liquidity_score': min(volume_ratio, 2.0),
                            'depth_score': min(depth_ratio, 1.0),
                            'validity_score': min(body_ratio * depth_ratio * volume_ratio, 5.0)  # 有效性评分
                        })
            
            return order_blocks
            
        except Exception as e:
            self.logger.error(f"订单块检测失败 {tf}: {e}")
            return []
    
    def _detect_fvg(self, df: pd.DataFrame, tf: str) -> List[Dict[str, Any]]:
        """检测公平价值缺口"""
        try:
            fvgs = []
            
            for i in range(3, len(df)):
                current = df.iloc[i]
                prev = df.iloc[i-1]
                prev2 = df.iloc[i-2]
                
                # 计算ATR和成交量指标
                atr = self._atr(df, 14).iloc[-1] if len(df) >= 14 else df['close'].std()
                volume_ma = df['volume'].rolling(20).mean().iloc[-1] if len(df) >= 20 else df['volume'].mean()
                current_volume = current['volume']
                volume_ratio = current_volume / volume_ma if volume_ma > 0 else 1.0
                
                # 看涨FVG：价格向上跳空 + 成交量确认
                if (prev['high'] < current['low'] and  # 缺口存在
                    prev2['close'] > prev2['open'] and    # 前一根是阳线
                    current['close'] > current['open']):  # 当前也是阳线
                    
                    gap_size = current['low'] - prev['high']
                    gap_ratio = gap_size / atr if atr > 0 else 0
                    
                    # 有效性验证：缺口大小和成交量要求
                    if gap_ratio > 0.2 and volume_ratio > 0.8:  # 缺口至少0.2ATR，成交量放大
                        fvgs.append({
                            'type': 'bullish_fvg',
                            'high': prev['high'],
                            'low': current['low'],
                            'gap_size': gap_size,
                            'gap_ratio': gap_ratio,
                            'volume_ratio': volume_ratio,
                            'strength': gap_ratio * volume_ratio,  # 综合强度
                            'atr_normalized': gap_ratio,
                            'liquidity_score': min(volume_ratio, 2.0),
                            'validity_score': min(gap_ratio * volume_ratio, 3.0)  # 有效性评分
                        })
                
                # 看跌FVG：价格向下跳空 + 成交量确认
                if (prev['low'] > current['high'] and  # 缺口存在
                    prev2['close'] < prev2['open'] and    # 前一根是阴线
                    current['close'] < current['open']):  # 当前也是阴线
                    
                    gap_size = prev['low'] - current['high']
                    gap_ratio = gap_size / atr if atr > 0 else 0
                    
                    # 有效性验证：缺口大小和成交量要求
                    if gap_ratio > 0.2 and volume_ratio > 0.8:  # 缺口至少0.2ATR，成交量放大
                        fvgs.append({
                            'type': 'bearish_fvg',
                            'high': current['high'],
                            'low': prev['low'],
                            'gap_size': gap_size,
                            'gap_ratio': gap_ratio,
                            'volume_ratio': volume_ratio,
                            'strength': gap_ratio * volume_ratio,  # 综合强度
                            'atr_normalized': gap_ratio,
                            'liquidity_score': min(volume_ratio, 2.0),
                            'validity_score': min(gap_ratio * volume_ratio, 3.0)  # 有效性评分
                        })
            
            return fvgs
            
        except Exception as e:
            self.logger.error(f"FVG检测失败 {tf}: {e}")
            return []
    
    def _detect_swing_points(self, df: pd.DataFrame, tf: str) -> List[Dict[str, Any]]:
        """检测摆动点"""
        try:
            swing_points = []
            
            # 使用简单的摆动点检测算法
            for i in range(2, len(df) - 2):
                current = df.iloc[i]
                prev1 = df.iloc[i-1]
                prev2 = df.iloc[i-2]
                next1 = df.iloc[i+1]
                next2 = df.iloc[i+2]
                
                # 看涨摆动点（低点）
                if (current['low'] < prev1['low'] and 
                    current['low'] < prev2['low'] and
                    current['low'] < next1['low'] and
                    current['low'] < next2['low']):
                    
                    swing_points.append({
                        'type': 'bullish_swing',
                        'price': current['low'],
                        'time': current.name,
                        'strength': self._calculate_swing_strength(current, df, 'bullish')
                    })
                
                # 看跌摆动点（高点）
                if (current['high'] > prev1['high'] and 
                    current['high'] > prev2['high'] and
                    current['high'] > next1['high'] and
                    current['high'] > next2['high']):
                    
                    swing_points.append({
                        'type': 'bearish_swing',
                        'price': current['high'],
                        'time': current.name,
                        'strength': self._calculate_swing_strength(current, df, 'bearish')
                    })
            
            return swing_points
            
        except Exception as e:
            self.logger.error(f"摆动点检测失败 {tf}: {e}")
            return []
    
    def _calculate_structure_score(self, bos_choch: Dict, order_blocks: List, fvg: List, swing_points: List) -> float:
        """计算结构强度评分"""
        try:
            # BOS/CHOCH评分
            bos_score = 0
            if bos_choch and 'structures' in bos_choch:
                for structure in bos_choch['structures']:
                    bos_score += structure.get('strength', 0) * structure.get('validity_score', 0)
                bos_score = min(bos_score / len(bos_choch['structures']) if bos_choch['structures'] else 0, 3.0)
            
            # 订单块评分
            ob_score = 0
            if order_blocks:
                for ob in order_blocks:
                    ob_score += ob.get('strength', 0) * ob.get('validity_score', 0)
                ob_score = min(ob_score / len(order_blocks) if order_blocks else 0, 5.0)
            
            # FVG评分
            fvg_score = 0
            if fvg:
                for fvg_item in fvg:
                    fvg_score += fvg_item.get('strength', 0) * fvg_item.get('validity_score', 0)
                fvg_score = min(fvg_score / len(fvg) if fvg else 0, 3.0)
            
            # 摆动点评分
            swing_score = 0
            if swing_points:
                for swing in swing_points:
                    swing_score += swing.get('strength', 0)
                swing_score = min(swing_score / len(swing_points) if swing_points else 0, 2.0)
            
            # 综合评分（加权平均）
            total_score = (
                self.config.structure_weights['bos_choch'] * min(bos_score / 3.0, 1.0) +
                self.config.structure_weights['ob_fvg'] * min((ob_score + fvg_score) / 8.0, 1.0) +
                self.config.structure_weights['swing_strength'] * min(swing_score / 2.0, 1.0)
            )
            
            return min(max(total_score, 0.0), 1.0)
            
        except Exception as e:
            self.logger.error(f"结构评分计算失败: {e}")
            return 0.0
    
    def _calculate_bos_strength(self, current: pd.Series, prev: pd.Series, prev2: pd.Series, atr: float) -> float:
        """计算BOS强度"""
        try:
            # 价格变化
            price_change = abs(current['close'] - prev2['close'])
            
            # 成交量变化
            volume_change = current['volume'] - prev2['volume']
            volume_ratio = current['volume'] / prev2['volume'] if prev2['volume'] > 0 else 1.0
            
            # 实体大小
            body_size = abs(current['close'] - current['open'])
            body_ratio = body_size / atr if atr > 0 else 0
            
            # 综合强度
            strength = (price_change / atr if atr > 0 else 0) * body_ratio * min(volume_ratio, 2.0)
            
            return min(strength, 3.0)
            
        except Exception as e:
            self.logger.error(f"BOS强度计算失败: {e}")
            return 0.0
    
    def _calculate_swing_strength(self, current: pd.Series, df: pd.DataFrame, direction: str) -> float:
        """计算摆动点强度"""
        try:
            # 获取前后数据
            idx = current.name
            idx_num = df.index.get_loc(idx)
            
            # 前后数据范围
            lookback = min(10, idx_num)
            lookahead = min(10, len(df) - idx_num - 1)
            
            if lookback < 3 or lookahead < 3:
                return 0.0
            
            # 计算相对高度/深度
            if direction == 'bullish':
                # 看涨摆动点：相对低点深度
                prev_data = df.iloc[idx_num-lookback:idx_num]
                next_data = df.iloc[idx_num+1:idx_num+lookahead+1]
                
                prev_high = prev_data['high'].max()
                next_high = next_data['high'].max()
                
                depth = min(prev_high, next_high) - current['low']
                avg_range = (prev_data['high'].max() - prev_data['low'].min() + 
                           next_data['high'].max() - next_data['low'].min()) / 2
                
                strength = depth / avg_range if avg_range > 0 else 0
            
            else:
                # 看跌摆动点：相对高点高度
                prev_data = df.iloc[idx_num-lookback:idx_num]
                next_data = df.iloc[idx_num+1:idx_num+lookahead+1]
                
                prev_low = prev_data['low'].min()
                next_low = next_data['low'].min()
                
                height = current['high'] - max(prev_low, next_low)
                avg_range = (prev_data['high'].max() - prev_data['low'].min() + 
                           next_data['high'].max() - next_data['low'].min()) / 2
                
                strength = height / avg_range if avg_range > 0 else 0
            
            return min(strength, 2.0)
            
        except Exception as e:
            self.logger.error(f"摆动点强度计算失败: {e}")
            return 0.0
    
    def _calculate_intelligent_bos_strength(self, df: pd.DataFrame, tf: str, atr: float) -> float:
        """智能BOS强度计算 - 基于价格行为的多维度分析"""
        try:
            # 基于时间框架的基准强度
            timeframe_base = {
                '1d': 0.8, '4h': 1.2, '1h': 1.5, '15m': 1.8, '3m': 2.0, '1m': 0.5
            }.get(tf, 1.5)
            
            # 价格波动性因子
            price_volatility = df['close'].std()
            volatility_factor = max(0.5, min(price_volatility / (df['close'].mean() * 0.01), 2.0))
            
            # 价格趋势因子
            short_ma = df['close'].tail(5).mean()
            long_ma = df['close'].tail(20).mean()
            trend_factor = 1.3 if short_ma > long_ma else 0.7  # 上升趋势增加强度
            
            # 价格范围因子
            recent_price_range = df['close'].max() - df['close'].min()
            range_factor = max(0.5, min(recent_price_range / (atr * 3), 2.0))
            
            # 成交量确认因子
            volume_avg = df['volume'].mean()
            recent_volume = df['volume'].tail(10).mean()
            volume_factor = max(0.5, min(recent_volume / volume_avg, 1.5)) if volume_avg > 0 else 1.0
            
            # 计算智能BOS强度
            intelligent_bos = timeframe_base * volatility_factor * trend_factor * range_factor * volume_factor
            
            # 限制在合理范围内
            bos_strength = max(0.1, min(intelligent_bos, 3.0))
            
            self.logger.debug(f"🔍 {tf} 智能BOS计算: 基准={timeframe_base}, 波动={volatility_factor:.2f}, 趋势={trend_factor:.2f}, 范围={range_factor:.2f}, 成交量={volume_factor:.2f}, 最终={bos_strength:.2f}")
            
            return bos_strength
            
        except Exception as e:
            self.logger.warning(f"智能BOS计算失败: {e}，使用备选计算")
            # 备选计算
            recent_price_range = df['close'].max() - df['close'].min()
            return max(0.1, min(recent_price_range / (atr * 2), 1.5)) if atr > 0 else max(0.1, recent_price_range / (df['close'].std() * 3))
    
    def _calculate_intelligent_fvg_count(self, df: pd.DataFrame, tf: str) -> int:
        """智能FVG数量计算 - 基于价格行为的多维度分析"""
        try:
            # 基于时间框架的基准数量
            timeframe_base = {
                '1d': 3, '4h': 8, '1h': 15, '15m': 25, '3m': 35, '1m': 45
            }.get(tf, 15)
            
            # 价格波动性因子
            price_volatility = df['close'].std()
            volatility_factor = max(0.5, min(price_volatility / (df['close'].mean() * 0.01), 2.0))
            
            # 价格趋势因子
            short_ma = df['close'].tail(5).mean()
            long_ma = df['close'].tail(20).mean()
            trend_factor = 1.2 if short_ma > long_ma else 0.8  # 上升趋势增加FVG数量
            
            # 价格范围因子
            price_range = df['high'].max() - df['low'].min()
            atr = self._atr(df, 14).iloc[-1] if len(df) >= 14 else price_volatility
            range_factor = max(0.5, min(price_range / (atr * 5), 2.0))
            
            # 成交量因子（FVG通常伴随低成交量）
            volume_avg = df['volume'].mean()
            recent_volume = df['volume'].tail(10).mean()
            volume_factor = max(0.5, min(volume_avg / recent_volume, 2.0)) if recent_volume > 0 else 1.0
            
            # 计算智能FVG数量
            intelligent_fvg = int(timeframe_base * volatility_factor * trend_factor * range_factor * volume_factor)
            
            # 限制在合理范围内
            fvg_count = max(1, min(intelligent_fvg, len(df) // 5))
            
            self.logger.debug(f"🔍 {tf} 智能FVG计算: 基准={timeframe_base}, 波动={volatility_factor:.2f}, 趋势={trend_factor:.2f}, 范围={range_factor:.2f}, 成交量={volume_factor:.2f}, 最终={fvg_count}")
            
            return fvg_count
            
        except Exception as e:
            self.logger.warning(f"智能FVG计算失败: {e}，使用备选计算")
            # 备选计算
            return max(1, min(len(df) // 10, 20))
    
    def _calculate_intelligent_ob_count(self, df: pd.DataFrame, tf: str) -> int:
        """智能OB数量计算 - 基于价格行为的多维度分析"""
        try:
            # 基于时间框架的基准数量
            timeframe_base = {
                '1d': 2, '4h': 6, '1h': 12, '15m': 18, '3m': 25, '1m': 30
            }.get(tf, 12)
            
            # 价格波动性因子
            price_volatility = df['close'].std()
            volatility_factor = max(0.5, min(price_volatility / (df['close'].mean() * 0.01), 2.0))
            
            # 成交量因子（OB通常伴随高成交量）
            volume_avg = df['volume'].mean()
            recent_volume = df['volume'].tail(10).mean()
            volume_factor = max(0.5, min(recent_volume / volume_avg, 2.0)) if volume_avg > 0 else 1.0
            
            # 价格趋势因子
            short_ma = df['close'].tail(5).mean()
            long_ma = df['close'].tail(20).mean()
            trend_factor = 1.2 if short_ma > long_ma else 0.8  # 上升趋势增加OB数量
            
            # 价格范围因子
            price_range = df['high'].max() - df['low'].min()
            atr = self._atr(df, 14).iloc[-1] if len(df) >= 14 else price_volatility
            range_factor = max(0.5, min(price_range / (atr * 5), 2.0))
            
            # 计算智能OB数量
            intelligent_ob = int(timeframe_base * volatility_factor * volume_factor * trend_factor * range_factor)
            
            # 限制在合理范围内
            ob_count = max(1, min(intelligent_ob, len(df) // 8))
            
            self.logger.debug(f"🔍 {tf} 智能OB计算: 基准={timeframe_base}, 波动={volatility_factor:.2f}, 成交量={volume_factor:.2f}, 趋势={trend_factor:.2f}, 范围={range_factor:.2f}, 最终={ob_count}")
            
            return ob_count
            
        except Exception as e:
            self.logger.warning(f"智能OB计算失败: {e}，使用备选计算")
            # 备选计算
            return max(1, min(len(df) // 15, 15))
    
    def _atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算平均真实范围"""
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            
            return atr
            
        except Exception as e:
            self.logger.error(f"ATR计算失败: {e}")
            return pd.Series([df['close'].std()] * len(df), index=df.index)


class MTFAnalyzer:
    """多时间框架分析器"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.smc_detector = SMCDetector(config, logger)
    
    def analyze_mtf_structures(self, multi_tf_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """分析多时间框架结构"""
        try:
            mtf_analysis = {}
            
            # 分析每个时间框架
            for tf, df in multi_tf_data.items():
                if len(df) < 10:
                    continue
                
                # 检测SMC结构
                smc_structures = self.smc_detector.detect_smc_structures(df, tf)
                
                # 分析趋势
                trend_data = self._analyze_trend(df)
                
                # 计算一致性
                consistency = self._calculate_tf_consistency(smc_structures, trend_data)
                
                mtf_analysis[tf] = {
                    'smc_structures': smc_structures,
                    'trend': trend_data['direction'],
                    'strength': trend_data['strength'],
                    'consistency': consistency,
                    'structure_score': smc_structures.get('structure_score', 0)
                }
            
            # 计算多时间框架一致性
            overall_consistency = self._calculate_overall_consistency(mtf_analysis)
            
            # 生成建议
            recommendation = self._generate_recommendation(mtf_analysis, overall_consistency)
            
            return {
                'timeframes': mtf_analysis,
                'consistency': overall_consistency,
                'recommendation': recommendation
            }
            
        except Exception as e:
            self.logger.error(f"多时间框架分析失败: {e}")
            return {}

    # ===== 公共适配方法（匹配测试期望的接口） =====
    def analyze_multiple_timeframes(self, market_data: Dict[str, pd.DataFrame], current_price: float) -> Dict[str, Any]:
        """适配器：分析多时间框架，返回测试期望的字段"""
        try:
            base = self.analyze_mtf_structures(market_data)
            tf_data = base.get('timeframes', {})
            mtf_signals = {}
            for tf, d in tf_data.items():
                direction = d.get('trend', 'neutral')
                consistency = d.get('consistency', 0.0)
                signal = 'BUY' if direction == 'bullish' else ('SELL' if direction == 'bearish' else 'HOLD')
                mtf_signals[tf] = {
                    'signal': signal,
                    'strength': consistency
                }
            consensus_result = self.calculate_mtf_consensus(mtf_signals)
            return {
                'mtf_signals': mtf_signals,
                'consensus': consensus_result.get('consensus', 'HOLD'),
                'overall_score': base.get('consistency', 0.0)
            }
        except Exception:
            return {'mtf_signals': {}, 'consensus': 'HOLD', 'overall_score': 0.0}

    def calculate_mtf_consensus(self, mtf_signals: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """适配器：计算多时间框架共识"""
        try:
            weights = getattr(self.config, 'mtf_weight', {
                '15m': 0.1, '1h': 0.3, '4h': 0.3, '1d': 0.3
            })
            total_w = 0.0
            score_sum = 0.0
            for tf, s in mtf_signals.items():
                w = float(weights.get(tf, 0.1))
                st = s.get('strength', 0.5)
                st = st if isinstance(st, (int, float)) else 0.5
                # BUY=1, HOLD=0.5, SELL=0
                val = 1.0 if s.get('signal') == 'BUY' else (0.0 if s.get('signal') == 'SELL' else 0.5)
                score_sum += val * w * st
                total_w += w
            weighted_score = (score_sum / total_w) if total_w > 0 else 0.5
            consensus = 'BUY' if weighted_score > 0.55 else ('SELL' if weighted_score < 0.45 else 'HOLD')
            confidence = abs(weighted_score - 0.5) * 2
            confidence = max(0.0, min(1.0, confidence))
            return {
                'consensus': consensus,
                'confidence': confidence,
                'weighted_score': weighted_score
            }
        except Exception:
            return {'consensus': 'HOLD', 'confidence': 0.0, 'weighted_score': 0.5}

    def calculate_timeframe_weight(self, tf: str) -> float:
        """适配器：返回指定时间框架的权重"""
        try:
            weights = getattr(self.config, 'mtf_weight', {'15m': 0.1, '1h': 0.3, '4h': 0.3, '1d': 0.3})
            return float(weights.get(tf, 0.1))
        except Exception:
            return 0.1

    def validate_mtf_signal(self, mtf_signals: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """适配器：对多时间框架信号进行简单验证"""
        try:
            consensus = self.calculate_mtf_consensus(mtf_signals)
            conf = consensus.get('confidence', 0.0)
            valid = conf >= 0.3
            reason = '一致性足够' if valid else '一致性不足'
            return {'valid': valid, 'reason': reason, 'confidence': conf}
        except Exception:
            return {'valid': False, 'reason': '计算失败', 'confidence': 0.0}
    
    def _analyze_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析趋势"""
        try:
            # 计算移动平均线
            ema_20 = df['close'].ewm(span=20).mean().iloc[-1]
            ema_50 = df['close'].ewm(span=50).mean().iloc[-1]
            ema_200 = df['close'].ewm(span=200).mean().iloc[-1]
            
            current_price = df['close'].iloc[-1]
            
            # 确定趋势方向
            if current_price > ema_20 > ema_50 > ema_200:
                direction = 'bullish'
                strength = min((current_price - ema_200) / ema_200 * 10, 1.0)
            elif current_price < ema_20 < ema_50 < ema_200:
                direction = 'bearish'
                strength = min((ema_200 - current_price) / ema_200 * 10, 1.0)
            else:
                direction = 'neutral'
                strength = 0.5
            
            return {
                'direction': direction,
                'strength': strength,
                'ema_20': ema_20,
                'ema_50': ema_50,
                'ema_200': ema_200
            }
            
        except Exception as e:
            self.logger.error(f"趋势分析失败: {e}")
            return {'direction': 'neutral', 'strength': 0.0}
    
    def _calculate_tf_consistency(self, smc_structures: Dict, trend_data: Dict) -> float:
        """计算时间框架内一致性"""
        try:
            # 结构评分（兼容strength/structure，并归一化到[0,1]）
            structure_score = smc_structures.get('structure_score', smc_structures.get('strength_score', 0))
            structure_score = min(max(structure_score, 0.0), 1.0)
            
            # 趋势强度
            trend_strength = trend_data.get('strength', 0)
            
            # 一致性评分
            consistency = (structure_score + trend_strength) / 2
            
            return min(max(consistency, 0.0), 1.0)
            
        except Exception as e:
            self.logger.error(f"时间框架一致性计算失败: {e}")
            return 0.0
    
    def _calculate_overall_consistency(self, mtf_analysis: Dict) -> float:
        """计算整体多时间框架一致性"""
        try:
            if not mtf_analysis:
                return 0.0
            
            # 计算加权平均（高时间框架权重更高）
            tf_weights = {
                '1d': 0.4, '4h': 0.3, '1h': 0.2, '15m': 0.1
            }
            
            weighted_sum = 0
            total_weight = 0
            
            for tf, data in mtf_analysis.items():
                consistency = data.get('consistency', 0)
                weight = tf_weights.get(tf, 0.1)
                weighted_sum += consistency * weight
                total_weight += weight
            
            if total_weight == 0:
                return 0.0
            
            overall_consistency = weighted_sum / total_weight
            
            return min(max(overall_consistency, 0.0), 1.0)
            
        except Exception as e:
            self.logger.error(f"整体一致性计算失败: {e}")
            return 0.0
    
    def _generate_recommendation(self, mtf_analysis: Dict, overall_consistency: float) -> str:
        """生成交易建议"""
        try:
            if overall_consistency >= 0.8:
                return "强烈建议交易 - 多时间框架高度一致"
            elif overall_consistency >= 0.6:
                return "建议交易 - 多时间框架基本一致"
            elif overall_consistency >= 0.4:
                return "谨慎交易 - 多时间框架部分一致"
            else:
                return "不建议交易 - 多时间框架不一致"
                
        except Exception as e:
            self.logger.error(f"建议生成失败: {e}")
            return "无法生成建议"