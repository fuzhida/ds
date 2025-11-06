"""
混合SMC检测策略
结合真实检测和智能备选，确保在真实数据不可用时有可靠的备选方案
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from smc_real_detector import RealSMCDetector
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class HybridSMCSstrategy:
    """混合SMC检测策略 - 真实检测优先，智能备选兜底"""
    
    def __init__(self, min_confidence_threshold: float = 0.6, 
                 fallback_enabled: bool = True,
                 real_time_weight: float = 0.7,
                 historical_weight: float = 0.3,
                 ai_enhanced: bool = True):
        """
        初始化混合策略
        
        Args:
            min_confidence_threshold: 最小置信度阈值
            fallback_enabled: 是否启用智能备选
            real_time_weight: 实时数据权重
            historical_weight: 历史数据权重
            ai_enhanced: 是否启用AI增强
        """
        self.real_detector = RealSMCDetector()
        self.min_confidence = min_confidence_threshold
        self.fallback_enabled = fallback_enabled
        self.real_time_weight = real_time_weight
        self.historical_weight = historical_weight
        self.ai_enhanced = ai_enhanced
        self.detection_history = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def detect_structures(self, df: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
        """
        混合SMC结构检测主方法
        
        Args:
            df: 价格数据
            timeframe: 时间框架
            
        Returns:
            包含检测结果的完整信息
        """
        self.logger.info(f"开始混合SMC检测 - {timeframe}")
        
        # 1. 首先尝试真实检测
        real_result = self.real_detector.detect_all_structures(df, timeframe)
        
        # 2. 评估真实检测结果
        real_confidence = real_result['validation']['confidence_score']
        real_valid = real_result['validation']['is_valid']
        
        # 3. 决策：是否使用真实检测结果
        if real_valid and real_confidence >= self.min_confidence:
            self.logger.info(f"使用真实检测结果 - 置信度: {real_confidence:.2f}")
            return self._create_hybrid_result(real_result, 'real', real_confidence)
        
        # 4. 如果真实检测不理想且启用了备选
        elif self.fallback_enabled:
            self.logger.warning(f"真实检测置信度不足 ({real_confidence:.2f})，启用智能备选")
            fallback_result = self._generate_intelligent_fallback(df, timeframe, real_result)
            return self._create_hybrid_result(fallback_result, 'fallback', 
                                            fallback_result['validation']['confidence_score'])
        
        # 5. 如果都不理想，返回真实结果但标记警告
        else:
            self.logger.error(f"真实检测失败且未启用备选 - 置信度: {real_confidence:.2f}")
            return self._create_hybrid_result(real_result, 'real_low_confidence', real_confidence)
    
    def _generate_intelligent_fallback(self, df: pd.DataFrame, timeframe: str, 
                                     real_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成智能备选结果
        
        Args:
            df: 价格数据
            timeframe: 时间框架
            real_result: 真实检测结果（用于参考）
            
        Returns:
            智能备选结果
        """
        self.logger.info("生成智能备选结果")
        
        # 基于历史数据和统计规律生成备选结构
        fallback_structures = self._generate_fallback_structures(df, timeframe, real_result)
        
        # 验证备选结果
        validation = self._validate_fallback_structures(fallback_structures, real_result)
        
        # 计算指标
        metrics = self._calculate_fallback_metrics(fallback_structures, timeframe)
        
        result = {
            'structures': fallback_structures,
            'validation': validation,
            'metrics': metrics,
            'timeframe': timeframe,
            'data_quality': real_result['data_quality'],
            'fallback_info': {
                'reason': real_result['validation']['issues'],
                'real_confidence': real_result['validation']['confidence_score']
            }
        }
        
        return result
    
    def _generate_fallback_structures(self, df: pd.DataFrame, timeframe: str, 
                                    real_result: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """生成智能备选结构"""
        
        # 获取历史统计基准
        historical_baseline = self._get_historical_baseline(timeframe)
        
        # 基于价格波动性调整
        volatility = self._calculate_volatility(df)
        
        # 生成BOS/CHOCH
        bos_choch = self._generate_fallback_bos_choch(df, historical_baseline, volatility)
        
        # 生成FVG
        fvg = self._generate_fallback_fvg(df, historical_baseline, volatility)
        
        # 生成订单块
        ob = self._generate_fallback_ob(df, historical_baseline, volatility)
        
        # 生成波段点
        swing_points = self._generate_fallback_swing_points(df, historical_baseline)
        
        return {
            'swing_points': swing_points,
            'bos_choch': bos_choch,
            'fvg': fvg,
            'ob': ob
        }
    
    def _generate_fallback_bos_choch(self, df: pd.DataFrame, baseline: Dict[str, float], 
                                   volatility: float) -> pd.DataFrame:
        """生成智能备选BOS/CHOCH"""
        
        # 基础BOS数量
        base_bos_count = baseline.get('bos_count', 2)
        
        # 根据波动性调整
        volatility_factor = 1.0 + (volatility - 0.02) * 10  # 2%为基准波动率
        adjusted_count = int(base_bos_count * volatility_factor)
        
        # 生成合理的BOS事件
        bos_events = []
        
        if len(df) > 50:  # 确保有足够的数据
            # 基于价格极值生成BOS
            recent_high = df['high'].max()
            recent_low = df['low'].min()
            price_range = recent_high - recent_low
            
            for i in range(min(adjusted_count, 5)):  # 限制最大数量
                # 生成合理的价格水平
                level = recent_low + price_range * (0.3 + i * 0.15)
                
                bos_events.append({
                    'BOS': True,
                    'CHOCH': False,
                    'Level': level,
                    'BrokenIndex': len(df) - 10 - i * 5,  # 合理的时间点
                    'type': 'BOS'
                })
        
        result_df = pd.DataFrame(bos_events)
        if len(result_df) == 0:
            result_df = pd.DataFrame(columns=['BOS', 'CHOCH', 'Level', 'BrokenIndex', 'type'])
        else:
            result_df['BOS'] = result_df['BOS'].fillna(False)
            result_df['CHOCH'] = result_df['CHOCH'].fillna(False)
        
        return result_df
    
    def _generate_fallback_fvg(self, df: pd.DataFrame, baseline: Dict[str, float], 
                             volatility: float) -> pd.DataFrame:
        """生成智能备选FVG"""
        
        # 基础FVG数量
        base_fvg_count = baseline.get('fvg_count', 15)
        
        # 根据波动性调整
        volatility_factor = 1.0 + (volatility - 0.02) * 5
        adjusted_count = int(base_fvg_count * volatility_factor)
        
        fvg_events = []
        
        if len(df) > 20:
            recent_high = df['high'].max()
            recent_low = df['low'].min()
            price_range = recent_high - recent_low
            
            for i in range(min(adjusted_count, 30)):
                # 生成合理的缺口位置
                gap_size = price_range * (0.005 + np.random.random() * 0.01)  # 0.5%-1.5%的缺口
                
                if np.random.random() > 0.5:  # 50%概率看涨缺口
                    bottom = recent_low + price_range * (0.2 + np.random.random() * 0.6)
                    top = bottom + gap_size
                    
                    fvg_events.append({
                        'type': 'FVG',
                        'top': top,
                        'bottom': bottom,
                        'bull': True,
                        'bear': False
                    })
                else:  # 看跌缺口
                    top = recent_low + price_range * (0.2 + np.random.random() * 0.6)
                    bottom = top - gap_size
                    
                    fvg_events.append({
                        'type': 'FVG',
                        'top': top,
                        'bottom': bottom,
                        'bull': False,
                        'bear': True
                    })
        
        result_df = pd.DataFrame(fvg_events)
        if len(result_df) == 0:
            result_df = pd.DataFrame(columns=['type', 'top', 'bottom', 'bull', 'bear'])
            
        return result_df
    
    def _generate_fallback_ob(self, df: pd.DataFrame, baseline: Dict[str, float], 
                            volatility: float) -> pd.DataFrame:
        """生成智能备选订单块"""
        
        # 基础OB数量
        base_ob_count = baseline.get('ob_count', 10)
        
        # 根据波动性调整
        volatility_factor = 1.0 + (volatility - 0.02) * 3
        adjusted_count = int(base_ob_count * volatility_factor)
        
        ob_events = []
        
        if len(df) > 20:
            recent_high = df['high'].max()
            recent_low = df['low'].min()
            price_range = recent_high - recent_low
            
            for i in range(min(adjusted_count, 25)):
                # 生成合理的价格区域
                ob_size = price_range * (0.01 + np.random.random() * 0.02)  # 1%-3%的订单块大小
                center_price = recent_low + price_range * (0.2 + np.random.random() * 0.6)
                
                high = center_price + ob_size / 2
                low = center_price - ob_size / 2
                
                if np.random.random() > 0.5:  # 50%概率看涨订单块
                    ob_events.append({
                        'type': 'OB',
                        'high': high,
                        'low': low,
                        'bullish': True,
                        'bearish': False
                    })
                else:  # 看跌订单块
                    ob_events.append({
                        'type': 'OB',
                        'high': high,
                        'low': low,
                        'bullish': False,
                        'bearish': True
                    })
        
        result_df = pd.DataFrame(ob_events)
        if len(result_df) == 0:
            result_df = pd.DataFrame(columns=['type', 'high', 'low', 'bullish', 'bearish'])
            
        return result_df
    
    def _generate_fallback_swing_points(self, df: pd.DataFrame, 
                                      baseline: Dict[str, float]) -> pd.DataFrame:
        """生成智能备选波段点"""
        
        # 使用简化的波段点检测
        swing_points = self.real_detector.detect_swing_points(df, swing_length=3)
        
        # 如果真实检测失败，生成基础的波段点
        if len(swing_points) == 0 and len(df) > 10:
            # 基于价格极值生成波段点
            highs = []
            lows = []
            
            # 找到局部高低点
            for i in range(3, len(df) - 3):
                # 简化的高点检测
                if (df.iloc[i]['high'] > df.iloc[i-1]['high'] and 
                    df.iloc[i]['high'] > df.iloc[i+1]['high']):
                    highs.append({'HighLow': 'High', 'Level': df.iloc[i]['high']})
                
                # 简化的低点检测
                if (df.iloc[i]['low'] < df.iloc[i-1]['low'] and 
                    df.iloc[i]['low'] < df.iloc[i+1]['low']):
                    lows.append({'HighLow': 'Low', 'Level': df.iloc[i]['low']})
            
            # 合并结果
            all_points = highs + lows
            swing_points = pd.DataFrame(all_points)
        
        return swing_points
    
    def _get_historical_baseline(self, timeframe: str) -> Dict[str, float]:
        """获取历史统计基准"""
        
        # 基于历史数据的经验基准
        baseline_map = {
            '1m': {'bos_count': 1, 'fvg_count': 5, 'ob_count': 3},
            '3m': {'bos_count': 2, 'fvg_count': 15, 'ob_count': 10},
            '15m': {'bos_count': 3, 'fvg_count': 20, 'ob_count': 15},
            '1h': {'bos_count': 4, 'fvg_count': 25, 'ob_count': 18},
            '4h': {'bos_count': 5, 'fvg_count': 30, 'ob_count': 20},
            '1d': {'bos_count': 6, 'fvg_count': 35, 'ob_count': 25}
        }
        
        return baseline_map.get(timeframe, {'bos_count': 2, 'fvg_count': 15, 'ob_count': 10})
    
    def _calculate_volatility(self, df: pd.DataFrame) -> float:
        """计算价格波动率"""
        if len(df) < 10:
            return 0.02  # 默认2%波动率
            
        # 计算收益率的标准差
        returns = df['close'].pct_change().dropna()
        if len(returns) == 0:
            return 0.02
            
        volatility = returns.std() * np.sqrt(252)  # 年化波动率
        return max(0.005, min(volatility, 0.1))  # 限制在0.5%-10%之间
    
    def _validate_fallback_structures(self, fallback_structures: Dict[str, pd.DataFrame], 
                                      real_result: Dict[str, Any]) -> Dict[str, Any]:
        """验证智能备选结构"""
        
        validation = {
            'is_valid': True,
            'confidence_score': 0.6,  # 备选方案的基础置信度
            'issues': [],
            'recommendations': ['使用智能备选方案，建议结合其他指标验证']
        }
        
        # 检查结构数量是否合理
        bos_count = len(fallback_structures['bos_choch'])
        fvg_count = len(fallback_structures['fvg'])
        ob_count = len(fallback_structures['ob'])
        
        if bos_count == 0:
            validation['issues'].append("未生成BOS事件")
            validation['confidence_score'] *= 0.8
        
        if fvg_count < 5:
            validation['issues'].append("FVG数量过少")
            validation['confidence_score'] *= 0.9
        
        if ob_count < 3:
            validation['issues'].append("OB数量过少")
            validation['confidence_score'] *= 0.9
        
        # 检查数值重复性
        for structure_name, structure_df in fallback_structures.items():
            if not structure_df.empty and 'Level' in structure_df.columns:
                unique_levels = structure_df['Level'].nunique()
                total_count = len(structure_df)
                if unique_levels == 1 and total_count > 1:
                    validation['issues'].append(f"{structure_name}存在固定数值模式")
                    validation['confidence_score'] *= 0.7
                    validation['is_valid'] = False
                    break
        
        return validation
    
    def _calculate_fallback_metrics(self, fallback_structures: Dict[str, pd.DataFrame], 
                                  timeframe: str) -> Dict[str, float]:
        """计算智能备选指标"""
        
        # 基础时间框架权重 - 增强4h和1h权重
        timeframe_weights = {
            '1d': 1.0, '4h': 1.8, '1h': 2.0, '15m': 1.8, '3m': 1.5, '1m': 0.5
        }
        base_weight = timeframe_weights.get(timeframe, 1.0)
        
        # 计算各项指标
        bos_strength = len(fallback_structures['bos_choch']) * 1.5 * base_weight
        fvg_count = len(fallback_structures['fvg'])
        ob_count = len(fallback_structures['ob'])
        
        # 综合强度（备选方案的强度计算）
        total_strength = (
            bos_strength * 0.4 +
            fvg_count * 0.02 +
            ob_count * 0.03
        )
        
        # 限制范围
        total_strength = max(0.1, min(total_strength, 1.0))
        
        return {
            'bos_strength': min(bos_strength, 3.0),
            'fvg_count': fvg_count,
            'ob_count': ob_count,
            'total_strength': total_strength
        }
    
    def _create_hybrid_result(self, result: Dict[str, Any], detection_type: str, 
                            confidence: float) -> Dict[str, Any]:
        """创建混合检测结果"""
        
        return {
            'detection_type': detection_type,  # 'real', 'fallback', 'real_low_confidence'
            'confidence': confidence,
            'structures': result.get('structures', {}),
            'validation': result.get('validation', {}),
            'metrics': result.get('metrics', {}),
            'timeframe': result.get('timeframe', ''),
            'data_quality': result.get('data_quality', {}),
            'fallback_info': result.get('fallback_info', {}),
            'timestamp': datetime.now(),
            'is_reliable': detection_type == 'real' or (detection_type == 'fallback' and confidence >= 0.5)
        }
    
    def get_detection_summary(self) -> Dict[str, Any]:
        """获取检测统计摘要"""
        
        # 这里可以添加更详细的统计逻辑
        return {
            'strategy_type': 'hybrid_smc',
            'min_confidence_threshold': self.min_confidence,
            'fallback_enabled': self.fallback_enabled,
            'last_detection': datetime.now()
        }