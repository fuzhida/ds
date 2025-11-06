#!/usr/bin/env python3
"""
SMC模块兼容性修复方案
解决交易机器人对稀疏数据的误解问题
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union
import logging

class SMCCompatibilityFix:
    """SMC模块兼容性修复类"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def validate_smc_data(self, data: pd.DataFrame, data_type: str) -> Dict[str, Any]:
        """
        验证SMC数据的有效性
        
        Args:
            data: SMC模块返回的数据
            data_type: 数据类型 ('highs_lows', 'bos_choch', 'fvg', 'ob', 'liquidity')
            
        Returns:
            验证结果字典
        """
        if data is None or not isinstance(data, pd.DataFrame):
            return {
                'is_valid': False,
                'error': '数据格式错误',
                'valid_rows': 0,
                'total_rows': 0,
                'sparsity_ratio': 1.0,
                'recommendation': '使用备选计算方案'
            }
        
        if data.empty:
            return {
                'is_valid': False,
                'error': '数据为空',
                'valid_rows': 0,
                'total_rows': 0,
                'sparsity_ratio': 1.0,
                'recommendation': '使用备选计算方案'
            }
        
        total_rows = len(data)
        
        # 根据不同数据类型检查有效数据
        if data_type == 'highs_lows':
            if 'HighLow' not in data.columns or 'Level' not in data.columns:
                return {
                    'is_valid': False,
                    'error': '缺少必要列',
                    'valid_rows': 0,
                    'total_rows': total_rows,
                    'sparsity_ratio': 1.0,
                    'recommendation': '使用备选计算方案'
                }
            valid_rows = data['HighLow'].notna().sum()
            
        elif data_type == 'bos_choch':
            expected_columns = ['BOS', 'CHOCH', 'Level', 'BrokenIndex']
            missing_columns = [col for col in expected_columns if col not in data.columns]
            if missing_columns:
                return {
                    'is_valid': False,
                    'error': f'缺少列: {missing_columns}',
                    'valid_rows': 0,
                    'total_rows': total_rows,
                    'sparsity_ratio': 1.0,
                    'recommendation': '使用备选计算方案'
                }
            # BOS或CHOCH任一列有有效数据即可
            valid_rows = (data['BOS'].notna() | data['CHOCH'].notna()).sum()
            
        elif data_type == 'fvg':
            if 'FVG' not in data.columns:
                return {
                    'is_valid': False,
                    'error': '缺少FVG列',
                    'valid_rows': 0,
                    'total_rows': total_rows,
                    'sparsity_ratio': 1.0,
                    'recommendation': '使用备选计算方案'
                }
            valid_rows = data['FVG'].notna().sum()
            
        elif data_type == 'ob':
            if 'OB' not in data.columns:
                return {
                    'is_valid': False,
                    'error': '缺少OB列',
                    'valid_rows': 0,
                    'total_rows': total_rows,
                    'sparsity_ratio': 1.0,
                    'recommendation': '使用备选计算方案'
                }
            valid_rows = data['OB'].notna().sum()
            
        elif data_type == 'liquidity':
            if 'Liquidity' not in data.columns:
                return {
                    'is_valid': False,
                    'error': '缺少Liquidity列',
                    'valid_rows': 0,
                    'total_rows': total_rows,
                    'sparsity_ratio': 1.0,
                    'recommendation': '使用备选计算方案'
                }
            valid_rows = data['Liquidity'].notna().sum()
            
        else:
            # 通用验证
            valid_rows = data.notna().any(axis=1).sum()
        
        sparsity_ratio = 1 - (valid_rows / total_rows) if total_rows > 0 else 1.0
        
        # 判断数据是否可用
        # SMC数据的稀疏性是正常现象，只要有一定比例的有效数据就可以使用
        min_valid_rows = max(1, total_rows * 0.01)  # 至少1%的有效数据
        
        is_valid = valid_rows >= min_valid_rows
        
        return {
            'is_valid': is_valid,
            'valid_rows': valid_rows,
            'total_rows': total_rows,
            'sparsity_ratio': sparsity_ratio,
            'recommendation': '数据可用' if is_valid else '使用备选计算方案',
            'data_quality': self._assess_data_quality(sparsity_ratio, valid_rows)
        }
    
    def _assess_data_quality(self, sparsity_ratio: float, valid_rows: int) -> str:
        """评估数据质量"""
        if sparsity_ratio < 0.5 and valid_rows >= 10:
            return '优秀'
        elif sparsity_ratio < 0.8 and valid_rows >= 5:
            return '良好'
        elif valid_rows >= 1:
            return '可用'
        else:
            return '不足'
    
    def process_smc_highs_lows(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理swing highs/lows数据"""
        validation = self.validate_smc_data(data, 'highs_lows')
        
        if not validation['is_valid']:
            self.logger.warning(f"Highs/Lows数据验证失败: {validation['error']}")
            # 返回空但格式正确的DataFrame
            return pd.DataFrame(columns=['HighLow', 'Level'])
        
        # 数据有效，返回原始数据
        return data.copy()
    
    def process_smc_bos_choch(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理BOS/CHOCH数据"""
        validation = self.validate_smc_data(data, 'bos_choch')
        
        if not validation['is_valid']:
            self.logger.warning(f"BOS/CHOCH数据验证失败: {validation['error']}")
            # 返回空但格式正确的DataFrame
            return pd.DataFrame(columns=['BOS', 'CHOCH', 'Level', 'BrokenIndex'])
        
        # 数据有效，返回原始数据
        return data.copy()
    
    def process_smc_fvg(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理FVG数据"""
        validation = self.validate_smc_data(data, 'fvg')
        
        if not validation['is_valid']:
            self.logger.warning(f"FVG数据验证失败: {validation['error']}")
            # 返回空但格式正确的DataFrame
            return pd.DataFrame(columns=['FVG', 'Top', 'Bottom', 'MitigatedIndex'])
        
        # 数据有效，返回原始数据
        return data.copy()
    
    def process_smc_ob(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理Order Blocks数据"""
        validation = self.validate_smc_data(data, 'ob')
        
        if not validation['is_valid']:
            self.logger.warning(f"OB数据验证失败: {validation['error']}")
            # 返回空但格式正确的DataFrame
            return pd.DataFrame(columns=['OB', 'Top', 'Bottom', 'MitigatedIndex'])
        
        # 数据有效，返回原始数据
        return data.copy()
    
    def calculate_sparsity_aware_metrics(self, data: pd.DataFrame, data_type: str, 
                                       df: pd.DataFrame, atr: float) -> Dict[str, float]:
        """
        计算考虑稀疏性的指标
        
        Args:
            data: SMC处理后的数据
            data_type: 数据类型
            df: 原始价格数据
            atr: 平均真实波幅
            
        Returns:
            指标字典
        """
        if data is None or data.empty:
            return self._get_default_metrics(data_type)
        
        metrics = {}
        
        if data_type == 'bos_choch':
            # BOS强度计算
            valid_events = data.dropna()
            if len(valid_events) > 0:
                # 计算最近的事件强度
                recent_events = valid_events.tail(5)  # 最近5个事件
                avg_strength = 0.5  # 默认强度
                
                # 如果有level数据，计算价格距离
                if 'Level' in recent_events.columns:
                    current_price = df['close'].iloc[-1]
                    levels = recent_events['Level'].dropna()
                    if len(levels) > 0:
                        last_level = levels.iloc[-1]
                        price_distance = abs(current_price - last_level) / atr if atr > 0 else 0.5
                        # 距离越近，强度越高
                        avg_strength = max(0.1, min(2.0, 2.0 / (1.0 + price_distance)))
                
                metrics['bos_strength'] = avg_strength
                metrics['bos_count'] = len(valid_events)
            else:
                metrics['bos_strength'] = 0.3
                metrics['bos_count'] = 0
                
        elif data_type == 'fvg':
            # FVG指标计算
            valid_fvgs = data.dropna()
            if len(valid_fvgs) > 0:
                metrics['fvg_count'] = len(valid_fvgs)
                # 计算平均强度（如果有强度数据）
                if 'strength' in valid_fvgs.columns:
                    avg_strength = valid_fvgs['strength'].mean()
                else:
                    avg_strength = 0.5
                metrics['avg_fvg_strength'] = avg_strength
            else:
                metrics['fvg_count'] = 0
                metrics['avg_fvg_strength'] = 0.0
                
        elif data_type == 'ob':
            # Order Blocks指标计算
            valid_obs = data.dropna()
            if len(valid_obs) > 0:
                metrics['ob_count'] = len(valid_obs)
                # 计算平均有效性评分
                if 'validity_score' in valid_obs.columns:
                    avg_validity = valid_obs['validity_score'].mean()
                else:
                    avg_validity = 2.0
                metrics['avg_ob_validity'] = avg_validity
            else:
                metrics['ob_count'] = 0
                metrics['avg_ob_validity'] = 0.0
                
        elif data_type == 'highs_lows':
            # Swing points指标计算
            valid_points = data.dropna()
            if len(valid_points) > 0:
                metrics['swing_count'] = len(valid_points)
                # 计算高低点分布
                if 'HighLow' in valid_points.columns:
                    high_count = (valid_points['HighLow'] == 1).sum()
                    low_count = (valid_points['HighLow'] == -1).sum()
                    metrics['swing_high_ratio'] = high_count / len(valid_points) if len(valid_points) > 0 else 0.5
                else:
                    metrics['swing_high_ratio'] = 0.5
            else:
                metrics['swing_count'] = 0
                metrics['swing_high_ratio'] = 0.5
        
        return metrics
    
    def _get_default_metrics(self, data_type: str) -> Dict[str, float]:
        """获取默认指标值"""
        defaults = {
            'bos_choch': {'bos_strength': 0.3, 'bos_count': 0},
            'fvg': {'fvg_count': 0, 'avg_fvg_strength': 0.0},
            'ob': {'ob_count': 0, 'avg_ob_validity': 0.0},
            'highs_lows': {'swing_count': 0, 'swing_high_ratio': 0.5}
        }
        return defaults.get(data_type, {})
    
    def create_fix_report(self) -> str:
        """创建修复报告"""
        return """
SMC模块兼容性修复报告
========================

问题诊断:
1. SMC模块本身功能正常，所有函数调用成功
2. 返回的数据格式正确（DataFrame）
3. 大部分值为NaN是市场结构的正常特征（稀疏信号）
4. 交易机器人错误地将稀疏性解释为模块故障

修复方案:
1. 实现数据验证机制，正确识别有效数据
2. 实现稀疏性感知的指标计算
3. 提供降级处理机制，当数据不足时使用合理默认值
4. 保持向后兼容性，不影响现有功能

使用说明:
1. 在交易机器人中导入此类
2. 用此类的方法替换原有的SMC数据处理逻辑
3. 系统将自动处理稀疏数据并提供合理指标
"""

# 测试函数
def test_smc_fix():
    """测试SMC修复方案"""
    print("测试SMC模块兼容性修复...")
    
    fix = SMCCompatibilityFix()
    
    # 创建模拟的稀疏SMC数据
    test_data = pd.DataFrame({
        'HighLow': [1.0, np.nan, np.nan, -1.0, np.nan, np.nan, 1.0],
        'Level': [42000, np.nan, np.nan, 41800, np.nan, np.nan, 42200]
    })
    
    # 测试验证
    result = fix.validate_smc_data(test_data, 'highs_lows')
    print(f"验证结果: {result}")
    
    # 测试处理
    processed = fix.process_smc_highs_lows(test_data)
    print(f"处理后数据形状: {processed.shape}")
    
    print("测试完成！")

if __name__ == "__main__":
    test_smc_fix()