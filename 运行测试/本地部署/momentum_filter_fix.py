#!/usr/bin/env python3
"""
动量过滤器修复补丁
解决FVG/OB数量不足导致的过滤异常问题
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

class EnhancedMomentumFilter:
    """增强的动量过滤器，修复FVG/OB检测异常问题"""
    
    def __init__(self, config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        # 修复：调整阈值以适应不同的市场条件
        self.min_fvg_count = 0  # 允许0个FVG
        self.min_ob_count = 0   # 允许0个OB
        self.min_total_structures = 0  # 总结构数量要求降低到0
        
        # 备选指标权重
        self.alternative_weights = {
            'volume_ratio': 0.3,
            'price_ema_ratio': 0.3,
            'rsi_momentum': 0.2,
            'volatility_adjustment': 0.2
        }
    
    def enhanced_intraday_momentum_filter(self, price_data: Dict[str, Any]) -> bool:
        """增强的动量过滤器，包含容错机制"""
        try:
            # 获取15分钟数据用于动量分析
            m15_df = price_data.get('multi_tf_data', {}).get('15m')
            current_price = price_data.get('price', 0)
            
            if m15_df is None or len(m15_df) < 10:  # 降低数据要求
                self.logger.info("动量过滤器：15分钟数据不足，使用备选分析")
                return self._fallback_momentum_analysis(price_data)
            
            # 计算备选动量指标
            momentum_score = self._calculate_alternative_momentum(m15_df, current_price, price_data)
            
            # 记录详细分析
            self.logger.info(f"增强动量分析 - 综合评分: {momentum_score:.2f}")
            
            # 动态阈值调整
            min_score = self._get_dynamic_threshold(m15_df, price_data)
            
            if momentum_score >= min_score:
                self.logger.info(f"✅ 增强动量过滤器通过 (评分: {momentum_score:.2f} >= 阈值: {min_score:.2f})")
                return True
            else:
                self.logger.info(f"❌ 增强动量过滤器失败 (评分: {momentum_score:.2f} < 阈值: {min_score:.2f})")
                return False
                
        except Exception as e:
            self.logger.warning(f"增强动量过滤器异常：{e}，回退到基础分析")
            return self._fallback_momentum_analysis(price_data)
    
    def _calculate_alternative_momentum(self, m15_df: pd.DataFrame, current_price: float, price_data: Dict[str, Any]) -> float:
        """计算备选动量指标"""
        scores = []
        
        # 1. 成交量动量 (30%权重)
        if 'volume' in m15_df.columns and len(m15_df) >= 20:
            vol_ma = m15_df['volume'].rolling(20).mean().iloc[-1]
            current_vol = m15_df['volume'].iloc[-1]
            vol_ratio = current_vol / vol_ma if vol_ma > 0 else 1.0
            vol_score = min(vol_ratio / 2.0, 1.0)  # 归一化到0-1
            scores.append(vol_score * self.alternative_weights['volume_ratio'])
        
        # 2. 价格-EMA动量 (30%权重)
        if 'ema_12' in m15_df.columns:
            ema12 = m15_df['ema_12'].iloc[-1]
            if current_price > ema12:
                price_ema_score = min((current_price - ema12) / (ema12 * 0.02), 1.0)
            else:
                price_ema_score = 0.0
            scores.append(price_ema_score * self.alternative_weights['price_ema_ratio'])
        
        # 3. RSI动量 (20%权重)
        technical_data = price_data.get('technical_data', {})
        rsi = technical_data.get('rsi', 50)
        # 将RSI转换为动量分数 (30-70范围映射到0-1)
        if 30 <= rsi <= 70:
            rsi_momentum = 1.0 - abs(rsi - 50) / 20.0  # 接近50时分数最高
        else:
            rsi_momentum = 0.3  # RSI超范围时给予基础分数
        scores.append(rsi_momentum * self.alternative_weights['rsi_momentum'])
        
        # 4. 波动性调整 (20%权重)
        if len(m15_df) >= 20:
            returns = m15_df['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(96)  # 年化波动率
            # 适度的波动性有利于动量 (10%-30%年化波动率最优)
            if 0.1 <= volatility <= 0.3:
                vol_adjustment = 1.0
            elif volatility < 0.1:
                vol_adjustment = volatility / 0.1  # 低波动率线性衰减
            else:
                vol_adjustment = max(0.0, 1.0 - (volatility - 0.3) / 0.2)  # 高波动率线性衰减
            scores.append(vol_adjustment * self.alternative_weights['volatility_adjustment'])
        
        return sum(scores) if scores else 0.5  # 默认中等分数
    
    def _get_dynamic_threshold(self, m15_df: pd.DataFrame, price_data: Dict[str, Any]) -> float:
        """获取动态阈值"""
        # 基础阈值
        base_threshold = 0.6
        
        # 根据市场条件调整
        adjustments = 0.0
        
        # 数据质量调整
        structures = price_data.get('smc_structures', {})
        tf_structures = structures.get('15m', {})
        fvg_count = tf_structures.get('fvg_count', 0)
        ob_count = tf_structures.get('ob_count', 0)
        
        # 如果SMC结构数据质量差，降低要求
        if fvg_count == 0 and ob_count == 0:
            adjustments -= 0.2  # 降低20%要求
            self.logger.info("检测到SMC结构数据缺失，降低动量阈值")
        
        # 时间调整（交易时段）
        import datetime
        current_hour = datetime.datetime.now().hour
        if 9 <= current_hour <= 16:  # 主要交易时段
            adjustments += 0.1
        elif 2 <= current_hour <= 6:  # 低流动性时段
            adjustments -= 0.15
        
        # 波动性调整
        if len(m15_df) >= 20:
            returns = m15_df['close'].pct_change().dropna()
            volatility = returns.std()
            if volatility < 0.001:  # 极低波动
                adjustments -= 0.1
            elif volatility > 0.01:  # 高波动
                adjustments += 0.05
        
        final_threshold = max(0.3, min(0.8, base_threshold + adjustments))
        self.logger.info(f"动态阈值计算: 基础={base_threshold}, 调整={adjustments:.2f}, 最终={final_threshold:.2f}")
        
        return final_threshold
    
    def _fallback_momentum_analysis(self, price_data: Dict[str, Any]) -> bool:
        """回退动量分析"""
        self.logger.info("使用回退动量分析")
        
        # 基础技术指标检查
        technical_data = price_data.get('technical_data', {})
        
        # RSI检查
        rsi = technical_data.get('rsi', 50)
        if not (25 < rsi < 75):  # 放宽RSI范围
            self.logger.info(f"回退分析：RSI超出范围 ({rsi})")
            return False
        
        # 价格趋势检查
        current_price = price_data.get('price', 0)
        sma_20 = technical_data.get('sma_20', current_price)
        
        if current_price < sma_20 * 0.99:  # 允许1%的偏差
            self.logger.info(f"回退分析：价格低于SMA20 ({current_price:.2f} < {sma_20:.2f})")
            return False
        
        self.logger.info("✅ 回退动量分析通过")
        return True
    
    def validate_smc_structures(self, structures: Dict[str, Any], tf: str = '15m') -> Dict[str, Any]:
        """验证SMC结构数据质量"""
        tf_structures = structures.get(tf, {})
        
        validation = {
            'is_valid': True,
            'fvg_count': tf_structures.get('fvg_count', 0),
            'ob_count': tf_structures.get('ob_count', 0),
            'total_structures': 0,
            'data_quality': 'good',
            'recommendations': []
        }
        
        total_structures = validation['fvg_count'] + validation['ob_count']
        validation['total_structures'] = total_structures
        
        # 数据质量评估
        if total_structures == 0:
            validation['data_quality'] = 'poor'
            validation['recommendations'].append('SMC结构数据缺失，建议使用增强动量过滤器')
        elif total_structures < 3:
            validation['data_quality'] = 'fair'
            validation['recommendations'].append('SMC结构数据较少，建议降低过滤要求')
        else:
            validation['data_quality'] = 'good'
        
        # 检查是否有异常模式
        if validation['fvg_count'] > 50:  # FVG数量异常多
            validation['recommendations'].append('FVG数量异常，可能存在数据质量问题')
            validation['is_valid'] = False
        
        return validation

# 快速修复函数
def quick_fix_momentum_filter(bot_instance, price_data: Dict[str, Any]) -> bool:
    """快速修复动量过滤器的函数"""
    logger = logging.getLogger(__name__)
    
    try:
        # 创建增强过滤器实例
        enhanced_filter = EnhancedMomentumFilter(bot_instance.config, logger)
        
        # 验证SMC结构数据
        structures = price_data.get('smc_structures', {})
        validation = enhanced_filter.validate_smc_structures(structures, '15m')
        
        logger.info(f"SMC结构验证结果: {validation}")
        
        # 使用增强过滤器
        result = enhanced_filter.enhanced_intraday_momentum_filter(price_data)
        
        logger.info(f"增强动量过滤器结果: {'通过' if result else '失败'}")
        return result
        
    except Exception as e:
        logger.error(f"增强动量过滤器异常: {e}")
        # 最终回退到基础RSI检查
        technical_data = price_data.get('technical_data', {})
        rsi = technical_data.get('rsi', 50)
        result = 30 < rsi < 70
        logger.info(f"最终回退到RSI检查: {result} (RSI: {rsi})")
        return result

# 测试函数
def test_enhanced_momentum_filter():
    """测试增强动量过滤器"""
    import pandas as pd
    import numpy as np
    
    # 创建测试数据
    test_data = {
        'price': 50000,
        'technical_data': {
            'rsi': 55,
            'sma_20': 49500,
            'ema_12': 49800
        },
        'multi_tf_data': {
            '15m': pd.DataFrame({
                'close': np.random.randn(100).cumsum() + 50000,
                'volume': np.random.randn(100) * 1000 + 10000,
                'ema_12': np.random.randn(100) * 100 + 49800
            })
        },
        'smc_structures': {
            '15m': {
                'fvg_count': 0,  # 模拟无FVG数据
                'ob_count': 0    # 模拟无OB数据
            }
        }
    }
    
    # 模拟配置
    class MockConfig:
        volume_confirmation_threshold = 0.8
        mtf_consensus_threshold = 0.6
        enable_smc_structures = True
        min_structure_score = 0.3
    
    config = MockConfig()
    logger = logging.getLogger(__name__)
    
    # 测试快速修复函数
    result = quick_fix_momentum_filter(type('MockBot', (), {'config': config})(), test_data)
    
    print(f"测试结果: {'通过' if result else '失败'}")
    return result

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("测试增强动量过滤器...")
    test_result = test_enhanced_momentum_filter()
    
    if test_result:
        print("✅ 增强动量过滤器测试通过")
    else:
        print("❌ 增强动量过滤器测试失败")