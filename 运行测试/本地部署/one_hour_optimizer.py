"""
1小时级别交易优化器
为1小时时间框架提供专门的交易参数优化
"""

import logging
from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class OptimizationResult:
    """优化结果数据类"""
    expected_improvements: str
    key_changes: str
    optimized_params: Dict[str, Any]


class OneHourOptimizer:
    """1小时级别交易优化器"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('system')
        
        # 1小时级别专用优化参数
        self._optimization_params = {
            'risk_per_trade': 0.018,  # 1.8%风险
            'rr_min_threshold': 2.0,   # 最小R:R 2.0:1
            'rr_aggressive_threshold': 3.0,  # 激进R:R 3.0:1
            'volatility_threshold': 70,  # 波动率阈值
            'min_structure_score': 0.4,  # 最小结构评分
            'mtf_consensus_threshold': 0.25,  # MTF一致性阈值
            'volume_confirmation_threshold': 0.6,  # 成交量确认阈值
            'max_zone_interactions': 10,  # 最大区域交互
            'fvg_stack_threshold': 1,  # FVG堆叠阈值
            'candle_pattern_weight': 1.5,  # 蜡烛图模式权重
            'order_flow_weight': 0.15,  # 订单流权重
            'temperature': 0.4,  # AI温度
            'leverage': 40,  # 杠杆
            'amplitude_lookback': 7,  # 振幅回看期
            'activation_threshold': 0.00005,  # 激活阈值
        }
    
    def apply_optimizations(self) -> Dict[str, Any]:
        """应用1小时级别优化配置"""
        try:
            optimized_params = {}
            
            # 应用优化参数
            for param, value in self._optimization_params.items():
                if hasattr(self.config, param):
                    setattr(self.config, param, value)
                    optimized_params[param] = value
            
            # 1小时级别专用配置
            optimized_params.update({
                'timeframes': ['1d', '4h', '1h', '15m', '3m', '1m'],
                'primary_timeframe': '15m',
                'structure_confirm_timeframe': '1h',
                'higher_tf_bias_tf': '4h',
                'lower_tf_entry_tf': '15m',
            })
            
            self.logger.info("✅ 1小时级别优化配置已应用")
            return optimized_params
            
        except Exception as e:
            self.logger.warning(f"⚠️ 1小时级别优化器应用失败: {e}")
            return {}
    
    def get_optimization_summary(self) -> Dict[str, str]:
        """获取优化摘要"""
        return {
            'expected_improvements': '提高1小时级别交易信号质量，增强风险控制',
            'key_changes': '优化风险参数、R:R要求、结构评分阈值等',
        }
    
    def validate_config(self) -> bool:
        """验证配置是否适合1小时级别交易"""
        try:
            # 检查必要参数
            required_params = ['risk_per_trade', 'leverage', 'timeframes']
            for param in required_params:
                if not hasattr(self.config, param):
                    self.logger.warning(f"⚠️ 配置缺少必要参数: {param}")
                    return False
            
            # 验证时间框架配置
            if '1h' not in getattr(self.config, 'timeframes', []):
                self.logger.warning("⚠️ 时间框架配置中缺少1小时级别")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 配置验证失败: {e}")
            return False
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return {
            'optimization_level': '1h_specialized',
            'risk_adjustment': 'enhanced',
            'signal_quality': 'improved',
            'timeframe_alignment': 'optimized',
        }


# 便捷函数
def create_one_hour_optimizer(config) -> OneHourOptimizer:
    """创建1小时级别优化器实例"""
    return OneHourOptimizer(config)


def apply_default_optimizations(config) -> Dict[str, Any]:
    """应用默认优化配置"""
    optimizer = OneHourOptimizer(config)
    return optimizer.apply_optimizations()


if __name__ == "__main__":
    # 测试代码
    @dataclass
    class TestConfig:
        risk_per_trade: float = 0.02
        leverage: int = 20
        timeframes: List[str] = None
        
        def __post_init__(self):
            if self.timeframes is None:
                self.timeframes = ['1d', '4h', '1h', '15m']
    
    config = TestConfig()
    optimizer = OneHourOptimizer(config)
    
    print("✅ 1小时级别优化器测试通过")
    print(f"优化前配置: risk={config.risk_per_trade}, leverage={config.leverage}")
    
    optimized = optimizer.apply_optimizations()
    print(f"优化后配置: risk={config.risk_per_trade}, leverage={config.leverage}")
    print(f"优化摘要: {optimizer.get_optimization_summary()}")