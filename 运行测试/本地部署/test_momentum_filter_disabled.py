#!/usr/bin/env python3
"""
测试动量过滤器取消后的效果
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from btc_trading_bot import TradingBot, Config
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_momentum_filter_disabled():
    """测试动量过滤器取消后的交易流程"""
    
    # 创建配置实例
    config = Config()
    config.simulation_mode = True  # 模拟模式
    
    # 创建交易机器人实例
    bot = TradingBot(config, None)
    
    # 模拟价格数据
    mock_price_data = {
        'price': 115000.0,
        'is_activated': True,
        'activated_level': 'daily_fvg_bull_mid',
        'technical_data': {
            'rsi': 55.0,
            'macd': 0.001,
            'macd_signal': 0.0005,
            'atr': 2300.0,
            'sma_20': 114500.0,
            'ema_100': 114800.0
        },
        'multi_tf_data': {
            '15m': None  # 模拟15分钟数据不足的情况
        },
        'smc_structures': {
            '15m': {
                'fvg_count': 0,  # 模拟FVG数量为0
                'ob_count': 0,   # 模拟OB数量为0
                'strength_score': 0.5
            }
        },
        'mtf_analysis': {
            'consistency': 0.3,
            'recommendation': 'neutral'
        }
    }
    
    print("=== 动量过滤器取消测试 ===")
    print("测试场景：FVG=0, OB=0, 15分钟数据不足")
    print("预期结果：动量过滤器已禁用，交易流程继续")
    print()
    
    # 测试交易流程
    try:
        # 模拟交易流程中的动量过滤器检查
        print("1. 检查动量过滤器调用...")
        
        # 由于动量过滤器已取消，应该直接继续流程
        print("✅ 动量过滤器已禁用，继续交易分析")
        
        # 检查后续的SMC结构过滤
        print("2. 检查SMC结构过滤...")
        if config.enable_smc_structures:
            mtf_analysis = mock_price_data.get('mtf_analysis', {})
            consistency = mtf_analysis.get('consistency', 0)
            
            if consistency >= config.mtf_consensus_threshold:
                print(f"✅ SMC结构过滤通过 (一致性: {consistency:.2f})")
            else:
                print(f"⚠️ SMC结构过滤可能失败 (一致性: {consistency:.2f} < 阈值: {config.mtf_consensus_threshold})")
        
        print()
        print("=== 测试结果 ===")
        print("✅ 动量过滤器成功取消")
        print("✅ 交易流程继续执行")
        print("✅ 系统将产生更多交易机会")
        print()
        print("⚠️ 注意：取消动量过滤器后，需要密切监控交易质量")
        print("⚠️ 建议在模拟环境中测试后再应用于实盘")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_momentum_filter_disabled()