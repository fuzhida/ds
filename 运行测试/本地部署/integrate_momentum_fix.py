#!/usr/bin/env python3
"""
集成动量过滤器修复到交易机器人
"""

import sys
import os
import logging
from typing import Dict, Any

# 添加当前目录到Python路径
sys.path.append('/Users/zhidafu/ds交易/ds/运行测试/本地部署')

# 导入修复模块
try:
    from momentum_filter_fix import EnhancedMomentumFilter, quick_fix_momentum_filter
    MOMENTUM_FIX_AVAILABLE = True
    print("✅ 动量过滤器修复模块加载成功")
except ImportError as e:
    print(f"❌ 动量过滤器修复模块加载失败: {e}")
    MOMENTUM_FIX_AVAILABLE = False

# 导入交易机器人
try:
    from btc_trading_bot import TradingBot as BTCTradingBot
    print("✅ 交易机器人模块加载成功")
except ImportError as e:
    print(f"❌ 交易机器人模块加载失败: {e}")
    BTCTRADINGBOT_AVAILABLE = False

def patch_trading_bot():
    """修补交易机器人的动量过滤器"""
    if not MOMENTUM_FIX_AVAILABLE:
        print("❌ 动量过滤器修复模块不可用")
        return False
    
    try:
        # 创建增强动量过滤器实例
        enhanced_filter = EnhancedMomentumFilter(None, logging.getLogger(__name__))
        
        # 保存原始方法
        if hasattr(BTCTradingBot, 'intraday_momentum_filter'):
            original_method = BTCTradingBot.intraday_momentum_filter
            print("✅ 已保存原始动量过滤器方法")
        else:
            print("❌ 未找到原始动量过滤器方法")
            return False
        
        # 定义新的动量过滤器方法
        def enhanced_intraday_momentum_filter(self, price_data: Dict[str, Any]) -> bool:
            """增强的动量过滤器，集成修复方案"""
            try:
                self.logger_system.info("使用增强动量过滤器")
                
                # 使用快速修复函数
                result = quick_fix_momentum_filter(self, price_data)
                
                # 记录结果
                if result:
                    self.logger_system.info("✅ 增强动量过滤器通过")
                else:
                    self.logger_system.info("❌ 增强动量过滤器失败")
                
                return result
                
            except Exception as e:
                self.logger_system.error(f"增强动量过滤器异常: {e}，回退到原始方法")
                # 回退到原始方法
                return original_method(self, price_data)
        
        # 替换方法
        BTCTradingBot.intraday_momentum_filter = enhanced_intraday_momentum_filter
        print("✅ 动量过滤器方法已成功替换为增强版本")
        
        return True
        
    except Exception as e:
        print(f"❌ 修补交易机器人失败: {e}")
        return False

def test_patched_bot():
    """测试修补后的交易机器人"""
    print("\n" + "="*50)
    print("测试修补后的交易机器人")
    print("="*50)
    
    try:
        # 导入配置类
        from btc_trading_bot import Config
        
        # 创建配置实例
        config = Config()
        
        # 创建交易机器人实例
        bot = BTCTradingBot(config)
        bot.logger_system = logging.getLogger(__name__)
        
        # 创建测试数据
        import pandas as pd
        import numpy as np
        
        test_data = {
            'price': 50000,
            'technical_data': {
                'rsi': 55,
                'sma_20': 49500,
                'ema_12': 49800,
                'atr': 200
            },
            'multi_tf_data': {
                '15m': pd.DataFrame({
                    'close': np.random.randn(100).cumsum() + 50000,
                    'volume': np.random.randn(100) * 1000 + 10000,
                    'ema_12': np.random.randn(100) * 100 + 49800,
                    'volume_ratio': np.random.randn(100) * 0.1 + 1.0
                })
            },
            'smc_structures': {
                '15m': {
                    'fvg_count': 0,  # 模拟无FVG数据
                    'ob_count': 0,  # 模拟无OB数据
                    'strength_score': 0.2
                }
            }
        }
        
        print("测试增强动量过滤器...")
        result = bot.intraday_momentum_filter(test_data)
        print(f"测试结果: {'通过' if result else '失败'}")
        
        return result
        
    except Exception as e:
        print(f"测试修补后的交易机器人失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 开始集成动量过滤器修复")
    print("="*60)
    
    # 1. 修补交易机器人
    print("步骤1: 修补交易机器人动量过滤器")
    patch_success = patch_trading_bot()
    
    if not patch_success:
        print("❌ 交易机器人修补失败，退出")
        return False
    
    # 2. 测试修补效果
    print("\n步骤2: 测试修补后的交易机器人")
    test_success = test_patched_bot()
    
    if test_success:
        print("\n✅ 动量过滤器修复集成成功！")
        print("交易机器人现在使用增强的动量过滤器")
        print("修复了FVG/OB数量不足导致的过滤异常问题")
    else:
        print("\n❌ 测试失败，但修补已完成")
    
    return test_success

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    success = main()
    
    if success:
        print("\n🎉 动量过滤器修复集成完成！")
        print("可以使用增强的交易机器人了")
    else:
        print("\n⚠️ 集成过程中遇到问题，但修复模块已准备就绪")