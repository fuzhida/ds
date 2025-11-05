#!/usr/bin/env python3
"""
SMC检测功能测试脚本
测试从TradingView转换的SMC检测功能
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入SMC检测模块
try:
    from smc_detection_tv import detect_smc_structures_tv, SMCDetector
    TV_SMC_AVAILABLE = True
    print("✓ TradingView SMC模块导入成功")
except ImportError as e:
    print(f"✗ TradingView SMC模块导入失败: {e}")
    TV_SMC_AVAILABLE = False

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_test_data(length=100, trend='random'):
    """生成测试数据"""
    np.random.seed(42)  # 确保可重复性
    
    if trend == 'up':
        # 上升趋势数据
        base = np.linspace(100, 120, length)
        noise = np.random.normal(0, 0.5, length)
        close = base + noise
        high = close + np.random.uniform(0, 1, length)
        low = close - np.random.uniform(0, 1, length)
        open_price = low + np.random.uniform(0, high - low)
    elif trend == 'down':
        # 下降趋势数据
        base = np.linspace(120, 100, length)
        noise = np.random.normal(0, 0.5, length)
        close = base + noise
        high = close + np.random.uniform(0, 1, length)
        low = close - np.random.uniform(0, 1, length)
        open_price = low + np.random.uniform(0, high - low)
    else:
        # 随机趋势数据
        close = np.random.normal(110, 5, length).cumsum()
        close = close - close.min() + 100  # 确保价格为正
        high = close + np.random.uniform(0, 2, length)
        low = close - np.random.uniform(0, 2, length)
        open_price = low + np.random.uniform(0, high - low)
    
    # 创建DataFrame
    dates = pd.date_range(start='2023-01-01', periods=length, freq='1h')
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': np.random.randint(1000, 10000, length)
    }, index=dates)
    
    return df

def test_smc_detection():
    """测试SMC检测功能"""
    if not TV_SMC_AVAILABLE:
        logger.error("TradingView SMC模块不可用，无法进行测试")
        return False
    
    # 测试不同趋势的数据
    test_cases = [
        ("上升趋势", "up"),
        ("下降趋势", "down"),
        ("随机趋势", "random")
    ]
    
    all_passed = True
    
    for name, trend in test_cases:
        logger.info(f"测试{name}数据...")
        df = generate_test_data(length=200, trend=trend)
        
        try:
            # 测试便捷函数
            result = detect_smc_structures_tv(
                df, 
                swing_length=5,
                structure_lookback=20,
                fvg_threshold=0.5,
                ob_threshold=0.3,
                liquidity_threshold=0.2
            )
            
            # 验证结果结构
            required_keys = ['market_structure', 'order_blocks', 'fair_value_gaps', 
                           'liquidity_levels', 'trend_direction', 'swing_points']
            
            missing_keys = [key for key in required_keys if key not in result]
            if missing_keys:
                logger.error(f"缺少必要的结果键: {missing_keys}")
                all_passed = False
                continue
            
            # 验证数据类型
            if not isinstance(result['market_structure'], list):
                logger.error("market_structure不是列表类型")
                all_passed = False
                continue
                
            if not isinstance(result['order_blocks'], list):
                logger.error("order_blocks不是列表类型")
                all_passed = False
                continue
                
            if not isinstance(result['fair_value_gaps'], list):
                logger.error("fair_value_gaps不是列表类型")
                all_passed = False
                continue
            
            # 输出检测结果摘要
            logger.info(f"市场结构: {result['trend_direction']}")
            logger.info(f"摆动点数量: {len(result['swing_points'])}")
            logger.info(f"订单块数量: {len(result['order_blocks'])}")
            logger.info(f"公允价值缺口数量: {len(result['fair_value_gaps'])}")
            logger.info(f"流动性区域数量: {len(result['liquidity_levels'])}")
            
            logger.info(f"✓ {name}数据测试通过")
            
        except Exception as e:
            logger.error(f"✗ {name}数据测试失败: {e}")
            all_passed = False
    
    return all_passed

def test_smc_detector_class():
    """测试SMCDetector类"""
    if not TV_SMC_AVAILABLE:
        logger.error("TradingView SMC模块不可用，无法进行测试")
        return False
    
    logger.info("测试SMCDetector类...")
    
    try:
        # 创建检测器实例
        detector = SMCDetector(
            swing_length=5,
            structure_lookback=20,
            fvg_threshold=0.5,
            ob_threshold=0.3,
            liquidity_threshold=0.2
        )
        
        # 生成测试数据
        df = generate_test_data(length=200, trend='random')
        
        # 检测市场结构
        result = detector.analyze(df)
        
        # 验证结果
        if not isinstance(result, dict):
            logger.error("SMCDetector.detect_market_structure返回的不是字典")
            return False
        
        logger.info("✓ SMCDetector类测试通过")
        return True
        
    except Exception as e:
        logger.error(f"✗ SMCDetector类测试失败: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("开始SMC检测功能测试")
    
    # 测试便捷函数
    test1_passed = test_smc_detection()
    
    # 测试SMCDetector类
    test2_passed = test_smc_detector_class()
    
    # 总结测试结果
    if test1_passed and test2_passed:
        logger.info("✓ 所有测试通过")
        return 0
    else:
        logger.error("✗ 部分测试失败")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)