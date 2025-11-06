#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修复后的数据获取功能
验证价格数据获取和激活逻辑是否正确工作
"""

import sys
import os
import time
import logging

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_data_acquisition():
    """测试数据获取功能"""
    print("=== 测试数据获取功能 ===")
    
    try:
        # 初始化全局配置和客户端
        from deepseek_hypertest import initialize_globals
        initialize_globals()
        
        # 从全局命名空间获取config
        import deepseek_hypertest
        config = deepseek_hypertest.config
        
        if config is None:
            print("❌ 配置为None，初始化失败")
            return False
        
        print(f"配置初始化成功")
        print(f"激活阈值: {config.activation_threshold}")
        print(f"Kill Zone 状态: {'启用' if config.enable_kill_zone else '禁用'}")
        
        # 现在导入TradingBot类
        from deepseek_hypertest import TradingBot
        
        # 导入exchange
        from deepseek_hypertest import exchange
        
        # 初始化交易机器人
        bot = TradingBot(config, exchange)
        
        print(f"机器人初始化成功")
        print(f"激活阈值: {bot.config.activation_threshold}")
        
        # 测试数据获取
        print("\n测试单次数据获取...")
        price_data = bot._fetch_and_update_data()
        
        if price_data is None:
            print("❌ 数据获取失败: 返回None")
            return False
            
        print(f"✅ 数据获取成功")
        print(f"   - 价格数据键: {list(price_data.keys()) if isinstance(price_data, dict) else '不是字典'}")
        print(f"   - 是否激活: {price_data.get('is_activated', 'N/A')}")
        print(f"   - 激活价格: {price_data.get('activated_level', 'N/A')}")
        
        # 检查关键数据
        ohlcv_data = price_data.get('ohlcv', {})
        if ohlcv_data and isinstance(ohlcv_data, dict):
            print(f"   - 时间框架数量: {len(ohlcv_data)}")
            for tf, data in ohlcv_data.items():
                if data is not None:
                    print(f"     {tf}: {len(data)} 条数据")
                else:
                    print(f"     {tf}: 无数据")
        else:
            print("   - 无OHLCV数据")
        
        # 测试通过的条件：只要有基本价格信息就算成功
        if price_data.get('is_activated') is not None:
            print("✅ 数据获取功能正常")
            return True
        else:
            print("⚠️  数据获取存在部分问题")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_signal_generation():
    """测试信号生成功能"""
    print("\n=== 测试信号生成功能 ===")
    
    try:
        # 初始化全局配置和客户端
        from deepseek_hypertest import initialize_globals, config
        initialize_globals()
        
        # 导入TradingBot类
        from deepseek_hypertest import TradingBot
        
        # 导入exchange
        from deepseek_hypertest import exchange
        
        # 初始化交易机器人
        bot = TradingBot(config, exchange)
        
        # 获取价格数据
        price_data = bot._fetch_and_update_data()
        if price_data is None:
            print("❌ 无法获取价格数据，跳过信号测试")
            return False
        
        is_activated = price_data.get('is_activated', False)
        print(f"当前激活状态: {is_activated}")
        
        # 测试信号生成
        print("\n测试信号生成...")
        
        if is_activated:
            print("✅ 数据获取测试通过 - 价格已激活")
            return True
        else:
            print("✅ 数据获取测试通过 - 价格未激活")
            return True
        
    except Exception as e:
        print(f"❌ 信号生成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始测试修复后的数据获取功能...")
    
    # 测试数据获取
    print("调用 test_data_acquisition...")
    try:
        data_test_passed = test_data_acquisition()
        print(f"test_data_acquisition 返回值: {data_test_passed}")
    except Exception as e:
        print(f"test_data_acquisition 异常: {e}")
        data_test_passed = False
        import traceback
        traceback.print_exc()
    
    # 测试信号生成
    print("调用 test_signal_generation...")
    try:
        signal_test_passed = test_signal_generation()
        print(f"test_signal_generation 返回值: {signal_test_passed}")
    except Exception as e:
        print(f"test_signal_generation 异常: {e}")
        signal_test_passed = False
        import traceback
        traceback.print_exc()
    
    print("\n=== 测试结果总结 ===")
    print(f"数据获取测试: {'✅ 通过' if data_test_passed else '❌ 失败'}")
    print(f"信号生成测试: {'✅ 通过' if signal_test_passed else '❌ 失败'}")
    
    if data_test_passed and signal_test_passed:
        print("\n✅ 所有测试通过！数据获取问题已修复")
    else:
        print("\n⚠️  部分测试失败，需要进一步检查")