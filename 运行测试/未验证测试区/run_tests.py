#!/usr/bin/env python3
"""
测试运行脚本
运行所有测试并生成报告
"""

import unittest
import sys
import os
import time
from io import StringIO

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_all_tests():
    """运行所有测试"""
    # 发现并加载所有测试
    loader = unittest.TestLoader()
    start_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests")
    suite = loader.discover(start_dir, pattern="test_*.py")
    
    # 创建测试运行器
    stream = StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2)
    
    # 运行测试
    print("开始运行所有测试...")
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    # 获取测试输出
    output = stream.getvalue()
    
    # 打印测试结果
    print("\n" + "="*80)
    print("测试结果摘要")
    print("="*80)
    print(f"运行时间: {end_time - start_time:.2f} 秒")
    print(f"测试总数: {result.testsRun}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print(f"跳过: {len(result.skipped)}")
    
    # 打印失败的测试
    if result.failures:
        print("\n失败的测试:")
        print("-"*80)
        for test, traceback in result.failures:
            print(f"测试: {test}")
            print(f"原因: {traceback}")
            print("-"*80)
    
    # 打印错误的测试
    if result.errors:
        print("\n错误的测试:")
        print("-"*80)
        for test, traceback in result.errors:
            print(f"测试: {test}")
            print(f"错误: {traceback}")
            print("-"*80)
    
    # 打印跳过的测试
    if result.skipped:
        print("\n跳过的测试:")
        print("-"*80)
        for test, reason in result.skipped:
            print(f"测试: {test}")
            print(f"原因: {reason}")
            print("-"*80)
    
    # 打印详细输出
    print("\n详细测试输出:")
    print("="*80)
    print(output)
    
    # 返回测试是否成功
    return len(result.failures) == 0 and len(result.errors) == 0

def run_specific_test(test_module):
    """运行特定测试模块"""
    try:
        # 导入测试模块
        module = __import__(f"tests.{test_module}", fromlist=[""])
        
        # 创建测试套件
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(module)
        
        # 创建测试运行器
        stream = StringIO()
        runner = unittest.TextTestRunner(stream=stream, verbosity=2)
        
        # 运行测试
        print(f"开始运行测试模块: {test_module}")
        start_time = time.time()
        result = runner.run(suite)
        end_time = time.time()
        
        # 获取测试输出
        output = stream.getvalue()
        
        # 打印测试结果
        print("\n" + "="*80)
        print(f"测试结果摘要 - {test_module}")
        print("="*80)
        print(f"运行时间: {end_time - start_time:.2f} 秒")
        print(f"测试总数: {result.testsRun}")
        print(f"失败: {len(result.failures)}")
        print(f"错误: {len(result.errors)}")
        print(f"跳过: {len(result.skipped)}")
        
        # 打印失败的测试
        if result.failures:
            print("\n失败的测试:")
            print("-"*80)
            for test, traceback in result.failures:
                print(f"测试: {test}")
                print(f"原因: {traceback}")
                print("-"*80)
        
        # 打印错误的测试
        if result.errors:
            print("\n错误的测试:")
            print("-"*80)
            for test, traceback in result.errors:
                print(f"测试: {test}")
                print(f"错误: {traceback}")
                print("-"*80)
        
        # 打印跳过的测试
        if result.skipped:
            print("\n跳过的测试:")
            print("-"*80)
            for test, reason in result.skipped:
                print(f"测试: {test}")
                print(f"原因: {reason}")
                print("-"*80)
        
        # 打印详细输出
        print("\n详细测试输出:")
        print("="*80)
        print(output)
        
        # 返回测试是否成功
        return len(result.failures) == 0 and len(result.errors) == 0
    
    except ImportError as e:
        print(f"错误: 无法导入测试模块 {test_module}: {e}")
        return False

def main():
    """主函数"""
    if len(sys.argv) > 1:
        # 运行特定测试模块
        test_module = sys.argv[1]
        success = run_specific_test(test_module)
    else:
        # 运行所有测试
        success = run_all_tests()
    
    # 根据测试结果设置退出代码
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()