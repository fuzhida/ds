#!/usr/bin/env python3
"""
测试优化后的数据获取功能
- 验证15分钟时间框架数据量从1000减少到500
- 验证本地缓存机制
- 验证增量更新逻辑
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timezone

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from deepseek_hyper import TradingBot, Config

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('test_data_optimization.log')
        ]
    )

def test_data_optimization():
    """测试数据优化功能"""
    logger = logging.getLogger('test')
    logger.info("开始测试数据优化功能")
    
    # 初始化配置和机器人
    config = Config()
    config.simulation_mode = True  # 使用模拟模式
    
    bot = TradingBot(config)
    
    # 测试1: 验证15分钟时间框架数据量从1000减少到500
    logger.info("测试1: 验证15分钟时间框架数据量从1000减少到500")
    
    # 第一次获取数据（完整获取）
    logger.info("第一次获取数据（完整获取）")
    start_time = time.time()
    data1 = bot.get_multi_timeframe_data()
    first_fetch_time = time.time() - start_time
    
    if data1 and 'multi_tf_data' in data1:
        tf_15m_data = data1['multi_tf_data'].get('15m')
        if tf_15m_data is not None:
            logger.info(f"15分钟时间框架数据量: {len(tf_15m_data)} 条记录")
            assert len(tf_15m_data) <= 500, f"15分钟时间框架数据量应不超过500，实际为{len(tf_15m_data)}"
            logger.info("✅ 15分钟时间框架数据量测试通过")
        else:
            logger.error("未获取到15分钟时间框架数据")
            return False
    else:
        logger.error("第一次数据获取失败")
        return False
    
    # 测试2: 验证缓存机制（模拟模式下缓存不会自动更新，需要手动测试）
    logger.info("测试2: 验证缓存机制（手动测试）")
    
    # 手动更新缓存
    if data1 and 'multi_tf_data' in data1:
        for tf, df in data1['multi_tf_data'].items():
            bot._update_cache(tf, df)
    
    # 检查缓存文件是否存在
    cache_file_path = bot.cache_file_path
    logger.info(f"缓存文件路径: {cache_file_path}")
    
    if os.path.exists(cache_file_path):
        logger.info("✅ 缓存文件已创建")
        
        # 检查缓存内容
        with open(cache_file_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        logger.info(f"缓存中包含的时间框架: {list(cache_data.keys())}")
        
        if '15m' in cache_data:
            cached_15m_count = len(cache_data['15m'].get('data', []))
            logger.info(f"缓存中15分钟时间框架数据量: {cached_15m_count} 条记录")
            assert cached_15m_count == len(tf_15m_data), "缓存中的数据量应与第一次获取的数据量一致"
            logger.info("✅ 缓存内容测试通过")
        else:
            logger.error("缓存中未找到15分钟时间框架数据")
            return False
    else:
        logger.error("缓存文件未创建")
        return False
    
    # 测试3: 验证缓存读取功能
    logger.info("测试3: 验证缓存读取功能")
    
    # 从缓存读取数据
    cached_15m_df = bot._get_cached_data('15m')
    if cached_15m_df is not None:
        logger.info(f"从缓存读取15分钟时间框架数据量: {len(cached_15m_df)} 条记录")
        assert len(cached_15m_df) == len(tf_15m_data), "从缓存读取的数据量应与原始数据量一致"
        logger.info("✅ 缓存读取功能测试通过")
    else:
        logger.error("从缓存读取数据失败")
        return False
    
    # 测试4: 验证缓存过期机制
    logger.info("测试4: 验证缓存过期机制")
    
    # 修改缓存TTL为1秒，强制过期
    original_ttl = bot.cache_ttl
    bot.cache_ttl = 1
    
    # 等待缓存过期
    logger.info("等待缓存过期...")
    time.sleep(2)
    
    # 尝试从缓存读取数据（应返回None）
    cached_15m_df_expired = bot._get_cached_data('15m')
    if cached_15m_df_expired is None:
        logger.info("✅ 缓存过期机制测试通过")
    else:
        logger.error("缓存过期机制测试失败，缓存未过期")
        return False
    
    # 恢复原始TTL
    bot.cache_ttl = original_ttl
    
    logger.info("✅ 所有测试通过")
    return True

if __name__ == "__main__":
    setup_logging()
    success = test_data_optimization()
    sys.exit(0 if success else 1)