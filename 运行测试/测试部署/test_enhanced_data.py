#!/usr/bin/env python3
"""
测试增强版数据结构 - 验证DeepSeek建议的改进是否有效
"""

import json
import logging
from datetime import datetime
from enhanced_mock_bot import EnhancedMockBot

def test_enhanced_data_structure():
    """测试增强版数据结构"""
    
    print("🔍 开始测试增强版数据结构...")
    
    # 创建增强版MockBot实例
    enhanced_bot = EnhancedMockBot()
    
    # 生成示例数据
    import random
    import numpy as np
    
    # 生成OHLC数据
    ohlc_data = []
    base_price = 42000
    for i in range(50):
        day = (i // 24) + 1
        hour = i % 24
        timestamp = f"2024-01-{day:02d}T{hour:02d}:00:00Z"
        open_price = base_price + random.uniform(-100, 100)
        close_price = open_price + random.uniform(-50, 50)
        high_price = max(open_price, close_price) + random.uniform(0, 50)
        low_price = min(open_price, close_price) - random.uniform(0, 50)
        volume = random.uniform(800, 1500)
        
        ohlc_data.append({
            "timestamp": timestamp,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume,
            "timeframe": "1h"
        })
        
        base_price = close_price
    
    # 生成市场深度数据
    market_depth = []
    for i in range(10):
        mid_price = 42000 + i * 10
        market_depth.append({
            "timestamp": f"2024-01-01T{(i%24):02d}:00:00Z",
            "bid_price": mid_price - 5,
            "ask_price": mid_price + 5,
            "bid_volume": random.uniform(100, 500),
            "ask_volume": random.uniform(100, 500)
        })
    
    # 生成时间与销售数据
    time_sales = []
    for i in range(100):
        time_sales.append({
            "timestamp": f"2024-01-01T{(i%24):02d}:{(i%60):02d}:{(i%60):02d}",
            "price": 42000 + random.uniform(-100, 100),
            "volume": random.uniform(0.1, 20),
            "side": random.choice(["buy", "sell"]),
            "aggressive": random.choice([True, False])
        })
    
    # 生成市场情绪数据
    market_sentiment = {
        "fear_greed_index": random.uniform(0, 100),
        "funding_rate": random.uniform(-0.01, 0.01),
        "open_interest_change": random.uniform(-5, 5),
        "long_short_ratio": random.uniform(0.8, 1.5)
    }
    
    # 测试1: 增强版数据提取
    print("\n📊 测试1: 增强版数据提取")
    enhanced_raw_data = enhanced_bot.extract_enhanced_raw_data(
        ohlc_data=ohlc_data,
        volume_data=[],
        market_depth=market_depth,
        time_sales=time_sales,
        market_sentiment=market_sentiment
    )
    
    # 验证增强版数据结构
    required_fields = [
        'enhanced_candlesticks', 'swing_points', 'volume_analysis',
        'market_depth', 'time_sales', 'market_sentiment',
        'multi_timeframe_context', 'liquidity_levels',
        'price_movements', 'order_flow_imbalance', 'market_microstructure'
    ]
    
    missing_fields = [field for field in required_fields if field not in enhanced_raw_data]
    if missing_fields:
        print(f"❌ 缺失字段: {missing_fields}")
        return False
    else:
        print("✅ 所有必需字段都存在")
    
    # 验证增强版K线数据
    enhanced_candlesticks = enhanced_raw_data['enhanced_candlesticks']
    if not enhanced_candlesticks:
        print("❌ 增强版K线数据为空")
        return False
    
    # 检查增强版K线字段
    enhanced_candle_fields = [
        'body_size', 'upper_wick', 'lower_wick', 'body_position',
        'body_ratio', 'gap_size', 'gap_direction', 'volume_profile',
        'engulfing', 'rejection', 'inside_bar'
    ]
    
    missing_candle_fields = [field for field in enhanced_candle_fields if field not in enhanced_candlesticks[0]]
    if missing_candle_fields:
        print(f"❌ 增强版K线缺失字段: {missing_candle_fields}")
        return False
    else:
        print("✅ 增强版K线字段完整")
    
    # 验证市场深度数据
    market_depth_data = enhanced_raw_data['market_depth']
    if not market_depth_data:
        print("⚠️ 市场深度数据为空（可选）")
    else:
        depth_fields = ['imbalance_ratio', 'spread_percentage', 'dominant_side', 'liquidity_score']
        missing_depth_fields = [field for field in depth_fields if field not in market_depth_data[0]]
        if missing_depth_fields:
            print(f"❌ 市场深度缺失字段: {missing_depth_fields}")
            return False
        else:
            print("✅ 市场深度字段完整")
    
    # 验证时间与销售数据
    time_sales_data = enhanced_raw_data['time_sales']
    if not time_sales_data:
        print("⚠️ 时间与销售数据为空（可选）")
    else:
        sales_fields = ['side', 'liquidity_removed', 'aggressive', 'large_order']
        missing_sales_fields = [field for field in sales_fields if field not in time_sales_data[0]]
        if missing_sales_fields:
            print(f"❌ 时间与销售缺失字段: {missing_sales_fields}")
            return False
        else:
            print("✅ 时间与销售字段完整")
    
    # 测试2: 增强版提示词生成
    print("\n📝 测试2: 增强版提示词生成")
    try:
        enhanced_prompt = enhanced_bot.generate_enhanced_smc_prompt(enhanced_raw_data)
        if len(enhanced_prompt) > 1000:  # 提示词应该足够长
            print(f"✅ 增强版提示词生成成功，长度: {len(enhanced_prompt)} 字符")
        else:
            print(f"❌ 增强版提示词过短: {len(enhanced_prompt)} 字符")
            return False
    except Exception as e:
        print(f"❌ 增强版提示词生成失败: {str(e)}")
        return False
    
    # 测试3: 数据质量评估
    print("\n🔬 测试3: 数据质量评估")
    
    # 检查数据完整性
    data_completeness = {
        'enhanced_candlesticks': len(enhanced_candlesticks) > 0,
        'swing_points': len(enhanced_raw_data['swing_points']) > 0,
        'volume_analysis': len(enhanced_raw_data['volume_analysis']) > 0,
        'market_depth': len(enhanced_raw_data['market_depth']) > 0,
        'time_sales': len(enhanced_raw_data['time_sales']) > 0,
        'market_sentiment': bool(enhanced_raw_data['market_sentiment']),
        'multi_timeframe_context': bool(enhanced_raw_data['multi_timeframe_context']),
        'liquidity_levels': len(enhanced_raw_data['liquidity_levels']) > 0,
        'price_movements': len(enhanced_raw_data['price_movements']) > 0,
        'order_flow_imbalance': bool(enhanced_raw_data['order_flow_imbalance']),
        'market_microstructure': bool(enhanced_raw_data['market_microstructure'])
    }
    
    completeness_score = sum(data_completeness.values()) / len(data_completeness)
    print(f"📈 数据完整性得分: {completeness_score:.2f} (1.0为满分)")
    
    if completeness_score >= 0.8:
        print("✅ 数据完整性良好")
    elif completeness_score >= 0.6:
        print("⚠️ 数据完整性一般")
    else:
        print("❌ 数据完整性不足")
        return False
    
    # 检查数据一致性
    consistency_issues = []
    
    # 检查K线数据时间顺序
    timestamps = [c['timestamp'] for c in enhanced_candlesticks]
    if timestamps != sorted(timestamps):
        consistency_issues.append("K线时间戳未按顺序排列")
    
    # 检查价格逻辑
    for i, candle in enumerate(enhanced_candlesticks):
        if not (candle['low'] <= candle['open'] <= candle['high'] and 
                candle['low'] <= candle['close'] <= candle['high']):
            consistency_issues.append(f"K线{i}价格逻辑错误")
    
    if consistency_issues:
        print(f"❌ 数据一致性问题: {consistency_issues}")
        return False
    else:
        print("✅ 数据一致性良好")
    
    # 测试4: SMC结构计算能力评估
    print("\n🧮 测试4: SMC结构计算能力评估")
    
    # 检查是否有足够的数据计算SMC结构
    smc_calculation_feasibility = {
        'BOS/CHOCH': len(enhanced_candlesticks) >= 10 and len(enhanced_raw_data['swing_points']) >= 2,
        'Order Blocks': len(enhanced_candlesticks) >= 20 and any(c['engulfing'] != 'none' for c in enhanced_candlesticks),
        'FVG': len(enhanced_candlesticks) >= 10 and any(c['gap_size'] > 0 for c in enhanced_candlesticks),
        'Liquidity Analysis': len(enhanced_raw_data['liquidity_levels']) >= 3,
        'Market Microstructure': bool(enhanced_raw_data['market_microstructure'])
    }
    
    print("SMC结构计算可行性评估:")
    for structure, feasible in smc_calculation_feasibility.items():
        status = "✅ 可行" if feasible else "❌ 不可行"
        print(f"  {structure}: {status}")
    
    feasibility_score = sum(smc_calculation_feasibility.values()) / len(smc_calculation_feasibility)
    print(f"📈 SMC结构计算可行性得分: {feasibility_score:.2f} (1.0为满分)")
    
    if feasibility_score >= 0.8:
        print("✅ SMC结构计算能力良好")
    elif feasibility_score >= 0.6:
        print("⚠️ SMC结构计算能力一般")
    else:
        print("❌ SMC结构计算能力不足")
        return False
    
    # 保存测试结果
    test_results = {
        "timestamp": datetime.now().isoformat(),
        "data_completeness": data_completeness,
        "completeness_score": completeness_score,
        "consistency_issues": consistency_issues,
        "smc_calculation_feasibility": smc_calculation_feasibility,
        "feasibility_score": feasibility_score,
        "overall_test_result": completeness_score >= 0.6 and feasibility_score >= 0.6 and not consistency_issues
    }
    
    with open("enhanced_data_test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)
    
    # 保存增强版数据示例
    with open("enhanced_data_example.json", "w") as f:
        # 转换numpy类型为Python原生类型
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            elif hasattr(obj, 'dtype'):  # numpy类型检查
                if obj.dtype == 'bool':
                    return bool(obj)
                elif 'int' in str(obj.dtype):
                    return int(obj)
                elif 'float' in str(obj.dtype):
                    return float(obj)
            return obj
        
        json.dump(convert_numpy_types(enhanced_raw_data), f, indent=2)
    
    # 保存增强版提示词
    with open("enhanced_prompt_example.txt", "w") as f:
        f.write(enhanced_prompt)
    
    print("\n🎉 增强版数据结构测试完成!")
    print(f"📊 测试结果已保存到 enhanced_data_test_results.json")
    print(f"📈 数据示例已保存到 enhanced_data_example.json")
    print(f"📝 提示词示例已保存到 enhanced_prompt_example.txt")
    
    return True

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    success = test_enhanced_data_structure()
    
    if success:
        print("\n✅ 所有测试通过! 增强版数据结构符合DeepSeek建议的改进要求。")
    else:
        print("\n❌ 测试失败! 增强版数据结构需要进一步改进。")