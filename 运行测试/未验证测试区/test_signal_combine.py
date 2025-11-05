#!/usr/bin/env python3
"""
测试信号组合过程
"""

import os
import logging
from dotenv import load_dotenv
from ai_signal_generator import AISignalGenerator
from config import Config
from datetime import datetime

def _combine_signals(ai_signal, consensus_signal, consensus_confidence, technical_signal, smc_signal, ob_overlay_result=None):
    """组合信号（从backtest_engine.py复制）"""
    try:
        # 获取各信号
        ai_signal_type = ai_signal.get("signal", "HOLD")
        ai_confidence = ai_signal.get("confidence", 0.1)
        
        tech_signal_type = technical_signal.get("signal", "HOLD")
        tech_confidence = technical_signal.get("confidence", 0.1)
        
        smc_signal_type = smc_signal.get("signal", "HOLD")
        smc_confidence = smc_signal.get("confidence", 0.1)
        
        # 信号权重
        ai_weight = 0.5
        consensus_weight = 0.2
        tech_weight = 0.2
        smc_weight = 0.1
        
        # 计算加权信号强度
        buy_strength = 0
        sell_strength = 0
        
        # AI信号
        if ai_signal_type == "BUY":
            buy_strength += ai_confidence * ai_weight
        elif ai_signal_type == "SELL":
            sell_strength += ai_confidence * ai_weight
        
        # 共识信号
        if consensus_signal == "BUY":
            buy_strength += consensus_confidence * consensus_weight
        elif consensus_signal == "SELL":
            sell_strength += consensus_confidence * consensus_weight
        
        # 技术信号
        if tech_signal_type == "BUY":
            buy_strength += tech_confidence * tech_weight
        elif tech_signal_type == "SELL":
            sell_strength += tech_confidence * tech_weight
        
        # SMC信号
        if smc_signal_type == "BUY":
            buy_strength += smc_confidence * smc_weight
        elif smc_signal_type == "SELL":
            sell_strength += smc_confidence * smc_weight
        
        # OB叠加置信度提升
        overlay_boost = 0.0
        overlay_info = ""
        if ob_overlay_result and ob_overlay_result.get('has_overlay', False):
            overlay_boost = ob_overlay_result.get('overlay_confidence_boost', 0.0)
            overlay_info = f", OB叠加置信度提升: {overlay_boost:.2f}"
            
            # 如果有OB叠加，且信号类型与叠加OB类型一致，增加对应信号强度
            narrow_ob = ob_overlay_result.get('narrow_ob_for_entry')
            if narrow_ob:
                ob_type = narrow_ob.get('type', '')
                if 'bullish' in ob_type and ai_signal_type == "BUY":
                    buy_strength += overlay_boost
                elif 'bearish' in ob_type and ai_signal_type == "SELL":
                    sell_strength += overlay_boost
        
        # 确定最终信号
        if buy_strength > sell_strength and buy_strength > 0.3:  # 降低阈值从0.5到0.3
            final_signal = "BUY"
            final_confidence = buy_strength
        elif sell_strength > buy_strength and sell_strength > 0.3:  # 降低阈值从0.5到0.3
            final_signal = "SELL"
            final_confidence = sell_strength
        else:
            final_signal = "HOLD"
            final_confidence = 0.5
        
        return {
            "signal": final_signal,
            "confidence": final_confidence,
            "reasoning": f"AI:{ai_signal_type}({ai_confidence:.2f}), 共识:{consensus_signal}({consensus_confidence:.2f}), 技术:{tech_signal_type}({tech_confidence:.2f}), SMC:{smc_signal_type}({smc_confidence:.2f}){overlay_info}",
            "ob_overlay_result": ob_overlay_result
        }
        
    except Exception as e:
        print(f"信号组合失败: {e}")
        return {"signal": "HOLD", "confidence": 0.1, "reasoning": f"信号组合失败: {str(e)}"}

def main():
    """主函数"""
    try:
        # 加载环境变量
        load_dotenv('1.env')
        
        # 创建配置
        config = Config()
        
        # 创建日志器
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger("Signal_Combine_Test")
        
        # 创建AI信号生成器
        ai_generator = AISignalGenerator(config, logger)
        
        # 创建测试市场数据
        market_data = {
            'symbol': 'BTC/USDT',
            'current_price': 50000.0,
            'technical_indicators': {
                'rsi': 55.0,
                'macd': {'macd': 100.0, 'signal': 90.0, 'histogram': 10.0},
                'bb': {'upper': 51000.0, 'middle': 50000.0, 'lower': 49000.0},
                'ema': {'ema_12': 49500.0, 'ema_26': 49000.0, 'ema_50': 48500.0},
                'overall_score': 0.6
            },
            'smc_structures': {
                'bos_choch': [{'type': 'BOS', 'direction': 'bullish', 'price': 49500.0, 'time': '2023-01-01T00:00:00Z'}],
                'order_blocks': [{'type': 'bullish', 'price': 49200.0, 'top': 49300.0, 'bottom': 49100.0}],
                'fair_value_gaps': [{'type': 'bullish', 'top': 49600.0, 'bottom': 49400.0}],
                'swing_points': [{'type': 'high', 'price': 50500.0, 'time': '2023-01-01T00:00:00Z'}],
                'overall_score': 0.6
            },
            'key_levels': {
                'support': [49000.0, 48000.0],
                'resistance': [51000.0, 52000.0],
                'ema': [49500.0, 49000.0, 48500.0]
            },
            'price_action': {
                'volatility': 0.02,
                'trend': 'bullish',
                'momentum': 0.6
            }
        }
        
        # 生成AI信号
        print('生成AI信号...')
        ai_signals = ai_generator.generate_signals(market_data)
        
        # 获取主要AI信号
        primary_ai_signal = ai_signals.get("primary", {})
        ai_signal = primary_ai_signal.get("signal", {})
        
        # 获取共识信号
        consensus_signal = ai_signals.get("consensus", "HOLD")
        consensus_confidence = ai_signals.get("consensus_confidence", 0.1)
        
        print(f'AI信号: {ai_signal}')
        print(f'共识信号: {consensus_signal} (置信度: {consensus_confidence:.2f})')
        
        # 创建技术信号
        technical_score = market_data['technical_indicators'].get('overall_score', 0.5)
        if technical_score > 0.6:
            tech_signal = "BUY"
        elif technical_score < 0.4:
            tech_signal = "SELL"
        else:
            tech_signal = "HOLD"
        
        technical_signal = {
            "signal": tech_signal,
            "confidence": abs(technical_score - 0.5) * 2,
            "score": technical_score
        }
        
        print(f'技术信号: {technical_signal}')
        
        # 创建SMC信号
        smc_score = market_data['smc_structures'].get('overall_score', 0.5)
        if smc_score > 0.6:
            smc_signal = "BUY"
        elif smc_score < 0.4:
            smc_signal = "SELL"
        else:
            smc_signal = "HOLD"
        
        smc_signal = {
            "signal": smc_signal,
            "confidence": abs(smc_score - 0.5) * 2,
            "score": smc_score
        }
        
        print(f'SMC信号: {smc_signal}')
        
        # 组合信号
        final_signal = _combine_signals(
            ai_signal, consensus_signal, consensus_confidence,
            technical_signal, smc_signal, None
        )
        
        print(f'最终信号: {final_signal}')
        
        # 检查是否满足交易条件
        signal_type = final_signal.get("signal", "HOLD")
        confidence = final_signal.get("confidence", 0.0)
        
        print(f'交易条件检查:')
        print(f'信号类型: {signal_type}')
        print(f'信号置信度: {confidence:.2f}')
        print(f'是否满足交易条件 (信号 != HOLD 且 置信度 > 0.6): {signal_type != "HOLD" and confidence > 0.6}')
        
    except Exception as e:
        print(f'测试失败: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()