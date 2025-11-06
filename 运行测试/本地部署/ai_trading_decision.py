"""
AI简化交易决策模块
将复杂的计算逻辑转化为AI提示词处理
"""

import json
from typing import Dict, Any, Optional

class AITradingDecision:
    """AI交易决策简化器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def extract_simplified_market_info(self, price_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        提取简化的市场信息，避免复杂计算
        """
        # 基础价格信息
        current_price = price_data.get('current_price', 0)
        
        # 简化结构信息提取
        smc_structures = price_data.get('smc_structures', {})
        structure_counts = self._extract_structure_counts(smc_structures)
        
        # 简化趋势判断
        trend_analysis = self._simplify_trend_analysis(price_data)
        
        # 风险等级评估
        risk_level = self._assess_risk_level(price_data)
        
        return {
            'current_price': current_price,
            'timeframes': ['1d', '4h', '1h', '15m', '3m'],
            'structure_counts': structure_counts,
            'trend_analysis': trend_analysis,
            'risk_level': risk_level,
            'market_condition': self._assess_market_condition(price_data)
        }
    
    def _extract_structure_counts(self, smc_structures: Dict[str, Any]) -> Dict[str, list]:
        """简化结构数量提取"""
        fvg_counts = []
        ob_counts = []
        
        for tf in ['1d', '4h', '1h', '15m', '3m']:
            if tf in smc_structures:
                structures = smc_structures[tf]
                fvg_counts.append(len(structures.get('fvg_events', [])))
                ob_counts.append(len(structures.get('ob_events', [])))
            else:
                fvg_counts.append(0)
                ob_counts.append(0)
                
        return {'fvg': fvg_counts, 'ob': ob_counts}
    
    def _simplify_trend_analysis(self, price_data: Dict[str, Any]) -> str:
        """简化趋势分析"""
        # 基于简单价格变化判断趋势
        price_changes = price_data.get('price_changes', {})
        
        if not price_changes:
            return 'neutral'
            
        # 简单趋势判断逻辑
        bullish_signals = 0
        bearish_signals = 0
        
        for tf, change in price_changes.items():
            if change > 0.01:  # 1%上涨
                bullish_signals += 1
            elif change < -0.01:  # 1%下跌
                bearish_signals += 1
                
        if bullish_signals > bearish_signals + 2:
            return 'bullish'
        elif bearish_signals > bullish_signals + 2:
            return 'bearish'
        else:
            return 'neutral'
    
    def _assess_risk_level(self, price_data: Dict[str, Any]) -> str:
        """简化风险评估"""
        volatility = price_data.get('volatility', 0)
        
        if volatility > 0.05:  # 5%波动率
            return 'high'
        elif volatility > 0.02:  # 2%波动率
            return 'medium'
        else:
            return 'low'
    
    def _assess_market_condition(self, price_data: Dict[str, Any]) -> str:
        """简化市场状态评估"""
        structure_counts = self._extract_structure_counts(price_data.get('smc_structures', {}))
        total_fvg = sum(structure_counts['fvg'])
        total_ob = sum(structure_counts['ob'])
        
        if total_fvg == 0 and total_ob > 20:
            return 'consolidation'  # 盘整状态
        elif total_fvg > 5:
            return 'volatile'  # 波动状态
        else:
            return 'normal'
    
    def generate_trading_prompt(self, market_info: Dict[str, Any]) -> str:
        """生成AI交易决策提示词"""
        
        prompt = f"""
        你是一个专业的AI交易员，专门从事BTC/USD的SMC/ICT策略分析。
        
        **当前市场状态分析：**
        - 当前价格：${market_info['current_price']:,.2f}
        - 时间框架覆盖：{', '.join(market_info['timeframes'])}
        - 市场结构：FVG总数={sum(market_info['structure_counts']['fvg'])}, OB总数={sum(market_info['structure_counts']['ob'])}
        - 趋势状态：{market_info['trend_analysis']}
        - 风险等级：{market_info['risk_level']}
        - 市场条件：{market_info['market_condition']}
        
        **交易策略要求：**
        - 主要策略：SMC/ICT机构订单流分析
        - 风险控制：最大回撤2%，严格止损
        - 信号优先级：结构强度 > 多时间框架一致性 > 流动性水平
        
        **请基于以上信息提供专业交易决策：**
        1. 交易方向建议（做多/做空/观望）
        2. 关键决策理由（基于SMC结构分析）
        3. 建议入场价格范围
        4. 止损位置建议
        5. 目标价位区间
        6. 仓位管理建议
        7. 风险提示
        
        请以JSON格式返回分析结果。
        """
        
        return prompt
    
    def parse_ai_response(self, response: str) -> Dict[str, Any]:
        """解析AI响应为结构化数据"""
        try:
            # 尝试解析JSON响应
            return json.loads(response)
        except:
            # 如果非JSON格式，提取关键信息
            return {
                'decision': self._extract_decision_from_text(response),
                'reasoning': response[:500],  # 限制长度
                'confidence': 0.7  # 默认置信度
            }
    
    def _extract_decision_from_text(self, text: str) -> str:
        """从文本中提取交易决策"""
        text_lower = text.lower()
        
        if '做多' in text_lower or 'long' in text_lower or 'buy' in text_lower:
            return 'long'
        elif '做空' in text_lower or 'short' in text_lower or 'sell' in text_lower:
            return 'short'
        else:
            return 'wait'

def create_ai_decision_maker(config: Dict[str, Any]) -> AITradingDecision:
    """创建AI决策器实例"""
    return AITradingDecision(config)

# 使用示例
if __name__ == "__main__":
    # 模拟配置
    config = {
        'symbol': 'BTC/USD',
        'risk_tolerance': 'medium',
        'max_drawdown': 0.02
    }
    
    # 创建决策器
    decision_maker = AITradingDecision(config)
    
    # 模拟市场数据
    sample_market_data = {
        'current_price': 110574.50,
        'smc_structures': {
            '1d': {'fvg_events': [], 'ob_events': [{}]*6},
            '4h': {'fvg_events': [], 'ob_events': [{}]*4},
            '1h': {'fvg_events': [], 'ob_events': [{}]*6},
            '15m': {'fvg_events': [], 'ob_events': [{}]*10},
            '3m': {'fvg_events': [], 'ob_events': [{}]*13}
        },
        'price_changes': {'1d': 0.005, '4h': -0.003, '1h': 0.001, '15m': 0.002, '3m': -0.001},
        'volatility': 0.006
    }
    
    # 提取简化信息
    market_info = decision_maker.extract_simplified_market_info(sample_market_data)
    
    # 生成提示词
    prompt = decision_maker.generate_trading_prompt(market_info)
    print("=== AI交易决策提示词 ===")
    print(prompt)
    print("\n=== 简化信息提取 ===")
    print(json.dumps(market_info, indent=2, ensure_ascii=False))