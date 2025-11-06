"""
纯原始数据 + AI决策交易系统
完全移除FVG/OB检测逻辑，只提供原始K线数据
将权重考虑转换为AI提示词
"""

import json
import time
from typing import Dict, Any, List
import os

class PureRawDataAITrader:
    """纯原始数据AI交易系统"""
    
    def __init__(self, config_file: str = "trading_config.json"):
        """
        从配置文件初始化交易系统
        
        Args:
            config_file: 配置文件路径
        """
        self.config = self._load_config(config_file)
        self.trading_style = self.config.get('trading_style', 'day_trading')
        self.risk_preference = self.config.get('risk_preference', 'moderate')
        
        # 初始化风格配置
        self.style_config = self._get_trading_style_config()
        self.risk_config = self._get_risk_preference_config()
        
        print(f"🎯 纯原始数据AI交易系统初始化完成")
        print(f"   配置文件: {config_file}")
        print(f"   交易风格: {self.trading_style}")
        print(f"   风险偏好: {self.risk_preference}")
        print(f"   监控品种: {', '.join(self.config.get('symbols', ['BTC/USD']))}")
        print(f"   🚫 已完全移除FVG/OB检测逻辑")
        print(f"   📊 仅提供原始K线数据给AI分析")
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """加载配置文件"""
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # 默认配置
            return {
                'trading_style': 'day_trading',
                'risk_preference': 'moderate',
                'symbols': ['BTC/USD']
            }
    
    def _get_trading_style_config(self) -> Dict[str, Any]:
        """获取交易风格配置（放弃日线及以上交易）"""
        styles = {
            'scalping': {
                'timeframes': ['1m', '3m', '5m'],
                'holding_period': '分钟级别',
                'profit_target': '0.5-1%',
                'description': '高频短线交易，快速进出',
                'max_trades_per_day': 20,
                'preferred_session': '亚洲/欧洲重叠时段'
            },
            'day_trading': {
                'timeframes': ['15m', '1h', '4h'],
                'holding_period': '日内交易',
                'profit_target': '1-3%',
                'description': '日内交易，不过夜持仓',
                'max_trades_per_day': 5,
                'preferred_session': '欧洲/美国重叠时段'
            },
            'swing_trading': {
                'timeframes': ['4h', '1h', '15m'],  # 放弃1d、3d，保留4h及以下
                'holding_period': '几天',
                'profit_target': '2-5%',
                'description': '短期波段交易，不过夜持仓',
                'max_trades_per_week': 3,
                'preferred_session': '任何时段'
            },
            'position_trading': {
                'timeframes': ['4h', '1h'],  # 放弃1d、3d、1w，最大4h
                'holding_period': '几天',
                'profit_target': '3-8%',
                'description': '短期持仓交易，不过夜持仓',
                'max_trades_per_month': 2,
                'preferred_session': '任何时段'
            }
        }
        return styles.get(self.trading_style, styles['day_trading'])
    
    def _get_risk_preference_config(self) -> Dict[str, Any]:
        """获取风险偏好配置"""
        risks = {
            'conservative': {
                'max_position_size': '1-2%',
                'stop_loss': '0.5-1%',
                'max_drawdown': '2%',
                'leverage': '无杠杆',
                'description': '保守型，严格控制风险',
                'risk_reward_ratio': '1:2',
                'max_daily_loss': '1%'
            },
            'moderate': {
                'max_position_size': '3-5%',
                'stop_loss': '1-2%',
                'max_drawdown': '5%',
                'leverage': '低杠杆(1-2x)',
                'description': '稳健型，平衡风险收益',
                'risk_reward_ratio': '1:3',
                'max_daily_loss': '3%'
            },
            'aggressive': {
                'max_position_size': '5-10%',
                'stop_loss': '2-3%',
                'max_drawdown': '10%',
                'leverage': '中杠杆(3-5x)',
                'description': '激进型，追求高收益',
                'risk_reward_ratio': '1:4',
                'max_daily_loss': '5%'
            }
        }
        return risks.get(self.risk_preference, risks['moderate'])
    
    def extract_pure_raw_data(self, symbol: str = 'BTC/USD') -> Dict[str, Any]:
        """
        提取纯原始K线数据（完全移除FVG/OB检测）
        
        Args:
            symbol: 交易品种
        """
        print(f"📊 开始提取 {symbol} 纯原始K线数据...")
        
        # 模拟不同品种的基础数据（实际中替换为真实API）
        symbol_data = {
            'BTC/USD': {
                'current_price': 110574.50,
                'volatility': 0.025,
                'typical_range': 2000.0
            },
            'ETH/USD': {
                'current_price': 3250.75,
                'volatility': 0.035,
                'typical_range': 150.0
            },
            'SOL/USD': {
                'current_price': 145.20,
                'volatility': 0.045,
                'typical_range': 8.0
            }
        }
        
        base_data = symbol_data.get(symbol, symbol_data['BTC/USD'])
        
        # 纯原始数据 - 不包含任何FVG/OB检测结果
        raw_data = {
            'timestamp': time.time(),
            'symbol': symbol,
            
            # 基础价格信息
            'price_info': {
                'current': base_data['current_price'],
                'open': base_data['current_price'] * 0.998,
                'high': base_data['current_price'] * 1.005,
                'low': base_data['current_price'] * 0.995,
                'close': base_data['current_price']
            },
            
            # 多时间框架K线数据
            'candles': {},
            
            # 基础市场指标
            'market_metrics': {
                '24h_change': 0.005,
                '24h_high': base_data['current_price'] * 1.01,
                '24h_low': base_data['current_price'] * 0.99,
                'volume_24h': 89214.6 if symbol == 'BTC/USD' else 24567.8,
                'volatility': base_data['volatility'],
                'typical_range': base_data['typical_range']
            },
            
            # 技术指标（基础计算）
            'technical_indicators': {
                'rsi_1h': 45.2,
                'rsi_4h': 52.1,
                'macd_1h': -12.5,
                'macd_4h': 8.3,
                'ema_20': base_data['current_price'] * 0.997,
                'ema_50': base_data['current_price'] * 0.995,
                'bollinger_upper': base_data['current_price'] * 1.01,
                'bollinger_lower': base_data['current_price'] * 0.99,
                'atr_1h': base_data['typical_range'] * 0.1,
                'atr_4h': base_data['typical_range'] * 0.2
            },
            
            # 日内交易专用数据
            'intraday_data': {
                # 成交量分析
                'volume_analysis': {
                    'volume_trend': 'stable',  # increasing, decreasing, stable
                    'volume_spike_detected': False,
                    'avg_volume_1h': 500.0,
                    'current_volume_ratio': 1.2,
                    'volume_profile': {
                        'support_levels': [base_data['current_price'] * 0.99, base_data['current_price'] * 0.985],
                        'resistance_levels': [base_data['current_price'] * 1.01, base_data['current_price'] * 1.015]
                    }
                },
                
                # 时间维度数据
                'time_analysis': {
                    'current_session': 'asia',  # asia, europe, us
                    'session_volatility': {
                        'asia': 0.015,
                        'europe': 0.025,
                        'us': 0.035
                    },
                    'intraday_high_low': {
                        'session_high': base_data['current_price'] * 1.008,
                        'session_low': base_data['current_price'] * 0.992,
                        'high_time': '10:30',
                        'low_time': '14:45'
                    }
                },
                
                # 动量指标
                'momentum_indicators': {
                    'price_momentum_1h': 0.002,
                    'price_momentum_4h': 0.005,
                    'breakout_signals': {
                        'resistance_break': False,
                        'support_break': False,
                        'consolidation_break': False
                    },
                    'trend_strength': 0.6  # 0-1, 1为最强
                },
                
                # 市场微观结构
                'market_microstructure': {
                    'order_book_depth': {
                        'bid_depth': 25000.0,
                        'ask_depth': 23000.0,
                        'depth_imbalance': 0.08
                    },
                    'large_orders': {
                        'recent_large_buy': 5000.0,
                        'recent_large_sell': 4500.0,
                        'order_flow': 'neutral'  # bullish, bearish, neutral
                    },
                    'liquidity_analysis': {
                        'liquidity_zones': [base_data['current_price'] * 0.995, base_data['current_price'] * 1.005],
                        'slippage_estimate': 0.001,
                        'market_depth_score': 0.7
                    }
                }
            }
        }
        
        # 为每个时间框架生成K线数据
        for timeframe in self.style_config['timeframes']:
            raw_data['candles'][timeframe] = {
                'open': base_data['current_price'] * (1 - base_data['volatility'] * 0.2),
                'high': base_data['current_price'] * (1 + base_data['volatility'] * 0.5),
                'low': base_data['current_price'] * (1 - base_data['volatility'] * 0.5),
                'close': base_data['current_price'],
                'volume': 1000.0,
                'timeframe': timeframe
            }
        
        print(f"✅ {symbol} 纯原始数据提取完成（无FVG/OB检测）")
        return raw_data
    
    def generate_ai_prompt_with_weight_considerations(self, raw_data: Dict[str, Any]) -> str:
        """
        生成包含权重考虑的AI提示词
        将原本的Python权重计算逻辑转换为AI提示词
        """
        
        prompt = f"""
你是一个专业的加密货币交易AI，专门为个性化交易需求提供决策支持。

## ⚠️ 重要交易限制
**我们已放弃日线(1d)及以上的交易，专注于日内时间框架**
- 最大时间框架: 4h
- 不过夜持仓，所有交易在日内完成
- 重点关注短期价格行为和日内趋势

## 📊 纯原始市场数据 - {raw_data['symbol']}
当前价格: ${raw_data['price_info']['current']:,.2f}
24小时变化: {raw_data['market_metrics']['24h_change']*100:.2f}%
波动率: {raw_data['market_metrics']['volatility']*100:.2f}%

## 🎯 交易风格配置（日内交易）
- 风格: {self.trading_style}
- 描述: {self.style_config['description']}
- 时间框架: {', '.join(self.style_config['timeframes'])} (最大4h)
- 持仓周期: {self.style_config['holding_period']}
- 盈利目标: {self.style_config['profit_target']}

## ⚖️ 风险偏好配置
- 偏好: {self.risk_preference}
- 描述: {self.risk_config['description']}
- 止损: {self.risk_config['stop_loss']}
- 风险回报比: {self.risk_config['risk_reward_ratio']}

## 📈 日内时间框架K线数据
"""
        
        # 添加K线数据
        for timeframe, candle in raw_data['candles'].items():
            prompt += f"""
**{timeframe}时间框架**:
- 开盘: ${candle['open']:,.2f}
- 最高: ${candle['high']:,.2f}
- 最低: ${candle['low']:,.2f}
- 收盘: ${candle['close']:,.2f}
- 成交量: {candle['volume']:,.1f}
"""
        
        prompt += f"""
## 🔍 技术指标
- RSI(1h): {raw_data['technical_indicators']['rsi_1h']:.1f}
- RSI(4h): {raw_data['technical_indicators']['rsi_4h']:.1f}
- MACD(1h): {raw_data['technical_indicators']['macd_1h']:.1f}
- MACD(4h): {raw_data['technical_indicators']['macd_4h']:.1f}
- EMA20: ${raw_data['technical_indicators']['ema_20']:,.2f}
- EMA50: ${raw_data['technical_indicators']['ema_50']:,.2f}
- 布林带上轨: ${raw_data['technical_indicators']['bollinger_upper']:,.2f}
- 布林带下轨: ${raw_data['technical_indicators']['bollinger_lower']:,.2f}
- ATR(1h): ${raw_data['technical_indicators']['atr_1h']:,.2f}
- ATR(4h): ${raw_data['technical_indicators']['atr_4h']:,.2f}

## 📊 日内交易专用数据

### 成交量分析
- 成交量趋势: {raw_data['intraday_data']['volume_analysis']['volume_trend']}
- 成交量异常: {'有' if raw_data['intraday_data']['volume_analysis']['volume_spike_detected'] else '无'}成交量放大
- 当前成交量比率: {raw_data['intraday_data']['volume_analysis']['current_volume_ratio']:.1f}

### 时间维度分析
- 当前交易时段: {raw_data['intraday_data']['time_analysis']['current_session']}
- 时段波动率: {raw_data['intraday_data']['time_analysis']['session_volatility'][raw_data['intraday_data']['time_analysis']['current_session']]*100:.1f}%
- 日内高点: ${raw_data['intraday_data']['time_analysis']['intraday_high_low']['session_high']:,.2f} (时间: {raw_data['intraday_data']['time_analysis']['intraday_high_low']['high_time']})
- 日内低点: ${raw_data['intraday_data']['time_analysis']['intraday_high_low']['session_low']:,.2f} (时间: {raw_data['intraday_data']['time_analysis']['intraday_high_low']['low_time']})

### 动量指标
- 1小时价格动量: {raw_data['intraday_data']['momentum_indicators']['price_momentum_1h']*100:.2f}%
- 4小时价格动量: {raw_data['intraday_data']['momentum_indicators']['price_momentum_4h']*100:.2f}%
- 趋势强度: {raw_data['intraday_data']['momentum_indicators']['trend_strength']*100:.0f}%
- 突破信号: {', '.join([k for k, v in raw_data['intraday_data']['momentum_indicators']['breakout_signals'].items() if v]) or '无'}

### 市场微观结构
- 订单簿深度: 买盘${raw_data['intraday_data']['market_microstructure']['order_book_depth']['bid_depth']:,.0f} / 卖盘${raw_data['intraday_data']['market_microstructure']['order_book_depth']['ask_depth']:,.0f}
- 深度不平衡: {raw_data['intraday_data']['market_microstructure']['order_book_depth']['depth_imbalance']*100:.1f}%
- 大单流向: {raw_data['intraday_data']['market_microstructure']['large_orders']['order_flow']}
- 流动性评分: {raw_data['intraday_data']['market_microstructure']['liquidity_analysis']['market_depth_score']*100:.0f}%

## 💡 权重考虑因素（请AI重点分析）

### 1. 时间框架权重（基于日内交易）
**重要提示**: 我们专注于日内交易，最大时间框架为4h：
- **{self.trading_style}风格**: 重点关注{self.style_config['timeframes'][-1]}和{self.style_config['timeframes'][-2]}时间框架
- 短期框架({self.style_config['timeframes'][0]})用于精确入场时机
- 4h框架用于日内趋势方向判断
- **放弃日线及以上分析**，所有决策基于日内数据

### 2. 价格水平权重
**关键价格区域**:
- 24小时高点: ${raw_data['market_metrics']['24h_high']:,.2f} (阻力)
- 24小时低点: ${raw_data['market_metrics']['24h_low']:,.2f} (支撑)
- EMA20: ${raw_data['technical_indicators']['ema_20']:,.2f} (动态支撑/阻力)
- EMA50: ${raw_data['technical_indicators']['ema_50']:,.2f} (趋势判断)

### 3. 波动率权重
**波动率考量**:
- 当前波动率: {raw_data['market_metrics']['volatility']*100:.2f}%
- 典型价格区间: ${raw_data['market_metrics']['typical_range']:,.2f}
- **风险提示**: 高波动率({'>3%'})需要更宽止损，低波动率({'<1%'})可能缺乏交易机会

### 4. 成交量权重
**成交量分析**:
- 24小时成交量: {raw_data['market_metrics']['volume_24h']:,.1f}
- **关键**: 关注价格突破时的成交量配合情况

### 5. 技术指标权重
**指标优先级**:
1. **RSI**: 超买(>70)/超卖(<30)区域重点关注
2. **MACD**: 金叉/死叉信号，结合趋势判断
3. **EMA**: 价格相对于EMA的位置判断趋势强度

## 🎯 决策要求

请基于以上原始数据和权重考虑因素，提供：

1. **交易决策**: BUY/SELL/WAIT
2. **入场区间**: 具体价格范围
3. **止损位置**: 基于风险偏好
4. **目标价位**: 基于盈利目标
5. **置信度**: 0-1之间的评分
6. **详细分析**: 结合权重因素的解释

**特别提醒**: 请直接基于原始K线数据进行分析，无需考虑FVG/OB等复杂结构检测。
"""
        
        return prompt
    
    def simulate_ai_analysis(self, prompt: str, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """模拟AI分析（实际中替换为真实API调用）"""
        print("🤖 开始AI分析（模拟）...")
        
        # 基于实际数据动态生成分析结果
        current_price = raw_data['price_info']['current']
        volatility = raw_data['market_metrics']['volatility']
        rsi_1h = raw_data['technical_indicators']['rsi_1h']
        rsi_4h = raw_data['technical_indicators']['rsi_4h']
        ema_20 = raw_data['technical_indicators']['ema_20']
        ema_50 = raw_data['technical_indicators']['ema_50']
        
        # 基于综合日内数据动态决定交易决策
        intraday_data = raw_data['intraday_data']
        
        # 计算综合得分
        score = 0
        
        # 1. 技术指标得分
        if rsi_1h < 30:
            score += 2  # 超卖区域
        elif rsi_1h > 70:
            score -= 2  # 超买区域
            
        if current_price > ema_20:
            score += 1  # 价格在EMA20之上
        else:
            score -= 1  # 价格在EMA20之下
            
        # 2. 成交量分析得分
        if intraday_data['volume_analysis']['volume_trend'] == 'increasing':
            score += 1
        if intraday_data['volume_analysis']['volume_spike_detected']:
            score += 2
            
        # 3. 动量指标得分
        if intraday_data['momentum_indicators']['trend_strength'] > 0.7:
            score += 1
        if any(intraday_data['momentum_indicators']['breakout_signals'].values()):
            score += 2
            
        # 4. 市场微观结构得分
        if intraday_data['market_microstructure']['large_orders']['order_flow'] == 'bullish':
            score += 1
        elif intraday_data['market_microstructure']['large_orders']['order_flow'] == 'bearish':
            score -= 1
            
        # 5. 流动性得分
        if intraday_data['market_microstructure']['liquidity_analysis']['market_depth_score'] > 0.7:
            score += 1
            
        # 根据综合得分决定交易决策
        if score >= 4:
            decision = 'BUY'
            confidence = min(0.9, 0.75 + score * 0.05)
        elif score <= -4:
            decision = 'SELL'
            confidence = min(0.85, 0.75 + abs(score) * 0.05)
        else:
            decision = 'WAIT'
            confidence = 0.75
        
        # 动态计算交易参数
        if decision == 'BUY':
            entry_range = {
                'buy': [current_price * 0.998, current_price * 1.002],
                'sell': [current_price * 1.015, current_price * 1.025]
            }
            stop_loss = current_price * (1 - volatility * 2)
            target_price = current_price * (1 + volatility * 3)
        elif decision == 'SELL':
            entry_range = {
                'buy': [current_price * 0.975, current_price * 0.985],
                'sell': [current_price * 0.998, current_price * 1.002]
            }
            stop_loss = current_price * (1 + volatility * 2)
            target_price = current_price * (1 - volatility * 3)
        else:
            entry_range = {
                'buy': [current_price * 0.995, current_price * 1.005],
                'sell': [current_price * 1.005, current_price * 1.015]
            }
            stop_loss = current_price * (1 + volatility * 2)
            target_price = current_price * (1 - volatility * 2)
        
        # 生成动态分析文本
        symbol = raw_data['symbol']
        analysis_text = f'''
📊 **基于纯原始数据的分析 - {symbol}**

**市场状态**: 当前价格${current_price:,.2f}，处于{'超卖' if rsi_1h < 30 else '超买' if rsi_1h > 70 else '中性'}区域。

**权重分析**:
1. **时间框架**: 价格相对于EMA20(${ema_20:,.2f})和EMA50(${ema_50:,.2f})的位置
2. **价格水平**: 24小时区间${raw_data['market_metrics']['24h_low']:,.2f}-${raw_data['market_metrics']['24h_high']:,.2f}
3. **波动率**: {volatility*100:.1f}%的波动率，适合日内交易
4. **技术指标**: RSI(1h)={rsi_1h:.1f}, RSI(4h)={rsi_4h:.1f}

**日内数据深度分析**:
- 成交量趋势：{intraday_data['volume_analysis']['volume_trend']}，{'有' if intraday_data['volume_analysis']['volume_spike_detected'] else '无'}异常放大
- 当前时段：{intraday_data['time_analysis']['current_session']}，波动率{intraday_data['time_analysis']['session_volatility'][intraday_data['time_analysis']['current_session']]*100:.1f}%
- 动量强度：{intraday_data['momentum_indicators']['trend_strength']*100:.0f}%，突破信号：{', '.join([k for k, v in intraday_data['momentum_indicators']['breakout_signals'].items() if v]) or '无'}
- 大单流向：{intraday_data['market_microstructure']['large_orders']['order_flow']}，流动性评分：{intraday_data['market_microstructure']['liquidity_analysis']['market_depth_score']*100:.0f}%

**决策理由**:
- {decision}决策基于当前技术指标和市场条件 (综合得分：{score})
- 置信度: {confidence*100:.1f}%
- 风险回报比符合{self.risk_preference}偏好要求

💡 **监控重点**:
- 价格突破EMA20(${ema_20:,.2f})可能触发趋势
- RSI指标进入超买/超卖区域时重点关注
- 重点关注成交量变化和订单簿深度
'''
        
        analysis_result = {
            'decision': decision,
            'confidence': confidence,
            'entry_range': entry_range,
            'stop_loss': round(stop_loss, 2),
            'target_price': round(target_price, 2),
            'analysis': analysis_text
        }
        
        print("✅ AI分析完成")
        return analysis_result
    
    def analyze_multiple_symbols(self) -> Dict[str, Any]:
        """分析多个交易品种"""
        start_time = time.time()
        symbols = self.config.get('symbols', ['BTC/USD'])
        
        print(f"🔍 开始分析 {len(symbols)} 个品种...")
        
        results = {}
        for symbol in symbols:
            print(f"\n--- 分析 {symbol} ---")
            
            # 1. 提取纯原始数据
            raw_data = self.extract_pure_raw_data(symbol)
            
            # 2. 生成包含权重考虑的AI提示词
            prompt = self.generate_ai_prompt_with_weight_considerations(raw_data)
            
            # 3. 模拟AI分析
            ai_result = self.simulate_ai_analysis(prompt, raw_data)
            
            results[symbol] = {
                'raw_data': raw_data,
                'prompt_preview': prompt[:500] + '...' if len(prompt) > 500 else prompt,
                'ai_analysis': ai_result,
                'timestamp': time.time()
            }
            
            print(f"✅ {symbol} 分析完成")
        
        total_time = time.time() - start_time
        
        # 保存结果
        result_file = f"pure_rawdata_analysis_{int(time.time())}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'config': {
                    'trading_style': self.trading_style,
                    'risk_preference': self.risk_preference,
                    'symbols': symbols
                },
                'results': results,
                'performance': {
                    'total_time': total_time,
                    'symbols_analyzed': len(symbols),
                    'avg_time_per_symbol': total_time / len(symbols) if len(symbols) > 0 else 0
                }
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n🎉 多品种分析完成!")
        print(f"   总耗时: {total_time:.3f}秒")
        print(f"   分析品种: {len(symbols)}个")
        print(f"   结果文件: {result_file}")
        print(f"   💡 系统特点: 纯原始数据 + AI权重分析，无FVG/OB检测")
        
        return results

# 主函数
def main():
    """主函数"""
    trader = PureRawDataAITrader("trading_config.json")
    trader.analyze_multiple_symbols()

if __name__ == "__main__":
    main()