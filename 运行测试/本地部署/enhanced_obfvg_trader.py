"""
增强版OB/FVG止损价AI交易系统
重新集成OB/FVG分析，专门用于设置止损价和爆仓价
考虑关键位置：OB/FVG/结构破坏/日开盘价/内日4小时高低点/本周开盘价/上周高低点
"""

import json
import time
import random
from typing import Dict, Any, List, Tuple
import os
from datetime import datetime, timedelta

class OBFVGAITrader:
    """OB/FVG增强止损价AI交易系统"""
    
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
        
        # 今日交易统计
        self.today_stats = {
            "initial_capital": 10000.0,  # 初始金额
            "today_pnl": 0.0,           # 今日盈亏金额
            "today_wins": 0,            # 今日胜场数
            "today_losses": 0,          # 今日败场数
            "today_trades": 0,          # 今日交易次数
            "positions": []             # 持仓情况
        }
        
        print(f"🎯 OB/FVG增强止损价AI交易系统初始化完成")
        print(f"   配置文件: {config_file}")
        print(f"   交易风格: {self.trading_style}")
        print(f"   风险偏好: {self.risk_preference}")
        print(f"   监控品种: {', '.join(self.config.get('symbols', ['BTC/USD']))}")
        print(f"   🔥 重新集成OB/FVG分析用于止损价设置")
        print(f"   📍 关键位置: OB/FVG/结构破坏/日开盘价/4h高低点/本周开盘价/上周高低点")
    
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
    
    def get_today_trading_stats(self) -> Dict[str, Any]:
        """获取今日交易统计"""
        # 计算今日胜率
        total_trades = self.today_stats["today_trades"]
        win_rate = (self.today_stats["today_wins"] / total_trades * 100) if total_trades > 0 else 0
        
        # 计算当前资金
        current_capital = self.today_stats["initial_capital"] + self.today_stats["today_pnl"]
        
        return {
            "initial_capital": self.today_stats["initial_capital"],
            "current_capital": current_capital,
            "today_pnl": self.today_stats["today_pnl"],
            "today_trades": self.today_stats["today_trades"],
            "today_wins": self.today_stats["today_wins"],
            "today_losses": self.today_stats["today_losses"],
            "today_win_rate": win_rate,
            "positions": self.today_stats["positions"].copy()
        }
    
    def update_trading_stats(self, symbol: str, result: str, pnl: float, position_size: float):
        """更新交易统计"""
        self.today_stats["today_trades"] += 1
        self.today_stats["today_pnl"] += pnl
        
        if result == "WIN":
            self.today_stats["today_wins"] += 1
        else:
            self.today_stats["today_losses"] += 1
        
        # 更新持仓情况
        if result == "OPEN":
            self.today_stats["positions"].append({
                "symbol": symbol,
                "position_size": position_size,
                "entry_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        elif result == "CLOSE":
            # 移除已平仓的持仓
            self.today_stats["positions"] = [
                pos for pos in self.today_stats["positions"] 
                if pos["symbol"] != symbol
            ]
    
    def detect_ob_fvg_key_levels(self, symbol: str, current_price: float) -> Dict[str, Any]:
        """
        检测OB/FVG关键位置用于止损价设置
        
        Args:
            symbol: 交易品种
            current_price: 当前价格
        """
        print(f"📍 开始检测 {symbol} 关键位置...")
        
        # 模拟关键位置检测（实际中替换为真实算法）
        key_levels = {
            # OB (Order Block) 检测
            'ob_levels': {
                'bullish_ob': self._generate_bullish_ob_levels(current_price),
                'bearish_ob': self._generate_bearish_ob_levels(current_price),
                'ob_strength': random.uniform(0.6, 0.9),  # OB强度评分
                'recent_ob_count': random.randint(2, 5)   # 近期OB数量
            },
            
            # FVG (Fair Value Gap) 检测
            'fvg_levels': {
                'bullish_fvg': self._generate_bullish_fvg_levels(current_price),
                'bearish_fvg': self._generate_bearish_fvg_levels(current_price),
                'fvg_strength': random.uniform(0.5, 0.8),  # FVG强度评分
                'recent_fvg_count': random.randint(1, 4)     # 近期FVG数量
            },
            
            # 结构破坏点
            'structure_break': {
                'bos_levels': self._generate_bos_levels(current_price),  # BOS (Break of Structure)
                'choch_levels': self._generate_choch_levels(current_price),  # CHOCH (Change of Character)
                'structure_strength': random.uniform(0.7, 0.95)
            },
            
            # 日内关键位置
            'intraday_levels': {
                'daily_open': self._calculate_daily_open(current_price),  # 日开盘价
                '4h_high_low': self._calculate_4h_high_low(current_price),  # 4小时高低点
                'intraday_vwap': current_price * random.uniform(0.998, 1.002)  # 日内VWAP
            },
            
            # 周级别关键位置
            'weekly_levels': {
                'week_open': self._calculate_week_open(current_price),  # 本周开盘价
                'prev_week_high_low': self._calculate_prev_week_high_low(current_price),  # 上周高低点
                'weekly_pivot': self._calculate_weekly_pivot(current_price)  # 周枢轴点
            },
            
            # 综合评分
            'overall_score': {
                'key_level_quality': random.uniform(0.6, 0.9),  # 关键位置质量
                'stop_loss_confidence': random.uniform(0.7, 0.95),  # 止损价置信度
                'risk_reward_enhancement': random.uniform(1.2, 2.0)  # 风险回报比增强
            }
        }
        
        print(f"✅ {symbol} 关键位置检测完成")
        print(f"   OB强度: {key_levels['ob_levels']['ob_strength']:.2f}")
        print(f"   FVG强度: {key_levels['fvg_levels']['fvg_strength']:.2f}")
        print(f"   结构强度: {key_levels['structure_break']['structure_strength']:.2f}")
        
        return key_levels
    
    def _generate_bullish_ob_levels(self, current_price: float) -> Dict[str, float]:
        """生成看涨OB水平"""
        return {
            'support': current_price * random.uniform(0.98, 0.995),
            'resistance': current_price * random.uniform(1.005, 1.02),
            'mid_point': current_price * random.uniform(0.995, 1.005),
            'volume_confirmation': random.uniform(0.7, 1.2)
        }
    
    def _generate_bearish_ob_levels(self, current_price: float) -> Dict[str, float]:
        """生成看跌OB水平"""
        return {
            'support': current_price * random.uniform(0.98, 0.995),
            'resistance': current_price * random.uniform(1.005, 1.02),
            'mid_point': current_price * random.uniform(0.995, 1.005),
            'volume_confirmation': random.uniform(0.7, 1.2)
        }
    
    def _generate_bullish_fvg_levels(self, current_price: float) -> Dict[str, float]:
        """生成看涨FVG水平"""
        return {
            'gap_top': current_price * random.uniform(1.002, 1.01),
            'gap_bottom': current_price * random.uniform(0.99, 0.998),
            'gap_size': current_price * random.uniform(0.002, 0.01),
            'retest_probability': random.uniform(0.6, 0.9)
        }
    
    def _generate_bearish_fvg_levels(self, current_price: float) -> Dict[str, float]:
        """生成看跌FVG水平"""
        return {
            'gap_top': current_price * random.uniform(1.002, 1.01),
            'gap_bottom': current_price * random.uniform(0.99, 0.998),
            'gap_size': current_price * random.uniform(0.002, 0.01),
            'retest_probability': random.uniform(0.6, 0.9)
        }
    
    def _generate_bos_levels(self, current_price: float) -> Dict[str, float]:
        """生成BOS结构破坏水平"""
        return {
            'breakout_level': current_price * random.uniform(0.995, 1.005),
            'invalidation_level': current_price * random.uniform(0.99, 1.01),
            'momentum_strength': random.uniform(0.5, 0.9)
        }
    
    def _generate_choch_levels(self, current_price: float) -> Dict[str, float]:
        """生成CHOCH结构破坏水平"""
        return {
            'reversal_level': current_price * random.uniform(0.995, 1.005),
            'confirmation_level': current_price * random.uniform(0.99, 1.01),
            'trend_change_probability': random.uniform(0.4, 0.8)
        }
    
    def _calculate_daily_open(self, current_price: float) -> float:
        """计算日开盘价"""
        return current_price * random.uniform(0.995, 1.005)
    
    def _calculate_4h_high_low(self, current_price: float) -> Dict[str, float]:
        """计算4小时高低点"""
        return {
            '4h_high': current_price * random.uniform(1.002, 1.01),
            '4h_low': current_price * random.uniform(0.99, 0.998),
            '4h_range': current_price * random.uniform(0.005, 0.02)
        }
    
    def _calculate_week_open(self, current_price: float) -> float:
        """计算本周开盘价"""
        return current_price * random.uniform(0.98, 1.02)
    
    def _calculate_prev_week_high_low(self, current_price: float) -> Dict[str, float]:
        """计算上周高低点"""
        return {
            'prev_week_high': current_price * random.uniform(1.01, 1.05),
            'prev_week_low': current_price * random.uniform(0.95, 0.99),
            'prev_week_range': current_price * random.uniform(0.05, 0.1)
        }
    
    def _calculate_weekly_pivot(self, current_price: float) -> Dict[str, float]:
        """计算周枢轴点"""
        return {
            'pivot': current_price * random.uniform(0.995, 1.005),
            'r1': current_price * random.uniform(1.01, 1.03),
            'r2': current_price * random.uniform(1.03, 1.06),
            's1': current_price * random.uniform(0.97, 0.99),
            's2': current_price * random.uniform(0.94, 0.97)
        }
    
    def extract_enhanced_market_data(self, symbol: str = 'BTC/USD') -> Dict[str, Any]:
        """
        提取增强版市场数据（包含OB/FVG关键位置）
        
        Args:
            symbol: 交易品种
        """
        print(f"📊 开始提取 {symbol} 增强版市场数据...")
        
        # 模拟不同品种的基础数据
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
        current_price = base_data['current_price']
        
        # 检测OB/FVG关键位置
        key_levels = self.detect_ob_fvg_key_levels(symbol, current_price)
        
        # 增强版市场数据
        enhanced_data = {
            'timestamp': time.time(),
            'symbol': symbol,
            
            # 基础价格信息
            'price_info': {
                'current': current_price,
                'open': current_price * 0.998,
                'high': current_price * 1.005,
                'low': current_price * 0.995,
                'close': current_price
            },
            
            # OB/FVG关键位置数据
            'key_levels': key_levels,
            
            # 多时间框架K线数据
            'candles': {},
            
            # 基础市场指标
            'market_metrics': {
                '24h_change': 0.005,
                '24h_high': current_price * 1.01,
                '24h_low': current_price * 0.99,
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
                'ema_20': current_price * 0.997,
                'ema_50': current_price * 0.995
            }
        }
        
        # 为每个时间框架生成K线数据
        for timeframe in self.style_config['timeframes']:
            enhanced_data['candles'][timeframe] = {
                'open': current_price * (1 - base_data['volatility'] * 0.2),
                'high': current_price * (1 + base_data['volatility'] * 0.5),
                'low': current_price * (1 - base_data['volatility'] * 0.5),
                'close': current_price,
                'volume': 1000.0,
                'timeframe': timeframe
            }
        
        print(f"✅ {symbol} 增强版市场数据提取完成")
        print(f"   🔥 已集成OB/FVG关键位置分析")
        
        return enhanced_data
    
    def generate_enhanced_ai_prompt(self, enhanced_data: Dict[str, Any]) -> str:
        """
        生成增强版AI提示词（包含OB/FVG止损价设置指导）
        """
        
        # 获取今日交易统计
        today_stats = self.get_today_trading_stats()
        
        # 构建持仓情况描述
        positions_info = ""
        if today_stats["positions"]:
            positions_info = "\n📊 **当前持仓情况**:"
            for pos in today_stats["positions"]:
                positions_info += f"\n- {pos['symbol']}: {pos['position_size']:.4f} 单位 (开仓时间: {pos['entry_time']})"
        else:
            positions_info = "\n📊 **当前持仓情况**: 无持仓"
        
        prompt = f"""
你是一个专业的加密货币交易AI，专门为个性化交易需求提供决策支持，特别擅长基于OB/FVG关键位置设置止损价。

## 📅 今日交易统计
**初始金额**: ${today_stats['initial_capital']:,.2f}
**当前资金**: ${today_stats['current_capital']:,.2f}
**今日盈亏**: ${today_stats['today_pnl']:,.2f}
**今日交易次数**: {today_stats['today_trades']}
**今日胜率**: {today_stats['today_win_rate']:.1f}%
**胜场/败场**: {today_stats['today_wins']}/{today_stats['today_losses']}
{positions_info}

## ⚠️ 重要交易限制
**我们已放弃日线(1d)及以上的交易，专注于日内时间框架**
- 最大时间框架: 4h
- 不过夜持仓，所有交易在日内完成
- 重点关注短期价格行为和日内趋势

## 📊 增强版市场数据 - {enhanced_data['symbol']}
当前价格: ${enhanced_data['price_info']['current']:,.2f}
24小时变化: {enhanced_data['market_metrics']['24h_change']*100:.2f}%
波动率: {enhanced_data['market_metrics']['volatility']*100:.2f}%

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

## 🔥 OB/FVG关键位置分析（用于止损价设置）

### 📍 OB (Order Block) 分析
**看涨OB水平**:
- 支撑: ${enhanced_data['key_levels']['ob_levels']['bullish_ob']['support']:,.2f}
- 阻力: ${enhanced_data['key_levels']['ob_levels']['bullish_ob']['resistance']:,.2f}
- 中点: ${enhanced_data['key_levels']['ob_levels']['bullish_ob']['mid_point']:,.2f}

**看跌OB水平**:
- 支撑: ${enhanced_data['key_levels']['ob_levels']['bearish_ob']['support']:,.2f}
- 阻力: ${enhanced_data['key_levels']['ob_levels']['bearish_ob']['resistance']:,.2f}
- 中点: ${enhanced_data['key_levels']['ob_levels']['bearish_ob']['mid_point']:,.2f}

**OB强度**: {enhanced_data['key_levels']['ob_levels']['ob_strength']:.2f}
**近期OB数量**: {enhanced_data['key_levels']['ob_levels']['recent_ob_count']}个

### 📈 FVG (Fair Value Gap) 分析
**看涨FVG**:
- 缺口顶部: ${enhanced_data['key_levels']['fvg_levels']['bullish_fvg']['gap_top']:,.2f}
- 缺口底部: ${enhanced_data['key_levels']['fvg_levels']['bullish_fvg']['gap_bottom']:,.2f}
- 缺口大小: ${enhanced_data['key_levels']['fvg_levels']['bullish_fvg']['gap_size']:,.2f}

**看跌FVG**:
- 缺口顶部: ${enhanced_data['key_levels']['fvg_levels']['bearish_fvg']['gap_top']:,.2f}
- 缺口底部: ${enhanced_data['key_levels']['fvg_levels']['bearish_fvg']['gap_bottom']:,.2f}
- 缺口大小: ${enhanced_data['key_levels']['fvg_levels']['bearish_fvg']['gap_size']:,.2f}

**FVG强度**: {enhanced_data['key_levels']['fvg_levels']['fvg_strength']:.2f}
**近期FVG数量**: {enhanced_data['key_levels']['fvg_levels']['recent_fvg_count']}个

### 🏗️ 结构破坏分析
**BOS (Break of Structure)**:
- 突破水平: ${enhanced_data['key_levels']['structure_break']['bos_levels']['breakout_level']:,.2f}
- 失效水平: ${enhanced_data['key_levels']['structure_break']['bos_levels']['invalidation_level']:,.2f}

**CHOCH (Change of Character)**:
- 反转水平: ${enhanced_data['key_levels']['structure_break']['choch_levels']['reversal_level']:,.2f}
- 确认水平: ${enhanced_data['key_levels']['structure_break']['choch_levels']['confirmation_level']:,.2f}

**结构强度**: {enhanced_data['key_levels']['structure_break']['structure_strength']:.2f}

### 📅 日内关键位置
**日开盘价**: ${enhanced_data['key_levels']['intraday_levels']['daily_open']:,.2f}

**4小时高低点**:
- 4h高点: ${enhanced_data['key_levels']['intraday_levels']['4h_high_low']['4h_high']:,.2f}
- 4h低点: ${enhanced_data['key_levels']['intraday_levels']['4h_high_low']['4h_low']:,.2f}
- 4h范围: ${enhanced_data['key_levels']['intraday_levels']['4h_high_low']['4h_range']:,.2f}

### 📆 周级别关键位置
**本周开盘价**: ${enhanced_data['key_levels']['weekly_levels']['week_open']:,.2f}

**上周高低点**:
- 上周高点: ${enhanced_data['key_levels']['weekly_levels']['prev_week_high_low']['prev_week_high']:,.2f}
- 上周低点: ${enhanced_data['key_levels']['weekly_levels']['prev_week_high_low']['prev_week_low']:,.2f}

**周枢轴点**:
- 枢轴: ${enhanced_data['key_levels']['weekly_levels']['weekly_pivot']['pivot']:,.2f}
- R1: ${enhanced_data['key_levels']['weekly_levels']['weekly_pivot']['r1']:,.2f}
- S1: ${enhanced_data['key_levels']['weekly_levels']['weekly_pivot']['s1']:,.2f}

## 💡 止损价设置策略（基于OB/FVG关键位置）

### 🎯 高盈亏比止损价设置原则
1. **OB支撑/阻力原则**: 将止损价设置在最近的OB支撑/阻力水平之外
2. **FVG缺口原则**: 利用FVG缺口作为天然止损屏障
3. **结构破坏原则**: 在结构破坏点设置止损，确保趋势确认
4. **日内高低点原则**: 结合4小时高低点设置动态止损

### 📊 关键位置权重
- **OB水平**: 最高权重，提供最强支撑/阻力
- **FVG缺口**: 中等权重，提供天然屏障
- **结构破坏点**: 高权重，确认趋势方向
- **日内高低点**: 中等权重，提供日内参考
- **周级别水平**: 低权重，提供背景参考

### 🚨 爆仓价设置指导
- 爆仓价应设置在关键支撑/阻力水平之外
- 考虑波动率因素，确保足够的缓冲空间
- 结合风险偏好设置合理的爆仓距离

## 🎯 决策要求

请基于以上增强版数据和OB/FVG关键位置分析，提供：

1. **交易决策**: BUY/SELL/WAIT
2. **入场区间**: 具体价格范围
3. **止损位置**: 基于OB/FVG关键位置的高盈亏比设置
4. **爆仓位置**: 结合关键位置和风险偏好的爆仓价
5. **目标价位**: 基于盈利目标
6. **置信度**: 0-1之间的评分
7. **详细分析**: 结合OB/FVG关键位置的解释

**特别强调**: 请重点利用OB/FVG关键位置来设置高盈亏比的止损价和爆仓价！
"""
        
        return prompt
    
    def simulate_enhanced_ai_analysis(self, prompt: str, enhanced_data: Dict[str, Any]) -> Dict[str, Any]:
        """模拟增强版AI分析（包含OB/FVG止损价逻辑）"""
        print("🤖 开始增强版AI分析（OB/FVG止损价设置）...")
        
        current_price = enhanced_data['price_info']['current']
        key_levels = enhanced_data['key_levels']
        
        # 基于OB/FVG关键位置计算止损价
        stop_loss_levels = self._calculate_stop_loss_levels(current_price, key_levels)
        liquidation_levels = self._calculate_liquidation_levels(current_price, key_levels)
        
        # 计算每单风险价值（假设标准仓位为1单位）
        entry_price = current_price
        stop_loss_price = stop_loss_levels['optimal_stop_loss']
        
        # 模拟仓位大小（基于当前价格的合理仓位）
        # 假设仓位大小为1000美元价值的仓位
        position_size = 1000 / entry_price  # 计算对应数量的代币
        
        position_risk = self.calculate_position_risk(entry_price, stop_loss_price, position_size)
        
        # 基于今日交易统计调整决策
        today_stats = self.get_today_trading_stats()
        
        # 根据今日表现调整决策逻辑
        if today_stats['today_win_rate'] > 70:  # 高胜率时更积极
            decision = 'BUY'
            confidence = min(0.90, 0.85 + (today_stats['today_win_rate'] - 70) * 0.01)
        elif today_stats['today_win_rate'] < 30:  # 低胜率时更保守
            decision = 'WAIT'
            confidence = max(0.60, 0.85 - (30 - today_stats['today_win_rate']) * 0.01)
        else:  # 中等胜率保持原策略
            decision = 'BUY'
            confidence = 0.85
        
        # 如果有持仓，考虑仓位管理
        if today_stats['positions']:
            # 持仓较多时更谨慎
            if len(today_stats['positions']) >= 3:
                decision = 'WAIT'
                confidence = 0.70
            elif len(today_stats['positions']) == 2:
                confidence = max(0.75, confidence - 0.05)
        
        # 模拟AI分析结果
        analysis_result = {
            'decision': decision,
            'confidence': confidence,
            'entry_range': {
                'buy': [current_price * 0.998, current_price * 1.002],
                'sell': [current_price * 1.005, current_price * 1.01]
            },
            'stop_loss': stop_loss_levels['optimal_stop_loss'],
            'liquidation': liquidation_levels['optimal_liquidation'],
            'target_price': current_price * 1.03,  # 3%目标
            'stop_loss_analysis': stop_loss_levels,
            'liquidation_analysis': liquidation_levels,
            'position_risk': position_risk,  # 新增：每单风险分析
            'analysis': f"""
📊 **基于OB/FVG关键位置的增强分析**

**市场状态**: 检测到强OB/FVG结构，提供高盈亏比交易机会。

**关键位置分析**:
1. **OB强度**: {key_levels['ob_levels']['ob_strength']:.2f} - 提供可靠支撑/阻力
2. **FVG强度**: {key_levels['fvg_levels']['fvg_strength']:.2f} - 缺口提供天然屏障
3. **结构强度**: {key_levels['structure_break']['structure_strength']:.2f} - 趋势确认度高

**止损价设置策略**:
- **OB支撑**: ${stop_loss_levels['ob_based_stop_loss']:,.2f}
- **FVG缺口**: ${stop_loss_levels['fvg_based_stop_loss']:,.2f}
- **结构失效**: ${stop_loss_levels['structure_based_stop_loss']:,.2f}
- **最优止损**: ${stop_loss_levels['optimal_stop_loss']:,.2f}

**爆仓价设置策略**:
- **关键支撑**: ${liquidation_levels['key_support_liquidation']:,.2f}
- **波动缓冲**: ${liquidation_levels['volatility_liquidation']:,.2f}
- **最优爆仓**: ${liquidation_levels['optimal_liquidation']:,.2f}

💸 **每单风险价值分析**:
{position_risk['analysis']}

💡 **交易优势**:
- 基于OB/FVG关键位置设置止损，提高盈亏比
- 结合结构破坏点确认趋势方向
- 日内高低点提供动态止损参考
- 风险回报比显著优于传统方法

🎯 **建议**: 基于强OB/FVG信号，建议入场交易
"""
        }
        
        print("✅ 增强版AI分析完成")
        return analysis_result
    
    def _calculate_stop_loss_levels(self, current_price: float, key_levels: Dict[str, Any]) -> Dict[str, float]:
        """计算基于OB/FVG的止损价水平"""
        
        # 基于OB的止损价
        ob_based_stop_loss = min(
            key_levels['ob_levels']['bullish_ob']['support'] * 0.995,
            key_levels['ob_levels']['bearish_ob']['support'] * 0.995
        )
        
        # 基于FVG的止损价
        fvg_based_stop_loss = min(
            key_levels['fvg_levels']['bullish_fvg']['gap_bottom'] * 0.998,
            key_levels['fvg_levels']['bearish_fvg']['gap_bottom'] * 0.998
        )
        
        # 基于结构破坏的止损价
        structure_based_stop_loss = min(
            key_levels['structure_break']['bos_levels']['invalidation_level'] * 0.997,
            key_levels['structure_break']['choch_levels']['confirmation_level'] * 0.997
        )
        
        # 基于日内高低点的止损价
        intraday_based_stop_loss = key_levels['intraday_levels']['4h_high_low']['4h_low'] * 0.995
        
        # 最优止损价（取最严格的值）
        optimal_stop_loss = min(
            ob_based_stop_loss,
            fvg_based_stop_loss,
            structure_based_stop_loss,
            intraday_based_stop_loss,
            current_price * 0.98  # 最小2%止损
        )
        
        return {
            'ob_based_stop_loss': ob_based_stop_loss,
            'fvg_based_stop_loss': fvg_based_stop_loss,
            'structure_based_stop_loss': structure_based_stop_loss,
            'intraday_based_stop_loss': intraday_based_stop_loss,
            'optimal_stop_loss': optimal_stop_loss,
            'stop_loss_distance_pct': (current_price - optimal_stop_loss) / current_price * 100
        }
    
    def _calculate_liquidation_levels(self, current_price: float, key_levels: Dict[str, Any]) -> Dict[str, float]:
        """计算基于关键位置的爆仓价水平"""
        
        # 基于关键支撑的爆仓价
        key_support_liquidation = min(
            key_levels['ob_levels']['bullish_ob']['support'] * 0.95,
            key_levels['intraday_levels']['4h_high_low']['4h_low'] * 0.93,
            key_levels['weekly_levels']['prev_week_high_low']['prev_week_low'] * 0.90
        )
        
        # 基于波动率的爆仓价
        volatility_liquidation = current_price * 0.85  # 15%波动缓冲
        
        # 最优爆仓价
        optimal_liquidation = min(key_support_liquidation, volatility_liquidation)
        
        return {
            'key_support_liquidation': key_support_liquidation,
            'volatility_liquidation': volatility_liquidation,
            'optimal_liquidation': optimal_liquidation,
            'liquidation_distance_pct': (current_price - optimal_liquidation) / current_price * 100
        }
    
    def calculate_position_risk(self, entry_price: float, stop_loss: float, position_size: float = None) -> Dict[str, Any]:
        """
        计算每单的风险价值
        
        Args:
            entry_price: 入场价格
            stop_loss: 止损价格
            position_size: 仓位大小（如果为None，则计算每单位风险）
        
        Returns:
            风险分析结果
        """
        
        # 计算止损距离（百分比）
        stop_loss_distance_pct = (entry_price - stop_loss) / entry_price * 100
        
        # 计算每单位风险价值
        risk_per_unit = entry_price - stop_loss
        
        # 如果提供了仓位大小，计算总风险
        if position_size is not None:
            # 精确计算公式：总风险 = (开单数量 * 开单价) - (开单数量 * 止损价)
            total_risk = (position_size * entry_price) - (position_size * stop_loss)
            # 验证公式正确性：total_risk = position_size * (entry_price - stop_loss)
            risk_percentage = (total_risk / (entry_price * position_size)) * 100
        else:
            total_risk = None
            risk_percentage = stop_loss_distance_pct
        
        # 风险等级评估
        if stop_loss_distance_pct <= 1:
            risk_level = "低风险"
            risk_color = "🟢"
        elif stop_loss_distance_pct <= 3:
            risk_level = "中等风险"
            risk_color = "🟡"
        elif stop_loss_distance_pct <= 5:
            risk_level = "高风险"
            risk_color = "🟠"
        else:
            risk_level = "极高风险"
            risk_color = "🔴"
        
        # 风险回报比评估（假设目标盈利为3%）
        target_profit_pct = 3.0
        risk_reward_ratio = target_profit_pct / stop_loss_distance_pct if stop_loss_distance_pct > 0 else float('inf')
        
        if risk_reward_ratio >= 3:
            rr_rating = "优秀"
            rr_color = "🟢"
        elif risk_reward_ratio >= 2:
            rr_rating = "良好"
            rr_color = "🟡"
        elif risk_reward_ratio >= 1:
            rr_rating = "一般"
            rr_color = "🟠"
        else:
            rr_rating = "较差"
            rr_color = "🔴"
        
        return {
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'stop_loss_distance_pct': stop_loss_distance_pct,
            'risk_per_unit': risk_per_unit,
            'position_size': position_size,
            'total_risk': total_risk,
            'risk_percentage': risk_percentage,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'target_profit_pct': target_profit_pct,
            'risk_reward_ratio': risk_reward_ratio,
            'rr_rating': rr_rating,
            'rr_color': rr_color,
            'analysis': f"""
📊 **每单风险分析**

**入场价格**: ${entry_price:,.2f}
**止损价格**: ${stop_loss:,.2f}
**止损距离**: {stop_loss_distance_pct:.2f}%

💸 **风险价值**:
- **每单位风险**: ${risk_per_unit:,.2f}
{'' if position_size is None else f'- **仓位大小**: {position_size} 单位'}
{'' if total_risk is None else f'- **总风险价值**: ${total_risk:,.2f}'}
- **风险占比**: {risk_percentage:.2f}%

🎯 **风险评估**:
- **风险等级**: {risk_color} {risk_level}
- **风险回报比**: {rr_color} {risk_reward_ratio:.2f}:1 ({rr_rating})
- **目标盈利**: {target_profit_pct}%

💡 **建议**:
{'- 风险控制良好，建议入场' if risk_reward_ratio >= 2 else '- 风险回报比一般，谨慎入场' if risk_reward_ratio >= 1 else '- 风险回报比较差，不建议入场'}
"""
        }
    
    def analyze_single_symbol(self, symbol: str) -> Dict[str, Any]:
        """分析单个品种（包含OB/FVG止损价设置）"""
        print(f"\n🎯 开始分析 {symbol}...")
        
        # 提取增强版市场数据
        enhanced_data = self.extract_enhanced_market_data(symbol)
        
        # 生成增强版AI提示词
        prompt = self.generate_enhanced_ai_prompt(enhanced_data)
        
        # 模拟AI分析
        analysis_result = self.simulate_enhanced_ai_analysis(prompt, enhanced_data)
        
        # 整合结果
        result = {
            'symbol': symbol,
            'timestamp': time.time(),
            'enhanced_data': enhanced_data,
            'ai_prompt': prompt,
            'analysis_result': analysis_result,
            'trading_style': self.trading_style,
            'risk_preference': self.risk_preference
        }
        
        # 显示今日交易统计
        today_stats = self.get_today_trading_stats()
        print(f"📅 **今日交易统计**")
        print(f"   初始金额: ${today_stats['initial_capital']:,.2f}")
        print(f"   当前资金: ${today_stats['current_capital']:,.2f}")
        print(f"   今日盈亏: ${today_stats['today_pnl']:,.2f}")
        print(f"   今日胜率: {today_stats['today_win_rate']:.1f}%")
        print(f"   交易次数: {today_stats['today_trades']} (胜/败: {today_stats['today_wins']}/{today_stats['today_losses']})")
        print(f"   持仓数量: {len(today_stats['positions'])}")
        
        print(f"✅ {symbol} 分析完成")
        print(f"   决策: {analysis_result['decision']}")
        print(f"   置信度: {analysis_result['confidence']:.2f}")
        print(f"   最优止损: ${analysis_result['stop_loss']:,.2f}")
        print(f"   最优爆仓: ${analysis_result['liquidation']:,.2f}")
        
        # 显示仓位大小信息
        position_size = analysis_result['position_risk']['position_size']
        entry_price = analysis_result['position_risk']['entry_price']
        stop_loss_price = analysis_result['position_risk']['stop_loss']
        print(f"   📦 仓位大小: {position_size:.4f} 单位")
        print(f"   💰 仓位价值: ${position_size * entry_price:,.2f}")
        
        # 显示风险分析结果（使用精确公式计算）
        risk_info = analysis_result['position_risk']
        print(f"   💸 每单风险金额: ${risk_info['total_risk']:,.2f}")
        print(f"   📊 止损距离: {risk_info['stop_loss_distance_pct']:.2f}%")
        print(f"   🎯 风险等级: {risk_info['risk_color']} {risk_info['risk_level']}")
        print(f"   📈 风险回报比: {risk_info['rr_color']} {risk_info['risk_reward_ratio']:.2f}:1")
        
        # 显示精确计算公式验证
        print(f"   🔢 公式验证: ({position_size:.4f} × ${entry_price:,.2f}) - ({position_size:.4f} × ${stop_loss_price:,.2f}) = ${risk_info['total_risk']:,.2f}")
        
        return result
    
    def analyze_multiple_symbols(self, symbols: List[str] = None) -> Dict[str, Any]:
        """分析多个品种"""
        if symbols is None:
            symbols = self.config.get('symbols', ['BTC/USD'])
        
        print(f"\n🚀 开始多品种OB/FVG增强分析...")
        print(f"   分析品种: {', '.join(symbols)}")
        print(f"   🔥 集成OB/FVG关键位置止损价设置")
        
        start_time = time.time()
        results = {}
        
        for symbol in symbols:
            results[symbol] = self.analyze_single_symbol(symbol)
        
        end_time = time.time()
        
        # 汇总结果
        summary = {
            'total_symbols': len(symbols),
            'analysis_time': end_time - start_time,
            'timestamp': time.time(),
            'results': results,
            'config': {
                'trading_style': self.trading_style,
                'risk_preference': self.risk_preference,
                'max_timeframe': '4h'
            }
        }
        
        print(f"\n✅ 多品种分析完成")
        print(f"   分析品种数: {len(symbols)}")
        print(f"   总耗时: {summary['analysis_time']:.3f}秒")
        print(f"   🔥 OB/FVG关键位置止损价设置已集成")
        
        return summary
    
    def save_analysis_results(self, analysis_results: Dict[str, Any], filename: str = None) -> str:
        """保存分析结果到JSON文件"""
        if filename is None:
            timestamp = int(time.time())
            filename = f"enhanced_obfvg_analysis_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        
        print(f"📁 分析结果已保存到: {filename}")
        return filename

def main():
    """主函数"""
    print("🎯 增强版OB/FVG止损价AI交易系统启动")
    print("🔥 重新集成OB/FVG分析用于高盈亏比止损价设置")
    
    # 初始化交易系统
    trader = OBFVGAITrader('trading_config.json')
    
    # 分析多个品种
    symbols = trader.config.get('symbols', ['BTC/USD', 'ETH/USD', 'SOL/USD'])
    analysis_results = trader.analyze_multiple_symbols(symbols)
    
    # 保存结果
    filename = trader.save_analysis_results(analysis_results)
    
    print(f"\n🎉 增强版OB/FVG止损价AI交易系统运行完成")
    print(f"   结果文件: {filename}")
    print(f"   🔥 已成功集成OB/FVG关键位置止损价设置")
    print(f"   📍 关键位置: OB/FVG/结构破坏/日开盘价/4h高低点/本周开盘价/上周高低点")

if __name__ == "__main__":
    main()