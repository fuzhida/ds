"""
回测模块 - 支持使用历史数据回测交易策略
"""

import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging

# 导入自定义模块
from config import Config
from data_manager import DataManager
from smc_analyzer import SMCDetector, MTFAnalyzer
from technical_analyzer import TechnicalIndicatorCalculator, KeyLevelsCalculator, PriceActionAnalyzer
from ai_signal_generator import AISignalGenerator
from risk_manager import RiskManager, PositionManager
from trading_executor import TradingExecutor


class BacktestEngine:
    """回测引擎"""
    
    def __init__(self, config_path: str = None):
        """初始化回测引擎"""
        # 加载配置
        self.config = Config(config_path)
        
        # 设置日志
        self.logger = logging.getLogger("BacktestEngine")
        self.logger.setLevel(logging.INFO)
        
        # 初始化组件
        self._initialize_components()
        
        # 回测状态
        self.initial_balance = 10000.0  # 初始资金
        self.current_balance = self.initial_balance
        self.positions = []  # 持仓列表
        self.trades = []  # 交易记录
        self.equity_curve = []  # 资金曲线
        self.performance_metrics = {}  # 性能指标
        
        self.logger.info("回测引擎初始化完成")
    
    def _initialize_components(self):
        """初始化组件"""
        try:
            # 初始化数据管理器
            self.data_manager = DataManager(self.config, self.logger)
            
            # 初始化SMC分析器
            self.smc_detector = SMCDetector(self.config, self.logger)
            self.mtf_analyzer = MTFAnalyzer(self.config, self.logger)
            
            # 初始化技术指标分析器
            self.technical_calculator = TechnicalIndicatorCalculator(self.config, self.logger)
            self.key_levels_calculator = KeyLevelsCalculator(self.config, self.logger)
            self.price_action_analyzer = PriceActionAnalyzer(self.config, self.logger)
            
            # 初始化AI信号生成器
            self.ai_signal_generator = AISignalGenerator(self.config, self.logger)
            
            # 初始化风险管理器
            self.risk_manager = RiskManager(self.config, self.logger)
            
            # 初始化仓位管理器
            self.position_manager = PositionManager(self.config, self.logger, self.risk_manager)
            
            self.logger.info("所有组件初始化成功")
            
        except Exception as e:
            self.logger.error(f"组件初始化失败: {e}")
            raise
    
    def run_backtest(self, symbol: str, timeframes: List[str], start_date: str, end_date: str, 
                    initial_balance: float = 10000.0) -> Dict[str, Any]:
        """
        运行回测
        :param symbol: 交易对
        :param timeframes: 时间框架列表
        :param start_date: 开始日期 (YYYY-MM-DD)
        :param end_date: 结束日期 (YYYY-MM-DD)
        :param initial_balance: 初始资金
        :return: 回测结果
        """
        try:
            self.logger.info(f"开始回测 {symbol} 从 {start_date} 到 {end_date}")
            
            # 设置初始资金
            self.initial_balance = initial_balance
            self.current_balance = initial_balance
            
            # 重置状态
            self.positions = []
            self.trades = []
            self.equity_curve = []
            
            # 获取历史数据
            historical_data = self._get_historical_data(symbol, timeframes, start_date, end_date)
            
            if not historical_data:
                raise ValueError("无法获取历史数据")
            
            # 获取主要时间框架数据
            main_tf = timeframes[0]
            data = historical_data[main_tf]
            
            # 逐个K线进行回测
            for i in range(len(data)):
                current_time = data.index[i]
                current_price = data['close'].iloc[i]
                
                # 获取当前时间点的数据切片
                data_slice = self._get_data_slice(historical_data, timeframes, i)
                
                # 分析市场
                analysis_result = self._analyze_market(data_slice, current_price)
                
                # 生成交易信号
                signal = self._generate_signal(analysis_result)
                
                # 执行交易
                self._execute_trade_logic(symbol, signal, current_price, current_time)
                
                # 更新持仓
                self._update_positions(symbol, current_price, current_time)
                
                # 记录资金曲线
                total_equity = self._calculate_total_equity(current_price)
                self.equity_curve.append({
                    'timestamp': current_time,
                    'balance': self.current_balance,
                    'equity': total_equity,
                    'price': current_price
                })
            
            # 计算性能指标
            self._calculate_performance_metrics()
            
            # 生成回测报告
            report = self._generate_report(symbol, start_date, end_date)
            
            self.logger.info("回测完成")
            return report
            
        except Exception as e:
            self.logger.error(f"回测失败: {e}")
            raise
    
    def _get_historical_data(self, symbol: str, timeframes: List[str], 
                            start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """获取历史数据"""
        try:
            # 转换日期为时间戳
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            
            # 计算数据点数量
            main_tf_minutes = {
                '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
                '1h': 60, '4h': 240, '1d': 1440
            }.get(timeframes[0], 15)
            
            total_minutes = (end_dt - start_dt).total_seconds() / 60
            limit = int(total_minutes / main_tf_minutes) + 100  # 额外获取一些数据
            
            # 获取数据
            data = {}
            for tf in timeframes:
                try:
                    # 尝试从真实交易所获取历史数据
                    market_data = self.data_manager.get_market_data(symbol, [tf], limit)
                    
                    # 从返回的字典中提取DataFrame
                    data[tf] = market_data.get(tf, pd.DataFrame())
                    
                    # 如果获取的数据为空或失败，使用模拟数据
                    if data[tf] is None or data[tf].empty:
                        self.logger.warning(f"无法获取{symbol} {tf}的真实数据，使用模拟数据")
                        data[tf] = self._generate_historical_ohlcv_data(tf, limit, start_dt, end_dt)
                    else:
                        self.logger.info(f"成功获取{symbol} {tf}的历史数据，共{len(data[tf])}条")
                        
                except Exception as e:
                    self.logger.warning(f"获取{symbol} {tf}历史数据失败: {e}，使用模拟数据")
                    data[tf] = self._generate_historical_ohlcv_data(tf, limit, start_dt, end_dt)
            
            return data
            
        except Exception as e:
            self.logger.error(f"获取历史数据失败: {e}")
            return {}
    
    def _generate_historical_ohlcv_data(self, timeframe: str, limit: int, 
                                      start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        """生成历史OHLCV数据"""
        try:
            # 根据时间框架确定时间间隔
            tf_minutes = {
                '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
                '1h': 60, '4h': 240, '1d': 1440
            }.get(timeframe, 15)
            
            # 生成时间序列
            times = []
            current_time = start_dt
            while current_time <= end_dt and len(times) < limit:
                times.append(current_time)
                current_time += timedelta(minutes=tf_minutes)
            
            # 生成价格数据
            base_price = 2000.0  # 基础价格
            prices = [base_price]
            
            for i in range(1, len(times)):
                # 随机波动
                change = np.random.normal(0, 0.002)  # 0.2%标准差
                new_price = prices[-1] * (1 + change)
                prices.append(new_price)
            
            # 生成OHLCV数据
            data = []
            for i, (time, price) in enumerate(zip(times, prices)):
                # 随机生成高低价
                high = price * (1 + abs(np.random.normal(0, 0.001)))
                low = price * (1 - abs(np.random.normal(0, 0.001)))
                
                # 确保高低价关系正确
                high = max(high, price)
                low = min(low, price)
                
                # 生成开盘价（第一根K线的开盘价等于收盘价）
                if i == 0:
                    open_price = price
                else:
                    # 上一根K线的收盘价
                    open_price = prices[i-1]
                
                # 生成成交量
                volume = np.random.normal(1000000, 200000)
                volume = max(volume, 100000)  # 最小成交量
                
                data.append({
                    'time': time,
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': price,
                    'volume': volume
                })
            
            # 转换为DataFrame
            df = pd.DataFrame(data)
            df.set_index('time', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"生成历史数据失败 {timeframe}: {e}")
            return pd.DataFrame()
    
    def _get_data_slice(self, historical_data: Dict[str, pd.DataFrame], 
                       timeframes: List[str], index: int) -> Dict[str, pd.DataFrame]:
        """获取当前时间点的数据切片"""
        try:
            data_slice = {}
            for tf in timeframes:
                if tf in historical_data and index < len(historical_data[tf]):
                    # 获取到当前时间点的数据
                    data_slice[tf] = historical_data[tf].iloc[:index+1]
            
            return data_slice
            
        except Exception as e:
            self.logger.error(f"获取数据切片失败: {e}")
            return {}
    
    def _analyze_market(self, data_slice: Dict[str, pd.DataFrame], current_price: float) -> Dict[str, Any]:
        """分析市场"""
        try:
            if not data_slice:
                return {}
            
            # 获取主要时间框架数据
            main_tf = list(data_slice.keys())[0]
            data = data_slice[main_tf]
            
            # 技术指标分析
            technical_indicators = self.technical_calculator.calculate_technical_indicators(data)
            
            # 关键水平分析
            key_levels = self.key_levels_calculator.calculate_key_levels({main_tf: data})
            
            # 价格行为分析
            price_action = self.price_action_analyzer.analyze_price_action(data)
            
            # SMC结构分析
            smc_structures = self.smc_detector.detect_smc_structures(data, main_tf)
            
            # 多时间框架分析
            mtf_analysis = self.mtf_analyzer.analyze_mtf_structures(data_slice)
            
            # 计算市场条件
            volatility = price_action.get("volatility", 0.5)
            trend_strength = technical_indicators.get("trend_strength", 0.5)
            
            return {
                "technical_indicators": technical_indicators,
                "key_levels": key_levels,
                "price_action": price_action,
                "smc_structures": smc_structures,
                "mtf_analysis": mtf_analysis,
                "volatility": volatility,
                "trend_strength": trend_strength,
                "current_price": current_price,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"市场分析失败: {e}")
            return {}
    
    def _generate_signal(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """生成交易信号"""
        try:
            if not analysis_result:
                return {"signal": "HOLD", "confidence": 0.1, "reasoning": "分析结果为空"}
            
            # 准备AI分析的市场数据
            ai_market_data = {
                "current_price": analysis_result.get("current_price", 0),
                "technical_indicators": analysis_result.get("technical_indicators", {}),
                "smc_structures": analysis_result.get("smc_structures", {}),
                "key_levels": analysis_result.get("key_levels", {}),
                "price_action": analysis_result.get("price_action", {})
            }
            
            # 生成AI信号
            ai_signals = self.ai_signal_generator.generate_signals(ai_market_data)
            
            # 添加调试日志
            self.logger.debug(f"AI市场数据: {ai_market_data}")
            self.logger.debug(f"AI信号: {ai_signals}")
            
            # 获取主要AI信号
            primary_ai_signal = ai_signals.get("primary", {})
            ai_signal = primary_ai_signal.get("signal", {})
            
            # 获取共识信号
            consensus_signal = ai_signals.get("consensus", "HOLD")
            consensus_confidence = ai_signals.get("consensus_confidence", 0.1)
            
            # 技术信号验证
            technical_signal = self._validate_technical_signal(analysis_result)
            
            # SMC信号验证
            smc_signal = self._validate_smc_signal(analysis_result)
            
            # 获取OB叠加结果
            ob_overlay_result = None
            smc_structures = analysis_result.get("smc_structures", {})
            if "ob_fvg_optimized" in smc_structures:
                ob_fvg_optimized = smc_structures["ob_fvg_optimized"]
                if "overlay_result" in ob_fvg_optimized:
                    ob_overlay_result = ob_fvg_optimized["overlay_result"]
            
            # 综合信号
            final_signal = self._combine_signals(
                ai_signal, consensus_signal, consensus_confidence,
                technical_signal, smc_signal, ob_overlay_result
            )
            
            return final_signal
            
        except Exception as e:
            self.logger.error(f"信号生成失败: {e}")
            return {"signal": "HOLD", "confidence": 0.1, "reasoning": f"信号生成失败: {str(e)}"}
    
    def _validate_technical_signal(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """技术信号验证"""
        try:
            technical_indicators = analysis_result.get("technical_indicators", {})
            
            # 获取技术评分
            technical_score = technical_indicators.get("overall_score", 0.5)
            
            # 确定技术信号
            if technical_score > 0.6:
                tech_signal = "BUY"
            elif technical_score < 0.4:
                tech_signal = "SELL"
            else:
                tech_signal = "HOLD"
            
            return {
                "signal": tech_signal,
                "confidence": abs(technical_score - 0.5) * 2,
                "score": technical_score
            }
        
        except Exception as e:
            self.logger.error(f"技术信号验证失败: {e}")
            return {"signal": "HOLD", "confidence": 0.1}
    
    def _validate_smc_signal(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """SMC信号验证"""
        try:
            smc_structures = analysis_result.get("smc_structures", {})
            
            # 获取SMC评分
            smc_score = smc_structures.get("overall_score", 0.5)
            
            # 确定SMC信号
            if smc_score > 0.6:
                smc_signal = "BUY"
            elif smc_score < 0.4:
                smc_signal = "SELL"
            else:
                smc_signal = "HOLD"
            
            return {
                "signal": smc_signal,
                "confidence": abs(smc_score - 0.5) * 2,
                "score": smc_score
            }
        
        except Exception as e:
            self.logger.error(f"SMC信号验证失败: {e}")
            return {"signal": "HOLD", "confidence": 0.1}
    
    def _combine_signals(self, ai_signal: Dict[str, Any], consensus_signal: str, 
                         consensus_confidence: float, technical_signal: Dict[str, Any], 
                         smc_signal: Dict[str, Any], ob_overlay_result: Dict[str, Any] = None) -> Dict[str, Any]:
        """组合信号"""
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
            self.logger.error(f"信号组合失败: {e}")
            return {"signal": "HOLD", "confidence": 0.1, "reasoning": f"信号组合失败: {str(e)}"}
    
    def _execute_trade_logic(self, symbol: str, signal: Dict[str, Any], 
                            current_price: float, current_time: datetime):
        """执行交易逻辑"""
        try:
            signal_type = signal.get("signal", "HOLD")
            confidence = signal.get("confidence", 0.0)
            
            # 检查信号强度
            if signal_type == "HOLD" or confidence < 0.4:  # 降低阈值从0.6到0.4
                return
            
            # 检查是否已有持仓
            has_position = any(p["symbol"] == symbol and p["status"] == "open" for p in self.positions)
            
            # 如果已有持仓，不重复开仓
            if has_position:
                return
            
            # 计算仓位大小
            risk_amount = self.current_balance * 0.02  # 风险2%
            position_result = self.risk_manager.calculate_position_size(
                signal, self.current_balance, current_price, current_price * 0.02  # 2%止损作为ATR
            )
            position_size = position_result.get("position_size", 0)
            
            # 计算止损和止盈
            ob_overlay_result = signal.get("ob_overlay_result")
            stop_loss = self._calculate_stop_loss_with_ob(current_price, signal_type, ob_overlay_result)
            take_profit = self._calculate_take_profit(current_price, signal_type, stop_loss)
            
            # 创建持仓
            position = {
                "id": len(self.positions) + 1,
                "symbol": symbol,
                "type": signal_type,
                "size": position_size,
                "entry_price": current_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "entry_time": current_time,
                "status": "open",
                "unrealized_pnl": 0.0,
                "realized_pnl": 0.0
            }
            
            self.positions.append(position)
            
            # 记录交易
            trade = {
                "id": len(self.trades) + 1,
                "symbol": symbol,
                "type": signal_type,
                "size": position_size,
                "entry_price": current_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "entry_time": current_time,
                "status": "open"
            }
            
            self.trades.append(trade)
            
            self.logger.info(f"开仓 {signal_type} {symbol} @ {current_price}, 数量: {position_size}, 止损: {stop_loss}, 止盈: {take_profit}")
            
        except Exception as e:
            self.logger.error(f"执行交易逻辑失败: {e}")
    
    def _calculate_stop_loss_with_ob(self, current_price: float, signal_type: str, 
                                    ob_overlay_result: Dict[str, Any] = None) -> float:
        """使用OB叠加结果计算止损"""
        try:
            # 默认止损距离
            default_stop_distance = current_price * 0.02  # 2%
            
            # 如果有OB叠加结果，使用OB边界计算止损
            if ob_overlay_result:
                wide_ob = ob_overlay_result.get('wide_ob_for_stop_loss')
                if wide_ob:
                    if signal_type == "BUY":
                        # 多单止损设于较宽OB下边界
                        stop_loss = wide_ob.get('low')
                        # 确保止损距离合理
                        if current_price - stop_loss > default_stop_distance * 3:
                            stop_loss = current_price - default_stop_distance
                    else:  # SELL
                        # 空单止损设于较宽OB上边界
                        stop_loss = wide_ob.get('high')
                        # 确保止损距离合理
                        if stop_loss - current_price > default_stop_distance * 3:
                            stop_loss = current_price + default_stop_distance
                    
                    return stop_loss
            
            # 默认止损计算
            if signal_type == "BUY":
                return current_price - default_stop_distance
            else:  # SELL
                return current_price + default_stop_distance
                
        except Exception as e:
            self.logger.error(f"计算止损失败: {e}")
            # 返回默认止损
            if signal_type == "BUY":
                return current_price * 0.98
            else:
                return current_price * 1.02
    
    def _calculate_take_profit(self, entry_price: float, signal_type: str, 
                               stop_loss: float) -> float:
        """计算止盈价格"""
        try:
            # 风险回报比
            risk_reward_ratio = 2.0
            
            if signal_type == "BUY":
                risk = entry_price - stop_loss
                return entry_price + risk * risk_reward_ratio
            else:  # SELL
                risk = stop_loss - entry_price
                return entry_price - risk * risk_reward_ratio
                
        except Exception as e:
            self.logger.error(f"计算止盈失败: {e}")
            # 返回默认止盈
            if signal_type == "BUY":
                return entry_price * 1.04
            else:
                return entry_price * 0.96
    
    def _update_positions(self, symbol: str, current_price: float, current_time: datetime):
        """更新持仓状态"""
        try:
            for position in self.positions:
                if position["symbol"] != symbol or position["status"] != "open":
                    continue
                
                # 计算未实现盈亏
                if position["type"] == "BUY":
                    unrealized_pnl = (current_price - position["entry_price"]) * position["size"]
                else:  # SELL
                    unrealized_pnl = (position["entry_price"] - current_price) * position["size"]
                
                position["unrealized_pnl"] = unrealized_pnl
                
                # 检查止损
                if position["type"] == "BUY" and current_price <= position["stop_loss"]:
                    self._close_position(position, current_price, current_time, "止损")
                elif position["type"] == "SELL" and current_price >= position["stop_loss"]:
                    self._close_position(position, current_price, current_time, "止损")
                
                # 检查止盈
                elif position["type"] == "BUY" and current_price >= position["take_profit"]:
                    self._close_position(position, current_price, current_time, "止盈")
                elif position["type"] == "SELL" and current_price <= position["take_profit"]:
                    self._close_position(position, current_price, current_time, "止盈")
            
        except Exception as e:
            self.logger.error(f"更新持仓失败: {e}")
    
    def _close_position(self, position: Dict[str, Any], exit_price: float, 
                       exit_time: datetime, reason: str):
        """平仓"""
        try:
            # 计算已实现盈亏
            if position["type"] == "BUY":
                realized_pnl = (exit_price - position["entry_price"]) * position["size"]
            else:  # SELL
                realized_pnl = (position["entry_price"] - exit_price) * position["size"]
            
            # 更新持仓状态
            position["status"] = "closed"
            position["exit_price"] = exit_price
            position["exit_time"] = exit_time
            position["exit_reason"] = reason
            position["realized_pnl"] = realized_pnl
            
            # 更新账户余额
            self.current_balance += realized_pnl
            
            # 更新交易记录
            for trade in self.trades:
                if trade["id"] == position["id"] and trade["status"] == "open":
                    trade["status"] = "closed"
                    trade["exit_price"] = exit_price
                    trade["exit_time"] = exit_time
                    trade["exit_reason"] = reason
                    trade["pnl"] = realized_pnl
                    break
            
            self.logger.info(f"平仓 {position['type']} {position['symbol']} @ {exit_price}, 盈亏: {realized_pnl:.2f}, 原因: {reason}")
            
        except Exception as e:
            self.logger.error(f"平仓失败: {e}")
    
    def _calculate_total_equity(self, current_price: float) -> float:
        """计算总权益"""
        try:
            total_equity = self.current_balance
            
            # 加上未实现盈亏
            for position in self.positions:
                if position["status"] == "open":
                    total_equity += position["unrealized_pnl"]
            
            return total_equity
            
        except Exception as e:
            self.logger.error(f"计算总权益失败: {e}")
            return self.current_balance
    
    def _calculate_performance_metrics(self):
        """计算性能指标"""
        try:
            # 获取已平仓交易
            closed_trades = [t for t in self.trades if t["status"] == "closed"]
            
            if not closed_trades:
                self.performance_metrics = {
                    "total_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "win_rate": 0.0,
                    "total_pnl": 0.0,
                    "total_return": 0.0,
                    "max_drawdown": 0.0,
                    "sharpe_ratio": 0.0,
                    "profit_factor": 0.0
                }
                return
            
            # 计算基本指标
            total_trades = len(closed_trades)
            winning_trades = len([t for t in closed_trades if t.get("pnl", 0) > 0])
            losing_trades = len([t for t in closed_trades if t.get("pnl", 0) < 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            
            # 计算盈亏指标
            total_pnl = sum(t.get("pnl", 0) for t in closed_trades)
            total_return = (self.current_balance - self.initial_balance) / self.initial_balance
            
            # 计算最大回撤
            max_drawdown = 0.0
            peak_equity = self.initial_balance
            
            for equity_point in self.equity_curve:
                if equity_point["equity"] > peak_equity:
                    peak_equity = equity_point["equity"]
                
                drawdown = (peak_equity - equity_point["equity"]) / peak_equity
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            # 计算夏普比率
            returns = []
            for i in range(1, len(self.equity_curve)):
                prev_equity = self.equity_curve[i-1]["equity"]
                curr_equity = self.equity_curve[i]["equity"]
                if prev_equity > 0:
                    returns.append((curr_equity - prev_equity) / prev_equity)
            
            sharpe_ratio = 0.0
            if returns:
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                if std_return > 0:
                    sharpe_ratio = avg_return / std_return * np.sqrt(252)  # 年化
            
            # 计算盈利因子
            gross_profit = sum(t.get("pnl", 0) for t in closed_trades if t.get("pnl", 0) > 0)
            gross_loss = abs(sum(t.get("pnl", 0) for t in closed_trades if t.get("pnl", 0) < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            self.performance_metrics = {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": win_rate,
                "total_pnl": total_pnl,
                "total_return": total_return,
                "max_drawdown": max_drawdown,
                "sharpe_ratio": sharpe_ratio,
                "profit_factor": profit_factor
            }
            
        except Exception as e:
            self.logger.error(f"计算性能指标失败: {e}")
    
    def _generate_report(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """生成回测报告"""
        try:
            # 替换符号中的斜杠，避免文件路径问题
            safe_symbol = symbol.replace('/', '_')
            
            report = {
                "symbol": symbol,
                "start_date": start_date,
                "end_date": end_date,
                "initial_balance": self.initial_balance,
                "final_balance": self.current_balance,
                "performance_metrics": self.performance_metrics,
                "trades": self.trades,
                "equity_curve": self.equity_curve
            }
            
            # 保存报告
            report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_reports")
            os.makedirs(report_dir, exist_ok=True)
            
            report_file = os.path.join(report_dir, f"backtest_{safe_symbol}_{start_date}_{end_date}.json")
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"回测报告已保存到: {report_file}")
            
            # 生成图表
            self._generate_charts(safe_symbol, start_date, end_date)
            
            return report
            
        except Exception as e:
            self.logger.error(f"生成回测报告失败: {e}")
            # 即使保存失败，也返回基本报告信息
            return {
                "symbol": symbol,
                "start_date": start_date,
                "end_date": end_date,
                "initial_balance": self.initial_balance,
                "final_balance": self.current_balance,
                "performance_metrics": self.performance_metrics,
                "trades": self.trades,
                "equity_curve": self.equity_curve,
                "error": str(e)
            }
    
    def _generate_charts(self, symbol: str, start_date: str, end_date: str):
        """生成回测图表"""
        try:
            if not self.equity_curve:
                return
            
            # 替换符号中的斜杠，避免文件路径问题
            safe_symbol = symbol.replace('/', '_')
            
            # 创建图表
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # 转换时间戳
            timestamps = [pd.to_datetime(point["timestamp"]) for point in self.equity_curve]
            equity_values = [point["equity"] for point in self.equity_curve]
            price_values = [point["price"] for point in self.equity_curve]
            
            # 绘制资金曲线
            ax1.plot(timestamps, equity_values, label='权益曲线', color='blue')
            ax1.set_title(f'{symbol} 回测权益曲线 ({start_date} 至 {end_date})')
            ax1.set_ylabel('账户权益')
            ax1.grid(True)
            ax1.legend()
            
            # 绘制价格曲线
            ax2.plot(timestamps, price_values, label='价格曲线', color='green')
            ax2.set_title(f'{symbol} 价格曲线')
            ax2.set_ylabel('价格')
            ax2.set_xlabel('时间')
            ax2.grid(True)
            ax2.legend()
            
            # 格式化x轴
            for ax in [ax1, ax2]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.xaxis.set_major_locator(mdates.MonthLocator())
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            
            # 保存图表
            chart_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_charts")
            os.makedirs(chart_dir, exist_ok=True)
            
            chart_file = os.path.join(chart_dir, f"backtest_{safe_symbol}_{start_date}_{end_date}.png")
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"回测图表已保存到: {chart_file}")
            
        except Exception as e:
            self.logger.error(f"生成回测图表失败: {e}")


def run_backtest_example():
    """运行回测示例"""
    try:
        # 创建回测引擎
        backtest = BacktestEngine()
        
        # 运行回测
        symbol = "BTC/USDT"
        timeframes = ["1h", "4h", "1d"]
        start_date = "2023-01-01"
        end_date = "2023-12-31"
        
        report = backtest.run_backtest(symbol, timeframes, start_date, end_date)
        
        # 打印结果
        print("回测结果:")
        print(f"初始资金: {report['initial_balance']:.2f}")
        print(f"最终资金: {report['final_balance']:.2f}")
        print(f"总收益率: {report['performance_metrics']['total_return']:.2%}")
        print(f"总交易次数: {report['performance_metrics']['total_trades']}")
        print(f"胜率: {report['performance_metrics']['win_rate']:.2%}")
        print(f"最大回撤: {report['performance_metrics']['max_drawdown']:.2%}")
        print(f"夏普比率: {report['performance_metrics']['sharpe_ratio']:.2f}")
        print(f"盈利因子: {report['performance_metrics']['profit_factor']:.2f}")
        
    except Exception as e:
        print(f"回测示例失败: {e}")


if __name__ == "__main__":
    run_backtest_example()