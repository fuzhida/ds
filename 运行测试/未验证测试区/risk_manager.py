"""
风险管理模块 - 包含风险控制、仓位管理和止损止盈功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timezone, timedelta
import math


class RiskManager:
    """风险管理器"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        
        # 风险参数
        self.max_risk_per_trade = config.max_risk_per_trade  # 每笔交易最大风险百分比
        self.max_daily_risk = config.max_daily_risk  # 每日最大风险百分比
        self.max_open_positions = config.max_open_positions  # 最大开仓数量
        self.risk_reward_ratio = config.risk_reward_ratio  # 风险回报比
        self.stop_loss_atr_multiplier = config.stop_loss_atr_multiplier  # ATR止损倍数
        self.take_profit_atr_multiplier = config.take_profit_atr_multiplier  # ATR止盈倍数
        self.trailing_stop_atr_multiplier = config.trailing_stop_atr_multiplier  # 移动止损ATR倍数
        self.trailing_stop_activation_atr = config.trailing_stop_activation_atr  # 移动止损激活ATR
        
        # 风险状态
        self.daily_risk_used = 0.0
        self.daily_risk_reset_time = self._get_next_reset_time()
        self.open_positions = []
        self.trade_history = []
    
    def _get_next_reset_time(self) -> datetime:
        """获取下一次风险重置时间"""
        now = datetime.now(timezone.utc)
        tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        return tomorrow
    
    def _reset_daily_risk(self):
        """重置每日风险使用量"""
        now = datetime.now(timezone.utc)
        if now >= self.daily_risk_reset_time:
            self.daily_risk_used = 0.0
            self.daily_risk_reset_time = self._get_next_reset_time()
            self.logger.info("每日风险使用量已重置")
    
    def calculate_position_size(self, signal: Dict[str, Any], account_balance: float, 
                              current_price: float, atr: float) -> Dict[str, Any]:
        """计算仓位大小"""
        try:
            self._reset_daily_risk()
            
            # 获取信号信息
            signal_type = signal.get("signal", "HOLD")
            confidence = signal.get("confidence", 0.5)
            
            # 检查是否可以开新仓
            if signal_type == "HOLD":
                return {"position_size": 0, "reason": "信号为HOLD，不开仓"}
            
            if len(self.open_positions) >= self.max_open_positions:
                return {"position_size": 0, "reason": f"已达到最大开仓数量 {self.max_open_positions}"}
            
            # 计算风险金额
            risk_amount = account_balance * self.max_risk_per_trade * confidence
            
            # 计算止损距离
            stop_loss_distance = atr * self.stop_loss_atr_multiplier
            
            # 计算仓位大小
            if stop_loss_distance > 0:
                position_size = risk_amount / stop_loss_distance
            else:
                position_size = 0
            
            # 检查每日风险限制
            new_risk = (position_size * stop_loss_distance) / account_balance
            if self.daily_risk_used + new_risk > self.max_daily_risk:
                remaining_risk = self.max_daily_risk - self.daily_risk_used
                position_size = (account_balance * remaining_risk) / stop_loss_distance
                
                if position_size <= 0:
                    return {"position_size": 0, "reason": "已达到每日最大风险限制"}
            
            # 应用仓位大小限制
            max_position_size = account_balance * 0.95 / current_price  # 最大95%的账户余额
            position_size = min(position_size, max_position_size)
            
            # 计算实际风险
            actual_risk = (position_size * stop_loss_distance) / account_balance
            
            return {
                "position_size": position_size,
                "risk_amount": position_size * stop_loss_distance,
                "risk_percentage": actual_risk,
                "stop_loss_distance": stop_loss_distance,
                "reason": "仓位计算成功"
            }
            
        except Exception as e:
            self.logger.error(f"仓位大小计算失败: {e}")
            return {"position_size": 0, "reason": f"计算失败: {str(e)}"}
    
    def calculate_stop_loss(self, signal_type: str, entry_price: float, 
                           atr: float, market_structure: Dict[str, Any] = None) -> float:
        """计算止损价格"""
        try:
            if signal_type == "BUY":
                # 做多止损在入场价下方
                stop_loss = entry_price - (atr * self.stop_loss_atr_multiplier)
                
                # 如果有市场结构信息，考虑关键支撑位
                if market_structure:
                    support_levels = market_structure.get("support_levels", [])
                    if support_levels:
                        nearest_support = max([level for level in support_levels if level < entry_price], default=stop_loss)
                        stop_loss = max(stop_loss, nearest_support)
                
                return stop_loss
                
            elif signal_type == "SELL":
                # 做空止损在入场价上方
                stop_loss = entry_price + (atr * self.stop_loss_atr_multiplier)
                
                # 如果有市场结构信息，考虑关键阻力位
                if market_structure:
                    resistance_levels = market_structure.get("resistance_levels", [])
                    if resistance_levels:
                        nearest_resistance = min([level for level in resistance_levels if level > entry_price], default=stop_loss)
                        stop_loss = min(stop_loss, nearest_resistance)
                
                return stop_loss
                
            else:
                return entry_price  # HOLD信号的止损就是入场价
                
        except Exception as e:
            self.logger.error(f"止损计算失败: {e}")
            return entry_price
    
    def calculate_take_profit(self, signal_type: str, entry_price: float, 
                             stop_loss: float, atr: float, 
                             market_structure: Dict[str, Any] = None) -> List[float]:
        """计算止盈价格列表"""
        try:
            if signal_type == "HOLD":
                return [entry_price]
            
            # 基于风险回报比计算基础止盈
            risk_distance = abs(entry_price - stop_loss)
            base_take_profit = entry_price + (risk_distance * self.risk_reward_ratio) if signal_type == "BUY" else entry_price - (risk_distance * self.risk_reward_ratio)
            
            # 基于ATR计算止盈
            atr_take_profit = entry_price + (atr * self.take_profit_atr_multiplier) if signal_type == "BUY" else entry_price - (atr * self.take_profit_atr_multiplier)
            
            # 合并止盈目标
            take_profits = [base_take_profit, atr_take_profit]
            
            # 如果有市场结构信息，考虑关键水平
            if market_structure:
                if signal_type == "BUY":
                    resistance_levels = market_structure.get("resistance_levels", [])
                    # 找到入场价上方的阻力位
                    valid_resistances = [level for level in resistance_levels if level > entry_price]
                    if valid_resistances:
                        nearest_resistance = min(valid_resistances)
                        take_profits.append(nearest_resistance)
                else:  # SELL
                    support_levels = market_structure.get("support_levels", [])
                    # 找到入场价下方的支撑位
                    valid_supports = [level for level in support_levels if level < entry_price]
                    if valid_supports:
                        nearest_support = max(valid_supports)
                        take_profits.append(nearest_support)
            
            # 去重并排序
            take_profits = list(set(take_profits))
            if signal_type == "BUY":
                take_profits = sorted([tp for tp in take_profits if tp > entry_price])
            else:  # SELL
                take_profits = sorted([tp for tp in take_profits if tp < entry_price], reverse=True)
            
            # 限制止盈目标数量
            take_profits = take_profits[:3]
            
            return take_profits if take_profits else [base_take_profit]
            
        except Exception as e:
            self.logger.error(f"止盈计算失败: {e}")
            return [entry_price]
    
    def calculate_trailing_stop(self, position: Dict[str, Any], current_price: float, 
                                atr: float) -> Optional[float]:
        """计算移动止损价格"""
        try:
            signal_type = position.get("signal_type", "HOLD")
            entry_price = position.get("entry_price", 0)
            current_stop_loss = position.get("stop_loss", entry_price)
            activation_price = position.get("trailing_stop_activation_price", None)
            
            if signal_type == "HOLD":
                return None
            
            # 计算移动止损激活价格
            if activation_price is None:
                activation_distance = atr * self.trailing_stop_activation_atr
                if signal_type == "BUY":
                    activation_price = entry_price + activation_distance
                else:  # SELL
                    activation_price = entry_price - activation_distance
                
                position["trailing_stop_activation_price"] = activation_price
            
            # 检查是否激活移动止损
            if signal_type == "BUY" and current_price < activation_price:
                return current_stop_loss
            elif signal_type == "SELL" and current_price > activation_price:
                return current_stop_loss
            
            # 计算新的移动止损
            trailing_distance = atr * self.trailing_stop_atr_multiplier
            
            if signal_type == "BUY":
                new_stop_loss = current_price - trailing_distance
                # 只有当新止损高于当前止损时才更新
                return max(new_stop_loss, current_stop_loss)
            else:  # SELL
                new_stop_loss = current_price + trailing_distance
                # 只有当新止损低于当前止损时才更新
                return min(new_stop_loss, current_stop_loss)
                
        except Exception as e:
            self.logger.error(f"移动止损计算失败: {e}")
            return None
    
    def add_position(self, position: Dict[str, Any]):
        """添加新仓位"""
        try:
            self.open_positions.append(position)
            
            # 更新每日风险使用量
            risk_amount = position.get("risk_amount", 0)
            account_balance = position.get("account_balance", 1)
            risk_percentage = risk_amount / account_balance
            self.daily_risk_used += risk_percentage
            
            self.logger.info(f"新仓位已添加: {position.get('signal_type')} {position.get('position_size')} @ {position.get('entry_price')}")
            
        except Exception as e:
            self.logger.error(f"添加仓位失败: {e}")
    
    def remove_position(self, position_id: str) -> Optional[Dict[str, Any]]:
        """移除仓位"""
        try:
            for i, position in enumerate(self.open_positions):
                if position.get("id") == position_id:
                    removed_position = self.open_positions.pop(i)
                    
                    # 添加到交易历史
                    self.trade_history.append(removed_position)
                    
                    self.logger.info(f"仓位已移除: {removed_position.get('signal_type')} {removed_position.get('position_size')} @ {removed_position.get('entry_price')}")
                    
                    return removed_position
            
            self.logger.warning(f"未找到仓位ID: {position_id}")
            return None
            
        except Exception as e:
            self.logger.error(f"移除仓位失败: {e}")
            return None
    
    def get_open_positions(self) -> List[Dict[str, Any]]:
        """获取当前开仓"""
        return self.open_positions.copy()
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """获取风险摘要"""
        try:
            self._reset_daily_risk()
            
            total_position_value = sum(pos.get("position_value", 0) for pos in self.open_positions)
            total_risk_amount = sum(pos.get("risk_amount", 0) for pos in self.open_positions)
            
            return {
                "daily_risk_used": self.daily_risk_used,
                "daily_risk_limit": self.max_daily_risk,
                "daily_risk_remaining": max(0, self.max_daily_risk - self.daily_risk_used),
                "open_positions_count": len(self.open_positions),
                "max_open_positions": self.max_open_positions,
                "total_position_value": total_position_value,
                "total_risk_amount": total_risk_amount,
                "next_risk_reset_time": self.daily_risk_reset_time.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"风险摘要生成失败: {e}")
            return {
                "daily_risk_used": 0,
                "daily_risk_limit": self.max_daily_risk,
                "daily_risk_remaining": self.max_daily_risk,
                "open_positions_count": 0,
                "max_open_positions": self.max_open_positions,
                "total_position_value": 0,
                "total_risk_amount": 0,
                "next_risk_reset_time": self.daily_risk_reset_time.isoformat()
            }
    
    def should_reduce_risk(self, market_conditions: Dict[str, Any]) -> bool:
        """判断是否应该降低风险"""
        try:
            # 检查市场波动性
            volatility = market_conditions.get("volatility", 0.5)
            if volatility > 0.8:  # 高波动性
                return True
            
            # 检查市场趋势强度
            trend_strength = market_conditions.get("trend_strength", 0.5)
            if trend_strength < 0.2:  # 弱趋势
                return True
            
            # 检查近期交易表现
            recent_trades = self.trade_history[-10:]  # 最近10笔交易
            if len(recent_trades) >= 5:
                win_rate = sum(1 for trade in recent_trades if trade.get("profit_loss", 0) > 0) / len(recent_trades)
                if win_rate < 0.3:  # 胜率低于30%
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"风险降低判断失败: {e}")
            return False
    
    def adjust_risk_parameters(self, market_conditions: Dict[str, Any]):
        """根据市场条件调整风险参数"""
        try:
            if self.should_reduce_risk(market_conditions):
                # 降低风险参数
                original_risk = self.max_risk_per_trade
                self.max_risk_per_trade *= 0.7  # 降低30%
                self.max_daily_risk *= 0.7  # 降低30%
                
                self.logger.info(f"风险参数已调整: 每笔交易风险从 {original_risk:.2%} 降低到 {self.max_risk_per_trade:.2%}")
            
        except Exception as e:
            self.logger.error(f"风险参数调整失败: {e}")
    
    def reset_risk_parameters(self):
        """重置风险参数到默认值"""
        try:
            self.max_risk_per_trade = self.config.max_risk_per_trade
            self.max_daily_risk = self.config.max_daily_risk
            
            self.logger.info("风险参数已重置到默认值")
            
        except Exception as e:
            self.logger.error(f"风险参数重置失败: {e}")


class PositionManager:
    """仓位管理器"""
    
    def __init__(self, config, logger, risk_manager: RiskManager):
        self.config = config
        self.logger = logger
        self.risk_manager = risk_manager
        self.positions = {}
        self.position_counter = 0
    
    def create_position(self, signal: Dict[str, Any], entry_price: float, 
                       position_size: float, stop_loss: float, 
                       take_profits: List[float], account_balance: float) -> Dict[str, Any]:
        """创建新仓位"""
        try:
            self.position_counter += 1
            position_id = f"pos_{self.position_counter}"
            
            signal_type = signal.get("signal", "HOLD")
            confidence = signal.get("confidence", 0.5)
            
            position = {
                "id": position_id,
                "signal_type": signal_type,
                "entry_price": entry_price,
                "position_size": position_size,
                "position_value": position_size * entry_price,
                "stop_loss": stop_loss,
                "take_profits": take_profits,
                "current_take_profit_index": 0,
                "confidence": confidence,
                "account_balance": account_balance,
                "risk_amount": position_size * abs(entry_price - stop_loss),
                "risk_percentage": (position_size * abs(entry_price - stop_loss)) / account_balance,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "status": "open",
                "profit_loss": 0.0,
                "profit_loss_percentage": 0.0,
                "highest_profit": 0.0,
                "lowest_loss": 0.0,
                "trailing_stop_activation_price": None,
                "notes": signal.get("reasoning", "")
            }
            
            self.positions[position_id] = position
            self.risk_manager.add_position(position)
            
            return position
            
        except Exception as e:
            self.logger.error(f"创建仓位失败: {e}")
            return {}
    
    def update_position(self, position_id: str, current_price: float, atr: float) -> Dict[str, Any]:
        """更新仓位状态"""
        try:
            if position_id not in self.positions:
                self.logger.error(f"未找到仓位ID: {position_id}")
                return {}
            
            position = self.positions[position_id]
            signal_type = position.get("signal_type", "HOLD")
            entry_price = position.get("entry_price", 0)
            position_size = position.get("position_size", 0)
            stop_loss = position.get("stop_loss", entry_price)
            take_profits = position.get("take_profits", [])
            current_take_profit_index = position.get("current_take_profit_index", 0)
            
            # 计算盈亏
            if signal_type == "BUY":
                profit_loss = (current_price - entry_price) * position_size
                profit_loss_percentage = (current_price - entry_price) / entry_price
            elif signal_type == "SELL":
                profit_loss = (entry_price - current_price) * position_size
                profit_loss_percentage = (entry_price - current_price) / entry_price
            else:
                profit_loss = 0
                profit_loss_percentage = 0
            
            # 更新最高盈利和最低亏损
            highest_profit = max(position.get("highest_profit", 0), profit_loss)
            lowest_loss = min(position.get("lowest_loss", 0), profit_loss)
            
            # 检查止损
            should_close = False
            close_reason = ""
            
            if signal_type == "BUY" and current_price <= stop_loss:
                should_close = True
                close_reason = "止损"
            elif signal_type == "SELL" and current_price >= stop_loss:
                should_close = True
                close_reason = "止损"
            
            # 检查止盈
            if not should_close and current_take_profit_index < len(take_profits):
                current_take_profit = take_profits[current_take_profit_index]
                
                if signal_type == "BUY" and current_price >= current_take_profit:
                    should_close = True
                    close_reason = f"止盈{current_take_profit_index + 1}"
                elif signal_type == "SELL" and current_price <= current_take_profit:
                    should_close = True
                    close_reason = f"止盈{current_take_profit_index + 1}"
            
            # 计算移动止损
            trailing_stop = self.risk_manager.calculate_trailing_stop(position, current_price, atr)
            if trailing_stop and trailing_stop != stop_loss:
                position["stop_loss"] = trailing_stop
                self.logger.info(f"仓位 {position_id} 移动止损已更新至 {trailing_stop}")
            
            # 更新仓位信息
            position["current_price"] = current_price
            position["profit_loss"] = profit_loss
            position["profit_loss_percentage"] = profit_loss_percentage
            position["highest_profit"] = highest_profit
            position["lowest_loss"] = lowest_loss
            position["updated_at"] = datetime.now(timezone.utc).isoformat()
            
            if should_close:
                position["status"] = "closed"
                position["close_reason"] = close_reason
                position["closed_at"] = datetime.now(timezone.utc).isoformat()
                position["close_price"] = current_price
                
                # 从风险管理器中移除
                self.risk_manager.remove_position(position_id)
                
                self.logger.info(f"仓位 {position_id} 已关闭: {close_reason}, 盈亏: {profit_loss:.2f}")
            
            return position
            
        except Exception as e:
            self.logger.error(f"更新仓位失败: {e}")
            return {}
    
    def get_position(self, position_id: str) -> Dict[str, Any]:
        """获取指定仓位"""
        return self.positions.get(position_id, {})
    
    def get_all_positions(self) -> Dict[str, Dict[str, Any]]:
        """获取所有仓位"""
        return self.positions.copy()
    
    def get_open_positions(self) -> Dict[str, Dict[str, Any]]:
        """获取所有开仓"""
        return {pos_id: pos for pos_id, pos in self.positions.items() if pos.get("status") == "open"}
    
    def close_position(self, position_id: str, close_price: float, close_reason: str = "手动关闭") -> Dict[str, Any]:
        """手动关闭仓位"""
        try:
            if position_id not in self.positions:
                self.logger.error(f"未找到仓位ID: {position_id}")
                return {}
            
            position = self.positions[position_id]
            
            if position.get("status") == "closed":
                self.logger.warning(f"仓位 {position_id} 已经关闭")
                return position
            
            # 更新仓位状态
            position["status"] = "closed"
            position["close_reason"] = close_reason
            position["close_price"] = close_price
            position["closed_at"] = datetime.now(timezone.utc).isoformat()
            
            # 计算最终盈亏
            signal_type = position.get("signal_type", "HOLD")
            entry_price = position.get("entry_price", 0)
            position_size = position.get("position_size", 0)
            
            if signal_type == "BUY":
                profit_loss = (close_price - entry_price) * position_size
                profit_loss_percentage = (close_price - entry_price) / entry_price
            elif signal_type == "SELL":
                profit_loss = (entry_price - close_price) * position_size
                profit_loss_percentage = (entry_price - close_price) / entry_price
            else:
                profit_loss = 0
                profit_loss_percentage = 0
            
            position["profit_loss"] = profit_loss
            position["profit_loss_percentage"] = profit_loss_percentage
            
            # 从风险管理器中移除
            self.risk_manager.remove_position(position_id)
            
            self.logger.info(f"仓位 {position_id} 已手动关闭: {close_reason}, 盈亏: {profit_loss:.2f}")
            
            return position
            
        except Exception as e:
            self.logger.error(f"手动关闭仓位失败: {e}")
            return {}