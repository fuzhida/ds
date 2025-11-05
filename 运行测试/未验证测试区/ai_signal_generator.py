"""
AI信号生成模块 - 包含AI分析和信号生成功能
"""

import pandas as pd
import numpy as np
import json
import requests
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timezone, timedelta
import os
import re
from abc import ABC, abstractmethod


class AISignalProvider(ABC):
    """AI信号提供者抽象基类"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
    
    @abstractmethod
    def generate_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成交易信号"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """检查服务是否可用"""
        pass


class DeepSeekSignalProvider(AISignalProvider):
    """DeepSeek AI信号提供者"""
    
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.api_key = config.deepseek_api_key
        self.base_url = config.deepseek_base_url
        self.model = config.deepseek_model
        self.timeout = config.ai_timeout
    
    def is_available(self) -> bool:
        """检查DeepSeek服务是否可用"""
        try:
            if not self.api_key:
                self.logger.warning("DeepSeek API密钥未配置")
                return False
            
            # 简单的API测试
            test_url = f"{self.base_url}/models"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(test_url, headers=headers, timeout=5)
            return response.status_code == 200
            
        except Exception as e:
            self.logger.error(f"DeepSeek服务可用性检查失败: {e}")
            return False
    
    def generate_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """使用DeepSeek生成交易信号"""
        try:
            if not self.is_available():
                return self._get_fallback_signal("DeepSeek服务不可用")
            
            # 准备提示词
            prompt = self._build_prompt(market_data)
            
            # 调用API
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "你是一个专业的交易分析师，基于提供的市场数据生成交易信号。"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 500
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                self.logger.error(f"DeepSeek API请求失败: {response.status_code} {response.text}")
                return self._get_fallback_signal(f"API请求失败: {response.status_code}")
            
            # 解析响应
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # 提取信号
            signal = self._extract_signal_from_response(content)
            
            return {
                "provider": "DeepSeek",
                "signal": signal,
                "confidence": signal.get("confidence", 0.5),
                "reasoning": signal.get("reasoning", ""),
                "raw_response": content,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"DeepSeek信号生成失败: {e}")
            return self._get_fallback_signal(f"生成失败: {str(e)}")
    
    def _build_prompt(self, market_data: Dict[str, Any]) -> str:
        """构建提示词"""
        try:
            # 获取当前价格
            current_price = market_data.get("current_price", 0)
            
            # 获取技术指标
            technical_indicators = market_data.get("technical_indicators", {})
            
            # 获取SMC结构
            smc_structures = market_data.get("smc_structures", {})
            
            # 获取关键水平
            key_levels = market_data.get("key_levels", {})
            
            # 获取价格行为
            price_action = market_data.get("price_action", {})
            
            # 构建提示词
            prompt = f"""
请分析以下市场数据并生成交易信号：

当前价格: {current_price}

技术指标:
- EMA: {technical_indicators.get('ema', {})}
- RSI: {technical_indicators.get('rsi', {})}
- MACD: {technical_indicators.get('macd', {})}
- 布林带: {technical_indicators.get('bollinger', {})}
- 成交量指标: {technical_indicators.get('volume', {})}

SMC结构:
- BOS/CHOCH: {smc_structures.get('bos_choch', {})}
- 订单块: {smc_structures.get('order_blocks', [])}
- 公平价值缺口: {smc_structures.get('fvg', [])}
- 摆动点: {smc_structures.get('swing_points', [])}

关键水平:
- 支撑: {key_levels.get('support', [])}
- 阻力: {key_levels.get('resistance', [])}
- EMA: {key_levels.get('ema', {})}

价格行为:
- 蜡烛图模式: {price_action.get('candlestick_patterns', {})}
- 价格效率: {price_action.get('price_efficiency', 0)}
- 波动性: {price_action.get('volatility', {})}
- 动量: {price_action.get('momentum', {})}

请基于以上信息，生成JSON格式的交易信号，包含以下字段:
- signal: "BUY"、"SELL"或"HOLD"
- confidence: 0-1之间的置信度
- reasoning: 详细的推理过程
- entry_price: 建议入场价格
- stop_loss: 建议止损价格
- take_profit: 建议止盈价格

请确保返回有效的JSON格式。
"""
            
            return prompt
            
        except Exception as e:
            self.logger.error(f"提示词构建失败: {e}")
            return "请基于市场数据生成交易信号"
    
    def _extract_signal_from_response(self, content: str) -> Dict[str, Any]:
        """从响应中提取信号"""
        try:
            # 尝试提取JSON
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                signal = json.loads(json_str)
                return signal
            
            # 尝试直接解析JSON
            try:
                signal = json.loads(content)
                return signal
            except:
                pass
            
            # 提取信号关键词
            signal = "HOLD"
            confidence = 0.5
            reasoning = content
            
            if "BUY" in content.upper() or "买入" in content or "做多" in content:
                signal = "BUY"
                confidence = 0.7
            elif "SELL" in content.upper() or "卖出" in content or "做空" in content:
                signal = "SELL"
                confidence = 0.7
            
            return {
                "signal": signal,
                "confidence": confidence,
                "reasoning": reasoning,
                "entry_price": None,
                "stop_loss": None,
                "take_profit": None
            }
            
        except Exception as e:
            self.logger.error(f"信号提取失败: {e}")
            return {
                "signal": "HOLD",
                "confidence": 0.3,
                "reasoning": f"信号提取失败: {str(e)}",
                "entry_price": None,
                "stop_loss": None,
                "take_profit": None
            }
    
    def _get_fallback_signal(self, reason: str) -> Dict[str, Any]:
        """获取备用信号"""
        return {
            "provider": "DeepSeek",
            "signal": {
                "signal": "HOLD",
                "confidence": 0.2,
                "reasoning": f"DeepSeek服务不可用: {reason}",
                "entry_price": None,
                "stop_loss": None,
                "take_profit": None
            },
            "confidence": 0.2,
            "reasoning": f"DeepSeek服务不可用: {reason}",
            "raw_response": "",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


class OpenAISignalProvider(AISignalProvider):
    """OpenAI AI信号提供者"""
    
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.api_key = config.openai_api_key
        self.base_url = config.openai_base_url
        self.model = config.openai_model
        self.timeout = config.ai_timeout
    
    def is_available(self) -> bool:
        """检查OpenAI服务是否可用"""
        try:
            if not self.api_key:
                self.logger.warning("OpenAI API密钥未配置")
                return False
            
            # 简单的API测试
            test_url = f"{self.base_url}/models"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(test_url, headers=headers, timeout=5)
            return response.status_code == 200
            
        except Exception as e:
            self.logger.error(f"OpenAI服务可用性检查失败: {e}")
            return False
    
    def generate_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """使用OpenAI生成交易信号"""
        try:
            if not self.is_available():
                return self._get_fallback_signal("OpenAI服务不可用")
            
            # 准备提示词
            prompt = self._build_prompt(market_data)
            
            # 调用API
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "你是一个专业的交易分析师，基于提供的市场数据生成交易信号。"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 500
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                self.logger.error(f"OpenAI API请求失败: {response.status_code} {response.text}")
                return self._get_fallback_signal(f"API请求失败: {response.status_code}")
            
            # 解析响应
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # 提取信号
            signal = self._extract_signal_from_response(content)
            
            return {
                "provider": "OpenAI",
                "signal": signal,
                "confidence": signal.get("confidence", 0.5),
                "reasoning": signal.get("reasoning", ""),
                "raw_response": content,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"OpenAI信号生成失败: {e}")
            return self._get_fallback_signal(f"生成失败: {str(e)}")
    
    def _build_prompt(self, market_data: Dict[str, Any]) -> str:
        """构建提示词"""
        try:
            # 获取当前价格
            current_price = market_data.get("current_price", 0)
            
            # 获取技术指标
            technical_indicators = market_data.get("technical_indicators", {})
            
            # 获取SMC结构
            smc_structures = market_data.get("smc_structures", {})
            
            # 获取关键水平
            key_levels = market_data.get("key_levels", {})
            
            # 获取价格行为
            price_action = market_data.get("price_action", {})
            
            # 构建提示词
            prompt = f"""
请分析以下市场数据并生成交易信号：

当前价格: {current_price}

技术指标:
- EMA: {technical_indicators.get('ema', {})}
- RSI: {technical_indicators.get('rsi', {})}
- MACD: {technical_indicators.get('macd', {})}
- 布林带: {technical_indicators.get('bollinger', {})}
- 成交量指标: {technical_indicators.get('volume', {})}

SMC结构:
- BOS/CHOCH: {smc_structures.get('bos_choch', {})}
- 订单块: {smc_structures.get('order_blocks', [])}
- 公平价值缺口: {smc_structures.get('fvg', [])}
- 摆动点: {smc_structures.get('swing_points', [])}

关键水平:
- 支撑: {key_levels.get('support', [])}
- 阻力: {key_levels.get('resistance', [])}
- EMA: {key_levels.get('ema', {})}

价格行为:
- 蜡烛图模式: {price_action.get('candlestick_patterns', {})}
- 价格效率: {price_action.get('price_efficiency', 0)}
- 波动性: {price_action.get('volatility', {})}
- 动量: {price_action.get('momentum', {})}

请基于以上信息，生成JSON格式的交易信号，包含以下字段:
- signal: "BUY"、"SELL"或"HOLD"
- confidence: 0-1之间的置信度
- reasoning: 详细的推理过程
- entry_price: 建议入场价格
- stop_loss: 建议止损价格
- take_profit: 建议止盈价格

请确保返回有效的JSON格式。
"""
            
            return prompt
            
        except Exception as e:
            self.logger.error(f"提示词构建失败: {e}")
            return "请基于市场数据生成交易信号"
    
    def _extract_signal_from_response(self, content: str) -> Dict[str, Any]:
        """从响应中提取信号"""
        try:
            # 尝试提取JSON
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                signal = json.loads(json_str)
                return signal
            
            # 尝试直接解析JSON
            try:
                signal = json.loads(content)
                return signal
            except:
                pass
            
            # 提取信号关键词
            signal = "HOLD"
            confidence = 0.5
            reasoning = content
            
            if "BUY" in content.upper() or "买入" in content or "做多" in content:
                signal = "BUY"
                confidence = 0.7
            elif "SELL" in content.upper() or "卖出" in content or "做空" in content:
                signal = "SELL"
                confidence = 0.7
            
            return {
                "signal": signal,
                "confidence": confidence,
                "reasoning": reasoning,
                "entry_price": None,
                "stop_loss": None,
                "take_profit": None
            }
            
        except Exception as e:
            self.logger.error(f"信号提取失败: {e}")
            return {
                "signal": "HOLD",
                "confidence": 0.3,
                "reasoning": f"信号提取失败: {str(e)}",
                "entry_price": None,
                "stop_loss": None,
                "take_profit": None
            }
    
    def _get_fallback_signal(self, reason: str) -> Dict[str, Any]:
        """获取备用信号"""
        return {
            "provider": "OpenAI",
            "signal": {
                "signal": "HOLD",
                "confidence": 0.2,
                "reasoning": f"OpenAI服务不可用: {reason}",
                "entry_price": None,
                "stop_loss": None,
                "take_profit": None
            },
            "confidence": 0.2,
            "reasoning": f"OpenAI服务不可用: {reason}",
            "raw_response": "",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


class AISignalGenerator:
    """AI信号生成器"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.providers = []
        self._initialize_providers()
    
    def _initialize_providers(self):
        """初始化AI信号提供者"""
        try:
            # 初始化DeepSeek提供者
            if hasattr(self.config, 'deepseek_api_key') and self.config.deepseek_api_key:
                deepseek_provider = DeepSeekSignalProvider(self.config, self.logger)
                if deepseek_provider.is_available():
                    self.providers.append(deepseek_provider)
                    self.logger.info("DeepSeek AI信号提供者初始化成功")
                else:
                    self.logger.warning("DeepSeek AI信号提供者不可用")
            
            # 初始化OpenAI提供者
            if hasattr(self.config, 'openai_api_key') and self.config.openai_api_key:
                openai_provider = OpenAISignalProvider(self.config, self.logger)
                if openai_provider.is_available():
                    self.providers.append(openai_provider)
                    self.logger.info("OpenAI AI信号提供者初始化成功")
                else:
                    self.logger.warning("OpenAI AI信号提供者不可用")
            
            if not self.providers:
                self.logger.warning("没有可用的AI信号提供者")
            
        except Exception as e:
            self.logger.error(f"AI信号提供者初始化失败: {e}")
    
    def generate_signals(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成AI交易信号"""
        try:
            signals = {}
            
            # 从每个提供者获取信号
            for provider in self.providers:
                try:
                    signal = provider.generate_signal(market_data)
                    signals[provider.__class__.__name__] = signal
                except Exception as e:
                    self.logger.error(f"{provider.__class__.__name__}信号生成失败: {e}")
            
            # 如果没有可用信号，返回默认信号
            if not signals:
                return {
                    "primary": {
                        "provider": "Fallback",
                        "signal": {
                            "signal": "HOLD",
                            "confidence": 0.2,
                            "reasoning": "没有可用的AI信号提供者",
                            "entry_price": None,
                            "stop_loss": None,
                            "take_profit": None
                        },
                        "confidence": 0.2,
                        "reasoning": "没有可用的AI信号提供者",
                        "raw_response": "",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    },
                    "all": {},
                    "consensus": "HOLD",
                    "consensus_confidence": 0.2
                }
            
            # 计算共识信号
            consensus = self._calculate_consensus(signals)
            
            # 选择主要信号
            primary_signal = self._select_primary_signal(signals, consensus)
            
            return {
                "primary": primary_signal,
                "all": signals,
                "consensus": consensus["signal"],
                "consensus_confidence": consensus["confidence"]
            }
            
        except Exception as e:
            self.logger.error(f"AI信号生成失败: {e}")
            return {
                "primary": {
                    "provider": "Fallback",
                    "signal": {
                        "signal": "HOLD",
                        "confidence": 0.1,
                        "reasoning": f"AI信号生成失败: {str(e)}",
                        "entry_price": None,
                        "stop_loss": None,
                        "take_profit": None
                    },
                    "confidence": 0.1,
                    "reasoning": f"AI信号生成失败: {str(e)}",
                    "raw_response": "",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                },
                "all": {},
                "consensus": "HOLD",
                "consensus_confidence": 0.1
            }
    
    def _calculate_consensus(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """计算信号共识"""
        try:
            signal_counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
            total_confidence = 0
            
            for provider_name, signal_data in signals.items():
                signal = signal_data.get("signal", {}).get("signal", "HOLD")
                confidence = signal_data.get("confidence", 0)
                
                signal_counts[signal] += 1
                total_confidence += confidence
            
            # 确定共识信号
            if signal_counts["BUY"] > signal_counts["SELL"] and signal_counts["BUY"] > signal_counts["HOLD"]:
                consensus_signal = "BUY"
            elif signal_counts["SELL"] > signal_counts["BUY"] and signal_counts["SELL"] > signal_counts["HOLD"]:
                consensus_signal = "SELL"
            else:
                consensus_signal = "HOLD"
            
            # 计算共识置信度
            total_providers = len(signals)
            consensus_confidence = total_confidence / total_providers if total_providers > 0 else 0
            
            return {
                "signal": consensus_signal,
                "confidence": consensus_confidence,
                "counts": signal_counts
            }
            
        except Exception as e:
            self.logger.error(f"共识计算失败: {e}")
            return {
                "signal": "HOLD",
                "confidence": 0.1,
                "counts": {"BUY": 0, "SELL": 0, "HOLD": 0}
            }
    
    def _select_primary_signal(self, signals: Dict[str, Any], consensus: Dict[str, Any]) -> Dict[str, Any]:
        """选择主要信号"""
        try:
            # 优先选择与共识一致的信号
            consensus_signal = consensus["signal"]
            
            # 找出与共识一致的信号
            matching_signals = []
            for provider_name, signal_data in signals.items():
                signal = signal_data.get("signal", {}).get("signal", "HOLD")
                if signal == consensus_signal:
                    matching_signals.append((provider_name, signal_data))
            
            # 如果没有匹配的信号，选择置信度最高的信号
            if not matching_signals:
                best_provider = None
                best_signal = None
                best_confidence = 0
                
                for provider_name, signal_data in signals.items():
                    confidence = signal_data.get("confidence", 0)
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_provider = provider_name
                        best_signal = signal_data
                
                return best_signal if best_signal else list(signals.values())[0]
            
            # 从匹配的信号中选择置信度最高的
            best_provider = None
            best_signal = None
            best_confidence = 0
            
            for provider_name, signal_data in matching_signals:
                confidence = signal_data.get("confidence", 0)
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_provider = provider_name
                    best_signal = signal_data
            
            return best_signal if best_signal else matching_signals[0][1]
            
        except Exception as e:
            self.logger.error(f"主要信号选择失败: {e}")
            return list(signals.values())[0] if signals else {
                "provider": "Fallback",
                "signal": {
                    "signal": "HOLD",
                    "confidence": 0.1,
                    "reasoning": "信号选择失败",
                    "entry_price": None,
                    "stop_loss": None,
                    "take_profit": None
                },
                "confidence": 0.1,
                "reasoning": "信号选择失败",
                "raw_response": "",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }