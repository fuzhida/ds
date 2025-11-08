"""
AI信号生成模块 - 包含AI分析和信号生成功能
"""

import pandas as pd
import numpy as np
import json
import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import Timeout, ReadTimeout, ConnectTimeout
try:
    from urllib3.util.retry import Retry
except Exception:
    Retry = None
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
        # 统一超时配置（读超时至少30秒）
        self.timeout = max(getattr(config, 'ai_timeout', 30) or 30, 30)
        # 配置会话与轻量重试策略
        try:
            session = requests.Session()
            if Retry is not None:
                retry = Retry(
                    total=2,
                    backoff_factor=0.5,
                    status_forcelist=[429, 500, 502, 503, 504],
                    allowed_methods={"GET", "POST"}
                )
                adapter = HTTPAdapter(max_retries=retry)
                session.mount("https://", adapter)
                session.mount("http://", adapter)
            self.session = session
        except Exception:
            # 兜底：直接使用requests模块
            self.session = requests
    
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
            
            response = self.session.get(test_url, headers=headers, timeout=5)
            return response.status_code == 200
            
        except Exception as e:
            self.logger.error(f"DeepSeek服务可用性检查失败: {e}")
            return False
    
    def generate_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """使用DeepSeek生成交易信号（返回扁平结构以便于消费）"""
        try:
            # 仅检查密钥是否配置，避免单测因可用性网络探测失败
            if not self.api_key:
                return self._get_fallback_signal("API key not configured")
            
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
                    {"role": "system", "content": "你是一个专业的交易分析师。严格返回JSON对象，无任何额外文字或标记。reasoning需简洁。"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": float(getattr(self.config, 'ai_temperature', 0.1) or 0.1),
                "top_p": 0.1,
                "max_tokens": int(getattr(self.config, 'ai_max_tokens', 1024) or 1024),
                "response_format": {"type": "json_object"}
            }

            # 轻量审计：准备基准审计条目（不记录密钥/headers）
            audit_base = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "symbol": getattr(self.config, "symbol", ""),
                "provider": "DeepSeek",
                "model": self.model,
                "messages": payload.get("messages", [])
            }
            
            try:
                response = self.session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=(10, max(self.timeout, 30))
                )
                attempt_no = 1
            except (Timeout, ReadTimeout, ConnectTimeout) as te:
                # 一次超时重试，延长读超时
                self.logger.error(f"DeepSeek请求超时，进行重试: {te}")
                try:
                    response = self.session.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=(10, max(self.timeout, 60))
                    )
                    attempt_no = 2
                except Exception as te2:
                    # 记录审计并返回降级信号
                    try:
                        self._append_audit_log({
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "symbol": getattr(self.config, "symbol", ""),
                            "provider": "DeepSeek",
                            "model": self.model,
                            "messages": payload.get("messages", []),
                            "error": f"timeout: {str(te2)}",
                            "attempt": 2
                        })
                    except Exception:
                        pass
                    return self._get_fallback_signal("API Timeout")

            # 解析响应（尽量记录 JSON，否则记录文本）
            try:
                resp_obj = response.json()
            except Exception:
                resp_obj = {"text": getattr(response, "text", "")}

            # 解析消息体（兼容思考模式 reasoning_content）
            result = resp_obj if isinstance(resp_obj, dict) else {}
            message = result.get("choices", [{}])[0].get("message", {}) or {}
            content = message.get("content", "") or ""
            reasoning_content = message.get("reasoning_content", "") or ""
            finish_reason = None
            try:
                choices0 = result.get("choices", [{}])[0]
                if isinstance(choices0, dict):
                    finish_reason = choices0.get("finish_reason")
            except Exception:
                finish_reason = None

            # 构建最终信号（仅在成功时）
            parsed_signal = None
            final_result = None
            if response.status_code == 200:
                # 若被截断，进行一次扩容重试
                if finish_reason == "length":
                    try:
                        payload_more_tokens = dict(payload)
                        payload_more_tokens["max_tokens"] = max(int(payload.get("max_tokens", 1024) or 1024), 2048)
                        response_len = self.session.post(
                            f"{self.base_url}/chat/completions",
                            headers=headers,
                            json=payload_more_tokens,
                            timeout=(10, max(self.timeout, 60))
                        )
                        attempt_no = 2
                        try:
                            resp_obj_len = response_len.json()
                        except Exception:
                            resp_obj_len = {"text": getattr(response_len, "text", "")}
                        result_len = resp_obj_len if isinstance(resp_obj_len, dict) else {}
                        message_len = result_len.get("choices", [{}])[0].get("message", {}) or {}
                        content = message_len.get("content", "") or ""
                        reasoning_content = message_len.get("reasoning_content", "") or ""
                        # 更新审计（截断重试）
                        try:
                            self._append_audit_log({
                                **audit_base,
                                "status_code": response_len.status_code,
                                "response": resp_obj_len,
                                "raw_content": content,
                                "raw_reasoning_content": reasoning_content,
                                "parsed": None,
                                "attempt": attempt_no,
                                "note": "retry_on_length"
                            })
                        except Exception:
                            pass
                        # 若重试成功，覆盖后续的构造
                        if response_len.status_code != 200:
                            # 保留第一次的内容，但继续走下方解析
                            self.logger.warning("DeepSeek响应截断扩容重试失败，使用首次返回内容解析")
                    except Exception as _len_e:
                        self.logger.error(f"DeepSeek响应截断扩容重试异常: {_len_e}")
                signal = self._extract_signal_from_response(content)
                reasoning = reasoning_content if reasoning_content else signal.get("reasoning", "") or content
                final_result = {
                    "signal": signal.get("signal", "HOLD"),
                    "confidence": float(signal.get("confidence", 0.0)),
                    "reasoning": reasoning,
                    "provider": "DeepSeek"
                }
                parsed_signal = final_result

            # 审计写入（尝试1）
            self._append_audit_log({
                **audit_base,
                "status_code": response.status_code,
                "response": resp_obj,
                "raw_content": content,
                "raw_reasoning_content": reasoning_content,
                "parsed": parsed_signal,
                "attempt": attempt_no
            })

            # 若失败，检测是否为 response_format 不支持导致的 400，进行一次回退重试
            if response.status_code != 200:
                err_str = ""
                if isinstance(resp_obj, dict):
                    err_str = json.dumps(resp_obj, ensure_ascii=False)
                needs_fallback = (response.status_code == 400) and ("response_format" in err_str or "json_object" in err_str)
                if needs_fallback:
                    try:
                        payload_fallback = dict(payload)
                        # 移除 response_format 参数
                        if isinstance(payload_fallback, dict):
                            payload_fallback.pop("response_format", None)
                        response2 = self.session.post(
                            f"{self.base_url}/chat/completions",
                            headers=headers,
                            json=payload_fallback,
                            timeout=(10, max(self.timeout, 60))
                        )
                        attempt_no = 2
                        try:
                            resp_obj2 = response2.json()
                        except Exception:
                            resp_obj2 = {"text": getattr(response2, "text", "")}
                        result2 = resp_obj2 if isinstance(resp_obj2, dict) else {}
                        message2 = result2.get("choices", [{}])[0].get("message", {}) or {}
                        content2 = message2.get("content", "") or ""
                        reasoning_content2 = message2.get("reasoning_content", "") or ""
                        parsed_signal2 = None
                        final_result2 = None
                        if response2.status_code == 200:
                            signal2 = self._extract_signal_from_response(content2)
                            reasoning2 = reasoning_content2 if reasoning_content2 else signal2.get("reasoning", "") or content2
                            final_result2 = {
                                "signal": signal2.get("signal", "HOLD"),
                                "confidence": float(signal2.get("confidence", 0.0)),
                                "reasoning": reasoning2,
                                "provider": "DeepSeek"
                            }
                            parsed_signal2 = final_result2
                        # 审计写入（尝试2）
                        self._append_audit_log({
                            **audit_base,
                            "status_code": response2.status_code,
                            "response": resp_obj2,
                            "raw_content": content2,
                            "raw_reasoning_content": reasoning_content2,
                            "parsed": parsed_signal2,
                            "attempt": attempt_no
                        })
                        if response2.status_code == 200 and final_result2 is not None:
                            return final_result2
                        else:
                            self.logger.error(f"DeepSeek API回退请求失败: {response2.status_code} {getattr(response2, 'text', '')}")
                    except Exception as _fb_e:
                        try:
                            self._append_audit_log({
                                **audit_base,
                                "error": f"fallback_error: {str(_fb_e)}",
                                "attempt": 2
                            })
                        except Exception:
                            pass
                # 若不满足回退条件或回退仍失败，返回备用信号
                self.logger.error(f"DeepSeek API请求失败: {response.status_code} {getattr(response, 'text', '')}")
                return self._get_fallback_signal(f"API Error: {response.status_code}")

            return final_result
            
        except Exception as e:
            # 审计记录：异常
            try:
                self._append_audit_log({"ts": datetime.now(timezone.utc).isoformat(),
                                        "symbol": getattr(self.config, "symbol", ""),
                                        "provider": "DeepSeek",
                                        "model": self.model,
                                        "messages": payload.get("messages", []) if 'payload' in locals() else [],
                                        "error": str(e)})
            except Exception:
                pass
            self.logger.error(f"DeepSeek信号生成失败: {e}")
            return self._get_fallback_signal(f"Error: {str(e)}")

    def _append_audit_log(self, entry: Dict[str, Any]) -> None:
        """追加写入轻量审计日志（失败时静默）。"""
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            logs_dir = os.path.join(base_dir, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            path = os.path.join(logs_dir, "deepseek_calls.jsonl")
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass
    
    def _build_prompt(self, market_data: Dict[str, Any]) -> str:
        """构建提示词"""
        try:
            # 获取当前价格
            current_price = market_data.get("current_price", 0)

            # 提取分析结果
            technical_indicators = market_data.get("technical_indicators", {}) or {}
            key_levels = market_data.get("key_levels", {}) or {}
            price_action = market_data.get("price_action", {}) or {}
            smc_structures = market_data.get("smc_structures", {}) or {}

            # 选择主要时间框架（优先配置的lower_tf_entry_tf/primary_timeframe，其次3m/1h）并汇总技术指标为最后一根的值
            def _select_tf(d: Dict[str, Any]) -> Optional[str]:
                if not isinstance(d, dict) or not d:
                    return None
                try:
                    candidates = []
                    ltf = getattr(self.config, 'lower_tf_entry_tf', None)
                    ptf = getattr(self.config, 'primary_timeframe', None)
                    if ltf:
                        candidates.append(ltf)
                    if ptf and ptf not in candidates:
                        candidates.append(ptf)
                    if '3m' not in candidates:
                        candidates.append('3m')
                    if '1h' not in candidates:
                        candidates.append('1h')
                    for c in candidates:
                        if c in d:
                            return c
                except Exception:
                    pass
                return next(iter(d.keys()))

            tf = _select_tf(technical_indicators)
            ti_tf = technical_indicators.get(tf, {}) if tf else {}

            def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
                try:
                    return float(x)
                except Exception:
                    return default

            # 技术指标摘要（最后一根）
            ema_series = ti_tf.get('ema') if isinstance(ti_tf.get('ema'), pd.Series) else None
            rsi_series = ti_tf.get('rsi') if isinstance(ti_tf.get('rsi'), pd.Series) else None
            atr_series = ti_tf.get('atr') if isinstance(ti_tf.get('atr'), pd.Series) else None
            ema_last = _safe_float(ema_series.iloc[-1]) if ema_series is not None and len(ema_series) > 0 else None
            rsi_last = _safe_float(rsi_series.iloc[-1]) if rsi_series is not None and len(rsi_series) > 0 else None
            atr_last = _safe_float(atr_series.iloc[-1]) if atr_series is not None and len(atr_series) > 0 else None
            macd_df = ti_tf.get('macd') if isinstance(ti_tf.get('macd'), pd.DataFrame) else None
            macd_last = {
                'macd': _safe_float(macd_df['macd'].iloc[-1]) if macd_df is not None and not macd_df.empty and 'macd' in macd_df.columns else None,
                'signal': _safe_float(macd_df['signal'].iloc[-1]) if macd_df is not None and not macd_df.empty and 'signal' in macd_df.columns else None,
                'histogram': _safe_float(macd_df['histogram'].iloc[-1]) if macd_df is not None and not macd_df.empty and 'histogram' in macd_df.columns else None,
            }
            bb_df = ti_tf.get('bollinger_bands') if isinstance(ti_tf.get('bollinger_bands'), pd.DataFrame) else None
            bb_last = {
                'upper': _safe_float(bb_df['upper'].iloc[-1]) if bb_df is not None and not bb_df.empty and 'upper' in bb_df.columns else None,
                'middle': _safe_float(bb_df['middle'].iloc[-1]) if bb_df is not None and not bb_df.empty and 'middle' in bb_df.columns else None,
                'lower': _safe_float(bb_df['lower'].iloc[-1]) if bb_df is not None and not bb_df.empty and 'lower' in bb_df.columns else None,
            }
            overall_score = _safe_float(ti_tf.get('overall_score', 0.0), 0.0)

            # 关键水平摘要
            support_lvls = key_levels.get('support', []) if isinstance(key_levels.get('support'), list) else []
            resistance_lvls = key_levels.get('resistance', []) if isinstance(key_levels.get('resistance'), list) else []
            pivots = key_levels.get('pivot_points', {}) if isinstance(key_levels.get('pivot_points'), dict) else {}
            vwap_series = key_levels.get('vwap')
            vwap_last = _safe_float(vwap_series.iloc[-1]) if isinstance(vwap_series, pd.Series) and len(vwap_series) > 0 else _safe_float(key_levels.get('vwap_last'))
            vwap_intraday_poc = _safe_float(key_levels.get('vwap_intraday_poc'))

            # SMC结构摘要
            bos_choch = smc_structures.get('bos_choch', {})
            order_blocks = smc_structures.get('order_blocks', {}) if isinstance(smc_structures.get('order_blocks'), dict) else {'bullish': [], 'bearish': []}
            fvg = smc_structures.get('fvg', {}) if isinstance(smc_structures.get('fvg'), dict) else {'bullish': [], 'bearish': []}
            swing_points = smc_structures.get('swing_points', {}) if isinstance(smc_structures.get('swing_points'), dict) else {'highs': [], 'lows': []}
            smc_score = _safe_float(smc_structures.get('overall_score', 0.0), 0.0)

            # 价格行为摘要
            pa_patterns = price_action.get('candlestick_patterns', {})
            pa_eff = _safe_float(price_action.get('price_efficiency', 0.0), 0.0)
            pa_vol = price_action.get('volatility', {}) if isinstance(price_action.get('volatility'), dict) else {}
            pa_mom = price_action.get('momentum', {}) if isinstance(price_action.get('momentum'), dict) else {}

            # 构建紧凑提示词（避免长列表，提供关键数值）
            # 注意：f字符串内的JSON样例需要避免未转义的大括号，这里使用单独变量插入
            json_schema_hint = (
                "{\n"
                "  \"signal\": \"BUY\" | \"SELL\" | \"HOLD\",\n"
                "  \"confidence\": 0.0-1.0,\n"
                "  \"reasoning\": \"简要推理\",\n"
                "  \"entry_price\": <number|null>,\n"
                "  \"stop_loss\": <number|null>,\n"
                "  \"take_profit\": <number|null>\n"
                "}"
            )

            prompt = f"""
请分析以下市场数据并生成交易信号（如无法确定，请返回HOLD）：

当前价格: {current_price}

技术指标（主TF: {tf or 'N/A'}）:
- EMA_last: {ema_last}
- RSI_14_last: {rsi_last}
- ATR_14_last: {atr_last}
- MACD_last: {macd_last}
- Bollinger_last: {bb_last}
- technical_overall_score: {overall_score}

关键水平:
- 支撑(最多5个): {support_lvls[:5]}
- 阻力(最多5个): {resistance_lvls[:5]}
- 枢轴点(P/R1/S1): {{'P': {pivots.get('pivot')}, 'R1': {pivots.get('r1')}, 'S1': {pivots.get('s1')}}}
- VWAP_last: {vwap_last}
- VWAP_intraday_POC: {vwap_intraday_poc}

SMC结构:
- BOS/CHOCH: {bos_choch}
- OB数量: {{'bullish': {len(order_blocks.get('bullish', []))}, 'bearish': {len(order_blocks.get('bearish', []))}}}
- FVG数量: {{'bullish': {len(fvg.get('bullish', []))}, 'bearish': {len(fvg.get('bearish', []))}}}
- 摆动点数量: {{'highs': {len(swing_points.get('highs', []))}, 'lows': {len(swing_points.get('lows', []))}}}
- smc_overall_score: {smc_score}

价格行为:
- 蜡烛图模式摘要: {pa_patterns}
- 价格效率: {pa_eff}
- 波动性: {pa_vol}
- 动量: {pa_mom}

请严格返回以下JSON结构（仅JSON，无其他文字）：
{json_schema_hint}

补充要求：
- 若为 HOLD 也尽量给出参考入场/止损/止盈（基于关键水平/ATR或最近波动），无法精确时以当前价格附近±1-2%估算；
- 数值统一为数字类型（不要字符串），保留2位小数；
- 入场价须接近当前价格；止损与止盈需符合风险收益逻辑（如RR≥1）。
"""

            return prompt

        except Exception as e:
            # 构建安全降级提示词：携带关键可用信息而不是仅返回通用短语
            try:
                # 兼容非dict输入，避免 .get 调用导致再次异常
                md = market_data if isinstance(market_data, dict) else {}
                cp = md.get('current_price', None)
                kl = md.get('key_levels', {}) or {}
                smc = md.get('smc_structures', {}) or {}
                pa = md.get('price_action', {}) or {}
                supports = kl.get('support', []) if isinstance(kl.get('support'), list) else []
                resistances = kl.get('resistance', []) if isinstance(kl.get('resistance'), list) else []
                bos_choch = smc.get('bos_choch', {}) if isinstance(smc, dict) else {}
                momentum = pa.get('momentum', {}) if isinstance(pa.get('momentum'), dict) else {}
                # 追加审计日志，记录异常与可用摘要
                try:
                    self._append_audit_log({
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "symbol": getattr(self.config, "symbol", ""),
                        "provider": "DeepSeek",
                        "event": "prompt_build_error",
                        "error": str(e),
                        "summary": {
                            "current_price": cp,
                            "support_count": len(supports),
                            "resistance_count": len(resistances),
                            "has_bos_choch": bool(bos_choch),
                            "has_momentum": bool(momentum)
                        }
                    })
                except Exception:
                    pass
                fallback_prompt = (
                    "请分析以下有限市场信息并生成交易信号（无法完整分析时返回HOLD）：\n"
                    f"当前价格: {cp}\n"
                    f"支撑: {supports[:5]}\n"
                    f"阻力: {resistances[:5]}\n"
                    f"SMC-BOS/CHOCH: {bos_choch}\n"
                    f"动量: {momentum}\n"
                    "请严格返回JSON对象：{\"signal\": \"BUY|SELL|HOLD\", \"confidence\": 0-1, \"reasoning\": \"简要推理\"}"
                )
                # 记录详细错误原因以便审计
                self.logger.error(f"提示词构建失败，使用降级提示词。原因: {e}")
                return fallback_prompt
            except Exception:
                # 最终兜底
                self.logger.error(f"提示词构建失败且降级提示词构造异常: {e}")
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
        """获取备用信号（扁平结构）"""
        return {
            "signal": "HOLD",
            "confidence": 0.0,
            "reasoning": reason,
            "provider": "DeepSeek"
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
        """使用OpenAI生成交易信号（返回扁平结构以便于消费）"""
        try:
            # 仅检查密钥是否配置，避免单测因可用性网络探测失败
            if not self.api_key:
                return self._get_fallback_signal("API key not configured")
            
            # 准备提示词
            prompt = self._build_prompt(market_data)
            
            # 使用 openai SDK（符合单测打桩）
            import openai
            try:
                resp = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是一个专业的交易分析师，基于提供的市场数据生成交易信号。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=500
                )
            except Exception as e:
                self.logger.error(f"OpenAI API请求异常: {e}")
                return self._get_fallback_signal("API Error: exception")

            # 解析响应（兼容 dict 与对象两种形式）
            try:
                if isinstance(resp, dict):
                    content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
                else:
                    content = resp.choices[0].message.content
            except Exception:
                return self._get_fallback_signal("API Error: bad response")
            
            # 提取信号并扁平化
            signal = self._extract_signal_from_response(content)
            return {
                "signal": signal.get("signal", "HOLD"),
                "confidence": float(signal.get("confidence", 0.0)),
                "reasoning": signal.get("reasoning", ""),
                "provider": "OpenAI"
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
            vwap_last = None
            vwap_intraday_poc = None
            try:
                vwap_last = market_data.get("key_levels", {}).get("vwap_last")
                vwap_intraday_poc = market_data.get("key_levels", {}).get("vwap_intraday_poc")
            except Exception:
                pass
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
- VWAP_last: {vwap_last}
- VWAP_intraday_POC: {vwap_intraday_poc}

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
        """获取备用信号（扁平结构）"""
        return {
            "signal": "HOLD",
            "confidence": 0.0,
            "reasoning": reason,
            "provider": "OpenAI"
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
            # 支持 list 与 dict 两种 provider 容器
            provider_items: List[Tuple[str, Any]] = []
            if isinstance(self.providers, dict):
                provider_items = list(self.providers.items())
            else:
                provider_items = [(p.__class__.__name__.lower(), p) for p in self.providers]

            # 统一输出结构：所有provider信号标准化为包含嵌套signal对象的形状
            def _normalize_signal(provider_key: str, s: Dict[str, Any]) -> Dict[str, Any]:
                try:
                    # 如果已为嵌套结构
                    if isinstance(s.get("signal"), dict):
                        nested = s.get("signal", {})
                        conf = float(nested.get("confidence", s.get("confidence", 0.0)))
                        return {
                            "provider": s.get("provider", provider_key.capitalize()),
                            "signal": {
                                "signal": nested.get("signal", "HOLD"),
                                "confidence": conf,
                                "reasoning": nested.get("reasoning", s.get("reasoning", "")),
                                "entry_price": nested.get("entry_price", s.get("entry_price")),
                                "stop_loss": nested.get("stop_loss", s.get("stop_loss")),
                                "take_profit": nested.get("take_profit", s.get("take_profit"))
                            },
                            "confidence": conf,
                            "reasoning": s.get("reasoning", nested.get("reasoning", "")),
                            "raw_response": s.get("raw_response", ""),
                            "timestamp": s.get("timestamp", datetime.now(timezone.utc).isoformat())
                        }
                    # 扁平结构 -> 标准化为嵌套
                    conf = float(s.get("confidence", 0.0))
                    return {
                        "provider": s.get("provider", provider_key.capitalize()),
                        "signal": {
                            "signal": s.get("signal", "HOLD"),
                            "confidence": conf,
                            "reasoning": s.get("reasoning", ""),
                            "entry_price": s.get("entry_price"),
                            "stop_loss": s.get("stop_loss"),
                            "take_profit": s.get("take_profit")
                        },
                        "confidence": conf,
                        "reasoning": s.get("reasoning", ""),
                        "raw_response": s.get("raw_response", ""),
                        "timestamp": s.get("timestamp", datetime.now(timezone.utc).isoformat())
                    }
                except Exception:
                    # 保护性降级
                    return {
                        "provider": s.get("provider", provider_key.capitalize()),
                        "signal": {
                            "signal": "HOLD",
                            "confidence": float(s.get("confidence", 0.0)),
                            "reasoning": s.get("reasoning", "")
                        },
                        "confidence": float(s.get("confidence", 0.0)),
                        "reasoning": s.get("reasoning", ""),
                        "raw_response": s.get("raw_response", ""),
                        "timestamp": s.get("timestamp", datetime.now(timezone.utc).isoformat())
                    }

            for provider_name, provider in provider_items:
                try:
                    signal = provider.generate_signal(market_data)
                    signals[provider_name] = _normalize_signal(provider_name, signal)
                except Exception as e:
                    self.logger.error(f"{provider_name}信号生成失败: {e}")
            
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
        """计算信号共识（兼容扁平/嵌套结构）"""
        try:
            signal_counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
            total_confidence = 0
            
            for provider_name, signal_data in signals.items():
                raw_signal = signal_data.get("signal", "HOLD")
                signal = raw_signal.get("signal") if isinstance(raw_signal, dict) else raw_signal
                confidence = signal_data.get("confidence", 0)
                
                signal_counts[signal] += 1
                total_confidence += confidence
            
            # 确定共识信号
            # 将 HOLD 视为中性，仅在 BUY 与 SELL 数量相等时返回 HOLD
            if signal_counts["BUY"] > signal_counts["SELL"]:
                consensus_signal = "BUY"
            elif signal_counts["SELL"] > signal_counts["BUY"]:
                consensus_signal = "SELL"
            else:
                consensus_signal = "HOLD"
            
            # 计算共识置信度
            total_providers = len(signals)
            # 当为 HOLD 时，置信度定义为 0（中性）
            if consensus_signal == "HOLD":
                consensus_confidence = 0.0
            else:
                # 仅统计与共识一致的置信度平均值
                matched_confidences: List[float] = []
                for s in signals.values():
                    rs = s.get("signal", "HOLD")
                    sig = rs.get("signal") if isinstance(rs, dict) else rs
                    if sig == consensus_signal:
                        matched_confidences.append(float(s.get("confidence", 0)))
                consensus_confidence = (sum(matched_confidences) / len(matched_confidences)) if matched_confidences else 0.0
            
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
        """选择主要信号（兼容扁平/嵌套结构）"""
        try:
            # 优先选择与共识一致的信号
            consensus_signal = consensus["signal"]
            
            # 找出与共识一致的信号
            matching_signals = []
            for provider_name, signal_data in signals.items():
                rs = signal_data.get("signal", "HOLD")
                sig = rs.get("signal") if isinstance(rs, dict) else rs
                if sig == consensus_signal:
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
                confidence = float(signal_data.get("confidence", 0))
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

    # ===== 以下为与单测对齐的公开方法 =====
    def generate_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """聚合各提供者信号，返回共识主信号（扁平结构）。"""
        try:
            signals_list: List[Dict[str, Any]] = []

            provider_items: List[Tuple[str, Any]] = []
            if isinstance(self.providers, dict):
                provider_items = list(self.providers.items())
            else:
                provider_items = [(p.__class__.__name__.lower(), p) for p in self.providers]

            for name, provider in provider_items:
                try:
                    s = provider.generate_signal(market_data)
                    signals_list.append(s)
                except Exception as e:
                    self.logger.error(f"{name}生成信号失败: {e}")

            total = len(signals_list)
            if total == 0:
                return {
                    "signal": "HOLD",
                    "confidence": 0.0,
                    "reasoning": "没有可用的AI信号提供者",
                    "consensus": {"agreement": 0, "total_providers": 0, "consensus_ratio": 0.0}
                }

            consensus = self.calculate_consensus(signals_list)
            primary = self.select_primary_signal(signals_list)

            # 当没有共识时，返回 HOLD
            result_signal = consensus["signal"] if consensus["signal"] in ("BUY", "SELL") else "HOLD"
            result_conf = consensus["confidence"] if result_signal != "HOLD" else 0.0

            return {
                "signal": result_signal,
                "confidence": result_conf,
                "reasoning": primary.get("reasoning", ""),
                "consensus": {
                    "agreement": consensus.get("agreement", 0),
                    "total_providers": consensus.get("total_providers", total),
                    "consensus_ratio": consensus.get("consensus_ratio", 0.0)
                }
            }
        except Exception as e:
            self.logger.error(f"AI信号聚合失败: {e}")
            return {
                "signal": "HOLD",
                "confidence": 0.0,
                "reasoning": f"AI信号聚合失败: {str(e)}",
                "consensus": {"agreement": 0, "total_providers": 0, "consensus_ratio": 0.0}
            }

    def calculate_consensus(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """根据各提供者扁平信号计算共识与置信度。"""
        try:
            buy = [s for s in signals if (s.get("signal") == "BUY")]
            sell = [s for s in signals if (s.get("signal") == "SELL")]
            total = len(signals)

            if len(buy) > len(sell):
                signal = "BUY"
                conf = sum(float(s.get("confidence", 0.0)) for s in buy) / len(buy)
                agreement = len(buy)
            elif len(sell) > len(buy):
                signal = "SELL"
                conf = sum(float(s.get("confidence", 0.0)) for s in sell) / len(sell)
                agreement = len(sell)
            else:
                signal = "HOLD"
                conf = 0.0
                agreement = 0

            return {
                "signal": signal,
                "confidence": conf,
                "agreement": agreement,
                "total_providers": total,
                "consensus_ratio": (agreement / total) if total > 0 else 0.0
            }
        except Exception as e:
            self.logger.error(f"共识计算失败: {e}")
            return {
                "signal": "HOLD",
                "confidence": 0.0,
                "agreement": 0,
                "total_providers": len(signals),
                "consensus_ratio": 0.0
            }

    def select_primary_signal(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """选择主信号：优先返回与共识一致的聚合结果，否则返回最高置信度信号。"""
        try:
            consensus = self.calculate_consensus(signals)
            if consensus["signal"] in ("BUY", "SELL"):
                # 聚合一致信号的置信度
                matched = [s for s in signals if s.get("signal") == consensus["signal"]]
                if matched:
                    avg_conf = sum(float(s.get("confidence", 0.0)) for s in matched) / len(matched)
                else:
                    avg_conf = 0.0
                return {
                    "signal": consensus["signal"],
                    "confidence": avg_conf,
                    "provider": "Consensus",
                    "reasoning": matched[0].get("reasoning", "") if matched else ""
                }

            # 无共识：返回最高置信度的信号
            best = None
            best_conf = -1.0
            for s in signals:
                c = float(s.get("confidence", 0.0))
                if c > best_conf:
                    best_conf = c
                    best = s
            return best or {"signal": "HOLD", "confidence": 0.0, "provider": "Consensus"}
        except Exception as e:
            self.logger.error(f"主信号选择失败: {e}")
            return {"signal": "HOLD", "confidence": 0.0, "provider": "Consensus"}

    def get_provider_status(self) -> Dict[str, bool]:
        """返回各提供者可用性状态。"""
        status: Dict[str, bool] = {}
        try:
            if isinstance(self.providers, dict):
                for name, provider in self.providers.items():
                    try:
                        status[name] = bool(provider.is_available()) if hasattr(provider, "is_available") else True
                    except Exception:
                        status[name] = False
            else:
                for p in self.providers:
                    name = p.__class__.__name__.lower()
                    try:
                        status[name] = bool(p.is_available()) if hasattr(p, "is_available") else True
                    except Exception:
                        status[name] = False
        except Exception as e:
            self.logger.error(f"获取提供者状态失败: {e}")
        return status