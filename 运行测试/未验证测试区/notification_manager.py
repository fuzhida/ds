"""
通知管理模块 - 负责交易信号和系统状态的通知
"""

import smtplib
import requests
import json
import os
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timezone
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


class NotificationManager:
    """通知管理器"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        
        # 邮件配置
        self.smtp_server = getattr(config, 'smtp_server', 'smtp.gmail.com')
        self.smtp_port = getattr(config, 'smtp_port', 587)
        self.email_username = getattr(config, 'email_username', '')
        self.email_password = getattr(config, 'email_password', '')
        self.notification_emails = getattr(config, 'notification_emails', [])
        
        # Webhook配置
        self.webhook_url = getattr(config, 'webhook_url', '')
        self.webhook_enabled = getattr(config, 'webhook_enabled', False)
        
        # 通知配置
        self.notify_on_trade = getattr(config, 'notify_on_trade', True)
        self.notify_on_signal = getattr(config, 'notify_on_signal', False)
        self.notify_on_error = getattr(config, 'notify_on_error', True)
        
        self.logger.info("通知管理器初始化完成")
    
    def send_trade_notification(self, trade_result: Dict[str, Any]) -> bool:
        """发送交易通知"""
        try:
            if not self.notify_on_trade:
                return True
            
            # 构建通知内容
            subject = f"交易通知 - {trade_result.get('signal', 'UNKNOWN')}"
            
            message = f"""
交易执行结果:
- 信号类型: {trade_result.get('signal', 'UNKNOWN')}
- 交易对: {trade_result.get('symbol', 'UNKNOWN')}
- 仓位大小: {trade_result.get('position_size', 0)}
- 入场价格: {trade_result.get('entry_price', 0)}
- 止损价格: {trade_result.get('stop_loss', 0)}
- 止盈价格: {trade_result.get('take_profit', [])}
- 执行时间: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
- 状态: {'成功' if trade_result.get('success', False) else '失败'}
- 原因: {trade_result.get('reason', '未知')}
"""
            
            # 发送邮件通知
            email_sent = self._send_email(subject, message)
            
            # 发送Webhook通知
            webhook_sent = self._send_webhook({
                'type': 'trade',
                'data': trade_result,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
            
            return email_sent or webhook_sent
            
        except Exception as e:
            self.logger.error(f"发送交易通知失败: {e}")
            return False
    
    def send_signal_notification(self, signal: Dict[str, Any]) -> bool:
        """发送信号通知"""
        try:
            if not self.notify_on_signal:
                return True
            
            # 构建通知内容
            subject = f"信号通知 - {signal.get('signal', 'HOLD')}"
            
            message = f"""
市场分析信号:
- 信号类型: {signal.get('signal', 'HOLD')}
- 置信度: {signal.get('confidence', 0):.2f}
- 交易对: {signal.get('symbol', 'UNKNOWN')}
- 分析时间: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
- 推理过程: {signal.get('reasoning', '无')}
"""
            
            # 添加组件信号
            components = signal.get('components', {})
            if components:
                message += "\n组件信号:\n"
                for comp_name, comp_data in components.items():
                    message += f"- {comp_name}: {comp_data.get('signal', 'UNKNOWN')} ({comp_data.get('confidence', 0):.2f})\n"
            
            # 发送邮件通知
            email_sent = self._send_email(subject, message)
            
            # 发送Webhook通知
            webhook_sent = self._send_webhook({
                'type': 'signal',
                'data': signal,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
            
            return email_sent or webhook_sent
            
        except Exception as e:
            self.logger.error(f"发送信号通知失败: {e}")
            return False
    
    def send_error_notification(self, error_message: str, error_type: str = "ERROR") -> bool:
        """发送错误通知"""
        try:
            if not self.notify_on_error:
                return True
            
            # 构建通知内容
            subject = f"系统错误通知 - {error_type}"
            
            message = f"""
系统错误报告:
- 错误类型: {error_type}
- 错误信息: {error_message}
- 发生时间: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
"""
            
            # 发送邮件通知
            email_sent = self._send_email(subject, message)
            
            # 发送Webhook通知
            webhook_sent = self._send_webhook({
                'type': 'error',
                'error_type': error_type,
                'error_message': error_message,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
            
            return email_sent or webhook_sent
            
        except Exception as e:
            self.logger.error(f"发送错误通知失败: {e}")
            return False
    
    def send_system_status(self, status: Dict[str, Any]) -> bool:
        """发送系统状态通知"""
        try:
            # 构建通知内容
            subject = "系统状态报告"
            
            message = f"""
系统状态报告:
- 运行状态: {'运行中' if status.get('is_running', False) else '已停止'}
- 暂停状态: {'已暂停' if status.get('is_paused', False) else '正常'}
- 最后信号: {status.get('last_signal', {}).get('signal', 'HOLD')}
- 最后分析时间: {status.get('last_analysis_time', '未知')}
- 开仓数量: {status.get('open_positions_count', 0)}
- 报告时间: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
"""
            
            # 添加风险摘要
            risk_summary = status.get('risk_summary', {})
            if risk_summary:
                message += "\n风险摘要:\n"
                message += f"- 每日风险剩余: {risk_summary.get('daily_risk_remaining', 0):.2%}\n"
                message += f"- 最大回撤: {risk_summary.get('max_drawdown', 0):.2%}\n"
            
            # 发送邮件通知
            email_sent = self._send_email(subject, message)
            
            # 发送Webhook通知
            webhook_sent = self._send_webhook({
                'type': 'status',
                'data': status,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
            
            return email_sent or webhook_sent
            
        except Exception as e:
            self.logger.error(f"发送系统状态通知失败: {e}")
            return False
    
    def _send_email(self, subject: str, message: str) -> bool:
        """发送邮件通知"""
        try:
            if not self.email_username or not self.email_password or not self.notification_emails:
                self.logger.debug("邮件配置不完整，跳过邮件通知")
                return False
            
            # 创建邮件
            msg = MIMEMultipart()
            msg['Subject'] = subject
            msg['From'] = self.email_username
            msg['To'] = ', '.join(self.notification_emails)
            
            # 添加邮件正文
            msg.attach(MIMEText(message, 'plain'))
            
            # 连接SMTP服务器并发送邮件
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_username, self.email_password)
                server.send_message(msg)
            
            self.logger.info(f"邮件通知已发送: {subject}")
            return True
            
        except Exception as e:
            self.logger.error(f"发送邮件通知失败: {e}")
            return False
    
    def _send_webhook(self, data: Dict[str, Any]) -> bool:
        """发送Webhook通知"""
        try:
            if not self.webhook_enabled or not self.webhook_url:
                self.logger.debug("Webhook配置不完整，跳过Webhook通知")
                return False
            
            # 发送POST请求
            response = requests.post(
                self.webhook_url,
                json=data,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            # 检查响应
            if response.status_code == 200:
                self.logger.info(f"Webhook通知已发送: {data.get('type', 'unknown')}")
                return True
            else:
                self.logger.error(f"Webhook通知发送失败: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"发送Webhook通知失败: {e}")
            return False
    
    def test_notifications(self) -> Dict[str, bool]:
        """测试通知功能"""
        try:
            results = {}
            
            # 测试邮件通知
            email_result = self._send_email(
                "测试通知",
                "这是一条测试通知，用于验证邮件通知功能是否正常工作。"
            )
            results['email'] = email_result
            
            # 测试Webhook通知
            webhook_result = self._send_webhook({
                'type': 'test',
                'message': '这是一条测试通知，用于验证Webhook通知功能是否正常工作。',
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
            results['webhook'] = webhook_result
            
            return results
            
        except Exception as e:
            self.logger.error(f"测试通知功能失败: {e}")
            return {'email': False, 'webhook': False}