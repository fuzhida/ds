# 🚀 DeepSeek AI 交易机器人

一个基于 DeepSeek AI 和 Hyperliquid 交易所的智能加密货币交易机器人，采用 SMC/ICT 交易策略，支持多时间框架分析和动态风险管理。

## ✨ 主要特性

### 🤖 AI 驱动的交易决策
- **DeepSeek AI 集成**: 使用先进的 AI 模型进行市场分析和信号生成
- **智能信号生成**: 结合技术指标和 AI 分析的混合策略
- **备用信号机制**: 当 AI 不可用时自动切换到基于规则的信号

### 📊 专业交易策略
- **SMC/ICT 策略**: Smart Money Concepts 和 Inner Circle Trader 方法论
- **多时间框架分析**: 支持 15m、1h、4h、1d 等多个时间框架
- **关键水平识别**: 自动识别支撑阻力、订单块、公允价值缺口等
- **Kill Zone 交易**: 在最佳交易时间窗口内执行交易

### 🛡️ 风险管理系统
- **动态杠杆**: 根据风险回报比自动调整杠杆
- **止损止盈**: 智能设置止损和止盈水平
- **资金管理**: 严格的仓位大小控制和风险限制
- **最大回撤控制**: 防止过度亏损的保护机制

### 🔧 技术特性
- **实时监控**: 价格监控、心跳检测、系统健康检查
- **日志系统**: 分类日志记录，便于调试和监控
- **回测功能**: 历史数据回测和性能分析
- **容错设计**: 网络异常重试、API 错误处理

## 📋 系统要求

- Python 3.9+
- 稳定的网络连接
- Hyperliquid 交易账户
- DeepSeek API 密钥

## 🛠️ 安装配置

### 1. 克隆项目
```bash
git clone <repository-url>
cd ds交易/ds/严格版
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 环境配置
创建 `1.env` 文件并配置以下环境变量：

```env
# DeepSeek AI API
DEEPSEEK_API_KEY=your_deepseek_api_key

# Hyperliquid 交易所
HYPERLIQUID_WALLET_ADDRESS=your_wallet_address
HYPERLIQUID_PRIVATE_KEY=your_private_key
```

### 4. 配置参数
编辑 `deepseek_hypertest.py` 中的 `Config` 类来调整交易参数：

```python
@dataclass
class Config:
    symbol: str = 'ETH/USDC:USDC'          # 交易对
    leverage: int = 25                      # 杠杆倍数
    risk_per_trade: float = 0.015          # 每笔交易风险 (1.5%)
    activation_threshold: float = 0.001     # 价格激活阈值 (0.1%)
    simulation_mode: bool = False           # 模拟模式开关
```

## 🚀 使用方法

### 启动交易机器人
```bash
python3 deepseek_hypertest.py
```

### 模拟模式测试
```python
# 在 Config 类中设置
simulation_mode: bool = True
```

### 回测功能
```python
# 使用历史数据进行回测
bot.backtest_from_file('historical_data.csv')
```

## 📊 监控和日志

### 日志文件
- `trading_bot.log`: 主要交易日志
- `heartbeat.log`: 系统心跳记录
- `signal_history.json`: 交易信号历史

### 实时监控
机器人提供以下监控功能：
- 💓 系统心跳 (每60秒)
- 📈 价格监控 (每3分钟)
- 🔍 关键水平检测
- ⚡ 动态止损止盈

### 日志示例
```
2025-10-27 16:57:35 [system] - INFO - Config validation successful: ETH/USDC:USDC | Leverage=25x | Risk=1.5% | Mode=Live
2025-10-27 16:57:35 [monitor] - INFO - 💓 Heartbeat: 16:57:35 | Position=No position | Balance=1001.97 USDC | Price=$4163.55
```

## ⚙️ 配置参数详解

### 基础配置
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `symbol` | ETH/USDC:USDC | 交易对 |
| `leverage` | 25 | 杠杆倍数 |
| `amount` | 0.01 | 基础交易数量 |
| `risk_per_trade` | 0.015 | 每笔交易风险比例 |

### 风险管理
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_margin_usage` | 0.60 | 最大保证金使用率 |
| `max_daily_loss_pct` | 0.12 | 最大日亏损比例 |
| `max_drawdown_pct` | 0.20 | 最大回撤比例 |
| `min_amount_usdc` | 50.0 | 最小持仓金额 |

### 技术指标
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `activation_threshold` | 0.001 | 价格激活阈值 |
| `rr_min_threshold` | 2.0 | 最小风险回报比 |
| `volume_confirmation_threshold` | 1.2 | 成交量确认阈值 |

### 时间设置
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `kill_zone_start_utc` | 8 | Kill Zone 开始时间 (UTC) |
| `kill_zone_end_utc` | 16 | Kill Zone 结束时间 (UTC) |
| `heartbeat_interval` | 60 | 心跳间隔 (秒) |
| `price_monitor_interval` | 180 | 价格监控间隔 (秒) |

## 🔒 安全注意事项

### API 密钥安全
- ✅ 使用环境变量存储敏感信息
- ✅ 不要将 API 密钥提交到版本控制
- ✅ 定期轮换 API 密钥
- ✅ 使用只读权限的 API 密钥进行测试

### 交易风险
- ⚠️ 加密货币交易存在高风险
- ⚠️ 建议先在模拟模式下测试
- ⚠️ 不要投入超过承受能力的资金
- ⚠️ 定期监控机器人运行状态

### 系统安全
- 🔐 确保服务器安全
- 🔐 使用防火墙保护
- 🔐 定期备份配置和日志
- 🔐 监控异常活动

## 🐛 故障排除

### 常见问题

#### 1. API 连接失败
```bash
# 检查网络连接
ping api.deepseek.com

# 验证 API 密钥
python3 -c "import os; from dotenv import load_dotenv; load_dotenv('1.env'); print('API Key:', os.getenv('DEEPSEEK_API_KEY')[:10] + '...')"
```

#### 2. 交易执行失败
- 检查账户余额是否充足
- 验证交易对是否正确
- 确认杠杆设置是否合理

#### 3. 价格数据异常
- 检查交易所 API 状态
- 验证网络连接稳定性
- 查看日志中的错误信息

### 日志分析
```bash
# 查看最新日志
tail -f trading_bot.log

# 搜索错误信息
grep "ERROR" trading_bot.log

# 查看心跳状态
tail -f heartbeat.log
```

## 🔍 数据真实性保障措施

### 已实现的数据真实性保护
- ✅ **固定数值模式检测**: 实时检测并防止固定数值模式污染系统
- ✅ **动态计算方法**: 基于真实市场数据的多因子动态计算机制
- ✅ **智能备用检测**: 当检测到固定模式时自动切换到智能计算
- ✅ **增强验证机制**: 多层次数据验证确保结果真实性

### 技术实现细节

#### 1. 固定数值模式检测
```python
def _detect_fixed_value_pattern(self, bos_strength: float, fvg_count: int, ob_count: int) -> bool:
    """检测固定数值模式，防止数据污染"""
    # 预设固定值列表
    fixed_bos_values = [0.7, 3.0, 2.5]
    fixed_fvg_values = [20, 29]
    fixed_ob_values = [8]
    
    # 检测固定模式
    if (bos_strength in fixed_bos_values or 
        fvg_count in fixed_fvg_values or 
        ob_count in fixed_ob_values):
        self.logger_system.error(f"🚨 检测到固定BOS强度模式: {bos_strength}，数据真实性严重受损！")
        return True
    return False
```

#### 2. 动态计算方法
- **BOS强度计算**: 基于价格变化、波动性、时间和价格因子
- **FVG数量计算**: 基于价格范围、时间和范围因子  
- **OB数量计算**: 基于成交量、时间和价格因子

#### 3. 智能备用检测
当检测到固定模式时，系统自动切换到基于真实数据的智能计算，确保不返回固定值。

### 验证机制
- ✅ 实时监控数据质量
- ✅ 异常值检测和处理
- ✅ 多维度数据验证
- ✅ 详细的日志记录和调试信息

## 📈 性能优化

### 已实现的优化
- ✅ 智能缓存机制减少 API 调用
- ✅ 异步处理提高响应速度
- ✅ 日志级别优化减少噪音
- ✅ 内存使用优化
- ✅ 网络重试机制

### 建议的优化
- 🔄 定期清理历史日志
- 🔄 监控系统资源使用
- 🔄 优化数据库查询
- 🔄 使用更快的网络连接

## 🤝 贡献指南

### 开发环境设置
```bash
# 安装开发依赖
pip install -r requirements-dev.txt

# 运行测试
python3 -m pytest tests/

# 代码格式化
black deepseek_hypertest.py
```

### 提交规范
- 使用清晰的提交信息
- 添加适当的测试用例
- 更新相关文档
- 遵循代码风格指南

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## ⚠️ 免责声明

本软件仅供教育和研究目的使用。使用本软件进行实际交易的风险由用户自行承担。开发者不对任何交易损失负责。

**重要提醒**：
- 加密货币交易存在极高风险
- 过去的表现不代表未来结果
- 请在充分了解风险的情况下使用
- 建议咨询专业的财务顾问

## 📞 支持和联系

如有问题或建议，请通过以下方式联系：

- 📧 Email: [your-email@example.com]
- 💬 Issues: [GitHub Issues](https://github.com/your-repo/issues)
- 📖 文档: [项目文档](https://your-docs-url.com)

---

**最后更新**: 2025-10-31  
**版本**: v1.1.0  
**状态**: 🟢 稳定运行 (已增强数据真实性保护)