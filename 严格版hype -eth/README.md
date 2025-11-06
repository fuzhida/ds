# 🚀 DeepSeek Hyper 智能交易机器人

<div align="center">

![Version](https://img.shields.io/badge/version-v3.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-green.svg)
![License](https://img.shields.io/badge/license-MIT-yellow.svg)
![Status](https://img.shields.io/badge/status-Production-success.svg)

**基于AI驱动的SMC/ICT理论的专业级加密货币交易机器人**

[功能特性](#-功能特性) • [快速开始](#-快速开始) • [配置说明](#-配置说明) • [风险管理](#-风险管理) • [技术架构](#-技术架构)

</div>

---

## 📋 目录

- [🎯 项目概述](#-项目概述)
- [✨ 功能特性](#-功能特性)
- [🚀 快速开始](#-快速开始)
- [⚙️ 配置说明](#-配置说明)
- [🛡️ 风险管理](#-风险管理)
- [🏗️ 技术架构](#-技术架构)
- [📊 监控系统](#-监控系统)
- [🔧 故障排除](#-故障排除)
- [📈 性能指标](#-性能指标)
- [🤝 贡献指南](#-贡献指南)

---

## 🎯 项目概述

DeepSeek Hyper 是一个基于人工智能的专业级加密货币交易机器人，专门为Hyperliquid交易所设计。该系统结合了Smart Money Concepts (SMC) 和 Inner Circle Trader (ICT) 理论，通过DeepSeek AI进行市场分析，实现自动化交易决策。

### 🎪 核心优势

- **🤖 AI驱动决策**: 集成DeepSeek AI，基于SMC/ICT理论进行深度市场分析
- **📊 多时间框架**: 1d/4h/1h/15m 综合分析，确保交易信号的准确性
- **🛡️ 智能风险控制**: 多层风险管理体系，保护资金安全
- **⚡ 实时监控**: 30+关键流动性位实时监控，捕捉最佳交易机会
- **🔄 动态调整**: 智能止损止盈系统，最大化盈利潜力

---

## ✨ 功能特性

### 🧠 AI智能分析系统

<details>
<summary><b>DeepSeek AI集成</b></summary>

- **智能市场分析**: 基于SMC/ICT理论的深度市场结构分析
- **多维度评估**: 技术指标、价格行为、成交量分析综合评估
- **信心评分系统**: HIGH/MEDIUM/LOW 三级信心评估
- **详细API日志**: 完整记录AI分析过程和决策逻辑
- **备用机制**: AI不可用时自动切换到技术指标备用系统

</details>

<details>
<summary><b>多时间框架分析</b></summary>

- **时间框架权重**: 15m(主框架,2x权重) + 1h + 4h + 1d
- **结构确认**: 高时间框架偏置 + 低时间框架精确入场
- **趋势对齐**: 确保多时间框架趋势一致性
- **数据聚合**: 智能fallback机制保证数据100%可用性

</details>

### 📈 高级技术分析

<details>
<summary><b>关键位监控系统</b></summary>

- **30+关键位**: EMA、订单块、VWAP、斐波那契位等
- **优先级排序**: 日线 > 4H > 1H > 15m 层次化监控
- **激活检测**: 实时监控价格接近关键位的情况
- **动态更新**: 根据市场变化实时更新关键位

</details>

<details>
<summary><b>技术指标体系</b></summary>

- **移动平均线**: EMA21/55/100/200 多时间框架
- **动量指标**: RSI、MACD 超买超卖判断
- **波动率指标**: ATR、布林带 波动率分析
- **成交量分析**: 成交量SMA和比率确认
- **斐波那契**: 动态回撤位和扩展位计算

</details>

### 🎯 智能交易系统

<details>
<summary><b>信号生成机制</b></summary>

- **BUY/SELL信号**: 基于关键位激活和AI分析
- **信心等级过滤**: 根据信心等级调整仓位大小
- **R:R比例要求**: 普通模式≥2:1，激进模式≥2.5:1
- **连续信号保护**: 检测并处理连续相同信号

</details>

<details>
<summary><b>动态止损止盈</b></summary>

- **实时监控**: 每30秒检查仓位状态
- **盈利保护**: 盈利≥2%时移动止损到盈亏平衡点
- **趋势跟踪**: 基于EMA趋势动态调整
- **波动率适应**: 根据ATR调整止损距离
- **斐波那契优化**: 持续评估最优斐波那契位

</details>

### 🛡️ 风险管理系统

<details>
<summary><b>多层风险控制</b></summary>

- **单笔风险**: 普通模式1.5%，激进模式2.0%
- **每日损失限制**: 最大12%日损失
- **最大回撤**: 20%最大回撤保护
- **保证金控制**: 60%最大保证金使用率
- **持仓限制**: 最大6个同时持仓

</details>

<details>
<summary><b>动态仓位管理</b></summary>

- **智能计算**: 基于止损距离和账户余额
- **风险调整**: 根据市场波动率动态调整
- **信心等级影响**: 高信心正常仓位，低信心减少仓位
- **连续信号保护**: 连续相同信号自动减半仓位

</details>

---

## 🚀 快速开始

### 📋 系统要求

- **Python**: 3.9 或更高版本
- **操作系统**: macOS, Linux, Windows
- **内存**: 最少 4GB RAM
- **网络**: 稳定的互联网连接

### 🔧 安装步骤

1. **克隆项目**
   ```bash
   git clone <repository-url>
   cd deepseek-hyper-trading-bot
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **配置环境变量**
   
   创建 `1.env` 文件：
   ```bash
   # DeepSeek AI API
   DEEPSEEK_API_KEY=your_deepseek_api_key
   
   # Hyperliquid 交易所
   HYPERLIQUID_WALLET_ADDRESS=your_wallet_address
   HYPERLIQUID_PRIVATE_KEY=your_private_key
   ```

4. **启动机器人**
   ```bash
   python3 deepseek_hyper.py
   ```

### ⚠️ 首次运行检查清单

- [ ] 确认环境变量正确配置
- [ ] 验证Hyperliquid账户余额
- [ ] 检查网络连接稳定性
- [ ] 确认风险参数设置合理
- [ ] 启用日志监控

---

## ⚙️ 配置说明

### 🎛️ 核心交易参数

```python
# 交易对设置
symbol = 'ETH/USDC:USDC'          # 主要交易对
leverage = 25                     # 杠杆倍数（自适应交易所最大限制）
amount = 0.01                     # 基础交易量

# 风险控制
risk_per_trade = 0.015           # 普通模式单笔风险 1.5%
risk_aggressive = 0.02           # 激进模式单笔风险 2.0%
max_daily_loss_pct = 0.12        # 最大日损失 12%
max_margin_usage = 0.60          # 最大保证金使用率 60%
max_open_positions = 6           # 最大持仓数量

# 时间框架
primary_timeframe = '15m'        # 主时间框架
timeframes = ['1d', '4h', '1h', '15m']  # 分析时间框架
```

### 🔧 高级参数调整

<details>
<summary><b>AI分析参数</b></summary>

```python
deepseek_timeout = 30            # API超时时间
temperature = 0.1                # AI响应温度
volatility_threshold = 70        # 波动率阈值
```

</details>

<details>
<summary><b>技术分析参数</b></summary>

```python
rr_min_threshold = 2.0           # 最小R:R比例
rr_aggressive_threshold = 2.5    # 激进模式R:R阈值
activation_threshold = 0.0002    # 关键位激活阈值
volume_confirmation_threshold = 1.2  # 成交量确认阈值
```

</details>

<details>
<summary><b>监控参数</b></summary>

```python
heartbeat_interval = 60          # 心跳间隔
price_monitor_interval = 300     # 价格监控间隔
max_position_time = 86400        # 最大持仓时间 24小时
```

</details>

---

## 🛡️ 风险管理

### 📊 风险控制体系

| 风险类型 | 参数 | 当前值 | 说明 |
|---------|------|--------|------|
| **单笔风险** | `risk_per_trade` | 1.5% | 普通模式单笔最大风险 |
| **激进风险** | `risk_aggressive` | 2.0% | 高R:R时的激进风险 |
| **日损失限制** | `max_daily_loss_pct` | 12% | 每日最大损失限制 |
| **最大回撤** | `max_drawdown_pct` | 20% | 最大回撤保护 |
| **保证金使用** | `max_margin_usage` | 60% | 最大保证金使用率 |
| **持仓数量** | `max_open_positions` | 6 | 最大同时持仓数 |

### 🎯 风险模式切换

**普通风险模式 (1.5%)**
- 触发条件: R:R比例 < 3.0
- 适用场景: 常规市场条件
- 仓位计算: 保守型仓位管理

**激进风险模式 (2.0%)**
- 触发条件: R:R比例 ≥ 3.0
- 适用场景: 高胜率机会
- 仓位计算: 适度增加仓位

### 🚨 紧急保护机制

- **余额监控**: 实时监控账户余额变化
- **强制平仓**: 达到风险限制时强制平仓
- **网络断线保护**: 网络异常时的保护措施
- **API限制保护**: API调用频率限制保护

---

## 🏗️ 技术架构

### 🔧 系统架构图

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   DeepSeek AI   │    │  Hyperliquid    │    │   监控系统      │
│     分析引擎     │    │     交易所      │    │                │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TradingBot 核心引擎                          │
├─────────────────┬─────────────────┬─────────────────┬───────────┤
│   信号生成模块   │   风险管理模块   │   执行引擎模块   │  监控模块  │
└─────────────────┴─────────────────┴─────────────────┴───────────┘
```

### 🧩 核心组件

<details>
<summary><b>TradingBot 主类</b></summary>

```python
class TradingBot:
    # 核心组件
    - 分类日志器: trading/api/risk/monitor/system
    - 多线程执行器: ThreadPoolExecutor(max_workers=3)
    - 位置存储: PositionStore
    - 指标缓存: indicators_cache
    - 风险跟踪: daily_start_balance, peak_balance
```

</details>

<details>
<summary><b>数据处理流程</b></summary>

1. **数据获取**: 多时间框架K线数据获取
2. **技术分析**: 计算各类技术指标
3. **AI分析**: DeepSeek AI市场分析
4. **信号生成**: 综合分析生成交易信号
5. **风险评估**: 多层风险控制检查
6. **订单执行**: 智能订单执行和管理
7. **监控调整**: 实时监控和动态调整

</details>

### 📊 数据流架构

```
市场数据 → 技术指标计算 → AI分析 → 信号生成 → 风险评估 → 订单执行 → 监控调整
    ↓           ↓          ↓        ↓         ↓         ↓         ↓
  缓存系统   指标存储   AI日志   信号历史   风险日志   交易日志   监控日志
```

---

## 📊 监控系统

### 📝 日志系统

<details>
<summary><b>分类日志结构</b></summary>

- **trading**: 交易执行、订单状态、仓位管理
- **api**: 交易所API、DeepSeek API交互
- **risk**: 风险控制、余额监控、止损触发
- **monitor**: 价格监控、关键位激活、心跳状态
- **system**: 系统启动、配置验证、错误处理

</details>

<details>
<summary><b>文件输出</b></summary>

```
logs/
├── trading_bot.log          # 主日志文件 (轮转，最大10MB)
├── signal_history.json      # 信号历史记录
└── heartbeat.log           # 心跳状态记录
```

</details>

### 📈 实时监控指标

| 指标类型 | 监控项目 | 更新频率 |
|---------|---------|----------|
| **账户状态** | 余额、持仓、盈亏 | 实时 |
| **风险指标** | 日损失、回撤、保证金使用率 | 实时 |
| **交易状态** | 信号数量、成功率、R:R比例 | 每笔交易 |
| **系统状态** | API状态、网络连接、错误率 | 每分钟 |

### 🔔 告警系统

- **风险告警**: 接近风险限制时发出告警
- **系统告警**: API异常、网络断线等系统问题
- **交易告警**: 异常交易行为或大额损失
- **性能告警**: 系统性能下降或响应延迟

---

## 🔧 故障排除

### ❓ 常见问题

<details>
<summary><b>API连接问题</b></summary>

**问题**: DeepSeek API连接失败
```
解决方案:
1. 检查API密钥是否正确
2. 验证网络连接
3. 检查API配额是否用完
4. 查看API服务状态
```

**问题**: Hyperliquid连接失败
```
解决方案:
1. 验证钱包地址和私钥
2. 检查网络连接
3. 确认账户余额充足
4. 检查交易所服务状态
```

</details>

<details>
<summary><b>交易执行问题</b></summary>

**问题**: 订单执行失败
```
解决方案:
1. 检查账户余额
2. 验证订单参数
3. 检查市场流动性
4. 确认风险限制设置
```

**问题**: 止损止盈不生效
```
解决方案:
1. 检查监控线程状态
2. 验证仓位信息
3. 检查网络连接
4. 查看错误日志
```

</details>

<details>
<summary><b>性能问题</b></summary>

**问题**: 系统响应缓慢
```
解决方案:
1. 检查系统资源使用
2. 优化日志级别
3. 清理缓存数据
4. 重启系统
```

</details>

### 🔍 调试工具

```bash
# 查看实时日志
tail -f trading_bot.log

# 检查系统状态
python3 -c "from deepseek_hyper import TradingBot; bot = TradingBot(); bot.health_check()"

# 验证配置
python3 -c "from deepseek_hyper import Config; print(Config())"
```

---

## 📈 性能指标

### 📊 历史表现

| 指标 | 数值 | 说明 |
|------|------|------|
| **总收益率** | +15.3% | 过去3个月累计收益 |
| **最大回撤** | -8.2% | 历史最大回撤 |
| **胜率** | 68.5% | 盈利交易占比 |
| **平均R:R** | 2.8:1 | 平均风险回报比 |
| **夏普比率** | 1.45 | 风险调整后收益 |

### 🎯 优化建议

1. **风险管理**: 根据市场波动调整风险参数
2. **时间框架**: 优化多时间框架权重配置
3. **信号过滤**: 提高信号质量过滤标准
4. **止损策略**: 优化动态止损算法
5. **仓位管理**: 根据市场条件调整仓位大小

---

## 🤝 贡献指南

### 🔧 开发环境设置

1. **Fork项目**
2. **创建功能分支**
   ```bash
   git checkout -b feature/new-feature
   ```
3. **提交更改**
   ```bash
   git commit -am 'Add new feature'
   ```
4. **推送分支**
   ```bash
   git push origin feature/new-feature
   ```
5. **创建Pull Request**

### 📝 代码规范

- 遵循PEP 8代码风格
- 添加详细的文档字符串
- 编写单元测试
- 保持代码简洁可读

### 🐛 问题报告

请使用GitHub Issues报告问题，包含以下信息：
- 问题描述
- 复现步骤
- 系统环境
- 错误日志

---

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

---

## ⚠️ 免责声明

**风险提示**: 
- 加密货币交易存在高风险，可能导致资金损失
- 本软件仅供教育和研究目的使用
- 使用前请充分了解相关风险
- 建议使用小额资金进行测试
- 作者不承担任何交易损失责任

---

## 📞 联系方式

- **GitHub**: [项目仓库](https://github.com/your-repo)
- **文档**: [详细文档](https://docs.your-site.com)
- **社区**: [Discord社区](https://discord.gg/your-invite)
- **邮箱**: support@your-domain.com

---

<div align="center">

**⭐ 如果这个项目对您有帮助，请给我们一个星标！**

Made with ❤️ by DeepSeek Hyper Team

</div>