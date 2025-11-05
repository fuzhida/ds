# PAXG交易机器人

基于智能市场结构(SMC)和AI分析的PAXG自动化交易系统。

## 功能特点

- **智能市场结构分析**: 识别BOS、CHOCH、订单块、FVG等关键市场结构
- **多时间框架分析**: 整合多个时间框架的数据进行综合分析
- **AI信号生成**: 集成DeepSeek和OpenAI等AI模型生成交易信号
- **风险管理**: 动态调整仓位大小、止损止盈和风险敞口
- **实时监控**: 自动监控市场变化并执行交易
- **模块化设计**: 清晰的代码结构，易于维护和扩展

## 项目结构

```
├── main.py                 # 主程序入口
├── trading_bot.py         # 核心交易机器人
├── config.py              # 配置管理
├── exchange_manager.py    # 交易所管理
├── smc_analyzer.py        # SMC分析模块
├── technical_analyzer.py   # 技术指标分析
├── ai_signal_generator.py  # AI信号生成
├── risk_manager.py        # 风险管理
├── trading_executor.py    # 交易执行
├── requirements.txt       # 依赖包列表
└── README.md             # 项目说明
```

## 安装与配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 环境变量配置

创建`.env`文件并配置以下变量:

```env
# 交易所配置
EXCHANGE_API_KEY=your_api_key
EXCHANGE_SECRET=your_secret
EXCHANGE_SANDBOX=true

# AI服务配置
OPENAI_API_KEY=your_openai_key
DEEPSEEK_API_KEY=your_deepseek_key

# 交易配置
SYMBOL=PAXG/USDT
BASE_CURRENCY=USDT
QUOTE_CURRENCY=PAXG
```

### 3. 自定义配置

创建`config.json`文件进行高级配置:

```json
{
  "trading": {
    "max_risk_per_trade": 0.02,
    "max_daily_risk": 0.05,
    "max_open_positions": 3,
    "min_signal_confidence": 0.6
  },
  "analysis": {
    "timeframes": ["15m", "1h", "4h", "1d"],
    "analysis_interval": 60
  }
}
```

## 使用方法

### 1. 运行交易机器人

```bash
python main.py --config config.json --log-level INFO
```

### 2. 查看状态

```bash
python main.py --mode status
```

### 3. 手动交易

```bash
# 买入
python main.py --mode manual-trade --signal BUY

# 卖出
python main.py --mode manual-trade --signal SELL
```

### 4. 关闭所有持仓

```bash
python main.py --mode close-all
```

## 命令行参数

- `--config`: 指定配置文件路径
- `--log-level`: 设置日志级别 (DEBUG, INFO, WARNING, ERROR)
- `--mode`: 运行模式 (run, status, manual-trade, close-all)
- `--signal`: 手动交易信号 (BUY, SELL, HOLD)

## 日志与报告

- 日志文件保存在`logs/`目录
- 交易报告保存在`reports/`目录

## 注意事项

1. **风险提示**: 交易有风险，请在充分了解风险后使用本系统
2. **测试建议**: 建议先在模拟环境测试，确认策略有效后再实盘交易
3. **监控**: 请定期检查交易日志和报告，确保系统正常运行
4. **备份**: 定期备份配置文件和交易记录

## 许可证

本项目仅供学习和研究使用。

## 贡献

欢迎提交Issue和Pull Request来改进项目。