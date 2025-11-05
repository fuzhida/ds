# Makefile for PAXG Trading Bot

.PHONY: help test test-unit test-integration test-specific lint format install clean run

# 默认目标
help:
	@echo "PAXG Trading Bot - 可用命令:"
	@echo "  test          - 运行所有测试"
	@echo "  test-unit     - 运行单元测试"
	@echo "  test-integration - 运行集成测试"
	@echo "  test-specific MODULE=模块名 - 运行特定测试模块"
	@echo "  lint          - 运行代码检查"
	@echo "  format        - 格式化代码"
	@echo "  install       - 安装依赖"
	@echo "  clean         - 清理临时文件"
	@echo "  run           - 运行交易机器人"

# 运行所有测试
test:
	python run_tests.py

# 运行单元测试
test-unit:
	python -m pytest tests/ -m "unit or not integration" -v

# 运行集成测试
test-integration:
	python -m pytest tests/ -m "integration" -v

# 运行特定测试模块
test-specific:
	python run_tests.py $(MODULE)

# 代码检查
lint:
	@echo "运行代码检查..."
	ruff check .
	@echo "代码检查完成"

# 代码格式化
format:
	@echo "格式化代码..."
	ruff format .
	@echo "代码格式化完成"

# 安装依赖
install:
	@echo "安装依赖..."
	pip install -r requirements.txt
	@echo "依赖安装完成"

# 清理临时文件
clean:
	@echo "清理临时文件..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	@echo "清理完成"

# 运行交易机器人
run:
	@echo "启动交易机器人..."
	python main.py

# 运行交易机器人（沙盒模式）
run-sandbox:
	@echo "启动交易机器人（沙盒模式）..."
	python main.py --sandbox

# 运行交易机器人（测试模式）
run-test:
	@echo "启动交易机器人（测试模式）..."
	python main.py --test-mode