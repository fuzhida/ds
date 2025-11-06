#!/bin/bash

# DeepSeek AI 交易机器人启动脚本
# 使用方法: ./start.sh [mode]
# mode: live (实盘) 或 sim (模拟)

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查Python版本
check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 未安装，请先安装 Python 3.9+"
        exit 1
    fi
    
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [[ $(echo "$python_version < 3.9" | bc -l) -eq 1 ]]; then
        print_error "Python 版本过低 ($python_version)，需要 3.9+"
        exit 1
    fi
    
    print_success "Python 版本检查通过: $python_version"
}

# 检查依赖
check_dependencies() {
    print_info "检查依赖包..."
    
    if [ ! -f "requirements.txt" ]; then
        print_error "requirements.txt 文件不存在"
        exit 1
    fi
    
    # 检查是否需要安装依赖
    if ! python3 -c "import openai, ccxt, pandas, numpy" &> /dev/null; then
        print_warning "检测到缺失依赖，正在安装..."
        pip3 install -r requirements.txt
    fi
    
    print_success "依赖检查完成"
}

# 检查环境变量
check_env() {
    print_info "检查环境配置..."
    
    if [ ! -f "1.env" ]; then
        if [ -f ".env.example" ]; then
            print_warning "1.env 文件不存在，请复制 .env.example 并配置"
            print_info "运行: cp .env.example 1.env"
        else
            print_error "环境配置文件不存在"
        fi
        exit 1
    fi
    
    # 检查必要的环境变量
    source 1.env
    if [ -z "$DEEPSEEK_API_KEY" ]; then
        print_error "DEEPSEEK_API_KEY 未配置"
        exit 1
    fi
    
    print_success "环境配置检查通过"
}

# 启动机器人
start_bot() {
    local mode=${1:-live}
    
    print_info "启动 DeepSeek AI 交易机器人..."
    print_info "模式: $mode"
    print_info "时间: $(date)"
    
    if [ "$mode" = "sim" ]; then
        print_warning "模拟模式启动"
        # 这里可以添加模拟模式的特殊配置
    else
        print_warning "实盘模式启动 - 请确保已充分测试！"
    fi
    
    # 创建日志目录
    mkdir -p logs
    
    # 启动机器人
    python3 deepseek_hypertest.py
}

# 显示帮助信息
show_help() {
    echo "DeepSeek AI 交易机器人启动脚本"
    echo ""
    echo "使用方法:"
    echo "  ./start.sh [mode]"
    echo ""
    echo "参数:"
    echo "  mode    运行模式 (live|sim)，默认: live"
    echo ""
    echo "示例:"
    echo "  ./start.sh          # 实盘模式"
    echo "  ./start.sh sim      # 模拟模式"
    echo "  ./start.sh --help   # 显示帮助"
    echo ""
}

# 主函数
main() {
    # 显示标题
    echo "=================================================="
    echo "🚀 DeepSeek AI 交易机器人"
    echo "=================================================="
    echo ""
    
    # 处理参数
    case "${1:-}" in
        --help|-h)
            show_help
            exit 0
            ;;
        sim|live|"")
            ;;
        *)
            print_error "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
    
    # 执行检查
    check_python
    check_dependencies
    check_env
    
    # 启动机器人
    start_bot "${1:-live}"
}

# 错误处理
trap 'print_error "脚本执行失败，退出码: $?"' ERR

# 执行主函数
main "$@"