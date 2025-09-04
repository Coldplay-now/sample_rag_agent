#!/bin/bash

# RAG Agent 启动脚本
# 用于快速启动RAG Agent系统

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# 检查Python版本
check_python() {
    print_info "检查Python环境..."
    
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 未安装，请先安装Python 3.8+"
        exit 1
    fi
    
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    required_version="3.8"
    
    if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
        print_error "Python版本过低，需要Python 3.8+，当前版本: $python_version"
        exit 1
    fi
    
    print_success "Python版本检查通过: $python_version"
}

# 检查虚拟环境
check_venv() {
    print_info "检查虚拟环境..."
    
    if [ ! -d "venv" ]; then
        print_warning "虚拟环境不存在，正在创建..."
        python3 -m venv venv
        print_success "虚拟环境创建完成"
    else
        print_success "虚拟环境已存在"
    fi
}

# 激活虚拟环境
activate_venv() {
    print_info "激活虚拟环境..."
    
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        print_success "虚拟环境已激活"
    else
        print_error "虚拟环境激活脚本不存在"
        exit 1
    fi
}

# 安装依赖
install_dependencies() {
    print_info "检查并安装依赖..."
    
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_success "依赖安装完成"
    else
        print_error "requirements.txt 文件不存在"
        exit 1
    fi
}

# 检查配置文件
check_config() {
    print_info "检查配置文件..."
    
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            print_warning "配置文件不存在，正在复制示例配置..."
            cp .env.example .env
            print_warning "请编辑 .env 文件，填入你的DeepSeek API密钥"
            print_warning "配置完成后重新运行此脚本"
            exit 1
        else
            print_error "配置文件和示例配置都不存在"
            exit 1
        fi
    else
        print_success "配置文件已存在"
        
        # 检查关键配置项
        if ! grep -q "DEEPSEEK_API_KEY=" .env || grep -q "DEEPSEEK_API_KEY=$" .env; then
            print_warning "DeepSeek API密钥未配置，请编辑 .env 文件"
            print_info "获取API密钥: https://platform.deepseek.com/api_keys"
        fi
        
        if ! grep -q "EMBEDDING_MODEL=" .env; then
            print_warning "嵌入模型未配置，将使用默认模型"
        fi
    fi
}

# 检查文档目录
check_documents() {
    print_info "检查文档目录..."
    
    if [ ! -d "data/documents" ]; then
        print_warning "文档目录不存在，正在创建..."
        mkdir -p data/documents
        print_success "文档目录创建完成"
    fi
    
    # 统计各种格式的文档
    md_count=$(find data/documents -name "*.md" -type f 2>/dev/null | wc -l)
    txt_count=$(find data/documents -name "*.txt" -type f 2>/dev/null | wc -l)
    pdf_count=$(find data/documents -name "*.pdf" -type f 2>/dev/null | wc -l)
    total_count=$((md_count + txt_count + pdf_count))
    
    if [ "$total_count" -eq 0 ]; then
        print_warning "文档目录为空，请添加一些文档文件"
        print_info "支持格式: .md, .txt, .pdf"
        print_info "示例：cp your_docs/*.md data/documents/"
    else
        print_success "找到 $total_count 个文档文件 (MD:$md_count, TXT:$txt_count, PDF:$pdf_count)"
    fi
}

# 运行系统测试
run_test() {
    print_info "运行系统测试..."
    
    # 检查是否有测试脚本
    if [ -f "test_error_handling.py" ]; then
        print_info "运行错误处理测试..."
        if python test_error_handling.py; then
            print_success "错误处理测试通过"
        else
            print_warning "错误处理测试部分失败，但系统仍可运行"
        fi
    fi
    
    # 运行主程序测试
    if python main.py --test; then
        print_success "系统测试通过"
        return 0
    else
        print_error "系统测试失败，请检查配置"
        return 1
    fi
}

# 启动RAG Agent
start_rag_agent() {
    print_info "启动RAG Agent..."
    echo
    echo "🚀 RAG Agent 正在启动..."
    echo "💡 使用 Ctrl+C 退出程序"
    echo
    
    python main.py "$@"
}

# 显示帮助信息
show_help() {
    echo "RAG Agent 启动脚本"
    echo
    echo "用法: $0 [选项]"
    echo
    echo "选项:"
    echo "  --help, -h          显示此帮助信息"
    echo "  --test, -t          只运行系统测试"
    echo "  --setup, -s         只进行环境设置"
    echo "  --rebuild, -r       重建文档索引"
    echo "  --config, -c        显示配置信息"
    echo "  --debug, -d         启用调试模式"
    echo "  --docs DIR          指定文档目录"
    echo "  --chat, -i          启动交互式聊天界面"
    echo "  --version, -v       显示版本信息"
    echo
    echo "示例:"
    echo "  $0                  # 正常启动"
    echo "  $0 --test          # 运行测试"
    echo "  $0 --rebuild       # 重建索引后启动"
    echo "  $0 --docs ./mydocs # 使用指定文档目录"
    echo
}

# 主函数
main() {
    echo "🤖 RAG Agent 启动脚本"
    echo "==================="
    echo
    
    # 解析命令行参数
    case "$1" in
        --help|-h)
            show_help
            exit 0
            ;;
        --test|-t)
            check_python
            check_venv
            activate_venv
            install_dependencies
            check_config
            run_test
            exit $?
            ;;
        --setup|-s)
            check_python
            check_venv
            activate_venv
            install_dependencies
            check_config
            check_documents
            print_success "环境设置完成"
            exit 0
            ;;
        --config|-c)
            print_info "当前配置信息:"
            if [ -f ".env" ]; then
                echo "📄 配置文件: .env"
                grep -v "API_KEY" .env | grep "=" || echo "  (配置为空)"
                if grep -q "DEEPSEEK_API_KEY=" .env && ! grep -q "DEEPSEEK_API_KEY=$" .env; then
                    echo "🔑 API密钥: 已配置"
                else
                    echo "🔑 API密钥: 未配置"
                fi
            else
                echo "❌ 配置文件不存在"
            fi
            exit 0
            ;;
        --version|-v)
            echo "RAG Agent v1.0.0"
            echo "基于DeepSeek API和FAISS的检索增强生成系统"
            exit 0
            ;;
        --chat|-i)
            check_python
            check_venv
            activate_venv
            install_dependencies
            check_config
            check_documents
            print_info "启动交互式聊天界面..."
            python chat_interface.py
            exit $?
            ;;
    esac
    
    # 环境检查和设置
    check_python
    check_venv
    activate_venv
    install_dependencies
    check_config
    check_documents
    
    echo
    print_success "环境检查完成，准备启动RAG Agent"
    echo
    
    # 启动程序
    start_rag_agent "$@"
}

# 错误处理
trap 'print_error "脚本执行被中断"; exit 1' INT TERM

# 检查是否在正确的目录
if [ ! -f "main.py" ]; then
    print_error "请在RAG Agent项目根目录下运行此脚本"
    exit 1
fi

# 运行主函数
main "$@"