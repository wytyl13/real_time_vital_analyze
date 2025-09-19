#!/bin/bash
if [ -f ".env" ]; then
    API_PORT=$(python -c "
import os
from dotenv import load_dotenv
load_dotenv()
print(os.getenv('API_PORT', '8891'))
")
    CONDA_ENV_PATH=$(python -c "
import os
from dotenv import load_dotenv
load_dotenv()
print(os.getenv('CONDA_ENV_PATH', '/work/soft/anaconda3/bin/'))
")
    CONDA_ENVIRONMENT=$(python -c "
import os
from dotenv import load_dotenv
load_dotenv()
print(os.getenv('CONDA_ENVIRONMENT', 'real_time_vital_analyze'))
")
else
    API_PORT=8891
    CONDA_ENV_PATH='/work/soft/anaconda3/bin/'
    CONDA_ENVIRONMENT='real_time_vital_analyze'
fi

# 从命令行参数获取 PROJECT_ROOT，如果未提供，则使用现有方式
if [ -n "$1" ]; then
    PROJECT_ROOT="$1"
else
    PROJECT_ROOT=$(dirname "$(readlink -f "$0")")
fi

check_and_kill_port() {
    local port=$1
    local pid=$(lsof -t -i :$port)

    if [ -n "$pid" ]; then
        echo "端口 $port 已被占用，正在终止进程 $pid"
        kill $pid
    else
        echo "端口 $port 未被占用"
    fi
}

check_and_kill_port $API_PORT
check_and_kill_port 5003

# 激活虚拟环境
CONDA_ENV=$CONDA_ENV_PATH
export PATH=$CONDA_ENV:$PATH
eval "$(conda shell.bash hook)"
conda init bash
conda activate $CONDA_ENVIRONMENT

timestamp=$(date +"%Y%m%d%H%M%S")
LOG_PATH=$PROJECT_ROOT/logs/api_server
LOCAL_FILE_SERVER_LOG_PATH=$PROJECT_ROOT/logs/local_file_server

LOG_FILE="$LOG_PATH/${timestamp}.log"

LOCAL_LOG_FILE="$LOCAL_FILE_SERVER_LOG_PATH/${timestamp}.log"

if [ ! -d "$LOG_PATH" ]; then
    # 目录不存在，创建它
    mkdir -p "$LOG_PATH"
fi
if [ ! -d "$LOCAL_FILE_SERVER_LOG_PATH" ]; then
    # 目录不存在，创建它
    mkdir -p "$LOCAL_FILE_SERVER_LOG_PATH"
fi
python -m api.table.init_tables
python -m api.table.redis_test
echo "日志文件路径: $LOG_FILE"
cd "$PROJECT_ROOT" || { echo "无法切换到项目目录: $PROJECT_ROOT"; exit 1; }
nohup python -m api.server.main_server -p $API_PORT > "$LOG_FILE" 2>&1 &
nohup python -m api.server.file_server > "$LOCAL_LOG_FILE" 2>&1 &
echo "检测脚本已在后台运行，输出日志位于: $LOG_FILE"