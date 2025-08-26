#!/bin/bash

# 睡眠监测系统一键部署脚本
# 使用方法: sudo bash sleep_monitor_init.sh <项目名称> <Redis端口> <PostgreSQL端口>
# 例如: sudo bash sleep_monitor_init.sh sleep_monitor 6379 5432

set -e

# 检查参数
if [ $# -ne 3 ]; then
    echo "使用方法: $0 <项目名称> <Redis端口> <PostgreSQL端口>"
    echo "例如: $0 sleep_monitor 6379 5432"
    exit 1
fi

PROJECT_NAME=$1
REDIS_PORT=$2
POSTGRES_PORT=$3

# 容器名称
REDIS_CONTAINER="${PROJECT_NAME}_redis"
POSTGRES_CONTAINER="${PROJECT_NAME}_postgres"

# 配置参数
POSTGRES_PASSWORD="123456"
POSTGRES_DB="sleep_monitor_db"
REDIS_PASSWORD=""  # 留空表示无密码
DATA_PATH="/mnt/data/$PROJECT_NAME"

echo "========================================"
echo "睡眠监测系统初始化"
echo "========================================"
echo "项目名称: $PROJECT_NAME"
echo "Redis端口: $REDIS_PORT"
echo "PostgreSQL端口: $POSTGRES_PORT"
echo "数据路径: $DATA_PATH"
echo "========================================"

# 检查端口是否被占用
check_port() {
    local port=$1
    local service=$2
    if netstat -tuln | grep -q ":$port "; then
        echo "错误: $service端口 $port 已被占用"
        exit 1
    fi
}

check_port $REDIS_PORT "Redis"
check_port $POSTGRES_PORT "PostgreSQL"

# 检查容器是否已存在
check_container() {
    local container=$1
    if docker ps -a --format "table {{.Names}}" | grep -q "^$container$"; then
        echo "警告: 容器 '$container' 已存在，将删除重建"
        docker rm -f $container
    fi
}

check_container $REDIS_CONTAINER
check_container $POSTGRES_CONTAINER

# 创建数据目录
echo "创建数据目录..."
sudo mkdir -p $DATA_PATH/redis
sudo mkdir -p $DATA_PATH/postgres
sudo mkdir -p $DATA_PATH/logs
sudo chown -R 999:999 $DATA_PATH/postgres  # PostgreSQL用户ID
sudo chown -R 1001:1001 $DATA_PATH/redis   # Redis用户ID

# 创建Redis配置文件
echo "创建Redis配置文件..."
cat > $DATA_PATH/redis.conf << 'EOF'
# Redis配置 - 针对睡眠监测系统优化
bind 0.0.0.0
port 6379
protected-mode no
tcp-backlog 511
timeout 0
tcp-keepalive 300

# 内存配置
maxmemory 1536mb
maxmemory-policy allkeys-lru
maxmemory-samples 5

# 持久化配置
save 900 1
save 300 10
save 60 10000
stop-writes-on-bgsave-error yes
rdbcompression yes
rdbchecksum yes
dbfilename dump.rdb

# AOF持久化
appendonly yes
appendfilename "appendonly.aof"
appendfsync everysec
no-appendfsync-on-rewrite no
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb
aof-load-truncated yes
aof-use-rdb-preamble yes

# 性能优化
stream-node-max-bytes 4kb
stream-node-max-entries 100
maxclients 1000

# 日志配置
loglevel notice
logfile ""
syslog-enabled no
databases 16

# 慢查询日志
slowlog-log-slower-than 10000
slowlog-max-len 128

# 延迟监控
latency-monitor-threshold 100
EOF

# 启动Redis容器

docker run -d \
    --name $REDIS_CONTAINER \
    --restart always \
    -p $REDIS_PORT:6379 \
    -v $DATA_PATH/redis:/data \
    -v $DATA_PATH/redis.conf:/usr/local/etc/redis/redis.conf:ro \
    --memory=2g \
    --cpus=1.0 \
    redis:7-alpine redis-server /usr/local/etc/redis/redis.conf

# 等待Redis启动
echo "等待Redis启动..."
max_attempts=30
attempt=0
while [ $attempt -lt $max_attempts ]; do
    if docker exec $REDIS_CONTAINER redis-cli ping > /dev/null 2>&1; then
        echo "Redis容器已就绪"
        break
    fi
    echo "等待中... ($((attempt + 1))/$max_attempts)"
    sleep 2
    attempt=$((attempt + 1))
done

if [ $attempt -eq $max_attempts ]; then
    echo "错误: Redis容器启动超时"
    exit 1
fi

# 启动PostgreSQL容器
echo "启动PostgreSQL容器..."
docker run -d \
    --name $POSTGRES_CONTAINER \
    --restart always \
    -e POSTGRES_PASSWORD=$POSTGRES_PASSWORD \
    -e POSTGRES_DB=$POSTGRES_DB \
    -v $DATA_PATH/postgres:/var/lib/postgresql/data \
    -p $POSTGRES_PORT:5432 \
    --memory=2g \
    --cpus=1.0 \
    postgres:15-alpine

# 等待PostgreSQL启动
echo "等待PostgreSQL启动..."
attempt=0
while [ $attempt -lt $max_attempts ]; do
    if docker exec $POSTGRES_CONTAINER pg_isready -U postgres > /dev/null 2>&1; then
        echo "PostgreSQL容器已就绪"
        break
    fi
    echo "等待中... ($((attempt + 1))/$max_attempts)"
    sleep 2
    attempt=$((attempt + 1))
done

if [ $attempt -eq $max_attempts ]; then
    echo "错误: PostgreSQL容器启动超时"
    exit 1
fi

# 创建应用配置文件
echo "创建应用配置文件..."
cat > $DATA_PATH/app_config.env << EOF
# 睡眠监测应用配置
REDIS_HOST=localhost
REDIS_PORT=$REDIS_PORT
REDIS_PASSWORD=$REDIS_PASSWORD
REDIS_DB=0

DATABASE_HOST=localhost
DATABASE_PORT=$POSTGRES_PORT
DATABASE_NAME=$POSTGRES_DB
DATABASE_USER=postgres
DATABASE_PASSWORD=$POSTGRES_PASSWORD

# 应用配置
LOG_LEVEL=INFO
MAX_DEVICES=1000
BATCH_SIZE=50
SLIDING_WINDOW_SIZE=20

# 连接URL
REDIS_URL=redis://localhost:$REDIS_PORT
DATABASE_URL=postgresql://postgres:$POSTGRES_PASSWORD@localhost:$POSTGRES_PORT/$POSTGRES_DB
EOF

# 创建Python requirements文件
echo "创建Python依赖文件..."
cat > $DATA_PATH/requirements.txt << 'EOF'
redis>=4.5.0
psycopg2-binary>=2.9.0
numpy>=1.21.0
torch>=1.9.0
scikit-learn>=1.0.0
sqlalchemy>=1.4.0
asyncpg>=0.25.0
python-dateutil>=2.8.0
pytz>=2021.1
pandas>=1.3.0
EOF

# 创建Docker网络
echo "创建Docker网络..."
docker network create ${PROJECT_NAME}_network 2>/dev/null || echo "网络已存在，跳过创建"

# 将容器添加到网络
docker network connect ${PROJECT_NAME}_network $REDIS_CONTAINER 2>/dev/null || echo "Redis已在网络中"
docker network connect ${PROJECT_NAME}_network $POSTGRES_CONTAINER 2>/dev/null || echo "PostgreSQL已在网络中"

# 创建管理脚本
echo "创建管理脚本..."

# 启动脚本
cat > $DATA_PATH/start.sh << EOF
#!/bin/bash
echo "启动睡眠监测系统..."
docker start $REDIS_CONTAINER
docker start $POSTGRES_CONTAINER
echo "系统已启动"
EOF

# 停止脚本
cat > $DATA_PATH/stop.sh << EOF
#!/bin/bash
echo "停止睡眠监测系统..."
docker stop $REDIS_CONTAINER 2>/dev/null || echo "Redis容器未运行"
docker stop $POSTGRES_CONTAINER 2>/dev/null || echo "PostgreSQL容器未运行"
echo "系统已停止"
EOF

# 监控脚本
cat > $DATA_PATH/monitor.sh << EOF
#!/bin/bash
echo "========================================"
echo "睡眠监测系统监控"
echo "========================================"

echo "容器状态:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep $PROJECT_NAME || echo "无运行的容器"

echo ""
echo "Redis状态:"
docker exec $REDIS_CONTAINER redis-cli ping 2>/dev/null || echo "Redis未运行"

echo ""
echo "PostgreSQL状态:"
docker exec $POSTGRES_CONTAINER pg_isready -U postgres 2>/dev/null || echo "PostgreSQL未运行"

echo ""
echo "Redis内存使用:"
docker exec $REDIS_CONTAINER redis-cli info memory | grep used_memory_human 2>/dev/null || echo "无法获取"

echo ""
echo "Stream信息:"
docker exec $REDIS_CONTAINER redis-cli xinfo stream device_data_stream 2>/dev/null || echo "Stream不存在"

echo ""
echo "数据库连接数:"
docker exec $POSTGRES_CONTAINER psql -U postgres -d $POSTGRES_DB -c "SELECT count(*) FROM pg_stat_activity;" 2>/dev/null || echo "无法获取"

echo ""
echo "表记录数:"
docker exec $POSTGRES_CONTAINER psql -U postgres -d $POSTGRES_DB -c "SELECT 'device_states' as table_name, count(*) FROM device_states UNION ALL SELECT 'sleep_data_state', count(*) FROM sleep_data_state UNION ALL SELECT 'real_time_vital_data', count(*) FROM real_time_vital_data;" 2>/dev/null || echo "无法获取"
EOF

# 清理脚本
cat > $DATA_PATH/cleanup.sh << EOF
#!/bin/bash
echo "清理睡眠监测系统..."
read -p "确定要删除所有容器和数据吗？(y/N): " confirm
if [[ \$confirm == [yY] || \$confirm == [yY][eE][sS] ]]; then
    docker stop $REDIS_CONTAINER $POSTGRES_CONTAINER 2>/dev/null
    docker rm $REDIS_CONTAINER $POSTGRES_CONTAINER 2>/dev/null
    docker network rm ${PROJECT_NAME}_network 2>/dev/null
    echo "容器已删除"
    read -p "是否删除数据目录 $DATA_PATH？(y/N): " confirm_data
    if [[ \$confirm_data == [yY] || \$confirm_data == [yY][eE][sS] ]]; then
        sudo rm -rf $DATA_PATH
        echo "数据已删除"
    fi
else
    echo "操作已取消"
fi
EOF

# 创建Python连接示例
cat > $DATA_PATH/redis_example.py << EOF
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Redis连接示例
"""
import redis
import os

# 从配置文件读取连接信息
REDIS_HOST = 'localhost'
REDIS_PORT = $REDIS_PORT
REDIS_DB = 0

# 创建Redis连接
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    decode_responses=True
)

# 测试连接
try:
    redis_client.ping()
    print("Redis连接成功")
    
    # 发布测试消息
    redis_client.xadd('device_data_stream', {
        'device_id': 'TEST_DEVICE',
        'data': 'test_message'
    })
    print("测试消息发布成功")
    
except Exception as e:
    print(f"Redis连接失败: {e}")
EOF

# 设置脚本权限
chmod +x $DATA_PATH/*.sh
chmod +x $DATA_PATH/*.py

# 显示结果
echo "========================================"
echo "睡眠监测系统初始化完成！"
echo "========================================"
echo "项目名称: $PROJECT_NAME"
echo ""
echo "容器信息:"
echo "  Redis容器: $REDIS_CONTAINER (端口: $REDIS_PORT)"
echo "  PostgreSQL容器: $POSTGRES_CONTAINER (端口: $POSTGRES_PORT)"
echo ""
echo "连接信息:"
echo "  Redis: redis://localhost:$REDIS_PORT"
echo "  PostgreSQL: postgresql://postgres:$POSTGRES_PASSWORD@localhost:$POSTGRES_PORT/$POSTGRES_DB"
echo ""
echo "数据目录: $DATA_PATH"
echo "配置文件: $DATA_PATH/app_config.env"
echo "依赖文件: $DATA_PATH/requirements.txt"
echo "连接示例: $DATA_PATH/redis_example.py"
echo ""
echo "管理命令:"
echo "  启动服务: bash $DATA_PATH/start.sh"
echo "  停止服务: bash $DATA_PATH/stop.sh"
echo "  监控状态: bash $DATA_PATH/monitor.sh"
echo "  完全清理: bash $DATA_PATH/cleanup.sh"
echo ""
echo "容器管理:"
echo "  查看Redis日志: docker logs $REDIS_CONTAINER"
echo "  查看PostgreSQL日志: docker logs $POSTGRES_CONTAINER"
echo "  Redis CLI: docker exec -it $REDIS_CONTAINER redis-cli"
echo "  PostgreSQL CLI: docker exec -it $POSTGRES_CONTAINER psql -U postgres -d $POSTGRES_DB"
echo ""
echo "在你的Python代码中使用:"
echo "  REDIS_URL = 'redis://localhost:$REDIS_PORT'"
echo "  DATABASE_URL = 'postgresql://postgres:$POSTGRES_PASSWORD@localhost:$POSTGRES_PORT/$POSTGRES_DB'"
echo "========================================"
