#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/08/21 10:38
@Author  : weiyutao
@File    : redis_test.py
"""

from pathlib import Path


from agent.config.sql_config import SqlConfig

redis_config = SqlConfig.from_file(Path("/work/ai/real_time_vital_analyze/config/yaml/redis.yaml"))


import redis
import time
import json

def test_redis_connection():
    """测试Redis连接"""
    try:
        # 创建Redis连接
        redis_client = redis.Redis(
            host=redis_config.host,
            port=redis_config.port,
            db=0,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5
        )
        
        # 1. 基本连接测试
        print("1. 测试基本连接...")
        response = redis_client.ping()
        print(f"   Ping响应: {response}")
        
        # 2. 基本读写测试
        print("\n2. 测试基本读写...")
        redis_client.set('test_key', 'hello_redis')
        value = redis_client.get('test_key')
        print(f"   写入: test_key = 'hello_redis'")
        print(f"   读取: test_key = {value}")
        
        # 3. 测试Stream操作（消息队列）
        print("\n3. 测试Stream操作...")
        stream_name = 'device_data_stream'
        
        # 发布消息
        message_id = redis_client.xadd(stream_name, {
            'device_id': 'TEST_DEVICE',
            'timestamp': time.time(),
            'data': 'test_message'
        })
        print(f"   消息发布成功: {message_id}")
        
        # 读取消息
        messages = redis_client.xread({stream_name: '0'}, count=1)
        print(f"   读取消息: {len(messages[0][1]) if messages else 0}条")
        
        # 4. 测试Redis信息
        print("\n4. Redis服务器信息...")
        info = redis_client.info()
        print(f"   Redis版本: {info.get('redis_version')}")
        print(f"   已用内存: {info.get('used_memory_human')}")
        print(f"   连接的客户端: {info.get('connected_clients')}")
        
        # 5. 测试哈希操作（设备状态存储）
        print("\n5. 测试设备状态存储...")
        device_state = {
            'device_id': 'TEST_DEVICE',
            'status': 'active',
            'last_update': str(time.time())
        }
        redis_client.hset('device_state:TEST_DEVICE', mapping=device_state)
        stored_state = redis_client.hgetall('device_state:TEST_DEVICE')
        print(f"   设备状态已存储: {stored_state}")
        
        # 6. 清理测试数据
        print("\n6. 清理测试数据...")
        redis_client.delete('test_key', 'device_state:TEST_DEVICE')
        redis_client.delete(stream_name)
        print("   测试数据已清理")
        
        print("\n✅ Redis连接测试全部通过!")
        return True
        
    except redis.ConnectionError as e:
        print(f"❌ Redis连接失败: {e}")
        print("   请检查Redis容器是否正在运行")
        return False
        
    except Exception as e:
        print(f"❌ Redis测试失败: {e}")
        return False

def test_redis_performance():
    """测试Redis性能"""
    try:
        redis_client = redis.Redis(host=redis_config.host, port=redis_config.port, db=0)
        
        print("\n📊 Redis性能测试...")
        
        # 批量写入测试
        start_time = time.time()
        for i in range(1000):
            redis_client.set(f'perf_test_{i}', f'value_{i}')
        write_time = time.time() - start_time
        print(f"   批量写入1000条: {write_time:.3f}秒")
        
        # 批量读取测试
        start_time = time.time()
        for i in range(1000):
            redis_client.get(f'perf_test_{i}')
        read_time = time.time() - start_time
        print(f"   批量读取1000条: {read_time:.3f}秒")
        
        # 清理测试数据
        for i in range(1000):
            redis_client.delete(f'perf_test_{i}')
        
        print(f"   写入QPS: {1000/write_time:.0f}")
        print(f"   读取QPS: {1000/read_time:.0f}")
        
    except Exception as e:
        print(f"❌ 性能测试失败: {e}")

if __name__ == '__main__':
    print("开始Redis连接测试...")
    print("=" * 50)
    
    success = test_redis_connection()
    
    if success:
        test_redis_performance()
    
    print("=" * 50)
    print("测试完成")