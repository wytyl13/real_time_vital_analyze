#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
设备队列存储解决方案
支持内存和Redis两种存储方式的设备数据管理
"""

import json
import time
import redis
import threading
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from collections import deque
import logging

from lockfree_queue import LockFreeQueue, AtomicDict











# 使用示例
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    def test_storage(storage: DeviceQueueStorageInterface, storage_name: str):
        """测试存储功能"""
        print(f"\n🧪 测试 {storage_name} 存储")
        print("=" * 50)
        
        # 模拟设备数据
        devices = ['device_001', 'device_002', 'device_003']
        
        # 添加数据
        for device_id in devices:
            for i in range(5):
                data = {
                    'device_id': device_id,
                    'temperature': 20 + i,
                    'humidity': 50 + i,
                    'timestamp': time.time()
                }
                success = storage.put_device_data(device_id, data)
                print(f"   设备 {device_id} 添加数据 {i}: {'成功' if success else '失败'}")
        
        # 查看统计
        print(f"\n📊 存储统计:")
        stats = storage.get_storage_stats()
        print(f"   {json.dumps(stats, indent=2, ensure_ascii=False)}")
        
        # 查看所有设备
        all_devices = storage.get_all_devices()
        print(f"\n📱 所有设备: {all_devices}")
        
        # 查看每个设备的数据
        for device_id in all_devices:
            queue_size = storage.get_device_queue_size(device_id)
            all_data = storage.get_all_device_data(device_id)
            print(f"\n📋 设备 {device_id}:")
            print(f"   队列大小: {queue_size}")
            print(f"   所有数据: {len(all_data)} 条")
            
            # 取出一条数据
            data = storage.get_device_data(device_id)
            if data:
                print(f"   取出数据: {data}")
    
    # 测试内存存储
    memory_storage = DeviceQueueStorageFactory.create_storage('memory')
    test_storage(memory_storage, "内存")
    
    # 测试Redis存储（需要Redis服务）
    try:
        redis_storage = DeviceQueueStorageFactory.create_storage(
            'redis',
            redis_config={
                'host': 'localhost',
                'port': 6379,
                'database': 0,
                'key_prefix': 'test_device_queue'
            }
        )
        test_storage(redis_storage, "Redis")
    except Exception as e:
        print(f"\n❌ Redis存储测试失败: {e}")
    
    # 测试混合存储
    try:
        hybrid_storage = DeviceQueueStorageFactory.create_storage(
            'hybrid',
            redis_config={
                'host': 'localhost',
                'port': 6379,
                'database': 0,
                'key_prefix': 'test_hybrid_queue'
            }
        )
        test_storage(hybrid_storage, "混合")
    except Exception as e:
        print(f"\n❌ 混合存储测试失败: {e}")