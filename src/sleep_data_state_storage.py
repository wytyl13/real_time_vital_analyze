#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/06/27 11:51
@Author  : weiyutao
@File    : sleep_data_storage.py
"""


import time
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Callable
from collections import deque
from typing import (
    Dict,
    Any
)
from pathlib import Path

from .tables.sleep_data_state import SleepDataState
from .provider.sql_provider import SqlProvider
from .state_smooth import EnhancedStateSmoother

SUB_ROOT_DIRECTORY = Path(__file__).parent
SQL_CONFIG_PATH = str(SUB_ROOT_DIRECTORY / "config" / "yaml" / "sql_config.yaml")

smoother = EnhancedStateSmoother(
    window_size=12,              # 增大窗口
    min_state_duration=60.0,     # 基础60秒
    confidence_threshold=0.75,   # 提高置信度
    hysteresis_margin=20.0       # 20秒滞后
)

sql_provider = SqlProvider(
    model=SleepDataState, 
    sql_config_path=SQL_CONFIG_PATH,
)


@dataclass
class RealTimeDataPoint:
    """轻量级实时数据点 - 只包含业务字段"""
    device_id: str
    timestamp: float
    breath_bpm: float
    breath_line: float
    heart_bpm: float
    heart_line: float
    reconstruction_error: float
    state: str

    def to_sleep_data_state(self, 
                          creator: str = "system", 
                          tenant_id: int = 0) -> SleepDataState:
        """转换为数据库存储对象"""
        return SleepDataState(
            device_id=self.device_id,
            timestamp=self.timestamp,
            breath_bpm=self.breath_bpm,
            breath_line=self.breath_line,
            heart_bpm=self.heart_bpm,
            heart_line=self.heart_line,
            reconstruction_error=self.reconstruction_error,
            state=self.state,
            creator=creator,
            tenant_id=tenant_id
        )


    def to_db_dict(self, 
                   creator: str = "system", 
                   tenant_id: int = 0) -> Dict[str, Any]:
        """转换为数据库存储格式的字典"""
        return {
            'device_id': self.device_id,
            'timestamp': self.timestamp,
            'breath_bpm': self.breath_bpm,
            'breath_line': self.breath_line,
            'heart_bpm': self.heart_bpm,
            'heart_line': self.heart_line,
            'reconstruction_error': self.reconstruction_error,
            'state': self.state,
            'creator': creator,
            'tenant_id': tenant_id
        }


class SleepDataStateStorage:
    """极简睡眠数据存储管理器 - 修复版"""
    
    def __init__(self, 
                 single_insert_db: Optional[Callable] = None,
                 batch_insert_db: Optional[Callable] = None,
                 buffer_duration: float = 60.0,      # 缓冲区时长(秒)
                 min_interval: float = 10.0,         # 最小存储间隔(秒)
                 max_interval: float = 60.0):        # 最大存储间隔(秒)
        
        self.single_insert_db = single_insert_db
        self.batch_insert_db = batch_insert_db
        self.buffer_duration = buffer_duration
        self.min_interval = min_interval
        self.max_interval = max_interval
        
        # 数据缓冲区（最近60秒）
        self.data_buffer: deque[RealTimeDataPoint] = deque()
        
        # 存储状态跟踪
        self.last_stored_data: Optional[RealTimeDataPoint] = None
        self.last_storage_time: Optional[float] = None
        
        # 异常状态定义 - 添加体动
        self.anomaly_states = {"呼吸暂停", "呼吸急促", "体动"}
        self.anomaly_detected = False
        self.anomaly_start_time: Optional[float] = None
        self.context_stored = False  # 标记是否已存储前60秒上下文
        
        print("FixedSleepDataStorage 初始化完成")
        print(f"缓冲区时长: {buffer_duration}秒")
        print(f"存储间隔: {min_interval}-{max_interval}秒")
        print(f"异常状态: {self.anomaly_states}")
        print("🔄 数据库自动处理重复数据")
    
    def add_data_point(self, 
                      device_id: str,
                      timestamp: float,
                      breath_bpm: float,
                      breath_line: float,
                      heart_bpm: float,
                      heart_line: float,
                      reconstruction_error: float,
                      state: str):
        """添加新的数据点"""
        
        # 创建数据点
        data_point = RealTimeDataPoint(
            device_id=device_id,
            timestamp=timestamp,
            breath_bpm=breath_bpm,
            breath_line=breath_line,
            heart_bpm=heart_bpm,
            heart_line=heart_line,
            reconstruction_error=reconstruction_error,
            state=state
        )
        
        smoothed_state = smoother.smooth_state(data_point.state, data_point.timestamp)
        data_point.state = smoothed_state
        # 1. 添加到缓冲区
        self.data_buffer.append(data_point)
        self._clean_buffer(timestamp)
        
        # 2. 检查是否需要存储
        should_store, reason = self._should_store(data_point)
        
        if should_store:
            if reason == "首次异常":
                # 首次异常：存储前60秒数据 + 当前异常数据
                self._store_anomaly_context(data_point)

                # 是否存储异常开始前60秒数据？仅存储当前数据
                # self._store_single_data(data_point, reason)
                self.context_stored = True
                
            elif reason == "持续异常":
                # 持续异常：只存储当前数据
                self._store_single_data(data_point, reason)
            elif reason == "异常结束":
                # 异常结束：存储当前数据并重置状态
                self._store_single_data(data_point, reason)
                self.context_stored = False
            else:
                # 正常情况：存储单个数据点
                self._store_single_data(data_point, reason)
            
            # 更新存储状态
            self.last_stored_data = data_point
            self.last_storage_time = timestamp
        
        # 3. 异常状态跟踪
        self._track_anomaly_state(data_point)
    
    def _clean_buffer(self, current_time: float):
        """清理超过60秒的缓冲区数据"""
        cutoff_time = current_time - self.buffer_duration
        
        while self.data_buffer and self.data_buffer[0].timestamp < cutoff_time:
            self.data_buffer.popleft()
    
    def _should_store(self, data_point: RealTimeDataPoint) -> tuple[bool, str]:
        """判断是否应该存储数据"""
        current_time = data_point.timestamp
        current_state = data_point.state
        is_current_anomaly = current_state in self.anomaly_states
        
        # 1. 第一次数据，必须存储
        if self.last_stored_data is None:
            return True, "首次数据"
        
        # 2. 异常状态检测
        if is_current_anomaly:
            if not self.anomaly_detected:
                # 首次检测到异常
                return True, "首次异常"
            else:
                # 持续异常状态，也要存储
                return True, "持续异常"
        
        # 3. 异常结束检测
        if not is_current_anomaly and self.anomaly_detected:
            return True, "异常结束"

        # 4. 状态变化，必须存储
        if current_state != self.last_stored_data.state:
            return True, "状态变化"
        
        # 5. 时间间隔检查
        time_since_last = current_time - self.last_storage_time
        
        # 达到最大间隔，必须存储
        if time_since_last >= self.max_interval:
            return True, "最大间隔"
        
        return False, "无需存储"
    
    def _track_anomaly_state(self, data_point: RealTimeDataPoint):
        """跟踪异常状态"""
        current_state = data_point.state
        
        if current_state in self.anomaly_states:
            if not self.anomaly_detected:
                # 首次检测到异常
                self.anomaly_detected = True
                self.anomaly_start_time = data_point.timestamp
                print(f"🚨 检测到异常状态: {current_state} at {time.strftime('%H:%M:%S', time.localtime(data_point.timestamp))}")
        else:
            if self.anomaly_detected:
                # 异常状态结束
                duration = data_point.timestamp - self.anomaly_start_time
                print(f"✅ 异常状态结束，持续时间: {duration:.1f}秒")
                
            self.anomaly_detected = False
            self.anomaly_start_time = None
    
    def _store_single_data(self, data_point: RealTimeDataPoint, reason: str):
        """存储单个数据点"""
        if self.single_insert_db:
            sleep_data_state = data_point.to_db_dict()
            self.single_insert_db(sleep_data_state)
        else:
            print(f"💾 存储数据 [{reason}]: {data_point.device_id} - {data_point.state} - "
                  f"{time.strftime('%H:%M:%S', time.localtime(data_point.timestamp))}")
    
    def _store_anomaly_context(self, anomaly_data_point: RealTimeDataPoint):
        """存储异常前60秒的数据，将存储数据的状态改为当前异常状态（不修改缓存）"""
        current_time = anomaly_data_point.timestamp
        context_start = current_time - 30.0
        current_anomaly_state = anomaly_data_point.state
        
        # 获取前60秒的数据，仅在存储时修改状态（缓存数据不变）
        context_data = []
        for data in self.data_buffer:
            if context_start <= data.timestamp <= current_time:
                # 转为存储格式
                sleep_data_state_ = data.to_db_dict()
                # 关键：只修改要存储的数据状态，缓存中的data对象保持不变
                sleep_data_state_["state"] = current_anomaly_state
                context_data.append(sleep_data_state_)
        
        # 按时间戳排序，确保插入顺序正确
        context_data.sort(key=lambda x: x["timestamp"])
        # print(context_data)
        # 批量存储（数据库中的状态已修改，但缓存保持原始状态）
        if self.batch_insert_db and context_data:
            self.batch_insert_db(context_data)
            # print(f"🚨 存储异常上下文数据:")
            # print(f"   异常类型: {current_anomaly_state}")
            # print(f"   数据点数量: {len(context_data)} 个")
            # print(f"   ⚠️  数据库存储状态已统一改为: {current_anomaly_state} (缓存保持原始状态)")
            # print(f"   时间范围: {time.strftime('%H:%M:%S', time.localtime(context_data[0]['timestamp']))} - "
            #       f"{time.strftime('%H:%M:%S', time.localtime(context_data[-1]['timestamp']))}")
        else:
            # print(f"🚨 存储异常上下文数据:")
            # print(f"   异常类型: {current_anomaly_state}")
            # print(f"   数据点数量: {len(context_data)} 个")
            # print(f"   ⚠️  数据库存储状态已统一改为: {current_anomaly_state} (缓存保持原始状态)")
            if context_data:
                print(f"   时间范围: {time.strftime('%H:%M:%S', time.localtime(context_data[0]['timestamp']))} - "
                      f"{time.strftime('%H:%M:%S', time.localtime(context_data[-1]['timestamp']))}")
    
    def get_buffer_stats(self) -> dict:
        """获取缓冲区统计信息"""
        if not self.data_buffer:
            return {
                "size": 0, 
                "duration": 0, 
                "states": {},
                "last_storage": self.last_storage_time,
                "anomaly_active": self.anomaly_detected
            }
        
        # 统计状态分布
        state_counts = {}
        for data in self.data_buffer:
            state_counts[data.state] = state_counts.get(data.state, 0) + 1
        
        duration = self.data_buffer[-1].timestamp - self.data_buffer[0].timestamp
        
        return {
            "size": len(self.data_buffer),
            "duration": duration,
            "states": state_counts,
            "last_storage": self.last_storage_time,
            "anomaly_active": self.anomaly_detected
        }
    
    def force_storage(self, reason: str = "手动触发"):
        """强制存储当前最新数据"""
        if self.data_buffer:
            latest_data = self.data_buffer[-1]
            self._store_single_data(latest_data, reason)
            self.last_stored_data = latest_data
            self.last_storage_time = latest_data.timestamp




# 使用示例
if __name__ == "__main__":
    # 创建存储管理器
    storage = SleepDataStateStorage(
        single_insert_db=sql_provider.add_record,
        batch_insert_db=sql_provider.bulk_insert_with_update,
        buffer_duration=60.0,
        min_interval=10.0,
        max_interval=30.0
    )
    
    # 模拟数据流
    device_id = "DEV001"
    base_time = time.time()
    
    test_scenarios = [
        (0, "在床正常", 70, 16),
        (1, "在床正常", 71, 17),
        (2, "清醒", 75, 18),
        (3, "清醒", 74, 17),
        (4, "浅睡眠", 65, 14),
        (5, "浅睡眠", 66, 15),
        (6, "呼吸暂停", 70, 8),        # 异常：会存储前60秒数据
        (7, "呼吸暂停", 72, 6),
        (8, "深睡眠", 60, 12),
        (9, "深睡眠", 58, 13),
        (10, "深睡眠", 58, 13),
        (11, "呼吸暂停", 58, 13),
        (12, "体动", 58, 13),
    ]
    
    print("开始模拟数据流...")
    print("=" * 60)
    
    for i, (offset, state, hr, br) in enumerate(test_scenarios):
        timestamp = base_time + offset
        
        # 模拟波形数据
        breath_line = 1.0
        heart_line = 1.0
        error = 0.2222
        
        print(f"\n[{i+1}] 输入数据: {state} | HR:{hr} | BR:{br} | "
              f"{time.strftime('%H:%M:%S', time.localtime(timestamp))}")
        
        # 添加数据点
        storage.add_data_point(
            device_id=device_id,
            timestamp=timestamp,
            breath_bpm=br,
            breath_line=breath_line,
            heart_bpm=hr,
            heart_line=heart_line,
            reconstruction_error=error,
            state=state
        )
        
        # 显示缓冲区状态
        stats = storage.get_buffer_stats()
        print(f"   缓冲区: {stats['size']}个数据点, 状态分布: {stats['states']}")
        print("-" * 40)
    
    print("\n" + "=" * 60)
    print("数据流模拟完成")
    print("查询时记得使用: ORDER BY timestamp ASC")