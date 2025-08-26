#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/08/21 10:37
@Author  : weiyutao
@File    : real_time_sleep_data_state.py
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, DateTime, Float, BigInteger, SmallInteger
from sqlalchemy.sql import func
from sqlalchemy.orm import sessionmaker
import asyncio

from api.table.base import Base

class SleepDataState(Base):
    """
    睡眠数据状态表 - 异步版本
    """
    __tablename__ = 'real_time_sleep_data_state'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True, comment='主键ID')
    device_id = Column(String(64), nullable=False, comment='设备编号')
    timestamp = Column(Float, nullable=False, comment='时间戳')
    breath_bpm = Column(Float, nullable=True, comment='呼吸率')
    breath_line = Column(Float, nullable=True, comment='呼吸线')
    heart_bpm = Column(Float, nullable=True, comment='心率')
    heart_line = Column(Float, nullable=True, comment='心线')
    reconstruction_error = Column(Float, nullable=True, comment='重建误差')
    state = Column(String(32), nullable=False, comment='状态')
    creator = Column(String(64), nullable=True, comment='创建人')
    create_time = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, comment='创建时间')
    updater = Column(String(64), nullable=True, comment='更新人')
    update_time = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False, comment='更新时间')
    tenant_id = Column(BigInteger, default=0, nullable=False, comment='租户编号')
    deleted = Column(SmallInteger, default=0, nullable=False, comment='是否删除')