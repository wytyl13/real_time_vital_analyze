#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/07/02 16:00
@Author  : weiyutao
@File    : device_data.py
"""

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, \
    BigInteger, ForeignKey, BINARY, Float, LargeBinary
from sqlalchemy.dialects.mysql import TINYINT
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class DeviceData(Base):
    """
    设备信息表
    对应 SX_DEVICE_DATA 表
    """
    __tablename__ = 'sx_device_data'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True, comment='主键id')
    device_code = Column(String(64), nullable=False, unique=True, comment='设备编号')
    scene = Column(String(128), nullable=True, comment='场景')
    wifi_name = Column(String(128), nullable=True, comment='WiFi名称')
    wifi_password = Column(String(256), nullable=True, comment='WiFi密码')
    username = Column(String(64), nullable=False, comment='用户名')
    user_id = Column(BigInteger, nullable=True, comment='用户ID')
    status = Column(String(16), default='active', nullable=False, comment='设备状态')
    creator = Column(String(64), nullable=True, comment='创建者')
    create_time = Column(DateTime, default=datetime.now, nullable=False, comment='创建时间')
    updater = Column(String(64), nullable=True, comment='更新者')
    update_time = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False, comment='更新时间')
    # 🔧 修改deleted字段为Boolean类型
    tenant_id = Column(BigInteger, default=0, nullable=False, comment='租户编号')
    deleted = Column(TINYINT, default=0, nullable=False, comment='是否删除')
    


if __name__ == '__main__':
    from pathlib import Path
    
    from real_time_vital_analyze.tables.device_data import DeviceData
    from real_time_vital_analyze.provider.sql_provider import SqlProvider
    ROOT_DIRECTORY = Path(__file__).parent.parent
    SQL_CONFIG_PATH = str(ROOT_DIRECTORY / "config" / "yaml" / "sql_config.yaml")
    sql_provider = SqlProvider(model=DeviceData, sql_config_path=SQL_CONFIG_PATH)
            
    existing_devices = sql_provider.get_record_by_condition(
        condition={"device_code": "13271C9D10004071111715B507"}
    )
    print(existing_devices)
