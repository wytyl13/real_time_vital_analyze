#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/08/06 14:35
@Author  : weiyutao
@File    : sleep_statistics_server.py
"""


from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import logging
from fastapi.encoders import jsonable_encoder

from ..tables.device_data import DeviceData
from ..tables.sleep_statistics_model import SleepStatistics
from ..provider.sql_provider import SqlProvider


class ListSleepStatistics(BaseModel):
    username: Optional[str] = None
    device_sn: Optional[str] = None

class SleepStatisticsServer:
    """睡眠统计服务类"""
    
    def __init__(self, sql_config_path: str):
        self.sql_config_path = sql_config_path
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def register_routes(self, app: FastAPI):
        """注册睡眠统计相关的路由"""
        app.post("/api/sleep_statistics")(self.post_sleep_statistics)
        app.get("/api/sleep_statistics")(self.get_sleep_statistics)
    
    
    
    async def get_sleep_statistics(
        self, 
        username: Optional[str] = None,
        device_sn: Optional[str] = None
    ):
        """
        GET请求 - 支持在url中传递username, device_sn等参数过滤睡眠数据统计信息
        Examples:
        - GET /api/sleep_statistics -> 不合法的请求，username和device_sn不能全为空
        - GET /api/sleep_statistics?username=john -> 获取john用户的设备信息
        - GET /api/sleep_statistics?device_sn=SN123456 -> 根据设备序列号查询
        """
        if not username and not device_sn:
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "用户名/设备编号不能同时为空", "timestamp": datetime.now().isoformat()}
            )
        
        try:
            condition = {}
            if username:
                condition["username"] = username
            if device_sn:
                condition["device_sn"] = device_sn
            
            if "username" in condition and "device_sn" not in condition:
                sql_provider_device = SqlProvider(model=DeviceData, sql_config_path=self.sql_config_path)
                device_result = sql_provider_device.get_record_by_condition(
                    condition=condition,
                    fields=["device_code"]
                )
            
                if not device_result:
                    return JSONResponse(
                        status_code=200,
                        content={"success": True, "data": [], "timestamp": datetime.now().isoformat()}
                    )
            
                sql_provider_sleep = SqlProvider(model=SleepStatistics, sql_config_path=self.sql_config_path)
                result = sql_provider_sleep.get_record_by_condition(
                    condition={"device_sn": device_result[0]["device_code"]},
                    fields=["sleep_start_time", "sleep_end_time", "health_report"]
                )
            
                json_compatible_result = jsonable_encoder(result)
                
                return JSONResponse(
                    status_code=200,
                    content={"success": True, "data": json_compatible_result, "timestamp": datetime.now().isoformat()}
                )
            else:
                try:
                    sql_provider = SqlProvider(model=SleepStatistics, sql_config_path=self.sql_config_path)
                    result = sql_provider.get_record_by_condition(
                        condition=condition,
                        fields=["sleep_start_time", "sleep_end_time", "health_report"]
                    )
                    
                    json_compatible_result = jsonable_encoder(result)
                    
                    return JSONResponse(
                        status_code=200,
                        content={"success": True, "data": json_compatible_result, "timestamp": datetime.now().isoformat()}
                    )
                    
                except Exception as e:
                    return JSONResponse(
                        status_code=500,
                        content={"success": False, "message": f"获取睡眠数据失败: {str(e)}", "timestamp": datetime.now().isoformat()}
                    )
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"success": False, "message": f"获取睡眠数据失败: {str(e)}", "timestamp": datetime.now().isoformat()}
            )
    
    
    async def post_sleep_statistics(self, list_params: ListSleepStatistics):
        """
        POST请求 - 支持在JSON请求体中传递username, device_sn等参数过滤睡眠数据统计信息
        Examples:
        - POST /api/sleep_statistics {}-> 不合法的请求，username和device_sn不能同时为空
        - POST /api/sleep_statistics {"username": "JOHN"} -> 获取john用户的设备信息
        - POST /api/sleep_statistics {"device_sn": "device_sn12345"} -> 根据设备序列号查询
        """
        # 无效代码-----------------------------------------------------------------------------------------
        try:
            username = list_params.username
            device_sn = list_params.device_sn
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": f"传参错误！{str(e)}", "data": None, "timestamp": datetime.now().isoformat()}
            )
        # 无效代码-----------------------------------------------------------------------------------------

        try:
            result = self.get_sleep_statistics(username=username, device_sn=device_sn)
            return result
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"success": False, "message": f"获取睡眠数据失败: {str(e)}", "timestamp": datetime.now().isoformat()}
            )
    
    