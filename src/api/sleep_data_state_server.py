#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/08/06 14:26
@Author  : weiyutao
@File    : device_data_server.py

200：执行成功，正确返回
400：请求参数错误
404：要删除或者更新的编号不存在
500：执行报错，错误返回
"""


from fastapi import FastAPI, HTTPException, Request, Query, Body
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import logging

# 导入你的数据模型
from ..tables.sleep_data_state import SleepDataState
from ..provider.sql_provider import SqlProvider


class SleepDataStateSchema(BaseModel):
    device_id: str
    timestamp: str
    wifi_password: str
    working_distance: Optional[float] = 1.5
    scene: Optional[str] = "睡眠监测"
    username: Optional[str] = ""
    user_id: Optional[int] = None


class ListDeviceData(BaseModel):
    username: Optional[str] = None
    device_sn: Optional[str] = None


class DeleteDeviceRequest(BaseModel):
    device_sn: Optional[str] = None



class DeviceDataServer:
    """设备管理服务类"""
    
    def __init__(self, sql_config_path: str):
        self.sql_config_path = sql_config_path
        self.logger = logging.getLogger(self.__class__.__name__)
    
    
    def register_routes(self, app: FastAPI):
        """注册设备相关的路由"""
        app.post("/api/device_info/save")(self.save_device)
        app.get("/api/device_info")(self.get_device_data_api)
        app.post("/api/device_info")(self.post_device_data_api)
        app.delete("/api/device_info")(self.delete_device_api)


    async def get_device_data_api(
        self,
        username: Optional[str] = Query(None, description="查询指定用户名的设备信息"),
        device_sn: Optional[str] = Query(None, description="查询指定设备序列号的设备信息")
    ):
        """
        GET请求 - 支持不传递任何参数获取所有设备信息，支持在url中传递username, device_sn等参数过滤设备信息
        Examples:
        - GET /api/device_info -> 获取所有设备信息
        - GET /api/device_info?username=john -> 获取john用户的设备信息
        - GET /api/device_info?device_sn=SN123456 -> 根据设备序列号查询
        """
        condition = {}
        try:
            if username is not None:
                condition["username"] = username
            if device_sn is not None:
                condition["device_code"] = device_sn
            condition["deleted"] = False
            sql_provider = SqlProvider(model=DeviceData, sql_config_path=self.sql_config_path)
            devices = sql_provider.get_record_by_condition(condition=condition)
            json_compatible_result = jsonable_encoder(devices)
            return JSONResponse(
                status_code=200,
                content={"success": True, "message": "成功！", "data": json_compatible_result, "timestamp": datetime.now().isoformat()}
            )
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"success": False, "message": f"数据库操作失败！{str(e)}", "data": None, "timestamp": datetime.now().isoformat()}
            )
            
            
    async def post_device_data_api(
        self,
        list_device_data: ListDeviceData = None
    ):
        """
        POST请求 - 支持不传递任何参数获取所有设备信息，支持在JSON请求体中传递username, device_sn等参数过滤设备信息
        Examples:
        - POST /api/device_info  {} ->  获取所有设备信息
        - POST /api/device_info  {"username": "JOHN"} -> 获取john用户的设备信息
        - POST /api/device_info  {"device_sn": "123456"} -> 根据设备序列号查询
        """
        
        # 无效代码-----------------------------------------------------------------------------------------
        try:
            username = list_device_data.username
            device_sn = list_device_data.device_sn
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": f"传参错误！{str(e)}", "data": None, "timestamp": datetime.now().isoformat()}
            )
        # 无效代码-----------------------------------------------------------------------------------------
        
        try:
            result = await self.get_device_data_api(username=username, device_sn=device_sn)
            return result
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"success": False, "message": f"数据库操作失败！{str(e)}", "data": None, "timestamp": datetime.now().isoformat()}
            )


    async def save_device(self, device_info: DeviceDataSchema):
        """
        POST请求 - 保存设备信息
        Examples:
        - POST /api/device_info/save  DeviceInfo ->  新增一条设备信息（device_code, wifi_name, wifi_password为必填项）
        """
        try:
            self.logger.info(f"收到设备保存请求: {device_info.device_code}")
            # 验证必要字段
            if not device_info.device_code:
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "message": "请提供设备编号", "data": None, "timestamp": datetime.now().isoformat()}
                )
            if not device_info.wifi_name:
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "message": "请提供wifi名称", "data": None, "timestamp": datetime.now().isoformat()}
                )
            if not device_info.wifi_password:
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "message": "请提供wifi密码", "data": None, "timestamp": datetime.now().isoformat()}
                )
            
            sql_provider = SqlProvider(model=DeviceData, sql_config_path=self.sql_config_path)
            existing_devices = sql_provider.get_record_by_condition(
                condition={"device_code": device_info.device_code}
            )
            if existing_devices:
                self.logger.info(f"找到 {len(existing_devices)} 个现有设备，进行删除操作")
                for device in existing_devices:
                    sql_provider.delete_record(record_id=device["id"], hard_delete=True)
            
            # 准备新设备数据
            device_data = {
                "device_code": device_info.device_code,
                "scene": device_info.scene,
                "wifi_name": device_info.wifi_name,
                "wifi_password": device_info.wifi_password,
                "username": device_info.username or "unknown",
                "user_id": device_info.user_id or 1,
                "status": "active",
                "creator": device_info.username or "system",
                "updater": device_info.username or "system",
                "deleted": False,
                "tenant_id": 0,
                "create_time": datetime.now(),
                "update_time": datetime.now()
            }
            
            result = sql_provider.add_record(device_data)
            
            if result:
                json_compatible_result = jsonable_encoder(result)
                return JSONResponse(
                    status_code=200,
                    content={"success": True, "message": "成功！", "data": json_compatible_result, "timestamp": datetime.now().isoformat()}
                )
            else:
                return JSONResponse(
                    status_code=500,
                    content={"success": False, "message": f"数据库操作失败！{str(e)}", "data": None, "timestamp": datetime.now().isoformat()}
                )
                
        except HTTPException:
            return JSONResponse(
                status_code=500,
                content={"success": False, "message": f"数据库操作失败！{str(e)}", "data": None, "timestamp": datetime.now().isoformat()}
            )
        except Exception as e:
            self.logger.error(f"保存设备信息异常: {str(e)}")
            import traceback
            traceback.print_exc()
            return JSONResponse(
                status_code=500,
                content={"success": False, "message": f"保存设备信息异常！{str(e)}, \n {traceback.print_exc()}", "data": None, "timestamp": datetime.now().isoformat()}
            )


    async def delete_device_api(
        self,
        device_sn: Optional[str] = Query(None, description="要删除的设备序列号"),
        request_body: Optional[DeleteDeviceRequest] = Body(None)
    ):
        """
        软删除，如果用户添加已经软删除的唯一键，则直接先删除软删除的内容，然后再添加新的即可
        DELETE请求 - 支持两种参数传递方式
        
        方式1 - 查询参数:
        DELETE /api/device_info?device_sn=13271C9D10004071111715B507
        
        方式2 - 请求体:
        DELETE /api/device_info
        Content-Type: application/json
        {
            "device_sn": "13271C9D10004071111715B507"
        }
        """
        try:
            # 获取设备序列号 - 查询参数优先
            target_device_sn = None
            
            if device_sn:
                target_device_sn = device_sn.strip()
                self.logger.info(f"使用查询参数删除设备: {target_device_sn}")
            elif request_body and request_body.device_sn:
                target_device_sn = request_body.device_sn.strip()
                self.logger.info(f"使用请求体删除设备: {target_device_sn}")
            
            # 参数验证
            if not target_device_sn:
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "message": "请提供设备序列号，支持查询参数(?device_sn=xxx)或请求体({'device_sn': 'xxx'})",
                        "data": None,
                        "timestamp": datetime.now().isoformat()
                    }
                )
            
            # 执行删除逻辑
            sql_provider = SqlProvider(model=DeviceData, sql_config_path=self.sql_config_path)
            devices = sql_provider.get_record_by_condition(
                condition={"device_code": target_device_sn}
            )
            
            if not devices:
                return JSONResponse(
                    status_code=404,
                    content={
                        "success": False,
                        "message": f"设备 {target_device_sn} 不存在",
                        "data": None,
                        "timestamp": datetime.now().isoformat()
                    }
                )
            
            # 删除设备
            for device in devices:
                sql_provider.delete_record(record_id=device["id"])
            
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": f"设备 {target_device_sn} 删除成功",
                    "data": None,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            self.logger.error(f"删除设备异常: {str(e)}")
            import traceback
            traceback.print_exc()
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": f"删除设备失败: {str(e)}",
                    "data": None,
                    "timestamp": datetime.now().isoformat()
                }
            )



if __name__ == '__main__':
    from pathlib import Path
    import asyncio
    from real_time_vital_analyze.tables.device_data import DeviceData
    from real_time_vital_analyze.provider.sql_provider import SqlProvider
    ROOT_DIRECTORY = Path(__file__).parent.parent
    SQL_CONFIG_PATH = str(ROOT_DIRECTORY / "config" / "yaml" / "sql_config.yaml")
    print(SQL_CONFIG_PATH)
    device_data_server = DeviceDataServer(sql_config_path=SQL_CONFIG_PATH)

    async def main():
        result = await device_data_server.get_device_data_api(
            device_sn="13271C9D10004071111715B507"
        )
        print(result)
    
    
    asyncio.run(main())