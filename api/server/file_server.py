from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException, Depends
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
import os
import mimetypes
from typing import Optional
import logging
from urllib.parse import unquote
from fastapi import UploadFile, File
from datetime import datetime
import requests
import base64
import time

class FileServer:
    """文件服务器类，提供文件访问和下载功能"""
    
    def __init__(self, root_directory: str):
        self.root_directory = Path(root_directory)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 确保根目录存在
        if not self.root_directory.exists():
            raise ValueError(f"根目录不存在: {self.root_directory}")
    
    
    def register_routes(self, app: FastAPI):
        """注册文件服务相关路由"""
        
        @app.post("/api/files/upload")
        async def upload_file(
            request: Request,
            file: Optional[UploadFile] = File(None),
            filename: Optional[str] = Form(""),
            user_id: Optional[str] = Form(""),
            extract_content: Optional[str] = Form("")
        ):
            """上传docx文件 - 支持文件上传和base64编码两种方式"""
            print(f"接收到参数:")
            print(f"filename: {filename}")
            print(f"user_id: {user_id}")
            print(f"extract_content: {extract_content}")
            try:
                content_type = request.headers.get('content-type', '')
                print(f"content_type: -------------------------------------------- {content_type}")
                # 检查是否是JSON请求（base64方式）
                if 'application/json' in content_type:
                    # 处理JSON请求（base64数据）
                    data = await request.json()
                    capabilityAssessmentDocument = data.get('capabilityAssessmentDocument')
                    filename = data.get('filename')
                    elderlyId = data.get('elderlyId')
                    elderlyAbilityEvaluationScore = data.get('elderlyAbilityEvaluationScore')
                    exercisesAbilityEvaluationScore = data.get('exercisesAbilityEvaluationScore')
                    mentalStateEvaluationScore = data.get('mentalStateEvaluationScore')
                    sensationPerceptionSocialParticipationScore = data.get('sensationPerceptionSocialParticipationScore')
                    print(f"user_id: ------------------------ {elderlyId}")
                    print(f"filename: ------------------------ {filename}")
                    print(f"elderlyAbilityEvaluationScore: ------------------------ {elderlyAbilityEvaluationScore}")
                    print(f"exercisesAbilityEvaluationScore: ------------------------ {exercisesAbilityEvaluationScore}")
                    print(f"mentalStateEvaluationScore: ------------------------ {mentalStateEvaluationScore}")
                    print(f"sensationPerceptionSocialParticipationScore: ------------------------ {sensationPerceptionSocialParticipationScore}")
                    if not capabilityAssessmentDocument and not file:
                        return JSONResponse(
                            status_code=400,
                            content={
                                "success": False, 
                                "message": "base64_data参数和file参数不能同时为空", 
                                "timestamp": datetime.now().isoformat()
                            }
                        )
                    
                    if not filename:
                        filename = f"document_{int(time.time())}.docx"
                    elif not filename.endswith('.docx'):
                        filename += '.docx'
                    
                    
                    if capabilityAssessmentDocument:
                        print(f"file: {filename}")
                        print(f"elderlyId: {elderlyId}")
                        print(f"file: {extract_content}")
                        try:
                            # 解码base64数据
                            import base64
                            file_content = base64.b64decode(capabilityAssessmentDocument)
                        except Exception as e:
                            return JSONResponse(
                                status_code=400,
                                content={
                                    "success": False, 
                                    "message": f"base64数据解码失败: {str(e)}", 
                                    "timestamp": datetime.now().isoformat()
                                }
                            )
                    
                        # 保存文件
                        upload_path = Path("/work/ai/real_time_vital_analyze/api/source")
                        upload_path.mkdir(parents=True, exist_ok=True)
                        
                        file_path = upload_path / filename
                        
                        with open(file_path, "wb") as buffer:
                            buffer.write(file_content)
                        
                        return JSONResponse(
                            status_code=200,
                            content={
                                "success": True, 
                                "message": f"Base64文件上传成功-{str(file_path)}", 
                                "user_id": user_id,
                                "filename": filename,
                                "file_path": str(file_path),
                                "timestamp": datetime.now().isoformat()
                            }
                        )
                else:
                    # 处理传统文件上传方式
                    if not file:
                        return JSONResponse(
                            status_code=400,
                            content={
                                "success": False, 
                                "message": "请提供文件或base64数据", 
                                "timestamp": datetime.now().isoformat()
                            }
                        )
                    
                    print(f"filename: {filename}")
                    print(f"user_id: {user_id}")
                    print(f"extract_content: {extract_content}")
                    
                    
                    # 检查文件类型
                    if not file.filename.endswith('.docx'):
                        return JSONResponse(
                            status_code=400,
                            content={
                                "success": False, 
                                "message": "只支持上传docx文件", 
                                "timestamp": datetime.now().isoformat()
                            }
                        )
                    
                    # 使用提供的文件名或原始文件名
                    final_filename = filename if filename else file.filename
                    if not final_filename.endswith('.docx'):
                        final_filename += '.docx'
                    
                    # 目标上传路径
                    upload_path = Path("/work/ai/real_time_vital_analyze/api/source")
                    upload_path.mkdir(parents=True, exist_ok=True)
                    
                    file_path = upload_path / final_filename
                    
                    # 保存文件
                    with open(file_path, "wb") as buffer:
                        content = await file.read()
                        buffer.write(content)
                    
                    return JSONResponse(
                        status_code=200,
                        content={
                            "success": True, 
                            "message": f"文件上传成功-{str(file_path)}", 
                            "user_id": user_id,
                            "filename": final_filename,
                            "file_path": str(file_path),
                            "timestamp": datetime.now().isoformat()
                        }
                    )
                
            except Exception as e:
                self.logger.error(f"文件上传失败: {str(e)}")
                return JSONResponse(
                    status_code=500,
                    content={
                        "success": False, 
                        "message": f"文件上传失败: {str(e)}", 
                        "timestamp": datetime.now().isoformat()
                    }
                )
        
        
        @app.get("/api/files/list")
        async def list_files(path: str = ""):
            """列出指定路径下的文件和文件夹"""
            try:
                target_path = self.root_directory / path if path else self.root_directory
                
                if not target_path.exists():
                    return JSONResponse(
                        status_code=400,
                        content={"success": False, "message": f"路径不存在", "timestamp": datetime.now().isoformat()}
                    )
                if not target_path.is_dir():
                    return JSONResponse(
                        status_code=400,
                        content={"success": False, "message": f"指定路径不是文件夹", "timestamp": datetime.now().isoformat()}
                    )
                
                items = []
                for item in target_path.iterdir():
                    item_info = {
                        "name": item.name,
                        "path": str(item.relative_to(self.root_directory)),
                        "is_directory": item.is_dir(),
                        "size": item.stat().st_size if item.is_file() else None,
                        "modified_time": item.stat().st_mtime
                    }
                    items.append(item_info)
                
                data = {
                    "current_path": str(target_path.relative_to(self.root_directory)),
                    "items": sorted(items, key=lambda x: (not x["is_directory"], x["name"]))
                }

                return JSONResponse(
                    status_code=200,
                    content={"success": True, "data": data, "timestamp": datetime.now().isoformat()}
                )
            except Exception as e:
                return JSONResponse(
                    status_code=500,
                    content={"success": False, "message": f"列出文件失败: {e}", "timestamp": datetime.now().isoformat()}
                )
        
        
        @app.get("/api/files/download/{file_path:path}")
        async def download_file(file_path: str):
            """下载指定文件"""
            try:
                # 对路径进行URL解码，处理中文文件名
                decoded_file_path = unquote(file_path)
                file_full_path = self.root_directory / decoded_file_path
                
                # 安全检查：确保文件在根目录内
                if not str(file_full_path.resolve()).startswith(str(self.root_directory.resolve())):
                    return JSONResponse(
                        status_code=400,
                        content={"success": False, "message": f"访问被拒绝！", "timestamp": datetime.now().isoformat()}
                    )
                
                if not file_full_path.exists():
                    return JSONResponse(
                        status_code=400,
                        content={"success": False, "message": f"文件不存在！", "timestamp": datetime.now().isoformat()}
                    )
                
                if not file_full_path.is_file():
                    return JSONResponse(
                        status_code=400,
                        content={"success": False, "message": f"指定路径不是文件", "timestamp": datetime.now().isoformat()}
                    )
                
                # 获取文件MIME类型
                mime_type, _ = mimetypes.guess_type(str(file_full_path))
                if mime_type is None:
                    mime_type = "application/octet-stream"
                
                self.logger.info(f"访问文件: {decoded_file_path}")
                
                # 修改这里：针对HTML文件直接在浏览器显示
                if file_full_path.suffix.lower() in ['.html', '.htm']:
                    from fastapi import Response
                    
                    # 读取HTML文件内容
                    with open(file_full_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    
                    return Response(
                        content=html_content,
                        media_type="text/html",
                        headers={
                            "Access-Control-Allow-Origin": "*",
                            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                            "Access-Control-Allow-Headers": "*",
                            "X-Frame-Options": "ALLOWALL",
                            "X-Content-Type-Options": "nosniff",
                            "Referrer-Policy": "no-referrer-when-downgrade",
                            "Content-Security-Policy": "default-src * 'unsafe-inline' 'unsafe-eval' data: blob:; script-src * 'unsafe-inline' 'unsafe-eval'; connect-src * 'unsafe-inline'; img-src * data: blob: 'unsafe-inline'; frame-src *; style-src * 'unsafe-inline'; font-src *;",
                        }
                    )
            except Exception as e:
                return JSONResponse(
                    status_code=500,
                    content={"success": False, "message": f"访问文件失败: {str(e)}", "timestamp": datetime.now().isoformat()}
                )
        
        
        @app.get("/api/files/download/{file_path:path}")
        async def download_file_(file_path: str):
            """下载指定文件"""
            try:
                # 对路径进行URL解码，处理中文文件名
                decoded_file_path = unquote(file_path)
                file_full_path = self.root_directory / decoded_file_path
                
                # 安全检查：确保文件在根目录内
                if not str(file_full_path.resolve()).startswith(str(self.root_directory.resolve())):
                    return JSONResponse(
                        status_code=400,
                        content={"success": False, "message": f"访问被拒绝！", "timestamp": datetime.now().isoformat()}
                    )
                
                if not file_full_path.exists():
                    return JSONResponse(
                        status_code=400,
                        content={"success": False, "message": f"文件不存在！", "timestamp": datetime.now().isoformat()}
                    )
                
                if not file_full_path.is_file():
                    return JSONResponse(
                        status_code=400,
                        content={"success": False, "message": f"指定路径不是文件", "timestamp": datetime.now().isoformat()}
                    )
                
                # 获取文件MIME类型
                mime_type, _ = mimetypes.guess_type(str(file_full_path))
                if mime_type is None:
                    mime_type = "application/octet-stream"
                
                self.logger.info(f"下载文件: {decoded_file_path}")
                
                return FileResponse(
                    path=str(file_full_path),
                    media_type=mime_type,
                    filename=file_full_path.name
                )
            except Exception as e:
                return JSONResponse(
                    status_code=500,
                    content={"success": False, "message": f"下载文件失败: {str(e)}", "timestamp": datetime.now().isoformat()}
                )
                

def create_app():
    app = FastAPI(
        title="OnlyOffice文档编辑器",
        description="支持URL传参编辑、内容替换和回调保存的OnlyOffice编辑器服务",
        version="2.1.0"
    )
    ROOT_DIRECTORY = Path(__file__).parent.parent.parent
    file_server = FileServer(str(ROOT_DIRECTORY / "api" / "source"))
    file_server.register_routes(app)
    return app


if __name__ == '__main__':
    import uvicorn
    app = create_app()
    print("启动OnlyOffice编辑器: http://localhost:8002")
    print("支持URL传参编辑、内容替换和回调保存功能")
    print("保存API: https://ai.shunxikj.com:5002/api/files/upload")
    print("使用方式: /edit_url?url=文档URL&filename=文件名&replace_information=[{\"from\":\"原文本\",\"to\":\"新文本\"}]")
    
    uvicorn.run(app, host="127.0.0.1", port=5003, log_level="info")

    
    