#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/09/02 15:10
@Author  : weiyutao
@File    : test_server.py
"""


#!/usr/bin/env python3
"""
OnlyOffice在线文档编辑器 - FastAPI版本，支持URL传参和回调保存
"""

from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pathlib import Path
import hashlib
import json
import time
import os
import requests
import tempfile
import urllib.parse
from urllib.parse import urlparse
import mimetypes
import logging
from typing import Optional
from datetime import datetime




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DocX文档字段提取工具 - 使用python-docx结构化提取
"""

from docx import Document
from docx.table import Table, _Cell
from docx.text.paragraph import Paragraph
from typing import Dict, Optional, Any, List, Union
import json
import re





class OnlyOfficeEditor:
    """OnlyOffice文档编辑器类"""
    
    def __init__(self, onlyoffice_server: str, save_api_url: str, jwt_secret: str = ""):
        self.onlyoffice_server = onlyoffice_server
        self.save_api_url = save_api_url
        self.jwt_secret = jwt_secret
        self.temp_dir = tempfile.gettempdir()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        
        # 文档类型映射
        self.type_mapping = {
            'doc': 'word', 'docx': 'word', 
            'xls': 'cell', 'xlsx': 'cell',
            'ppt': 'slide', 'pptx': 'slide'
        }


    def download_file_from_url(self, url: str, filename: Optional[str] = None) -> tuple[str, str]:
        """从URL下载文件到临时目录"""
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # 如果没有提供filename，尝试从响应中获取
            if not filename:
                # 首先尝试从Content-Disposition头获取
                content_disposition = response.headers.get('content-disposition')
                if content_disposition:
                    import re
                    match = re.search(r'filename[*]?=([^;]+)', content_disposition)
                    if match:
                        filename = match.group(1).strip('"\'')
                
                # 如果还没有filename，从URL路径获取
                if not filename:
                    parsed_url = urlparse(url)
                    filename = os.path.basename(parsed_url.path)
                
                # 如果仍然没有filename，使用默认名称
                if not filename or '.' not in filename:
                    content_type = response.headers.get('content-type', '')
                    if 'wordprocessing' in content_type:
                        filename = 'document.docx'
                    elif 'spreadsheet' in content_type:
                        filename = 'document.xlsx'
                    elif 'presentation' in content_type:
                        filename = 'document.pptx'
                    else:
                        filename = 'document.docx'
            
            # 确保filename有正确的扩展名
            if '.' not in filename:
                filename += '.docx'
            
            # 生成临时文件路径
            temp_path = os.path.join(self.temp_dir, f"onlyoffice_{int(time.time())}_{filename}")
            
            # 保存文件
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            return temp_path, filename
            
        except Exception as e:
            raise Exception(f"下载文件失败: {str(e)}")


    def upload_file_to_api(self, file_content: bytes, filename: str) -> dict:
            """上传文件到保存API"""
            try:
                files = {
                    'file': (filename, file_content, 'application/octet-stream')
                }
                data = {
                    'filename': filename
                }
                
                response = requests.post(self.save_api_url, files=files, data=data, timeout=30)
                response.raise_for_status()
                
                return response.json()
            except Exception as e:
                raise Exception(f"上传文件到API失败: {str(e)}")


    def cleanup_temp_files(self):
        """清理临时文件"""
        try:
            for filename in os.listdir(self.temp_dir):
                if filename.startswith('onlyoffice_') and os.path.isfile(os.path.join(self.temp_dir, filename)):
                    file_path = os.path.join(self.temp_dir, filename)
                    # 删除超过1小时的临时文件
                    if time.time() - os.path.getctime(file_path) > 3600:
                        os.remove(file_path)
                        self.logger.info(f"清理临时文件: {file_path}")
        except Exception as e:
            self.logger.error(f"清理临时文件出错: {e}")


    def generate_onlyoffice_config(self, file_path: str, filename: str, document_url: str) -> dict:
        """生成OnlyOffice配置"""
        file_ext = filename.lower().split('.')[-1]
        document_type = self.type_mapping.get(file_ext, 'word')
        
        # 生成文档密钥
        file_stat = os.stat(file_path)
        doc_key = hashlib.md5(f"{document_url}-{file_stat.st_mtime}".encode()).hexdigest()
        
        # 文件服务URL
        local_file_url = f"http://ai.shunxikj.com:8002/serve_temp_file?path={urllib.parse.quote(file_path)}"
        
        config = {
            "documentType": document_type,
            "document": {
                "fileType": file_ext,
                "key": doc_key,
                "title": filename,
                "url": local_file_url,
                "permissions": {
                    "comment": True,
                    "copy": True,
                    "download": True,
                    "edit": True,
                    "fillForms": True,              # 表单填写权限
                    "modifyFilter": True,           # 修改筛选器权限
                    "modifyContentControl": True,   # 修改内容控件权限
                    "review": True,
                    "print": True,
                    "changeHistory": True,          # 修改历史权限
                    "rename": True                  # 重命名权限
                }
            },
            "editorConfig": {
                "lang": "zh-CN",
                "mode": "edit",
                "callbackUrl": f"http://ai.shunxikj.com:8002/callback?key={doc_key}&filename={urllib.parse.quote(filename)}&original_url={urllib.parse.quote(document_url)}",
                "user": {
                    "id": "user-1",
                    "name": "编辑用户"
                },
                "customization": {
                    "leftMenu": False,
                    "toolbar": True,
                    "autosave": True,
                    "forcesave": False,
                    "submitForm": False,
                    "compactToolbar": False,
                    "toolbarNoTabs": False,
                    "hideRightMenu": False,
                    "about": False,
                    "feedback": False,
                    "goback": False
                },
                "coEditing": {
                    "mode": "fast",
                    "change": True
                }
            },
            "width": "100%",
            "height": "100%"
        }
        
        # JWT token处理
        if self.jwt_secret:
            try:
                import jwt
                payload = config.copy()
                payload["iat"] = int(time.time())
                payload["exp"] = int(time.time()) + 3600
                config["token"] = jwt.encode(payload, self.jwt_secret, algorithm='HS256')
            except Exception as e:
                self.logger.warning(f"JWT token生成失败: {e}")
        
        return config


    def register_routes(self, app: FastAPI):
        """注册OnlyOffice编辑器相关路由"""
        @app.get("/edit_url", response_class=HTMLResponse)
        async def edit_url(
            url: str = Query(..., description="文档URL"),
            filename: Optional[str] = Query(None, description="自定义文件名")
        ):
            """编辑在线文档 - 支持URL参数"""
            try:
                if not url:
                    raise HTTPException(status_code=400, detail="缺少url参数")
                
                # 下载文件到临时目录
                file_path, auto_filename = self.download_file_from_url(url, filename)
                
                # 使用自定义filename（如果提供）
                final_filename = filename if filename else auto_filename
                
                # 生成OnlyOffice配置
                config = self.generate_onlyoffice_config(file_path, final_filename, url)
                
                # 生成HTML模板
                html_template = f'''
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>编辑: {final_filename}</title>
    <style>
        body {{ margin: 0; padding: 0; overflow: hidden; height: 100vh; }}
        #editor {{ width: 100%; height: 100vh; border: none; }}
        .loading {{
            position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
            text-align: center; z-index: 1000; background: rgba(255,255,255,0.9);
            padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .spinner {{
            width: 40px; height: 40px; border: 4px solid #f3f3f3;
            border-top: 4px solid #4f46e5; border-radius: 50%;
            animation: spin 1s linear infinite; margin: 0 auto 10px;
        }}
        @keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}
    </style>
</head>
<body>
    <div class="loading" id="loading">
        <div class="spinner"></div>
        <div>正在加载编辑器...</div>
        <div style="font-size: 12px; margin-top: 10px;">{final_filename}</div>
        <div style="font-size: 10px; margin-top: 5px;">来源: {url}</div>
    </div>
    
    <div id="editor"></div>
    
    <script src="{self.onlyoffice_server}/web-apps/apps/api/documents/api.js"></script>
    <script>
        let config = {json.dumps(config)};
        
        console.log('文档URL:', '{url}');
        console.log('文件名:', '{final_filename}');
        console.log('表单权限:', config.document.permissions.fillForms);
        console.log('内容控件权限:', config.document.permissions.modifyContentControl);
        console.log('编辑模式:', config.editorConfig.mode);
        console.log('完整配置:', config);
        
        window.onload = function() {{
            try {{
                config.events = {{
                    'onAppReady': function() {{
                        console.log('编辑器已就绪，可以编辑表单字段');
                        document.getElementById('loading').style.display = 'none';
                    }},
                    'onDocumentReady': function() {{
                        console.log('文档已加载完成，表单字段应该可以编辑');
                    }},
                    'onRequestEditRights': function() {{
                        console.log('请求编辑权限');
                    }},
                    'onError': function(event) {{
                        console.error('编辑器错误:', event);
                        let errorMsg = '未知错误';
                        if (event && event.data) {{
                            switch(event.data) {{
                                case 1: errorMsg = '文档加载错误'; break;
                                case 2: errorMsg = '回调URL错误'; break;
                                case 3: errorMsg = '内部服务器错误'; break;
                                case 4: errorMsg = '文档密钥错误'; break;
                                case 5: errorMsg = '回调文档状态错误'; break;
                                case 6: errorMsg = '回调文档格式错误'; break;
                                default: errorMsg = `错误代码: ${{event.data}}`;
                            }}
                        }}
                        document.getElementById('loading').innerHTML = 
                            '<div>编辑器错误: ' + errorMsg + '</div>';
                    }}
                }};
                
                new DocsAPI.DocEditor("editor", config);
                console.log('编辑器实例已创建');
            }} catch(error) {{
                console.error('初始化失败:', error);
                document.getElementById('loading').innerHTML = 
                    '<div>加载失败: ' + error.message + '</div>';
            }}
        }};
    </script>
</body>
</html>
                '''
                
                return HTMLResponse(content=html_template, status_code=200)
                
            except Exception as e:
                self.logger.error(f"编辑文档失败: {e}")
                raise HTTPException(status_code=500, detail=f"处理文档失败: {str(e)}")
        
        
        @app.get("/serve_temp_file")
        async def serve_temp_file(path: str = Query(..., description="临时文件路径")):
            """提供临时文件服务"""
            try:
                if not path:
                    raise HTTPException(status_code=400, detail="文件路径无效")
                
                # 解码路径
                file_path = urllib.parse.unquote(path)
                
                # 安全检查：确保文件在临时目录中
                if not file_path.startswith(self.temp_dir):
                    raise HTTPException(status_code=403, detail="文件路径无效")
                
                if not os.path.exists(file_path):
                    raise HTTPException(status_code=404, detail="文件不存在")
                
                # 获取文件扩展名来设置正确的Content-Type
                file_ext = os.path.splitext(file_path)[1].lower()
                content_type_mapping = {
                    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                    '.doc': 'application/msword',
                    '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    '.xls': 'application/vnd.ms-excel',
                    '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
                    '.ppt': 'application/vnd.ms-powerpoint'
                }
                content_type = content_type_mapping.get(file_ext, 'application/octet-stream')
                
                filename = os.path.basename(file_path)
                
                return FileResponse(
                    path=file_path,
                    media_type=content_type,
                    filename=filename
                )
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"文件服务错误: {e}")
                raise HTTPException(status_code=500, detail=f"文件服务错误: {str(e)}")
        
        
        @app.post("/callback")
        async def callback(
            request: Request,
            key: str = Query("unknown", description="文档密钥"),
            filename: str = Query("document.docx", description="文件名"),
            original_url: str = Query("", description="原始URL")
        ):
            """OnlyOffice回调 - 支持保存到API"""
            try:
                data = await request.json()
                
                # 解码filename
                filename = urllib.parse.unquote(filename)
                
                self.logger.info(f"OnlyOffice回调 - Key: {key}, Filename: {filename}, Status: {data.get('status')}")
                
                # 处理保存状态
                status = data.get('status', 0)
                if status == 2:
                    # 文档准备保存
                    download_url = data.get('url')
                    if download_url:
                        self.logger.info(f"文档需要保存，下载URL: {download_url}")
                        try:
                            # 下载编辑后的文档
                            response = requests.get(download_url, timeout=30)
                            response.raise_for_status()
                            
                            # 上传到保存API
                            upload_result = self.upload_file_to_api(response.content, filename)
                            self.logger.info(f"文档已保存到API: {upload_result}")
                            
                            # 清理临时文件
                            self.cleanup_temp_files()
                            
                        except Exception as e:
                            self.logger.error(f"保存文档时出错: {e}")
                            return JSONResponse(
                                status_code=200,
                                content={"error": 1, "message": str(e)}
                            )
                
                return JSONResponse(
                    status_code=200,
                    content={"error": 0}
                )
                
            except Exception as e:
                self.logger.error(f"回调处理错误: {e}")
                return JSONResponse(
                    status_code=200,
                    content={"error": 1, "message": str(e)}
                )
        
        
        @app.get("/health")
        async def health():
            """健康检查"""
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "data": {
                        "status": "healthy",
                        "onlyoffice_server": self.onlyoffice_server,
                        "save_api_url": self.save_api_url,
                        "temp_dir": self.temp_dir
                    },
                    "timestamp": datetime.now().isoformat()
                }
            )


def create_app():
    app = FastAPI(
        title="OnlyOffice文档编辑器",
        description="支持URL传参编辑和回调保存的OnlyOffice编辑器服务",
        version="2.0.0"
    )
    # 配置参数
    ONLYOFFICE_SERVER = 'http://ai.shunxikj.com:8442'
    SAVE_API_URL = 'https://ai.shunxikj.com:5002/api/files/upload'
    JWT_SECRET = ''  # 暂时禁用JWT

    editor = OnlyOfficeEditor(
        onlyoffice_server=ONLYOFFICE_SERVER,
        save_api_url=SAVE_API_URL,
        jwt_secret=JWT_SECRET
    )
    editor.register_routes(app)
    return app

            
if __name__ == '__main__':
    import uvicorn
    app = create_app()
    print("启动OnlyOffice编辑器: http://localhost:8002")
    print("支持URL传参编辑和回调保存功能")
    print("保存API: https://ai.shunxikj.com:5002/api/files/upload")
    print("使用方式: /edit_url?url=文档URL&filename=文件名")
    
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")