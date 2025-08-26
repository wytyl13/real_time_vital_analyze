from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import os
import mimetypes
from typing import Optional
import logging
from urllib.parse import unquote

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
        
        @app.get("/api/files/list")
        async def list_files(path: str = ""):
            """列出指定路径下的文件和文件夹"""
            try:
                target_path = self.root_directory / path if path else self.root_directory
                
                if not target_path.exists():
                    raise HTTPException(status_code=404, detail="路径不存在")
                
                if not target_path.is_dir():
                    raise HTTPException(status_code=400, detail="指定路径不是文件夹")
                
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
                
                return {
                    "current_path": str(target_path.relative_to(self.root_directory)),
                    "items": sorted(items, key=lambda x: (not x["is_directory"], x["name"]))
                }
                
            except Exception as e:
                self.logger.error(f"列出文件失败: {e}")
                raise HTTPException(status_code=500, detail=f"列出文件失败: {str(e)}")
        
        @app.get("/api/files/download/{file_path:path}")
        async def download_file(file_path: str):
            """下载指定文件"""
            try:
                # 对路径进行URL解码，处理中文文件名
                decoded_file_path = unquote(file_path)
                file_full_path = self.root_directory / decoded_file_path
                
                # 安全检查：确保文件在根目录内
                if not str(file_full_path.resolve()).startswith(str(self.root_directory.resolve())):
                    raise HTTPException(status_code=403, detail="访问被拒绝")
                
                if not file_full_path.exists():
                    raise HTTPException(status_code=404, detail="文件不存在")
                
                if not file_full_path.is_file():
                    raise HTTPException(status_code=400, detail="指定路径不是文件")
                
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
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"下载文件失败: {e}")
                raise HTTPException(status_code=500, detail=f"下载文件失败: {str(e)}")
        
        @app.get("/api/files/info/{file_path:path}")
        async def get_file_info(file_path: str):
            """获取文件详细信息"""
            try:
                file_full_path = self.root_directory / file_path
                
                # 安全检查
                if not str(file_full_path.resolve()).startswith(str(self.root_directory.resolve())):
                    raise HTTPException(status_code=403, detail="访问被拒绝")
                
                if not file_full_path.exists():
                    raise HTTPException(status_code=404, detail="文件不存在")
                
                stat = file_full_path.stat()
                mime_type, _ = mimetypes.guess_type(str(file_full_path))
                
                return {
                    "name": file_full_path.name,
                    "path": file_path,
                    "size": stat.st_size,
                    "mime_type": mime_type,
                    "is_directory": file_full_path.is_dir(),
                    "created_time": stat.st_ctime,
                    "modified_time": stat.st_mtime,
                    "accessed_time": stat.st_atime
                }
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"获取文件信息失败: {e}")
                raise HTTPException(status_code=500, detail=f"获取文件信息失败: {str(e)}")
        
        @app.get("/api/files/search")
        async def search_files(query: str, path: str = ""):
            """搜索文件"""
            try:
                search_path = self.root_directory / path if path else self.root_directory
                
                if not search_path.exists():
                    raise HTTPException(status_code=404, detail="搜索路径不存在")
                
                results = []
                for file_path in search_path.rglob("*"):
                    if query.lower() in file_path.name.lower():
                        results.append({
                            "name": file_path.name,
                            "path": str(file_path.relative_to(self.root_directory)),
                            "is_directory": file_path.is_dir(),
                            "size": file_path.stat().st_size if file_path.is_file() else None
                        })
                
                return {
                    "query": query,
                    "search_path": str(search_path.relative_to(self.root_directory)),
                    "results": results[:100]  # 限制结果数量
                }
                
            except Exception as e:
                self.logger.error(f"搜索文件失败: {e}")
                raise HTTPException(status_code=500, detail=f"搜索文件失败: {str(e)}")