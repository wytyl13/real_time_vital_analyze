#!/usr/bin/env python3
"""
OnlyOffice在线文档编辑器 - 支持动态docx_url参数
"""

from flask import Flask, request, render_template_string, jsonify, send_file
import hashlib
import json
import time
import os
import urllib.parse
from urllib.parse import urlparse

app = Flask(__name__)

# OnlyOffice配置
ONLYOFFICE_SERVER = 'http://ai.shunxikj.com:5555'
JWT_SECRET = 'THMDlTXvRQJogxp2uCTG9DJRlhu4cor8'

# 允许的文件路径前缀（安全限制）
ALLOWED_PATH_PREFIXES = [
    '/work/ai/real_time_vital_analyze/api/source/',
    '/var/documents/',
    '/home/documents/'
]

@app.route('/')
def index():
    """首页 - 支持docx_url参数"""
    docx_url = request.args.get('docx_url')
    
    if docx_url:
        # 如果提供了docx_url参数，跳转到编辑器
        encoded_url = urllib.parse.quote(docx_url, safe='')
        return f'<script>window.location.href="/edit_url?url={encoded_url}";</script>'
    else:
        # 如果没有提供参数，显示使用说明
        return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>OnlyOffice在线文档编辑器</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 50px; background: #f5f5f5; }
        .container { max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .title { color: #333; text-align: center; margin-bottom: 30px; }
        .usage { background: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
        .example { background: #e3f2fd; padding: 15px; border-radius: 5px; margin-top: 10px; }
        .code { font-family: monospace; background: #fff; padding: 2px 5px; border: 1px solid #ddd; border-radius: 3px; }
        .form-group { margin-bottom: 15px; }
        .form-control { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; box-sizing: border-box; }
        .btn { background: #4f46e5; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
        .btn:hover { background: #3730a3; }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="title">OnlyOffice在线文档编辑器</h1>
        
        <div class="usage">
            <h3>使用方法：</h3>
            <p>在URL后添加 <span class="code">docx_url</span> 参数来指定要编辑的文档路径</p>
            
            <div class="example">
                <strong>示例：</strong><br>
                <span class="code">http://1.71.15.121:8002/?docx_url=/work/ai/real_time_vital_analyze/api/source/5.docx</span>
            </div>
        </div>
        
        <form action="/" method="get">
            <div class="form-group">
                <label for="docx_url">文档路径：</label>
                <input type="text" id="docx_url" name="docx_url" class="form-control" 
                       placeholder="/work/ai/real_time_vital_analyze/api/source/your_file.docx">
            </div>
            <button type="submit" class="btn">打开文档</button>
        </form>
        
        <div class="usage">
            <h3>支持的文件类型：</h3>
            <ul>
                <li>Word文档：.doc, .docx</li>
                <li>Excel表格：.xls, .xlsx</li>
                <li>PowerPoint演示：.ppt, .pptx</li>
            </ul>
        </div>
    </div>
</body>
</html>
        ''')

@app.route('/edit_url')
def edit_url():
    """编辑在线文档 - 从URL参数获取文件路径"""
    # 从URL参数获取文件路径
    file_url = request.args.get('url')
    if not file_url:
        return "缺少文件路径参数", 400
    
    # URL解码
    file_path = urllib.parse.unquote(file_url)
    
    # 安全检查：确保文件路径在允许的目录中
    is_allowed = False
    for prefix in ALLOWED_PATH_PREFIXES:
        if file_path.startswith(prefix):
            is_allowed = True
            break
    
    if not is_allowed:
        return f"文件路径不被允许：{file_path}<br>允许的路径前缀：{ALLOWED_PATH_PREFIXES}", 403
    
    # 基本检查
    if not os.path.exists(file_path):
        return f"文件不存在：{file_path}", 404
    
    filename = os.path.basename(file_path)
    file_ext = filename.lower().split('.')[-1]
    
    # 检查文件类型
    supported_formats = ['doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx']
    if file_ext not in supported_formats:
        return f"不支持的文件格式：{file_ext}，支持的格式：{supported_formats}", 400
    
    # 文档类型映射
    type_mapping = {
        'doc': 'word', 'docx': 'word', 
        'xls': 'cell', 'xlsx': 'cell',
        'ppt': 'slide', 'pptx': 'slide'
    }
    document_type = type_mapping.get(file_ext, 'word')
    
    # 生成文档密钥
    file_stat = os.stat(file_path)
    doc_key = hashlib.md5(f"{file_path}-{file_stat.st_mtime}".encode()).hexdigest()
    
    # 文件服务URL
    encoded_path = urllib.parse.quote(file_path, safe='')
    local_file_url = f"http://ai.shunxikj.com:8002/serve_file?path={encoded_path}"
    
    # OnlyOffice配置
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
                "changeHistory": True,          # 添加：修改历史权限
                "rename": True                  # 添加：重命名权限
            }
        },
        "editorConfig": {
            "lang": "zh-CN",
            "mode": "edit",
            "callbackUrl": f"http://ai.shunxikj.com:8002/callback?key={doc_key}",
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
                "compactToolbar": False,        # 添加：不使用紧凑工具栏
                "toolbarNoTabs": False,         # 添加：显示工具栏标签
                "hideRightMenu": False,         # 添加：不隐藏右键菜单
                "about": False,                 # 隐藏关于信息
                "feedback": False,              # 隐藏反馈
                "goback": {
                    "url": f"http://1.71.15.121:8002/",  # 返回首页
                    "text": "返回"
                }
            },
            # 添加：协作设置
            "coEditing": {
                "mode": "fast",                 # 快速协作模式
                "change": True                  # 允许修改
            }
        },
        "width": "100%",
        "height": "100%"
    }
    
    # JWT token
    if JWT_SECRET:
        try:
            import jwt
            payload = config.copy()
            payload["iat"] = int(time.time())
            payload["exp"] = int(time.time()) + 3600
            config["token"] = jwt.encode(payload, JWT_SECRET, algorithm='HS256')
        except ImportError:
            print("警告：JWT库未安装，将在不使用token的情况下运行")
        except Exception as e:
            print(f"JWT生成失败：{e}")
    
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>编辑: {{ filename }}</title>
    <style>
        body { margin: 0; padding: 0; overflow: hidden; height: 100vh; }
        #editor { width: 100%; height: 100vh; border: none; }
        .loading {
            position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
            text-align: center; z-index: 1000; background: rgba(255,255,255,0.9);
            padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .spinner {
            width: 40px; height: 40px; border: 4px solid #f3f3f3;
            border-top: 4px solid #4f46e5; border-radius: 50%;
            animation: spin 1s linear infinite; margin: 0 auto 10px;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .file-info {
            position: fixed; top: 10px; right: 10px; background: rgba(0,0,0,0.7);
            color: white; padding: 5px 10px; border-radius: 5px; font-size: 12px; z-index: 999;
        }
    </style>
</head>
<body>
    <div class="file-info">正在编辑：{{ filename }}</div>
    
    <div class="loading" id="loading">
        <div class="spinner"></div>
        <div>正在加载编辑器...</div>
        <div style="font-size: 12px; color: #666; margin-top: 10px;">文件：{{ filename }}</div>
    </div>
    
    <div id="editor"></div>
    
    <script src="{{ onlyoffice_server }}/web-apps/apps/api/documents/api.js"></script>
    <script>
        let config = {{ config|safe }};
        
        window.onload = function() {
            try {
                config.events = {
                    'onAppReady': function() {
                        document.getElementById('loading').style.display = 'none';
                        console.log('OnlyOffice编辑器已准备就绪');
                    },
                    'onDocumentReady': function() {
                        console.log('文档已加载完成');
                    },
                    'onError': function(event) {
                        console.error('OnlyOffice错误:', event);
                        document.getElementById('loading').innerHTML = 
                            '<div style="color: red;">加载失败: ' + (event.data || '未知错误') + '</div>';
                    }
                };
                
                new DocsAPI.DocEditor("editor", config);
            } catch(error) {
                console.error('编辑器初始化失败:', error);
                document.getElementById('loading').innerHTML = 
                    '<div style="color: red;">加载失败: ' + error.message + '</div>';
            }
        };
    </script>
</body>
</html>
    ''', 
    filename=filename,
    config=json.dumps(config),
    onlyoffice_server=ONLYOFFICE_SERVER)

@app.route('/serve_file')
def serve_file():
    """提供文件服务"""
    file_path = urllib.parse.unquote(request.args.get('path', ''))
    
    if not file_path:
        return "缺少文件路径参数", 400
    
    # 安全检查：确保文件路径在允许的目录中
    is_allowed = False
    for prefix in ALLOWED_PATH_PREFIXES:
        if file_path.startswith(prefix):
            is_allowed = True
            break
    
    if not is_allowed:
        return "文件路径不被允许", 403
    
    if not os.path.exists(file_path):
        return "文件不存在", 404
    
    try:
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        return f"文件服务错误: {str(e)}", 500

@app.route('/callback', methods=['POST'])
def callback():
    """OnlyOffice回调"""
    try:
        data = request.get_json() or {}
        print(f"OnlyOffice回调: {data}")
        
        # 根据状态处理不同的回调
        status = data.get('status', 0)
        if status == 1:
            print("文档正在编辑")
        elif status == 2:
            print("文档准备保存")
            # 这里可以添加保存逻辑
        elif status == 3:
            print("文档保存出错")
        elif status == 4:
            print("文档关闭，无修改")
        elif status == 6:
            print("文档正在编辑，但当前用户已断开连接")
        elif status == 7:
            print("文档保存出错，强制保存")
        
        return {"error": 0}
    except Exception as e:
        print(f"回调处理错误: {e}")
        return {"error": 1}

if __name__ == '__main__':
    print("启动OnlyOffice编辑器服务...")
    print("服务地址: http://1.71.15.121:8002")
    print("使用方法: http://1.71.15.121:8002/?docx_url=/your/file/path.docx")
    print(f"允许的文件路径前缀: {ALLOWED_PATH_PREFIXES}")
    app.run(debug=True, host='0.0.0.0', port=8002)