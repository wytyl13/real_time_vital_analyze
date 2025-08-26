#!/usr/bin/env python3
"""
OnlyOffice在线文档编辑器 - 最简化版本
"""

from flask import Flask, request, render_template_string, jsonify, send_file
import hashlib
import json
import time
import os

app = Flask(__name__)

# OnlyOffice配置
ONLYOFFICE_SERVER = 'http://ai.shunxikj.com:5555'
JWT_SECRET = 'THMDlTXvRQJogxp2uCTG9DJRlhu4cor8'

@app.route('/')
def index():
    """首页直接跳转到编辑器"""
    return '<script>window.location.href="/edit_url?url=dummy";</script>'

@app.route('/edit_url')
def edit_url():
    """编辑在线文档"""
    # 固定文件路径
    file_path = '/work/ai/real_time_vital_analyze/api/source/5.docx'
    
    # 基本检查
    if not os.path.exists(file_path):
        return f"文件不存在：{file_path}", 400
    
    filename = os.path.basename(file_path)
    file_ext = filename.lower().split('.')[-1]
    
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
    local_file_url = f"http://ai.shunxikj.com:8002/serve_file?path={file_path}"
    
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
                "goback": False                 # 隐藏返回按钮
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
        except:
            pass
    
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
            padding: 20px; border-radius: 10px;
        }
        .spinner {
            width: 40px; height: 40px; border: 4px solid #f3f3f3;
            border-top: 4px solid #4f46e5; border-radius: 50%;
            animation: spin 1s linear infinite; margin: 0 auto 10px;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="loading" id="loading">
        <div class="spinner"></div>
        <div>正在加载编辑器...</div>
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
                    }
                };
                
                new DocsAPI.DocEditor("editor", config);
            } catch(error) {
                document.getElementById('loading').innerHTML = 
                    '<div>加载失败: ' + error.message + '</div>';
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
    file_path = request.args.get('path')
    if not file_path or not file_path.startswith('/work/ai/real_time_vital_analyze/api/source/'):
        return "文件路径无效", 403
    
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
        return {"error": 0}
    except:
        return {"error": 1}

if __name__ == '__main__':
    print("启动OnlyOffice编辑器: http://localhost:8002")
    app.run(debug=True, host='0.0.0.0', port=8002)