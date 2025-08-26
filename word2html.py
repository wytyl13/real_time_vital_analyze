"""
最简单的OnlyOffice Word文档可编辑应用
部署后即可上传和编辑Word文档
"""

from flask import Flask, request, jsonify, render_template, send_file, url_for
import os
import json
import time
import hashlib
import requests
from werkzeug.utils import secure_filename
from pathlib import Path

app = Flask(__name__)
app.config['SECRET_KEY'] = 'simple-onlyoffice-key'
app.config['UPLOAD_FOLDER'] = 'documents'
app.config['ONLYOFFICE_SERVER'] = 'http://onlyoffice-server:80'

# 创建文档目录
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 支持的文件类型
ALLOWED_EXTENSIONS = {'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_document_key(filepath):
    """生成文档唯一标识"""
    stat = os.stat(filepath)
    return hashlib.md5(f"{filepath}_{stat.st_mtime}".encode()).hexdigest()

@app.route('/')
def index():
    """主页面 - 显示文档列表"""
    documents = []
    
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if allowed_file(filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            stat = os.stat(filepath)
            documents.append({
                'filename': filename,
                'size': f"{stat.st_size / 1024:.1f} KB",
                'modified': time.strftime('%Y-%m-%d %H:%M', time.localtime(stat.st_mtime)),
                'edit_url': url_for('edit_document', filename=filename)
            })
    
    return render_template('simple_editor.html', documents=documents)

@app.route('/upload', methods=['POST'])
def upload_file():
    """上传文档"""
    if 'file' not in request.files:
        return jsonify({'error': '没有文件'}), 400
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': '文件格式不支持'}), 400
    
    filename = secure_filename(file.filename)
    
    # 如果文件存在，添加时间戳
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(filepath):
        name, ext = os.path.splitext(filename)
        filename = f"{name}_{int(time.time())}{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    file.save(filepath)
    
    return jsonify({
        'success': True,
        'filename': filename,
        'edit_url': url_for('edit_document', filename=filename)
    })

@app.route('/edit/<filename>')
def edit_document(filename):
    """编辑文档页面"""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return '文档不存在', 404
    
    # OnlyOffice编辑器配置
    config = {
        "document": {
            "fileType": Path(filename).suffix[1:].lower(),
            "key": get_document_key(filepath),
            "title": filename,
            "url": url_for('serve_document', filename=filename, _external=True),
            "permissions": {
                "edit": True,
                "download": True,
                "print": True
            }
        },
        "documentType": get_document_type(filename),
        "editorConfig": {
            "mode": "edit",
            "lang": "zh-CN",
            "callbackUrl": url_for('save_callback', filename=filename, _external=True),
            "user": {
                "id": "user1",
                "name": "Editor"
            }
        },
        "height": "600px",
        "width": "100%"
    }
    
    return render_template('simple_document_editor.html', 
                         filename=filename,
                         config=json.dumps(config),
                         onlyoffice_server=app.config['ONLYOFFICE_SERVER'])

def get_document_type(filename):
    """获取文档类型"""
    ext = Path(filename).suffix.lower()
    if ext in ['.docx', '.doc']:
        return 'word'
    elif ext in ['.xlsx', '.xls']:
        return 'cell'
    elif ext in ['.pptx', '.ppt']:
        return 'slide'
    return 'word'

@app.route('/documents/<filename>')
def serve_document(filename):
    """提供文档访问"""
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

@app.route('/callback/<filename>', methods=['POST'])
def save_callback(filename):
    """OnlyOffice保存回调"""
    try:
        data = request.get_json()
        status = data.get('status')
        
        if status in [2, 3]:  # 文档准备保存或保存错误
            doc_url = data.get('url')
            if doc_url:
                # 下载并保存文档
                response = requests.get(doc_url)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                print(f"文档已保存: {filename}")
        
        return jsonify({"error": 0})
    
    except Exception as e:
        print(f"保存回调错误: {e}")
        return jsonify({"error": 1}), 500

@app.route('/delete/<filename>', methods=['POST'])
def delete_document(filename):
    """删除文档"""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if os.path.exists(filepath):
        os.remove(filepath)
        return jsonify({'success': True})
    
    return jsonify({'error': '文件不存在'}), 404

if __name__ == '__main__':
    print("🚀 OnlyOffice编辑器启动中...")
    print(f"📁 文档目录: {os.path.abspath(app.config['UPLOAD_FOLDER'])}")
    print(f"🌐 OnlyOffice服务器: {app.config['ONLYOFFICE_SERVER']}")
    print(f"💻 访问地址: http://localhost:5000")
    
    # 检查OnlyOffice服务器
    try:
        response = requests.get(f"{app.config['ONLYOFFICE_SERVER']}/healthcheck", timeout=5)
        if response.json().get('status') == 'ok':
            print("✅ OnlyOffice服务器连接成功")
        else:
            print("❌ OnlyOffice服务器状态异常")
    except:
        print("⚠️  警告: 无法连接OnlyOffice服务器")
        print("   请确保运行: docker run -d -p 8080:80 onlyoffice/documentserver")
    
    app.run(debug=True, host='0.0.0.0', port=5000)