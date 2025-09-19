#!/usr/bin/env python3
"""
极简HTTP服务器 - 一行命令启动
"""
import http.server
import socketserver

PORT = 8890

with socketserver.TCPServer(("0.0.0.0", PORT), http.server.SimpleHTTPRequestHandler) as httpd:
    print(f"服务器运行在 http://0.0.0.0:{PORT}")
    print(f"本地访问: http://localhost:{PORT}")
    print("按 Ctrl+C 停止")
    httpd.serve_forever()