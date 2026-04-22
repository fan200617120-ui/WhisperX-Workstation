#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FireRedASR2S WebUI Professional Edition
Copyright 2026 光影的故事2018

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import gradio as gr
import subprocess
import sys
import time
import os
import argparse
from pathlib import Path

# ==================== 配置路径 ====================
BASE_DIR = Path(__file__).parent
PYTHON_EXE = BASE_DIR / "python_embeded" / "python.exe"
if not PYTHON_EXE.exists():
    PYTHON_EXE = sys.executable  # 降级到系统 Python

SCRIPTS_DIR = BASE_DIR / "core"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)  # 确保输出目录存在

def launch_script(script_name):
    """启动指定脚本"""
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        return f"❌ 脚本 {script_name} 不存在"
    try:
        subprocess.Popen([str(PYTHON_EXE), str(script_path)], cwd=str(SCRIPTS_DIR))
        return f"✅ 已启动 {script_name}，请查看新窗口"
    except Exception as e:
        return f"❌ 启动失败：{e}"

def open_folder(path):
    """使用系统文件管理器打开文件夹"""
    folder = Path(path)
    if not folder.exists():
        return f"❌ 目录不存在：{path}"
    try:
        if sys.platform == "win32":
            os.startfile(str(folder))
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(folder)])
        else:
            subprocess.Popen(["xdg-open", str(folder)])
        return f"📂 已打开：{folder}"
    except Exception as e:
        return f"❌ 打开失败：{e}"

def refresh_status():
    """刷新系统状态（可扩展为获取CPU/内存等信息）"""
    return f"状态：运行中 ({time.strftime('%H:%M:%S')})"

# ==================== 自定义 CSS（电脑自适应）====================
custom_css = """
body {
    background: radial-gradient(circle at top, #1f2a37, #0b1014);
    min-height: 100vh;
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 20px;
    color: #fff;
}
.gradio-container {
    background: transparent !important;
    max-width: 900px !important;
    width: 100% !important;
    margin: 0 auto !important;
}
.panel {
    background: rgba(255,255,255,0.03);
    border-radius: 24px;
    padding: 28px;
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 20px 60px rgba(0,0,0,0.4);
}
.title {
    font-size: 24px;
    font-weight: 600;
    margin-bottom: 6px;
    color: white;
}
.sub {
    font-size: 13px;
    color: #9aa3b2;
    margin-bottom: 26px;
}
.btn-group {
    display: flex;
    flex-direction: column;
    gap: 14px;
    margin-bottom: 22px;
}
.btn-group .gr-button {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 18px !important;
    padding: 20px 22px !important;
    color: white !important;
    font-size: 16px !important;
    font-weight: 500 !important;
    text-align: left !important;
    backdrop-filter: blur(10px) !important;
    box-shadow: none !important;
    transition: all 0.2s ease;
}
.btn-group .gr-button:hover {
    background: rgba(255,255,255,0.1) !important;
    border-color: rgba(255,255,255,0.2) !important;
    transform: translateY(-2px);
}
.status {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 16px;
    font-size: 14px;
    color: #cdd7e3;
    margin-bottom: 18px;
}
.notice {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 14px;
    font-size: 12px;
    color: #9aa3b2;
    line-height: 1.5;
    margin-bottom: 12px;
}
.copyright {
    text-align: center;
    font-size: 11px;
    color: #6b7280;
}
.bili {
    color: #3b82f6;
    text-decoration: none;
}
"""

# ==================== 构建界面 ====================
with gr.Blocks(css=custom_css, title="语音字幕工作站") as demo:
    with gr.Column(elem_classes="panel"):
        gr.HTML('<div class="title">语音字幕工作站</div>')
        gr.HTML('<div class="sub">本地AI工具 · 高效简洁 · 一键启动</div>')

        # 按钮组
        with gr.Column(elem_classes="btn-group"):
            btn_asr = gr.Button("📄 语音转字幕", elem_id="btn_asr")
            btn_align = gr.Button("🎬 语音转字幕（对齐）", elem_id="btn_align")  # 新增对齐脚本按钮
            btn_clean = gr.Button("✨ 字幕清洗", elem_id="btn_clean")
            btn_trans = gr.Button("🔤 字幕转换", elem_id="btn_trans")
            btn_translate = gr.Button("🤖 在线AI助手", elem_id="btn_translate")
            btn_json_extract = gr.Button("📋 双语字幕API版", elem_id="btn_json_extract")

        # 状态显示
        status = gr.Textbox(
            label="",
            value=refresh_status(),
            elem_classes="status",
            interactive=False
        )

        # 新增快捷目录按钮
        with gr.Row():
            btn_open_root = gr.Button("📁 打开根目录", size="sm")
            btn_open_output = gr.Button("📂 打开输出目录", size="sm")

        # 页脚版权与免责声明
        gr.HTML("""
        <div class="notice">
            注意事项：<br>
            • 本工具仅用于个人学习与视频剪辑使用<br>
            • 禁止用于商业用途及侵权行为<br>            
            • 使用前确保模型与依赖环境正常配置
        </div>
        <div style="text-align: center; color: #666; font-size: 0.9em;">
            <p>本软件包不提供任何模型文件，模型由用户自行从官方渠道获取。用户需自行遵守模型的原许可证。</p>
            <p>本软件包按“原样”提供，不提供任何明示或暗示的担保。使用本软件所产生的一切风险由用户自行承担。</p>
            <p>本软件包开发者不对因使用本软件而导致的任何直接或间接损失负责。</p>       
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 8px; margin: 15px auto; max-width: 600px;">
                <p style="color: white; font-weight: bold; margin: 5px 0; font-size: 1em;">🎬 更新请关注B站up主：光影的故事2018</p>
                <p style="color: white; margin: 5px 0; font-size: 0.9em;">
                🔗 <strong>B站主页</strong>: 
                <a href="https://space.bilibili.com/381518712" target="_blank" style="color: #ffdd40; text-decoration: none; font-weight: bold;">
                space.bilibili.com/381518712
                </a>
                </p>
            </div>
        </div>
        <div style="text-align: center; color: #666; margin-top: 10px; font-size: 0.9em;">
            © 原创 WebUI 代码 © 2026 光影紐扣 版权所有 | 基于 FireRedASR2S (Apache 2.0) 二次开发
        </div>
        """)

    # ==================== 绑定点击事件 ====================
    btn_asr.click(fn=lambda: launch_script("whisperX.py"), outputs=status)
    btn_align.click(fn=lambda: launch_script("whisperX_sub_align.py"), outputs=status)  # 绑定对齐脚本
    btn_clean.click(fn=lambda: launch_script("clean_subtitle.py"), outputs=status)
    btn_trans.click(fn=lambda: launch_script("subtitle_utils.py"), outputs=status)
    btn_translate.click(fn=lambda: launch_script("AI_translator.py"), outputs=status)
    btn_json_extract.click(fn=lambda: launch_script("subtitle_translator_pro.py"), outputs=status)

    # 打开目录事件
    btn_open_root.click(fn=lambda: open_folder(BASE_DIR), outputs=status)
    btn_open_output.click(fn=lambda: open_folder(OUTPUT_DIR), outputs=status)

if __name__ == "__main__":
    # 解析命令行参数，支持指定端口（多网口备选可通过多次运行不同端口实现）
    parser = argparse.ArgumentParser(description="语音字幕工作站 WebUI")
    parser.add_argument("--port", type=int, default=7868,
                        help="服务端口（默认 7868，可自定义如 7968、7988）")
    args = parser.parse_args()

    print(f"启动 WebUI 于端口 {args.port}，访问地址：http://127.0.0.1:{args.port}")
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        inbrowser=True
    )