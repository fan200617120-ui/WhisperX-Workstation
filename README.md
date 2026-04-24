# WhisperX-Workstation
WhisperX Studio – All-in-One Speech &amp; Subtitle Workstation
# 🎙️ WhisperX 语音字幕工作站 – 使用手册

> 轻舟渡万境，一智载千寻 
> 基于 FireRedASR2S 二次开发，整合语音识别、字幕处理、AI 翻译、格式转换等一站式工具集。

---

## 📋 目录

1. [项目简介](#项目简介)
2. [系统要求](#系统要求)
3. [快速安装](#快速安装)
4. [工具模块详解](#工具模块详解)
   - 4.1 语音转字幕 (whisperX)
   - 4.2 字幕清洗 (clean_subtitle)
   - 4.3 字幕处理工具箱 (subtitle_utils)
   - 4.4 在线 AI 助手 (AI_translator)
   - 4.5 双语字幕翻译 Pro (subtitle_translator_pro)
5. [启动器界面](#启动器界面)
6. [常见问题](#常见问题)
7. [许可证与法律声明](#许可证与法律声明)

---

## 项目简介

**WhisperX 语音字幕工作站** 是一个本地化的 AI 辅助视频/音频处理工具包，主要功能包括：

- 🎤 **语音识别转字幕** – 基于 faster-whisper，支持 GPU 加速，输出 SRT/JSON/TXT。
- ✨ **字幕清洗** – 去除语气词、重复词，支持自定义词库。
- 🔧 **字幕处理** – 双语合并、SRT↔ASS 互转、拼音添加、LRC 转 SRT、文本转字幕。
- 🤖 **在线 AI 助手** – 集成常用翻译/大模型网站快捷入口 + 专业提示词模板。
- 🌍 **双语字幕翻译 Pro** – 使用大模型 API（DeepSeek/通义千问等）进行上下文感知翻译，支持批量处理。

所有工具均采用便携式设计，解压即用，无需额外配置 Python 环境。

---
![WhisperX-Workstation 计算精度设置界面](https://raw.githubusercontent.com/fan200617120-ui/WhisperX-Workstation/main/image/2026-04-06_234604.png)
## 系统要求

| 组件 | 最低要求 | 推荐配置 |
|------|----------|----------|
| 操作系统 | Windows 10 / 11 | Windows 11 |
| CPU | Intel Core i5 或同等 | Intel Core i7 / AMD Ryzen 7 |
| 内存 | 8 GB | 16 GB |
| 显卡 | 无（CPU 模式） | NVIDIA GTX 1060 或更高 (CUDA) |
| 存储空间 | 5 GB（含基础模型） | 10 GB+ |
| 依赖 | 解压后即可运行（已内置 Python） | — |

> 注意：语音识别首次使用需下载模型（约 1.5~3 GB），请保持网络畅通。

---

## 快速安装

1. **下载压缩包**  
   从发布页下载 `WhisperX_Workstation.zip` 并解压到任意目录（路径建议不含中文和空格）。

2. **目录结构**  
   ```
   whisperX/
   ├── core/               # 所有核心脚本
   ├── ffmpeg/             # 内置 FFmpeg（视频处理）
   ├── python_embeded/     # 便携版 Python 环境
   ├── pretrained_models/  # 语音识别模型（自动下载）
   ├── output/             # 所有输出文件
   ├── logs/               # 错误日志
   ├── preset/             # 用户配置预设
   ├── Index_Public_release.py   # 主启动器
   └── 启动器.bat          # Windows 快捷启动
   ```

3. **首次启动**  
   - 双击 `启动器.bat` 或运行 `Index_Public_release.py`。
   - 主界面会显示 5 个工具按钮，点击即可在独立窗口中打开对应工具。
   - **语音转字幕** 首次使用时需要选择并加载模型（建议选择 `medium` 或 `large-v3`，会自动下载）。

4. **安装额外依赖（可选）**  
   如需使用 **繁简转换** 功能，请手动安装 opencc：
   ```cmd
   .\python_embeded\python.exe -m pip install opencc-python-reimplemented
   ```
   如需使用 **拼音添加** 功能，请安装 pypinyin：
   ```cmd
   .\python_embeded\python.exe -m pip install pypinyin
   ```

---

## 工具模块详解

### 4.1 语音转字幕 (whisperX)

**启动方式**：点击主界面「语音转字幕」按钮，或在 `core` 目录下运行 `whisperX.py`。

**功能**：
- 音频/视频转文字（支持麦克风录制、文件上传）。
- 自动生成 SRT 字幕、JSON 时间戳、纯文本。
- 视频字幕硬/软嵌入。
- **强制对齐**：将用户提供的稿子文本与音频对齐，生成逐字/逐词精准字幕。

**使用步骤**：

| 标签页 | 操作说明 |
|--------|----------|
| **音频识别** | 上传音频 → 选择模型（首次需加载）→ 点击「开始识别」→ 预览并下载 SRT/TXT/JSON。 |
| **视频字幕** | 上传视频 → 选择字幕嵌入模式（软/硬）→ 开始处理 → 得到带字幕的视频文件。 |
| **批量处理** | 上传多个音频文件 → 批量识别并保存字幕。 |
| **字幕自动打轴** | 上传音频 + 粘贴文稿（段落间用空行分隔）→ 选择对齐粒度（字符级/单词级）→ 生成逐词/整句子幕。 |

**系统状态面板**：显示 GPU 信息、模型加载状态、输出目录等。

> 💡 **提示**：若使用 GTX 1080 等旧显卡，建议选择 CUDA 12.6 版本（代码已适配），并关闭“激进模式”以保证稳定性。

---

### 4.2 字幕清洗 (clean_subtitle)

**启动方式**：主界面「字幕清洗」按钮，或运行 `core/clean_subtitle.py`。

**功能**：
- 去除语气词（如“那个”、“然后”、“啊”、“嗯”等）。
- 删除自定义的重复词或冗余词。
- 支持 SRT 格式保留时间轴，或纯文本清洗。

**使用步骤**：
1. 上传 SRT/TXT 文件或直接粘贴文本。
2. 按需勾选 **激进模式**（会去除句中语气词）。
3. 在 **自定义词库** 区域添加要删除的词（每行一个），可加载/保存为 TXT 文件。
4. 点击「开始清洗」→ 预览结果并下载。

**内置语气词库示例**：`啊`、`呀`、`就是`、`那个`、`然后`、`哎哟喂` 等。

---

### 4.3 字幕处理工具箱 (subtitle_utils)

**启动方式**：主界面「字幕转换」按钮，或运行 `core/subtitle_utils.py`。  
**端口**：`http://127.0.0.1:18006`（可修改）

**功能一览**：

| 标签页 | 功能 |
|--------|------|
| 双语合并 | 将中文和英文字幕合并为上下对照格式。 |
| SRT转TXT | 提取字幕纯文本。 |
| 添加拼音 | 为中文 SRT 添加拼音（可选带声调/数字）。 |
| 文本转字幕 | 按时间码格式（如 `0:00`）的文本转为 SRT。 |
| LRC转SRT | 将歌词文件转换为字幕。 |
| 格式转换 | SRT↔ASS 互转、ASS 提取 TXT、TXT 简易转 SRT。 |
| **繁简转换** | 使用 OpenCC 进行简繁转换（需安装依赖）。 |

**使用示例**：  
- **双语合并**：上传中文 SRT 和英文 SRT → 合并为一行中文一行英文的 SRT。  
- **繁简转换**：上传 SRT/ASS/TXT → 选择转换模式（繁→简 / 简→繁 / 台湾正体 / 香港繁体）→ 输出同格式或跨格式文件。

---

### 4.4 在线 AI 助手 (AI_translator)

**启动方式**：主界面「在线AI助手」按钮，或运行 `core/AI_translator.py`。  
**端口**：`http://127.0.0.1:18009`

**功能**：
- **快速入口**：一键打开 DeepL、有道、DeepSeek、通义千问、Kimi 等网站。
- **提示词模板**：内置专业提示词（SRT 翻译、语义断句、双语字幕、文本润色等），可复制后粘贴到上述网站使用。

**使用步骤**：
1. 打开「快速入口」标签页，点击任意网站按钮即可跳转。
2. 切换到「提示词模板」标签页，选择需要的模板（如“SRT 字幕翻译”）。
3. 复制提示词内容，粘贴到打开的翻译/大模型网站中，并将自己的字幕内容附在提示词后面即可。

---

### 4.5 双语字幕翻译 Pro (subtitle_translator_pro)

**启动方式**：主界面「双语字幕API版」按钮，或运行 `core/subtitle_translator_pro.py`。  
**端口**：`http://127.0.0.1:7869`

**功能**：
- 调用 OpenAI 格式的大模型 API（支持 DeepSeek、通义千问、Moonshot、Ollama 等）进行字幕翻译。
- 支持**上下文感知**（前后窗口）、批量翻译加速。
- 输出双语 SRT（上下对照/原文优先/仅译文）。

**使用步骤**：
1. 获取 API Key（如从 DeepSeek 官网注册）。
2. 在界面中填写：
   - `API Key`
   - `API Base URL`（预设了几个常用服务）
   - `模型名称`（如 `deepseek-chat`、`qwen-plus`）
3. 上传 SRT 文件或粘贴字幕内容。
4. 设置目标语言（默认“简体中文”）、批量大小、上下文窗口等。
5. 点击「开始翻译」，等待结果并下载。

> ⚠️ 注意：该工具消耗 API 额度，建议先少量测试。

---

## 启动器界面

主启动器 `Index_Public_release.py` 提供统一入口，运行后显示五个按钮：

- 📄 **语音转字幕** → 启动 whisperX.py（端口 18006）
- ✨ **字幕清洗** → 启动 clean_subtitle.py（端口 18007）
- 🔤 **字幕转换** → 启动 subtitle_utils.py（端口 18006）
- 🤖 **在线AI助手** → 启动 AI_translator.py（端口 18009）
- 📋 **双语字幕API版** → 启动 subtitle_translator_pro.py（端口 7869）

所有工具运行在独立窗口中，互不干扰。底部显示状态栏和版权信息。

---

## 常见问题

### 1. 启动时报错“未找到嵌入版 Python”
   - 请确保解压完整，`python_embeded` 文件夹与 `core` 同级。

### 2. 语音转字幕时提示“Audio data must be floating-point”
   - 已在新版代码中修复（自动归一化整数音频）。如果仍出现，请更新 `whisperX.py` 到最新版本。

### 3. 视频字幕处理失败（找不到 ffmpeg）
   - 检查 `ffmpeg/bin` 目录下是否有 `ffmpeg.exe`（Windows）或 `ffmpeg`（Linux/Mac）。便携版已内置。

### 4. 模型下载慢或失败
   - 首次使用会从 Hugging Face 下载模型（如 `base`、`medium`）。可预先将模型文件夹放入 `pretrained_models` 目录（命名如 `Systran--faster-whisper-medium`）。

### 5. 强制对齐结果不准确
   - 确保稿子文本与音频内容完全一致（包括标点符号）。
   - 使用“字符级”对齐适合中文，“单词级”适合英文。
   - 勾选“根据空行断句”可严格按段落分割。

### 6. 繁简转换提示缺少 opencc
   - 运行命令：`.\python_embeded\python.exe -m pip install opencc-python-reimplemented`

### 7. 双语翻译 Pro 返回空结果
   - 检查 API Key 是否有效，Base URL 是否以 `/v1` 结尾。
   - 降低批量大小（例如改为 3），关闭上下文窗口再试。

---

## 许可证与法律声明

本项目基于 **Apache License, Version 2.0** 开源。

### 附加声明

- 本软件包**不提供任何模型文件**，模型由用户自行从官方渠道（如 Hugging Face）获取。用户需自行遵守模型的原许可证（如 MIT、Apache 或 CC 等）。
- 本软件包按“原样”提供，不提供任何明示或暗示的担保。使用本软件所产生的一切风险由用户自行承担。
- 开发者不对因使用本软件而导致的任何直接或间接损失负责。
- 禁止将本软件用于商业用途或侵权行为（仅供个人学习与视频剪辑）。

### 致谢

- FireRedASR2S 原始项目 (Apache 2.0)
- faster-whisper (MIT)
- gradio (Apache 2.0)
- opencc-python-reimplemented (Apache 2.0)
- pypinyin (MIT)

---

## 📧 联系与更新

- **B站主页**：[光影的故事2018](https://space.bilibili.com/381518712)
- 欢迎反馈 Bug、提出建议。本手册随代码同步更新。

**版本**：v2.0 (2026-04-06)

通过网盘分享的文件：wishperX
链接: https://pan.baidu.com/s/1QAfmoKyrsb0xK6ACAY8ENg?pwd=6688 提取码: 6688


*轻舟渡万境，一智载千寻 — 愿你的创作如行云流水。*
