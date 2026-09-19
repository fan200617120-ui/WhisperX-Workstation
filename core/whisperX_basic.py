#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WhisperX 语音识别独立增强版
- 音频识别 / 视频字幕 / 批量处理
- 可选：使用 wav2vec2 对齐模型精细化单词/字符时间戳
- 自动扫描本地 faster-whisper 模型和对齐模型
- 不包含文稿对齐，无额外 pyannote 依赖
Copyright 2026 光影的故事2018
"""

import sys, os, json, logging, traceback, time, gc, threading, atexit, tempfile, hashlib, re, subprocess, shutil, uuid
from pathlib import Path
from typing import List, Dict, Optional, Tuple

sys.setrecursionlimit(10000)

# ==================== 日志 ====================
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

def clean_old_logs(days=7):
    cutoff = time.time() - days*24*3600
    for f in LOG_DIR.glob("error_*.log"):
        if f.stat().st_mtime < cutoff:
            try: f.unlink()
            except OSError: pass
clean_old_logs()
log_file = LOG_DIR / f"error_{time.strftime('%Y%m%d')}.log"
logging.basicConfig(filename=log_file, level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== 路径 ====================
CURRENT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
BASE_DIR = Path(__file__).parent.absolute()
ROOT_DIR = BASE_DIR.parent
DEFAULT_OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
PRESET_DIR = ROOT_DIR / "preset"
PRESET_DIR.mkdir(exist_ok=True)
CONFIG_FILE = PRESET_DIR / "settings.json"
config_lock = threading.RLock()

# 临时文件目录（硬字幕滤镜需要把 srt 复制到无中文的路径）
CACHE_DIR = ROOT_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# ==================== FFmpeg ====================
PORTABLE_FFMPEG_DIR = ROOT_DIR / "ffmpeg" / "bin"
if sys.platform == "win32":
    PORTABLE_FFMPEG_EXE = PORTABLE_FFMPEG_DIR / "ffmpeg.exe"
else:
    PORTABLE_FFMPEG_EXE = PORTABLE_FFMPEG_DIR / "ffmpeg"
if PORTABLE_FFMPEG_EXE.exists():
    os.environ["PATH"] = str(PORTABLE_FFMPEG_DIR) + os.pathsep + os.environ.get("PATH", "")
    FFMPEG_PATH = str(PORTABLE_FFMPEG_EXE)
    print(f"[OK] 已自动加载内置 FFmpeg: {FFMPEG_PATH}")
else:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        FFMPEG_PATH = system_ffmpeg
        print(f"[OK] 使用系统已安装的 FFmpeg: {FFMPEG_PATH}")
    else:
        FFMPEG_PATH = "ffmpeg"
        print("[WARN] 未找到内置 FFmpeg，视频处理可能失败，请将 ffmpeg 放入 ffmpeg/bin 目录。")

def load_settings():
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f: return json.load(f)
        except Exception as e:
            # 修复：原为裸 `except: return {}`，settings.json 损坏时会静默回退默认值，
            # 用户看不到任何提示（输出目录、截断长度都悄悄变了）。
            print(f"[WARN] 读取设置失败（{CONFIG_FILE}）: {e}，将使用默认值")
            return {}
    return {}

def save_settings(settings):
    # 修复：原为裸 `except: pass`，设置写盘失败（磁盘满 / 无权限 / 文件被占用）
    # 会被完全静默，用户改了输出目录却不知道为什么没生效。
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f: json.dump(settings, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] 设置保存失败（{CONFIG_FILE}）: {e}")
        logger.warning(f"设置保存失败: {e}")

# ==================== 导入核心依赖 ====================
try:
    import gradio as gr
    import torch, numpy as np, librosa, soundfile as sf
    from faster_whisper import WhisperModel
    print(f"PyTorch: {torch.__version__}, CUDA可用: {torch.cuda.is_available()}")
    print("faster-whisper: 已导入")
except ImportError as e:
    print(f"基础依赖缺失: {e}")
    sys.exit(1)

# 尝试导入 whisperx.align（faster-whisper 已在上面）
try:
    from whisperx import load_align_model, align as whisperx_align
    WHISPERX_ALIGN_AVAILABLE = True
    print("whisperx.align: 可用")
except ImportError:
    WHISPERX_ALIGN_AVAILABLE = False
    print("提示: 未找到 whisperx.align，精细对齐选项将不可用。")

# ==================== 工具函数 ====================
MAX_OUTPUT_TEXT_LENGTH = 50000

# 精度候选（顺序即下拉框显示顺序，与原默认保持一致：int8_float32 优先）
_COMPUTE_PREFERENCE = ["int8_float32", "float16", "float32", "int8_float16",
                       "int8", "int16", "bfloat16"]


def supported_compute_types(device):
    """查询 CTranslate2 在该设备上**真正支持**的精度。

    修复：界面原本无条件提供 ["int8_float32", "float16", "float32"]，
    但 float16 能不能用取决于显卡。实测本机 GTX 1080（算力 6.1）在
    ctranslate2 4.4.0 下 CUDA 只支持 {int8, int8_float32, float32}，
    **不支持 float16** —— 用户在下拉框选了 float16 必然报
    "Requested float16 compute type, but the target device or backend
     do not support efficient float16 computation."。
    现改为按实际支持的精度构造下拉框，并在加载时做运行时校验。
    """
    try:
        import ctranslate2
        supported = set(ctranslate2.get_supported_compute_types(device))
    except Exception as e:
        print(f"[WARN] 无法查询 {device} 支持的精度（{e}），回退默认列表")
        supported = {"int8_float32", "float32"}
        if device != "cpu":
            supported.add("float16")
    choices = [c for c in _COMPUTE_PREFERENCE if c in supported]
    return choices or ["int8_float32", "float32"]


def run_ffmpeg(cmd, timeout=3600):
    """执行 ffmpeg 命令；失败时把 stderr 的真实原因包进异常抛出。

    修复两处：
    1) 原代码 `subprocess.run(..., capture_output=True)` 把 stderr 收走却从不输出，
       异常分支只回显 str(e)，用户永远只能看到 "returned non-zero exit status 1"。
    2) text=True 未指定 encoding，Windows 下按 GBK 解码 ffmpeg 的 UTF-8 输出，
       文件名含中文时会抛 UnicodeDecodeError，stderr 缓冲区为空。
    """
    try:
        return subprocess.run(cmd, check=True, capture_output=True,
                              text=True, encoding='utf-8', errors='replace',
                              timeout=timeout)
    except subprocess.CalledProcessError as e:
        stderr = (getattr(e, 'stderr', '') or '').strip()
        tail = stderr[-800:] if stderr else '(ffmpeg 未输出错误信息)'
        raise RuntimeError(f"FFmpeg 执行失败（退出码 {e.returncode}）:\n{tail}") from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"FFmpeg 执行超时（超过 {timeout} 秒），可能是视频过长或进程卡住") from e


def _align_fallback_note(reason: str) -> str:
    """精细对齐失败时的用户可见提示。

    修复：原先对齐失败只 print / 写日志，界面照常显示「完成」，
    用户以为拿到了精细对齐的时间戳，实际是 ASR 原始时间戳。
    """
    r = (reason or "").strip().replace("\n", " ")
    if len(r) > 140:
        r = r[:140] + "…"
    return f"⚠ 精细对齐未生效，已回退为 ASR 原始时间戳（原因：{r}）"


def safe_text(text: str, max_len: int = MAX_OUTPUT_TEXT_LENGTH) -> str:
    if len(text) > max_len:
        return text[:max_len] + "\n\n[注意] 返回内容过长已截断，完整结果已保存至输出目录。"
    return text

def seconds_to_srt_time(seconds: float) -> str:
    total_ms = round(seconds * 1000)
    hours = total_ms // 3600000
    minutes = (total_ms % 3600000) // 60000
    secs = (total_ms % 60000) // 1000
    ms = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"

def _join_segment_texts(segments):
    """按相邻边界决定连接符，避免中文被插入空格、英文被粘在一起。

    修复：原实现是 " ".join(...)，中文 txt 导出成「第一句 第二句」；
    而 whisperX.py 用的是「整篇只要有汉字就全部空串连接」，
    两者都不对——正确做法是逐对判断相邻两段的边界字符。
    """
    parts = []
    for i, seg in enumerate(segments):
        t = (seg.get("text") or "").strip()
        if not t:
            continue
        if parts:
            prev = parts[-1]
            prev_cjk = bool(re.search(r'[\u4e00-\u9fff\u3040-\u30ff]', prev[-1:]))
            cur_cjk = bool(re.search(r'[\u4e00-\u9fff\u3040-\u30ff]', t[:1]))
            # 两侧都非 CJK 时才补空格（英文句间需要空格）
            if not prev_cjk and not cur_cjk:
                parts.append(" ")
        parts.append(t)
    return "".join(parts)

def format_result_to_outputs(result):
    if not result or not isinstance(result, dict):
        return "无结果", "{}", "", []
    segments = result.get("segments", [])
    text = _join_segment_texts(segments)
    ts_json = json.dumps(segments, ensure_ascii=False, indent=2)
    srt = []
    for i, seg in enumerate(segments, 1):
        srt.append(str(i))
        srt.append(f"{seconds_to_srt_time(seg['start'])} --> {seconds_to_srt_time(seg['end'])}")
        srt.append(seg["text"])
        srt.append("")
    srt_text = "\n".join(srt)
    extra = f"语言: {result.get('language','未知')} (概率: {result.get('language_probability',0):.2f})"
    full = f"{text}\n\n[元数据] {extra}"
    return full, ts_json, srt_text, segments

def save_outputs(base_name, full_text, ts_json, srt_text, language, model_info):
    # 修复：原时间戳只到秒，同一秒内多次运行会覆盖同名输出文件
    ts = time.strftime("%Y%m%d_%H%M%S") + f"{int(time.time() * 1000) % 1000:03d}"
    if base_name:
        safe = re.sub(r'[^\w\u4e00-\u9fff\-\.]', '', Path(base_name).stem)
        prefix = f"{safe}_{ts}"
    else:
        prefix = f"whisperx_{ts}"
    saved = {}
    txt_path = OUTPUT_DIR / f"{prefix}.txt"
    with open(txt_path, 'w', encoding='utf-8') as f: f.write(full_text)
    saved['txt'] = str(txt_path)
    if ts_json and ts_json != "{}":
        json_path = OUTPUT_DIR / f"{prefix}.json"
        with open(json_path, 'w', encoding='utf-8') as f: f.write(ts_json)
        saved['json'] = str(json_path)
    if srt_text.strip():
        srt_path = OUTPUT_DIR / f"{prefix}.srt"
        with open(srt_path, 'w', encoding='utf-8') as f: f.write(srt_text)
        saved['srt'] = str(srt_path)
    return saved

def get_system_info():
    info = []
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory/1e9
        allocated = torch.cuda.memory_allocated(0)/1e9
        info.append(f"显卡: {gpu} ({total:.1f} GB)")
        info.append(f"已分配显存: {allocated:.1f} GB")
    else:
        info.append("设备: CPU模式")
    with manager.lock:
        if manager.asr_model:
            info.append(f"ASR模型: {manager.current_asr_model_name}")
            info.append(f"计算类型: {manager.current_compute_type}")
        else:
            info.append("ASR模型: 未加载")
    info.append(f"输出目录: {OUTPUT_DIR}")
    if WHISPERX_ALIGN_AVAILABLE:
        info.append("精细对齐: 可用")
    else:
        info.append("精细对齐: 不可用 (缺少 whisperx.align)")
    return "\n".join(info)

def generate_output_filename(base_input, ts_str, suffix="", default="recording"):
    original = None
    if isinstance(base_input, str) and os.path.exists(base_input):
        original = Path(base_input).stem
    elif isinstance(base_input, dict) and base_input.get('path') and os.path.exists(base_input['path']):
        original = Path(base_input['path']).stem
    elif isinstance(base_input, tuple):
        original = default
    if not original: original = default
    safe = re.sub(r'[^\w\u4e00-\u9fff\-]', '', original)
    if not safe: safe = default
    parts = [safe, ts_str]
    if suffix: parts.append(suffix)
    return "_".join(parts)

# ==================== 对齐模型语言匹配（修复子串误匹配） ====================
# 修复：本文件原先用「语言全名」子串去匹配目录名 ——
#   key = {"zh": "chinese-zh-cn", "en": "english", ...}[detected]
#   if key in disp.lower()      # "chinese-zh-cn" in "wav2vec2-zh"  ->  False !
# 而用户实际的目录名是 wav2vec2-zh / wav2vec2-en，所以「auto」**永远匹配不到本地模型**，
# 会回退到在线模型名，离线环境下白等两分多钟后失败（实测 166 秒）。
# whisperX.py / whisperX_pro.py / whisperX_sub_align.py 用的都是下面这套
# 「先边界匹配语言代码」的逻辑，只有本文件漏改了。
ALIGN_LANG_KEYWORDS = {
    "zh": ["chinese", "mandarin", "zh-cn"], "en": ["english"], "ja": ["japanese"],
    "fr": ["french"], "de": ["german"], "es": ["spanish"], "pt": ["portuguese"],
    "it": ["italian"], "nl": ["dutch"], "hu": ["hungarian"], "ru": ["russian"],
    "pl": ["polish"], "vi": ["vietnamese"], "tr": ["turkish"], "ko": ["korean"],
    "ar": ["arabic"], "sv": ["swedish"], "uk": ["ukrainian"], "fi": ["finnish"],
    "da": ["danish"], "no": ["norwegian"], "cs": ["czech"], "ro": ["romanian"],
    "el": ["greek"], "he": ["hebrew"], "hi": ["hindi"], "th": ["thai"], "id": ["indonesian"],
}


def match_local_align_by_lang(lang, local_align):
    """先边界匹配语言代码，再按语言全名匹配；修复 'en' 子串误命中 'french' 的问题。

    例：目录 wav2vec2-zh 能被 'zh' 命中（`-zh` 处于边界）；
        wav2vec2-en 不会被 'zh' 命中。
    """
    lang = (lang or "").strip().lower()
    if not lang or not local_align:
        return None
    for disp, path in local_align:
        if re.search(rf'(?:^|[_-]){re.escape(lang)}(?:[_-]|$)', disp.lower()):
            return path
    for kw in ALIGN_LANG_KEYWORDS.get(lang, []):
        for disp, path in local_align:
            if kw in disp.lower():
                return path
    return None


# ==================== 模型管理器 ====================
class WhisperXManager:
    def __init__(self):
        self.asr_model = None
        self.current_asr_model_name = None
        self.current_device = None
        self.current_compute_type = None
        self.asr_in_use = 0   # 转写占用计数：转写进行中禁止卸载/切换 ASR 模型，防止显存峰值翻倍
        self.align_in_use = 0  # 对齐占用计数：对齐进行中禁止卸载对齐模型
        self.settings = load_settings()
        self.temp_files = []
        self.lock = threading.RLock()
        # 对齐模型缓存
        self.align_model = None
        self.align_metadata = None
        self.align_model_lang = None
        self.last_align_error = ""  # 最近一次精细对齐的失败原因（用于在结果里如实告知用户）

    def get_available_local_models(self):
        models = []
        models_dir = ROOT_DIR / "pretrained_models"
        if not models_dir.exists(): return []
        for item in models_dir.iterdir():
            if not item.is_dir(): continue
            if (item/"model.bin").exists() or (item/"config.json").exists() or (item/"pytorch_model.bin").exists():
                models.append((item.name, str(item)))
        return models

    def get_local_align_models(self):
        models = []
        models_dir = ROOT_DIR / "pretrained_models"
        if not models_dir.exists():
            return models
        for item in models_dir.iterdir():
            if not item.is_dir():
                continue
            if "wav2vec2" in item.name.lower() or "xlsr" in item.name.lower():
                if ((item / "pytorch_model.bin").exists() or
                    (item / "model.bin").exists() or
                    (item / "config.json").exists()):
                    models.append((item.name, str(item)))
        return models

    def load_asr_model(self, model_size, device, compute_type, language=None):
        with self.lock:
            # 修复：原实现缺少 CPU + float16 校验（whisperX.py 有），
            # faster-whisper 在 CPU 上用 float16 会直接抛错，而错误原因又会被
            # UI 的 outputs 覆盖掉，用户只看到「未加载」不知为何。
            # 现改为按「该设备实际支持的精度」做通用校验 —— 不只是 CPU，
            # 实测 GTX 1080 上 CUDA 也不支持 float16。
            _ok_types = supported_compute_types(device)
            if compute_type not in _ok_types:
                return False, (f"{device} 不支持 {compute_type}，"
                               f"请选择: {'、'.join(_ok_types)}")
            if not model_size:
                return False, "未发现模型：请先在 pretrained_models 目录放置模型文件，然后刷新页面后重试"
            local_path = ROOT_DIR / "pretrained_models" / model_size
            if local_path.exists() and (local_path / "model.bin").exists():
                model_name_or_path = str(local_path)
                local_only = True
            else:
                known = ["tiny","base","small","medium","large-v2","large-v3","large-v3-turbo"]
                if model_size in known:
                    model_name_or_path = model_size
                    local_only = False
                else:
                    available = self.get_available_local_models()
                    found = False
                    for disp, path in available:
                        if disp == model_size:
                            model_name_or_path = path
                            local_only = True
                            found = True
                            break
                    if not found:
                        model_name_or_path = model_size
                        local_only = False
            if self.asr_model is not None and self.current_asr_model_name == model_name_or_path and self.current_device == device and self.current_compute_type == compute_type:
                return True, f"ASR模型已加载: {model_size}"
            # 修复：转写占用期间禁止切换 ASR 模型 —— 旧模型被 del 后仍被转写线程的
            # 本地引用持有，显存不会释放，新模型再加载会峰值翻倍（与 whisperX.py 一致）
            if self.asr_in_use > 0:
                return False, "有转写任务正在进行，暂不能切换 ASR 模型，请等待其完成后再试"
            self.unload_models()
            try:
                self.asr_model = WhisperModel(model_name_or_path, device=device, compute_type=compute_type, local_files_only=local_only)
                self.current_asr_model_name = model_name_or_path
                self.current_device = device
                self.current_compute_type = compute_type
                return True, f"ASR模型加载成功: {model_size}"
            except Exception as e:
                logger.error(traceback.format_exc())
                return False, f"加载ASR模型失败: {str(e)}"

    def unload_models(self):
        # 修复：补上转写/对齐占用保护（whisperX.py / whisperX_pro.py 都有）。
        # 旧模型被 del 后仍被转写线程的局部引用持有，显存不会释放，
        # 实际效果是白白丢弃模型还可能触发新旧模型同时驻留（显存峰值翻倍）。
        with self.lock:
            if self.asr_in_use > 0 or self.align_in_use > 0:
                return False, "模型正在使用中（转写/对齐进行中），请等待任务完成后再卸载"
            if self.asr_model:
                del self.asr_model
                self.asr_model = None
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            # 同时释放对齐模型
            self.unload_align_model()
            return True, "模型已卸载"

    def unload_align_model(self):
        with self.lock:
            if self.align_model is not None:
                del self.align_model
                self.align_model = None
                self.align_metadata = None
                self.align_model_lang = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def transcribe(self, audio_path, language=None, beam_size=5, vad_filter=True, word_timestamps=True, initial_prompt=None):
        # 修复：补上转写占用计数（whisperX.py / whisperX_pro.py 都有，只有本文件缺）。
        # faster-whisper 的 segments 是惰性生成器，旧模型被 del 后仍被本函数的局部引用持有，
        # 显存不会释放，再加载新模型时新旧同时驻留、峰值翻倍，容易 OOM。
        # 计数需覆盖下方完整消费过程，所以用 try/finally 包住。
        with self.lock:
            if self.asr_model is None:
                return None, "ASR模型未加载"
            model = self.asr_model
            self.asr_in_use += 1
        try:
            try:
                segments, info = model.transcribe(audio_path, language=language, beam_size=beam_size, vad_filter=vad_filter, word_timestamps=word_timestamps, initial_prompt=initial_prompt)
            except Exception as e:
                if vad_filter and ("onnx" in str(e).lower() or "vad" in str(e).lower()):
                    print(f"VAD 失败，关闭 VAD 重试。错误: {e}")
                    segments, info = model.transcribe(audio_path, language=language, beam_size=beam_size, vad_filter=False, word_timestamps=word_timestamps, initial_prompt=initial_prompt)
                else:
                    return None, str(e)
            sentences = []
            all_words = []
            for seg in segments:
                s = {"start": seg.start, "end": seg.end, "text": seg.text.strip()}
                if seg.words:
                    words = [{"word": w.word, "start": w.start, "end": w.end} for w in seg.words]
                    s["words"] = words
                    all_words.extend(words)
                sentences.append(s)
            result = {"language": info.language, "language_probability": info.language_probability, "segments": sentences, "words": all_words}
            return result, None
        finally:
            with self.lock:
                self.asr_in_use = max(0, self.asr_in_use - 1)

    def apply_whisperx_align(self, result, audio_path, language, device, model_choice):
        """
        使用 whisperx.align 精细化单词时间戳。
        model_choice: 可以是 "auto" 或本地模型显示名。
        返回更新后的 result 或原 result（失败时）。
        """
        if not WHISPERX_ALIGN_AVAILABLE:
            self.last_align_error = "未安装 whisperx.align，无法精细对齐"
            return result
        self.last_align_error = ""
        self.align_in_use += 1
        try:
            # 决定对齐模型
            local_align = self.get_local_align_models()
            align_model_path = None
            if model_choice == "auto":
                # 根据检测到的语言自动选择
                detected = result.get("language", "en")
                # 修复：原实现用「语言全名」子串匹配目录名
                # （"chinese-zh-cn" in "wav2vec2-zh" -> False），
                # 导致 auto 永远匹配不到本地的 wav2vec2-zh / wav2vec2-en，
                # 直接掉到下面的在线回退分支，离线环境白等 166 秒后失败。
                # 现改用与 whisperX.py / _pro.py / _sub_align.py 一致的边界匹配。
                align_model_path = match_local_align_by_lang(detected, local_align)
                if align_model_path:
                    print(f"自动匹配到本地对齐模型: {Path(align_model_path).name}")
                if not align_model_path:
                    # 回退到在线模型名
                    online_map = {
                        "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
                        "en": "jonatasgrosman/wav2vec2-large-xlsr-53-english",
                        "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
                        "fr": "jonatasgrosman/wav2vec2-large-xlsr-53-french",
                        "de": "jonatasgrosman/wav2vec2-large-xlsr-53-german",
                        "es": "jonatasgrosman/wav2vec2-large-xlsr-53-spanish",
                        "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
                        "it": "jonatasgrosman/wav2vec2-large-xlsr-53-italian",
                        "nl": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
                        "hu": "jonatasgrosman/wav2vec2-large-xlsr-53-hungarian",
                    }
                    align_model_path = online_map.get(detected.lower(), None)
                    if not align_model_path:
                        print(f"未找到语言 {detected} 的自动对齐模型，跳过精细对齐。")
                        return result
            else:
                # 手动选择模型，优先在本地查找
                for disp, path in local_align:
                    if disp == model_choice:
                        align_model_path = path
                        break
                if not align_model_path:
                    align_model_path = model_choice  # 当作在线模型名

            # 加载对齐模型（按需缓存，按语言+模型路径）
            cache_key = f"{language}_{align_model_path}"
            if (self.align_model is None or self.align_model_lang != cache_key):
                print(f"加载对齐模型: {align_model_path}")
                self.align_model, self.align_metadata = load_align_model(
                    language_code=language or result.get("language", "en"),
                    device=device,
                    model_name=align_model_path,
                    model_dir=str(ROOT_DIR / "pretrained_models")
                )
                self.align_model_lang = cache_key

            # 执行对齐
            print("正在执行 wav2vec2 精细对齐...")
            aligned = whisperx_align(
                result["segments"],
                self.align_model,
                self.align_metadata,
                audio_path,
                device,
                return_char_alignments=False
            )
            # 更新 segments 和 words
            if "segments" in aligned:
                result["segments"] = aligned["segments"]
            new_words = []
            for seg in result["segments"]:
                if "words" in seg:
                    new_words.extend(seg["words"])
            result["words"] = new_words
            print("精细对齐完成。")
        except Exception as e:
            # 修复：原来只 print，界面照常显示「完成」，用户以为拿到了精细对齐的时间戳。
            # 现记录到实例属性，由调用方在结果里如实告知。
            print(f"精细对齐出错: {e}，将使用原始时间戳。")
            logger.warning(f"精细对齐出错: {e}，将使用原始时间戳。")
            self.last_align_error = str(e)
        finally:
            self.align_in_use = max(0, self.align_in_use - 1)
        return result

    def cleanup_temp(self):
        cleaned = 0
        for f in self.temp_files[:]:
            try:
                os.unlink(f)
                self.temp_files.remove(f)
                cleaned += 1
            except OSError: pass
        return cleaned

    def _prepare_audio(self, audio_input):
        """把输入规整成「可直接交给 faster-whisper 的音频路径」。

        现在音频页用的是 gr.Audio(type="filepath")，传进来的就是路径，
        第一个分支直接命中、原文件交给 faster-whisper 自行解码（它会统一重采样到
        16kHz 单声道），不再需要额外转码 —— 与 whisperX.py / _pro.py 的音频页一致。

        下面的 tuple 分支保留作为兜底：万一有调用方仍传入
        (采样率, ndarray) 形式的音频数据，仍能正常处理。
        """
        try:
            if isinstance(audio_input, str) and os.path.exists(audio_input):
                return audio_input
            if isinstance(audio_input, tuple) and len(audio_input)==2:
                sr, data = audio_input
                if data is None: return None
                if data.ndim > 1: data = np.mean(data, axis=1)
                if sr != 16000:
                    from scipy import signal
                    n_samples = int(len(data) * 16000 / sr)
                    data = signal.resample(data, n_samples)
                    sr = 16000
                temp_hash = hashlib.md5(data.tobytes() + str(time.time()).encode()).hexdigest()[:8]
                temp_path = os.path.join(tempfile.gettempdir(), f"whisperx_temp_{temp_hash}.wav")
                sf.write(temp_path, data, sr)
                self.temp_files.append(temp_path)
                return temp_path
            return None
        except Exception as e:
            print(f"音频转换失败: {e}")
            return None

manager = WhisperXManager()

# ==================== 核心函数 ====================
def ensure_model_loaded(model_size, device, compute_type, language):
    success, msg = manager.load_asr_model(model_size, device, compute_type, language)
    if not success: raise RuntimeError(msg)

def transcribe_audio(audio, model_size, device, compute_type, language, beam_size, vad_filter, hotwords, enable_align, align_model, progress=gr.Progress()):
    if audio is None: return "请上传或录制音频", "", ""
    progress(0, desc="初始化...")
    try: ensure_model_loaded(model_size, device, compute_type, language)
    except RuntimeError as e: return str(e), "", ""
    progress(0.3, desc="转写中...")
    audio_path = manager._prepare_audio(audio)
    if not audio_path: return "音频处理失败", "", ""
    try:
        prompt = hotwords.strip() if hotwords else None
        result, err = manager.transcribe(audio_path, language=language, beam_size=beam_size, vad_filter=vad_filter, word_timestamps=True, initial_prompt=prompt)
        if err: return f"错误: {err}", "", ""
        # 可选精细对齐
        if enable_align:
            progress(0.6, desc="精细对齐...")
            result = manager.apply_whisperx_align(result, audio_path, language, device, align_model)
        align_note = _align_fallback_note(manager.last_align_error) if enable_align else ""
        progress(0.7, desc="生成输出...")
        full_text, tsjson, srt_text, _ = format_result_to_outputs(result)
        # 音频页改为 type="filepath" 后，audio 就是「缓存目录/哈希/原文件名」这样的路径，
        # 这里能取到真正的源文件名，输出文件不再是千篇一律的 whisperx_<时间戳>。
        base = audio if isinstance(audio, str) and os.path.exists(audio) else None
        saved = save_outputs(base, full_text, tsjson, srt_text, language=result.get("language","未知"), model_info=model_size)
        save_info = "文件已保存:\n"
        if saved.get('txt'): save_info += f" {Path(saved['txt']).name}\n"
        if saved.get('json'): save_info += f" {Path(saved['json']).name}\n"
        if saved.get('srt'): save_info += f" {Path(saved['srt']).name}\n"
        if align_note:
            save_info += f"\n{align_note}\n"
        full_text = save_info + "\n" + full_text
        progress(1.0, desc="完成")
        return safe_text(full_text), safe_text(tsjson), safe_text(srt_text)
    finally:
        manager.cleanup_temp()

def transcribe_video(video, model_size, device, compute_type, language, beam_size, vad_filter, subtitle_mode, hotwords, enable_align, align_model, progress=gr.Progress()):
    temp_audio_path = None
    temp_srt_path = None
    try:
        if video is None: return "请上传视频文件", "", ""
        progress(0, desc="初始化...")
        ensure_model_loaded(model_size, device, compute_type, language)
        progress(0.2, desc="提取音频...")
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_audio.close()
        audio_path = temp_audio.name
        temp_audio_path = audio_path
        cmd = [FFMPEG_PATH, "-i", video, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-y", audio_path]
        run_ffmpeg(cmd)
        progress(0.4, desc="转写中...")
        prompt = hotwords.strip() if hotwords else None
        result, err = manager.transcribe(audio_path, language=language, beam_size=beam_size, vad_filter=vad_filter, word_timestamps=True, initial_prompt=prompt)
        if err: return f"识别失败: {err}", "", ""
        if enable_align:
            progress(0.6, desc="精细对齐...")
            result = manager.apply_whisperx_align(result, audio_path, language, device, align_model)
        align_note = _align_fallback_note(manager.last_align_error) if enable_align else ""
        progress(0.7, desc="生成字幕...")
        full_text, tsjson, srt_text, _ = format_result_to_outputs(result)
        base = video if isinstance(video, str) and os.path.exists(video) else None
        saved = save_outputs(base, full_text, tsjson, srt_text, language=result.get("language","未知"), model_info=model_size)
        srt_path = saved.get('srt')
        if not srt_path: return "处理完成，未生成字幕。", "", ""
        progress(0.8, desc="嵌入字幕...")
        ts = time.strftime("%Y%m%d_%H%M%S")
        prefix = generate_output_filename(video, ts, subtitle_mode, "video")
        out_path = OUTPUT_DIR / f"{prefix}.mp4"
        vid_str = str(video).replace('\\','/')
        out_str = str(out_path).replace('\\','/')
        if subtitle_mode == "soft":
            srt_str = str(srt_path).replace('\\','/')
            cmd = [FFMPEG_PATH, "-i", vid_str, "-i", srt_str, "-c", "copy", "-c:s", "mov_text", "-metadata:s:s:0", "language=chi", "-y", out_str]
        else:
            # 修复：ffmpeg 的 subtitles 滤镜要求路径里的冒号转义成 \:，
            # 原写法直接把 "D:/..." 塞进去，滤镜会把它当成 original_size 参数解析，
            # 报 "Unable to parse option value ... as image size"，硬字幕永远生成不出来
            # （实测：转义后正常产出 mp4，未转义不产出任何文件）。
            # 做法与 whisperX.py / whisperX_pro.py 保持一致：
            # 先复制到无中文的临时路径，再转义 : 和 '。
            temp_srt_path = str(CACHE_DIR / f"sub_{uuid.uuid4().hex[:8]}.srt")
            shutil.copy2(str(srt_path), temp_srt_path)
            escaped_srt = temp_srt_path.replace('\\', '/').replace(':', '\\:').replace("'", "\\'")
            font_name = "Microsoft YaHei" if sys.platform == "win32" else "Arial"
            vf_str = (f"subtitles='{escaped_srt}':force_style='FontName={font_name},"
                      f"FontSize=24,PrimaryColour=&H00FFFFFF,OutlineColour=&H000000,BorderStyle=3'")
            cmd = [FFMPEG_PATH, "-i", vid_str, "-vf", vf_str, "-c:a", "copy", "-y", out_str]
        run_ffmpeg(cmd)
        result_msg = f"✅ 处理完成！输出视频: {out_path.name}\n字幕文件已保存至 output 目录。"
        if align_note:
            result_msg += f"\n\n{align_note}"
        progress(1.0, desc="完成")
        return safe_text(result_msg), "", ""
    except Exception as e:
        logger.error(traceback.format_exc())
        return f"处理视频失败: {str(e)}", "", ""
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try: os.unlink(temp_audio_path)
            except Exception: pass
        if temp_srt_path and os.path.exists(temp_srt_path):
            try: os.unlink(temp_srt_path)
            except Exception: pass
        manager.cleanup_temp()

def transcribe_batch(files, model_size, device, compute_type, language, beam_size, vad_filter, hotwords, enable_align, align_model, progress=gr.Progress()):
    """批量转写。

    修复：原实现遇到失败就 `continue` 静默跳过，最后无条件返回
    「✅ 批量处理完成，共 N 个文件」，N 还是「输入文件数」而不是「成功数」。
    用户看到成功提示，去 output 目录却找不到文件。
    现改为逐文件记录结果，并在末尾如实汇报成功/失败明细。
    """
    if not files: return "请选择音频文件"
    try: ensure_model_loaded(model_size, device, compute_type, language)
    except RuntimeError as e: return str(e)
    total = len(files)
    ok_list, fail_list = [], []
    for i, fobj in enumerate(files, 1):
        fp = fobj.name if hasattr(fobj, 'name') else str(fobj)
        name = os.path.basename(fp)
        progress((i - 1) / total, desc=f"处理 {i}/{total}: {name}")
        ap = manager._prepare_audio(fp)
        if not ap:
            fail_list.append(f"{name}：音频预处理失败（文件损坏或 ffmpeg 不可用）")
            continue
        try:
            prompt = hotwords.strip() if hotwords else None
            result, err = manager.transcribe(ap, language=language, beam_size=beam_size, vad_filter=vad_filter, word_timestamps=True, initial_prompt=prompt)
            if err:
                fail_list.append(f"{name}：转写失败 - {err}")
                continue
            if enable_align:
                result = manager.apply_whisperx_align(result, ap, language, device, align_model)
            full_text, tsjson, srt_text, _ = format_result_to_outputs(result)
            save_outputs(fp, full_text, tsjson, srt_text, language=result.get("language","未知"), model_info=model_size)
            if enable_align and manager.last_align_error:
                ok_list.append(f"{name}（精细对齐未生效，已用 ASR 原始时间戳）")
            else:
                ok_list.append(name)
        except Exception as e:
            logger.error(traceback.format_exc())
            fail_list.append(f"{name}：{e}")
        finally:
            manager.cleanup_temp()

    progress(1.0, desc="完成")
    if fail_list and not ok_list:
        head = f"❌ 批量处理失败：{total} 个文件全部未成功。"
    elif fail_list:
        head = f"⚠️ 批量处理完成：成功 {len(ok_list)} / {total} 个，失败 {len(fail_list)} 个。"
    else:
        head = f"✅ 批量处理完成，共 {total} 个文件。"
    lines = [head]
    if ok_list:
        lines.append("\n成功：" + "、".join(ok_list))
    if fail_list:
        lines.append("\n失败明细：")
        lines.extend(f"  • {x}" for x in fail_list)
    lines.append("\n详细结果请查看 output 目录。")
    return "\n".join(lines)

def load_model_click(model_size, device, compute_type, language):
    success, msg = manager.load_asr_model(model_size, device, compute_type, language)
    # 修复：原来返回 (msg, get_system_info())，而 UI 里 outputs 写成了
    # [status_display, status_display]（同一个组件出现两次），后写的会覆盖前写的，
    # 导致「加载失败」的真实原因（如 CPU 不支持 float16）永远不显示。
    # 现合并成一个字符串、只绑定一个输出组件。
    return f"{msg}\n\n{get_system_info()}"

def unload_model_click():
    # 修复：unload_models 现在会在转写/对齐进行中拒绝卸载并给出原因，
    # 这里要把结果显示出来，而不是无条件报「模型已卸载」
    ok, msg = manager.unload_models()
    return f"{msg}\n\n{get_system_info()}"

def refresh_status(): return get_system_info()

def health_check():
    info = get_system_info()
    with manager.lock:
        if manager.asr_model is None: info += "\n\n[警告] ASR模型未加载，请先加载模型。"
        else: info += "\n\n[信息] 系统已就绪。"
    return info

def toggle_align_controls(enable_align):
    """联动：勾选精细对齐时显示模型选择下拉框"""
    return gr.update(visible=enable_align and WHISPERX_ALIGN_AVAILABLE)

# ==================== 界面 ====================
def create_interface():
    settings = manager.settings
    default_output_dir = settings.get("output_dir", str(DEFAULT_OUTPUT_DIR))
    global OUTPUT_DIR
    with config_lock:
        OUTPUT_DIR = Path(default_output_dir)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    local_models = manager.get_available_local_models()
    model_choices = [disp for disp, _ in local_models]
    if not model_choices: model_choices = ["tiny","base","small","medium","large-v2","large-v3","large-v3-turbo"]
    # 修复：原写法 ["cuda" if torch.cuda.is_available() else "cpu", "cpu"]
    # 在无 CUDA 的机器上会得到 ["cpu", "cpu"]，下拉框出现两个一样的选项。
    device_choices = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
    # 修复：原写法硬编码 ["int8_float32","float16","float32"]，无条件提供 float16。
    # 实测 GTX 1080（算力 6.1）在 ctranslate2 4.4.0 下 CUDA 不支持 float16，
    # 用户选了必然加载失败。现按该设备实际支持的精度构造。
    compute_choices = supported_compute_types(device_choices[0])
    print(f"[OK] {device_choices[0]} 支持的精度: {compute_choices}")

    # 对齐模型列表
    align_local = manager.get_local_align_models()
    align_options = ["auto"] + [name for name, _ in align_local]

    with gr.Blocks(title="WhisperX 语音识别（增强版）", theme=gr.themes.Default()) as demo:
        gr.Markdown("# 🎤 WhisperX 语音识别（时间戳精细对齐可选）\n输出目录: `{}`".format(OUTPUT_DIR))
        with gr.Accordion("系统状态", open=False):
            with gr.Row():
                status_display = gr.Textbox(label="系统状态", value=get_system_info(), lines=6, interactive=False, scale=4)
                with gr.Column(scale=1):
                    refresh_btn = gr.Button("刷新", variant="secondary")
                    health_btn = gr.Button("健康检查", variant="secondary")
            health_btn.click(health_check, outputs=[status_display])
        with gr.Row():
            device = gr.Dropdown(label="设备", choices=device_choices, value=device_choices[0])
            model_size = gr.Dropdown(label="模型大小", choices=model_choices, value=model_choices[0] if model_choices else "medium")
            compute_type = gr.Dropdown(label="计算类型", choices=compute_choices, value=compute_choices[0])
            language = gr.Textbox(label="语言代码", value="zh", placeholder="zh/en/ja...")
        with gr.Row():
            load_btn = gr.Button("加载模型", variant="primary")
            unload_btn = gr.Button("卸载模型", variant="stop")
            beam_size = gr.Slider(label="Beam Size", minimum=1, maximum=10, value=5, step=1)
            vad_filter = gr.Checkbox(label="启用 VAD 过滤", value=False, info="若 onnxruntime 不可用请关闭")
        with gr.Row():
            enable_align = gr.Checkbox(label="使用 wav2vec2 精细对齐（提升时间戳准确度）", value=False, interactive=WHISPERX_ALIGN_AVAILABLE)
            align_model_dropdown = gr.Dropdown(
                label="对齐模型选择", choices=align_options, value="auto", visible=False,
                info="auto: 根据语言自动选择；或手动指定本地/在线模型"
            )
            if not WHISPERX_ALIGN_AVAILABLE:
                gr.Markdown("⚠️ **whisperx.align 不可用，精细对齐功能已禁用。如需使用，请安装 whisperx 及依赖。**")

        # 联动：勾选时显示模型选择下拉框
        enable_align.change(
            toggle_align_controls,
            inputs=[enable_align],
            outputs=[align_model_dropdown]
        )

        load_btn.click(load_model_click, inputs=[model_size, device, compute_type, language], outputs=[status_display])
        unload_btn.click(unload_model_click, outputs=[status_display])
        refresh_btn.click(refresh_status, outputs=[status_display])
        gr.Markdown("---")
        with gr.Tabs():
            with gr.Tab("音频识别"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # 修复：原来用 type="numpy"，Gradio 把音频解码成
                        # (采样率, ndarray) 元组传进来，**原始文件名丢失**，
                        # 导致输出文件恒为 whisperx_<时间戳>.txt，无法与源文件对应。
                        # 改为 type="filepath"：Gradio 会把上传文件存到缓存目录并
                        # **保留原文件名**（processing_utils.save_file_to_cache 里
                        # 用 Path(file_path).name 命名），于是输出名能带上源文件名。
                        # 本项目另外两个脚本（whisperX.py / whisperX_pro.py）的音频页
                        # 一直用的就是 gr.File 传路径，这条路径是验证过的。
                        # 麦克风录音会拿到 Gradio 的临时文件名，属可接受的次要场景。
                        audio_input = gr.Audio(label="选择或录制音频", type="filepath",
                                               sources=["upload", "microphone"])
                        hotwords_audio = gr.Textbox(label="热词/提示词", lines=2, value="")
                        with gr.Row():
                            t_btn = gr.Button("开始识别", variant="primary")
                            c_btn = gr.Button("清空", variant="secondary")
                    with gr.Column(scale=2):
                        with gr.Tabs():
                            with gr.Tab("识别文本"): text_out = gr.Textbox(label="结果", lines=8, show_copy_button=False)
                            with gr.Tab("时间戳"): json_out = gr.Textbox(label="JSON", lines=8, show_copy_button=False)
                            with gr.Tab("SRT"): srt_out = gr.Textbox(label="SRT字幕", lines=8, show_copy_button=False)
                t_btn.click(
                    transcribe_audio,
                    inputs=[audio_input, model_size, device, compute_type, language, beam_size, vad_filter, hotwords_audio, enable_align, align_model_dropdown],
                    outputs=[text_out, json_out, srt_out]
                ).then(refresh_status, outputs=[status_display])
                c_btn.click(lambda: [None, "", "", "", ""], outputs=[audio_input, hotwords_audio, text_out, json_out, srt_out])
            with gr.Tab("视频字幕"):
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input = gr.Video(label="选择视频", sources=["upload"])
                        sub_mode = gr.Radio(label="嵌入模式", choices=["soft","hard"], value="soft")
                        hotwords_video = gr.Textbox(label="热词/提示词", lines=2, value="")
                        with gr.Row():
                            vt_btn = gr.Button("开始处理", variant="primary")
                            vc_btn = gr.Button("清空", variant="secondary")
                    with gr.Column(scale=2):
                        with gr.Tabs():
                            with gr.Tab("识别文本"): v_text = gr.Textbox(label="结果", lines=8)
                            with gr.Tab("时间戳"): v_json = gr.Textbox(label="JSON", lines=8)
                            with gr.Tab("SRT"): v_srt = gr.Textbox(label="SRT", lines=8)
                vt_btn.click(
                    transcribe_video,
                    inputs=[video_input, model_size, device, compute_type, language, beam_size, vad_filter, sub_mode, hotwords_video, enable_align, align_model_dropdown],
                    outputs=[v_text, v_json, v_srt]
                ).then(refresh_status, outputs=[status_display])
                vc_btn.click(lambda: [None, "", "", "", ""], outputs=[video_input, hotwords_video, v_text, v_json, v_srt])
            with gr.Tab("批量处理"):
                with gr.Row():
                    with gr.Column(scale=1):
                        files_input = gr.Files(label="上传多个音频", file_types=[".wav",".mp3",".m4a",".flac",".ogg"], file_count="multiple")
                        hotwords_batch = gr.Textbox(label="热词/提示词", lines=2, value="")
                        with gr.Row():
                            bt_btn = gr.Button("批量识别", variant="primary")
                            bc_btn = gr.Button("清空", variant="secondary")
                    with gr.Column(scale=2):
                        batch_out = gr.Textbox(label="结果", lines=8)
                bt_btn.click(
                    transcribe_batch,
                    inputs=[files_input, model_size, device, compute_type, language, beam_size, vad_filter, hotwords_batch, enable_align, align_model_dropdown],
                    outputs=[batch_out]
                ).then(refresh_status, outputs=[status_display])
                bc_btn.click(lambda: [None, "", ""], outputs=[files_input, hotwords_batch, batch_out])
        gr.Markdown("---")
        gr.HTML("<div style='text-align:center;color:#666;'>© 2026 光影的故事2018</div>")
        demo.load(refresh_status, outputs=[status_display])
    return demo

@atexit.register
def cleanup():
    manager.unload_models()
    manager.cleanup_temp()
    clean_old_logs()

def main():
    demo = create_interface()
    for port in [18006,18007,18008,18009,18010]:
        try:
            demo.queue().launch(server_name="127.0.0.1", server_port=port, inbrowser=True, show_error=True)
            break
        except OSError:
            print(f"端口 {port} 被占用，尝试下一个...")
            continue
    else:
        print("所有端口均被占用，请手动指定空闲端口。")

if __name__ == "__main__":
    main()