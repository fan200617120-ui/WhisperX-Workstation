#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WhisperX WebUI 全功能版 - 自动扫描本地模型，支持多语种强制对齐
包含：音频识别、视频字幕、批量处理、强制对齐（WhisperX精细对齐+回退算法）
新增：热词/提示词功能（initial_prompt），多语种对齐模型选择
Copyright 2026 光影的故事2018
"""

import sys
import os
import json
import logging
import traceback
import time
import gc
import threading
import atexit
import tempfile
import hashlib
import re
import base64
import subprocess
import shutil
import html
from pathlib import Path
from datetime import timedelta
from typing import List, Dict, Optional, Tuple

sys.setrecursionlimit(10000)

# ==================== 日志设置 ====================
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

def clean_old_logs(days=7):
    cutoff = time.time() - days * 24 * 3600
    for f in LOG_DIR.glob("error_*.log"):
        if f.stat().st_mtime < cutoff:
            try:
                f.unlink()
            except:
                pass

clean_old_logs()
log_file = LOG_DIR / f"error_{time.strftime('%Y%m%d')}.log"
logging.basicConfig(filename=log_file, level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# ==================== 路径设置 ====================
CURRENT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
BASE_DIR = Path(__file__).parent.absolute()
ROOT_DIR = BASE_DIR.parent
DEFAULT_OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
ALIGN_OUTPUT_DIR = OUTPUT_DIR / "字幕自动打轴"
ALIGN_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
PRESET_DIR = ROOT_DIR / "preset"
PRESET_DIR.mkdir(exist_ok=True)
CONFIG_FILE = PRESET_DIR / "settings.json"
config_lock = threading.RLock()

# ==================== FFmpeg (跨平台) ====================
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
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_settings(settings):
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
    except:
        pass

# ==================== 导入依赖 ====================
try:
    import gradio as gr
    import torch
    import numpy as np
    import librosa
    import soundfile as sf
    from faster_whisper import WhisperModel
    print(f"PyTorch: {torch.__version__}, CUDA可用: {torch.cuda.is_available()}")
    print("faster-whisper: 已导入")
except ImportError as e:
    print(f"基础依赖缺失: {e}")
    sys.exit(1)

# 尝试导入 whisperx 对齐模块
try:
    from whisperx import load_align_model, align
    WHISPERX_ALIGN_AVAILABLE = True
    print("whisperx.align: 已导入")
except ImportError:
    print("警告: 未找到 whisperx.align 模块，多语种精细对齐功能将不可用。")
    print("请运行: pip install whisperx")
    WHISPERX_ALIGN_AVAILABLE = False

# ==================== 工具函数 ====================
def seconds_to_srt_time(seconds):
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    ms = int((td.total_seconds() - total_seconds) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"

def words_to_srt(words_with_time):
    lines = []
    for i, w in enumerate(words_with_time, 1):
        lines.append(str(i))
        lines.append(f"{seconds_to_srt_time(w['start'])} --> {seconds_to_srt_time(w['end'])}")
        lines.append(w["word"])
        lines.append("")
    return "\n".join(lines)

def sentences_to_srt(sentences):
    lines = []
    for i, s in enumerate(sentences, 1):
        lines.append(str(i))
        lines.append(f"{seconds_to_srt_time(s['start'])} --> {seconds_to_srt_time(s['end'])}")
        lines.append(s["text"])
        lines.append("")
    return "\n".join(lines)

def force_align_char_level(reference_text, transcribed_words, audio_duration=None):
    """字符级对齐（中文等）"""
    if not transcribed_words:
        return []
    ref_chars = [ch for ch in reference_text if not ch.isspace()]
    hyp_chars = []
    char_to_word_idx = []
    for w_idx, w in enumerate(transcribed_words):
        for ch in w["word"]:
            if not ch.isspace():
                hyp_chars.append(ch)
                char_to_word_idx.append(w_idx)
    if not char_to_word_idx:
        default_end = audio_duration if audio_duration else 1.0
        return [{"word": ch, "start": 0.0, "end": default_end} for ch in ref_chars]
    hyp_idx = 0
    match_map = []
    for r_char in ref_chars:
        found = False
        for offset in range(20):
            check_idx = hyp_idx + offset
            if check_idx < len(hyp_chars) and hyp_chars[check_idx] == r_char:
                match_map.append(check_idx)
                hyp_idx = check_idx + 1
                found = True
                break
        if not found:
            match_map.append(hyp_idx - 1 if hyp_idx > 0 else 0)
    aligned = []
    for i, r_char in enumerate(ref_chars):
        matched_hyp_idx = match_map[i]
        if matched_hyp_idx < len(char_to_word_idx):
            w_idx = char_to_word_idx[matched_hyp_idx]
        else:
            w_idx = char_to_word_idx[-1]
        start_t = transcribed_words[w_idx]["start"]
        end_t = transcribed_words[w_idx]["end"]
        count_in_word = 1
        total_in_word = 1
        for j in range(i - 1, -1, -1):
            if match_map[j] < len(char_to_word_idx) and char_to_word_idx[match_map[j]] == w_idx:
                count_in_word += 1
                total_in_word += 1
            else:
                break
        for j in range(i + 1, len(ref_chars)):
            if match_map[j] < len(char_to_word_idx) and char_to_word_idx[match_map[j]] == w_idx:
                total_in_word += 1
            else:
                break
        word_duration = end_t - start_t
        char_duration = word_duration / total_in_word if total_in_word > 0 else 0
        char_start = start_t + (count_in_word - 1) * char_duration
        char_end = char_start + char_duration
        aligned.append({"word": r_char, "start": char_start, "end": char_end})
    return aligned

def force_align_word_level(reference_text, transcribed_words, audio_duration=None):
    """单词级对齐（英文等）"""
    if not transcribed_words:
        return []
    ref_words = reference_text.split()
    hyp_words = [w["word"] for w in transcribed_words]
    aligned = []
    hyp_idx = 0
    for ref_w in ref_words:
        found = False
        for offset in range(20):
            if hyp_idx + offset >= len(hyp_words):
                break
            hyp_w = hyp_words[hyp_idx + offset]
            ref_clean = re.sub(r'[^\w]', '', ref_w.lower())
            hyp_clean = re.sub(r'[^\w]', '', hyp_w.lower())
            if ref_clean == hyp_clean or ref_clean in hyp_clean or hyp_clean in ref_clean:
                w_idx = hyp_idx + offset
                aligned.append({
                    "word": ref_w,
                    "start": transcribed_words[w_idx]["start"],
                    "end": transcribed_words[w_idx]["end"]
                })
                hyp_idx = w_idx + 1
                found = True
                break
        if not found:
            if hyp_idx < len(transcribed_words):
                start = transcribed_words[hyp_idx]["start"]
                end = transcribed_words[hyp_idx]["end"]
            else:
                start = aligned[-1]["end"] if aligned else 0.0
                end = start + 0.3
            aligned.append({"word": ref_w, "start": start, "end": end})
    return aligned

def _generate_merged_srt(aligned_chars, sentences, paragraphs,
                         merge_punctuations, merge_max_words, merge_max_chars,
                         merge_max_duration, merge_by_newline,
                         merge_by_punc, merge_by_silence, merge_by_wordcount,
                         merge_by_charcount, merge_by_duration, silence_threshold):
    """根据合并策略生成合并字幕"""
    if merge_by_newline:
        return sentences_to_srt(sentences)
    punc_set = set(merge_punctuations) if merge_punctuations else set()
    merged_segments = []
    current_chars = []
    current_start = None
    for i, ch_info in enumerate(aligned_chars):
        if current_start is None:
            current_start = ch_info["start"]
        current_chars.append(ch_info)
        should_split = False
        if merge_by_punc and punc_set and ch_info["word"] in punc_set:
            should_split = True
        if merge_by_silence and i < len(aligned_chars) - 1:
            gap = aligned_chars[i+1]["start"] - ch_info["end"]
            if gap > silence_threshold:
                should_split = True
        text_so_far = "".join([c["word"] for c in current_chars])
        if merge_by_wordcount and len(current_chars) >= merge_max_words:
            should_split = True
        if merge_by_charcount and len(text_so_far) >= merge_max_chars:
            should_split = True
        duration = ch_info["end"] - current_start
        if merge_by_duration and duration >= merge_max_duration:
            should_split = True
        if should_split:
            merged_segments.append({
                "start": current_start,
                "end": ch_info["end"],
                "text": text_so_far.strip()
            })
            current_chars = []
            current_start = None
    if current_chars:
        text = "".join([c["word"] for c in current_chars]).strip()
        if text:
            merged_segments.append({
                "start": current_chars[0]["start"],
                "end": current_chars[-1]["end"],
                "text": text
            })
    return sentences_to_srt(merged_segments)

def format_result_to_outputs(result):
    if not result or not isinstance(result, dict):
        return "无结果", "{}", "", []
    text = " ".join([seg["text"] for seg in result.get("segments", [])])
    segments = result.get("segments", [])
    timestamps_json = json.dumps(segments, ensure_ascii=False, indent=2)
    srt_lines = []
    for i, seg in enumerate(segments, 1):
        start = seconds_to_srt_time(seg["start"])
        end = seconds_to_srt_time(seg["end"])
        srt_lines.append(str(i))
        srt_lines.append(f"{start} --> {end}")
        srt_lines.append(seg["text"])
        srt_lines.append("")
    srt_text = "\n".join(srt_lines)
    extra = f"语言: {result.get('language', '未知')} (概率: {result.get('language_probability', 0):.2f})"
    full_text = f"{text}\n\n[元数据] {extra}"
    return full_text, timestamps_json, srt_text, segments

def generate_subtitle_html(segments, audio_path, max_size_mb=5):
    if not audio_path or not os.path.exists(audio_path) or not segments:
        return '<div style="padding:20px; text-align:center; color:#999;">暂无字幕预览</div>'
    try:
        size = os.path.getsize(audio_path) / (1024 * 1024)
        if size > max_size_mb:
            return (
                '<div style="padding:20px; text-align:center; color:#666;">'
                f'[提示] 音频文件过大 ({size:.1f} MB)，超过 {max_size_mb} MB 预览限制。</div>'
            )
    except:
        pass
    mime_map = {
        '.wav': 'audio/wav', '.mp3': 'audio/mpeg', '.m4a': 'audio/mp4',
        '.flac': 'audio/flac', '.ogg': 'audio/ogg', '.aac': 'audio/aac'
    }
    ext = os.path.splitext(audio_path)[1].lower()
    mime_type = mime_map.get(ext, 'audio/wav')
    try:
        with open(audio_path, "rb") as f:
            audio_data = f.read()
        b64_data = base64.b64encode(audio_data).decode('utf-8')
    except Exception as e:
        return f'<div style="color:red;">音频加载失败: {e}</div>'

    audio_html = (
        '<div style="margin-bottom:15px; padding:15px; background:#f8f9fa; border-radius:8px; border:1px solid #ddd;">'
        '<div style="margin-bottom:8px; font-weight:bold; color:#333;">音频回放</div>'
        '<audio id="preview-audio" controls style="width:100%;">'
        f'<source src="data:{mime_type};base64,{b64_data}" type="{mime_type}">'
        '</audio></div>'
    )

    cues_parts = [
        '<div id="subtitle-container" style="height:500px; overflow-y:auto; border:1px solid #ccc; '
        'padding:10px; border-radius:5px; background-color:#fff;">'
    ]
    for i, seg in enumerate(segments):
        safe_text = html.escape(seg["text"])
        cues_parts.append(
            f'<div class="subtitle-row" data-index="{i}" data-start="{seg["start"]}" data-end="{seg["end"]}" '
            f'style="padding:10px; margin:5px 0; border-radius:5px; border-bottom:1px solid #eee; '
            f'cursor:pointer; display:flex; align-items:flex-start;">'
            f'<span style="color:#666; font-size:0.85em; margin-right:15px; font-family:monospace; '
            f'background:#eee; padding:2px 6px; border-radius:4px; min-width:60px; text-align:center;">'
            f'{seg["start"]:.2f}s</span>'
            f'<span class="text-content" style="font-size:1.1rem; color:#333; line-height:1.5;">{safe_text}</span>'
            f'</div>'
        )
    cues_parts.append('</div>')
    cues_html = '\n'.join(cues_parts)

    js_script = """
<script>
(function() {
    let audio = document.getElementById("preview-audio");
    let container = document.getElementById("subtitle-container");
    if (!audio || !container) return;
    let rows = container.querySelectorAll(".subtitle-row");
    let clickBound = false;
    function updateHighlight() {
        let currentTime = audio.currentTime;
        rows.forEach(row => {
            let start = parseFloat(row.getAttribute("data-start"));
            let end = parseFloat(row.getAttribute("data-end"));
            if (currentTime >= start && currentTime < end) {
                if (!row.classList.contains("active")) {
                    row.classList.add("active");
                    row.style.backgroundColor = "#e3f2fd";
                    row.style.borderLeft = "5px solid #2196f3";
                    row.style.fontWeight = "bold";
                    row.style.transform = "scale(1.01)";
                    row.style.boxShadow = "0 2px 5px rgba(0,0,0,0.1)";
                    row.scrollIntoView({ behavior: "smooth", block: "center" });
                }
            } else {
                if (row.classList.contains("active")) {
                    row.classList.remove("active");
                    row.style.backgroundColor = "";
                    row.style.borderLeft = "";
                    row.style.fontWeight = "";
                    row.style.transform = "";
                    row.style.boxShadow = "";
                }
            }
        });
    }
    if (!clickBound) {
        rows.forEach(row => {
            row.onclick = function() {
                let start = parseFloat(row.getAttribute("data-start"));
                audio.currentTime = start;
                audio.play();
            };
        });
        clickBound = true;
    }
    audio.addEventListener("timeupdate", updateHighlight);
    setTimeout(updateHighlight, 100);
})();
</script>
"""
    return audio_html + cues_html + js_script

def save_outputs(base_name, full_text, timestamps_json, srt_text, language, model_info):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if base_name:
        safe = re.sub(r'[^\w\u4e00-\u9fff\-\.]', '', Path(base_name).stem)
        prefix = f"{safe}_{timestamp}"
    else:
        prefix = f"whisperx_{timestamp}"
    saved = {}
    txt_path = OUTPUT_DIR / f"{prefix}.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(full_text)
    saved['txt'] = str(txt_path)
    if timestamps_json and timestamps_json != "{}":
        json_path = OUTPUT_DIR / f"{prefix}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(timestamps_json)
        saved['json'] = str(json_path)
    if srt_text.strip():
        srt_path = OUTPUT_DIR / f"{prefix}.srt"
        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write(srt_text)
        saved['srt'] = str(srt_path)
    return saved

def get_system_info(align_model_info: str = ""):
    info = []
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        allocated = torch.cuda.memory_allocated(0) / 1e9
        info.append(f"显卡: {gpu_name} ({total:.1f} GB)")
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
    info.append(f"打轴输出: {ALIGN_OUTPUT_DIR}")
    if align_model_info:
        info.append(f"对齐模型: {align_model_info}")
    return "\n".join(info)

def generate_output_filename(base_input, timestamp_str, custom_suffix="", default_name="recording"):
    original_name = None
    if isinstance(base_input, str) and os.path.exists(base_input):
        original_name = Path(base_input).stem
    elif isinstance(base_input, dict) and base_input.get('path') and os.path.exists(base_input['path']):
        original_name = Path(base_input['path']).stem
    elif isinstance(base_input, tuple):
        original_name = default_name
    if not original_name:
        original_name = default_name
    safe_name = re.sub(r'[^\w\u4e00-\u9fff\-]', '', original_name)
    if not safe_name:
        safe_name = default_name
    parts = [safe_name, timestamp_str]
    if custom_suffix:
        parts.append(custom_suffix)
    return "_".join(parts)

# ==================== 语言到对齐模型的映射表 ====================
LANGUAGE_ALIGN_MODEL_MAP = {
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "en": "jonatasgrosman/wav2vec2-large-xlsr-53-english",
    "fr": "jonatasgrosman/wav2vec2-large-xlsr-53-french",
    "de": "jonatasgrosman/wav2vec2-large-xlsr-53-german",
    "it": "jonatasgrosman/wav2vec2-large-xlsr-53-italian",
    "es": "jonatasgrosman/wav2vec2-large-xlsr-53-spanish",
    "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "nl": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
    "hu": "jonatasgrosman/wav2vec2-large-xlsr-53-hungarian",
}

def get_align_model_from_language(language: str, local_models: List[Tuple[str, str]]) -> Tuple[Optional[str], bool]:
    if not language:
        return None, False
    lang = language.strip().lower()
    if lang in LANGUAGE_ALIGN_MODEL_MAP:
        online_id = LANGUAGE_ALIGN_MODEL_MAP[lang]
        for display, path in local_models:
            if online_id.split("/")[-1].replace("-", "") in display.replace("-", "").replace("_", "").lower():
                return path, True
        return online_id, False
    return None, False

# ==================== 模型管理器 ====================
class WhisperXManager:
    def __init__(self):
        self.asr_model = None
        self.align_model = None
        self.align_metadata = None
        self.current_asr_model_name = None
        self.current_device = None
        self.current_compute_type = None
        self.settings = load_settings()
        self.temp_files = []
        self.lock = threading.RLock()

    def get_available_local_models(self):
        """扫描 faster-whisper 模型"""
        models = []
        models_dir = ROOT_DIR / "pretrained_models"
        if not models_dir.exists():
            return []
        for item in models_dir.iterdir():
            if item.is_dir() and "faster-whisper" in item.name.lower():
                if (item / "model.bin").exists() or (item / "config.json").exists() or (item / "pytorch_model.bin").exists():
                    match = re.search(r'faster-whisper-(\w+(?:-\w+)?)', item.name.lower())
                    if match:
                        display_name = match.group(1)
                    else:
                        display_name = item.name
                    models.append((display_name, str(item)))
        return models

    def get_local_align_models(self):
        """扫描本地 wav2vec2 对齐模型"""
        models = []
        models_dir = ROOT_DIR / "pretrained_models"
        if not models_dir.exists():
            return []
        for item in models_dir.iterdir():
            if not item.is_dir():
                continue
            name_lower = item.name.lower()
            if "wav2vec2" in name_lower or "xlsr" in name_lower:
                if (item / "pytorch_model.bin").exists() or (item / "model.bin").exists() or (item / "config.json").exists():
                    models.append((item.name, str(item)))
        return models

    def load_asr_model(self, model_size, device, compute_type, language=None):
        with self.lock:
            if (self.asr_model is not None and
                self.current_asr_model_name == model_size and
                self.current_device == device and
                self.current_compute_type == compute_type):
                return True, f"ASR模型已加载: {model_size}"

            local_models = self.get_available_local_models()
            local_path = None
            for display_name, path in local_models:
                if display_name == model_size:
                    local_path = path
                    break

            if local_path and os.path.exists(local_path):
                model_name_or_path = local_path
                local_files_only = True
                print(f"使用本地模型: {model_name_or_path}")
            else:
                std_local_path = ROOT_DIR / "pretrained_models" / model_size
                if std_local_path.exists() and (std_local_path / "model.bin").exists():
                    model_name_or_path = str(std_local_path)
                    local_files_only = True
                    print(f"使用本地模型: {model_name_or_path}")
                else:
                    model_name_or_path = model_size
                    local_files_only = False
                    print(f"将从 HuggingFace 下载模型: {model_name_or_path}")

            self.unload_models()
            try:
                self.asr_model = WhisperModel(
                    model_name_or_path,
                    device=device,
                    compute_type=compute_type,
                    local_files_only=local_files_only
                )
                self.current_asr_model_name = model_name_or_path
                self.current_device = device
                self.current_compute_type = compute_type
                return True, f"ASR模型加载成功: {model_size}"
            except Exception as e:
                logging.error(traceback.format_exc())
                return False, f"加载ASR模型失败: {str(e)}"

    def unload_models(self):
        with self.lock:
            if self.asr_model:
                del self.asr_model
                self.asr_model = None
            self.unload_align_model()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def load_align_model(self, language_code: str, device: str, model_name: str = None, model_dir: str = None):
        """加载 wav2vec2 对齐模型"""
        if not WHISPERX_ALIGN_AVAILABLE:
            raise RuntimeError("whisperx.align 模块不可用")
        self.unload_align_model()
        self.align_model, self.align_metadata = load_align_model(
            language_code=language_code,
            device=device,
            model_name=model_name,
            model_dir=model_dir
        )
        return self.align_model, self.align_metadata

    def unload_align_model(self):
        if self.align_model is not None:
            del self.align_model
            self.align_model = None
            self.align_metadata = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def transcribe(self, audio_path, language=None, beam_size=5, vad_filter=True, word_timestamps=True, initial_prompt=None):
        with self.lock:
            if self.asr_model is None:
                return None, "ASR模型未加载"
        try:
            segments, info = self.asr_model.transcribe(
                audio_path, language=language, beam_size=beam_size,
                vad_filter=vad_filter, word_timestamps=word_timestamps,
                initial_prompt=initial_prompt
            )
            sentences = []
            all_words = []
            for seg in segments:
                sentence = {"start": seg.start, "end": seg.end, "text": seg.text.strip()}
                if seg.words:
                    words = [{"word": w.word, "start": w.start, "end": w.end} for w in seg.words]
                    sentence["words"] = words
                    all_words.extend(words)
                sentences.append(sentence)
            result = {
                "language": info.language,
                "language_probability": info.language_probability,
                "segments": sentences,
                "words": all_words
            }
            return result, None
        except Exception as e:
            return None, str(e)

    def transcribe_with_segments(self, audio_path, language=None, beam_size=5, vad_filter=True, initial_prompt=None):
        with self.lock:
            if self.asr_model is None:
                return None, "ASR模型未加载"
        try:
            segments, info = self.asr_model.transcribe(
                audio_path, language=language, beam_size=beam_size,
                vad_filter=vad_filter, word_timestamps=True,
                initial_prompt=initial_prompt
            )
            seg_list = []
            for seg in segments:
                seg_dict = {"start": seg.start, "end": seg.end, "text": seg.text.strip()}
                if seg.words:
                    seg_dict["words"] = [{"word": w.word, "start": w.start, "end": w.end} for w in seg.words]
                seg_list.append(seg_dict)
            return {"language": info.language, "segments": seg_list}, None
        except Exception as e:
            return None, str(e)

    def cleanup_temp(self):
        cleaned = 0
        for f in self.temp_files[:]:
            try:
                os.unlink(f)
                self.temp_files.remove(f)
                cleaned += 1
            except:
                pass
        return cleaned

    def _prepare_audio(self, audio_input):
        try:
            if isinstance(audio_input, str) and os.path.exists(audio_input):
                return audio_input
            if isinstance(audio_input, tuple) and len(audio_input) == 2:
                sr, data = audio_input
                if data is None:
                    return None
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                if np.issubdtype(data.dtype, np.integer):
                    data = data.astype(np.float32)
                    max_val = np.abs(data).max()
                    if max_val > 0:
                        data = data / max_val
                else:
                    data = data.astype(np.float32)
                    data = np.clip(data, -1.0, 1.0)
                if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                if sr != 16000:
                    data = librosa.resample(data, orig_sr=sr, target_sr=16000)
                    sr = 16000
                data = np.clip(data, -1.0, 1.0)
                temp_hash = hashlib.md5(data.tobytes() + str(time.time()).encode()).hexdigest()[:8]
                temp_path = os.path.join(tempfile.gettempdir(), f"whisperx_temp_{temp_hash}.wav")
                sf.write(temp_path, data, sr)
                self.temp_files.append(temp_path)
                return temp_path
            return None
        except Exception as e:
            print(f"音频转换失败: {e}")
            import traceback
            traceback.print_exc()
            return None

manager = WhisperXManager()

# ==================== WebUI 核心函数 ====================
def ensure_model_loaded(model_size, device, compute_type, language, progress=None):
    success, msg = manager.load_asr_model(model_size, device, compute_type, language)
    if not success:
        raise RuntimeError(msg)
    return manager

def transcribe_audio(audio, model_size, device, compute_type, language, beam_size, vad_filter,
                     hotwords, progress=gr.Progress()):
    if audio is None:
        return "请上传或录制音频", "", "", ""
    progress(0, desc="初始化...")
    try:
        ensure_model_loaded(model_size, device, compute_type, language, progress)
    except RuntimeError as e:
        return str(e), "", "", ""
    progress(0.3, desc="转写中...")
    audio_path = manager._prepare_audio(audio)
    if audio_path is None:
        return "音频处理失败", "", "", ""
    try:
        result, error = manager.transcribe(
            audio_path, language=language, beam_size=beam_size,
            vad_filter=vad_filter, word_timestamps=True,
            initial_prompt=hotwords if hotwords.strip() else None
        )
        if error:
            return f"错误: {error}", "", "", ""
        progress(0.7, desc="生成输出...")
        full_text, timestamps_json, srt_text, segments = format_result_to_outputs(result)
        base_name = None
        if isinstance(audio, str) and os.path.exists(audio):
            base_name = audio
        saved = save_outputs(
            base_name, full_text, timestamps_json, srt_text,
            language=result.get("language", "未知"), model_info=model_size
        )
        preview_max_size = manager.settings.get("preview_max_size_mb", 5)
        subtitle_html = generate_subtitle_html(segments, audio_path, preview_max_size)
        save_info = "文件已保存:\n"
        if saved.get('txt'):
            save_info += f" {Path(saved['txt']).name}\n"
        if saved.get('json'):
            save_info += f" {Path(saved['json']).name}\n"
        if saved.get('srt'):
            save_info += f" {Path(saved['srt']).name}\n"
        full_text = save_info + "\n" + full_text
        progress(1.0, desc="完成")
        return full_text, timestamps_json, srt_text, subtitle_html
    finally:
        manager.cleanup_temp()

def transcribe_video(video, model_size, device, compute_type, language, beam_size, vad_filter,
                     subtitle_mode, hotwords, progress=gr.Progress()):
    temp_audio_path = None
    try:
        if video is None:
            return "请上传视频文件", "", ""
        progress(0, desc="初始化...")
        ensure_model_loaded(model_size, device, compute_type, language, progress)
        progress(0.2, desc="提取视频音频...")
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_audio.close()
        audio_path = temp_audio.name
        temp_audio_path = audio_path
        cmd = [FFMPEG_PATH, "-i", video, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-y", audio_path]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        progress(0.4, desc="转写音频...")
        result, error = manager.transcribe(
            audio_path, language=language, beam_size=beam_size,
            vad_filter=vad_filter, word_timestamps=True,
            initial_prompt=hotwords if hotwords.strip() else None
        )
        if error:
            return f"识别失败: {error}", "", ""
        progress(0.6, desc="生成字幕...")
        full_text, timestamps_json, srt_text, segments = format_result_to_outputs(result)
        base_name = video if isinstance(video, str) and os.path.exists(video) else None
        saved = save_outputs(
            base_name, full_text, timestamps_json, srt_text,
            language=result.get("language", "未知"), model_info=model_size
        )
        save_info = "文件已保存:\n"
        if saved.get('txt'):
            save_info += f" {Path(saved['txt']).name}\n"
        if saved.get('json'):
            save_info += f" {Path(saved['json']).name}\n"
        if saved.get('srt'):
            save_info += f" {Path(saved['srt']).name}\n"
        srt_path = saved.get('srt')
        if not srt_path:
            result_msg = f"处理完成，但未生成字幕（可能音频无语音）。\n{save_info}"
            return result_msg, "", ""
        progress(0.8, desc="嵌入字幕...")
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        prefix = generate_output_filename(video, timestamp_str, subtitle_mode, default_name="video")
        out_path = OUTPUT_DIR / f"{prefix}.mp4"
        srt_path_str = str(srt_path).replace('\\', '/')
        video_path_str = str(video).replace('\\', '/')
        out_path_str = str(out_path).replace('\\', '/')
        if subtitle_mode == "soft":
            cmd = [FFMPEG_PATH, "-i", video_path_str, "-i", srt_path_str,
                   "-c", "copy", "-c:s", "mov_text",
                   "-metadata:s:s:0", "language=chi", "-y", out_path_str]
        else:
            safe_srt = srt_path_str.replace("'", "'\\''")
            vf = f"subtitles='{safe_srt}':force_style='FontName=Microsoft YaHei,FontSize=24,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,BorderStyle=3'"
            cmd = [FFMPEG_PATH, "-i", video_path_str,
                   "-vf", vf,
                   "-c:a", "copy", "-y", out_path_str]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        result_msg = f"处理完成！\n输出视频: {out_path.name}\n{save_info}"
        combined_text = f"{result_msg}\n\n[识别文本]\n{full_text}"
        progress(1.0, desc="完成")
        return combined_text, timestamps_json, srt_text
    except Exception as e:
        logging.error(traceback.format_exc())
        return f"处理视频失败: {str(e)}", "", ""
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
            except:
                pass
        manager.cleanup_temp()

def transcribe_batch(files, model_size, device, compute_type, language, beam_size, vad_filter,
                     hotwords, progress=gr.Progress()):
    if not files:
        return "请选择音频文件", "", ""
    try:
        ensure_model_loaded(model_size, device, compute_type, language, progress)
    except RuntimeError as e:
        return str(e), "", ""
    results_text = []
    total = len(files)
    for i, file_obj in enumerate(files, 1):
        file_path = file_obj.name if hasattr(file_obj, 'name') else str(file_obj)
        progress(i / total, desc=f"处理 {i}/{total}: {os.path.basename(file_path)}")
        audio_path = manager._prepare_audio(file_path)
        if audio_path is None:
            results_text.append(f"[{os.path.basename(file_path)}]\n错误: 音频处理失败\n")
            continue
        try:
            result, error = manager.transcribe(
                audio_path, language=language, beam_size=beam_size,
                vad_filter=vad_filter, word_timestamps=True,
                initial_prompt=hotwords if hotwords.strip() else None
            )
            if error:
                results_text.append(f"[{os.path.basename(file_path)}]\n错误: {error}\n")
            else:
                full_text, timestamps_json, srt_text, _ = format_result_to_outputs(result)
                saved = save_outputs(
                    file_path, full_text, timestamps_json, srt_text,
                    language=result.get("language", "未知"), model_info=model_size
                )
                saved_files = []
                if saved.get('txt'):
                    saved_files.append(f"{Path(saved['txt']).name}")
                if saved.get('json'):
                    saved_files.append(f"{Path(saved['json']).name}")
                if saved.get('srt'):
                    saved_files.append(f"{Path(saved['srt']).name}")
                file_list = "\n ".join(saved_files) if saved_files else "无文件保存"
                results_text.append(f"[{os.path.basename(file_path)}]\n已保存:\n {file_list}\n")
        finally:
            manager.cleanup_temp()
    return "\n".join(results_text), "", ""

def force_align_wrapper(
    audio, reference_text, model_size, device, compute_type, language, beam_size, vad_filter,
    align_granularity, hotwords,
    align_model_choice, auto_match_align,  # 新增对齐模型参数
    merge_punctuations, merge_max_words, merge_max_chars, merge_max_duration,
    merge_silence_threshold, merge_by_punc, merge_by_silence, merge_by_wordcount,
    merge_by_charcount, merge_by_duration, merge_by_newline,
    progress=gr.Progress()
):
    if audio is None:
        return "请上传音频文件", "", "", get_system_info()
    if not reference_text.strip():
        return "请粘贴稿子文本", "", "", get_system_info()

    # ---- 1. 确定对齐模型 ----
    local_align_models = manager.get_local_align_models()
    align_model_path = None
    align_model_display = "默认（简单算法）"
    use_whisperx_align = False

    if auto_match_align and language:
        model_info, is_local = get_align_model_from_language(language, local_align_models)
        if model_info:
            align_model_path = model_info
            align_model_display = f"{model_info} (自动匹配{'本地' if is_local else '在线'})"
            use_whisperx_align = WHISPERX_ALIGN_AVAILABLE
        else:
            align_model_display = f"未找到语言 '{language}' 的对齐模型，将使用简单算法"
    else:
        if align_model_choice and align_model_choice != "无（使用默认）":
            found = False
            for disp, path in local_align_models:
                if disp == align_model_choice:
                    align_model_path = path
                    align_model_display = f"{disp} (本地)"
                    found = True
                    use_whisperx_align = WHISPERX_ALIGN_AVAILABLE
                    break
            if not found:
                align_model_path = align_model_choice
                align_model_display = f"{align_model_choice} (在线)"
                use_whisperx_align = WHISPERX_ALIGN_AVAILABLE
        else:
            use_whisperx_align = WHISPERX_ALIGN_AVAILABLE
            align_model_path = None
            align_model_display = "默认（WhisperX自动选择）" if use_whisperx_align else "默认（简单算法）"

    status_text = get_system_info(align_model_display)

    # ---- 2. 加载ASR模型并转写 ----
    progress(0.1, desc="加载ASR模型...")
    try:
        ensure_model_loaded(model_size, device, compute_type, language, progress)
    except RuntimeError as e:
        return str(e), "", "", status_text

    progress(0.3, desc="转写音频...")
    audio_path = manager._prepare_audio(audio)
    if not audio_path:
        return "音频处理失败", "", "", status_text

    try:
        result, err = manager.transcribe_with_segments(
            audio_path, language, beam_size, vad_filter,
            initial_prompt=hotwords if hotwords.strip() else None
        )
        if err:
            return f"转写失败: {err}", "", "", status_text

        # ---- 3. 精细对齐（如果可用） ----
        if use_whisperx_align:
            progress(0.6, desc=f"加载对齐模型: {align_model_display}...")
            try:
                model_name_for_align = align_model_path
                model_dir_for_align = str(ROOT_DIR / "pretrained_models")
                if align_model_choice == "无（使用默认）" and auto_match_align == False:
                    model_name_for_align = None

                align_model, align_metadata = manager.load_align_model(
                    language_code=language or result.get("language", "en"),
                    device=device,
                    model_name=model_name_for_align,
                    model_dir=model_dir_for_align
                )
                progress(0.7, desc="执行精细对齐...")
                aligned_result = align(
                    transcript=result["segments"],
                    model=align_model,
                    align_model_metadata=align_metadata,
                    audio=audio_path,
                    device=device,
                    return_char_alignments=(align_granularity == "char")
                )
                result = aligned_result
                manager.unload_align_model()
            except Exception as e:
                progress(0.7, desc="对齐失败，回退到简单算法")
                print(f"警告: whisperx.align 执行失败，将使用简单对齐算法。错误: {e}")
                manager.unload_align_model()
                use_whisperx_align = False

        # ---- 4. 提取单词时间戳 ----
        words = []
        if use_whisperx_align:
            for seg in result.get("segments", []):
                if "words" in seg and seg["words"]:
                    words.extend(seg["words"])
                elif "char-segments" in seg and align_granularity == "char":
                    for char_seg in seg["char-segments"]:
                        words.append({"word": char_seg["char"], "start": char_seg["start"], "end": char_seg["end"]})
        else:
            for seg in result.get("segments", []):
                if "words" in seg and seg["words"]:
                    words.extend(seg["words"])

        if not words:
            return "未能获得单词时间戳", "", "", status_text

        # ---- 5. 文稿匹配 ----
        progress(0.8, desc="匹配文稿...")
        try:
            data, sr = sf.read(audio_path)
            duration = len(data) / sr
        except:
            duration = None

        if align_granularity == "char":
            aligned = force_align_char_level(reference_text, words, duration)
        else:
            aligned = force_align_word_level(reference_text, words, duration)

        if not aligned:
            return "对齐失败，请检查稿子与音频是否匹配", "", "", status_text

        # ---- 6. 生成字幕 ----
        word_srt = words_to_srt(aligned)
        paragraphs = re.split(r'\n\s*\n', reference_text.strip())
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        char_pos = 0
        sentences = []
        if align_granularity == "char":
            for para in paragraphs:
                para_char_count = len([ch for ch in para if not ch.isspace()])
                start_idx = char_pos
                end_idx = char_pos + para_char_count
                if start_idx >= len(aligned):
                    break
                if end_idx > len(aligned):
                    end_idx = len(aligned)
                seg_words = aligned[start_idx:end_idx]
                if seg_words:
                    sentences.append({
                        "start": seg_words[0]["start"],
                        "end": seg_words[-1]["end"],
                        "text": "".join([w["word"] for w in seg_words])
                    })
                char_pos = end_idx
        else:
            word_idx = 0
            for para in paragraphs:
                para_words = para.split()
                para_word_count = len(para_words)
                start_idx = word_idx
                end_idx = word_idx + para_word_count
                if start_idx >= len(aligned):
                    break
                if end_idx > len(aligned):
                    end_idx = len(aligned)
                seg_words = aligned[start_idx:end_idx]
                if seg_words:
                    sentences.append({
                        "start": seg_words[0]["start"],
                        "end": seg_words[-1]["end"],
                        "text": " ".join([w["word"] for w in seg_words])
                    })
                word_idx = end_idx

        sent_srt = sentences_to_srt(sentences)
        merged_srt = _generate_merged_srt(
            aligned, sentences, paragraphs,
            merge_punctuations, merge_max_words, merge_max_chars,
            merge_max_duration, merge_by_newline,
            merge_by_punc, merge_by_silence, merge_by_wordcount,
            merge_by_charcount, merge_by_duration, merge_silence_threshold
        )

        # ---- 7. 保存文件 ----
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        prefix = f"align_{timestamp}"
        word_path = ALIGN_OUTPUT_DIR / f"{prefix}_words.srt"
        sent_path = ALIGN_OUTPUT_DIR / f"{prefix}_sentences.srt"
        merged_path = ALIGN_OUTPUT_DIR / f"{prefix}_merged.srt"
        with open(word_path, "w", encoding="utf-8") as f:
            f.write(word_srt)
        with open(sent_path, "w", encoding="utf-8") as f:
            f.write(sent_srt)
        with open(merged_path, "w", encoding="utf-8") as f:
            f.write(merged_srt)

        progress(1.0, desc="完成")
        return word_srt, sent_srt, merged_srt, status_text
    finally:
        manager.cleanup_temp()

def load_model_click(model_size, device, compute_type, language):
    success, msg = manager.load_asr_model(model_size, device, compute_type, language)
    return msg, get_system_info()

def unload_model_click():
    manager.unload_models()
    return "模型已卸载", get_system_info()

def refresh_status():
    return get_system_info()

def health_check():
    info = get_system_info()
    with manager.lock:
        if manager.asr_model is None:
            info += "\n\n[警告] ASR模型未加载，请先加载模型。"
        else:
            info += "\n\n[信息] 系统已就绪。"
    return info

def refresh_align_model_list():
    models = manager.get_local_align_models()
    choices = ["无（使用默认）"] + [disp for disp, _ in models]
    return gr.update(choices=choices, value="无（使用默认）")

# ==================== 创建 Gradio 界面 ====================
def create_interface():
    settings = manager.settings
    default_output_dir = settings.get("output_dir", str(DEFAULT_OUTPUT_DIR))

    global OUTPUT_DIR, ALIGN_OUTPUT_DIR
    with config_lock:
        OUTPUT_DIR = Path(default_output_dir)
        ALIGN_OUTPUT_DIR = OUTPUT_DIR / "字幕自动打轴"
        ALIGN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    local_models_with_path = manager.get_available_local_models()
    local_display_names = [name for name, path in local_models_with_path]
    default_models = ["tiny", "base", "small", "medium", "large-v2", "large-v3", "large-v3-turbo"]
    model_choices = local_display_names + [m for m in default_models if m not in local_display_names]
    if not model_choices:
        model_choices = default_models

    local_align_models = manager.get_local_align_models()
    align_choices = ["无（使用默认）"] + [disp for disp, _ in local_align_models]

    device_choices = ["cuda" if torch.cuda.is_available() else "cpu", "cpu"]
    compute_type_choices = ["int8_float32", "float16", "float32"]

    with gr.Blocks(title="WhisperX WebUI 全功能版", theme=gr.themes.Default()) as demo:
        gr.Markdown(
            "# WhisperX 语音识别与字幕自动打轴系统\n"
            "支持多语种、单词级时间戳、视频字幕嵌入、强制对齐（WhisperX精细对齐+回退算法）\n"
            f"输出目录: `{OUTPUT_DIR}`\n"
            f"字幕自动打轴输出: `{ALIGN_OUTPUT_DIR}`"
        )

        with gr.Accordion("系统状态信息 (点击展开/折叠)", open=False):
            with gr.Row():
                status_display = gr.Textbox(
                    label="系统状态", value=get_system_info(), lines=6, interactive=False, scale=4
                )
                with gr.Column(scale=1):
                    refresh_btn = gr.Button("刷新状态", variant="secondary")
                    health_btn = gr.Button("健康检查", variant="secondary")
            health_btn.click(health_check, outputs=[status_display])

        with gr.Row():
            with gr.Column(scale=1):
                device = gr.Dropdown(label="设备", choices=device_choices, value=device_choices[0])
                model_size = gr.Dropdown(
                    label="模型大小", choices=model_choices,
                    value=model_choices[0] if model_choices else "medium"
                )
            with gr.Column(scale=1):
                compute_type = gr.Dropdown(label="计算类型", choices=compute_type_choices, value="int8_float32")
                language = gr.Textbox(label="语言代码 (留空自动检测)", value="zh", placeholder="例如: zh, en, ja")

        with gr.Row():
            with gr.Column(scale=1):
                load_btn = gr.Button("加载模型", variant="primary")
                unload_btn = gr.Button("卸载模型", variant="stop")
            with gr.Column(scale=1):
                beam_size = gr.Slider(label="Beam Size", minimum=1, maximum=10, value=5, step=1)
                vad_filter = gr.Checkbox(label="启用 VAD 过滤", value=True)

        load_btn.click(load_model_click, inputs=[model_size, device, compute_type, language],
                       outputs=[status_display, status_display])
        unload_btn.click(unload_model_click, outputs=[status_display, status_display])
        refresh_btn.click(refresh_status, outputs=[status_display])

        gr.Markdown("---")

        with gr.Tabs():

            with gr.Tab("音频识别"):
                with gr.Row():
                    with gr.Column(scale=1):
                        audio_input = gr.Audio(label="选择或录制音频", type="numpy",
                                               sources=["upload", "microphone"])
                        hotwords_audio = gr.Textbox(
                            label="热词/提示词 (initial_prompt)",
                            placeholder="例如：以下是关于人工智能的讨论，重点关注术语：Transformer、扩散模型",
                            lines=2,
                            value=""
                        )
                        with gr.Row():
                            transcribe_btn = gr.Button("开始识别", variant="primary")
                            clear_btn = gr.Button("清空", variant="secondary")
                    with gr.Column(scale=2):
                        with gr.Tabs():
                            with gr.Tab("识别文本"):
                                text_output = gr.Textbox(label="结果", lines=15, show_copy_button=True)
                            with gr.Tab("时间戳 (JSON)"):
                                json_output = gr.Textbox(label="时间戳数据", lines=15, show_copy_button=True)
                            with gr.Tab("SRT字幕"):
                                srt_output = gr.Textbox(label="SRT字幕", lines=15, show_copy_button=True)
                            with gr.Tab("字幕预览"):
                                preview_output = gr.HTML(label="字幕预览",
                                                         value=generate_subtitle_html([], None))

                transcribe_btn.click(
                    transcribe_audio,
                    inputs=[audio_input, model_size, device, compute_type, language, beam_size, vad_filter, hotwords_audio],
                    outputs=[text_output, json_output, srt_output, preview_output]
                ).then(refresh_status, outputs=[status_display])

                clear_btn.click(
                    lambda: [None, "", "", "", "", generate_subtitle_html([], None)],
                    outputs=[audio_input, hotwords_audio, text_output, json_output, srt_output, preview_output]
                )

            with gr.Tab("视频字幕"):
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input = gr.Video(label="选择视频文件", sources=["upload"], interactive=True)
                        subtitle_mode = gr.Radio(label="字幕嵌入模式", choices=["soft", "hard"], value="soft")
                        hotwords_video = gr.Textbox(
                            label="热词/提示词 (initial_prompt)",
                            placeholder="例如：以下是关于人工智能的讨论，重点关注术语：Transformer、扩散模型",
                            lines=2,
                            value=""
                        )
                        with gr.Row():
                            video_transcribe_btn = gr.Button("开始处理", variant="primary")
                            video_clear_btn = gr.Button("清空", variant="secondary")
                    with gr.Column(scale=2):
                        with gr.Tabs():
                            with gr.Tab("识别文本"):
                                video_text_output = gr.Textbox(label="结果", lines=15, show_copy_button=True)
                            with gr.Tab("时间戳 (JSON)"):
                                video_json_output = gr.Textbox(label="时间戳数据", lines=15, show_copy_button=True)
                            with gr.Tab("SRT字幕"):
                                video_srt_output = gr.Textbox(label="SRT字幕", lines=15, show_copy_button=True)

                video_transcribe_btn.click(
                    transcribe_video,
                    inputs=[video_input, model_size, device, compute_type, language, beam_size, vad_filter, subtitle_mode, hotwords_video],
                    outputs=[video_text_output, video_json_output, video_srt_output]
                ).then(refresh_status, outputs=[status_display])

                video_clear_btn.click(
                    lambda: [None, "", "", "", ""],
                    outputs=[video_input, hotwords_video, video_text_output, video_json_output, video_srt_output]
                )

            with gr.Tab("批量处理"):
                with gr.Row():
                    with gr.Column(scale=1):
                        file_input = gr.Files(
                            label="上传多个音频文件",
                            file_types=[".wav", ".mp3", ".m4a", ".flac", ".ogg"],
                            file_count="multiple"
                        )
                        hotwords_batch = gr.Textbox(
                            label="热词/提示词 (initial_prompt)",
                            placeholder="例如：以下是关于人工智能的讨论，重点关注术语：Transformer、扩散模型",
                            lines=2,
                            value=""
                        )
                        batch_transcribe_btn = gr.Button("批量识别", variant="primary")
                        batch_clear = gr.Button("清空", variant="secondary")
                    with gr.Column(scale=2):
                        batch_output = gr.Textbox(label="批量结果", lines=20, show_copy_button=True)

                batch_transcribe_btn.click(
                    transcribe_batch,
                    inputs=[file_input, model_size, device, compute_type, language, beam_size, vad_filter, hotwords_batch],
                    outputs=[batch_output, gr.State(), gr.State()]
                ).then(refresh_status, outputs=[status_display])

                batch_clear.click(
                    lambda: [None, "", ""],
                    outputs=[file_input, hotwords_batch, batch_output]
                )

            with gr.Tab("字幕自动打轴 (文稿生字幕)"):
                with gr.Row():
                    with gr.Column(scale=1):
                        align_audio = gr.Audio(label="选择音频文件", type="filepath", sources=["upload"])
                        align_text = gr.Textbox(
                            label="粘贴稿子文本", lines=12,
                            placeholder="将稿子文本粘贴到这里...\n段落之间请用空行（连续两个换行符）分隔。"
                        )
                        align_granularity = gr.Radio(
                            label="对齐粒度", choices=[("字符级（中文）", "char"), ("单词级（英文等）", "word")],
                            value="char", info="字符级适合中文（逐字对齐），单词级适合英文、日文罗马字等"
                        )
                        hotwords_align = gr.Textbox(
                            label="热词/提示词 (initial_prompt)",
                            placeholder="例如：以下是关于人工智能的讨论，重点关注术语：Transformer、扩散模型",
                            lines=2,
                            value=""
                        )
                        gr.Markdown("###  多语种对齐模型")
                        with gr.Row():
                            align_model_dropdown = gr.Dropdown(
                                label="对齐模型", choices=align_choices, value="无（使用默认）",
                                interactive=True
                            )
                            refresh_align_btn = gr.Button("刷新列表", size="sm")
                        auto_match_check = gr.Checkbox(
                            label="根据语言代码自动匹配（推荐）", value=True,
                            info="勾选后将根据左侧语言代码自动选择最合适的对齐模型"
                        )
                        with gr.Accordion("字幕合并参数", open=False):
                            merge_punctuations = gr.Textbox(label="句末标点符号", value="。！？.!?")
                            with gr.Row():
                                merge_max_words = gr.Slider(label="最大词数", minimum=5, maximum=50,
                                                            value=20, step=1)
                                merge_max_chars = gr.Slider(label="最大字符数", minimum=5, maximum=100,
                                                            value=30, step=5)
                            with gr.Row():
                                merge_max_duration = gr.Slider(label="最大时长 (秒)", minimum=1.0,
                                                                maximum=20.0, value=10.0, step=0.5)
                                merge_silence_threshold = gr.Slider(label="静音阈值 (秒)", minimum=0.1,
                                                                    maximum=1.0, value=0.3, step=0.05)
                            with gr.Row():
                                merge_by_punc = gr.Checkbox(label="根据标点断句", value=True)
                                merge_by_silence = gr.Checkbox(label="根据静音断句", value=True)
                                merge_by_wordcount = gr.Checkbox(label="根据词数断句", value=True)
                            with gr.Row():
                                merge_by_charcount = gr.Checkbox(label="根据字符数断句", value=True)
                                merge_by_duration = gr.Checkbox(label="根据时长断句", value=True)
                                merge_by_newline = gr.Checkbox(
                                    label="根据空行断句", value=True,
                                    info="强烈推荐：勾选此项将严格按稿子中的空行分段"
                                )

                        with gr.Row():
                            align_btn = gr.Button("生成精准字幕", variant="primary")
                            align_clear = gr.Button("清空", variant="secondary")

                    with gr.Column(scale=2):
                        align_status_box = gr.Textbox(
                            label="对齐状态", value=get_system_info(), lines=5,
                            interactive=False, show_copy_button=True
                        )
                        with gr.Tabs():
                            with gr.Tab("逐词/逐字 SRT"):
                                align_word_output = gr.Textbox(label="逐词字幕", lines=40,
                                                                show_copy_button=True)
                            with gr.Tab("整句 SRT (按空行)"):
                                align_sent_output = gr.Textbox(label="整句子幕", lines=40,
                                                               show_copy_button=True)
                            with gr.Tab("合并字幕"):
                                align_merged_output = gr.Textbox(label="合并后的字幕", lines=40,
                                                                  show_copy_button=True)

                align_btn.click(
                    force_align_wrapper,
                    inputs=[
                        align_audio, align_text, model_size, device, compute_type, language,
                        beam_size, vad_filter, align_granularity, hotwords_align,
                        align_model_dropdown, auto_match_check,
                        merge_punctuations, merge_max_words, merge_max_chars,
                        merge_max_duration, merge_silence_threshold, merge_by_punc, merge_by_silence,
                        merge_by_wordcount, merge_by_charcount, merge_by_duration, merge_by_newline
                    ],
                    outputs=[align_word_output, align_sent_output, align_merged_output, align_status_box]
                ).then(refresh_status, outputs=[status_display])

                align_clear.click(
                    lambda: [None, "", "", "", "", "", "", "", "", get_system_info()],
                    outputs=[align_audio, align_text, hotwords_align, align_word_output,
                             align_sent_output, align_merged_output, align_status_box]
                )

                refresh_align_btn.click(
                    refresh_align_model_list,
                    outputs=[align_model_dropdown]
                )

            with gr.Tab("系统信息"):
                with gr.Column():
                    system_info_text = gr.Textbox(label="详细信息", value=get_system_info(),
                                                  lines=20, show_copy_button=True)
                    with gr.Row():
                        output_dir_input = gr.Textbox(label="输出目录", value=str(OUTPUT_DIR),
                                                      interactive=True, scale=3)
                        update_output_btn = gr.Button("更新输出目录", variant="secondary", scale=1)
                    with gr.Row():
                        preview_max_size = gr.Slider(
                            label="字幕预览最大文件大小 (MB)", minimum=1, maximum=100,
                            value=manager.settings.get("preview_max_size_mb", 5), step=1
                        )
                    def save_preview_size(size):
                        manager.settings["preview_max_size_mb"] = size
                        save_settings(manager.settings)
                        return f"预览大小已保存为 {size} MB"
                    preview_max_size.change(save_preview_size, inputs=[preview_max_size], outputs=[gr.State()])

                    with gr.Row():
                        open_output_btn = gr.Button("打开输出目录")
                        open_log_btn = gr.Button("打开日志文件夹")
                        clear_cache_btn = gr.Button("清理临时文件")
                    with gr.Row():
                        save_config_btn = gr.Button("保存当前配置", variant="primary")
                        preset_files = sorted([f.name for f in PRESET_DIR.glob("preset_*.json")],
                                              reverse=True)
                        preset_selector = gr.Dropdown(label="选择预设文件", choices=preset_files,
                                                      value=None, interactive=True)
                        load_config_btn = gr.Button("加载所选配置", variant="secondary")
                        refresh_preset_btn = gr.Button("刷新列表", variant="secondary", size="sm")
                    config_status = gr.Textbox(label="配置状态", interactive=False)

                def update_output_dir(new_dir, new_preview_size):
                    global OUTPUT_DIR, ALIGN_OUTPUT_DIR
                    try:
                        p = Path(new_dir)
                        p.mkdir(parents=True, exist_ok=True)
                        with config_lock:
                            OUTPUT_DIR = p
                            ALIGN_OUTPUT_DIR = p / "字幕自动打轴"
                            ALIGN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                            manager.settings["output_dir"] = str(p)
                            manager.settings["preview_max_size_mb"] = new_preview_size
                            save_settings(manager.settings)
                        return f"输出目录已更新为 {p}，预览阈值已设为 {new_preview_size} MB", get_system_info()
                    except Exception as e:
                        return f"更新失败: {e}", get_system_info()

                update_output_btn.click(update_output_dir,
                                        inputs=[output_dir_input, preview_max_size],
                                        outputs=[config_status, system_info_text])

                def open_output():
                    os.startfile(str(OUTPUT_DIR))
                    return "已打开输出目录"

                open_output_btn.click(open_output, outputs=[config_status])

                def open_log():
                    os.startfile(str(LOG_DIR))
                    return "已打开日志文件夹"

                open_log_btn.click(open_log, outputs=[config_status])

                def clear_cache():
                    cleaned = manager.cleanup_temp()
                    return f"清理了 {cleaned} 个临时文件"

                clear_cache_btn.click(clear_cache, outputs=[config_status])

                def save_current_config():
                    config = {
                        "model_size": model_size.value,
                        "device": device.value,
                        "compute_type": compute_type.value,
                        "language": language.value,
                        "beam_size": beam_size.value,
                        "vad_filter": vad_filter.value,
                        "align_granularity": align_granularity.value,
                        "merge_punctuations": merge_punctuations.value,
                        "merge_max_words": merge_max_words.value,
                        "merge_max_chars": merge_max_chars.value,
                        "merge_max_duration": merge_max_duration.value,
                        "merge_silence_threshold": merge_silence_threshold.value,
                        "merge_by_punc": merge_by_punc.value,
                        "merge_by_silence": merge_by_silence.value,
                        "merge_by_wordcount": merge_by_wordcount.value,
                        "merge_by_charcount": merge_by_charcount.value,
                        "merge_by_duration": merge_by_duration.value,
                        "merge_by_newline": merge_by_newline.value,
                        "preview_max_size_mb": preview_max_size.value,
                    }
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    preset_path = PRESET_DIR / f"preset_{timestamp}.json"
                    with open(preset_path, "w", encoding="utf-8") as f:
                        json.dump(config, f, ensure_ascii=False, indent=2)
                    new_choices = sorted([f.name for f in PRESET_DIR.glob("preset_*.json")], reverse=True)
                    return f"配置已保存到 {preset_path}", gr.update(choices=new_choices)

                save_config_btn.click(save_current_config, outputs=[config_status, preset_selector])

                def refresh_preset_list():
                    new_choices = sorted([f.name for f in PRESET_DIR.glob("preset_*.json")], reverse=True)
                    return gr.update(choices=new_choices)

                refresh_preset_btn.click(refresh_preset_list, outputs=[preset_selector])

                def load_selected_config(filename):
                    if not filename:
                        return ["请先选择一个预设文件"] + [gr.update() for _ in range(19)]
                    file_path = PRESET_DIR / filename
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            cfg = json.load(f)
                    except Exception as e:
                        return [f"加载失败: {e}"] + [gr.update() for _ in range(19)]
                    updates = [
                        gr.update(value=cfg.get("model_size", "medium")),
                        gr.update(value=cfg.get("device", device_choices[0])),
                        gr.update(value=cfg.get("compute_type", "int8_float32")),
                        gr.update(value=cfg.get("language", "zh")),
                        gr.update(value=cfg.get("beam_size", 5)),
                        gr.update(value=cfg.get("vad_filter", True)),
                        gr.update(value=cfg.get("align_granularity", "char")),
                        gr.update(value=cfg.get("merge_punctuations", "。！？.!?")),
                        gr.update(value=cfg.get("merge_max_words", 20)),
                        gr.update(value=cfg.get("merge_max_chars", 30)),
                        gr.update(value=cfg.get("merge_max_duration", 10.0)),
                        gr.update(value=cfg.get("merge_silence_threshold", 0.3)),
                        gr.update(value=cfg.get("merge_by_punc", True)),
                        gr.update(value=cfg.get("merge_by_silence", True)),
                        gr.update(value=cfg.get("merge_by_wordcount", True)),
                        gr.update(value=cfg.get("merge_by_charcount", True)),
                        gr.update(value=cfg.get("merge_by_duration", True)),
                        gr.update(value=cfg.get("merge_by_newline", False)),
                        gr.update(value=cfg.get("preview_max_size_mb", 5)),
                    ]
                    return [f"配置已加载: {filename}"] + updates

                load_config_btn.click(
                    load_selected_config,
                    inputs=[preset_selector],
                    outputs=[
                        config_status, model_size, device, compute_type, language,
                        beam_size, vad_filter, align_granularity,
                        merge_punctuations, merge_max_words, merge_max_chars,
                        merge_max_duration, merge_silence_threshold,
                        merge_by_punc, merge_by_silence, merge_by_wordcount,
                        merge_by_charcount, merge_by_duration, merge_by_newline, preview_max_size
                    ]
                )

        gr.Markdown("---")
        gr.HTML("""
<div class="notice" style="margin: 10px 0; padding: 10px; background: transparent; border-left: 4px solid #ff9800; font-size: 0.9em;">
    注意事项：<br>
    &bull; 本工具仅用于个人学习与视频剪辑使用<br>
    &bull; 禁止用于商业用途及侵权行为<br>
    &bull; 使用前确保模型与依赖环境正常配置
</div>
<div style="text-align: center; color: #666; font-size: 0.9em; margin-top: 15px;">
    <p>本软件包不提供任何模型文件，模型由用户自行从官方渠道获取。用户需自行遵守模型的原许可证。</p>
    <p>本软件包按"原样"提供，不提供任何明示或暗示的担保。使用本软件所产生的一切风险由用户自行承担。</p>
    <p>本软件包开发者不对因使用本软件而导致的任何直接或间接损失负责。</p>
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 8px; margin: 15px auto; max-width: 600px;">
        <p style="color: white; font-weight: bold; margin: 5px 0; font-size: 1em;">更新请关注B站up主：光影的故事2018</p>
        <p style="color: white; margin: 5px 0; font-size: 0.9em;">
            <strong>B站主页</strong>:
            <a href="https://space.bilibili.com/381518712" target="_blank" style="color: #ffdd40; text-decoration: none; font-weight: bold;">
                space.bilibili.com/381518712
            </a>
        </p>
    </div>
</div>
<div style="text-align: center; color: #666; margin-top: 10px; font-size: 0.9em;">
    &copy; 原创 WebUI 代码 &copy; 2026 光影紐扣 版权所有
</div>
""")

        demo.load(refresh_status, outputs=[status_display])

    return demo

# ==================== 退出清理 ====================
@atexit.register
def cleanup():
    print("正在退出，清理资源...")
    manager.unload_models()
    manager.cleanup_temp()
    clean_old_logs()
    print("清理完成")

# ==================== 主函数 ====================
def main():
    demo = create_interface()
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=18006,
        inbrowser=True,
        show_error=True
    )

if __name__ == "__main__":
    main()
