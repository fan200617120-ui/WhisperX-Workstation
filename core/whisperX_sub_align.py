#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
字幕自动打轴独立 UI 版（字幕自动打轴通用版）- 生产级优化
基于 WhisperX，提供图形界面进行文稿与音频的强制对齐，支持副文稿挂载生成双语字幕
布局：左侧音频+主/副文稿，右侧所有参数折叠面板
修复：输出绑定错误、线程安全、音频输入类型、WhisperX fallback、字符时长下限等
修复(2026.04.22)：WhisperX 字符级对齐提取逻辑 + VAD 离线环境容错
Copyright 2026 光影的故事2018
"""

import sys
import os
import re
import time
import shutil
import gc
import threading
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union

CURRENT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import gradio as gr
    import torch
    import numpy as np
    import librosa
    import soundfile as sf
    from faster_whisper import WhisperModel
except ImportError as e:
    print(f"缺少基础依赖库: {e}")
    sys.exit(1)

try:
    from whisperx import load_align_model, align
    WHISPERX_ALIGN_AVAILABLE = True
except ImportError:
    print("警告: 未找到 whisperx.align 模块，多语种精细对齐功能将不可用。")
    WHISPERX_ALIGN_AVAILABLE = False

# ==================== 工具函数 ====================
def seconds_to_srt_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def words_to_srt(words_with_time: List[Dict]) -> str:
    lines = []
    for i, w in enumerate(words_with_time, 1):
        lines.append(str(i))
        lines.append(f"{seconds_to_srt_time(w['start'])} --> {seconds_to_srt_time(w['end'])}")
        lines.append(w["word"])
        lines.append("")
    return "\n".join(lines)

def sentences_to_srt(sentences: List[Dict]) -> str:
    lines = []
    for i, s in enumerate(sentences, 1):
        lines.append(str(i))
        lines.append(f"{seconds_to_srt_time(s['start'])} --> {seconds_to_srt_time(s['end'])}")
        lines.append(s["text"])
        lines.append("")
    return "\n".join(lines)

def force_align_char_level(reference_text: str, transcribed_words: List[Dict],
                           audio_duration: Optional[float] = None) -> List[Dict]:
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
            else:
                break
        for j in range(i + 1, len(ref_chars)):
            if match_map[j] < len(char_to_word_idx) and char_to_word_idx[match_map[j]] == w_idx:
                total_in_word += 1
            else:
                break
        word_duration = end_t - start_t
        # 修复：字符时长下限 0.02 秒，避免字幕闪烁
        char_duration = max(word_duration / total_in_word, 0.02) if total_in_word > 0 else 0.02
        char_start = start_t + (count_in_word - 1) * char_duration
        char_end = char_start + char_duration
        aligned.append({"word": r_char, "start": char_start, "end": char_end})
    return aligned

def force_align_word_level(reference_text: str, transcribed_words: List[Dict],
                           audio_duration: Optional[float] = None) -> List[Dict]:
    if not transcribed_words:
        return []
    ref_words = reference_text.split()
    hyp_words = [w["word"] for w in transcribed_words]
    aligned = []
    hyp_idx = 0
    # 计算平均词时长用于 fallback
    avg_duration = 0.3
    if len(transcribed_words) > 1:
        total_dur = transcribed_words[-1]["end"] - transcribed_words[0]["start"]
        avg_duration = max(total_dur / len(transcribed_words), 0.1)
    for ref_w in ref_words:
        found = False
        for offset in range(20):
            if hyp_idx + offset >= len(hyp_words):
                break
            hyp_w = hyp_words[hyp_idx + offset]
            ref_clean = re.sub(r'[^\w\u4e00-\u9fff]', '', ref_w.lower(), flags=re.UNICODE)
            hyp_clean = re.sub(r'[^\w\u4e00-\u9fff]', '', hyp_w.lower(), flags=re.UNICODE)
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
                start = transcribed_words[-1]["end"] if transcribed_words else 0.0
                end = start + avg_duration
            aligned.append({"word": ref_w, "start": start, "end": end})
    return aligned

def generate_merged_srt(aligned_chars: List[Dict], sentences: List[Dict], paragraphs: List[str],
                        merge_punctuations: str, merge_max_words: int, merge_max_chars: int,
                        merge_max_duration: float, merge_by_newline: bool,
                        merge_by_punc: bool, merge_by_silence: bool, merge_by_wordcount: bool,
                        merge_by_charcount: bool, merge_by_duration: bool,
                        silence_threshold: float) -> str:
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

# ==================== FFmpeg ====================
def find_ffmpeg():
    portable_dir = PROJECT_ROOT / "ffmpeg" / "bin"
    if sys.platform == "win32":
        exe = portable_dir / "ffmpeg.exe"
    else:
        exe = portable_dir / "ffmpeg"
    if exe.exists():
        return str(exe)
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    return "ffmpeg"
FFMPEG_PATH = find_ffmpeg()

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

# ==================== 模型管理器（线程安全） ====================
class AlignModelManager:
    def __init__(self):
        self.model = None
        self.current_model_name = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "int8_float32"
        self.align_model = None
        self.align_metadata = None
        self.lock = threading.RLock()  # 可重入锁，支持同一线程多次获取

    def get_local_models(self):
        models = []
        models_dir = PROJECT_ROOT / "pretrained_models"
        if not models_dir.exists():
            return []
        for item in models_dir.iterdir():
            if item.is_dir() and "faster-whisper" in item.name.lower():
                if (item / "model.bin").exists() or (item / "config.json").exists():
                    match = re.search(r'faster-whisper-(\w+(?:-\w+)?)', item.name.lower())
                    display = match.group(1) if match else item.name
                    models.append((display, str(item)))
        return models

    def get_local_align_models(self):
        models = []
        models_dir = PROJECT_ROOT / "pretrained_models"
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

    def load_model(self, model_size, device, compute_type):
        with self.lock:
            if self.model is not None and self.current_model_name == model_size:
                return True, f"模型 {model_size} 已加载"
            local_models = self.get_local_models()
            model_path = None
            for disp, path in local_models:
                if disp == model_size:
                    model_path = path
                    break
            if model_path:
                model_name_or_path = model_path
                local_only = True
            else:
                model_name_or_path = model_size
                local_only = False
            try:
                self.model = WhisperModel(
                    model_name_or_path,
                    device=device,
                    compute_type=compute_type,
                    local_files_only=local_only
                )
                self.current_model_name = model_size
                self.device = device
                self.compute_type = compute_type
                return True, f"模型 {model_size} 加载成功"
            except Exception as e:
                return False, f"加载失败: {e}"

    def transcribe_with_segments(self, audio_path, language=None, beam_size=5, vad_filter=True, initial_prompt=None):
        with self.lock:
            if self.model is None:
                return None, "模型未加载"
        try:
            segments, info = self.model.transcribe(
                audio_path, language=language, beam_size=beam_size,
                vad_filter=vad_filter, word_timestamps=True,
                initial_prompt=initial_prompt
            )
        except Exception as e:
            # 修复：离线环境下 VAD 模型缺失时的自动降级
            if vad_filter and ("vad" in str(e).lower() or "offline" in str(e).lower()):
                print(f"VAD 模型加载失败，自动关闭 VAD 并重试。原始错误: {e}")
                segments, info = self.model.transcribe(
                    audio_path, language=language, beam_size=beam_size,
                    vad_filter=False, word_timestamps=True,
                    initial_prompt=initial_prompt
                )
            else:
                return None, str(e)

        seg_list = []
        for seg in segments:
            seg_dict = {"start": seg.start, "end": seg.end, "text": seg.text.strip()}
            if seg.words:
                seg_dict["words"] = [{"word": w.word, "start": w.start, "end": w.end} for w in seg.words]
            seg_list.append(seg_dict)
        return {"language": info.language, "segments": seg_list}, None

    def load_align_model(self, language_code: str, device: str, model_name: str = None, model_dir: str = None):
        with self.lock:
            if not WHISPERX_ALIGN_AVAILABLE:
                raise RuntimeError("whisperx.align 模块不可用")
            if self.align_model is not None:
                del self.align_model
                self.align_model = None
                self.align_metadata = None
                self._clean_gpu_memory()
            self.align_model, self.align_metadata = load_align_model(
                language_code=language_code,
                device=device,
                model_name=model_name,
                model_dir=model_dir
            )
            return self.align_model, self.align_metadata

    def unload_align_model(self):
        with self.lock:
            if self.align_model is not None:
                del self.align_model
                self.align_model = None
                self.align_metadata = None
                self._clean_gpu_memory()

    def unload_system(self):
        with self.lock:
            if self.model is not None:
                del self.model
                self.model = None
                self.current_model_name = None
                self._clean_gpu_memory()
            self.unload_align_model()

    def _clean_gpu_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

manager = AlignModelManager()

def get_system_status(align_model_info: str = ""):
    lines = []
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        lines.append(f"显卡: {gpu_name} ({total_mem:.1f} GB)")
    else:
        lines.append("设备: CPU 模式")
    with manager.lock:
        lines.append(f"ASR模型: {manager.current_model_name or '未加载'}")
        lines.append(f"计算类型: {manager.compute_type}")
    lines.append(f"FFmpeg: {'已找到' if FFMPEG_PATH != 'ffmpeg' else '未找到'}")
    if align_model_info:
        lines.append(f"对齐模型: {align_model_info}")
    return "\n".join(lines)

# ==================== 公共提取单词函数（已修复） ====================
def extract_words_from_result(result: Dict, align_granularity: str, use_whisperx_align: bool) -> List[Dict]:
    """
    统一的单词/字符时间戳提取逻辑。
    修复：正确处理 WhisperX 字符级对齐输出中的 `chars` 字段。
    """
    words = []
    for seg in result.get("segments", []):
        if "words" not in seg or not seg["words"]:
            continue
        seg_words = seg["words"]
        for i, w in enumerate(seg_words):
            # --- WhisperX 字符级对齐路径 ---
            if use_whisperx_align and align_granularity == "char" and "chars" in w:
                for c in w["chars"]:
                    if "char" in c and "start" in c and "end" in c:
                        words.append({"word": c["char"], "start": c["start"], "end": c["end"]})
                    else:
                        # 极端情况：字符缺失时间戳，使用单词边界兜底
                        char_text = c.get("char", "")
                        if char_text:
                            words.append({"word": char_text, "start": w.get("start", 0.0), "end": w.get("end", 0.0)})
            else:
                # --- 单词级路径（WhisperX 单词级或 faster-whisper 回退）---
                if "start" in w and "end" in w and "word" in w:
                    words.append({"word": w["word"], "start": w["start"], "end": w["end"]})
                else:
                    # 缺失时间戳时的插值估算
                    prev_word = seg_words[i-1] if i > 0 else None
                    next_word = seg_words[i+1] if i < len(seg_words)-1 else None
                    start = prev_word["end"] if prev_word and "end" in prev_word else seg.get("start", 0.0)
                    end = next_word["start"] if next_word and "start" in next_word else seg.get("end", start+0.01)
                    word_text = w.get("word", "")
                    if word_text:
                        words.append({"word": word_text, "start": start, "end": end})
    return words

def safe_audio_path(audio_input: Union[str, tuple, dict, None]) -> Optional[str]:
    """安全提取音频文件路径"""
    if audio_input is None:
        return None
    if isinstance(audio_input, str):
        return audio_input
    if isinstance(audio_input, tuple):
        return audio_input[0] if len(audio_input) > 0 else None
    if isinstance(audio_input, dict):
        return audio_input.get("name")
    return None

# ==================== 核心对齐函数（双语版，优化后） ====================
def run_alignment(
    audio_file, primary_text, secondary_text, secondary_lang, enable_dual,
    model_size, device, compute_type, primary_lang, beam_size, vad_filter,
    hotwords,
    align_sync_lang, align_model_manual,
    align_granularity,
    merge_punctuations, merge_max_words, merge_max_chars, merge_max_duration,
    merge_silence_threshold, merge_by_punc, merge_by_silence, merge_by_wordcount,
    merge_by_charcount, merge_by_duration, merge_by_newline,
    progress=gr.Progress()
):
    if audio_file is None:
        return "错误: 请上传音频文件", "", "", "", "", "", get_system_status()
    if not primary_text or not primary_text.strip():
        return "错误: 请粘贴主文稿", "", "", "", "", "", get_system_status()

    # ---- 1. 确定对齐模型 ----
    local_align_models = manager.get_local_align_models()
    align_model_path = None
    align_model_display = ""
    use_whisperx_align = False

    if align_sync_lang and primary_lang and primary_lang != "auto":
        model_info, is_local = get_align_model_from_language(primary_lang, local_align_models)
        if model_info:
            align_model_path = model_info
            align_model_display = f"{model_info} (自动匹配{'本地' if is_local else '在线'})"
            use_whisperx_align = WHISPERX_ALIGN_AVAILABLE
        else:
            align_model_display = f"未找到语言 '{primary_lang}' 的对齐模型，将使用简单算法"
            use_whisperx_align = False
    else:
        if align_sync_lang and primary_lang == "auto":
            align_model_display = "语言为 auto，无法自动匹配对齐模型，将使用简单算法"
            use_whisperx_align = False
        else:
            if align_model_manual and align_model_manual != "无（使用默认）":
                found = False
                for disp, path in local_align_models:
                    if disp == align_model_manual:
                        align_model_path = path
                        align_model_display = f"{disp} (本地手动)"
                        found = True
                        use_whisperx_align = WHISPERX_ALIGN_AVAILABLE
                        break
                if not found:
                    align_model_path = align_model_manual
                    align_model_display = f"{align_model_manual} (在线手动)"
                    use_whisperx_align = WHISPERX_ALIGN_AVAILABLE
            else:
                align_model_display = "默认（由WhisperX自动选择）"
                use_whisperx_align = WHISPERX_ALIGN_AVAILABLE
                align_model_path = None

    system_info = get_system_status(align_model_display)

    # ---- 2. 加载ASR模型并转写 ----
    progress(0.1, desc="加载ASR模型...")
    success, msg = manager.load_model(model_size, device, compute_type)
    if not success:
        return f"错误: {msg}", "", "", "", "", "", system_info

    progress(0.3, desc="转写音频...")
    audio_path = safe_audio_path(audio_file)
    if not audio_path or not os.path.exists(audio_path):
        return "错误: 无法获取有效的音频文件路径", "", "", "", "", "", system_info

    asr_language = None if primary_lang == "auto" else primary_lang
    initial_prompt = hotwords if hotwords and hotwords.strip() else None
    result, err = manager.transcribe_with_segments(
        audio_path, language=asr_language, beam_size=beam_size, vad_filter=vad_filter,
        initial_prompt=initial_prompt
    )
    if err:
        return f"错误: 转写失败 - {err}", "", "", "", "", "", system_info

    # 保存原始结果副本，用于 fallback
    original_result = result.copy()

    # ---- 3. 精细对齐 ----
    if use_whisperx_align:
        progress(0.6, desc=f"加载对齐模型: {align_model_display}...")
        try:
            model_name_for_align = align_model_path
            model_dir_for_align = str(PROJECT_ROOT / "pretrained_models")
            if not align_sync_lang and align_model_manual == "无（使用默认）":
                model_name_for_align = None

            align_model, align_metadata = manager.load_align_model(
                language_code=asr_language or result.get("language", "en"),
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
            result = original_result  # 恢复原始结果，避免结构不一致

    # ---- 4. 提取单词时间戳（统一函数，已修复） ----
    words = extract_words_from_result(result, align_granularity, use_whisperx_align)

    if not words:
        return "错误: 未检测到有效的单词时间戳", "", "", "", "", "", system_info

    # ---- 5. 文稿匹配 ----
    progress(0.8, desc="匹配主文稿...")
    try:
        data, sr = sf.read(audio_path)
        duration = len(data) / sr
    except:
        duration = None

    if align_granularity == "char":
        aligned = force_align_char_level(primary_text, words, duration)
    else:
        aligned = force_align_word_level(primary_text, words, duration)

    if not aligned:
        return "错误: 对齐失败，请检查主文稿与音频是否匹配", "", "", "", "", "", system_info

    # ---- 6. 生成基础字幕 ----
    word_srt = words_to_srt(aligned)

    paragraphs = re.split(r'\n\s*\n', primary_text.strip())
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    char_pos = 0
    sentences = []
    if align_granularity == "char":
        for para in paragraphs:
            para_char_count = len([ch for ch in para if ch.strip()])  # 过滤所有空白字符
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
    # 使用关键字参数调用，避免位置错误
    merged_srt = generate_merged_srt(
        aligned_chars=aligned,
        sentences=sentences,
        paragraphs=paragraphs,
        merge_punctuations=merge_punctuations,
        merge_max_words=merge_max_words,
        merge_max_chars=merge_max_chars,
        merge_max_duration=merge_max_duration,
        merge_by_newline=merge_by_newline,
        merge_by_punc=merge_by_punc,
        merge_by_silence=merge_by_silence,
        merge_by_wordcount=merge_by_wordcount,
        merge_by_charcount=merge_by_charcount,
        merge_by_duration=merge_by_duration,
        silence_threshold=merge_silence_threshold
    )

    # ---- 7. 通用双语挂载（宽松匹配） ----
    dual_srt = ""
    secondary_srt = ""
    warning_msg = ""

    if enable_dual and secondary_text and secondary_text.strip():
        sec_paragraphs = [p.strip() for p in re.split(r'\n\s*\n', secondary_text.strip()) if p.strip()]
        len_diff = abs(len(sec_paragraphs) - len(sentences))
        if len_diff <= 1:  # 允许 ±1 段误差
            # 对齐段落数：如果差1，截断较长的
            if len(sec_paragraphs) > len(sentences):
                sec_paragraphs = sec_paragraphs[:len(sentences)]
                warning_msg = f"⚠️ 副文稿段落数多 {len_diff} 段，已自动截断"
            elif len(sentences) > len(sec_paragraphs):
                # 补空行
                sec_paragraphs += [""] * (len(sentences) - len(sec_paragraphs))
                warning_msg = f"⚠️ 副文稿段落数少 {len_diff} 段，已补充空行"
            # 生成副文稿单语 SRT
            sec_lines = []
            for i, (seg, sec_text) in enumerate(zip(sentences, sec_paragraphs), 1):
                sec_lines.append(str(i))
                sec_lines.append(f"{seconds_to_srt_time(seg['start'])} --> {seconds_to_srt_time(seg['end'])}")
                sec_lines.append(sec_text)
                sec_lines.append("")
            secondary_srt = "\n".join(sec_lines)
            # 生成双语 SRT
            dual_lines = []
            for i, (seg, sec_text) in enumerate(zip(sentences, sec_paragraphs), 1):
                dual_lines.append(str(i))
                dual_lines.append(f"{seconds_to_srt_time(seg['start'])} --> {seconds_to_srt_time(seg['end'])}")
                dual_lines.append(seg['text'])
                dual_lines.append(sec_text)
                dual_lines.append("")
            dual_srt = "\n".join(dual_lines)
        else:
            warning_msg = f"⚠️ 段落数相差 {len_diff} 段（超过1），跳过双语生成"

    # ---- 8. 保存文件 ----
    output_dir = PROJECT_ROOT / "output" / "字幕自动打轴"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_name = Path(audio_path).stem
    prefix = f"{base_name}_align_{timestamp}"

    safe_lang_tag = ""
    if secondary_lang and secondary_lang.strip():
        lang_clean = re.sub(r'[\\/*?:"<>|]', '', secondary_lang.strip())
        safe_lang_tag = f"_{lang_clean}"

    # 保存主文稿字幕
    word_path = output_dir / f"{prefix}_words.srt"
    sent_path = output_dir / f"{prefix}_sentences.srt"
    merged_path = output_dir / f"{prefix}_merged.srt"

    with open(word_path, "w", encoding="utf-8") as f:
        f.write(word_srt)
    with open(sent_path, "w", encoding="utf-8") as f:
        f.write(sent_srt)
    with open(merged_path, "w", encoding="utf-8") as f:
        f.write(merged_srt)

    status = (f"对齐完成！\n"
              f"逐词字幕: {word_path.name}\n"
              f"整句字幕: {sent_path.name}\n"
              f"合并字幕: {merged_path.name}\n"
              f"对齐模型: {align_model_display}")

    if secondary_srt:
        sec_path = output_dir / f"{prefix}{safe_lang_tag}_secondary.srt"
        with open(sec_path, "w", encoding="utf-8") as f:
            f.write(secondary_srt)
        status += f"\n副文稿单语: {sec_path.name}"

    if dual_srt:
        dual_path = output_dir / f"{prefix}{safe_lang_tag}_dual.srt"
        with open(dual_path, "w", encoding="utf-8") as f:
            f.write(dual_srt)
        status += f"\n双语字幕: {dual_path.name}"

    if warning_msg:
        status += f"\n{warning_msg}"

    # 返回：status, word_srt, sent_srt, merged_srt, secondary_srt, dual_srt, system_info
    return status, word_srt, sent_srt, merged_srt, secondary_srt, dual_srt, system_info

def clear_outputs():
    return "", "", "", "", "", "", get_system_status()

def refresh_align_model_list():
    models = manager.get_local_align_models()
    choices = ["无（使用默认）"] + [disp for disp, _ in models]
    return gr.update(choices=choices, value="无（使用默认）")

def toggle_align_model_manual(sync: bool):
    return gr.update(visible=not sync)

# ==================== 创建界面（右侧所有参数面板） ====================
def create_ui():
    local_models = manager.get_local_models()
    local_names = [name for name, _ in local_models]
    default_models = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
    model_choices = local_names + [m for m in default_models if m not in local_names]
    if not model_choices:
        model_choices = default_models

    local_align_models = manager.get_local_align_models()
    align_choices = ["无（使用默认）"] + [disp for disp, _ in local_align_models]

    with gr.Blocks(title="字幕自动打轴（通用版）", theme=gr.themes.Default()) as demo:
        gr.Markdown("# 🎬 字幕自动打轴（多语种通用版）")

        with gr.Row():
            # ========== 左侧列：音频和文稿 ==========
            with gr.Column(scale=1):
                audio_input = gr.Audio(label="选择音频文件", type="filepath", sources=["upload"])
                primary_text = gr.Textbox(
                    label="主文稿（对齐用）",
                    lines=26,
                    placeholder="粘贴与音频语言一致的稿子...\n段落之间用空行分隔",
                    info="此文稿将用于强制对齐，语言需与音频一致"
                )
                secondary_text = gr.Textbox(
                    label="副文稿（挂载用，可选）",
                    lines=26,
                    placeholder="粘贴任意语种的翻译稿...\n段落结构尽量与主文稿一致",
                    info="不参与对齐，仅挂载时间轴。段落数相差1段内会自动调整"
                )
                with gr.Row():
                    secondary_lang = gr.Textbox(
                        label="副文稿语言标记",
                        placeholder="如：en / ja / fr",
                        value="",
                        scale=1
                    )
                    enable_dual = gr.Checkbox(label="生成双语字幕", value=False, scale=1)

                gr.Markdown("""
                <div style="margin-top: 20px; padding: 12px; background: rgba(255,255,255,0.05); border-radius: 8px; border-left: 4px solid #ff9800; font-size: 0.9em;">
                <strong>使用提示：</strong><br>
                • 音频格式支持 wav/mp3/m4a/flac 等<br>
                • 主文稿空行将作为字幕分段依据（勾选“按空行分段”）<br>
                • 副文稿段落数相差1段内会自动调整，超出则跳过<br>
                • 生成的字幕文件保存在 <code>output/字幕自动打轴</code> 目录下
                </div>
                """)

            # ========== 右侧列：所有参数面板 ==========
            with gr.Column(scale=2):
                # 任务状态与系统信息并排
                with gr.Row():
                    status_box = gr.Textbox(
                        label="任务状态",
                        value="等待开始",
                        lines=4,
                        interactive=False,
                        show_copy_button=True,
                        scale=1
                    )
                    system_box = gr.Textbox(
                        label="系统信息",
                        value=get_system_status(),
                        lines=4,
                        interactive=False,
                        show_copy_button=True,
                        scale=1
                    )

                with gr.Accordion("⚙️ 模型与识别参数", open=True):
                    model_dropdown = gr.Dropdown(
                        label="ASR模型", choices=model_choices,
                        value=model_choices[0] if model_choices else "medium"
                    )
                    with gr.Row():
                        device_dropdown = gr.Dropdown(
                            label="设备", choices=["cuda", "cpu"],
                            value="cuda" if torch.cuda.is_available() else "cpu"
                        )
                        compute_dropdown = gr.Dropdown(
                            label="计算类型", choices=["int8_float32", "float16", "float32"],
                            value="int8_float32"
                        )
                    with gr.Row():
                        primary_lang = gr.Dropdown(
                            label="主语言",
                            choices=["auto", "zh", "en", "ja", "fr", "de", "es", "it", "pt", "nl", "hu"],
                            value="zh",
                            info="决定 ASR 识别语言，也用于自动匹配对齐模型"
                        )
                        beam_slider = gr.Slider(label="Beam Size", minimum=1, maximum=10, value=5, step=1)
                    vad_check = gr.Checkbox(label="启用 VAD 过滤", value=True)
                    hotwords_box = gr.Textbox(
                        label="热词/提示词 (initial_prompt)",
                        placeholder="例如：以下是关于人工智能的讨论，重点关注术语：Transformer、扩散模型",
                        lines=2,
                        value=""
                    )

                with gr.Accordion("🌐 对齐模型设置", open=True):
                    align_sync_lang = gr.Checkbox(
                        label="对齐模型跟随主语言自动匹配",
                        value=True,
                        info="取消勾选可手动指定 wav2vec2 对齐模型。注意：主语言为 auto 时无法自动匹配"
                    )
                    align_model_manual = gr.Dropdown(
                        label="手动选择对齐模型",
                        choices=align_choices,
                        value="无（使用默认）",
                        visible=False,
                        info="仅当上方取消勾选时才生效"
                    )
                    refresh_align_btn = gr.Button("刷新对齐模型列表", size="sm")
                    align_granularity = gr.Radio(
                        label="对齐粒度", choices=[("字符级（中文）", "char"), ("单词级（英文）", "word")],
                        value="char"
                    )

                with gr.Accordion("📝 字幕合并规则", open=True):
                    with gr.Row():
                        merge_newline = gr.Checkbox(label="按空行分段（推荐）", value=True)
                        merge_punc = gr.Checkbox(label="按标点断句", value=True)
                        merge_silence = gr.Checkbox(label="按静音断句", value=True)
                    with gr.Row():
                        merge_wordcount = gr.Checkbox(label="按词数断句", value=True)
                        merge_charcount = gr.Checkbox(label="按字符数断句", value=True)
                        merge_duration = gr.Checkbox(label="按时长断句", value=True)
                    with gr.Row():
                        punc_box = gr.Textbox(label="句末标点", value="，；。！？,;.!?", scale=2)
                        silence_slider = gr.Slider(label="静音阈值 (秒)", minimum=0.1, maximum=1.0, value=0.3, step=0.05, scale=1)
                    with gr.Row():
                        max_words_slider = gr.Slider(label="最大词数", minimum=5, maximum=50, value=20, step=1)
                        max_chars_slider = gr.Slider(label="最大字符数", minimum=5, maximum=100, value=30, step=5)
                        max_duration_slider = gr.Slider(label="最大时长 (秒)", minimum=1.0, maximum=20.0, value=10.0, step=0.5)

                with gr.Row():
                    run_btn = gr.Button("开始对齐", variant="primary", size="lg")
                    clear_btn = gr.Button("清空", variant="secondary")

                with gr.Tabs():
                    with gr.Tab("逐词/逐字 SRT"):
                        word_output = gr.Textbox(label="逐词字幕", lines=20, show_copy_button=True)
                    with gr.Tab("整句 SRT"):
                        sent_output = gr.Textbox(label="整句字幕", lines=20, show_copy_button=True)
                    with gr.Tab("合并字幕"):
                        merged_output = gr.Textbox(label="合并后的字幕", lines=20, show_copy_button=True)
                    with gr.Tab("副文稿单语 SRT"):
                        secondary_output = gr.Textbox(label="副文稿字幕", lines=20, show_copy_button=True)
                    with gr.Tab("双语 SRT（主上、副下）"):
                        dual_output = gr.Textbox(label="双语字幕", lines=20, show_copy_button=True)

        # 绑定事件
        align_sync_lang.change(
            toggle_align_model_manual,
            inputs=align_sync_lang,
            outputs=align_model_manual
        )

        run_btn.click(
            run_alignment,
            inputs=[
                audio_input, primary_text, secondary_text, secondary_lang, enable_dual,
                model_dropdown, device_dropdown, compute_dropdown,
                primary_lang, beam_slider, vad_check,
                hotwords_box,
                align_sync_lang, align_model_manual,
                align_granularity,
                punc_box, max_words_slider, max_chars_slider, max_duration_slider,
                silence_slider, merge_punc, merge_silence, merge_wordcount,
                merge_charcount, merge_duration, merge_newline
            ],
            outputs=[status_box, word_output, sent_output, merged_output, secondary_output, dual_output, system_box]
        )

        clear_btn.click(
            clear_outputs,
            outputs=[status_box, word_output, sent_output, merged_output, secondary_output, dual_output, system_box]
        ).then(
            lambda: [None, "", "", "", False],
            outputs=[audio_input, primary_text, secondary_text, secondary_lang, enable_dual]
        )

        refresh_align_btn.click(
            refresh_align_model_list,
            outputs=[align_model_manual]
        )

        # 页脚
        gr.HTML("""
        <div style="text-align: center; color: #666; font-size: 0.85em; margin-top: 20px;">
            <p>本软件包按“原样”提供，不提供任何明示或暗示的担保。使用本软件所产生的一切风险由用户自行承担。</p>
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 12px; border-radius: 8px; margin: 15px auto; max-width: 600px;">
                <p style="color: white; font-weight: bold; margin: 5px 0; font-size: 1em;">更新请关注B站up主：光影的故事2018</p>
                <p style="color: white; margin: 5px 0; font-size: 0.9em;">
                <strong>B站主页</strong>: 
                <a href="https://space.bilibili.com/381518712" target="_blank" style="color: #ffdd40; text-decoration: none; font-weight: bold;">
                space.bilibili.com/381518712
                </a>
                </p>
            </div>
            <p>© 原创 WebUI 代码 © 2026 光影紐扣 版权所有</p>
        </div>
        """)

    return demo

def main():
    demo = create_ui()
    ports_to_try = [7966, 7967, 7968, 7969, 7970]
    for port in ports_to_try:
        try:
            demo.launch(
                server_name="127.0.0.1",
                server_port=port,
                inbrowser=True
            )
            break
        except OSError:
            print(f"端口 {port} 被占用，尝试下一个...")
            continue
    else:
        print("所有端口均被占用，请手动指定空闲端口。")

if __name__ == "__main__":
    main()