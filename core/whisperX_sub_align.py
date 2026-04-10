#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
字幕自动打轴独立 UI 版（支持多语种对齐模型选择 + 热词/提示词）
基于 WhisperX，提供图形界面进行文稿与音频的强制对齐
修复：单词时间戳格式兼容问题，热词功能完善，增强单词时间戳提取（自动补全缺失字段）
Copyright 2026 光影的故事2018
"""

import sys
import os
import re
import time
import json
import shutil
import gc
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# 设置路径
CURRENT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 导入依赖
try:
    import gradio as gr
    import torch
    import numpy as np
    import librosa
    import soundfile as sf
    from faster_whisper import WhisperModel
except ImportError as e:
    print(f"缺少基础依赖库: {e}")
    print("请确保已安装 faster-whisper, gradio, librosa, soundfile")
    sys.exit(1)

# 尝试导入 whisperx 对齐模块
try:
    from whisperx import load_align_model, align
    WHISPERX_ALIGN_AVAILABLE = True
except ImportError:
    print("警告: 未找到 whisperx.align 模块，多语种精细对齐功能将不可用。")
    print("请运行: pip install whisperx")
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
    """字符级对齐（中文等），要求每个元素包含 'word', 'start', 'end'"""
    if not transcribed_words:
        return []
    ref_chars = [ch for ch in reference_text if not ch.isspace()]
    hyp_chars = []
    char_to_word_idx = []
    for w_idx, w in enumerate(transcribed_words):
        word_text = w.get("word", w.get("text", ""))
        for ch in word_text:
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
        start_t = transcribed_words[w_idx].get("start", 0.0)
        end_t = transcribed_words[w_idx].get("end", start_t + 0.1)
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

def force_align_word_level(reference_text: str, transcribed_words: List[Dict],
                           audio_duration: Optional[float] = None) -> List[Dict]:
    """单词级对齐（英文等），要求每个元素包含 'word', 'start', 'end'"""
    if not transcribed_words:
        return []
    ref_words = reference_text.split()
    hyp_words = [w.get("word", w.get("text", "")) for w in transcribed_words]
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
                    "start": transcribed_words[w_idx].get("start", 0.0),
                    "end": transcribed_words[w_idx].get("end", 0.3)
                })
                hyp_idx = w_idx + 1
                found = True
                break
        if not found:
            if hyp_idx < len(transcribed_words):
                start = transcribed_words[hyp_idx].get("start", 0.0)
                end = transcribed_words[hyp_idx].get("end", start + 0.3)
            else:
                start = aligned[-1]["end"] if aligned else 0.0
                end = start + 0.3
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

# ==================== 模型管理器 ====================
class AlignModelManager:
    def __init__(self):
        self.model = None
        self.current_model_name = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "int8_float32"
        self.align_model = None
        self.align_metadata = None

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
        if self.model is None:
            return None, "模型未加载"
        try:
            segments, info = self.model.transcribe(
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

    def load_align_model(self, language_code: str, device: str, model_name: str = None, model_dir: str = None):
        if not WHISPERX_ALIGN_AVAILABLE:
            raise RuntimeError("whisperx.align 模块不可用")
        if self.align_model is not None:
            del self.align_model
            self.align_model = None
            self.align_metadata = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
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

manager = AlignModelManager()

def get_system_status(align_model_info: str = ""):
    lines = []
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        lines.append(f"显卡: {gpu_name} ({total_mem:.1f} GB)")
    else:
        lines.append("设备: CPU 模式")
    lines.append(f"ASR模型: {manager.current_model_name or '未加载'}")
    lines.append(f"计算类型: {manager.compute_type}")
    lines.append(f"FFmpeg: {'已找到' if FFMPEG_PATH != 'ffmpeg' else '未找到'}")
    if align_model_info:
        lines.append(f"对齐模型: {align_model_info}")
    return "\n".join(lines)

# ==================== UI 处理函数 ====================
def run_alignment(
    audio_file, text_content,
    model_size, device, compute_type, language, beam_size, vad_filter,
    hotwords,
    align_model_choice, auto_match_align,
    align_granularity,
    merge_punctuations, merge_max_words, merge_max_chars, merge_max_duration,
    merge_silence_threshold, merge_by_punc, merge_by_silence, merge_by_wordcount,
    merge_by_charcount, merge_by_duration, merge_by_newline,
    progress=gr.Progress()
):
    if audio_file is None:
        return "错误: 请上传音频文件", "", "", "", get_system_status()
    if not text_content or not text_content.strip():
        return "错误: 请粘贴稿子文本", "", "", "", get_system_status()

    # ---- 1. 确定对齐模型 ----
    local_align_models = manager.get_local_align_models()
    align_model_path = None
    align_model_display = ""
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
            align_model_display = "默认（由WhisperX自动选择）"
            use_whisperx_align = WHISPERX_ALIGN_AVAILABLE
            align_model_path = None

    status_text = get_system_status(align_model_display)

    # ---- 2. 加载ASR模型并转写 ----
    progress(0.1, desc="加载ASR模型...")
    success, msg = manager.load_model(model_size, device, compute_type)
    if not success:
        return f"错误: {msg}", "", "", "", status_text

    progress(0.3, desc="转写音频...")
    audio_path = audio_file
    initial_prompt = hotwords if hotwords and hotwords.strip() else None
    result, err = manager.transcribe_with_segments(
        audio_path, language, beam_size, vad_filter,
        initial_prompt=initial_prompt
    )
    if err:
        return f"错误: 转写失败 - {err}", "", "", "", status_text

    # ---- 3. 精细对齐（如果可用） ----
    if use_whisperx_align:
        progress(0.6, desc=f"加载对齐模型: {align_model_display}...")
        try:
            model_name_for_align = align_model_path
            model_dir_for_align = str(PROJECT_ROOT / "pretrained_models")
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

    # ---- 4. 提取单词时间戳（增强版：自动补全缺失字段） ----
    words = []
    if use_whisperx_align:
        # 精细对齐结果（whisperx.align 输出）
        for seg in result.get("segments", []):
            if "words" in seg and seg["words"]:
                seg_words = seg["words"]
                for i, w in enumerate(seg_words):
                    if "start" in w and "end" in w and "word" in w:
                        words.append({"word": w["word"], "start": w["start"], "end": w["end"]})
                    else:
                        # 为缺失时间的单词估算时间
                        prev_word = seg_words[i-1] if i > 0 else None
                        next_word = seg_words[i+1] if i < len(seg_words)-1 else None
                        if prev_word and "end" in prev_word:
                            start = prev_word["end"]
                        else:
                            start = seg.get("start", 0.0)
                        if next_word and "start" in next_word:
                            end = next_word["start"]
                        else:
                            end = seg.get("end", start + 0.01)
                        word_text = w.get("word", w.get("char", ""))
                        if word_text:
                            words.append({"word": word_text, "start": start, "end": end})
            elif "char-segments" in seg and align_granularity == "char":
                for char_seg in seg["char-segments"]:
                    if "char" in char_seg and "start" in char_seg and "end" in char_seg:
                        words.append({"word": char_seg["char"], "start": char_seg["start"], "end": char_seg["end"]})
    else:
        # 回退：使用 faster-whisper 自带单词时间戳
        for seg in result.get("segments", []):
            if "words" in seg and seg["words"]:
                seg_words = seg["words"]
                for i, w in enumerate(seg_words):
                    if "start" in w and "end" in w and "word" in w:
                        words.append({"word": w["word"], "start": w["start"], "end": w["end"]})
                    else:
                        prev_word = seg_words[i-1] if i > 0 else None
                        next_word = seg_words[i+1] if i < len(seg_words)-1 else None
                        start = prev_word["end"] if prev_word and "end" in prev_word else seg.get("start", 0.0)
                        end = next_word["start"] if next_word and "start" in next_word else seg.get("end", start + 0.01)
                        word_text = w.get("word", "")
                        if word_text:
                            words.append({"word": word_text, "start": start, "end": end})

    if not words:
        return "错误: 未检测到有效的单词时间戳", "", "", "", status_text

    # ---- 5. 文稿匹配 ----
    progress(0.8, desc="匹配文稿...")
    try:
        data, sr = sf.read(audio_path)
        duration = len(data) / sr
    except:
        duration = None

    if align_granularity == "char":
        aligned = force_align_char_level(text_content, words, duration)
    else:
        aligned = force_align_word_level(text_content, words, duration)

    if not aligned:
        return "错误: 对齐失败，请检查稿子与音频是否匹配", "", "", "", status_text

    # ---- 6. 生成字幕 ----
    word_srt = words_to_srt(aligned)

    paragraphs = re.split(r'\n\s*\n', text_content.strip())
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
    merged_srt = generate_merged_srt(
        aligned, sentences, paragraphs,
        merge_punctuations, merge_max_words, merge_max_chars,
        merge_max_duration, merge_by_newline,
        merge_by_punc, merge_by_silence, merge_by_wordcount,
        merge_by_charcount, merge_by_duration, merge_silence_threshold
    )

    # ---- 7. 保存文件 ----
    output_dir = PROJECT_ROOT / "output" / "字幕自动打轴"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_name = Path(audio_path).stem
    prefix = f"{base_name}_align_{timestamp}"
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
              f"整句子幕: {sent_path.name}\n"
              f"合并字幕: {merged_path.name}\n"
              f"对齐模型: {align_model_display}")

    return status, word_srt, sent_srt, merged_srt, status_text

def clear_outputs():
    return "", "", "", "", get_system_status()

def refresh_align_model_list():
    models = manager.get_local_align_models()
    choices = ["无（使用默认）"] + [disp for disp, _ in models]
    return gr.update(choices=choices, value="无（使用默认）")

# ==================== 创建界面 ====================
def create_ui():
    local_models = manager.get_local_models()
    local_names = [name for name, _ in local_models]
    default_models = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
    model_choices = local_names + [m for m in default_models if m not in local_names]
    if not model_choices:
        model_choices = default_models

    local_align_models = manager.get_local_align_models()
    align_choices = ["无（使用默认）"] + [disp for disp, _ in local_align_models]

    with gr.Blocks(title="字幕自动打轴独立版", theme=gr.themes.Default()) as demo:
        gr.Markdown("# 🎬 字幕自动打轴（文稿生字幕）")

        with gr.Row():
            # ========== 左侧列 ==========
            with gr.Column(scale=1):
                audio_input = gr.Audio(label="选择音频文件", type="filepath", sources=["upload"])
                text_input = gr.Textbox(
                    label="稿子文本",
                    lines=10,
                    placeholder="将稿子粘贴到这里...\n段落之间请用空行分隔。",
                    info="支持中英文混合，段落之间请留空行"
                )

                with gr.Group():
                    gr.Markdown("### ⚙️ 模型与识别参数")
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
                        language_box = gr.Textbox(label="语言代码", value="zh", placeholder="留空自动检测")
                        beam_slider = gr.Slider(label="Beam Size", minimum=1, maximum=10, value=5, step=1)
                    vad_check = gr.Checkbox(label="启用 VAD 过滤", value=True)

                    hotwords_box = gr.Textbox(
                        label="热词/提示词 (initial_prompt)",
                        placeholder="例如：以下是关于人工智能的讨论，重点关注术语：Transformer、扩散模型",
                        lines=2,
                        value=""
                    )

                    gr.Markdown("### 🌐 多语种对齐模型")
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

                gr.Markdown("""
                <div style="margin-top: 20px; padding: 12px; background: rgba(255,255,255,0.05); border-radius: 8px; border-left: 4px solid #ff9800; font-size: 0.9em;">
                <strong>注意事项：</strong><br>
                • 本工具仅用于个人学习与视频剪辑使用<br>
                • 禁止用于商业用途及侵权行为<br>
                • 使用前确保模型与依赖环境正常配置
                </div>
                """)

            # ========== 右侧列 ==========
            with gr.Column(scale=2):
                status_box = gr.Textbox(
                    label="系统状态",
                    value=get_system_status(),
                    lines=5,
                    interactive=False,
                    show_copy_button=True
                )

                gr.Markdown("### 对齐与合并参数")
                with gr.Group():
                    granularity_radio = gr.Radio(
                        label="对齐粒度", choices=[("字符级（中文）", "char"), ("单词级（英文）", "word")],
                        value="char"
                    )
                    gr.Markdown("**合并规则**")
                    with gr.Row():
                        merge_newline = gr.Checkbox(label="按空行分段（推荐）", value=True)
                        merge_punc = gr.Checkbox(label="按标点断句", value=True)
                        merge_silence = gr.Checkbox(label="按静音断句", value=True)
                    with gr.Row():
                        merge_wordcount = gr.Checkbox(label="按词数断句", value=True)
                        merge_charcount = gr.Checkbox(label="按字符数断句", value=True)
                        merge_duration = gr.Checkbox(label="按时长断句", value=True)
                    with gr.Row():
                        punc_box = gr.Textbox(label="句末标点", value="。！？.!?", scale=2)
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
                    with gr.Tab("整句 SRT（按空行）"):
                        sent_output = gr.Textbox(label="整句子幕", lines=20, show_copy_button=True)
                    with gr.Tab("合并字幕"):
                        merged_output = gr.Textbox(label="合并后的字幕", lines=20, show_copy_button=True)

                gr.Markdown("""
                ---
                **使用提示**  
                - 音频文件格式支持 wav/mp3/m4a/flac 等  
                - 稿子文本中的空行将作为字幕分段依据（勾选“按空行分段”时）  
                - 生成的字幕文件保存在 `output/字幕自动打轴` 目录下
                """)

        # 绑定事件
        run_btn.click(
            run_alignment,
            inputs=[
                audio_input, text_input,
                model_dropdown, device_dropdown, compute_dropdown,
                language_box, beam_slider, vad_check,
                hotwords_box,
                align_model_dropdown, auto_match_check,
                granularity_radio,
                punc_box, max_words_slider, max_chars_slider, max_duration_slider,
                silence_slider, merge_punc, merge_silence, merge_wordcount,
                merge_charcount, merge_duration, merge_newline
            ],
            outputs=[status_box, word_output, sent_output, merged_output, status_box]
        )

        clear_btn.click(
            clear_outputs,
            outputs=[status_box, word_output, sent_output, merged_output, status_box]
        ).then(
            lambda: [None, "", ""],
            outputs=[audio_input, text_input, hotwords_box]
        )

        refresh_align_btn.click(
            refresh_align_model_list,
            outputs=[align_model_dropdown]
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
    ports_to_try = [7966, 7967, 7969, 7962, 7970]
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
