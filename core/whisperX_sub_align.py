#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
字幕自动打轴独立 UI 版 —— 锚点增强·强制断行重构版
- 强制断行分段：最高优先级，完全按原始文稿换行划分段落
- 字符锚点 + 时间密度锚点用于微调
Copyright 2026 光影的故事2018
"""

import sys, os, re, time, shutil, gc, threading, subprocess, logging, tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union

CURRENT_DIR = Path(__file__).parent.absolute()
if (CURRENT_DIR.parent / "pretrained_models").exists() or (CURRENT_DIR.parent / "preset").exists():
    PROJECT_ROOT = CURRENT_DIR.parent
else:
    PROJECT_ROOT = CURRENT_DIR
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import gradio as gr
    import torch
    import numpy as np
    import soundfile as sf
    from faster_whisper import WhisperModel
except ImportError as e:
    print(f"缺少基础依赖库: {e}")
    sys.exit(1)

# ============= 修复 whisperx 导入 =============
WHISPERX_ALIGN_AVAILABLE = False
try:
    import whisperx.alignment
    from whisperx.alignment import load_align_model, align
    WHISPERX_ALIGN_AVAILABLE = True
except ImportError:
    print("警告: 未找到 whisperx.alignment 模块，精细对齐不可用")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============ 工具函数 ============
MAX_OUTPUT_TEXT_LENGTH = 20000
current_max_output_length = MAX_OUTPUT_TEXT_LENGTH

def safe_text(text: str, max_len: int = None) -> str:
    if max_len is None:
        max_len = current_max_output_length
    if not text:
        return ""
    if len(text) > max_len:
        if text.strip().startswith("1\n"):
            return text[:max_len] + "\n\n[注意] 返回的字幕过长已截断，完整结果已保存至 output/字幕自动打轴 目录。"
        return text[:max_len] + "\n\n[注意] 返回内容过长已截断，完整结果已保存至 output/字幕自动打轴 目录。"
    return text

def seconds_to_srt_time(seconds: float) -> str:
    total_ms = round(seconds * 1000)
    hours = total_ms // 3600000
    minutes = (total_ms % 3600000) // 60000
    secs = (total_ms % 60000) // 1000
    millis = total_ms % 1000
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

def normalize_text_for_alignment(text: str, granularity: str) -> str:
    if granularity == "char":
        return re.sub(r'[^\w\u4e00-\u9fff]', '', text, flags=re.UNICODE)
    else:
        return re.sub(r'[^\w\s\u4e00-\u9fff]', '', text, flags=re.UNICODE).strip()

def _ensure_monotonic(sentences: List[Dict]) -> List[Dict]:
    if not sentences:
        return []
    result = [sentences[0].copy()]
    for i in range(1, len(sentences)):
        s = sentences[i].copy()
        if s["start"] < result[-1]["end"]:
            s["start"] = result[-1]["end"]
        if s["end"] <= s["start"]:
            s["end"] = s["start"] + 0.1
        result.append(s)
    return result

# ---------- 时间密度锚点（微调）----------
def _time_density_split_points(items: List[Dict], start_idx: int, end_idx: int, min_gap: float = 0.18) -> List[int]:
    split_indices = []
    for i in range(start_idx, end_idx - 1):
        gap = items[i + 1]["start"] - items[i]["end"]
        if gap > min_gap:
            split_indices.append(i + 1)
    return split_indices

def _refine_by_density(sentence: Dict, items: List[Dict], s_idx: int, e_idx: int,
                       boundary_window: int = 3, min_gap: float = 0.18) -> Dict:
    if e_idx - s_idx <= 1:
        return sentence
    splits = _time_density_split_points(items, s_idx, e_idx, min_gap=min_gap)
    if not splits:
        return sentence
    start = sentence["start"]
    end = sentence["end"]
    for sp in splits:
        if abs(sp - s_idx) <= boundary_window:
            start = items[sp]["start"]
            break
    for sp in splits:
        if abs(sp - e_idx) <= boundary_window:
            end = items[sp]["end"] if sp < len(items) else items[-1]["end"]
            break
    if end <= start:
        end = start + 0.1
    return {"start": start, "end": end, "text": sentence["text"]}

# ---------- 锚点组合（字符锚点 + 密度锚点）----------
def anchor_refine_sentences(
    sentences: List[Dict],
    aligned_items: List[Dict],
    primary_language: str,
    anchor_char_count: int = 3,
    use_anchor_start: bool = False,
    use_anchor_end: bool = False,
    use_anchor_mean: bool = False,
    granularity: str = "char",
    enable_density_anchor: bool = True,
    density_min_gap: float = 0.18,
    density_boundary_window: int = 3,
) -> List[Dict]:
    """字符锚点 + 密度锚点微调"""
    if not aligned_items:
        return sentences

    sentence_ranges = []
    cur_idx = 0
    n_items = len(aligned_items)
    for sent in sentences:
        t_start = sent["start"]
        t_end = sent["end"]
        while cur_idx < n_items and aligned_items[cur_idx]["end"] < t_start:
            cur_idx += 1
        start_idx = cur_idx
        while cur_idx < n_items and aligned_items[cur_idx]["start"] <= t_end:
            cur_idx += 1
        end_idx = cur_idx
        if start_idx >= end_idx:
            if sentence_ranges:
                prev_end_idx = sentence_ranges[-1][1]
                start_idx = min(prev_end_idx, n_items - 1)
                end_idx = min(start_idx + 1, n_items)
            else:
                start_idx = 0
                end_idx = min(1, n_items)
        sentence_ranges.append((start_idx, end_idx))

    is_cjk = primary_language in ("zh", "ja", "ko")
    def is_anchor_unit(text: str) -> bool:
        if is_cjk:
            return bool(re.search(r'[\u4e00-\u9fff]', text))
        else:
            return bool(re.search(r'[A-Za-z]', text))

    refined_char = []
    for i, (sent, (s_idx, e_idx)) in enumerate(zip(sentences, sentence_ranges)):
        seg_items = aligned_items[s_idx:e_idx]
        if not seg_items:
            refined_char.append(sent.copy())
            continue
        seg_text = sent["text"]
        anchor_indices = [j for j, item in enumerate(seg_items) if is_anchor_unit(item["word"])]
        start_default = seg_items[0]["start"]
        end_default = seg_items[-1]["end"]
        if not anchor_indices:
            refined_char.append({"start": start_default, "end": end_default, "text": seg_text})
            continue
        n = min(anchor_char_count, len(anchor_indices))
        front_idx = anchor_indices[:n]
        back_idx = anchor_indices[-n:]
        start_front = sum(seg_items[k]["start"] for k in front_idx) / n
        end_front = sum(seg_items[k]["end"] for k in front_idx) / n
        start_back = sum(seg_items[k]["start"] for k in back_idx) / n
        end_back = sum(seg_items[k]["end"] for k in back_idx) / n
        if use_anchor_mean:
            seg_start = (start_front + start_back) / 2
            seg_end = (end_front + end_back) / 2
        else:
            seg_start = start_front if use_anchor_start else start_default
            seg_end = end_back if use_anchor_end else end_default
        if seg_end <= seg_start:
            seg_end = seg_start + 0.5
        refined_char.append({"start": seg_start, "end": seg_end, "text": seg_text})

    if enable_density_anchor:
        refined = []
        for i, (sent_char, (s_idx, e_idx)) in enumerate(zip(refined_char, sentence_ranges)):
            refined.append(_refine_by_density(sent_char, aligned_items, s_idx, e_idx,
                                              boundary_window=density_boundary_window,
                                              min_gap=density_min_gap))
    else:
        refined = refined_char

    return _ensure_monotonic(refined)

# ============ 字符/词对齐算法 ============
def force_align_char_level(reference_text, transcribed_words, audio_duration=None):
    if not transcribed_words:
        return []
    ref_chars = [ch for ch in reference_text if ch.strip() and (ch.isalnum() or '\u4e00' <= ch <= '\u9fff')]
    if not ref_chars:
        return []
    hyp_chars = []
    char_to_word_idx = []
    for w_idx, w in enumerate(transcribed_words):
        for ch in w["word"]:
            if ch.strip() and (ch.isalnum() or '\u4e00' <= ch <= '\u9fff'):
                hyp_chars.append(ch)
                char_to_word_idx.append(w_idx)
    if not char_to_word_idx:
        default_end = audio_duration if audio_duration else 1.0
        return [{"word": ch, "start": 0.0, "end": default_end} for ch in ref_chars]
    hyp_idx = 0
    match_map = []
    for r_char in ref_chars:
        found = False
        for offset in range(30):
            check_idx = hyp_idx + offset
            if check_idx < len(hyp_chars) and hyp_chars[check_idx] == r_char:
                match_map.append(check_idx)
                hyp_idx = check_idx + 1
                found = True
                break
        if not found:
            match_map.append(-1)
    aligned = []
    prev_time = transcribed_words[0]["start"] if transcribed_words else 0.0
    for i, r_char in enumerate(ref_chars):
        matched_hyp_idx = match_map[i]
        if matched_hyp_idx != -1 and matched_hyp_idx < len(char_to_word_idx):
            w_idx = char_to_word_idx[matched_hyp_idx]
            start_t = transcribed_words[w_idx]["start"]
            end_t = transcribed_words[w_idx]["end"]
            count_in_word = 1
            total_in_word = 1
            for j in range(i - 1, -1, -1):
                if match_map[j] != -1 and match_map[j] < len(char_to_word_idx) and char_to_word_idx[match_map[j]] == w_idx:
                    count_in_word += 1
                else:
                    break
            for j in range(i + 1, len(ref_chars)):
                if match_map[j] != -1 and match_map[j] < len(char_to_word_idx) and char_to_word_idx[match_map[j]] == w_idx:
                    total_in_word += 1
                else:
                    break
            word_duration = end_t - start_t
            char_duration = max(word_duration / total_in_word, 0.02) if total_in_word > 0 else 0.02
            char_start = start_t + (count_in_word - 1) * char_duration
            char_end = char_start + char_duration
            aligned.append({"word": r_char, "start": char_start, "end": char_end})
            prev_time = char_end
        else:
            next_time = transcribed_words[-1]["end"] if transcribed_words else (audio_duration or 1.0)
            for j in range(i + 1, len(ref_chars)):
                if match_map[j] != -1 and match_map[j] < len(char_to_word_idx):
                    next_time = transcribed_words[char_to_word_idx[match_map[j]]]["start"]
                    break
            if next_time <= prev_time:
                next_time = prev_time + 0.1
            mid_time = (prev_time + next_time) / 2.0
            avg_dur = min(0.05, (next_time - prev_time) * 0.4)
            aligned.append({"word": r_char, "start": mid_time - avg_dur/2, "end": mid_time + avg_dur/2})
            prev_time = mid_time
    return aligned

def force_align_word_level(reference_text, transcribed_words, audio_duration=None):
    if not transcribed_words:
        return []
    ref_words = reference_text.split()
    if not ref_words:
        return []
    hyp_words = [w["word"] for w in transcribed_words]
    aligned = []
    hyp_idx = 0
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
            if not ref_clean or not hyp_clean:
                continue
            if ref_clean == hyp_clean:
                w_idx = hyp_idx + offset
                aligned.append({"word": ref_w, "start": transcribed_words[w_idx]["start"], "end": transcribed_words[w_idx]["end"]})
                hyp_idx = w_idx + 1
                found = True
                break
            else:
                shorter_len = min(len(ref_clean), len(hyp_clean))
                longer_len = max(len(ref_clean), len(hyp_clean))
                len_ratio = shorter_len / longer_len if longer_len > 0 else 0
                if len_ratio >= 0.5 and shorter_len >= 2:
                    if ref_clean in hyp_clean or hyp_clean in ref_clean:
                        w_idx = hyp_idx + offset
                        aligned.append({"word": ref_w, "start": transcribed_words[w_idx]["start"], "end": transcribed_words[w_idx]["end"]})
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

def match_paragraphs_to_aligned(aligned_chars, norm_paragraphs, original_paragraphs):
    # 保持原逻辑不变
    if not aligned_chars or not norm_paragraphs:
        return []
    aligned_text = "".join([c["word"] for c in aligned_chars])
    n_items = len(aligned_chars)
    n_text = len(aligned_text)
    cum_len = [0]
    for p in norm_paragraphs:
        cum_len.append(cum_len[-1] + len(p))
    if aligned_text == "".join(norm_paragraphs) and n_text == n_items:
        sentences = []
        for i, para in enumerate(original_paragraphs):
            s_idx = cum_len[i]
            e_idx = cum_len[i + 1]
            s_idx = max(0, min(s_idx, n_items - 1))
            e_idx = max(s_idx + 1, min(e_idx, n_items))
            seg = aligned_chars[s_idx:e_idx]
            t_start = seg[0]["start"]
            t_end = seg[-1]["end"]
            if t_end <= t_start:
                t_end = t_start + 0.1
            sentences.append({"start": t_start, "end": t_end, "text": para})
        return _ensure_monotonic(sentences)
    non_empty_items = [(i, np_, op) for i, (np_, op) in enumerate(zip(norm_paragraphs, original_paragraphs)) if np_]
    if not non_empty_items:
        return []
    matched_ranges = {}
    search_pos = 0
    for orig_idx, norm_p, orig_p in non_empty_items:
        para_len = len(norm_p)
        if para_len > n_text:
            matched_ranges[orig_idx] = (0, n_text)
            continue
        best_start = -1
        best_ratio = -1.0
        margin_back = max(20, para_len)
        margin_fwd = max(100, para_len * 3)
        w_start = max(0, search_pos - margin_back)
        w_end_max = n_text - para_len
        if w_end_max < 0:
            w_end_max = 0
        w_end = min(w_end_max, search_pos + margin_fwd)
        expected_pos = cum_len[orig_idx] if orig_idx < len(cum_len) else search_pos
        if w_end < w_start:
            best_start = max(0, min(expected_pos, n_text - para_len)) if para_len <= n_text else 0
            best_ratio = 0.0
        else:
            for s in range(w_start, w_end + 1):
                seg = aligned_text[s:s + para_len]
                if len(seg) < para_len:
                    continue
                score = sum(1 for a, b in zip(seg, norm_p) if a == b)
                ratio = score / para_len
                dist = abs(s - expected_pos)
                if ratio > best_ratio or (ratio == best_ratio and best_start != -1 and dist < abs(best_start - expected_pos)):
                    best_ratio = ratio
                    best_start = s
        if best_start == -1:
            best_start = max(0, min(search_pos, n_text - para_len))
            best_ratio = 0.0
        end_pos = min(best_start + para_len, n_text)
        matched_ranges[orig_idx] = (best_start, end_pos)
        search_pos = end_pos
    sentences = []
    for i in range(len(original_paragraphs)):
        if i in matched_ranges:
            s_c, e_c = matched_ranges[i]
            s_idx = max(0, min(s_c, n_items - 1))
            e_idx = max(s_idx + 1, min(e_c, n_items))
            seg = aligned_chars[s_idx:e_idx]
            if seg:
                t_start = seg[0]["start"]
                t_end = seg[-1]["end"]
            else:
                prev_end = sentences[-1]["end"] if sentences else 0.0
                t_start = prev_end
                t_end = prev_end + 0.3
        else:
            prev_end = sentences[-1]["end"] if sentences else 0.0
            next_start = None
            for k in range(i + 1, len(original_paragraphs)):
                if k in matched_ranges:
                    sc, _ = matched_ranges[k]
                    nidx = max(0, min(sc, n_items - 1))
                    next_start = aligned_chars[nidx]["start"]
                    break
            if next_start is not None and next_start > prev_end:
                t_start = prev_end
                t_end = (prev_end + next_start) / 2
            else:
                t_start = prev_end
                t_end = prev_end + 0.3
        if t_end <= t_start:
            t_end = t_start + 0.1
        sentences.append({"start": t_start, "end": t_end, "text": original_paragraphs[i]})
    return _ensure_monotonic(sentences)

def match_word_paragraphs_to_aligned(aligned_words, norm_paragraphs, original_paragraphs):
    # 保持原逻辑
    if not aligned_words or not norm_paragraphs:
        return []
    n_aligned = len(aligned_words)
    aligned_word_texts = [w["word"] for w in aligned_words]
    aligned_full_text = " ".join(aligned_word_texts)
    non_empty_items = [(i, np_.strip(), op) for i, (np_, op) in enumerate(zip(norm_paragraphs, original_paragraphs)) if np_.strip()]
    if not non_empty_items:
        return []
    matched_word_ranges = {}
    search_char_pos = 0
    for orig_idx, norm_p, orig_p in non_empty_items:
        para_len = len(norm_p)
        best_start = -1
        best_ratio = 0.0
        margin_back = max(10, para_len)
        margin_fwd = max(80, para_len * 3)
        w_start = max(0, search_char_pos - margin_back)
        w_end_max = len(aligned_full_text) - para_len
        if w_end_max < 0:
            w_end_max = 0
        w_end = min(w_end_max, search_char_pos + margin_fwd)
        if w_end < w_start:
            best_start = search_char_pos
            best_ratio = 0.0
        else:
            for s in range(w_start, w_end + 1):
                seg = aligned_full_text[s:s + para_len]
                if len(seg) < para_len:
                    continue
                score = sum(1 for a, b in zip(seg, norm_p) if a == b)
                ratio = score / para_len if para_len > 0 else 0
                if ratio > best_ratio or (ratio == best_ratio and best_start != -1 and abs(s - search_char_pos) < abs(best_start - search_char_pos)):
                    best_ratio = ratio
                    best_start = s
        if best_start == -1:
            best_start = search_char_pos
        end_char_pos = min(best_start + para_len, len(aligned_full_text))
        prefix = aligned_full_text[:best_start]
        word_start = prefix.count(' ')
        suffix = aligned_full_text[:end_char_pos]
        word_end = suffix.count(' ')
        if (end_char_pos <= len(aligned_full_text) and end_char_pos > 0 and aligned_full_text[end_char_pos - 1] != ' '):
            word_end = word_end + 1
        word_end = max(word_start + 1, word_end)
        word_start = max(0, min(word_start, n_aligned - 1))
        word_end = max(word_start + 1, min(word_end, n_aligned))
        matched_word_ranges[orig_idx] = (word_start, word_end)
        search_char_pos = end_char_pos
    sentences = []
    for i in range(len(original_paragraphs)):
        if i in matched_word_ranges:
            ws, we = matched_word_ranges[i]
            seg = aligned_words[ws:we]
            t_start = seg[0]["start"]
            t_end = seg[-1]["end"]
        else:
            prev_end = sentences[-1]["end"] if sentences else 0.0
            next_start = None
            for k in range(i + 1, len(original_paragraphs)):
                if k in matched_word_ranges:
                    ws_k, _ = matched_word_ranges[k]
                    ws_k = max(0, min(ws_k, n_aligned - 1))
                    next_start = aligned_words[ws_k]["start"]
                    break
            if next_start is not None and next_start > prev_end:
                t_start = prev_end
                t_end = (prev_end + next_start) / 2
            else:
                t_start = prev_end
                t_end = prev_end + 0.3
        if t_end <= t_start:
            t_end = t_start + 0.1
        sentences.append({"start": t_start, "end": t_end, "text": original_paragraphs[i]})
    return _ensure_monotonic(sentences)

def generate_merged_srt(aligned_chars, sentences, paragraphs, merge_punctuations, merge_max_words, merge_max_chars, merge_max_duration, merge_by_newline, merge_by_punc, merge_by_silence, merge_by_wordcount, merge_by_charcount, merge_by_duration, silence_threshold, align_granularity="char"):
    if merge_by_newline:
        return sentences_to_srt(sentences)
    punc_set = set(merge_punctuations) if merge_punctuations else set()
    if merge_by_punc and punc_set and align_granularity == "char":
        has_punc = any(c["word"] in punc_set for c in aligned_chars[:500])
        if not has_punc:
            return sentences_to_srt(sentences)
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
            gap = aligned_chars[i + 1]["start"] - ch_info["end"]
            if gap > 0 and gap > silence_threshold:
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
            merged_segments.append({"start": current_start, "end": ch_info["end"], "text": text_so_far.strip()})
            current_chars = []
            current_start = None
    if current_chars:
        text = "".join([c["word"] for c in current_chars]).strip()
        if text:
            merged_segments.append({"start": current_chars[0]["start"], "end": current_chars[-1]["end"], "text": text})
    return sentences_to_srt(merged_segments)

# ============ FFmpeg ============
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

def get_ffprobe_path():
    ffmpeg_dir = os.path.dirname(FFMPEG_PATH)
    if sys.platform == "win32":
        ffprobe_name = "ffprobe.exe"
    else:
        ffprobe_name = "ffprobe"
    candidate = os.path.join(ffmpeg_dir, ffprobe_name)
    if os.path.isfile(candidate):
        return candidate
    sys_ffprobe = shutil.which("ffprobe")
    return sys_ffprobe if sys_ffprobe else "ffprobe"

def get_audio_duration_robust(audio_path: str) -> Optional[float]:
    try:
        return sf.info(audio_path).duration
    except Exception:
        pass
    try:
        ffprobe_cmd = get_ffprobe_path()
        cmd = [ffprobe_cmd, "-v", "error", "-show_entries", "format=duration",
               "-of", "default=noprint_wrappers=1:nokey=1", audio_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception:
        pass
    try:
        data, sr = sf.read(audio_path)
        return len(data) / sr
    except Exception:
        return None

# ============ 缓存与离线设置 ============
def setup_offline_env():
    root_cache = PROJECT_ROOT / "cache"
    root_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(root_cache))
    align_cache = PROJECT_ROOT / "pretrained_models" / "cache"
    align_cache.mkdir(parents=True, exist_ok=True)
    return align_cache

ALIGN_CACHE_DIR = setup_offline_env()

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

# ============ 模型管理器 ============
class AlignModelManager:
    def __init__(self):
        self.model = None
        self.current_model_name = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "int8_float32"
        self.align_model = None
        self.align_metadata = None
        self.align_model_lang = None
        self.lock = threading.RLock()
        self.keep_align_model_loaded = False

    def get_local_models(self):
        models = []
        models_dir = PROJECT_ROOT / "pretrained_models"
        if not models_dir.exists():
            return models
        for item in models_dir.iterdir():
            if not item.is_dir():
                continue
            if (item / "model.bin").exists() or (item / "config.json").exists() or (item / "pytorch_model.bin").exists():
                models.append((item.name, str(item)))
        return models

    def get_local_align_models(self):
        models = []
        models_dir = PROJECT_ROOT / "pretrained_models"
        if not models_dir.exists():
            return models
        for item in models_dir.iterdir():
            if not item.is_dir():
                continue
            if "wav2vec2" in item.name.lower() or "xlsr" in item.name.lower():
                if ((item / "pytorch_model.bin").exists() or (item / "model.bin").exists() or (item / "config.json").exists()):
                    models.append((item.name, str(item)))
        return models

    def load_model(self, model_size, device, compute_type):
        with self.lock:
            if (self.model is not None and self.current_model_name == model_size and
                self.device == device and self.compute_type == compute_type):
                return True, f"模型 {model_size} 已加载"
            local_models = self.get_local_models()
            model_path = None
            for disp, path in local_models:
                if disp == model_size:
                    model_path = path
                    break
            local_only = bool(model_path)
            model_name_or_path = model_path if model_path else model_size
            try:
                self.model = WhisperModel(model_name_or_path, device=device,
                                          compute_type=compute_type,
                                          local_files_only=local_only)
                self.current_model_name = model_size
                self.device = device
                self.compute_type = compute_type
                return True, f"模型 {model_size} 加载成功"
            except Exception as e:
                return False, f"加载失败: {e}"

    def transcribe_with_segments(self, audio_path, language=None, beam_size=5,
                                  vad_filter=True, vad_parameters=None, initial_prompt=None):
        with self.lock:
            if self.model is None:
                return None, "模型未加载"
            try:
                segments, info = self.model.transcribe(
                    audio_path, language=language, beam_size=beam_size,
                    vad_filter=vad_filter,
                    vad_parameters=vad_parameters if vad_filter else None,
                    word_timestamps=True, initial_prompt=initial_prompt)
            except Exception as e:
                if vad_filter and ("vad" in str(e).lower() or "offline" in str(e).lower()):
                    logger.warning(f"VAD 失败，回退无 VAD。错误: {e}")
                    segments, info = self.model.transcribe(
                        audio_path, language=language, beam_size=beam_size,
                        vad_filter=False, word_timestamps=True,
                        initial_prompt=initial_prompt)
                else:
                    return None, str(e)
            seg_list = []
            for seg in segments:
                seg_dict = {"start": seg.start, "end": seg.end, "text": seg.text.strip()}
                if seg.words:
                    seg_dict["words"] = [{"word": w.word, "start": w.start, "end": w.end} for w in seg.words]
                seg_list.append(seg_dict)
            return {"language": info.language, "segments": seg_list}, None

    def load_align_model(self, language_code, device, model_name=None, model_dir=None):
        with self.lock:
            if not WHISPERX_ALIGN_AVAILABLE:
                raise RuntimeError("whisperx.align 不可用")
            cache_key = f"{language_code}_{model_name}_{device}"
            if self.keep_align_model_loaded and self.align_model is not None:
                if self.align_model_lang == cache_key:
                    return self.align_model, self.align_metadata
            if not self.keep_align_model_loaded:
                if self.align_model is not None:
                    del self.align_model
                    self.align_model = None
                    self.align_metadata = None
                    self._clean_gpu_memory()
            self.align_model, self.align_metadata = load_align_model(
                language_code=language_code, device=device,
                model_name=model_name, model_dir=model_dir)
            self.align_model_lang = cache_key
            return self.align_model, self.align_metadata

    def unload_align_model(self):
        with self.lock:
            if self.align_model is not None and not self.keep_align_model_loaded:
                del self.align_model
                self.align_model = None
                self.align_metadata = None
                self.align_model_lang = None
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

def get_system_status(align_model_info=""):
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
    if manager.keep_align_model_loaded:
        lines.append("对齐模型保持: 是")
    return "\n".join(lines)

def extract_words_from_result(result, align_granularity, use_whisperx_align):
    words = []
    for seg in result.get("segments", []):
        if "words" not in seg or not seg["words"]:
            continue
        seg_words = seg["words"]
        for i, w in enumerate(seg_words):
            if use_whisperx_align and align_granularity == "char" and "chars" in w:
                chars_list = w["chars"]
                valid_chars = [c for c in chars_list if "char" in c]
                if not valid_chars:
                    words.append({"word": w.get("word", ""), "start": w.get("start", 0.0), "end": w.get("end", 0.0)})
                    continue
                if all("start" in c and "end" in c for c in valid_chars):
                    for c in valid_chars:
                        words.append({"word": c["char"], "start": c["start"], "end": c["end"]})
                else:
                    word_dur = w["end"] - w["start"]
                    char_dur = word_dur / len(valid_chars) if len(valid_chars) > 0 else word_dur
                    for idx, c in enumerate(valid_chars):
                        words.append({"word": c["char"], "start": w["start"] + idx * char_dur, "end": w["start"] + (idx+1) * char_dur})
            else:
                if "start" in w and "end" in w and "word" in w:
                    words.append({"word": w["word"], "start": w["start"], "end": w["end"]})
                else:
                    prev_word = seg_words[i - 1] if i > 0 else None
                    next_word = seg_words[i + 1] if i < len(seg_words) - 1 else None
                    start = (prev_word["end"] if prev_word and "end" in prev_word else seg.get("start", 0.0))
                    end = (next_word["start"] if next_word and "start" in next_word else seg.get("end", start + 0.01))
                    word_text = w.get("word", "")
                    if word_text:
                        words.append({"word": word_text, "start": start, "end": end})
    return words

def safe_audio_path(audio_input):
    if audio_input is None:
        return None
    if isinstance(audio_input, str):
        return audio_input
    if isinstance(audio_input, tuple):
        return audio_input[0] if len(audio_input) > 0 else None
    if isinstance(audio_input, dict):
        return audio_input.get("name") or audio_input.get("path")
    return None

# ============ 核心对齐流程 ============
def run_alignment(
    audio_file, primary_text, secondary_text, secondary_lang, enable_dual,
    model_size, device, compute_type, primary_lang, beam_size,
    vad_filter, vad_threshold, vad_min_speech, vad_min_silence,
    hotwords, align_sync_lang, align_model_manual, align_granularity,
    merge_punctuations, merge_max_words, merge_max_chars, merge_max_duration,
    merge_silence_threshold, merge_by_punc, merge_by_silence,
    merge_by_wordcount, merge_by_charcount, merge_by_duration,
    merge_by_newline, keep_align_loaded,
    force_preprocess,
    anchor_enable_start, anchor_enable_end, anchor_enable_mean,
    anchor_char_count,
    anchor_density_enable, anchor_forced_linebreak_enable,
    density_min_gap, density_boundary_window, linebreak_window_frames,
    progress=gr.Progress(),
):
    if audio_file is None:
        return "错误: 请上传音频文件", "", "", "", "", "", "", get_system_status()
    if not primary_text or not primary_text.strip():
        return "错误: 请粘贴主文稿", "", "", "", "", "", "", get_system_status()

    manager.keep_align_model_loaded = keep_align_loaded
    temp_files = []
    original_filename = None

    try:
        audio_path = safe_audio_path(audio_file)
        if not audio_path or not os.path.exists(audio_path):
            return "错误: 无法获取有效的音频文件路径", "", "", "", "", "", "", get_system_status()
        original_filename = Path(audio_path).stem

        if force_preprocess:
            tmp = tempfile.NamedTemporaryFile(suffix="_16k_mono.wav", delete=False)
            tmp.close()
            temp_preprocessed = tmp.name
            cmd = [FFMPEG_PATH, "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", temp_preprocessed]
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                temp_files.append(temp_preprocessed)
                audio_path = temp_preprocessed
                logger.info("音频已预处理为 16kHz 单声道")
            except Exception as e:
                logger.warning(f"音频预处理失败: {e}，使用原始音频")
                try:
                    os.unlink(temp_preprocessed)
                except Exception:
                    pass

        # ---- 对齐模型选择（离线优先） ----
        local_align_models = manager.get_local_align_models()
        use_whisperx_align = False
        align_model_name_for_load = None
        align_model_dir_for_load = str(ALIGN_CACHE_DIR)
        align_model_display = ""

        def match_local_by_lang(lang: str) -> Optional[str]:
            lang = lang.strip().lower()
            for disp, path in local_align_models:
                d = disp.lower()
                if re.search(rf'(?:^|[_-]){re.escape(lang)}(?:[_-]|$)', d):
                    return path
            lang_map = {"zh": ["chinese", "mandarin"], "en": ["english"], "ja": ["japanese"]}
            for kw in lang_map.get(lang, [lang]):
                for disp, path in local_align_models:
                    if kw in disp.lower():
                        return path
            return None

        if align_model_manual and align_model_manual != "无（使用默认）":
            for disp, path in local_align_models:
                if disp == align_model_manual:
                    align_model_path = path
                    align_model_display = f"{disp} (本地)"
                    break
            else:
                align_model_path = align_model_manual
                align_model_display = f"{align_model_manual} (在线)"
            use_whisperx_align = WHISPERX_ALIGN_AVAILABLE
        elif align_sync_lang and primary_lang and primary_lang != "auto":
            local_path = match_local_by_lang(primary_lang)
            if local_path:
                align_model_path = local_path
                align_model_display = f"{Path(local_path).name} (本地)"
                use_whisperx_align = WHISPERX_ALIGN_AVAILABLE
            else:
                online_id = LANGUAGE_ALIGN_MODEL_MAP.get(primary_lang.strip().lower())
                logger.warning(f"未在本地找到语言 '{primary_lang}' 的对齐模型，请下载放入 pretrained_models/。在线 ID: {online_id}")
                align_model_display = f"本地无 '{primary_lang}' 模型，已降级"
                use_whisperx_align = False
        else:
            align_model_display = "未启用精细对齐"
            use_whisperx_align = False

        if use_whisperx_align and align_model_path:
            if os.path.isdir(align_model_path):
                align_model_name_for_load = align_model_path
            else:
                align_model_name_for_load = align_model_path

        system_info = get_system_status(align_model_display)

        # ---- 加载ASR模型并转写 ----
        progress(0.1, desc="加载ASR模型...")
        success, msg = manager.load_model(model_size, device, compute_type)
        if not success:
            return f"错误: {msg}", "", "", "", "", "", "", system_info

        progress(0.3, desc="转写音频...")
        asr_language = None if primary_lang == "auto" else primary_lang
        initial_prompt = hotwords if hotwords and hotwords.strip() else None
        vad_parameters = {
            "onset": vad_threshold, "offset": vad_threshold,
            "min_speech_duration_ms": vad_min_speech,
            "min_silence_duration_ms": vad_min_silence,
        } if vad_filter else None

        result, err = manager.transcribe_with_segments(
            audio_path, language=asr_language, beam_size=beam_size,
            vad_filter=vad_filter, vad_parameters=vad_parameters,
            initial_prompt=initial_prompt)
        if err:
            return f"错误: 转写失败 - {err}", "", "", "", "", "", "", system_info
        original_result = result.copy()

        # ---- 精细对齐 ----
        if use_whisperx_align:
            progress(0.6, desc=f"加载对齐模型: {align_model_display}...")
            try:
                align_model, align_metadata = manager.load_align_model(
                    language_code=asr_language or result.get("language", "en"),
                    device=device,
                    model_name=align_model_name_for_load,
                    model_dir=align_model_dir_for_load)
                progress(0.7, desc="执行精细对齐...")
                aligned_result = align(
                    transcript=result["segments"], model=align_model,
                    align_model_metadata=align_metadata, audio=audio_path,
                    device=device, return_char_alignments=(align_granularity == "char"))
                result = aligned_result
                if not manager.keep_align_model_loaded:
                    manager.unload_align_model()
            except Exception as e:
                logger.warning(f"whisperx.align 失败，回退简单算法: {e}")
                if not manager.keep_align_model_loaded:
                    manager.unload_align_model()
                use_whisperx_align = False
                result = original_result

        words = extract_words_from_result(result, align_granularity, use_whisperx_align)
        if not words:
            return "错误: 未检测到有效的单词时间戳", "", "", "", "", "", "", system_info

        # ---- 文稿匹配（关键分支：强制断行分段） ----
        progress(0.8, desc="匹配主文稿...")
        duration = get_audio_duration_robust(audio_path)
        normalized_primary = normalize_text_for_alignment(primary_text, align_granularity)

        if align_granularity == "char":
            aligned = force_align_char_level(normalized_primary, words, duration)
        else:
            aligned = force_align_word_level(normalized_primary, words, duration)
        if not aligned:
            return "错误: 对齐失败，请检查主文稿与音频是否匹配", "", "", "", "", "", "", system_info

        word_srt = words_to_srt(aligned)

        # ★ 强制断行分段模式
        if anchor_forced_linebreak_enable:
            lines = [line.strip() for line in primary_text.splitlines() if line.strip()]
            if not lines:
                return "错误: 文稿无有效行", "", "", "", "", "", "", system_info
            norm_lines = [normalize_text_for_alignment(line, align_granularity) for line in lines]
            if align_granularity == "char":
                sentences = match_paragraphs_to_aligned(aligned, norm_lines, lines)
            else:
                sentences = match_word_paragraphs_to_aligned(aligned, norm_lines, lines)
            merged_srt = sentences_to_srt(sentences)
            # 强制模式下忽略所有合并规则
        else:
            paragraphs = re.split(r'\n\s*\n', primary_text.strip())
            paragraphs = [p.strip() for p in paragraphs if p.strip()]
            if not paragraphs:
                return "错误: 主文稿无有效段落", "", "", "", "", "", "", system_info
            norm_paragraphs = [normalize_text_for_alignment(p, align_granularity) for p in paragraphs]
            if align_granularity == "char":
                sentences = match_paragraphs_to_aligned(aligned, norm_paragraphs, paragraphs)
            else:
                sentences = match_word_paragraphs_to_aligned(aligned, norm_paragraphs, paragraphs)
            merged_srt = generate_merged_srt(
                aligned, sentences, paragraphs,
                merge_punctuations, merge_max_words, merge_max_chars,
                merge_max_duration, merge_by_newline, merge_by_punc,
                merge_by_silence, merge_by_wordcount, merge_by_charcount,
                merge_by_duration, merge_silence_threshold, align_granularity)

        # ---- 锚点微调（字符锚点 + 密度锚点） ----
        anchor_srt = ""
        anchor_used = anchor_enable_start or anchor_enable_end or anchor_enable_mean
        if anchor_used and aligned:
            if primary_lang != "auto":
                anchor_lang = primary_lang
            else:
                anchor_lang = result.get("language", "en") if isinstance(result, dict) else "en"
            sentences = anchor_refine_sentences(
                sentences, aligned, primary_language=anchor_lang,
                anchor_char_count=anchor_char_count,
                use_anchor_start=anchor_enable_start,
                use_anchor_end=anchor_enable_end,
                use_anchor_mean=anchor_enable_mean,
                granularity=align_granularity,
                enable_density_anchor=anchor_density_enable,
                density_min_gap=density_min_gap,
                density_boundary_window=int(density_boundary_window),
            )
            anchor_srt = sentences_to_srt(sentences)

        sent_srt = sentences_to_srt(sentences)

        # ---- 双语挂载（容错降为1） ----
        dual_srt = ""
        secondary_srt = ""
        warning_msg = ""
        if enable_dual and secondary_text and secondary_text.strip():
            sec_paragraphs = [p.strip() for p in re.split(r'\n\s*\n', secondary_text.strip()) if p.strip()]
            len_diff = abs(len(sec_paragraphs) - len(sentences))
            if len_diff <= 1:
                if len(sec_paragraphs) > len(sentences):
                    sec_paragraphs = sec_paragraphs[:len(sentences)]
                    warning_msg = f"警告: 副文稿段落数多 {len_diff} 段，已截断"
                elif len(sentences) > len(sec_paragraphs):
                    sec_paragraphs += [""] * (len(sentences) - len(sec_paragraphs))
                    warning_msg = f"警告: 副文稿段落数少 {len_diff} 段，已补空行"
                sec_lines, dual_lines = [], []
                for i, (seg, sec_text) in enumerate(zip(sentences, sec_paragraphs), 1):
                    time_str = f"{seconds_to_srt_time(seg['start'])} --> {seconds_to_srt_time(seg['end'])}"
                    sec_lines.extend([str(i), time_str, sec_text, ""])
                    dual_lines.extend([str(i), time_str, seg["text"], sec_text, ""])
                secondary_srt = "\n".join(sec_lines)
                dual_srt = "\n".join(dual_lines)
            else:
                warning_msg = f"警告: 段落数相差 {len_diff} 段（>1），跳过双语生成"

        # 单词级 CJK 警告
        if align_granularity == "word" and primary_lang in ("zh", "ja", "ko"):
            warning_msg += "\n" if warning_msg else ""
            warning_msg += "提示: 单词级对齐不适合中日韩语言，建议使用字符级对齐"

        if warning_msg:
            system_info += f"\n{warning_msg}"

        # ---- 保存 ----
        output_dir = PROJECT_ROOT / "output" / "字幕自动打轴"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = original_filename or "unknown_audio"
        prefix = f"{base_name}_align_{timestamp}"
        safe_lang_tag = ""
        if secondary_lang and secondary_lang.strip():
            lang_clean = re.sub(r'[\\/*?:"<>|]', '', secondary_lang.strip())
            safe_lang_tag = f"_{lang_clean}"
        word_path = output_dir / f"{prefix}_words.srt"
        sent_path = output_dir / f"{prefix}_sentences.srt"
        merged_path = output_dir / f"{prefix}_merged.srt"
        with open(word_path, "w", encoding="utf-8") as f: f.write(word_srt)
        with open(sent_path, "w", encoding="utf-8") as f: f.write(sent_srt)
        with open(merged_path, "w", encoding="utf-8") as f: f.write(merged_srt)

        status = f"对齐完成！\n逐词字幕: {word_path.name}\n整句字幕: {sent_path.name}\n合并字幕: {merged_path.name}\n对齐模型: {align_model_display}"
        if anchor_used and anchor_srt:
            anchor_path = output_dir / f"{prefix}_anchor.srt"
            with open(anchor_path, "w", encoding="utf-8") as f: f.write(anchor_srt)
            status += f"\n锚点字幕: {anchor_path.name}"
        if secondary_srt:
            sec_path = output_dir / f"{prefix}{safe_lang_tag}_secondary.srt"
            with open(sec_path, "w", encoding="utf-8") as f: f.write(secondary_srt)
            status += f"\n副文稿单语: {sec_path.name}"
        if dual_srt:
            dual_path = output_dir / f"{prefix}{safe_lang_tag}_dual.srt"
            with open(dual_path, "w", encoding="utf-8") as f: f.write(dual_srt)
            status += f"\n双语字幕: {dual_path.name}"

        return (status, safe_text(word_srt), safe_text(sent_srt), safe_text(merged_srt),
                safe_text(secondary_srt) if secondary_srt else "",
                safe_text(dual_srt) if dual_srt else "",
                safe_text(anchor_srt) if anchor_srt else "",
                system_info)
    finally:
        for f in temp_files:
            try:
                os.unlink(f)
            except Exception:
                pass

def clear_outputs():
    return "", "", "", "", "", "", "", get_system_status()

def refresh_align_model_list():
    models = manager.get_local_align_models()
    choices = ["无（使用默认）"] + [disp for disp, _ in models]
    return gr.update(choices=choices, value="无（使用默认）")

def toggle_align_model_manual(sync: bool):
    return gr.update(visible=not sync)

def open_output_folder():
    output_dir = PROJECT_ROOT / "output" / "字幕自动打轴"
    output_dir.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        os.startfile(str(output_dir))
    else:
        subprocess.Popen(["xdg-open" if shutil.which("xdg-open") else "open", str(output_dir)])

def set_max_length(val):
    global current_max_output_length
    current_max_output_length = int(val)
    return get_system_status()

# ============ 界面 ============
def create_ui():
    local_models = manager.get_local_models()
    model_choices = [name for name, _ in local_models]
    default_models = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
    model_choices = model_choices + [m for m in default_models if m not in model_choices]
    if not model_choices:
        model_choices = default_models

    local_align = manager.get_local_align_models()
    align_choices = ["无（使用默认）"] + [disp for disp, _ in local_align]

    with gr.Blocks(title="字幕自动打轴", theme=gr.themes.Default()) as demo:
        gr.Markdown("# 字幕自动打轴（中英双语通用）")

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.File(label="选择音频文件", file_types=[".wav", ".mp3", ".m4a", ".flac", ".ogg"])
                audio_preview = gr.Audio(label="音频预览", interactive=False, visible=False)
                force_preprocess_check = gr.Checkbox(label="强制预处理为 16kHz 单声道 (推荐大文件)", value=True)
                primary_text = gr.Textbox(label="主文稿（对齐用）", lines=20, placeholder="粘贴稿子...")
                secondary_text = gr.Textbox(label="副文稿（挂载用，可选）", lines=20, placeholder="翻译稿...")
                with gr.Row():
                    secondary_lang = gr.Textbox(label="副文稿语言标记", value="", scale=1)
                    enable_dual = gr.Checkbox(label="生成双语字幕", value=False, scale=1)

            with gr.Column(scale=2):
                with gr.Row():
                    status_box = gr.Textbox(label="任务状态", value="等待开始", lines=4, interactive=False)
                    system_box = gr.Textbox(label="系统信息", value=get_system_status(), lines=4, interactive=False)

                with gr.Accordion("模型与识别参数", open=True):
                    model_drop = gr.Dropdown(label="ASR模型", choices=model_choices, value=model_choices[0] if model_choices else "medium")
                    with gr.Row():
                        device_drop = gr.Dropdown(label="设备", choices=["cuda", "cpu"], value="cuda" if torch.cuda.is_available() else "cpu")
                        compute_drop = gr.Dropdown(label="计算类型", choices=["int8_float32", "float16", "float32"], value="int8_float32")
                    with gr.Row():
                        primary_lang = gr.Dropdown(label="主语言", choices=["auto", "zh", "en", "ja", "fr", "de", "es", "it", "pt", "nl", "hu"], value="zh")
                        beam_slider = gr.Slider(label="Beam Size", minimum=1, maximum=10, value=5, step=1)
                    hotwords_box = gr.Textbox(label="热词/提示词", lines=2, value="")

                with gr.Accordion("VAD 高级设置", open=False):
                    vad_filter = gr.Checkbox(label="启用 VAD 过滤", value=True)
                    vad_threshold = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="语音检测阈值")
                    vad_min_speech = gr.Slider(100, 1000, value=250, step=50, label="最短语音 (ms)")
                    vad_min_silence = gr.Slider(50, 1000, value=100, step=50, label="最短静音 (ms)")

                with gr.Accordion("对齐模型设置", open=True):
                    align_sync_lang = gr.Checkbox(label="对齐模型跟随主语言自动匹配", value=True)
                    align_model_manual = gr.Dropdown(label="手动选择对齐模型", choices=align_choices, value="无（使用默认）", visible=False)
                    refresh_align_btn = gr.Button("刷新对齐模型列表", size="sm")
                    align_granularity = gr.Radio(label="对齐粒度", choices=[("字符级", "char"), ("单词级", "word")], value="char")
                    keep_align_loaded = gr.Checkbox(label="保持对齐模型加载", value=False)

                # ★ 最醒目的强制断行锚点放在合并规则顶部
                with gr.Accordion("字幕合并规则", open=True):
                    gr.Markdown("### 🔥 强制断行锚点（最高优先级）")
                    anchor_forced_linebreak = gr.Checkbox(
                        label="📌 强制断行分段（按稿子原始换行，忽略所有自动规则）",
                        value=True,
                        info="专为已排版的规整文稿设计，完全依照文稿换行，首尾字定位时间。开启后下方断句选项自动无效。"
                    )
                    gr.Markdown("---")
                    with gr.Row():
                        merge_newline = gr.Checkbox(label="按空行分段", value=True)
                        merge_punc = gr.Checkbox(label="按标点断句", value=True)
                        merge_silence = gr.Checkbox(label="按静音断句", value=True)
                    with gr.Row():
                        merge_wordcount = gr.Checkbox(label="按词数断句", value=True)
                        merge_charcount = gr.Checkbox(label="按字符数断句", value=True)
                        merge_duration = gr.Checkbox(label="按时长断句", value=True)
                    with gr.Row():
                        punc_box = gr.Textbox(label="句末标点", value="，；。！？,;.!?", scale=2)
                        silence_slider = gr.Slider(0.1, 1.0, value=0.3, step=0.05, label="静音阈值 (秒)")
                    with gr.Row():
                        max_words_slider = gr.Slider(5, 50, value=20, step=1, label="最大词数")
                        max_chars_slider = gr.Slider(5, 100, value=30, step=5, label="最大字符数")
                        max_duration_slider = gr.Slider(1.0, 20.0, value=10.0, step=0.5, label="最大时长 (秒)")

                with gr.Accordion("锚点增强 (中英实验性)", open=False):
                    with gr.Row():
                        anchor_start = gr.Checkbox(label="前锚点", value=False)
                        anchor_end = gr.Checkbox(label="后锚点", value=False)
                        anchor_mean = gr.Checkbox(label="前后均值", value=False)
                        anchor_count_slider = gr.Slider(1, 5, value=3, step=1, label="锚点参考单位数")
                    with gr.Row():
                        anchor_density = gr.Checkbox(label="时间密度锚点", value=True,
                                                     info="利用静音间隙抑制边界飘移")
                        # 移除 linebreak 相关，已集成在顶部
                    with gr.Row():
                        density_min_gap_slider = gr.Slider(0.05, 0.5, value=0.18, step=0.01, label="密度最小间隙(秒)")
                        density_boundary_window_slider = gr.Slider(1, 6, value=3, step=1, label="密度边界窗口(单位数)")

                with gr.Accordion("输出控制", open=False):
                    with gr.Row():
                        open_output_btn = gr.Button("打开输出目录", variant="secondary")
                        max_text_len_slider = gr.Slider(2000, 50000, value=current_max_output_length, step=2000, label="界面最大显示字符数")
                    open_output_btn.click(open_output_folder, inputs=None, outputs=None)
                    max_text_len_slider.change(set_max_length, inputs=[max_text_len_slider], outputs=[system_box])

                with gr.Row():
                    run_btn = gr.Button("开始对齐", variant="primary", size="lg")
                    clear_btn = gr.Button("清空")

                with gr.Tabs():
                    with gr.Tab("逐词/逐字 SRT"):
                        word_output = gr.Textbox(label="逐词字幕", lines=20, show_copy_button=True)
                    with gr.Tab("整句 SRT"):
                        sent_output = gr.Textbox(label="整句字幕", lines=20, show_copy_button=True)
                    with gr.Tab("合并字幕"):
                        merged_output = gr.Textbox(label="合并字幕", lines=20, show_copy_button=True)
                    with gr.Tab("锚点增强 SRT"):
                        anchor_output = gr.Textbox(label="锚点增强字幕", lines=20, show_copy_button=True)
                    with gr.Tab("副文稿单语 SRT"):
                        secondary_output = gr.Textbox(label="副文稿字幕", lines=20, show_copy_button=True)
                    with gr.Tab("双语 SRT"):
                        dual_output = gr.Textbox(label="双语字幕", lines=20, show_copy_button=True)

        def update_audio_preview(file_path):
            if file_path:
                return gr.update(value=file_path, visible=True)
            return gr.update(value=None, visible=False)
        audio_input.change(update_audio_preview, inputs=[audio_input], outputs=[audio_preview])
        align_sync_lang.change(toggle_align_model_manual, inputs=[align_sync_lang], outputs=[align_model_manual])

        run_btn.click(
            run_alignment,
            inputs=[
                audio_input, primary_text, secondary_text, secondary_lang, enable_dual,
                model_drop, device_drop, compute_drop, primary_lang, beam_slider,
                vad_filter, vad_threshold, vad_min_speech, vad_min_silence,
                hotwords_box, align_sync_lang, align_model_manual, align_granularity,
                punc_box, max_words_slider, max_chars_slider, max_duration_slider,
                silence_slider, merge_punc, merge_silence,
                merge_wordcount, merge_charcount, merge_duration,
                merge_newline, keep_align_loaded,
                force_preprocess_check,
                anchor_start, anchor_end, anchor_mean, anchor_count_slider,
                anchor_density, anchor_forced_linebreak,
                density_min_gap_slider, density_boundary_window_slider, gr.State(2),  # 占位
            ],
            outputs=[status_box, word_output, sent_output, merged_output, secondary_output, dual_output, anchor_output, system_box],
        )

        clear_btn.click(
            clear_outputs,
            outputs=[status_box, word_output, sent_output, merged_output, secondary_output, dual_output, anchor_output, system_box],
        ).then(
            lambda: [None, None, "", "", "", False],
            outputs=[audio_input, audio_preview, primary_text, secondary_text, secondary_lang, enable_dual],
        )

        refresh_align_btn.click(refresh_align_model_list, outputs=[align_model_manual])

        gr.HTML("""
        <div style="text-align: center; color: #666; font-size: 0.85em; margin-top: 20px;">
        <p>本软件包按"原样"提供，不提供任何明示或暗示的担保。</p>
        <p>更新请关注B站up主：光影的故事2018</p>
        </div>
        """)

    return demo

def main():
    model_root = PROJECT_ROOT / "pretrained_models"
    if not model_root.exists():
        print(f"警告: 模型目录 {model_root} 不存在，请确保模型已下载。")
    demo = create_ui()
    demo.queue(default_concurrency_limit=1)
    ports = [18001, 18002, 18003, 18004, 18005]
    for p in ports:
        try:
            demo.launch(server_name="127.0.0.1", server_port=p, inbrowser=True, show_error=True, max_file_size=500 * 1024 * 1024)
            break
        except OSError:
            print(f"端口 {p} 被占用，尝试下一个...")
            continue
    else:
        print("所有端口均被占用，请手动指定空闲端口。")
        sys.exit(1)

if __name__ == "__main__":
    main()