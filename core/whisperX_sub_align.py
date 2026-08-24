#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
字幕自动打轴独立 UI 版 —— 空行断句 · 原文索引对齐（修复版）
- 空行断句（推荐）：按文稿空行分段（可开关，真实生效）
- 叠加断句规则（标点、静音、字数/词数、时长）进行段内细分
- 字符锚点 + 时间密度锚点用于微调（默认关闭）
- 支持 28 种语言对齐
- 支持批量文件夹自动配对
Copyright 2026 光影的故事2018

"""
import sys, os, re, time, shutil, gc, threading, subprocess, logging, tempfile, copy, atexit, traceback, difflib
from pathlib import Path
from typing import List, Dict, Optional

CURRENT_DIR = Path(__file__).parent.absolute()
if (CURRENT_DIR.parent / "pretrained_models").exists() or (CURRENT_DIR.parent / "preset").exists():
    PROJECT_ROOT = CURRENT_DIR.parent
else:
    PROJECT_ROOT = CURRENT_DIR
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import gradio as gr
    import torch
    import soundfile as sf
    from faster_whisper import WhisperModel
except ImportError as e:
    print(f"缺少基础依赖库: {e}")
    sys.exit(1)

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

_SRT_HEAD_RE = re.compile(r'^\s*\d+\s*\n\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}\s*-->')

def _looks_like_srt(text: str) -> bool:
    """修复：用"序号行 + 时间码行"格式识别 SRT，替代原 "1\n" 魔法字符串判断。"""
    return bool(_SRT_HEAD_RE.match(text))

def safe_text(text: str, max_len: int = None) -> str:
    if max_len is None:
        max_len = current_max_output_length
    if not text:
        return ""
    if len(text) > max_len:
        if _looks_like_srt(text):
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

def normalize_file_input(f):
    """统一解析 Gradio 5 文件输入，返回 (路径, 原始文件名)。"""
    if f is None:
        return None, None
    if isinstance(f, dict):
        path = f.get('name') or f.get('path')
        orig = f.get('orig_name') or f.get('name') or (os.path.basename(str(path)) if path else None)
        return (str(path) if path else None), orig
    if isinstance(f, (list, tuple)):
        if len(f) > 0:
            return normalize_file_input(f[0])
        return None, None
    if hasattr(f, 'orig_name') or hasattr(f, 'path'):
        path = getattr(f, 'path', None) or getattr(f, 'name', None)
        if path is None:
            return None, None
        path = str(path)
        orig = getattr(f, 'orig_name', None) or os.path.basename(path)
        return path, orig
    if isinstance(f, (str, Path)):
        p = str(f)
        return p, os.path.basename(p)
    return None, None

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

# ============ 原文索引对齐核心（本版核心重构） ============
def _core_text(s: str) -> str:
    """匹配用核心文本：去标点、统一小写。"""
    return re.sub(r'[^\w\u4e00-\u9fff]', '', s, flags=re.UNICODE).lower()

def build_alignment_units(primary_text: str, granularity: str, split_paragraphs: bool = True):
    """
    解析主文稿，生成对齐单元序列。
    返回。
    单元保留原文索引，后续可从 full_original 精确切片重建文本（保留标点/空格）。
    """
    text = primary_text.strip()
    if not text:
        return "", [], [], []
    if split_paragraphs:
        paragraphs = [re.sub(r'\s+', ' ', p.strip())
                      for p in re.split(r'\n\s*\n', text) if p.strip()]
    else:
        # 修复：不启用空行分段时整篇作为单段
        paragraphs = [re.sub(r'\s+', ' ', text)]
    if not paragraphs:
        return "", [], [], []
    full_original = "\n\n".join(paragraphs)
    para_spans = []
    pos = 0
    for p in paragraphs:
        para_spans.append((pos, pos + len(p)))
        pos += len(p) + 2
    units = []
    if granularity == "char":
        for i, ch in enumerate(full_original):
            if ch.isalnum() or '\u4e00' <= ch <= '\u9fff':
                units.append({"text": ch, "orig_idx": i, "orig_end": i + 1, "core": ch.lower()})
    else:
        for m in re.finditer(r'\S+', full_original):
            word = m.group(0)
            core = _core_text(word)
            if core:
                units.append({"text": word, "orig_idx": m.start(), "orig_end": m.end(), "core": core})
    return full_original, units, para_spans, paragraphs

def _expand_hyp_words(transcribed_words: List[Dict], granularity: str):
    """把 ASR 词序列展开为匹配核心序列，返回 (cores, owner_idx)。"""
    cores, owners = [], []
    for w_idx, w in enumerate(transcribed_words):
        core = _core_text(w.get("word", ""))
        if not core:
            continue
        if granularity == "char":
            for ch in core:
                cores.append(ch)
                owners.append(w_idx)
        else:
            cores.append(core)
            owners.append(w_idx)
    return cores, owners

def _sequence_match(ref, hyp):
    """返回 match_map: ref_idx -> hyp_idx（未匹配为 -1）。
    短序列用 SequenceMatcher 全局对齐；超长降级为贪心+失配重同步（修复死锁）。"""
    n_ref, n_hyp = len(ref), len(hyp)
    match = [-1] * n_ref
    if n_ref == 0 or n_hyp == 0:
        return match
    if n_ref <= 3000 and n_ref * n_hyp <= 4000000:
        sm = difflib.SequenceMatcher(None, hyp, ref, autojunk=False)
        for b in sm.get_matching_blocks():
            for k in range(b.size):
                match[b.b + k] = b.a + k
        return match
    lookahead = 20
    hi = 0
    misses = 0
    for ri in range(n_ref):
        found = False
        for off in range(lookahead):
            ci = hi + off
            if ci < n_hyp and hyp[ci] == ref[ri]:
                match[ri] = ci
                hi = ci + 1
                found = True
                misses = 0
                break
        if not found:
            misses += 1
            if misses >= lookahead:  # 连续失配：重新同步
                # 修复：在有限窗口内查找 hyp 中下一个与 ref[ri] 相等的位置并回填匹配；
                # 原盲跳推进 misses 可能越过正确同步点，导致长文稿段落整体偏移
                nxt = -1
                for ci in range(hi, min(hi + 200, n_hyp)):
                    if hyp[ci] == ref[ri]:
                        nxt = ci
                        break
                if nxt >= 0:
                    match[ri] = nxt
                    hi = nxt + 1
                else:
                    hi = min(hi + misses, n_hyp)
                misses = 0
    return match

def force_align_units(units: List[Dict], transcribed_words: List[Dict],
                      granularity: str = "char", audio_duration: Optional[float] = None) -> List[Dict]:
    """
    将文稿 units 与 ASR 词时间戳强制对齐。
    返回与 units 等长: [{"word","start","end","orig_idx","orig_end"}]
    """
    n = len(units)
    if n == 0 or not transcribed_words:
        return []
    unit_cores = [u["core"] for u in units]
    hyp_cores, hyp_owners = _expand_hyp_words(transcribed_words, granularity)
    match_map = _sequence_match(unit_cores, hyp_cores)
    n_hyp = len(hyp_cores)

    aligned: List[Optional[Dict]] = [None] * n
    for ui in range(n):
        hi = match_map[ui]
        if hi < 0 or hi >= n_hyp:
            continue
        w = transcribed_words[hyp_owners[hi]]
        if "start" not in w or "end" not in w:
            continue
        u = units[ui]
        if granularity == "char":
            # 同词内字符均分词时长
            before = 0
            j = ui - 1
            while j >= 0 and 0 <= match_map[j] < n_hyp and hyp_owners[match_map[j]] == hyp_owners[hi]:
                before += 1
                j -= 1
            after = 0
            j = ui + 1
            while j < n and 0 <= match_map[j] < n_hyp and hyp_owners[match_map[j]] == hyp_owners[hi]:
                after += 1
                j += 1
            total = before + after + 1
            word_dur = max(w["end"] - w["start"], 0.02)
            char_dur = max(word_dur / total, 0.02)
            st = w["start"] + before * char_dur
            aligned[ui] = {"word": u["text"], "start": st, "end": st + char_dur,
                           "orig_idx": u["orig_idx"], "orig_end": u["orig_end"]}
        else:
            aligned[ui] = {"word": u["text"], "start": w["start"], "end": w["end"],
                           "orig_idx": u["orig_idx"], "orig_end": u["orig_end"]}

    # 未匹配 unit：按区间均分插值
    default_start = transcribed_words[0].get("start", 0.0)
    default_end = transcribed_words[-1].get("end", audio_duration or default_start + 1.0)
    ui = 0
    while ui < n:
        if aligned[ui] is not None:
            ui += 1
            continue
        a = ui
        while ui < n and aligned[ui] is None:
            ui += 1
        b = ui
        prev_t = aligned[a - 1]["end"] if a > 0 and aligned[a - 1] is not None else default_start
        next_t = aligned[b]["start"] if b < n and aligned[b] is not None else default_end
        if next_t <= prev_t:
            next_t = prev_t + 0.05 * (b - a)
        span = max(next_t - prev_t, 0.05 * (b - a))
        per = max(span / (b - a), 0.02)
        for k in range(a, b):
            st = prev_t + (k - a) * per
            u = units[k]
            aligned[k] = {"word": u["text"], "start": st, "end": st + per,
                          "orig_idx": u["orig_idx"], "orig_end": u["orig_end"]}
    return aligned

def build_paragraph_sentences(aligned: List[Dict], para_spans, paragraphs):
    """基于原文索引精确划分段落（替代旧版 O(n^2) 模糊匹配死代码）。
    返回 (sentences, unit_para)。"""
    n = len(aligned)
    n_para = len(paragraphs)
    if n == 0:
        return [], []
    unit_para = [0] * n
    para_idxs = [[] for _ in range(n_para)]  # 单次遍历分桶，避免 O(n_para×n)
    pi = 0
    for i in range(n):
        oi = aligned[i]["orig_idx"]
        while pi < n_para - 1 and oi >= para_spans[pi][1]:
            pi += 1
        unit_para[i] = pi
        para_idxs[pi].append(i)
    sentences = []
    for p in range(n_para):
        idxs = para_idxs[p]
        if idxs:
            t_start = aligned[idxs[0]]["start"]
            t_end = aligned[idxs[-1]]["end"]
            if t_end <= t_start:
                t_end = t_start + 0.1
            sentences.append({"start": t_start, "end": t_end, "text": paragraphs[p]})
        else:
            prev_end = sentences[-1]["end"] if sentences else 0.0
            next_start = None
            for q in range(p + 1, n_para):
                if para_idxs[q]:
                    next_start = aligned[para_idxs[q][0]]["start"]
                    break
            if next_start is not None and next_start > prev_end:
                t_start, t_end = prev_end, (prev_end + next_start) / 2
            else:
                t_start, t_end = prev_end, prev_end + 0.3
            sentences.append({"start": t_start, "end": t_end, "text": paragraphs[p]})
    return _ensure_monotonic(sentences), unit_para

def count_words(text: str) -> int:
    """词数统计：CJK 按字计，西文按空白分词计。"""
    cjk = len(re.findall(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', text))
    latin = len(re.findall(r"[A-Za-z0-9]+(?:['\u2019\-][A-Za-z0-9]+)*", text))
    return cjk + latin

def generate_merged_srt(
    aligned: List[Dict], full_original: str, para_spans,
    merge_punctuations: str, merge_max_words: int, merge_max_chars: int,
    merge_max_duration: float, merge_by_punc: bool, merge_by_silence: bool,
    merge_by_wordcount: bool, merge_by_charcount: bool, merge_by_duration: bool,
    silence_threshold: float, unit_para: List[int],
) -> List[Dict]:  # 修复：原注解 str 与实际返回的 segments 列表不符
    """
    修复版合并逻辑：
    - 文本从 full_original 按原文索引切片重建，标点与空格完整保留；
    - 标点断句：相邻单元之间的原文区间内出现句末标点即在处断开；
    - 段落边界强制断开（不再跨段粘连）；
    - 静音/词数/字符数/时长规则在段内生效；
    - 词数/字符数增量累计（避免每个单元反复切片的 O(n^2) 热路径）；
    - 返回 segments 列表（供双语挂载使用，保证与合并字幕分段一致）。
    """
    if not aligned:
        return []
    punc_set = set(merge_punctuations) if merge_punctuations else set()
    n = len(aligned)

    def text_for_range(a: int, b: int) -> str:
        """units [a, b) 的原文切片：延伸到下一单元起点，句末标点自然归入本句。"""
        start_o = aligned[a]["orig_idx"]
        if b < n and unit_para[b] == unit_para[b - 1]:
            end_o = aligned[b]["orig_idx"]
        else:
            end_o = para_spans[unit_para[b - 1]][1]
        seg = full_original[start_o:end_o]
        return re.sub(r'\s+', ' ', seg).strip()

    merged_segments = []
    seg_start_idx = 0
    seg_cursor = aligned[0]["orig_idx"]  # 已累计到原文中的位置
    cum_chars = 0   # 当前段内非空白字符数（增量）
    cum_words = 0   # 当前段内词数（增量）
    need_count = merge_by_wordcount or merge_by_charcount
    for i in range(n):
        should_split = False
        # 1) 段落边界强制断开
        if i + 1 < n and unit_para[i + 1] != unit_para[i]:
            should_split = True
        # 2) 标点断句（修复：基于原文区间检测，真实生效）
        if not should_split and merge_by_punc and punc_set and i + 1 < n:
            between = full_original[aligned[i]["orig_end"]:aligned[i + 1]["orig_idx"]]
            if any(ch in punc_set for ch in between):
                should_split = True
        # 3) 静音断句
        if not should_split and merge_by_silence and i + 1 < n:
            gap = aligned[i + 1]["start"] - aligned[i]["end"]
            if gap > silence_threshold:
                should_split = True
        # 4) 词数 / 字符数（增量累计到含 i 的原文区间）/ 时长
        if not should_split and (need_count or merge_by_duration):
            if need_count:
                if i + 1 < n and unit_para[i + 1] == unit_para[i]:
                    end_o = aligned[i + 1]["orig_idx"]
                else:
                    end_o = para_spans[unit_para[i]][1]
                chunk = full_original[seg_cursor:end_o]
                cum_chars += len(re.sub(r'\s+', '', chunk))
                cum_words += count_words(chunk)
                seg_cursor = end_o
            if merge_by_wordcount and cum_words >= merge_max_words:
                should_split = True
            if not should_split and merge_by_charcount and cum_chars >= merge_max_chars:
                should_split = True
            if not should_split and merge_by_duration:
                duration = aligned[i]["end"] - aligned[seg_start_idx]["start"]
                if duration >= merge_max_duration:
                    should_split = True
        if should_split:
            text = text_for_range(seg_start_idx, i + 1)
            if text:
                seg_start = aligned[seg_start_idx]["start"]
                seg_end = aligned[i]["end"]
                if seg_end <= seg_start:
                    seg_end = seg_start + 0.1
                merged_segments.append({"start": seg_start, "end": seg_end, "text": text})
            seg_start_idx = i + 1
            if i + 1 < n:
                seg_cursor = aligned[i + 1]["orig_idx"]
            cum_chars = 0
            cum_words = 0
    if seg_start_idx < n:
        text = text_for_range(seg_start_idx, n)
        if text:
            seg_start = aligned[seg_start_idx]["start"]
            seg_end = aligned[n - 1]["end"]
            if seg_end <= seg_start:
                seg_end = seg_start + 0.1
            merged_segments.append({"start": seg_start, "end": seg_end, "text": text})
    return merged_segments

# ---------- 锚点（保留原逻辑） ----------
def _time_density_split_points(items, start_idx, end_idx, min_gap=0.18):
    split_indices = []
    for i in range(start_idx, end_idx - 1):
        gap = items[i + 1]["start"] - items[i]["end"]
        if gap > min_gap:
            split_indices.append(i + 1)
    return split_indices

def _refine_by_density(sentence, items, s_idx, e_idx, boundary_window=3, min_gap=0.18):
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
            # 句尾应落在间隙起点（即上一词条结束之后），而非间隙后词条的结束
            end = items[sp]["start"] if sp < len(items) else items[-1]["end"]
            break
    if end <= start:
        end = start + 0.1
    return {"start": start, "end": end, "text": sentence["text"]}

def anchor_refine_sentences(sentences, aligned_items, primary_language,
                            anchor_char_count=3, use_anchor_start=False, use_anchor_end=False,
                            use_anchor_mean=False,
                            enable_density_anchor=True, density_min_gap=0.18,
                            density_boundary_window=3):
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

    def is_anchor_unit(text):
        if is_cjk:
            return bool(re.search(r'[\u4e00-\u9fff]', text))
        return bool(re.search(r'[A-Za-z]', text))

    refined_char = []
    for sent, (s_idx, e_idx) in zip(sentences, sentence_ranges):
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
        n_anc = min(anchor_char_count, len(anchor_indices))
        front_idx = anchor_indices[:n_anc]
        back_idx = anchor_indices[-n_anc:]
        start_front = sum(seg_items[k]["start"] for k in front_idx) / n_anc
        end_front = sum(seg_items[k]["end"] for k in front_idx) / n_anc
        start_back = sum(seg_items[k]["start"] for k in back_idx) / n_anc
        end_back = sum(seg_items[k]["end"] for k in back_idx) / n_anc
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
        refined = [_refine_by_density(sc, aligned_items, s_idx, e_idx,
                                      boundary_window=density_boundary_window,
                                      min_gap=density_min_gap)
                   for sc, (s_idx, e_idx) in zip(refined_char, sentence_ranges)]
    else:
        refined = refined_char
    return _ensure_monotonic(refined)

# ============ FFmpeg 工具函数 ============
def find_ffmpeg():
    portable_dir = PROJECT_ROOT / "ffmpeg" / "bin"
    exe = portable_dir / ("ffmpeg.exe" if sys.platform == "win32" else "ffmpeg")
    if exe.exists():
        return str(exe)
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    return "ffmpeg"

FFMPEG_PATH = find_ffmpeg()

def get_ffprobe_path():
    ffmpeg_dir = os.path.dirname(FFMPEG_PATH)
    ffprobe_name = "ffprobe.exe" if sys.platform == "win32" else "ffprobe"
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
        cmd = [get_ffprobe_path(), "-v", "error", "-show_entries", "format=duration",
               "-of", "default=noprint_wrappers=1:nokey=1", audio_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception:
        pass
    try:
        # 修复：兜底改为流式分块统计帧数（原全量 sf.read 把音频解码为
        # float32 数组，100MB 文件内存峰值可达数百 MB）
        if os.path.getsize(audio_path) <= 100 * 1024 * 1024:
            with sf.SoundFile(audio_path) as f:
                sr = f.samplerate
                total_frames = 0
                for block in f.blocks(blocksize=1 << 20, dtype="float32", always_2d=True):
                    total_frames += block.shape[0]
                if sr > 0:
                    return total_frames / sr
    except Exception:
        return None
    return None

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
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "fr": "jonatasgrosman/wav2vec2-large-xlsr-53-french",
    "de": "jonatasgrosman/wav2vec2-large-xlsr-53-german",
    "es": "jonatasgrosman/wav2vec2-large-xlsr-53-spanish",
    "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
    "it": "jonatasgrosman/wav2vec2-large-xlsr-53-italian",
    "nl": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
    "hu": "jonatasgrosman/wav2vec2-large-xlsr-53-hungarian",
    "ru": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
    "pl": "jonatasgrosman/wav2vec2-large-xlsr-53-polish",
    "vi": "jonatasgrosman/wav2vec2-large-xlsr-53-vietnamese",
    "tr": "jonatasgrosman/wav2vec2-large-xlsr-53-turkish",
    "ko": "jonatasgrosman/wav2vec2-large-xlsr-53-korean",
    "ar": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
    "sv": "jonatasgrosman/wav2vec2-large-xlsr-53-swedish",
    "uk": "jonatasgrosman/wav2vec2-large-xlsr-53-ukrainian",
    "fi": "jonatasgrosman/wav2vec2-large-xlsr-53-finnish",
    "da": "jonatasgrosman/wav2vec2-large-xlsr-53-danish",
    "no": "jonatasgrosman/wav2vec2-large-xlsr-53-norwegian",
    "cs": "jonatasgrosman/wav2vec2-large-xlsr-53-czech",
    "ro": "jonatasgrosman/wav2vec2-large-xlsr-53-romanian",
    "el": "jonatasgrosman/wav2vec2-large-xlsr-53-greek",
    "he": "jonatasgrosman/wav2vec2-large-xlsr-53-hebrew",
    "hi": "jonatasgrosman/wav2vec2-large-xlsr-53-hindi",
    "th": "jonatasgrosman/wav2vec2-large-xlsr-53-thai",
    "id": "jonatasgrosman/wav2vec2-large-xlsr-53-indonesian",
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
        self.active_tasks = 0  # 并发任务计数：防止任务间互卸对方正使用的对齐模型
        self.asr_in_use = 0    # 转写占用计数：转写进行中禁止切换 ASR 模型，防止显存峰值翻倍

    def get_local_models(self):
        """修复：过滤 wav2vec2/xlsr 对齐模型目录，防止默认 ASR 模型选错。"""
        models = []
        models_dir = PROJECT_ROOT / "pretrained_models"
        if not models_dir.exists():
            return models
        for item in models_dir.iterdir():
            if not item.is_dir():
                continue
            name = item.name.lower()
            if "wav2vec2" in name or "xlsr" in name:
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
            if device == "cpu" and compute_type == "float16":
                return False, "CPU 模式不支持 float16，请选择 int8_float32 或 float32"
            if (self.model is not None and self.current_model_name == model_size
                    and self.device == device and self.compute_type == compute_type):
                return True, f"模型 {model_size} 已加载"
            # 修复：转写占用期间禁止切换/重载 ASR 模型——旧模型被 del 后仍被
            # 转写线程的本地引用持有，显存不会释放，新模型再加载会峰值翻倍
            if self.asr_in_use > 0:
                return False, "有转写任务正在进行，暂不能切换 ASR 模型，请等待其完成后再试"
            local_models = self.get_local_models()
            model_path = None
            for disp, path in local_models:
                if disp == model_size:
                    model_path = path
                    break
            local_only = bool(model_path)
            model_name_or_path = model_path if model_path else model_size
            # 修复：切换/重载 ASR 模型前先卸载旧模型，防止新旧模型同时驻留显存
            if self.model is not None:
                del self.model
                self.model = None
                self.current_model_name = None
                self._clean_gpu_memory()
            try:
                self.model = WhisperModel(model_name_or_path, device=device,
                                          compute_type=compute_type, local_files_only=local_only)
                self.current_model_name = model_size
                self.device = device
                self.compute_type = compute_type
                return True, f"模型 {model_size} 加载成功"
            except Exception as e:
                return False, f"加载失败: {e}"

    def transcribe_with_segments(self, audio_path, language=None, beam_size=5,
                                 vad_filter=True, vad_parameters=None, initial_prompt=None):
        # 修复：锁内只取模型引用，推理在锁外执行，
        # 避免转写期间阻塞系统信息面板等 UI 事件
        with self.lock:
            model = self.model
            if model is None:
                return None, "模型未加载"
            # 修复：转写期间占用计数（faster-whisper 的 segments 为惰性生成器，
            # 计数需覆盖下方完整消费过程），防止并发任务切换 ASR 模型
            self.asr_in_use += 1
        try:
            try:
                segments, info = model.transcribe(
                    audio_path, language=language, beam_size=beam_size,
                    vad_filter=vad_filter,
                    vad_parameters=vad_parameters if vad_filter else None,
                    word_timestamps=True, initial_prompt=initial_prompt)
            except Exception as e:
                if vad_filter and ("vad" in str(e).lower() or "offline" in str(e).lower()):
                    logger.warning(f"VAD 失败，回退无 VAD。错误: {e}")
                    segments, info = model.transcribe(
                        audio_path, language=language, beam_size=beam_size,
                        vad_filter=False, word_timestamps=True, initial_prompt=initial_prompt)
                else:
                    return None, str(e)
            seg_list = []
            for seg in segments:
                seg_dict = {"start": seg.start, "end": seg.end, "text": seg.text.strip()}
                if seg.words:
                    seg_dict["words"] = [{"word": w.word, "start": w.start, "end": w.end} for w in seg.words]
                seg_list.append(seg_dict)
            return {"language": info.language, "segments": seg_list}, None
        finally:
            with self.lock:
                self.asr_in_use = max(0, self.asr_in_use - 1)

    def load_align_model(self, language_code, device, model_name=None, model_dir=None):
        with self.lock:
            if not WHISPERX_ALIGN_AVAILABLE:
                raise RuntimeError("whisperx.align 不可用")
            cache_key = f"{language_code}_{model_name}_{device}"
            if self.align_model is not None and self.align_model_lang == cache_key:
                return self.align_model, self.align_metadata
            # 修复：语言/设备/模型变化时先卸载旧模型释放显存
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
            if (self.align_model is not None and not self.keep_align_model_loaded
                    and self.active_tasks == 0):
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
        # 修复：keep 偏好移入锁内读取，避免与任务结束时锁内写入竞态
        if manager.keep_align_model_loaded:
            lines.append("对齐模型保持: 是")
    lines.append(f"FFmpeg: {'已找到' if FFMPEG_PATH != 'ffmpeg' else '未找到'}")
    if align_model_info:
        lines.append(f"对齐模型: {align_model_info}")
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
                valid_chars = [c for c in chars_list if "char" in c and c["char"].strip() != ""]
                if not valid_chars:
                    # 修复：缺时间戳的词条直接跳过，避免 0 时刻幽灵词拉偏插值
                    if "start" in w and "end" in w:
                        words.append({"word": w.get("word", ""), "start": w["start"], "end": w["end"]})
                    continue
                if all("start" in c and "end" in c for c in valid_chars):
                    for c in valid_chars:
                        words.append({"word": c["char"], "start": c["start"], "end": c["end"]})
                elif "start" in w and "end" in w:
                    word_dur = w["end"] - w["start"]
                    char_dur = word_dur / len(valid_chars) if len(valid_chars) > 0 else word_dur
                    for idx, c in enumerate(valid_chars):
                        words.append({"word": c["char"], "start": w["start"] + idx * char_dur,
                                      "end": w["start"] + (idx + 1) * char_dur})
            else:
                if "start" in w and "end" in w and "word" in w:
                    words.append({"word": w["word"], "start": w["start"], "end": w["end"]})
                else:
                    prev_word = seg_words[i - 1] if i > 0 else None
                    next_word = seg_words[i + 1] if i < len(seg_words) - 1 else None
                    start = (prev_word["end"] if prev_word and "end" in prev_word else seg.get("start", 0.0))
                    end = (next_word["start"] if next_word and "start" in next_word else seg.get("end", start + 0.01))
                    if end <= start:  # 修复：前后词时间倒挂兜底，防止负时长词条向下游传播
                        end = start + 0.01
                    word_text = w.get("word", "")
                    if word_text:
                        words.append({"word": word_text, "start": start, "end": end})
    return words

# ============ 核心对齐函数 ============
def run_alignment(
    audio_file, primary_text, secondary_text, secondary_lang, enable_dual,
    model_size, device, compute_type, primary_lang, beam_size,
    vad_filter, vad_threshold, vad_min_speech, vad_min_silence, hotwords,
    align_sync_lang, align_model_manual, align_granularity,
    merge_punctuations, merge_max_words, merge_max_chars, merge_max_duration,
    merge_silence_threshold, merge_by_punc, merge_by_silence, merge_by_wordcount,
    merge_by_charcount, merge_by_duration, merge_by_newline, keep_align_loaded,
    force_preprocess, anchor_enable_start, anchor_enable_end, anchor_enable_mean,
    anchor_char_count, anchor_density_enable, density_min_gap, density_boundary_window,
    progress=gr.Progress(),  # 修复：默认值注入，保证进度条生效
):
    if progress is None or not callable(progress):
        # 批量任务传入 None：使用空操作进度对象，避免脱离 Gradio 事件上下文
        def progress(*args, **kwargs):
            pass
    if audio_file is None:
        return "错误: 请上传音频文件", "", "", "", "", "", "", get_system_status()
    if not primary_text or not primary_text.strip():
        return "错误: 请粘贴主文稿", "", "", "", "", "", "", get_system_status()
    # 修复：不再在任务启动时锁外改写全局 keep 偏好（并发任务会互相覆盖，
    # 导致卸载行为与先启动任务的用户意愿相反）；改由最后结束的任务在
    # finally 锁内更新，顺序使用时行为与原版一致
    with manager.lock:
        manager.active_tasks += 1
    temp_files = []
    try:
        # 修复：统一解析 Gradio 5 文件输入
        audio_path, orig_name_full = normalize_file_input(audio_file)
        if not audio_path or not os.path.exists(audio_path):
            return "错误: 无法获取有效的音频文件路径", "", "", "", "", "", "", get_system_status()
        original_filename = Path(orig_name_full).stem if orig_name_full else Path(audio_path).stem

        if force_preprocess:
            tmp = tempfile.NamedTemporaryFile(suffix="_16k_mono.wav", delete=False)
            tmp.close()
            temp_preprocessed = tmp.name
            cmd = [FFMPEG_PATH, "-y", "-i", audio_path, "-ar", "16000",
                   "-ac", "1", "-c:a", "pcm_s16le", temp_preprocessed]
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
                temp_files.append(temp_preprocessed)
                audio_path = temp_preprocessed
                logger.info("音频已预处理为 16kHz 单声道")
            except Exception as e:
                logger.warning(f"音频预处理失败: {e}，使用原始音频")
                try:
                    os.unlink(temp_preprocessed)
                except Exception:
                    pass

        # ---- 对齐模型选择 ----
        local_align_models = manager.get_local_align_models()
        use_whisperx_align = False
        align_model_name_for_load = None
        align_model_dir_for_load = str(ALIGN_CACHE_DIR)
        align_model_display = ""
        align_model_path = None

        def match_local_by_lang(lang: str) -> Optional[str]:
            lang = (lang or "").strip().lower()
            if not lang:
                return None
            for disp, path in local_align_models:
                if re.search(rf'(?:^|[_-]){re.escape(lang)}(?:[_-]|$)', disp.lower()):
                    return path
            lang_names = {
                "zh": ["chinese", "mandarin"], "en": ["english"], "ja": ["japanese"],
                "fr": ["french"], "de": ["german"], "es": ["spanish"], "pt": ["portuguese"],
                "it": ["italian"], "nl": ["dutch"], "hu": ["hungarian"], "ru": ["russian"],
                "pl": ["polish"], "vi": ["vietnamese"], "tr": ["turkish"], "ko": ["korean"],
                "ar": ["arabic"], "sv": ["swedish"], "uk": ["ukrainian"], "fi": ["finnish"],
                "da": ["danish"], "no": ["norwegian"], "cs": ["czech"], "ro": ["romanian"],
                "el": ["greek"], "he": ["hebrew"], "hi": ["hindi"], "th": ["thai"],
                "id": ["indonesian"],
            }
            for kw in lang_names.get(lang, [lang]):
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
            align_model_name_for_load = align_model_path
        system_info = get_system_status(align_model_display)

        # ---- 加载 ASR 模型并转写 ----
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
        # 修复：转写后立即保存检测语言（后续 result 可能被替换为无 language 键的对齐结果）
        detected_lang = result.get("language", "en")
        original_result = copy.deepcopy(result)

        # ---- 精细对齐 ----
        if use_whisperx_align:
            progress(0.6, desc=f"加载对齐模型: {align_model_display}...")
            try:
                align_model, align_metadata = manager.load_align_model(
                    language_code=asr_language or detected_lang,
                    device=device,
                    model_name=align_model_name_for_load,
                    model_dir=align_model_dir_for_load)
                progress(0.7, desc="执行精细对齐...")
                aligned_result = align(
                    transcript=result["segments"],
                    model=align_model,
                    align_model_metadata=align_metadata,
                    audio=audio_path,
                    device=device,
                    return_char_alignments=(align_granularity == "char"))
                result = aligned_result
                # 修复：移除此处主动卸载调用——任务活跃期间 active_tasks>=1，
                # unload_align_model 必然短路，从未生效；统一由 finally 收尾
            except Exception as e:
                logger.warning(f"whisperx.align 失败，回退简单算法: {e}")
                use_whisperx_align = False
                result = original_result

        words = extract_words_from_result(result, align_granularity, use_whisperx_align)
        if not words:
            return "错误: 未检测到有效的单词时间戳", "", "", "", "", "", "", system_info

        # ---- 文稿匹配（原文索引对齐） ----
        progress(0.8, desc="匹配主文稿...")
        duration = get_audio_duration_robust(audio_path)
        full_original, units, para_spans, paragraphs = build_alignment_units(
            primary_text, align_granularity, split_paragraphs=merge_by_newline)  # 修复：开关真实生效
        if not units:
            return "错误: 主文稿无有效内容", "", "", "", "", "", "", system_info
        aligned = force_align_units(units, words, align_granularity, duration)
        if not aligned:
            return "错误: 对齐失败，请检查主文稿与音频是否匹配", "", "", "", "", "", "", system_info
        word_srt = words_to_srt(aligned)

        initial_sentences, unit_para = build_paragraph_sentences(aligned, para_spans, paragraphs)
        merged_segments = generate_merged_srt(
            aligned, full_original, para_spans,
            merge_punctuations, merge_max_words, merge_max_chars, merge_max_duration,
            merge_by_punc, merge_by_silence, merge_by_wordcount,
            merge_by_charcount, merge_by_duration,
            merge_silence_threshold, unit_para)
        merged_srt = sentences_to_srt(merged_segments)
        sent_srt = sentences_to_srt(initial_sentences)
        sentences_for_anchor = initial_sentences

        # ---- 锚点微调（可选） ----
        anchor_srt = ""
        anchor_used = anchor_enable_start or anchor_enable_end or anchor_enable_mean
        if anchor_used and aligned and sentences_for_anchor:
            anchor_lang = primary_lang if primary_lang != "auto" else detected_lang
            refined_sentences = anchor_refine_sentences(
                sentences_for_anchor, aligned,
                primary_language=anchor_lang,
                anchor_char_count=anchor_char_count,
                use_anchor_start=anchor_enable_start,
                use_anchor_end=anchor_enable_end,
                use_anchor_mean=anchor_enable_mean,
                enable_density_anchor=anchor_density_enable,
                density_min_gap=density_min_gap,
                density_boundary_window=int(density_boundary_window))
            anchor_srt = sentences_to_srt(refined_sentences)
        else:
            refined_sentences = sentences_for_anchor

        # ---- 双语挂载 ----
        dual_srt = ""
        secondary_srt = ""
        warning_msg = ""
        if enable_dual and secondary_text and secondary_text.strip():
            # 修复：双语挂载基于合并字幕分段（与 merged_srt 粒度一致），
            # 而非按空行分段的整段句，避免粒度不匹配
            sec_paragraphs = [re.sub(r'\s+', ' ', p.strip())
                              for p in re.split(r'\n\s*\n', secondary_text.strip()) if p.strip()]
            len_diff = abs(len(sec_paragraphs) - len(merged_segments))
            if len_diff <= 1:
                if len(sec_paragraphs) > len(merged_segments):
                    sec_paragraphs = sec_paragraphs[:len(merged_segments)]
                    warning_msg = f"警告: 副文稿段落数多 {len_diff} 段，已截断"
                elif len(merged_segments) > len(sec_paragraphs):
                    sec_paragraphs += [""] * (len(merged_segments) - len(sec_paragraphs))
                    warning_msg = f"警告: 副文稿段落数少 {len_diff} 段，已补空行"
                sec_lines, dual_lines = [], []
                for i, (seg, sec_text) in enumerate(zip(merged_segments, sec_paragraphs), 1):
                    time_str = f"{seconds_to_srt_time(seg['start'])} --> {seconds_to_srt_time(seg['end'])}"
                    sec_lines.extend([str(i), time_str, sec_text, ""])
                    dual_lines.extend([str(i), time_str, seg["text"], sec_text, ""])
                secondary_srt = "\n".join(sec_lines)
                dual_srt = "\n".join(dual_lines)
            else:
                warning_msg = f"警告: 段落数相差 {len_diff} 段（>1），跳过双语生成"
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
        with open(word_path, "w", encoding="utf-8") as f:
            f.write(word_srt)
        with open(sent_path, "w", encoding="utf-8") as f:
            f.write(sent_srt)
        with open(merged_path, "w", encoding="utf-8") as f:
            f.write(merged_srt)
        status = (f"对齐完成！\n逐词字幕: {word_path.name}\n整句字幕: {sent_path.name}\n"
                  f"合并字幕: {merged_path.name}\n对齐模型: {align_model_display}")
        if anchor_used and anchor_srt:
            anchor_path = output_dir / f"{prefix}_anchor.srt"
            with open(anchor_path, "w", encoding="utf-8") as f:
                f.write(anchor_srt)
            status += f"\n锚点字幕: {anchor_path.name}"
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
        return (status, safe_text(word_srt), safe_text(sent_srt), safe_text(merged_srt),
                safe_text(secondary_srt) if secondary_srt else "",
                safe_text(dual_srt) if dual_srt else "",
                safe_text(anchor_srt) if anchor_srt else "",
                system_info)
    except Exception as e:
        # 修复：增加整体异常捕获，避免直接抛 traceback 给前端
        logger.error(traceback.format_exc())
        return f"错误: 处理异常 - {e}", "", "", "", "", "", "", get_system_status()
    finally:
        for f in temp_files:
            try:
                os.unlink(f)
            except Exception:
                pass
        # 修复：任务结束时递减活跃计数；全部任务结束后在锁内以最后结束
        # 任务的 keep 偏好决定是否释放对齐模型（消除并发覆盖竞态）
        with manager.lock:
            manager.active_tasks = max(0, manager.active_tasks - 1)
            if manager.active_tasks == 0:
                manager.keep_align_model_loaded = keep_align_loaded
                if not manager.keep_align_model_loaded:
                    manager.unload_align_model()

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

def read_text_robust(path) -> str:
    """修复：文稿读取支持 UTF-8(BOM)/GB18030 编码回退（原硬编码 utf-8，
    GBK 编码文稿会直接 UnicodeDecodeError）；全部失败则以替换符兜底。"""
    for enc in ("utf-8-sig", "gb18030"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()

# ---------- 批量处理函数 ----------
def batch_folder_align(
    folder_path, secondary_text, secondary_lang, enable_dual,
    model_size, device, compute_type, primary_lang, beam_size,
    vad_filter, vad_threshold, vad_min_speech, vad_min_silence, hotwords,
    align_sync_lang, align_model_manual, align_granularity,
    punc_box, max_words_slider, max_chars_slider, max_duration_slider, silence_slider,
    merge_punc, merge_silence, merge_wordcount, merge_charcount, merge_duration,
    merge_newline, keep_align_loaded, force_preprocess_check,
    anchor_start, anchor_end, anchor_mean, anchor_count_slider,
    anchor_density, density_min_gap_slider, density_boundary_window_slider,
    progress=gr.Progress(),
):
    if not folder_path or not os.path.isdir(folder_path):
        return "错误: 请提供有效的文件夹路径"
    folder = Path(folder_path)
    audio_exts = {'.mp3', '.wav', '.m4a', '.flac', '.ogg'}
    audio_files = [f for f in folder.iterdir() if f.suffix.lower() in audio_exts]
    if not audio_files:
        return "文件夹中未找到支持的音频文件"
    pairs = []
    for audio_f in audio_files:
        txt_f = folder / (audio_f.stem + '.txt')
        if txt_f.exists():
            pairs.append((audio_f, txt_f))
    if not pairs:
        return "未找到任何同名 .txt 文稿文件，请确保音频和文稿文件名相同（除扩展名外）"
    total = len(pairs)
    results = []
    for i, (audio_path, txt_path) in enumerate(pairs, 1):
        progress(i / total, desc=f"处理 {i}/{total}: {audio_path.name}")
        try:
            primary_text = read_text_robust(txt_path)
            if not primary_text.strip():
                results.append(f"❌ {audio_path.name}: 文稿为空")
                continue
            status, *_ = run_alignment(
                str(audio_path), primary_text, secondary_text, secondary_lang,
                enable_dual, model_size, device, compute_type, primary_lang, beam_size,
                vad_filter, vad_threshold, vad_min_speech, vad_min_silence, hotwords,
                align_sync_lang, align_model_manual, align_granularity,
                punc_box, max_words_slider, max_chars_slider, max_duration_slider,
                silence_slider, merge_punc, merge_silence, merge_wordcount,
                merge_charcount, merge_duration, merge_newline, keep_align_loaded,
                force_preprocess_check, anchor_start, anchor_end, anchor_mean,
                anchor_count_slider, anchor_density, density_min_gap_slider,
                density_boundary_window_slider, progress=None)
            if status.startswith("错误"):  # 修复：前缀匹配替代 "错误" in status 包含判定，避免误判
                results.append(f"❌ {audio_path.name}: {status.split(chr(10))[0]}")
            else:
                results.append(f"✅ {audio_path.name}: 已完成")
        except Exception as e:
            results.append(f"❌ {audio_path.name}: 异常 - {str(e)}")
    total_success = sum(1 for r in results if r.startswith("✅"))
    return f"批量处理完成！成功: {total_success}/{total}\n" + "\n".join(results)

# ============ UI 界面 ============
def create_ui():
    local_models = manager.get_local_models()
    model_choices = [name for name, _ in local_models]
    default_models = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
    model_choices = model_choices + [m for m in default_models if m not in model_choices]
    if not model_choices:
        model_choices = default_models
    local_align = manager.get_local_align_models()
    align_choices = ["无（使用默认）"] + [disp for disp, _ in local_align]
    all_languages = ["auto"] + list(LANGUAGE_ALIGN_MODEL_MAP.keys())

    with gr.Blocks(title="字幕自动打轴", theme=gr.themes.Default()) as demo:
        gr.Markdown("# 🎬 字幕自动打轴（支持 28 种语言）")
        with gr.Tabs():
            with gr.Tab("单文件处理"):
                with gr.Row():
                    with gr.Column(scale=1):
                        audio_input = gr.File(label="选择音频文件",
                                              file_types=[".wav", ".mp3", ".m4a", ".flac", ".ogg"])
                        audio_preview = gr.Audio(label="音频预览", interactive=False, visible=False)
                        force_preprocess_check = gr.Checkbox(
                            label="强制预处理为 16kHz 单声道 (推荐大文件)", value=True)
                        primary_text = gr.Textbox(label="主文稿（对齐用）", lines=20,
                                                  placeholder="粘贴稿子...")
                        secondary_text = gr.Textbox(
                            label="副文稿（挂载用，可选）", lines=20,
                            placeholder="翻译稿...\n（双语挂载说明：副文稿需按空行分段，"
                                        "段数须与合并字幕条数一致（相差≤1），否则跳过双语生成）")
                        with gr.Row():
                            secondary_lang = gr.Textbox(label="副文稿语言标记", value="", scale=1)
                            enable_dual = gr.Checkbox(label="生成双语字幕", value=False, scale=1)
                    with gr.Column(scale=2):
                        with gr.Row():
                            status_box = gr.Textbox(label="任务状态", value="等待开始",
                                                    lines=4, interactive=False)
                            system_box = gr.Textbox(label="系统信息", value=get_system_status(),
                                                    lines=4, interactive=False)
                        with gr.Accordion("模型与识别参数", open=True):
                            model_drop = gr.Dropdown(label="ASR模型", choices=model_choices,
                                                     value=model_choices[0] if model_choices else "medium")
                            with gr.Row():
                                device_drop = gr.Dropdown(
                                    label="设备",
                                    choices=["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"],
                                    value="cuda" if torch.cuda.is_available() else "cpu")
                                compute_drop = gr.Dropdown(label="计算类型",
                                                           choices=["int8_float32", "float16", "float32"],
                                                           value="int8_float32")
                            with gr.Row():
                                primary_lang = gr.Dropdown(label="主语言", choices=all_languages, value="zh")
                                beam_slider = gr.Slider(label="Beam Size", minimum=1, maximum=10, value=5, step=1)
                            hotwords_box = gr.Textbox(label="热词/提示词", lines=2, value="")
                        with gr.Accordion("VAD 高级设置", open=False):
                            vad_filter = gr.Checkbox(label="启用 VAD 过滤", value=True)
                            vad_threshold = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="语音检测阈值")
                            vad_min_speech = gr.Slider(100, 1000, value=250, step=50, label="最短语音 (ms)")
                            vad_min_silence = gr.Slider(50, 1000, value=100, step=50, label="最短静音 (ms)")
                        with gr.Accordion("对齐模型设置", open=True):
                            align_sync_lang = gr.Checkbox(label="对齐模型跟随主语言自动匹配", value=True)
                            align_model_manual = gr.Dropdown(label="手动选择对齐模型",
                                                             choices=align_choices,
                                                             value="无（使用默认）", visible=False)
                            refresh_align_btn = gr.Button("刷新对齐模型列表", size="sm")
                            align_granularity = gr.Radio(
                                label="对齐粒度",
                                choices=[("字符级（中日韩推荐）", "char"), ("单词级（西文推荐）", "word")],
                                value="char")
                            keep_align_loaded = gr.Checkbox(label="保持对齐模型加载", value=False)
                        with gr.Accordion("字幕合并规则", open=True):
                            with gr.Row():
                                merge_newline = gr.Checkbox(label="按空行分段(推荐)", value=True)
                                merge_punc = gr.Checkbox(label="按标点断句", value=False)
                                merge_silence = gr.Checkbox(label="按静音断句", value=False)
                            with gr.Row():
                                merge_wordcount = gr.Checkbox(label="按词数断句", value=False)
                                merge_charcount = gr.Checkbox(label="按字符数断句", value=False)
                                merge_duration = gr.Checkbox(label="按时长断句", value=False)
                            with gr.Row():
                                punc_box = gr.Textbox(label="句末标点", value="，；。！？,;.!?", scale=2)
                                silence_slider = gr.Slider(0.1, 1.0, value=0.3, step=0.05, label="静音阈值 (秒)")
                            with gr.Row():
                                max_words_slider = gr.Slider(5, 50, value=20, step=1, label="最大词数")
                                max_chars_slider = gr.Slider(5, 100, value=30, step=5, label="最大字符数")
                                max_duration_slider = gr.Slider(1.0, 20.0, value=10.0, step=0.5, label="最大时长 (秒)")
                        with gr.Accordion("锚点增强 (实验性，默认关闭)", open=False):
                            with gr.Row():
                                anchor_start = gr.Checkbox(label="前锚点", value=False)
                                anchor_end = gr.Checkbox(label="后锚点", value=False)
                                anchor_mean = gr.Checkbox(label="前后均值", value=False)
                            anchor_count_slider = gr.Slider(1, 5, value=3, step=1, label="锚点参考单位数")
                            with gr.Row():
                                anchor_density = gr.Checkbox(label="时间密度锚点", value=False)
                            with gr.Row():
                                density_min_gap_slider = gr.Slider(0.05, 0.5, value=0.18, step=0.01,
                                                                   label="密度最小间隙(秒)")
                                density_boundary_window_slider = gr.Slider(1, 6, value=3, step=1,
                                                                            label="密度边界窗口(单位数)")
                        with gr.Accordion("输出控制", open=False):
                            open_output_btn = gr.Button("打开输出目录", variant="secondary")
                            max_text_len_slider = gr.Slider(2000, 50000, value=current_max_output_length,
                                                            step=2000, label="界面最大显示字符数")
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
                        # 修复：解析 FileInfo 后传路径
                        path, _ = normalize_file_input(file_path)
                        if path and os.path.exists(path):
                            return gr.update(value=path, visible=True)
                        return gr.update(value=None, visible=False)

                    audio_input.change(update_audio_preview, inputs=[audio_input], outputs=[audio_preview])
                    align_sync_lang.change(toggle_align_model_manual, inputs=[align_sync_lang],
                                           outputs=[align_model_manual])
                    run_btn.click(
                        run_alignment,
                        inputs=[
                            audio_input, primary_text, secondary_text, secondary_lang, enable_dual,
                            model_drop, device_drop, compute_drop, primary_lang, beam_slider,
                            vad_filter, vad_threshold, vad_min_speech, vad_min_silence, hotwords_box,
                            align_sync_lang, align_model_manual, align_granularity,
                            punc_box, max_words_slider, max_chars_slider, max_duration_slider,
                            silence_slider, merge_punc, merge_silence, merge_wordcount,
                            merge_charcount, merge_duration, merge_newline, keep_align_loaded,
                            force_preprocess_check, anchor_start, anchor_end, anchor_mean,
                            anchor_count_slider, anchor_density, density_min_gap_slider,
                            density_boundary_window_slider],
                        outputs=[status_box, word_output, sent_output, merged_output,
                                 secondary_output, dual_output, anchor_output, system_box])
                    clear_btn.click(
                        clear_outputs,
                        outputs=[status_box, word_output, sent_output, merged_output,
                                 secondary_output, dual_output, anchor_output, system_box]
                    ).then(
                        lambda: [None, None, "", "", "", False],
                        outputs=[audio_input, audio_preview, primary_text,
                                 secondary_text, secondary_lang, enable_dual])
                    refresh_align_btn.click(refresh_align_model_list, outputs=[align_model_manual])
                    open_output_btn.click(open_output_folder, inputs=None, outputs=None)
                    max_text_len_slider.change(set_max_length, inputs=[max_text_len_slider],
                                               outputs=[system_box])

            with gr.Tab("批量处理（文件夹自动配对）"):
                gr.Markdown("""
### 文件夹自动配对
将音频文件（如 `.mp3`、`.wav`）和同名 `.txt` 文稿放在同一文件夹中，程序会自动匹配并批量生成字幕。

> ⚠️ 注意：左侧填写的"副文稿"会应用到**所有**文件（按段落序号挂载）。
> 若各文件需要不同副文稿，请逐个使用"单文件处理"。
""")
                batch_folder_input = gr.Textbox(label="文件夹路径", placeholder="例如：D:/my_audio_folder")
                batch_status = gr.Textbox(label="批量处理状态", lines=8, interactive=False)
                with gr.Row():
                    batch_run_btn = gr.Button("开始批量处理", variant="primary")
                    batch_clear_btn = gr.Button("清空")
                batch_run_btn.click(
                    batch_folder_align,
                    inputs=[
                        batch_folder_input, secondary_text, secondary_lang, enable_dual,
                        model_drop, device_drop, compute_drop, primary_lang, beam_slider,
                        vad_filter, vad_threshold, vad_min_speech, vad_min_silence, hotwords_box,
                        align_sync_lang, align_model_manual, align_granularity,
                        punc_box, max_words_slider, max_chars_slider, max_duration_slider,
                        silence_slider, merge_punc, merge_silence, merge_wordcount,
                        merge_charcount, merge_duration, merge_newline, keep_align_loaded,
                        force_preprocess_check, anchor_start, anchor_end, anchor_mean,
                        anchor_count_slider, anchor_density, density_min_gap_slider,
                        density_boundary_window_slider],
                    outputs=[batch_status])
                batch_clear_btn.click(lambda: ("", ""), outputs=[batch_folder_input, batch_status])

        gr.HTML("""
<div style="text-align: center; color: #666; font-size: 0.85em; margin-top: 20px;">
  <p>本软件包按"原样"提供，不提供任何明示或暗示的担保。</p>
  <p>更新请关注B站up主：光影的故事2018</p>
</div>
""")
        return demo

def _cleanup():
    try:
        manager.unload_system()
    except Exception:
        pass

atexit.register(_cleanup)  # 修复：补充退出清理

def main():
    model_root = PROJECT_ROOT / "pretrained_models"
    if not model_root.exists():
        print(f"警告: 模型目录 {model_root} 不存在，请确保模型已下载。")
    demo = create_ui()
    demo.queue(default_concurrency_limit=1)
    ports = [18001, 18002, 18003, 18004, 18005]
    for p in ports:
        try:
            demo.launch(server_name="127.0.0.1", server_port=p, inbrowser=True,
                        show_error=True, max_file_size=500 * 1024 * 1024)
            break
        except OSError:
            print(f"端口 {p} 被占用，尝试下一个...")
            continue
    else:
        print("所有端口均被占用，请手动指定空闲端口。")
        sys.exit(1)

if __name__ == "__main__":
    main()
