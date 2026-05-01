#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WhisperX 语音识别独立增强版（参数全开放 / 稳定版）
- 音频识别 / 视频字幕 / 批量处理
- 可选精细时间戳对齐（wav2vec2）
- 字幕断句控制（最长时长/字符数/标点/静音间隔，可独立组合启用）
- VAD 高级参数（onset/offset/时长/静音间隔）
- 界面显示最大字符数可调（默认5000，防前端崩溃）
- 一键打开输出目录
- 音频预处理（FFmpeg 16k mono）可控开关，避免内存溢出
Copyright 2026 光影的故事2018
"""

import sys, os, json, logging, traceback, time, gc, threading, atexit, tempfile, hashlib, re, subprocess, shutil, math, uuid
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set

sys.setrecursionlimit(10000)

# ==================== 日志 ====================
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

def clean_old_logs(days=7):
    cutoff = time.time() - days*24*3600
    for f in LOG_DIR.glob("error_*.log"):
        if f.stat().st_mtime < cutoff:
            try: f.unlink()
            except: pass
clean_old_logs()
log_file = LOG_DIR / f"error_{time.strftime('%Y%m%d')}.log"
logging.basicConfig(filename=log_file, level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== 路径 ====================
CURRENT_DIR = Path(__file__).parent.absolute()
if (CURRENT_DIR.parent / "pretrained_models").exists() or (CURRENT_DIR.parent / "preset").exists():
    PROJECT_ROOT = CURRENT_DIR.parent
else:
    PROJECT_ROOT = CURRENT_DIR
sys.path.insert(0, str(PROJECT_ROOT))
BASE_DIR = CURRENT_DIR
ROOT_DIR = PROJECT_ROOT
DEFAULT_OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
# 提前创建输出目录（修复 Bug 8）
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PRESET_DIR = ROOT_DIR / "preset"
PRESET_DIR.mkdir(exist_ok=True)
CONFIG_FILE = PRESET_DIR / "settings.json"
config_lock = threading.RLock()

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
        except: return {}
    return {}

def save_settings(settings):
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f: json.dump(settings, f, ensure_ascii=False, indent=2)
    except: pass

# ==================== 导入依赖 ====================
try:
    import gradio as gr
    import torch, numpy as np, librosa, soundfile as sf
    from faster_whisper import WhisperModel
    print(f"PyTorch: {torch.__version__}, CUDA可用: {torch.cuda.is_available()}")
    print("faster-whisper: 已导入")
except ImportError as e:
    print(f"基础依赖缺失: {e}")
    sys.exit(1)

try:
    from whisperx import load_align_model, align as whisperx_align
    WHISPERX_ALIGN_AVAILABLE = True
    print("whisperx.align: 可用")
except ImportError:
    WHISPERX_ALIGN_AVAILABLE = False
    print("提示: 未找到 whisperx.align，精细对齐功能将不可用。")

# ==================== 工具函数 ====================
DEFAULT_MAX_OUTPUT_LENGTH = 5000
current_max_output_length = DEFAULT_MAX_OUTPUT_LENGTH

def safe_text(text: str, max_len: int = None) -> str:
    """对纯文本/日志进行截断，但对 JSON/SRT 等结构化内容限制整个输出框的展示度，不破坏结构。"""
    if max_len is None:
        max_len = current_max_output_length
    if not isinstance(text, str):
        return str(text)
    if len(text) > max_len:
        if text.strip().startswith("[") or text.strip().startswith("{"):
            return text[:max_len] + "\n\n[注意] 返回的 JSON 过长已截断，完整结果已保存至输出目录。"
        if re.match(r'^\d+\n', text):
            return text[:max_len] + "\n\n[注意] 返回的 SRT 过长已截断，完整结果已保存至输出目录。"
        return text[:max_len] + "\n\n[注意] 返回内容过长已截断，完整结果已保存至输出目录。"
    return text

def seconds_to_srt_time(seconds: float) -> str:
    total_ms = round(seconds * 1000)
    hours = total_ms // 3600000
    minutes = (total_ms % 3600000) // 60000
    secs = (total_ms % 60000) // 1000
    ms = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"

def split_long_segments(segments,
                        enable_duration_split=False, max_duration=None,
                        enable_char_split=False, max_chars=None,
                        enable_silence_split=False, max_silence_ms=None,
                        enable_punc_split=False, punc_set=None):
    """
    根据多种策略独立或组合拆分过长的句子片段。
    """
    if not segments:
        return segments

    # 先根据标点拆分
    if enable_punc_split and punc_set:
        segments = _split_by_punc(segments, punc_set)

    new_segments = []
    for seg in segments:
        text = seg['text'].strip()
        start = seg['start']
        end = seg['end']
        duration = end - start
        if duration <= 0 or len(text) == 0:
            new_segments.append(seg)
            continue

        words = seg.get('words', [])
        # --- 静音分割 ---
        if enable_silence_split and words and max_silence_ms is not None:
            # 收集静音间隙边界，将 segment 切分为多个子段
            sub_segments_times = []
            last_end = start
            sub_start = start
            for w in words:
                w_start = w.get('start', last_end)
                if w_start - last_end > max_silence_ms / 1000.0:
                    if sub_start < last_end:
                        sub_segments_times.append((sub_start, last_end))
                    sub_start = w_start
                last_end = max(last_end, w.get('end', w_start))
            # 最后一段
            if sub_start < end:
                sub_segments_times.append((sub_start, end))
            if not sub_segments_times:
                sub_segments_times = [(start, end)]
        else:
            sub_segments_times = [(start, end)]

        # 对每个子段（已按静音切分）应用时长/字数分割
        for sub_start, sub_end in sub_segments_times:
            sub_duration = sub_end - sub_start
            if sub_duration <= 0:
                continue
            # 提取该子段内的单词
            sub_words = [w for w in words if w.get('start', 0) >= sub_start and w.get('end', 0) <= sub_end + 1e-4]
            # 计算该子段内的文本
            sub_text = " ".join([w['word'] for w in sub_words]) if sub_words else ""
            # 如果只是时间片段但没有单词，用原文本（按比例分割）
            if not sub_text:
                # 简单按时间比例截取原始文本（不够精确但保证不崩溃）
                total_duration = end - start
                ratio_start = (sub_start - start) / total_duration if total_duration > 0 else 0
                ratio_end = (sub_end - start) / total_duration if total_duration > 0 else 1
                start_char = int(ratio_start * len(text))
                end_char = int(ratio_end * len(text))
                sub_text = text[start_char:end_char].strip()
                sub_words = []
            # 再对子段应用时长/字数分割
            splitted = _split_by_time_and_chars(
                sub_text, sub_start, sub_end, sub_words,
                enable_duration_split, max_duration,
                enable_char_split, max_chars
            )
            new_segments.extend(splitted)
    return new_segments

def _split_by_punc(segments, punc_set):
    """按标点符号拆分句子，同时保持时间戳。"""
    result = []
    for seg in segments:
        text = seg['text']
        if not any(p in text for p in punc_set):
            result.append(seg)
            continue
        start = seg['start']
        end = seg['end']
        duration = end - start
        char_dur = duration / len(text) if len(text) > 0 else 0
        sub_texts = re.split(f'([{re.escape("".join(punc_set))}])', text)
        current = ""
        current_start_idx = 0
        for part in sub_texts:
            if not part:
                continue
            current += part
            if part and part[-1] in punc_set:
                if current.strip():
                    end_idx = current_start_idx + len(current)
                    sub_end = start + end_idx * char_dur
                    if sub_end > end:
                        sub_end = end
                    result.append({
                        'start': start + current_start_idx * char_dur,
                        'end': sub_end,
                        'text': current.strip()
                    })
                current_start_idx += len(current)
                current = ""
        if current.strip():
            sub_end = end
            result.append({
                'start': start + current_start_idx * char_dur,
                'end': sub_end,
                'text': current.strip()
            })
    return result

def _split_by_time_and_chars(text, start, end, words,
                             enable_duration_split, max_duration,
                             enable_char_split, max_chars):
    """根据时长和字数限制进一步分割一个句子片段。"""
    duration = end - start
    char_count = len(text)
    need_split = False
    if enable_duration_split and max_duration is not None and duration > max_duration:
        need_split = True
    if enable_char_split and max_chars is not None and char_count > max_chars:
        need_split = True
    if not need_split or duration <= 0 or char_count == 0:
        return [{'start': start, 'end': end, 'text': text.strip(), 'words': words}]

    # 计算分割份数
    num_splits = 1
    if enable_duration_split and max_duration is not None:
        num_splits = max(num_splits, math.ceil(duration / max_duration))
    if enable_char_split and max_chars is not None:
        num_splits = max(num_splits, math.ceil(char_count / max_chars))

    # 改为按词数均匀分割，避免字符索引混乱（修复 Bug 2）
    if words and len(words) > 1:
        words_per_seg = math.ceil(len(words) / num_splits)
        result = []
        for i in range(0, len(words), words_per_seg):
            chunk_words = words[i:i+words_per_seg]
            if not chunk_words:
                continue
            sub_start = chunk_words[0]['start']
            sub_end = chunk_words[-1]['end']
            sub_text = "".join([w['word'] for w in chunk_words]).strip()
            result.append({
                'start': sub_start,
                'end': sub_end,
                'text': sub_text,
                'words': chunk_words
            })
        return result
    else:
        # 无单词信息时按时间均分，文本按比例截取（兜底）
        time_per_seg = duration / num_splits
        chars_per_seg = math.ceil(char_count / num_splits)
        result = []
        for i in range(num_splits):
            sub_start = start + i * time_per_seg
            sub_end = start + (i+1) * time_per_seg if i < num_splits-1 else end
            start_idx = i * chars_per_seg
            end_idx = min(start_idx + chars_per_seg, char_count)
            sub_text = text[start_idx:end_idx].strip()
            if sub_text:
                result.append({
                    'start': sub_start,
                    'end': sub_end,
                    'text': sub_text,
                    'words': []
                })
        return result

def format_result_to_outputs(result, **kwargs):
    """
    将原始识别结果转换为文本、JSON、SRT。
    断句参数通过关键字传入，内部调用 split_long_segments。
    """
    if not result or not isinstance(result, dict):
        return "无结果", "{}", "", []

    segments = split_long_segments(
        result.get("segments", []),
        enable_duration_split=kwargs.get('enable_duration_split', False),
        max_duration=kwargs.get('max_duration'),
        enable_char_split=kwargs.get('enable_char_split', False),
        max_chars=kwargs.get('max_chars'),
        enable_silence_split=kwargs.get('enable_silence_split', False),
        max_silence_ms=kwargs.get('max_silence_ms'),
        enable_punc_split=kwargs.get('enable_punc_split', False),
        punc_set=set(kwargs.get('punc_chars', '')) if kwargs.get('enable_punc_split') else None
    )

    has_cjk = any(re.search(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', seg.get("text", ""))
                  for seg in segments)
    join_char = "" if has_cjk else " "
    text = join_char.join([seg["text"] for seg in segments])
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
    ts = time.strftime("%Y%m%d_%H%M%S")
    try:
        original_stem = Path(base_name).stem
    except Exception:
        original_stem = "recording"
    safe_stem = re.sub(r'[^\w\u4e00-\u9fff\-\.]', '', original_stem)
    if not safe_stem:
        safe_stem = "whisperx"
    prefix = f"{safe_stem}_{ts}"
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

def open_output_folder():
    if sys.platform == "win32":
        os.startfile(str(OUTPUT_DIR))
    else:
        subprocess.Popen(["xdg-open" if shutil.which("xdg-open") else "open", str(OUTPUT_DIR)])

# ==================== 模型管理器 ====================
class WhisperXManager:
    def __init__(self):
        self.asr_model = None
        self.current_asr_model_name = None
        self.current_device = None
        self.current_compute_type = None
        self.settings = load_settings()
        self.temp_files = []
        self.lock = threading.RLock()
        self.align_model = None
        self.align_metadata = None
        self.align_model_lang = None
        self.original_input_name = None

    def get_available_local_models(self):
        models = []
        models_dir = ROOT_DIR / "pretrained_models"
        if not models_dir.exists(): return []
        for item in models_dir.iterdir():
            if not item.is_dir(): continue
            if (item / "config.json").exists() or (item / "tokenizer.json").exists() or (item / "model.bin").exists():
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
            # 优先使用本地路径（修复硬编码 Bug）
            local_path = ROOT_DIR / "pretrained_models" / model_size
            if local_path.exists() and ((local_path / "model.bin").exists() or (local_path / "config.json").exists()):
                model_name_or_path = str(local_path)
                local_only = True
            else:
                # 如果是已知的远程模型，允许下载（可配置）
                known = ["tiny","base","small","medium","large-v2","large-v3","large-v3-turbo"]
                if model_size in known:
                    model_name_or_path = model_size
                    local_only = False
                else:
                    # 尝试从扫描到的本地模型中匹配（以防用户自定义名称）
                    available = self.get_available_local_models()
                    for disp, path in available:
                        if disp == model_size:
                            model_name_or_path = path
                            local_only = True
                            break
                    else:
                        model_name_or_path = model_size
                        local_only = False
            if self.asr_model is not None and self.current_asr_model_name == model_name_or_path and self.current_device == device and self.current_compute_type == compute_type:
                return True, f"ASR模型已加载: {model_size}"
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
        with self.lock:
            if self.asr_model:
                del self.asr_model
                self.asr_model = None
                self.current_asr_model_name = None
                self.current_device = None
                self.current_compute_type = None
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            self.unload_align_model()

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

    def transcribe(self, audio_path, language=None, beam_size=5, vad_filter=True,
                   vad_parameters=None, word_timestamps=True, initial_prompt=None):
        if self.asr_model is None: return None, "ASR模型未加载"
        try:
            segments, info = self.asr_model.transcribe(
                audio_path,
                language=language,
                beam_size=beam_size,
                vad_filter=vad_filter,
                vad_parameters=vad_parameters if vad_filter else None,
                word_timestamps=word_timestamps,
                initial_prompt=initial_prompt
            )
        except Exception as e:
            if vad_filter and ("onnx" in str(e).lower() or "vad" in str(e).lower()):
                print(f"VAD 失败，关闭 VAD 重试。错误: {e}")
                segments, info = self.asr_model.transcribe(
                    audio_path,
                    language=language,
                    beam_size=beam_size,
                    vad_filter=False,
                    word_timestamps=word_timestamps,
                    initial_prompt=initial_prompt
                )
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

    def apply_whisperx_align(self, result, audio_path, language, device, model_choice):
        if not WHISPERX_ALIGN_AVAILABLE:
            return result
        try:
            local_align = self.get_local_align_models()
            align_model_path = None
            lang_map = {
                "zh": "chinese-zh-cn", "en": "english", "ja": "japanese",
                "fr": "french", "de": "german", "es": "spanish",
                "pt": "portuguese", "it": "italian", "nl": "dutch", "hu": "hungarian",
                "ru": "russian", "pl": "polish", "vi": "vietnamese", "tr": "turkish",
                "ko": "korean", "ar": "arabic", "sv": "swedish", "uk": "ukrainian",
                "fi": "finnish", "da": "danish", "no": "norwegian", "cs": "czech",
                "ro": "romanian", "el": "greek", "he": "hebrew", "hi": "hindi",
                "th": "thai", "id": "indonesian", "ms": "malay", "ca": "catalan",
                "fa": "persian", "tl": "filipino", "sk": "slovak", "bg": "bulgarian",
                "hr": "croatian", "et": "estonian", "lv": "latvian", "lt": "lithuanian",
                "sl": "slovenian", "sr": "serbian", "mk": "macedonian", "sq": "albanian",
                "hy": "armenian", "ka": "georgian", "az": "azerbaijani", "eu": "basque",
                "gl": "galician", "mt": "maltese", "cy": "welsh",
            }
            detected = result.get("language", "en")
            if model_choice == "auto":
                key = lang_map.get(detected.lower(), detected.lower())
                for disp, path in local_align:
                    if key in disp.lower():
                        align_model_path = path
                        break
                if not align_model_path:
                    online_map = { ... }  # 省略，原样
                    align_model_path = online_map.get(detected.lower())
                    if not align_model_path:
                        print(f"未找到语言 {detected} 的对齐模型，跳过精细对齐。")
                        return result
            else:
                for disp, path in local_align:
                    if disp == model_choice:
                        align_model_path = path
                        break
                if not align_model_path:
                    align_model_path = model_choice
            cache_key = f"{language}_{align_model_path}"
            if self.align_model is None or self.align_model_lang != cache_key:
                print(f"加载对齐模型: {align_model_path}")
                self.align_model, self.align_metadata = load_align_model(
                    language_code=language or result.get("language", "en"),
                    device=device,
                    model_name=align_model_path,
                    model_dir=str(ROOT_DIR / "pretrained_models")
                )
                self.align_model_lang = cache_key
            print("执行精细对齐...")
            aligned = whisperx_align(
                result["segments"],
                self.align_model,
                self.align_metadata,
                audio_path,
                device,
                return_char_alignments=False
            )
            if "segments" in aligned:
                result["segments"] = aligned["segments"]
            new_words = []
            for seg in result["segments"]:
                if "words" in seg:
                    new_words.extend(seg["words"])
            result["words"] = new_words
            print("精细对齐完成。")
        except Exception as e:
            print(f"精细对齐出错: {e}，将使用原始时间戳。")
        return result

    def cleanup_temp(self):
        cleaned = 0
        for f in self.temp_files[:]:
            try:
                os.unlink(f)
                self.temp_files.remove(f)
                cleaned += 1
            except: pass
        return cleaned

    def set_original_input_name(self, name):
        self.original_input_name = name

    def _prepare_audio(self, audio_input, force_preprocess=False):
        """
        返回用于转写的音频路径（16kHz mono）
        """
        if isinstance(audio_input, (str, Path)) and os.path.exists(str(audio_input)):
            if not force_preprocess:
                return str(audio_input)
            tmp = tempfile.NamedTemporaryFile(suffix="_16k_mono.wav", delete=False)
            tmp.close()
            temp_16k = tmp.name
            try:
                cmd = [
                    FFMPEG_PATH, "-y", "-i", str(audio_input),
                    "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                    temp_16k
                ]
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except (subprocess.CalledProcessError, Exception) as e:
                try: os.unlink(temp_16k)
                except: pass
                logger.error(f"音频预处理失败: {e}")
                return None
            self.temp_files.append(temp_16k)
            return temp_16k
        # 处理 numpy 数组
        if isinstance(audio_input, tuple) and len(audio_input)==2:
            sr, data = audio_input
            if data is None: return None
            # 修复 Bug 3：转为 float32 并归一化
            if np.issubdtype(data.dtype, np.integer):
                data = data.astype(np.float32) / np.iinfo(data.dtype).max
            elif data.dtype != np.float32:
                data = data.astype(np.float32)
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            if sr != 16000:
                try:
                    data = librosa.resample(data, orig_sr=sr, target_sr=16000)
                except Exception as e:
                    raise RuntimeError(f"音频重采样失败: {e}")
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            temp_path = tmp.name
            try:
                sf.write(temp_path, data, 16000)
            except Exception:
                try: os.unlink(temp_path)
                except: pass
                return None
            self.temp_files.append(temp_path)
            return temp_path
        return None

manager = WhisperXManager()

# ==================== 转录函数 ====================
def ensure_model_loaded(model_size, device, compute_type, language):
    success, msg = manager.load_asr_model(model_size, device, compute_type, language)
    if not success: raise RuntimeError(msg)

def build_split_kwargs(enable_duration, max_dur, enable_chars, max_chars,
                       enable_silence, max_silence, enable_punc, punc_chars):
    return {
        'enable_duration_split': enable_duration,
        'max_duration': max_dur if enable_duration else None,
        'enable_char_split': enable_chars,
        'max_chars': max_chars if enable_chars else None,
        'enable_silence_split': enable_silence,
        'max_silence_ms': max_silence if enable_silence else None,
        'enable_punc_split': enable_punc,
        'punc_chars': punc_chars if enable_punc else ''
    }

def transcribe_audio(audio, model_size, device, compute_type, language, beam_size,
                     vad_filter, vad_onset, vad_offset, vad_min_speech, vad_min_silence,
                     hotwords, enable_align, align_model,
                     enable_duration_split, max_duration,
                     enable_char_split, max_chars,
                     enable_silence_split, max_silence_ms,
                     enable_punc_split, punc_chars,
                     force_preprocess,
                     progress=gr.Progress()):
    if audio is None: return "请上传音频文件", "", ""
    original_name = None
    if isinstance(audio, (str, Path)):
        original_name = str(audio)
    elif isinstance(audio, dict) and audio.get('name'):
        original_name = audio['name']
    # 修复：当 original_name 为空时，设为 "recording"
    if not original_name:
        original_name = "recording"
    manager.set_original_input_name(original_name)

    progress(0, desc="初始化...")
    try: ensure_model_loaded(model_size, device, compute_type, language)
    except RuntimeError as e: return str(e), "", ""
    progress(0.2, desc="预处理音频...")
    audio_path = manager._prepare_audio(audio, force_preprocess=force_preprocess)
    if not audio_path: return "音频处理失败", "", ""
    try:
        prompt = hotwords.strip() if hotwords else None
        vad_params = None
        if vad_filter:
            vad_params = {
                "onset": vad_onset,
                "offset": vad_offset,
                "min_speech_duration_ms": vad_min_speech,
                "min_silence_duration_ms": vad_min_silence
            }
        result, err = manager.transcribe(audio_path, language=language, beam_size=beam_size,
                                        vad_filter=vad_filter, vad_parameters=vad_params,
                                        word_timestamps=True, initial_prompt=prompt)
        if err: return f"错误: {err}", "", ""
        if enable_align:
            progress(0.6, desc="精细对齐...")
            result = manager.apply_whisperx_align(result, audio_path, language, device, align_model)
        progress(0.7, desc="生成输出...")
        split_args = build_split_kwargs(enable_duration_split, max_duration,
                                        enable_char_split, max_chars,
                                        enable_silence_split, max_silence_ms,
                                        enable_punc_split, punc_chars)
        full_text, tsjson, srt_text, _ = format_result_to_outputs(result, **split_args)
        saved = save_outputs(original_name, full_text, tsjson, srt_text,
                             language=result.get("language","未知"), model_info=model_size)
        save_info = "文件已保存:\n"
        if saved.get('txt'): save_info += f" {Path(saved['txt']).name}\n"
        if saved.get('json'): save_info += f" {Path(saved['json']).name}\n"
        if saved.get('srt'): save_info += f" {Path(saved['srt']).name}\n"
        full_text = save_info + "\n" + full_text
        progress(1.0, desc="完成")
        return safe_text(full_text), safe_text(tsjson), safe_text(srt_text)
    finally:
        manager.cleanup_temp()

def transcribe_video(video, model_size, device, compute_type, language, beam_size,
                     vad_filter, vad_onset, vad_offset, vad_min_speech, vad_min_silence,
                     subtitle_mode, hotwords, enable_align, align_model,
                     enable_duration_split, max_duration,
                     enable_char_split, max_chars,
                     enable_silence_split, max_silence_ms,
                     enable_punc_split, punc_chars,
                     progress=gr.Progress()):
    temp_audio_path = None
    safe_srt_temp = None
    try:
        if video is None:
            return "请上传视频文件", "", ""
        # 兼容多种 Gradio 返回类型
        video_path = None
        if isinstance(video, dict):
            video_path = video.get('name') or video.get('path')
        elif isinstance(video, (str, Path)):
            video_path = str(video)
        if not video_path or not os.path.exists(str(video_path)):
            return "无法获取视频路径", "", ""

        progress(0, desc="初始化...")
        ensure_model_loaded(model_size, device, compute_type, language)
        progress(0.2, desc="提取音频...")
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_audio.close()
        audio_path = temp_audio.name
        temp_audio_path = audio_path
        cmd = [FFMPEG_PATH, "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-y", audio_path]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        progress(0.4, desc="转写中...")
        prompt = hotwords.strip() if hotwords else None
        vad_params = None
        if vad_filter:
            vad_params = {
                "onset": vad_onset,
                "offset": vad_offset,
                "min_speech_duration_ms": vad_min_speech,
                "min_silence_duration_ms": vad_min_silence
            }
        result, err = manager.transcribe(audio_path, language=language, beam_size=beam_size,
                                        vad_filter=vad_filter, vad_parameters=vad_params,
                                        word_timestamps=True, initial_prompt=prompt)
        if err:
            return f"识别失败: {err}", "", ""
        if enable_align:
            progress(0.6, desc="精细对齐...")
            result = manager.apply_whisperx_align(result, audio_path, language, device, align_model)
        progress(0.7, desc="生成字幕...")
        split_args = build_split_kwargs(enable_duration_split, max_duration,
                                        enable_char_split, max_chars,
                                        enable_silence_split, max_silence_ms,
                                        enable_punc_split, punc_chars)
        full_text, tsjson, srt_text, _ = format_result_to_outputs(result, **split_args)
        saved = save_outputs(video_path, full_text, tsjson, srt_text,
                             language=result.get("language","未知"), model_info=model_size)
        srt_path = saved.get('srt')
        if not srt_path:
            return "处理完成，未生成字幕。", safe_text(tsjson), safe_text(srt_text)
        progress(0.8, desc="嵌入字幕...")
        ts = time.strftime("%Y%m%d_%H%M%S")
        video_stem = Path(video_path).stem
        safe_stem = re.sub(r'[^\w\u4e00-\u9fff\-\.]', '', video_stem)
        prefix = f"{safe_stem}_{ts}"
        out_path = OUTPUT_DIR / f"{prefix}.mp4"

        detected_lang = result.get("language", "en")
        lang_code_map = {"zh": "chi", "en": "eng", "ja": "jpn", "ko": "kor", "fr": "fre", "de": "ger", "es": "spa"}
        sub_lang_code = lang_code_map.get(detected_lang, "eng")

        if subtitle_mode == "soft":
            cmd = [FFMPEG_PATH, "-i", video_path, "-i", str(srt_path),
                   "-c", "copy", "-c:s", "mov_text",
                   "-metadata:s:s:0", f"language={sub_lang_code}",
                   "-y", str(out_path)]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        else:
            safe_srt_temp = os.path.join(tempfile.gettempdir(), f"sub_{uuid.uuid4().hex[:8]}.srt")
            shutil.copy2(str(srt_path), safe_srt_temp)
            # 修复路径转义（Bug 6）
            escaped_srt = safe_srt_temp.replace('\\', '/')
            escaped_srt = escaped_srt.replace(':', '\\:').replace("'", "\\'")
            font_name = "Arial"
            if sys.platform == "win32":
                font_name = "Microsoft YaHei"
            vf_str = f"subtitles='{escaped_srt}':force_style='FontName={font_name},FontSize=24,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,BorderStyle=3'"
            cmd = [FFMPEG_PATH, "-i", video_path, "-vf", vf_str, "-c:a", "copy", "-y", str(out_path)]
            subprocess.run(cmd, check=True, capture_output=True, text=True)

        result_msg = f"✅ 处理完成！输出视频: {out_path.name}\n字幕文件已保存至 output 目录。\n\n【识别文本】\n{full_text}"
        progress(1.0, desc="完成")
        return safe_text(result_msg), safe_text(tsjson), safe_text(srt_text)
    except Exception as e:
        logger.error(traceback.format_exc())
        return f"处理视频失败: {str(e)}", "", ""
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try: os.unlink(temp_audio_path)
            except: pass
        if safe_srt_temp and os.path.exists(safe_srt_temp):
            try: os.unlink(safe_srt_temp)
            except: pass
        manager.cleanup_temp()

def transcribe_batch(files, model_size, device, compute_type, language, beam_size,
                     vad_filter, vad_onset, vad_offset, vad_min_speech, vad_min_silence,
                     hotwords, enable_align, align_model,
                     enable_duration_split, max_duration,
                     enable_char_split, max_chars,
                     enable_silence_split, max_silence_ms,
                     enable_punc_split, punc_chars,
                     force_preprocess,
                     progress=gr.Progress()):
    if not files: return "请选择音频文件"
    try: ensure_model_loaded(model_size, device, compute_type, language)
    except RuntimeError as e: return str(e)
    results_text = []
    failed_text = []
    total = len(files)
    for i, fobj in enumerate(files, 1):
        # 修复：使用 orig_name 保留原始文件名
        orig_name = getattr(fobj, 'orig_name', None) or os.path.basename(str(fobj))
        fp = str(fobj) if hasattr(fobj, '__fspath__') else str(fobj)
        progress(i/total, desc=f"处理 {i}/{total}: {orig_name}")
        ap = manager._prepare_audio(fp, force_preprocess=force_preprocess)
        if not ap:
            failed_text.append(f"❌ {orig_name}: 音频预处理失败")
            continue
        try:
            prompt = hotwords.strip() if hotwords else None
            vad_params = None
            if vad_filter:
                vad_params = {
                    "onset": vad_onset,
                    "offset": vad_offset,
                    "min_speech_duration_ms": vad_min_speech,
                    "min_silence_duration_ms": vad_min_silence
                }
            result, err = manager.transcribe(ap, language=language, beam_size=beam_size,
                                            vad_filter=vad_filter, vad_parameters=vad_params,
                                            word_timestamps=True, initial_prompt=prompt)
            if err:
                failed_text.append(f"❌ {orig_name}: 转写失败 - {err}")
                continue
            if enable_align:
                result = manager.apply_whisperx_align(result, ap, language, device, align_model)
            split_args = build_split_kwargs(enable_duration_split, max_duration,
                                            enable_char_split, max_chars,
                                            enable_silence_split, max_silence_ms,
                                            enable_punc_split, punc_chars)
            full_text, tsjson, srt_text, _ = format_result_to_outputs(result, **split_args)
            # 使用原始文件名保存
            save_outputs(orig_name, full_text, tsjson, srt_text,
                         language=result.get("language","未知"), model_info=model_size)
            results_text.append(f"✅ {orig_name}: 已保存")
        except Exception as e:
            failed_text.append(f"❌ {orig_name}: {str(e)}")
        finally:
            manager.cleanup_temp()
    msg = f"✅ 批量处理完成！成功 {len(results_text)} 个，失败 {len(failed_text)} 个。\n"
    if results_text:
        msg += "\n【成功】\n" + "\n".join(results_text)
    if failed_text:
        msg += "\n\n【失败】\n" + "\n".join(failed_text)
    msg += "\n\n详细结果请查看 output 目录。"
    return msg

def load_model_click(model_size, device, compute_type, language):
    success, msg = manager.load_asr_model(model_size, device, compute_type, language)
    info = get_system_info()
    return msg + "\n" + info

def unload_model_click():
    manager.unload_models()
    msg = "模型已卸载"
    info = get_system_info()
    return msg + "\n" + info

def refresh_status(): return get_system_info()

def health_check():
    info = get_system_info()
    with manager.lock:
        if manager.asr_model is None: info += "\n\n[警告] ASR模型未加载，请先加载模型。"
        else: info += "\n\n[信息] 系统已就绪。"
    return info

def toggle_align_controls(enable_align):
    return gr.update(visible=enable_align and WHISPERX_ALIGN_AVAILABLE)

# ==================== 界面 ====================
def create_interface():
    settings = manager.settings
    default_output_dir = settings.get("output_dir", str(DEFAULT_OUTPUT_DIR))
    global OUTPUT_DIR, current_max_output_length
    with config_lock:
        OUTPUT_DIR = Path(default_output_dir)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    local_models = manager.get_available_local_models()
    model_choices = [disp for disp, _ in local_models]
    if not model_choices:
        gr.Warning("未在 pretrained_models 目录中发现任何模型，请先放置模型文件。")
    device_choices = ["cuda" if torch.cuda.is_available() else "cpu", "cpu"]
    compute_choices = ["int8_float32", "float16", "float32"]

    align_local = manager.get_local_align_models()
    align_options = ["auto"] + [name for name, _ in align_local]

    with gr.Blocks(title="WhisperX 语音识别增强版", theme=gr.themes.Default()) as demo:
        gr.Markdown("# 🎤 WhisperX 语音识别 (稳定版)\n输出目录: `{}`".format(OUTPUT_DIR))

        with gr.Accordion("⚙️ 全局设置 (模型/文本截断)", open=True):
            with gr.Row():
                device = gr.Dropdown(label="设备", choices=device_choices, value=device_choices[0])
                model_size = gr.Dropdown(label="模型大小", choices=model_choices, value=model_choices[0] if model_choices else None, interactive=bool(model_choices))
                compute_type = gr.Dropdown(label="计算类型", choices=compute_choices, value="int8_float32")
                language = gr.Textbox(label="语言代码", value="zh", placeholder="zh/en/ja...")
            with gr.Row():
                beam_size = gr.Slider(label="Beam Size", minimum=1, maximum=10, value=5, step=1)
                max_text_len = gr.Slider(label="界面最大显示字符数", minimum=1000, maximum=20000, value=DEFAULT_MAX_OUTPUT_LENGTH, step=1000,
                                         info="超过此长度将截断，建议保持 5000～10000 避免前端报错")
            with gr.Row():
                load_btn = gr.Button("加载模型", variant="primary")
                unload_btn = gr.Button("卸载模型", variant="stop")
                refresh_btn = gr.Button("刷新状态", variant="secondary")
                health_btn = gr.Button("健康检查", variant="secondary")

        with gr.Accordion("📊 系统状态", open=False):
            status_display_ctrl = gr.Textbox(label="状态", value=get_system_info(), lines=5, interactive=False)
            with gr.Row():
                open_output_btn = gr.Button("打开输出目录", variant="secondary")

        load_btn.click(load_model_click, inputs=[model_size, device, compute_type, language], outputs=[status_display_ctrl])
        unload_btn.click(unload_model_click, outputs=[status_display_ctrl])
        refresh_btn.click(refresh_status, outputs=[status_display_ctrl])
        health_btn.click(health_check, outputs=[status_display_ctrl])
        open_output_btn.click(open_output_folder, inputs=None, outputs=None)

        gr.Markdown("---")

        # 断句控制组件
        def build_split_controls():
            with gr.Accordion("✂️ 断句控制", open=False):
                with gr.Row():
                    enable_duration = gr.Checkbox(label="启用时长断句", value=False)
                    duration_slider = gr.Slider(label="最长单句时长 (秒)", minimum=1, maximum=60, value=15, step=1)
                with gr.Row():
                    enable_chars = gr.Checkbox(label="启用字数断句", value=False)
                    chars_slider = gr.Slider(label="最长单句字符数", minimum=1, maximum=200, value=50, step=5)
                with gr.Row():
                    enable_silence = gr.Checkbox(label="启用静音停顿断句", value=False)
                    silence_slider = gr.Slider(label="最大静音间隔 (毫秒)", minimum=100, maximum=2000, value=500, step=50,
                                              info="单词/字之间超过此静音时长即在此处断开")
                with gr.Row():
                    enable_punc = gr.Checkbox(label="启用标点断句", value=False)
                    punc_text = gr.Textbox(label="断句标点", value="。！？!?……，,;；：:", placeholder="输入用于断句的标点")
            return [enable_duration, duration_slider, enable_chars, chars_slider,
                    enable_silence, silence_slider, enable_punc, punc_text]

        with gr.Tabs():
            # 音频识别
            with gr.Tab("音频识别"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 上传音频")
                        audio_input = gr.File(
                            label="选择音频文件",
                            file_types=[".wav", ".mp3", ".m4a", ".flac", ".ogg"]
                        )
                        audio_preview = gr.Audio(
                            label="🎧 音频预览（上传后可试听）",
                            interactive=False,
                            visible=False
                        )
                        force_preprocess_check = gr.Checkbox(
                            label="⚡ 强制预处理为 16kHz 单声道 (推荐大文件)", value=True
                        )
                        hotwords_audio = gr.Textbox(label="热词/提示词", lines=2, value="")
                        split_ctrls_audio = build_split_controls()
                        with gr.Accordion("VAD 高级设置", open=False):
                            vad_enable_audio = gr.Checkbox(label="启用 VAD 过滤", value=False)
                            vad_onset_audio = gr.Slider(label="语音起始阈值 (onset)", minimum=0.0, maximum=1.0, value=0.6, step=0.05)
                            vad_offset_audio = gr.Slider(label="语音结束阈值 (offset)", minimum=0.0, maximum=1.0, value=0.4, step=0.05)
                            vad_min_speech_audio = gr.Slider(label="最短语音 (毫秒)", minimum=100, maximum=1000, value=250, step=50)
                            vad_min_silence_audio = gr.Slider(label="最短静音 (毫秒)", minimum=50, maximum=1000, value=100, step=50)
                        enable_align_audio = gr.Checkbox(label="使用 wav2vec2 精细对齐", value=False, interactive=WHISPERX_ALIGN_AVAILABLE)
                        align_model_audio = gr.Dropdown(label="对齐模型", choices=align_options, value="auto", visible=False)
                        enable_align_audio.change(toggle_align_controls, inputs=[enable_align_audio], outputs=[align_model_audio])
                        with gr.Row():
                            t_btn = gr.Button("开始识别", variant="primary")
                            c_btn = gr.Button("清空", variant="secondary")
                    with gr.Column(scale=2):
                        text_out = gr.Textbox(label="识别文本", lines=8)
                        json_out = gr.Textbox(label="时间戳 JSON", lines=8)
                        srt_out = gr.Textbox(label="SRT 字幕", lines=8)

                def update_audio_preview(file_path):
                    if file_path:
                        return gr.update(value=file_path, visible=True)
                    return gr.update(value=None, visible=False)

                audio_input.change(update_audio_preview, inputs=[audio_input], outputs=[audio_preview])
                t_btn.click(
                    transcribe_audio,
                    inputs=[audio_input, model_size, device, compute_type, language, beam_size,
                            vad_enable_audio, vad_onset_audio, vad_offset_audio, vad_min_speech_audio, vad_min_silence_audio,
                            hotwords_audio, enable_align_audio, align_model_audio] +
                            split_ctrls_audio + [force_preprocess_check],
                    outputs=[text_out, json_out, srt_out]
                ).then(refresh_status, outputs=[status_display_ctrl])
                c_btn.click(lambda: [None, None, "", "", "", ""], outputs=[audio_input, audio_preview, hotwords_audio, text_out, json_out, srt_out])

            # 视频字幕
            with gr.Tab("视频字幕"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 视频处理\n> 音频会自动转为 16kHz 单声道")
                        video_input = gr.Video(label="选择视频", sources=["upload"])
                        sub_mode = gr.Radio(label="嵌入模式", choices=["soft","hard"], value="soft")
                        hotwords_video = gr.Textbox(label="热词/提示词", lines=2, value="")
                        split_ctrls_video = build_split_controls()
                        with gr.Accordion("VAD 高级设置", open=False):
                            vad_enable_video = gr.Checkbox(label="启用 VAD 过滤", value=False)
                            vad_onset_video = gr.Slider(label="语音起始阈值 (onset)", minimum=0.0, maximum=1.0, value=0.6, step=0.05)
                            vad_offset_video = gr.Slider(label="语音结束阈值 (offset)", minimum=0.0, maximum=1.0, value=0.4, step=0.05)
                            vad_min_speech_video = gr.Slider(label="最短语音 (毫秒)", minimum=100, maximum=1000, value=250, step=50)
                            vad_min_silence_video = gr.Slider(label="最短静音 (毫秒)", minimum=50, maximum=1000, value=100, step=50)
                        enable_align_video = gr.Checkbox(label="使用 wav2vec2 精细对齐", value=False, interactive=WHISPERX_ALIGN_AVAILABLE)
                        align_model_video = gr.Dropdown(label="对齐模型", choices=align_options, value="auto", visible=False)
                        enable_align_video.change(toggle_align_controls, inputs=[enable_align_video], outputs=[align_model_video])
                        with gr.Row():
                            vt_btn = gr.Button("开始处理", variant="primary")
                            vc_btn = gr.Button("清空", variant="secondary")
                    with gr.Column(scale=2):
                        v_text = gr.Textbox(label="识别文本", lines=8)
                        v_json = gr.Textbox(label="时间戳 JSON", lines=8)
                        v_srt = gr.Textbox(label="SRT 字幕", lines=8)
                vt_btn.click(
                    transcribe_video,
                    inputs=[video_input, model_size, device, compute_type, language, beam_size,
                            vad_enable_video, vad_onset_video, vad_offset_video, vad_min_speech_video, vad_min_silence_video,
                            sub_mode, hotwords_video, enable_align_video, align_model_video] +
                            split_ctrls_video,
                    outputs=[v_text, v_json, v_srt]
                ).then(refresh_status, outputs=[status_display_ctrl])
                vc_btn.click(lambda: [None, "", "", "", ""], outputs=[video_input, hotwords_video, v_text, v_json, v_srt])

            # 批量处理
            with gr.Tab("批量处理"):
                with gr.Row():
                    with gr.Column(scale=1):
                        files_input = gr.Files(label="上传多个音频", file_types=[".wav",".mp3",".m4a",".flac",".ogg"], file_count="multiple")
                        force_preprocess_batch = gr.Checkbox(
                            label="⚡ 强制预处理为 16kHz 单声道", value=True
                        )
                        hotwords_batch = gr.Textbox(label="热词/提示词", lines=2, value="")
                        split_ctrls_batch = build_split_controls()
                        with gr.Accordion("VAD 高级设置", open=False):
                            vad_enable_batch = gr.Checkbox(label="启用 VAD 过滤", value=False)
                            vad_onset_batch = gr.Slider(label="语音起始阈值 (onset)", minimum=0.0, maximum=1.0, value=0.6, step=0.05)
                            vad_offset_batch = gr.Slider(label="语音结束阈值 (offset)", minimum=0.0, maximum=1.0, value=0.4, step=0.05)
                            vad_min_speech_batch = gr.Slider(label="最短语音 (毫秒)", minimum=100, maximum=1000, value=250, step=50)
                            vad_min_silence_batch = gr.Slider(label="最短静音 (毫秒)", minimum=50, maximum=1000, value=100, step=50)
                        enable_align_batch = gr.Checkbox(label="使用 wav2vec2 精细对齐", value=False, interactive=WHISPERX_ALIGN_AVAILABLE)
                        align_model_batch = gr.Dropdown(label="对齐模型", choices=align_options, value="auto", visible=False)
                        enable_align_batch.change(toggle_align_controls, inputs=[enable_align_batch], outputs=[align_model_batch])
                        with gr.Row():
                            bt_btn = gr.Button("批量识别", variant="primary")
                            bc_btn = gr.Button("清空", variant="secondary")
                    with gr.Column(scale=2):
                        batch_out = gr.Textbox(label="结果", lines=8)
                bt_btn.click(
                    transcribe_batch,
                    inputs=[files_input, model_size, device, compute_type, language, beam_size,
                            vad_enable_batch, vad_onset_batch, vad_offset_batch, vad_min_speech_batch, vad_min_silence_batch,
                            hotwords_batch, enable_align_batch, align_model_batch] +
                            split_ctrls_batch + [force_preprocess_batch],
                    outputs=[batch_out]
                ).then(refresh_status, outputs=[status_display_ctrl])
                bc_btn.click(lambda: [None, "", ""], outputs=[files_input, hotwords_batch, batch_out])

        gr.Markdown("---")
        gr.HTML("<div style='text-align:center;color:#666;'>© 2026 光影的故事2018</div>")

        def set_max_length(val):
            global current_max_output_length
            current_max_output_length = int(val)
            return get_system_info()
        max_text_len.change(set_max_length, inputs=[max_text_len], outputs=[status_display_ctrl])
        demo.load(refresh_status, outputs=[status_display_ctrl])

    return demo

# 移除 atexit，改用 Gradio 关闭事件（可选）
def cleanup():
    manager.unload_models()
    manager.cleanup_temp()
    clean_old_logs()

def main():
    demo = create_interface()
    for port in [18006,18007,18008,18009,18010]:
        try:
            demo.queue().launch(
                server_name="127.0.0.1",
                server_port=port,
                inbrowser=True,
                show_error=True,
                max_file_size=500 * 1024 * 1024
            )
            break
        except OSError:
            print(f"端口 {port} 被占用，尝试下一个...")
            continue
    else:
        print("所有端口均被占用，请手动指定空闲端口。")
        sys.exit(1)
    # 注册退出清理
    atexit.register(cleanup)

if __name__ == "__main__":
    main()