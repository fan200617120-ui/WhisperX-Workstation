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

import sys, os, json, logging, traceback, time, gc, threading, atexit, tempfile, hashlib, re, subprocess, shutil
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
            except: pass
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

def format_result_to_outputs(result):
    if not result or not isinstance(result, dict):
        return "无结果", "{}", "", []
    text = " ".join([seg["text"] for seg in result.get("segments", [])])
    segments = result.get("segments", [])
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
        # 对齐模型缓存
        self.align_model = None
        self.align_metadata = None
        self.align_model_lang = None

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
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            # 同时释放对齐模型
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

    def transcribe(self, audio_path, language=None, beam_size=5, vad_filter=True, word_timestamps=True, initial_prompt=None):
        if self.asr_model is None: return None, "ASR模型未加载"
        try:
            segments, info = self.asr_model.transcribe(audio_path, language=language, beam_size=beam_size, vad_filter=vad_filter, word_timestamps=word_timestamps, initial_prompt=initial_prompt)
        except Exception as e:
            if vad_filter and ("onnx" in str(e).lower() or "vad" in str(e).lower()):
                print(f"VAD 失败，关闭 VAD 重试。错误: {e}")
                segments, info = self.asr_model.transcribe(audio_path, language=language, beam_size=beam_size, vad_filter=False, word_timestamps=word_timestamps, initial_prompt=initial_prompt)
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
        """
        使用 whisperx.align 精细化单词时间戳。
        model_choice: 可以是 "auto" 或本地模型显示名。
        返回更新后的 result 或原 result（失败时）。
        """
        if not WHISPERX_ALIGN_AVAILABLE:
            return result
        try:
            # 决定对齐模型
            local_align = self.get_local_align_models()
            align_model_path = None
            if model_choice == "auto":
                # 根据检测到的语言自动选择
                detected = result.get("language", "en")
                lang_map = {
                    "zh": "chinese-zh-cn",
                    "en": "english",
                    "ja": "japanese",
                    "fr": "french",
                    "de": "german",
                    "es": "spanish",
                    "pt": "portuguese",
                    "it": "italian",
                    "nl": "dutch",
                    "hu": "hungarian",
                }
                key = lang_map.get(detected.lower(), detected.lower())
                # 在本地模型中搜索
                for disp, path in local_align:
                    if key in disp.lower():
                        align_model_path = path
                        break
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

    def _prepare_audio(self, audio_input):
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
        progress(0.7, desc="生成输出...")
        full_text, tsjson, srt_text, _ = format_result_to_outputs(result)
        base = audio if isinstance(audio, str) and os.path.exists(audio) else None
        saved = save_outputs(base, full_text, tsjson, srt_text, language=result.get("language","未知"), model_info=model_size)
        save_info = "文件已保存:\n"
        if saved.get('txt'): save_info += f" {Path(saved['txt']).name}\n"
        if saved.get('json'): save_info += f" {Path(saved['json']).name}\n"
        if saved.get('srt'): save_info += f" {Path(saved['srt']).name}\n"
        full_text = save_info + "\n" + full_text
        progress(1.0, desc="完成")
        return safe_text(full_text), safe_text(tsjson), safe_text(srt_text)
    finally:
        manager.cleanup_temp()

def transcribe_video(video, model_size, device, compute_type, language, beam_size, vad_filter, subtitle_mode, hotwords, enable_align, align_model, progress=gr.Progress()):
    temp_audio_path = None
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
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        progress(0.4, desc="转写中...")
        prompt = hotwords.strip() if hotwords else None
        result, err = manager.transcribe(audio_path, language=language, beam_size=beam_size, vad_filter=vad_filter, word_timestamps=True, initial_prompt=prompt)
        if err: return f"识别失败: {err}", "", ""
        if enable_align:
            progress(0.6, desc="精细对齐...")
            result = manager.apply_whisperx_align(result, audio_path, language, device, align_model)
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
        srt_str = str(srt_path).replace('\\','/')
        vid_str = str(video).replace('\\','/')
        out_str = str(out_path).replace('\\','/')
        if subtitle_mode == "soft":
            cmd = [FFMPEG_PATH, "-i", vid_str, "-i", srt_str, "-c", "copy", "-c:s", "mov_text", "-metadata:s:s:0", "language=chi", "-y", out_str]
        else:
            cmd = [FFMPEG_PATH, "-i", vid_str, "-vf", f"subtitles='{srt_str}':force_style='FontName=Microsoft YaHei,FontSize=24,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,BorderStyle=3'", "-c:a", "copy", "-y", out_str]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        result_msg = f"✅ 处理完成！输出视频: {out_path.name}\n字幕文件已保存至 output 目录。"
        progress(1.0, desc="完成")
        return safe_text(result_msg), "", ""
    except Exception as e:
        logger.error(traceback.format_exc())
        return f"处理视频失败: {str(e)}", "", ""
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try: os.unlink(temp_audio_path)
            except: pass
        manager.cleanup_temp()

def transcribe_batch(files, model_size, device, compute_type, language, beam_size, vad_filter, hotwords, enable_align, align_model, progress=gr.Progress()):
    if not files: return "请选择音频文件"
    try: ensure_model_loaded(model_size, device, compute_type, language)
    except RuntimeError as e: return str(e)
    total = len(files)
    for i, fobj in enumerate(files, 1):
        fp = fobj.name if hasattr(fobj, 'name') else str(fobj)
        progress(i/total, desc=f"处理 {i}/{total}: {os.path.basename(fp)}")
        ap = manager._prepare_audio(fp)
        if not ap: continue
        try:
            prompt = hotwords.strip() if hotwords else None
            result, err = manager.transcribe(ap, language=language, beam_size=beam_size, vad_filter=vad_filter, word_timestamps=True, initial_prompt=prompt)
            if err: continue
            if enable_align:
                result = manager.apply_whisperx_align(result, ap, language, device, align_model)
            full_text, tsjson, srt_text, _ = format_result_to_outputs(result)
            save_outputs(fp, full_text, tsjson, srt_text, language=result.get("language","未知"), model_info=model_size)
        finally:
            manager.cleanup_temp()
    return f"✅ 批量处理完成，共 {total} 个文件。\n详细结果请查看 output 目录。"

def load_model_click(model_size, device, compute_type, language):
    success, msg = manager.load_asr_model(model_size, device, compute_type, language)
    return msg, get_system_info()

def unload_model_click():
    manager.unload_models()
    return "模型已卸载", get_system_info()

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
    device_choices = ["cuda" if torch.cuda.is_available() else "cpu", "cpu"]
    compute_choices = ["int8_float32", "float16", "float32"]

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
            compute_type = gr.Dropdown(label="计算类型", choices=compute_choices, value="int8_float32")
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

        load_btn.click(load_model_click, inputs=[model_size, device, compute_type, language], outputs=[status_display, status_display])
        unload_btn.click(unload_model_click, outputs=[status_display, status_display])
        refresh_btn.click(refresh_status, outputs=[status_display])
        gr.Markdown("---")
        with gr.Tabs():
            with gr.Tab("音频识别"):
                with gr.Row():
                    with gr.Column(scale=1):
                        audio_input = gr.Audio(label="选择或录制音频", type="numpy", sources=["upload","microphone"])
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