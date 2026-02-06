import os
import re
import json
import asyncio
import aiohttp
import logging
import uuid
import base64
import random
import functools
import shutil
import time
from io import BytesIO
from PIL import Image as PILImage, ImageDraw, ImageFont
from typing import List, Dict, Any, Optional

import astrbot.api.message_components as Comp
from astrbot.api.all import *
from astrbot.api.event import filter, MessageEventResult
from astrbot.api import llm_tool

# 导入 TTS 工具和 Record 组件
try:
    from .tts_utils import SiliconFlowTTS
except ImportError:
    SiliconFlowTTS = None

try:
    from astrbot.api.message_components import Record
except ImportError:
    try:
        from astrbot.core.message.components import Record
    except ImportError:
        Record = None

# 导入 AI 抠图工具（rembg）
try:
    from rembg import remove as rembg_remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    logging.warning("[Magic Wardrobe] rembg 未安装，将使用传统抠图算法。安装: pip install rembg")

@register("astrbot_plugin_magic_wardrobe", "AkiKa", "AI 魔法衣橱", "1.2.1")
class MagicWardrobePlugin(Star):
    def __init__(self, context: Context, config: Optional[dict] = None):
        super().__init__(context)
        self.config = config or {}
        # 使用绝对路径，确保图片能被正确加载
        self.data_dir = os.path.abspath(os.path.join("data", "plugins", "astrbot_plugin_magic_wardrobe"))
        self.last_char_url = None # 存储模型生成的最后一张图片
        self.last_payload_preview: Optional[Dict[str, Any]] = None
        self.last_payload_at: Optional[float] = None
        self.last_prompt_preview: Optional[Dict[str, Any]] = None
        self._pending_tts: Dict[str, str] = {}
        self._pending_tts_tasks: Dict[str, asyncio.Task] = {}
        self._active_preset: Optional[str] = None
        self._active_preset_path = os.path.join(self.data_dir, "active_preset.txt")

        # 尝试从持久化文件中加载最后一张图
        self.cache_path = os.path.join(self.data_dir, "last_url.txt")
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r") as f: self.last_char_url = f.read().strip()
            except: pass
        self.background_index_path = os.path.join(self.data_dir, "background_index.json")
        self.background_index = self._load_background_index()
        self.actions_data = self._load_actions_data()
        self.models_config = self._load_models_config()
        self._sync_schema_presets()
        self._active_preset = self._load_active_preset()

        self.font_dir = self._prepare_dir("fonts", "ziti")
        self.component_dir = self._prepare_dir("components", "zujian")

        # 确保目录存在
        for d in ["background", "character", "border", "fonts", "components", "presets"]:
            os.makedirs(os.path.join(self.data_dir, d), exist_ok=True)

        # 初始化 TTS 客户端
        self.tts_client = None
        if SiliconFlowTTS and self.config.get("enable_tts", False):
            # 优先使用 TTS 专用 API Key，如果没有则使用图像生成的 API Key
            tts_api_key = self.config.get("tts_api_key", "") or self.config.get("api_key", "")
            if tts_api_key:
                fmt = self.config.get("tts_format", "wav")
                self.tts_client = SiliconFlowTTS(
                    api_url="https://api.siliconflow.cn/v1",
                    api_key=tts_api_key,
                    model="FunAudioLLM/CosyVoice2-0.5B",
                    fmt=fmt,
                    speed=self.config.get("tts_speed", 1.0),
                    gain=self.config.get("tts_gain", 0),
                    sample_rate=self.config.get("tts_sample_rate", 44100),
                )
                logging.info("[Magic Wardrobe] TTS 客户端已初始化")
            else:
                logging.warning("[Magic Wardrobe] TTS 已启用但未配置 API Key")

        # 启动 WebUI 任务
        self.webui_task = asyncio.create_task(self._run_webui())
        logging.info("[Magic Wardrobe] Plugin loaded")

    def _can_tts(self) -> bool:
        return bool(self.config.get("enable_tts", False) and self.tts_client and Record)

    def _should_tts_for_image(self) -> bool:
        return self._can_tts()

    def _should_tts_for_text(self, text: str) -> bool:
        if not self._can_tts():
            return False
        try:
            prob = float(self.config.get("tts_other_probability", 0.0))
        except (TypeError, ValueError):
            prob = 0.0
        if prob <= 0:
            return False
        text = text.strip()
        if not text:
            return False
        min_len = int(self.config.get("tts_text_min_length", 5) or 0)
        max_len = int(self.config.get("tts_text_max_length", 0) or 0)
        if min_len > 0 and len(text) < min_len:
            return False
        if max_len > 0 and len(text) > max_len:
            return False
        return random.random() <= prob

    def _save_last_url(self, url: str):
        self.last_char_url = url
        try:
            with open(self.cache_path, "w") as f: f.write(url)
        except: pass

    def _apply_generated_character_to_layout(self, cached_path: str) -> None:
        if not cached_path or not os.path.exists(cached_path):
            return
        try:
            char_dir = os.path.join(self.data_dir, "character")
            if os.path.commonpath([cached_path, char_dir]) != char_dir:
                return
            rel = os.path.relpath(cached_path, char_dir)
            preset = self._active_preset or self.config.get("active_template", "default")
            layout_path = os.path.join(self.data_dir, "presets", f"{preset}.json")
            if not os.path.exists(layout_path):
                return
            with open(layout_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                data["character_asset"] = rel
                with open(layout_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            return

    def _get_fallback_image_path(self) -> Optional[str]:
        path = (self.config.get("fallback_image", "") or "").strip()
        if not path:
            path = os.path.join(self.data_dir, "fallback.png")
        if os.path.exists(path):
            return path
        return None

    async def _cache_generated_character(self, url: str) -> str:
        if not url:
            return url
        if url.startswith("file://") or os.path.exists(url):
            return url
        try:
            img = await self._download_image(url)
            if not img:
                return url
            out_dir = os.path.join(self.data_dir, "character", "generated")
            os.makedirs(out_dir, exist_ok=True)
            out_file = os.path.join(out_dir, f"char_{uuid.uuid4().hex[:8]}.png")
            img.save(out_file, format="PNG")
            logging.info("[Magic Wardrobe] Cached generated character: %s", out_file)
            return out_file
        except Exception as e:
            logging.error("[Magic Wardrobe] Cache generated character failed: %s", e)
            return url

    def _prepare_dir(self, new_name: str, old_name: str) -> str:
        new_path = os.path.join(self.data_dir, new_name)
        os.makedirs(new_path, exist_ok=True)
        return new_path

    def _load_background_index(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.background_index_path):
            return []
        try:
            with open(self.background_index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                return []
            cleaned = []
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                tag = entry.get("tag")
                file_name = entry.get("file")
                if not tag or not file_name:
                    continue
                file_path = os.path.join(self.data_dir, "background", file_name)
                if os.path.exists(file_path):
                    cleaned.append(entry)
            return cleaned
        except Exception:
            return []

    def _load_models_config(self) -> Dict[str, Any]:
        paths = [
            os.path.join(self.data_dir, "models.json"),
            os.path.join(os.path.dirname(__file__), "models.json"),
        ]
        for path in paths:
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        return data
                except Exception as e:
                    logging.warning("[Magic Wardrobe] models.json load failed: %s", e)
        return {
            "default_model": "Qwen/Qwen-Image-Edit-2509",
            "models": {},
        }

    def _get_model_config(self, model_name: str) -> Dict[str, Any]:
        models = self.models_config.get("models") if isinstance(self.models_config, dict) else None
        if isinstance(models, dict):
            cfg = models.get(model_name)
            if isinstance(cfg, dict):
                return cfg
        return {}

    def _get_model_endpoint(self, model_cfg: Dict[str, Any]) -> str:
        endpoint = (model_cfg or {}).get("endpoint") or ""
        if endpoint:
            return endpoint
        return "https://api.siliconflow.cn/v1/images/generations"

    def _default_model_slots(self, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        model = model_name or (
            self.models_config.get("default_model")
            if isinstance(self.models_config, dict)
            else "Qwen/Qwen-Image-Edit-2509"
        )
        cfg = self._get_model_config(model)
        slots = cfg.get("slots") if isinstance(cfg, dict) else None
        if not isinstance(slots, list) or not slots:
            slots = [{"key": "image", "role": "person", "weight": 1.0}]
        normalized = []
        for slot in slots:
            if not isinstance(slot, dict):
                continue
            normalized.append(
                {
                    "key": slot.get("key") or "image",
                    "role": slot.get("role") or "person",
                    "weight": slot.get("weight", 1.0),
                    "asset": "",
                }
            )
        return normalized

    def _make_payload_preview(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        preview: Dict[str, Any] = {}
        for key, value in payload.items():
            if key.startswith("image"):
                if isinstance(value, str) and value.startswith("data:image"):
                    preview[key] = f"<base64:{len(value)}>"
                else:
                    preview[key] = "<image>"
            else:
                preview[key] = value
        return preview

    def _resolve_output_mode(self, layout: Dict[str, Any]) -> Dict[str, bool]:
        mode = (
            (layout.get("output_mode") or "").strip()
            or (self.config.get("output_mode", "") or "").strip()
            or "image_only"
        )
        mapping = {
            "image_only": {"bot_text": False, "tts": False},
            "image_text": {"bot_text": True, "tts": False},
            "image_tts": {"bot_text": False, "tts": True},
            "image_text_tts": {"bot_text": True, "tts": True},
            # 兼容旧值：对话框改为独立开关
            "image_dialogue": {"bot_text": False, "tts": False},
            "image_dialogue_tts": {"bot_text": False, "tts": True},
        }
        return mapping.get(mode, mapping["image_only"])

    def _list_presets(self) -> List[str]:
        preset_dir = os.path.join(self.data_dir, "presets")
        if not os.path.exists(preset_dir):
            return []
        presets = []
        for file_name in os.listdir(preset_dir):
            if file_name.endswith(".json"):
                presets.append(file_name.replace(".json", ""))
        return presets

    def _sync_schema_presets(self):
        schema_path = os.path.join(os.path.dirname(__file__), "_conf_schema.json")
        if not os.path.exists(schema_path):
            return
        try:
            with open(schema_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            active_template = data.get("active_template")
            if not isinstance(active_template, dict):
                return
            presets = self._list_presets()
            if not presets:
                presets = ["default"]
            preset_set = []
            for name in presets:
                if name not in preset_set:
                    preset_set.append(name)
            preset_set.sort(key=lambda x: (0 if x == "default" else 1, x))
            if active_template.get("options") == preset_set:
                return
            active_template["options"] = preset_set
            data["active_template"] = active_template
            with open(schema_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            return

    def _load_active_preset(self) -> Optional[str]:
        try:
            if os.path.exists(self._active_preset_path):
                with open(self._active_preset_path, "r", encoding="utf-8") as f:
                    name = (f.read() or "").strip()
                    return name or None
        except Exception:
            return None
        return None

    def _set_active_preset(self, name: str) -> None:
        name = (name or "").strip()
        if not name:
            return
        self._active_preset = name
        try:
            with open(self._active_preset_path, "w", encoding="utf-8") as f:
                f.write(name)
        except Exception:
            pass

    def _load_actions_data(self) -> Dict[str, Any]:
        paths = [
            os.path.join(self.data_dir, "poses.json"),
            os.path.join(os.path.dirname(__file__), "poses.json"),
        ]
        for path in paths:
            if not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
            except Exception:
                continue
        return {}

    def _fill_action_defaults(
        self,
        full_body_action: str,
        hand_gesture: str,
        pose: str,
        expression: str,
        camera_angle: str,
    ) -> Dict[str, str]:
        fields = {
            "full_body_action": (full_body_action or "").strip(),
            "hand_gesture": (hand_gesture or "").strip(),
            "pose": (pose or "").strip(),
            "expression": (expression or "").strip(),
            "camera_angle": (camera_angle or "").strip(),
        }
        if not self.actions_data:
            return fields

        presets = list(self.actions_data.get("presets", {}).values())
        preset = None
        if presets:
            preset = random.choice(presets)

        if preset:
            if not fields["full_body_action"]:
                fields["full_body_action"] = preset.get("full_body_action", "")
            if not fields["hand_gesture"]:
                fields["hand_gesture"] = preset.get("hand_gesture", "")
            if not fields["pose"]:
                fields["pose"] = preset.get("pose", "")
            if not fields["expression"]:
                fields["expression"] = preset.get("expression", "")
            if not fields["camera_angle"]:
                fields["camera_angle"] = preset.get("camera_angle", "")

        pools = {
            "full_body_action": self.actions_data.get("full_body_actions", []),
            "hand_gesture": self.actions_data.get("hand_gestures", []),
            "pose": self.actions_data.get("poses", []),
            "expression": self.actions_data.get("expressions", []),
            "camera_angle": self.actions_data.get("camera_angles", []),
        }
        for key, pool in pools.items():
            if not fields[key] and pool:
                fields[key] = random.choice(pool)

        return fields




    def _save_background_index(self):
        try:
            with open(self.background_index_path, "w", encoding="utf-8") as f:
                json.dump(self.background_index, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _sanitize_tag(self, tag: str) -> str:
        cleaned = re.sub(r"[\\/:*?\"<>|]", "", tag)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned[:20] if cleaned else "scene"

    def _extract_scene_tag(self, text: str) -> str:
        if not text:
            return ""
        match = re.search(r"[【\[]([^\]\n】]{1,20})[】\]]", text)
        if not match:
            return ""
        return self._sanitize_tag(match.group(1))

    def _strip_scene_tags(self, text: str) -> str:
        if not text:
            return text
        cleaned = re.sub(r"[【\[]([^\]\n】]{1,20})[】\]]", "", text)
        return cleaned.strip()

    def _infer_scene_tag(self, text: str) -> str:
        if not text:
            return ""
        candidates = [
            ("海滩", "海滩"),
            ("沙滩", "海滩"),
            ("海边", "海边"),
            ("泳池", "泳池"),
            ("教室", "教室"),
            ("操场", "操场"),
            ("公园", "公园"),
            ("花园", "花园"),
            ("咖啡", "咖啡厅"),
            ("图书馆", "图书馆"),
            ("卧室", "卧室"),
            ("街道", "街道"),
            ("舞台", "舞台"),
            ("办公室", "办公室"),
            ("雨", "雨天"),
            ("雪", "雪景"),
            ("夜", "夜景"),
        ]
        for key, tag in candidates:
            if key in text:
                return self._sanitize_tag(tag)
        return ""

    def _infer_time_of_day(self, text: str) -> str:
        if any(k in text for k in ["清晨", "早上", "清早"]):
            return "morning"
        if any(k in text for k in ["傍晚", "黄昏"]):
            return "evening"
        if any(k in text for k in ["夜", "晚上", "夜晚"]):
            return "night"
        return "day"

    def _infer_weather(self, text: str) -> str:
        if "雪" in text:
            return "snowy"
        if "雨" in text:
            return "rainy"
        if any(k in text for k in ["阴", "多云"]):
            return "cloudy"
        return "clear"

    def _infer_mood(self, text: str) -> str:
        if "浪漫" in text:
            return "romantic"
        if any(k in text for k in ["安静", "宁静"]):
            return "peaceful"
        if any(k in text for k in ["热闹", "活力"]):
            return "lively"
        return "neutral"

    def _get_background_entry(self, tag: str) -> Optional[Dict[str, Any]]:
        for entry in self.background_index:
            if entry.get("tag") == tag:
                file_name = entry.get("file")
                if not file_name:
                    continue
                file_path = os.path.join(self.data_dir, "background", file_name)
                if os.path.exists(file_path):
                    return entry
        return None

    def _remove_background_entry(self, file_name: str):
        original_len = len(self.background_index)
        self.background_index = [
            entry for entry in self.background_index
            if entry.get("file") != file_name
        ]
        if len(self.background_index) != original_len:
            self._save_background_index()

    def _rename_background_entry(self, old_name: str, new_name: str):
        changed = False
        for entry in self.background_index:
            if entry.get("file") == old_name:
                entry["file"] = new_name
                changed = True
        if changed:
            self._save_background_index()

    def _add_background_entry(self, tag: str, file_name: str):
        self.background_index = [
            entry for entry in self.background_index
            if entry.get("tag") != tag
        ]
        self.background_index.insert(0, {
            "tag": tag,
            "file": file_name,
        })
        limit = int(self.config.get("background_cache_limit", 10))
        if limit > 0 and len(self.background_index) > limit:
            overflow = self.background_index[limit:]
            self.background_index = self.background_index[:limit]
            for entry in overflow:
                file_path = os.path.join(self.data_dir, "background", entry.get("file", ""))
                if file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except Exception:
                        pass
        self._save_background_index()

    async def _save_background_asset(self, tag: str, url: str) -> Optional[str]:
        img = await self._download_image(url)
        if not img:
            return None
        safe_tag = self._sanitize_tag(tag)
        file_name = f"[{safe_tag}]_{uuid.uuid4().hex[:8]}.png"
        file_path = os.path.join(self.data_dir, "background", file_name)
        try:
            img.save(file_path, format="PNG")
            return file_path
        except Exception:
            return None

    async def _prepare_auto_background(self, event: AstrMessageEvent, text: str, explicit_tag: str = ""):
        if event.get_extra("_magic_wardrobe_auto_bg_done", False):
            return
        layout = self._load_layout()
        enable_ai_bg = layout.get("enable_ai_background")
        auto_bg = layout.get("auto_background")
        if enable_ai_bg is None:
            enable_ai_bg = self.config.get("enable_ai_background", False)
        if auto_bg is None:
            auto_bg = self.config.get("auto_background", False)
        if not enable_ai_bg:
            return
        if not auto_bg:
            return
        if event.get_extra("_magic_wardrobe_bg_locked", False):
            return

        tag = explicit_tag or self._extract_scene_tag(text) or self._extract_scene_tag(event.message_str or "")
        if not tag:
            tag = self._infer_scene_tag(event.message_str or text)
        if not tag:
            tag = "scene"

        entry = self._get_background_entry(tag)
        if entry:
            file_path = os.path.join(self.data_dir, "background", entry.get("file", ""))
            if os.path.exists(file_path):
                self.last_background_url = file_path
                event.set_extra("_magic_wardrobe_auto_bg_done", True)
                return

        time_of_day = self._infer_time_of_day(text)
        weather = self._infer_weather(text)
        mood = self._infer_mood(text)
        prompt = self._build_background_prompt(tag, time_of_day, weather, mood)
        background_url = await self._generate_background_image(prompt)
        if background_url.startswith("❌"):
            return
        saved_path = await self._save_background_asset(tag, background_url)
        if saved_path:
            self._add_background_entry(tag, os.path.basename(saved_path))
            self.last_background_url = saved_path
        else:
            self.last_background_url = background_url
        event.set_extra("_magic_wardrobe_auto_bg_done", True)

    def _is_allowed(self, event: AstrMessageEvent):
        """
        会话过滤逻辑：
        1. 如果未启用过滤器（use_whitelist=False），则所有会话都允许
        2. 如果启用了过滤器，支持两种过滤方式：
           - SID 过滤：通过 session_list 配置（如 default_123456）
           - QQ 号过滤：通过 qq_number_list 配置（如 123456789）
        3. 过滤模式：
           - 白名单模式（whitelist_mode=True）：仅名单内会话生效
           - 黑名单模式（whitelist_mode=False）：仅名单外会话生效
        """
        if not hasattr(event, "unified_msg_origin"):
            return True

        # 如果未启用过滤器，默认所有会话都允许
        use_filter = self.config.get("use_whitelist", False)
        if not use_filter:
            return True

        # 获取配置的名单
        session_list = self.config.get("session_list", [])
        qq_number_list = self.config.get("qq_number_list", [])

        # 如果两个名单都为空，默认允许所有会话（全开）
        if not session_list and not qq_number_list:
            return True

        sid = event.unified_msg_origin
        is_in_list = False

        # 检查 SID 是否在会话列表中
        if session_list:
            is_in_list = any(sid.startswith(item) for item in session_list)

        # 检查 QQ 号是否在 QQ 号列表中（如果SID检查未匹配）
        if not is_in_list and qq_number_list:
            # 尝试从 event 中提取 QQ 号
            # 支持多种格式：group_123456, private_123456, qq_123456 等
            qq_number = None
            
            # 从 unified_msg_origin 提取 QQ 号
            # 格式示例: "qq_group_123456", "qq_private_123456", "default_123456"
            parts = sid.split('_')
            if len(parts) >= 2:
                # 取最后一部分作为潜在的 QQ 号
                potential_qq = parts[-1]
                if potential_qq.isdigit():
                    qq_number = potential_qq
            
            # 也支持从 sender_id 获取（如果有）
            if not qq_number and hasattr(event, "sender_id"):
                qq_number = str(event.sender_id)
            
            # 检查 QQ 号是否在列表中
            if qq_number:
                is_in_list = qq_number in qq_number_list
                if is_in_list:
                    logging.info(f"[Magic Wardrobe] QQ 号 {qq_number} 在过滤列表中")

        # 白名单模式：仅名单内生效；黑名单模式：仅名单外生效
        whitelist_mode = self.config.get("whitelist_mode", True)
        return is_in_list if whitelist_mode else not is_in_list


    async def _run_webui(self):
        from quart import Quart, send_from_directory, request, jsonify
        import hypercorn.asyncio
        from hypercorn.config import Config as HyperConfig

        app = Quart(__name__)
        webui_dir = os.path.join(os.path.dirname(__file__), "webui")

        @app.route("/")
        async def index(): return await send_from_directory(webui_dir, "index.html")

        @app.route("/api/layout", methods=["GET", "POST"])
        async def api_layout():
            if request.method == "GET": 
                preset = request.args.get("name")
                return jsonify(self._load_layout(preset))
            layout = await request.json
            preset_name = layout.get("preset_name", "default")
            with open(os.path.join(self.data_dir, "presets", f"{preset_name}.json"), "w", encoding="utf-8") as f:
                json.dump(layout, f, indent=2, ensure_ascii=False)
            self._sync_schema_presets()
            self._set_active_preset(preset_name)
            return jsonify({"status": "success"})

        @app.route("/api/presets/new", methods=["POST"])
        async def api_new_preset():
            data = await request.json
            name = data.get("name", "new_preset")
            path = os.path.join(self.data_dir, "presets", f"{name}.json")
            if os.path.exists(path): return jsonify({"status": "error", "message": "exists"})
            # 基础模板
            base = {"canvas_width": 1280, "canvas_height": 720, "box_width": 800, "box_height": 220, "box_left": 240, "box_top": 450, "box_color": "#000000b4", "radius": 20}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(base, f, indent=2, ensure_ascii=False)
            self._sync_schema_presets()
            return jsonify({"status": "success"})

        @app.route("/api/assets", methods=["GET"])
        async def api_assets():
            def get_files(d, ext=(".png", ".jpg", ".jpeg", ".webp")):
                p = os.path.join(self.data_dir, d)
                if not os.path.exists(p): os.makedirs(p)
                return [f for f in os.listdir(p) if f.lower().endswith(ext)]

            return jsonify({
                "backgrounds": get_files("background"),
                "characters": get_files("character"),
                "borders": get_files("border"),
                "components": get_files("components"),
                "fonts": get_files("fonts", (".ttf", ".otf")),
                "presets": [f.replace(".json", "") for f in os.listdir(os.path.join(self.data_dir, "presets")) if f.endswith(".json")]
            })

        @app.route("/api/models", methods=["GET"])
        async def api_models():
            return jsonify(self.models_config or {})

        @app.route("/api/poses", methods=["GET"])
        async def api_poses():
            return jsonify(self.actions_data or {})

        @app.route("/api/payload-preview", methods=["GET"])
        async def api_payload_preview():
            return jsonify({
                "payload": self.last_payload_preview or {},
                "timestamp": self.last_payload_at,
            })

        @app.route("/api/prompt-preview", methods=["GET"])
        async def api_prompt_preview():
            return jsonify({
                "preview": self.last_prompt_preview or {},
                "timestamp": self.last_payload_at,
            })

        @app.route("/api/upload", methods=["POST"])
        async def api_upload():
            files = await request.files
            form = await request.form
            t = form.get("type", "background")
            if t in ("component", "components"):
                t = "components"
            if t in ("font", "fonts"):
                t = "fonts"
            if "file" not in files: return jsonify({"status": "error", "message": "no file"})
            f = files["file"]
            path = os.path.join(self.data_dir, t)
            if not os.path.exists(path): os.makedirs(path)
            await f.save(os.path.join(path, f.filename))
            return jsonify({"status": "success"})

        @app.route("/api/delete", methods=["POST"])
        async def api_delete():
            data = await request.json
            t = data.get("type")
            name = data.get("name")
            if not t or not name: return jsonify({"status": "error", "message": "missing params"})
            if t in ("component", "components"):
                t = "components"
            if t in ("font", "fonts"):
                t = "fonts"
            path = os.path.join(self.data_dir, t, name)
            if os.path.exists(path):
                os.remove(path)
                if t == "background":
                    self._remove_background_entry(name)
                return jsonify({"status": "success"})
            return jsonify({"status": "error", "message": "file not found"})

        @app.route("/api/rename", methods=["POST"])
        async def api_rename():
            data = await request.json
            t = data.get("type")
            name = data.get("name")
            new_name = data.get("new_name")
            if not t or not name or not new_name:
                return jsonify({"status": "error", "message": "missing params"})
            if t in ("component", "components"):
                t = "components"
            if t in ("font", "fonts"):
                t = "fonts"
            new_name = os.path.basename(new_name)
            if not new_name:
                return jsonify({"status": "error", "message": "invalid new name"})
            src = os.path.join(self.data_dir, t, name)
            if not os.path.exists(src):
                return jsonify({"status": "error", "message": "file not found"})
            root, ext = os.path.splitext(name)
            new_root, new_ext = os.path.splitext(new_name)
            if not new_ext:
                new_name = new_root + ext
            dst = os.path.join(self.data_dir, t, new_name)
            if os.path.exists(dst):
                return jsonify({"status": "error", "message": "target exists"})
            os.rename(src, dst)
            if t == "background":
                self._rename_background_entry(name, new_name)
            return jsonify({"status": "success"})

        @app.route("/api/raw/<type>/<name>")
        async def api_raw(type, name): 
            try:
                from urllib.parse import unquote
                name = unquote(name)
            except Exception:
                pass
            if type in ("component", "components"):
                type = "components"
            if type in ("font", "fonts"):
                type = "fonts"
            return await send_from_directory(os.path.join(self.data_dir, type), name)

        port = self.config.get('webui_port', 18765)
        conf = HyperConfig()
        conf.bind = [f"0.0.0.0:{port}"]
        try:
            await hypercorn.asyncio.serve(app, conf)
        except Exception as e:
            logging.error(f"Hypercorn error on port {port}: {e}")

    def _load_layout(self, preset=None):
        if not preset:
            preset = self._active_preset or self.config.get("active_template", "default")
        path = os.path.join(self.data_dir, "presets", f"{preset}.json")
        if not os.path.exists(path): path = os.path.join(self.data_dir, "default_layout.json")
        
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    if "text_overlays" not in data:
                        data["text_overlays"] = []
                    # 角色立绘位置与尺寸
                    if "character_width" not in data:
                        data["character_width"] = 600
                    if "character_left" not in data:
                        data["character_left"] = 340
                    if "model_name" not in data:
                        data["model_name"] = (
                            self.models_config.get("default_model")
                            if isinstance(self.models_config, dict)
                            else "Qwen/Qwen-Image-Edit-2509"
                        )
                    if "model_slots" not in data or not isinstance(data.get("model_slots"), list):
                        data["model_slots"] = self._default_model_slots(data.get("model_name"))
                    if "output_mode" not in data:
                        data["output_mode"] = "image_only"
                    if "enable_dialogue_box" not in data:
                        data["enable_dialogue_box"] = True
                    if "ui_theme" not in data:
                        data["ui_theme"] = "blue"
                    if "prompt_extra" not in data:
                        data["prompt_extra"] = ""
                    if "negative_prompt" not in data:
                        data["negative_prompt"] = ""
                    if "negative_prompt_extra" not in data:
                        data["negative_prompt_extra"] = ""
                    if "use_char_as_full" not in data:
                        data["use_char_as_full"] = False
                    if "output_image_size" not in data:
                        data["output_image_size"] = "1024x1024"
                    if "num_inference_steps" not in data:
                        data["num_inference_steps"] = 20
                    if "cfg_scale" not in data:
                        data["cfg_scale"] = 4.0
                    if "background_model" not in data:
                        data["background_model"] = "Kwai-Kolors/Kolors"
                    if "enable_ai_background" not in data:
                        data["enable_ai_background"] = False
                    if "auto_background" not in data:
                        data["auto_background"] = False
                    if "use_char_as_full" not in data:
                        data["use_char_as_full"] = True
                    if "show_name_box" not in data:
                        data["show_name_box"] = True
                    if "name_text" not in data:
                        data["name_text"] = "角色名"
                    if "name_box_width" not in data:
                        data["name_box_width"] = 220
                    if "name_box_height" not in data:
                        data["name_box_height"] = 56
                    if "name_box_left" not in data:
                        data["name_box_left"] = int(data.get("box_left", 520)) + 20
                    if "name_box_top" not in data:
                        data["name_box_top"] = max(0, int(data.get("box_top", 160)) - 70)
                    if "name_box_color" not in data:
                        data["name_box_color"] = "#d7e8ffcc"
                    if "name_text_color" not in data:
                        data["name_text_color"] = "#1e3a8a"
                    if "name_font_size" not in data:
                        data["name_font_size"] = 26
                    if "name_padding" not in data:
                        data["name_padding"] = 12
                    if "name_radius" not in data:
                        data["name_radius"] = 16
                return data
            except: pass
        return {"canvas_width": 1280, "canvas_height": 720, "background_color": "#2c3e50", "box_width": 700, "box_height": 340, "box_left": 520, "box_top": 160, "character_width": 520, "character_left": 0, "font_size": 32, "text_color": "#ffffff", "radius": 20, "padding": 20, "text_overlays": [], "use_char_as_full": False, "show_name_box": True, "name_text": "角色名", "name_box_width": 220, "name_box_height": 56, "name_box_left": 540, "name_box_top": 90, "name_box_color": "#d7e8ffcc", "name_text_color": "#1e3a8a", "name_font_size": 26, "name_padding": 12, "name_radius": 16}


    @filter.on_llm_request(priority=-1000)
    async def handle_llm_request(self, event: AstrMessageEvent, request):
        if not self.config.get("enable_tool", True):
            return
        if event.get_extra("_magic_wardrobe_direct_outfit", False):
            return

        user_text = event.message_str or ""
        if not user_text.strip():
            return

        if not self._should_trigger_outfit(user_text):
            if self.config.get("guard_outfit_chatter", True):
                guard_hint = (
                    "Unless the user explicitly asks for outfits, images, or scene changes, "
                    "respond normally and do not mention changing clothes, background, or generating images. "
                    "Do not call any tools unless the user explicitly asks for outfits or images."
                )
                if guard_hint not in request.system_prompt:
                    request.system_prompt = (request.system_prompt + "\n" + guard_hint).strip()
            return

        event.set_extra("enable_streaming", False)
        event.set_extra("_magic_wardrobe_outfit_requested", True)
        event.set_extra("_magic_wardrobe_outfit_text", self._extract_clothing_desc(user_text))
        hint = (
            "When the user asks to view or change clothing or outfit, "
            "call the change_outfit tool. Use the user's request as the clothing description. "
            "If the user did not specify actions, pick suitable full_body_action, hand_gesture, "
            "pose, expression, and camera_angle yourself. "
            "Call only change_outfit. Do not call any other tools."
        )
        if hint not in request.system_prompt:
            request.system_prompt = (request.system_prompt + "\n" + hint).strip()
        layout = self._load_layout()
        enable_ai_bg = layout.get("enable_ai_background") if isinstance(layout, dict) else None
        auto_bg = layout.get("auto_background") if isinstance(layout, dict) else None
        if (enable_ai_bg if enable_ai_bg is not None else self.config.get("enable_ai_background", False)) and (auto_bg if auto_bg is not None else self.config.get("auto_background", False)):
            scene_hint = (
                "If the user mentions a scene or location, prefix your reply with a short scene tag in 【】, "
                "for example: 【海边】 or 【教室】. Keep the tag short."
            )
            if scene_hint not in request.system_prompt:
                request.system_prompt = (request.system_prompt + "\n" + scene_hint).strip()
        logging.info("[Magic Wardrobe] Injected tool prompt for outfit request")

    def _should_trigger_outfit(self, text: str) -> bool:
        lowered = text.lower()
        clothing_keywords = [
            "女仆装",
            "女仆",
            "水手服",
            "泳装",
            "礼服",
            "制服",
            "洛丽塔",
            "黑丝",
            "旗袍",
            "汉服",
            "睡衣",
            "便服",
            "私服",
        ]
        english = ["outfit", "dress", "swimsuit", "maid", "uniform", "lolita", "sailor"]
        if any(k in text for k in clothing_keywords):
            return True
        if any(k in lowered for k in english):
            return True

        image_verbs = [
            "看看",
            "来张",
            "来个",
            "来一张",
            "来套",
            "换装",
            "换个",
            "换一套",
            "穿上",
            "穿一身",
            "穿个",
            "画",
            "生成",
            "出图",
            "发一张",
            "给我看",
        ]
        image_nouns = [
            "图",
            "图片",
            "照片",
            "立绘",
            "插画",
            "图像",
        ]
        return any(v in text for v in image_verbs) and any(n in text for n in image_nouns)


    @filter.regex(r"(看看|来张|来个|换|穿|泳装|水手服|女仆|礼服|制服|洛丽塔|黑丝|旗袍)")
    async def handle_direct_outfit(self, event: AstrMessageEvent):
        if not self.config.get("enable_tool", True):
            return
        if self.config.get("prefer_llm_action", True):
            return
        if not self.config.get("only_change_on_request", True):
            return
        if not self._is_allowed(event):
            return

        text = (event.message_str or "").strip()
        if not text:
            return
        if not self._should_trigger_outfit(text):
            return

        clothing = self._extract_clothing_desc(text)
        if not clothing:
            return
        try:
            event.set_extra("enable_streaming", False)
            await self._run_direct_outfit(event, clothing)
        except Exception as e:
            logging.error(f"[Magic Wardrobe] Direct outfit failed: {e}", exc_info=True)

    def _extract_clothing_desc(self, text: str) -> str:
        cleaned = re.sub(r"^(看看|来张|来个|换|穿|给我|要|来套)+", "", text).strip()
        cleaned = cleaned.strip(" \t\r\n,.;:!?，。？！")
        return cleaned or text.strip()


    def _parse_wardrobe_request(self, clothing: str):
        """???????????????(??, ??)?

        ????????/??????????(None, clothing_text)?
        ??????? change_outfit ???
        """
        clothing_text = (clothing or "").strip()
        return None, clothing_text


    async def _run_direct_outfit(self, event: AstrMessageEvent, clothing: str) -> bool:
        if not clothing:
            return False

        action_fields = self._fill_action_defaults("", "", "", "", "")
        prompt_data = {
            "clothing": clothing,
            "full_body_action": action_fields["full_body_action"],
            "hand_gesture": action_fields["hand_gesture"],
            "pose": action_fields["pose"],
            "expression": action_fields["expression"],
            "camera_angle": action_fields["camera_angle"],
        }

        logging.info("[Magic Wardrobe] Direct outfit request: %s", clothing)
        image_result = await self._generate_ai_image(prompt_data)
        if not image_result or image_result.startswith("❌"):
            logging.warning("[Magic Wardrobe] Direct outfit failed: %s", image_result)
            event.set_extra("_magic_wardrobe_failed", True)
            event.set_extra("_magic_wardrobe_failed_reason", image_result or "direct outfit failed")
            return False

        cached_url = await self._cache_generated_character(image_result)
        self._save_last_url(cached_url)
        self._apply_generated_character_to_layout(cached_url)
        logging.info(
            "[Magic Wardrobe] Direct outfit updated image: %s",
            cached_url[:80],
        )
        event.set_extra("_magic_wardrobe_direct_outfit", True)
        event.set_extra("_magic_wardrobe_tool_used", True)
        return True

    @filter.on_decorating_result(priority=-1000)
    async def handle_decorating(self, event: AstrMessageEvent, *args, **kwargs):
        """装饰结果处理：将 LLM 回复合成到图片中"""
        if event.get_extra("_magic_wardrobe_streaming_fallback", False):
            return
        if event.get_extra("_magic_wardrobe_image_sent", False):
            return

        result = event.get_result()
        if not result:
            return

        # 关键修改：强制修改 result_content_type 为 LLM_RESULT，防止被跳过
        try:
            from astrbot.core.message.message_event_result import ResultContentType
            if result.result_content_type == ResultContentType.STREAMING_FINISH:
                result.result_content_type = ResultContentType.LLM_RESULT
                logging.info("[Magic Wardrobe] 已将 STREAMING_FINISH 修改为 LLM_RESULT")
        except Exception as e:
            logging.warning(f"[Magic Wardrobe] 修改 result_content_type 失败: {e}")

        # 强制继续传播，不要停止事件
        try:
            event.continue_event()
        except Exception:
            pass

        # 如果已被停止，强制继续
        try:
            if event.is_stopped():
                logging.info("[Magic Wardrobe] Event was stopped, forcing continue")
                event.continue_event()
        except Exception:
            pass

        # 检查是否是命令直接生成的图片，如果是则跳过
        if hasattr(result, '_skip_magic_wardrobe') and result._skip_magic_wardrobe:
            logging.info("[Magic Wardrobe] 检测到命令直接生成的图片，跳过装饰器处理")
            return

        if not self.config.get("enable_render", True):
            return
        if not self._is_allowed(event):
            return

        # 检查是否已经是图片，避免无限循环合成
        if any(isinstance(m, Comp.Image) or type(m).__name__ == "Image" for m in result.chain):
            return

        # 如果开启了全局拦截，或者是因为 LLM 回复触发
        is_llm = result.is_llm_result()
        intercept_all = self.config.get("intercept_all_to_image", False)

        if not (intercept_all or is_llm):
            return

        if not intercept_all and not event.get_extra("_magic_wardrobe_outfit_requested", False):
            return

        plain_text = self._sanitize_render_text(result.get_plain_text())
        if not plain_text.strip():
            return
        if event.get_extra("_magic_wardrobe_failed", False):
            fallback_path = self._get_fallback_image_path()
            if fallback_path:
                result.chain = [Comp.Image.fromFileSystem(fallback_path)]
                logging.info("[Magic Wardrobe] Fallback image sent due to generation failure")
                return
        if event.get_extra("_magic_wardrobe_outfit_requested", False) and not event.get_extra("_magic_wardrobe_tool_used", False):
            clothing = event.get_extra("_magic_wardrobe_outfit_text") or self._extract_clothing_desc(event.message_str or "")
            try:
                await self._run_direct_outfit(event, clothing)
            except Exception as e:
                logging.warning(f"[Magic Wardrobe] Direct outfit fallback failed: {e}")
        explicit_tag = self._extract_scene_tag(plain_text) or self._extract_scene_tag(event.message_str or "")
        await self._prepare_auto_background(event, plain_text, explicit_tag)
        plain_text = self._strip_scene_tags(plain_text)
        if not plain_text.strip():
            return

        threshold = self.config.get("render_threshold", 0)
        if threshold > 0 and len(plain_text) < threshold:
            return

        try:
            # 使用最后一次生成的形象
            char_url = self.last_char_url

            logging.info(f"[Magic Wardrobe] 开始合成图片, char_url={char_url[:50] if char_url else 'None'}")

            # 合成图片
            layout = self._load_layout()
            output = self._resolve_output_mode(layout)
            render_text = plain_text if layout.get("enable_dialogue_box", True) else ""

            final_image_path = await self._composite_image(char_url, render_text)

            logging.info(f"[Magic Wardrobe] Image composed: {final_image_path}")

            audio_path = None
            if output.get("tts") and self._should_tts_for_image():
                audio_path = await self._synth_tts_audio(plain_text)

            chain = [Comp.Image.fromFileSystem(final_image_path)]
            if output.get("bot_text"):
                chain.append(Comp.Plain(plain_text))
            if audio_path:
                chain.append(Record(file=str(audio_path)))
                event.set_extra("_magic_wardrobe_record_after", True)
            result.chain = chain
            event.set_extra("_magic_wardrobe_image_sent", True)
        except Exception as e:
            logging.error(f"[Magic Wardrobe] 渲染失败: {e}", exc_info=True)

    @filter.on_llm_response(priority=-1000)
    async def handle_llm_response(self, event: AstrMessageEvent, response):
        if not self.config.get("enable_render", True):
            return
        if not self._is_allowed(event):
            return
        if event.get_extra("_magic_wardrobe_failed", False):
            return

        config = self.context.get_config(event.unified_msg_origin)
        streaming_response = config.get("provider_settings", {}).get(
            "streaming_response",
            False,
        )
        enable_streaming = event.get_extra("enable_streaming")
        if enable_streaming is not None:
            streaming_response = bool(enable_streaming)
        if not streaming_response:
            return

        if not self.config.get("intercept_all_to_image", False) and not event.get_extra("_magic_wardrobe_outfit_requested", False):
            return

        if event.get_extra("_magic_wardrobe_streaming_fallback", False):
            return
        if event.get_extra("_magic_wardrobe_image_sent", False):
            return

        text = ""
        if response and getattr(response, "result_chain", None):
            text = response.result_chain.get_plain_text()
        if not text and response and getattr(response, "completion_text", None):
            text = response.completion_text or ""
        text = self._sanitize_render_text(text)
        if not text.strip():
            return
        if event.get_extra("_magic_wardrobe_outfit_requested", False) and not event.get_extra("_magic_wardrobe_tool_used", False):
            clothing = event.get_extra("_magic_wardrobe_outfit_text") or self._extract_clothing_desc(event.message_str or "")
            try:
                await self._run_direct_outfit(event, clothing)
            except Exception as e:
                logging.warning(f"[Magic Wardrobe] Direct outfit fallback failed: {e}")
        explicit_tag = self._extract_scene_tag(text) or self._extract_scene_tag(event.message_str or "")
        await self._prepare_auto_background(event, text, explicit_tag)
        text = self._strip_scene_tags(text)
        if not text.strip():
            return

        threshold = self.config.get("render_threshold", 0)
        if threshold > 0 and len(text) < threshold:
            return

        try:
            char_url = self.last_char_url
            logging.info(
                f"[Magic Wardrobe] Streaming fallback render, char_url={char_url[:50] if char_url else 'None'}"
            )
            layout = self._load_layout()
            output = self._resolve_output_mode(layout)
            render_text = text if layout.get("enable_dialogue_box", True) else ""
            final_image_path = await self._composite_image(char_url, render_text)
            from astrbot.core.message.message_event_result import MessageChain

            image_chain = MessageChain()
            audio_path = None
            if output.get("tts") and self._should_tts_for_image():
                audio_path = await self._synth_tts_audio(text)
            chain = [Comp.Image.fromFileSystem(final_image_path)]
            if output.get("bot_text"):
                chain.append(Comp.Plain(text))
            if audio_path:
                chain.append(Record(file=str(audio_path)))
            image_chain.chain = chain
            event.set_extra("_magic_wardrobe_image_sent", True)
            await self.context.send_message(event.unified_msg_origin, image_chain)

            event.set_extra("_magic_wardrobe_streaming_fallback", True)
        except Exception as e:
            logging.error(f"[Magic Wardrobe] Streaming fallback failed: {e}", exc_info=True)

    @filter.on_decorating_result(priority=-900)
    async def handle_plain_tts(self, event: AstrMessageEvent, *args, **kwargs):
        if event.get_extra("_magic_wardrobe_streaming_fallback", False):
            return
        if not self._is_allowed(event):
            return
        if not self._can_tts():
            return

        result = event.get_result()
        if not result or not result.is_llm_result():
            return

        if any(isinstance(m, Comp.Image) or type(m).__name__ == "Image" for m in result.chain):
            return
        if Record and any(isinstance(m, Record) or type(m).__name__ == "Record" for m in result.chain):
            return

        plain_text = self._sanitize_render_text(result.get_plain_text())
        if not plain_text.strip():
            return
        if not self._should_tts_for_text(plain_text):
            return

        audio_path = await self._synth_tts_audio(plain_text)
        if not audio_path:
            return
        mode = (self.config.get("tts_plain_mode", "text_with_voice") or "").strip()
        if mode == "voice_only":
            new_chain = []
            for comp in result.chain:
                if isinstance(comp, Comp.Plain) or type(comp).__name__ == "Plain":
                    continue
                new_chain.append(comp)
            new_chain.append(Record(file=str(audio_path)))
            result.chain = new_chain
        else:
            result.chain.append(Record(file=str(audio_path)))
        logging.info("[Magic Wardrobe] Added TTS for plain response")

    if hasattr(filter, "after_message_sent"):
        @filter.after_message_sent(priority=-1000)
        async def after_message_sent(self, event: AstrMessageEvent):
            umo = event.unified_msg_origin
            pending = self._pending_tts_tasks.pop(umo, None)
            if pending and not pending.done():
                pending.cancel()

            text = self._pending_tts.pop(umo, "")
            if not text:
                return

            try:
                logging.info("[Magic Wardrobe] Sending TTS after image")
                await self._send_tts_voice(event, text)
            except Exception as e:
                logging.error(f"[Magic Wardrobe] TTS send failed: {e}", exc_info=True)
            finally:
                event.set_extra("_magic_wardrobe_tts_pending", False)

    @command("魔法衣橱")
    async def command_help(self, event: AstrMessageEvent):
        help_text = "✨ AI 魔法衣橱 使用指南\n" \
                    "直接对话让 AI 生成图片，例如：\n" \
                    "- '画一张你微笑的图片'\n" \
                    "- '换上红色连衣裙'\n" \
                    "- '来张泳装自拍'\n\n" \
                    "管理后台：访问 WebUI 进行布局配置"
        yield event.plain_result(help_text)


    @llm_tool("change_outfit")
    async def change_outfit(
        self,
        event: AstrMessageEvent,
        clothing: str,
        full_body_action: str = "",
        hand_gesture: str = "",
        pose: str = "",
        expression: str = "",
        camera_angle: str = ""
    ) -> str:
        """
        核心绘图/换装工具。当用户要求你【生成图片】、【画图】、【自拍】、【换装】、【变身】或【展示动作表情】时必须调用。

        Args:
            clothing(string): 必须提供。角色的着装描述。如果不需换装，请根据当前人设描述默认服装，或根据场景自由发挥(如: "Casual outfit", "Swimsuit")。
            full_body_action(string): 全身动作描述（如：站立、坐着、奔跑、跳跃、躺着）。
            hand_gesture(string): 手部动作描述（如：挥手、比心、托腮、拿着书、双手合十）。
            pose(string): 姿势描述（如：侧身、回头、弯腰、伸懒腰）。
            expression(string): 角色的表情描述（如：害羞、微笑、生气、惊讶、思考）。
            camera_angle(string): 镜头角度（如：正面、侧面、背面、俯视、仰视、特写）。
        """
        try:
            print(f"[Magic Wardrobe DEBUG] change_outfit 工具被调用", flush=True)
            print(f"[Magic Wardrobe DEBUG] 参数 - clothing:{clothing}, expression:{expression}, pose:{pose}", flush=True)
            logging.info(f"[Magic Wardrobe] change_outfit 工具被调用")
            logging.info(f"[Magic Wardrobe] 参数 - clothing:{clothing}, expression:{expression}, pose:{pose}")

            enable_tool_value = self.config.get("enable_tool", True)
            print(f"[Magic Wardrobe DEBUG] enable_tool = {enable_tool_value}", flush=True)

            if not enable_tool_value:
                print(f"[Magic Wardrobe DEBUG] 工具已被禁用，返回", flush=True)
                logging.warning("[Magic Wardrobe] 工具已被禁用")
                return "工具已被禁用"

            print(f"[Magic Wardrobe DEBUG] 开始封装提示词数据", flush=True)
            wardrobe_asset, wardrobe_tag = self._parse_wardrobe_request(clothing)
            if wardrobe_tag:
                clothing = wardrobe_tag

            action_fields = self._fill_action_defaults(
                full_body_action,
                hand_gesture,
                pose,
                expression,
                camera_angle,
            )
            full_body_action = action_fields["full_body_action"]
            hand_gesture = action_fields["hand_gesture"]
            pose = action_fields["pose"]
            expression = action_fields["expression"]
            camera_angle = action_fields["camera_angle"]

            # 封装提示词数据
            prompt_data = {
                "clothing": clothing,
                "full_body_action": full_body_action,
                "hand_gesture": hand_gesture,
                "pose": pose,
                "expression": expression,
                "camera_angle": camera_angle
            }
            if wardrobe_asset:
                prompt_data["_clothing_asset"] = wardrobe_asset
            else:
                prompt_data["_ignore_clothing_asset"] = True

            print(f"[Magic Wardrobe DEBUG] 准备调用 _generate_ai_image", flush=True)
            logging.info(f"[Magic Wardrobe] 开始调用 AI 生成图片")
            image_result = await self._generate_ai_image(prompt_data)
            print(f"[Magic Wardrobe DEBUG] AI 生成结果: {image_result[:100] if image_result else 'None'}", flush=True)
            logging.info(f"[Magic Wardrobe] AI 生成结果: {image_result[:100] if image_result else 'None'}")

            if image_result.startswith("❌"):
                event.set_extra("_magic_wardrobe_failed", True)
                event.set_extra("_magic_wardrobe_failed_reason", image_result)
                return image_result

            cached_url = await self._cache_generated_character(image_result)
            self._save_last_url(cached_url)

            # 关键修改：返回成功消息给 LLM，让它知道图片已生成
            # LLM 会根据这个结果生成一个中文回复
            # 然后 on_decorating_result 装饰器会拦截这个回复并合成到图片中
            logging.info(f"[Magic Wardrobe] AI图片生成完成: {cached_url[:50]}...")
            print(f"[Magic Wardrobe DEBUG] 工具执行完成，返回成功消息", flush=True)

            # 返回一个简洁的成功消息，告诉 LLM 图片已生成
            # LLM 会基于这个结果生成一个自然的中文回复
            action_desc = f"{full_body_action} {hand_gesture} {pose}".strip()
            event.set_extra("_magic_wardrobe_tool_used", True)
            return f"图片已成功生成。服装：{clothing}，表情：{expression}，动作：{action_desc}，镜头：{camera_angle}。请用中文自然地回复用户，描述你换上新装扮后的感受。"

        except Exception as e:
            logging.error(f"[Magic Wardrobe] change_outfit 发生异常: {e}", exc_info=True)
            print(f"[Magic Wardrobe DEBUG] 发生异常: {e}", flush=True)
            return f"❌ 换装失败: {str(e)}"

    @llm_tool("generate_scene_background")
    async def generate_scene_background(self, event: AstrMessageEvent, scene_description: str, time_of_day: str = "day", weather: str = "clear", mood: str = "neutral") -> str:
        """
        AI 场景背景生成工具。根据对话内容和场景需求，自动生成合适的背景图片。

        当对话涉及特定场景、地点、环境时调用此工具（如：教室、海边、咖啡厅、公园等）。

        Args:
            scene_description(string): 必须提供。场景的核心描述（如：classroom, beach, cafe, park, bedroom）。
            time_of_day(string): 时间段（如：morning, day, evening, night）。
            weather(string): 天气状况（如：clear, cloudy, rainy, snowy）。
            mood(string): 场景氛围（如：peaceful, lively, romantic, mysterious）。
        """
        try:
            # 检查是否启用 AI 背景生成
            layout = self._load_layout()
            enable_ai_bg = layout.get("enable_ai_background") if isinstance(layout, dict) else None
            if not (enable_ai_bg if enable_ai_bg is not None else self.config.get("enable_ai_background", False)):
                logging.info("[Magic Wardrobe] AI 背景生成未启用，使用静态背景")
                return "AI 背景生成功能未启用，将使用静态背景库。"

            logging.info(f"[Magic Wardrobe] generate_scene_background 被调用")
            logging.info(f"[Magic Wardrobe] 场景参数 - scene:{scene_description}, time:{time_of_day}, weather:{weather}, mood:{mood}")

            # 构建背景生成提示词
            background_prompt = self._build_background_prompt(scene_description, time_of_day, weather, mood)

            # 调用背景生成 API
            background_url = await self._generate_background_image(background_prompt)

            if background_url.startswith("❌"):
                return background_url

            scene_tag = self._sanitize_tag(scene_description)
            saved_path = await self._save_background_asset(scene_tag, background_url)
            if saved_path:
                self._add_background_entry(scene_tag, os.path.basename(saved_path))
                self.last_background_url = saved_path
            else:
                self.last_background_url = background_url
            event.set_extra("_magic_wardrobe_bg_locked", True)

            logging.info(f"[Magic Wardrobe] 背景生成完成: {background_url[:50]}...")
            return f"场景背景已生成：{scene_description}（{time_of_day}, {weather}, {mood}）。"

        except Exception as e:
            logging.error(f"[Magic Wardrobe] generate_scene_background 发生异常: {e}", exc_info=True)
            return f"❌ 背景生成失败: {str(e)}"

    def _build_background_prompt(self, scene: str, time: str, weather: str, mood: str) -> str:
        """构建背景生成的提示词"""
        prompt_parts = [
            f"{scene} background",
            f"{time} time",
            f"{weather} weather",
            f"{mood} atmosphere",
            "high quality, detailed, anime style, galgame background",
            "no characters, empty scene, landscape view"
        ]
        return ", ".join(prompt_parts)

    async def _generate_background_image(self, prompt: str) -> str:
        layout = self._load_layout()
        """调用 API 生成背景图片"""
        api_key = self.config.get("api_key", "")
        api_channel = self.config.get("api_channel", "siliconflow")

        # 根据渠道选择 API Key
        if api_channel == "modelscope":
            api_key = self.config.get("modelscope_api_key", "")
            if not api_key:
                return "❌ 请先在配置中填入 ModelScope API Key。"
        elif not api_key:
            return "❌ 请先在配置中填入 SiliconFlow API Key。"

        model = (layout.get("background_model") or "").strip() or self.config.get("background_model", "black-forest-labs/FLUX.1-schnell")

        steps = layout.get("num_inference_steps") or 20

        # 构建 API 请求
        if api_channel == "siliconflow":
            api_url = "https://api.siliconflow.cn/v1/images/generations"
        else:  # modelscope
            api_url = "https://api.modelscope.cn/v1/images/generations"

        payload = {
            "model": model,
            "prompt": prompt,
            "image_size": "1280x720",
            "num_inference_steps": int(steps),
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(api_url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logging.error(f"[Magic Wardrobe] 背景生成 API 错误: {error_text}")
                        return f"❌ API 请求失败 ({resp.status}): {error_text[:100]}"

                    result = await resp.json()
                    
                    # 兼容多种 API 返回格式
                    # SiliconFlow: {"images": [{"url": "..."}]}
                    # 或: {"data": [{"url": "..."}]}
                    images_data = result.get("images") or result.get("data")
                    
                    if images_data and len(images_data) > 0:
                        image_item = images_data[0]
                        
                        # 支持多种格式
                        if isinstance(image_item, dict):
                            if "url" in image_item:
                                return image_item["url"]
                            elif "b64_json" in image_item:
                                return f"data:image/png;base64,{image_item['b64_json']}"
                        elif isinstance(image_item, str):
                            # 直接返回URL字符串
                            return image_item
                    
                    logging.error(f"[Magic Wardrobe] 背景API返回格式异常: {result}")
                    return "❌ API 返回格式异常，未找到图片数据。"

        except asyncio.TimeoutError:
            return "❌ 背景生成超时，请稍后重试。"
        except Exception as e:
            logging.error(f"[Magic Wardrobe] 背景生成异常: {e}", exc_info=True)
            return f"❌ 背景生成失败: {str(e)}"

    async def _generate_ai_image(self, prompt_input: Any):
        api_key = self.config.get("api_key", "")
        if not api_key: return "❌ 请先在配置中填入 SiliconFlow API Key。"
        
        if isinstance(prompt_input, str):
            prompt_data = {"clothing": prompt_input, "upper_action": "", "lower_action": "", "expression": ""}
        else:
            prompt_data = prompt_input

        layout = self._load_layout()
        char_asset = layout.get("character_asset")

        # 使用布局或 models.json 的模型
        layout_model = layout.get("model_name") if isinstance(layout, dict) else None
        model = (
            layout_model
            or (self.models_config.get("default_model") if isinstance(self.models_config, dict) else None)
            or "Qwen/Qwen-Image-Edit-2509"
        )
        model_cfg = self._get_model_config(model)
        supports = model_cfg.get("supports") if isinstance(model_cfg, dict) else {}
        if not isinstance(supports, dict):
            supports = {}
        is_image_edit = supports.get("image_edit")
        if is_image_edit is None:
            is_image_edit = "Qwen-Image-Edit" in model
        
        # 构建基础 payload，严格按照官网 API 规范
        # 参考: https://docs.siliconflow.cn/api-reference/image-generation
        # ⚠️ Qwen-Image-Edit 不支持 strength/image_size 参数，仅支持 cfg
        payload = {"model": model, "prompt": ""}
        layout_steps = layout.get("num_inference_steps")
        steps = layout_steps if layout_steps is not None else self.config.get("num_inference_steps", 20)
        if steps:
            payload["num_inference_steps"] = steps
        if supports.get("cfg", True):
            layout_cfg = layout.get("cfg_scale")
            cfg_val = layout_cfg if layout_cfg is not None else self.config.get("cfg_scale", 4.0)
            payload["cfg"] = cfg_val

        # 处理图片转 Base64 逻辑
        def to_b64(path):
            with PILImage.open(path).convert("RGB") as img:
                # 不再缩小图片，保持原始尺寸上传给Qwen
                # 这样Qwen返回的图片尺寸和本地素材一致
                buf = BytesIO()
                img.save(buf, format="JPEG")
                return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
        def pil_to_b64(img: PILImage.Image) -> str:
            img = img.convert("RGB")
            # 不再缩小图片，保持原始尺寸
            buf = BytesIO()
            img.save(buf, format="JPEG")
            return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
        # ??????? models.json ???
        has_identity_ref = False
        resolved_images: List[Dict[str, Any]] = []
        slots = model_cfg.get("slots") if isinstance(model_cfg, dict) else None
        if not isinstance(slots, list) or not slots:
            slots = [{"key": "image", "role": "person"}]
        max_images = int(supports.get("max_images", 1) or 1)

        layout_slots = layout.get("model_slots") if isinstance(layout, dict) else None
        layout_slots_map: Dict[str, Any] = {}
        if isinstance(layout_slots, list):
            for s in layout_slots:
                if isinstance(s, dict) and s.get("key"):
                    layout_slots_map[s.get("key")] = s

        for slot in slots:
            if not isinstance(slot, dict):
                continue
            key = slot.get("key")
            role = (slot.get("role") or "person").strip().lower()
            layout_slot = layout_slots_map.get(key)
            if layout_slot:
                role = (layout_slot.get("role") or role).strip().lower()
            slot_asset = ""
            if layout_slot:
                slot_asset = (layout_slot.get("asset") or "").strip()
            if not key or not role:
                continue

            img_b64 = None
            if role == "person":
                if slot_asset:
                    p1 = os.path.join(self.data_dir, "character", slot_asset)
                    if os.path.exists(p1):
                        img_b64 = to_b64(p1)
                elif char_asset:
                    p1 = os.path.join(self.data_dir, "character", char_asset)
                    if os.path.exists(p1):
                        img_b64 = to_b64(p1)
                elif self.config.get("use_last_character_reference", True) and self.last_char_url:
                    try:
                        ref_img = await self._download_image(self.last_char_url)
                        if ref_img:
                            ref_img = ref_img.convert("RGB")
                            # 不再缩小，保持原始尺寸
                            buf = BytesIO()
                            ref_img.save(buf, format="JPEG")
                            img_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
                    except Exception as e:
                        logging.warning("[Magic Wardrobe] Last character reference load failed: %s", e)
                if img_b64:
                    has_identity_ref = True
            elif role == "scene":
                bg_asset = slot_asset or layout.get("background_asset")
                if slot_asset == "__ai__":
                    if getattr(self, "last_background_url", None):
                        try:
                            bg_img = await self._download_image(self.last_background_url)
                            if bg_img:
                                img_b64 = pil_to_b64(bg_img)
                        except Exception as e:
                            logging.warning(f"[Magic Wardrobe] Scene ref from AI background failed: {e}")
                elif bg_asset:
                    p_bg = os.path.join(self.data_dir, "background", bg_asset)
                    if os.path.exists(p_bg):
                        img_b64 = to_b64(p_bg)
            elif role == "style":
                # style role removed
                pass

            if img_b64:
                resolved_images.append({"key": key, "role": role, "img": img_b64})

        # 重新排序：有场景图时强制放第一个槽位，避免尺寸问题
        key_order = [s.get("key") for s in slots if isinstance(s, dict) and s.get("key")]
        key_order = key_order[:max_images]
        ordered_payload: Dict[str, str] = {}
        scene_img = next((r for r in resolved_images if r.get("role") == "scene"), None)
        person_img = next((r for r in resolved_images if r.get("role") == "person"), None)
        used_imgs = set()

        if key_order:
            if scene_img:
                ordered_payload[key_order[0]] = scene_img["img"]
                used_imgs.add(id(scene_img))
            elif person_img:
                ordered_payload[key_order[0]] = person_img["img"]
                used_imgs.add(id(person_img))

        if len(key_order) > 1:
            if person_img and id(person_img) not in used_imgs:
                ordered_payload[key_order[1]] = person_img["img"]
                used_imgs.add(id(person_img))

        # 填充其余未使用的图片
        for r in resolved_images:
            if len(ordered_payload) >= len(key_order):
                break
            if id(r) in used_imgs:
                continue
            next_key = key_order[len(ordered_payload)]
            ordered_payload[next_key] = r["img"]
            used_imgs.add(id(r))

        payload.update(ordered_payload)

        # 处理衣装引用（如果有）
        clothing_asset = None
        if isinstance(prompt_data, dict):
            if not prompt_data.get("_ignore_clothing_asset"):
                clothing_asset = prompt_data.get("_clothing_asset")

        # 构建 Prompt: 严格限制为服装+动作描述，强制纯色背景便于抠图
        # 不使用 ai_persona 和 ai_style，避免画风和背景污染
        
        # ✨ 智能对比色背景选择：根据服装颜色自动选择对比度高的背景
        def get_contrast_background(clothing_desc: str) -> str:
            """
            根据服装描述智能选择对比色背景，确保抠图成功率
            
            策略：
            - 黑色系服装 → 浅灰/白色背景
            - 白色系服装 → 中灰背景 (#808080)
            - 深色系服装 → 浅色背景
            - 浅色系服装 → 深色背景
            - 彩色服装 → 中性灰背景
            """
            clothing_lower = clothing_desc.lower()
            
            # 检测黑色系
            black_keywords = ["black", "黑色", "黑", "暗", "dark"]
            if any(k in clothing_lower for k in black_keywords):
                return "light gray background (#E0E0E0), soft white background"
            
            # 检测白色系
            white_keywords = ["white", "白色", "白", "light", "cream", "米白"]
            if any(k in clothing_lower for k in white_keywords):
                return "medium gray background (#808080)"
            
            # 检测深色系（蓝、紫、棕等）
            dark_keywords = ["navy", "深蓝", "purple", "紫", "brown", "棕", "深"]
            if any(k in clothing_lower for k in dark_keywords):
                return "light background (#D0D0D0)"
            
            # 检测浅色系（粉、黄、浅蓝等）
            light_keywords = ["pink", "粉", "yellow", "黄", "cyan", "浅", "淡"]
            if any(k in clothing_lower for k in light_keywords):
                return "neutral gray background (#A0A0A0)"
            
            # 默认：中性灰背景（适合大多数彩色服装）
            return "neutral gray background (#B0B0B0), simple solid background"
        
        clothing_desc = prompt_data.get('clothing', '')
        auto_background = get_contrast_background(clothing_desc)
        logging.info(f"[Magic Wardrobe] 服装: {clothing_desc} → 智能背景: {auto_background}")
        
        identity_prompt = ""
        if has_identity_ref:
            identity_prompt = (
                "Keep the exact same character identity, face, hair style and hair color "
                "as the reference image."
            )

        # 核心提示词：仅包含角色、服装、动作
        clothing_asset = None
        if isinstance(prompt_data, dict):
            if not prompt_data.get("_ignore_clothing_asset"):
                clothing_asset = prompt_data.get("_clothing_asset")

        if is_image_edit:
            prompt_parts = []
            if identity_prompt:
                prompt_parts.append(identity_prompt)
            # 服装描述
            if clothing_asset:
                prompt_parts.append(f"Outfit style based on image2: {prompt_data.get('clothing')}")
            else:
                prompt_parts.append(f"Outfit: {prompt_data.get('clothing')}")
        else:
            # 普通模型：简化描述
            prompt_parts = [f"1girl, wearing {prompt_data.get('clothing')}"]
            if identity_prompt:
                prompt_parts.append(identity_prompt)

        # 动作描述
        # 稳定性模板（可配置开关）
        if self.config.get("enable_stability_prompt", True):
            stability_prompt = (model_cfg.get("stability_prompt") or "").strip()
            if stability_prompt:
                prompt_parts.append(stability_prompt)

        # 动作描述
        if prompt_data.get('full_body_action'):
            prompt_parts.append(f"{prompt_data.get('full_body_action')}")
        if prompt_data.get('hand_gesture'):
            prompt_parts.append(f"{prompt_data.get('hand_gesture')}")
        if prompt_data.get('pose'):
            prompt_parts.append(f"{prompt_data.get('pose')}")
        if prompt_data.get('expression'):
            prompt_parts.append(f"{prompt_data.get('expression')}")
        if prompt_data.get('camera_angle'):
            prompt_parts.append(f"{prompt_data.get('camera_angle')}")

        # 智能对比色背景（根据服装颜色自动选择），便于后期抠图
        background_hint = f"{auto_background}, plain background, studio lighting"
        
        # 用户自定义追加（如果有）
        layout_prompt_extra = (layout.get("prompt_extra", "") or "").strip()
        prompt_extra = layout_prompt_extra or (self.config.get("prompt_extra", "") or "").strip()
        prompt_tail = f", {prompt_extra}" if prompt_extra else ""
        
        # 最终Prompt：服装+动作+纯色背景
        payload["prompt"] = ", ".join(prompt_parts) + f", {background_hint}{prompt_tail}"

        # 负面提示词：排除低质量和复杂背景，确保纯色背景
        # 使用配置中的 negative_prompt，如果没有则使用默认值
        default_negative = (
            "complex background, detailed background, scenic background, outdoor background, "
            "landscape, scenery, beach, sky, clouds, trees, buildings, furniture, "
            "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, "
            "cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, "
            "username, blurry, torn clothes, ripped clothes, damaged clothes, "
            "different hair color, different hairstyle, different person, different face"
        )
        
        # 优先使用用户配置的 negative_prompt
        layout_negative = (layout.get("negative_prompt", "") or "").strip()
        user_negative = layout_negative or (self.config.get("negative_prompt", "") or "").strip()
        layout_neg_extra = (layout.get("negative_prompt_extra", "") or "").strip()
        negative_extra = layout_neg_extra or (self.config.get("negative_prompt_extra", "") or "").strip()
        stability_negative = ""
        # ????????????
        if self.config.get("enable_stability_prompt", True):
            stability_negative = (model_cfg.get("stability_negative") or "").strip()

        if user_negative:
            payload["negative_prompt"] = user_negative
            if stability_negative:
                payload["negative_prompt"] += f", {stability_negative}"
            if negative_extra:
                payload["negative_prompt"] += f", {negative_extra}"
        else:
            payload["negative_prompt"] = default_negative
            if stability_negative:
                payload["negative_prompt"] += f", {stability_negative}"
            if negative_extra:
                payload["negative_prompt"] += f", {negative_extra}"

        logging.info(f"[Magic Wardrobe] AI Prompt: {payload['prompt']}")
        logging.info(f"[Magic Wardrobe] Negative Prompt: {payload['negative_prompt']}")
        print(f"[Magic Wardrobe DEBUG] AI Prompt: {payload['prompt']}", flush=True)
        print(f"[Magic Wardrobe DEBUG] Negative Prompt: {payload['negative_prompt']}", flush=True)
        self.last_payload_preview = self._make_payload_preview(payload)
        self.last_prompt_preview = {
            "model": model,
            "prompt_parts": prompt_parts,
            "background_hint": background_hint,
            "prompt_extra": prompt_extra,
            "final_prompt": payload.get("prompt", ""),
            "negative": {
                "base": user_negative or default_negative,
                "stability_negative": stability_negative,
                "negative_extra": negative_extra,
                "final_negative": payload.get("negative_prompt", ""),
            },
        }
        self.last_payload_at = time.time()

        # 接口路由选择
        url = self._get_model_endpoint(model_cfg)
        if supports.get("image_size", True):
            # ??? image_size ??????
            output_size = layout.get("output_image_size") or self.config.get("output_image_size", "1024x1024")
            payload["image_size"] = output_size
            payload["batch_size"] = 1

        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=payload, headers=headers, timeout=60) as resp:
                    resp_json = await resp.json()
                    if resp.status == 200:
                        data = resp_json.get("images") or resp_json.get("data")
                        return data[0].get("url") if data else "❌ 无返回数据"
                    else:
                        # 尝试切换 singular URL 重试 (如果是因为路由不对导致的 404/405)
                        if resp.status in [404, 405] and "images" in url:
                             singular_url = url.replace("images", "image")
                             async with session.post(singular_url, json=payload, headers=headers, timeout=60) as resp2:
                                 resp_json = await resp2.json()
                                 if resp2.status == 200:
                                     data = resp_json.get("images") or resp_json.get("data")
                                     return data[0].get("url") if data else "❌ 无返回数据"
                        
                        error_msg = resp_json.get("message", await resp.text())
                        return f"❌ API 错误: {error_msg}"
            except Exception as e:
                return f"❌ 请求失败: {str(e)}"
        return "❌ 未知错误。"

    async def _composite_image(self, char_url: Optional[str], text: str):
        layout = self._load_layout()
        canvas_w, canvas_h = int(layout.get("canvas_width", 1280)), int(layout.get("canvas_height", 720))
        canvas = PILImage.new("RGBA", (canvas_w, canvas_h), (0,0,0,0))

        # 背景处理优先级：
        # 1. AI 生成的背景（如果启用且生成成功）
        # 2. WebUI 配置的背景（background_asset）
        # 3. 随机背景（如果启用）
        # 4. 纯色背景
        
        bg_img = None
        layout_ai_bg = layout.get("enable_ai_background") if isinstance(layout, dict) else None
        ai_background_enabled = layout_ai_bg if layout_ai_bg is not None else self.config.get("enable_ai_background", False)
        
        # 优先级1：AI 生成的背景
        if ai_background_enabled and hasattr(self, 'last_background_url') and self.last_background_url:
            try:
                bg_img = await self._download_image(self.last_background_url)
                if bg_img:
                    bg_img = self._resize_cover(bg_img, canvas_w, canvas_h)
                    logging.info("[Magic Wardrobe] ✓ 使用 AI 生成的背景（优先级最高）")
                else:
                    logging.info("[Magic Wardrobe] AI 背景下载失败，尝试备用方案")
            except Exception as e:
                logging.warning(f"[Magic Wardrobe] AI 背景加载失败: {e}，尝试备用方案")
                bg_img = None

        # 优先级2：WebUI 配置的静态背景
        if bg_img is None:
            bg_asset = layout.get("background_asset")
            
            # 优先级3：随机背景（如果启用）
            if layout.get("random_background", False) or self.config.get("random_background", False):
                bg_dir = os.path.join(self.data_dir, "background")
                if os.path.exists(bg_dir):
                    bgs = [f for f in os.listdir(bg_dir) if f.endswith((".png", ".jpg", ".jpeg", ".webp"))]
                    if bgs:
                        bg_asset = random.choice(bgs)
                        logging.info(f"[Magic Wardrobe] ✓ 随机选择背景: {bg_asset}")

            if bg_asset:
                bg_path = os.path.join(self.data_dir, "background", bg_asset)
                if os.path.exists(bg_path):
                    bg_img = PILImage.open(bg_path).convert("RGBA")
                    logging.info(f"[Magic Wardrobe] ✓ 使用静态背景: {bg_asset}")

                    # 优先使用用户自定义的裁剪区域
                    bg_crop = layout.get("bg_crop")
                    if bg_crop and all(k in bg_crop for k in ["x", "y", "width", "height"]):
                        try:
                            left = max(0, int(bg_crop["x"]))
                            top = max(0, int(bg_crop["y"]))
                            right = min(bg_img.width, left + int(bg_crop["width"]))
                            bottom = min(bg_img.height, top + int(bg_crop["height"]))

                            if right > left and bottom > top:
                                bg_img = bg_img.crop((left, top, right, bottom))
                                bg_img = bg_img.resize((canvas_w, canvas_h), PILImage.Resampling.LANCZOS)
                            else:
                                raise ValueError("Invalid crop size")
                        except Exception as e:
                            logging.warning(f"Custom crop failed, falling back to cover mode: {e}")
                            bg_img = self._resize_cover(bg_img, canvas_w, canvas_h)
                    else:
                        bg_img = self._resize_cover(bg_img, canvas_w, canvas_h)

        # 合成背景到画布
        if bg_img:
            canvas.alpha_composite(bg_img)
        else:
            bg_color = layout.get("background_color", "#2c3e50")
            canvas.paste(bg_color, [0,0,canvas_w,canvas_h])

        def smart_background_removal(img: PILImage.Image) -> PILImage.Image:
            """
            AI智能背景移除：优先使用 rembg (U2-Net) 进行高精度人物分割
            若 rembg 不可用或失败，降级为传统颜色距离算法
            """
            img = img.convert("RGBA")
            
            # 方法1：尝试使用 rembg 进行 AI 抠图（推荐，精度最高）
            use_ai_segmentation = self.config.get("use_ai_segmentation", True)
            
            if use_ai_segmentation and REMBG_AVAILABLE:
                try:
                    logging.info("[Magic Wardrobe] 使用 rembg AI 智能抠图...")
                    # 将 PIL Image 转为 bytes
                    img_bytes = BytesIO()
                    img.save(img_bytes, format='PNG')
                    img_bytes.seek(0)
                    
                    # 使用 rembg 移除背景（U2-Net 模型）
                    output_bytes = rembg_remove(img_bytes.read())
                    
                    # 转回 PIL Image
                    result_img = PILImage.open(BytesIO(output_bytes)).convert("RGBA")
                    
                    logging.info("[Magic Wardrobe] ✓ rembg AI 抠图成功")
                    return result_img
                    
                except Exception as e:
                    logging.warning(f"[Magic Wardrobe] rembg 抠图失败: {e}，降级为传统算法")
            
            # 方法2：传统颜色距离算法（备用方案）
            def sample_border_color(step: int = 10) -> tuple[int, int, int]:
                """采样图片边缘颜色，推断背景色"""
                width, height = img.size
                pixels = img.load()
                samples = []
                # 采样四条边
                for x in range(0, width, step):
                    samples.append(pixels[x, 0][:3])
                    samples.append(pixels[x, height - 1][:3])
                for y in range(0, height, step):
                    samples.append(pixels[0, y][:3])
                    samples.append(pixels[width - 1, y][:3])
                
                if not samples:
                    return (255, 255, 255)
                
                # 计算平均值
                r = sum(p[0] for p in samples) / len(samples)
                g = sum(p[1] for p in samples) / len(samples)
                b = sum(p[2] for p in samples) / len(samples)
                return (int(r), int(g), int(b))
            
            def is_solid_color(color: tuple[int, int, int]) -> bool:
                """判断颜色是否为纯色（RGB三通道接近）"""
                r, g, b = color
                max_diff = max(abs(r-g), abs(g-b), abs(r-b))
                return max_diff < 20
            
            def calculate_border_coverage(target_color: tuple[int, int, int], threshold: int = 30) -> float:
                """
                计算边缘区域中与目标颜色匹配的像素占比
                用于判断该颜色是否真的是背景色（而非衣服颜色）
                """
                width, height = img.size
                pixels = img.load()
                border_pixels = []
                
                # 采样边缘区域（外围10%区域）
                border_thickness = max(int(width * 0.1), int(height * 0.1), 5)
                
                # 上边缘
                for y in range(min(border_thickness, height)):
                    for x in range(width):
                        border_pixels.append(pixels[x, y][:3])
                
                # 下边缘
                for y in range(max(height - border_thickness, 0), height):
                    for x in range(width):
                        border_pixels.append(pixels[x, y][:3])
                
                # 左边缘
                for x in range(min(border_thickness, width)):
                    for y in range(border_thickness, height - border_thickness):
                        border_pixels.append(pixels[x, y][:3])
                
                # 右边缘
                for x in range(max(width - border_thickness, 0), width):
                    for y in range(border_thickness, height - border_thickness):
                        border_pixels.append(pixels[x, y][:3])
                
                if not border_pixels:
                    return 0.0
                
                # 计算匹配像素数
                tr, tg, tb = target_color
                matched = 0
                for r, g, b in border_pixels:
                    dist = abs(r - tr) + abs(g - tg) + abs(b - tb)
                    if dist <= threshold:
                        matched += 1
                
                coverage = matched / len(border_pixels)
                return coverage
            
            def is_background_color(color: tuple[int, int, int]) -> bool:
                """
                智能判断颜色是否为背景色
                
                策略：
                1. 必须是纯色（RGB三通道接近）
                2. 必须在边缘区域大量出现（覆盖率 > 60%）
                3. 特殊处理：绿幕、极黑、极白（明显的背景色）
                """
                r, g, b = color
                
                # 检测绿幕（明显的背景色，直接返回）
                if g > max(r, b) + 30 and g > 100:
                    return True
                
                # 检测极黑（亮度 < 30）
                if r < 30 and g < 30 and b < 30:
                    return True
                
                # 检测极白（亮度 > 220）
                if r > 220 and g > 220 and b > 220:
                    # 但需要验证白色是否真的在边缘大量出现
                    coverage = calculate_border_coverage(color, threshold=20)
                    if coverage > 0.6:  # 边缘覆盖率 > 60%
                        return True
                    else:
                        logging.info(f"[Magic Wardrobe] 检测到白色但覆盖率不足 ({coverage:.1%})，可能是白色衣服，跳过移除")
                        return False
                
                # 检测中性灰/灰色背景（我们生成的智能对比色背景）
                if is_solid_color(color):
                    # 计算亮度
                    brightness = (r + g + b) / 3
                    
                    # 灰色范围：亮度在 50-200 之间
                    if 50 <= brightness <= 200:
                        # 验证灰色是否在边缘大量出现
                        coverage = calculate_border_coverage(color, threshold=30)
                        if coverage > 0.6:  # 边缘覆盖率 > 60%
                            logging.info(f"[Magic Wardrobe] 检测到灰色背景 RGB{color}，边缘覆盖率 {coverage:.1%}")
                            return True
                        else:
                            logging.info(f"[Magic Wardrobe] 检测到灰色但覆盖率不足 ({coverage:.1%})，可能是服装，跳过移除")
                            return False
                
                return False
            
            def remove_by_color_distance(
                target: tuple[int, int, int],
                threshold: int = 50,
                feather: int = 30,
            ) -> PILImage.Image:
                """根据颜色距离移除背景"""
                bg_r, bg_g, bg_b = target
                data = list(img.getdata())
                new_data = []
                removed = 0
                
                for r, g, b, a in data:
                    # 计算曼哈顿距离
                    dist = abs(r - bg_r) + abs(g - bg_g) + abs(b - bg_b)
                    
                    if dist <= threshold:
                        # 完全移除
                        new_data.append((r, g, b, 0))
                        removed += 1
                    elif dist <= threshold + feather:
                        # 羽化边缘
                        alpha = int(a * (dist - threshold) / max(feather, 1))
                        new_data.append((r, g, b, alpha))
                        removed += 1
                    else:
                        # 保留
                        new_data.append((r, g, b, a))
                
                img.putdata(new_data)
                removed_ratio = removed / max(len(data), 1)
                logging.info(
                    f"[Magic Wardrobe] 背景移除完成 - "
                    f"背景色:RGB{target}, 处理像素:{removed}({removed_ratio:.2%})"
                )
                return img
            
            # 采样边缘颜色
            border_color = sample_border_color()
            
            # 判断是否需要移除背景
            if is_background_color(border_color):
                logging.info(f"[Magic Wardrobe] 传统算法检测到背景色: RGB{border_color}，开始移除")
                return remove_by_color_distance(border_color, threshold=50, feather=30)
            else:
                logging.info(f"[Magic Wardrobe] 传统算法未检测到明显背景色: RGB{border_color}，保持原图")
                return img

        def _maybe_cutout(img):
            try:
                if img.mode != "RGBA":
                    img = img.convert("RGBA")
                alpha = img.split()[-1]
                lo, hi = alpha.getextrema()
                if lo == 255 and hi == 255:
                    return smart_background_removal(img)
                return img
            except Exception:
                return img

        char_img = None
        char_asset = layout.get("character_asset")
        if char_asset:
            char_path = os.path.join(self.data_dir, "character", char_asset)
            if os.path.exists(char_path):
                char_img = PILImage.open(char_path).convert("RGBA")
                char_img = _maybe_cutout(char_img)
                logging.info(f"[Magic Wardrobe] 使用角色素材: {char_asset}, 原始尺寸: {char_img.size}")

        if not char_img and char_url:
            logging.info(f"[Magic Wardrobe] Downloading generated image: {char_url[:100]}...")
            try:
                char_img = await self._download_image(char_url)
                if char_img:
                    logging.info(f"[Magic Wardrobe] AI生成图片下载成功, 原始尺寸: {char_img.size}, mode: {char_img.mode}")
                    char_img = _maybe_cutout(char_img)
                    logging.info(f"[Magic Wardrobe] 背景移除后尺寸: {char_img.size}, mode: {char_img.mode}")
                else:
                    logging.error("[Magic Wardrobe] Downloaded image is empty")
            except Exception as e:
                logging.error(f"[Magic Wardrobe] Download generated image failed: {e}", exc_info=True)

        use_char_as_full = bool(layout.get("use_char_as_full", False))
        if char_asset:
            use_char_as_full = False
        if char_url and int(layout.get("character_width", 0) or 0) > 0:
            use_char_as_full = False
        if char_img and not bg_img and not layout.get("background_asset") and not ai_background_enabled and use_char_as_full:
            try:
                full_bg = self._resize_cover(char_img.convert("RGBA"), canvas_w, canvas_h)
                canvas.alpha_composite(full_bg)
                char_img = None
            except Exception as e:
                logging.warning(f"[Magic Wardrobe] Use generated image as background failed: {e}")

        if char_img:
            # 不检测透明边，直接按整张图片高度缩放
            # character_width 现在作为目标高度使用
            char_h_target = int(layout.get("character_width", 600))
            
            # 按整张图片的高度缩放
            ratio = char_h_target / char_img.height
            new_w = int(char_img.width * ratio)
            new_h = int(char_img.height * ratio)
            
            logging.info(f"[Magic Wardrobe] 图片原始尺寸: {char_img.width}x{char_img.height}")
            logging.info(f"[Magic Wardrobe] 按图片高度缩放: {char_img.height}px -> {char_h_target}px, 比例: {ratio:.4f}")
            logging.info(f"[Magic Wardrobe] 缩放后尺寸: {new_w}x{new_h}")
            
            char_img = char_img.resize((new_w, new_h), PILImage.Resampling.LANCZOS)

            char_left = int(layout.get("character_left", 0))
            char_h = new_h
            
            # 始终贴底对齐，如果超出画布，上半部分会被裁掉
            char_bottom = canvas_h - char_h
            if char_h > canvas_h:
                logging.warning(f"[Magic Wardrobe] 角色高度 {char_h}px 超出画布 {canvas_h}px，贴底显示，上半部分将被裁剪")
            else:
                logging.info(f"[Magic Wardrobe] 角色位置: X={char_left}, Y={char_bottom} (贴底)")
            
            char_layer = PILImage.new("RGBA", canvas.size, (0,0,0,0))
            char_layer.paste(char_img, (char_left, char_bottom))
            canvas.alpha_composite(char_layer)

        draw_dialogue = bool(text.strip()) and layout.get("enable_dialogue_box", True)
        if draw_dialogue:
            overlay = PILImage.new("RGBA", canvas.size, (0,0,0,0))
            draw = ImageDraw.Draw(overlay)
            bl, bt, bw, bh = int(layout.get("box_left", 400)), int(layout.get("box_top", 100)), int(layout.get("box_width", 800)), int(layout.get("box_height", 300))
            box_color_str = layout.get("box_color", "#000000b4")
            font_path = self._get_font_path(layout.get("body_font"))
            font = None
        
            logging.info(f"[Magic Wardrobe] 对话框参数 - 位置:({bl},{bt}) 尺寸:({bw}x{bh}) 颜色:{box_color_str}")
            logging.info(f"[Magic Wardrobe] 要绘制的文本: {text}")
        
            def hex_to_rgba(hex_str):
                hex_str = hex_str.lstrip("#")
                if len(hex_str) == 8:
                    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4, 6))
                elif len(hex_str) == 6:
                    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4)) + (255,)
                return (0, 0, 0, 150)

            def apply_opacity(color_rgba, opacity: float):
                r, g, b, a = color_rgba
                alpha = int(a * max(0.0, min(1.0, opacity)))
                return (r, g, b, max(0, min(255, alpha)))

            # 绘制对话框背景
            draw.rounded_rectangle([bl, bt, bl + bw, bt + bh], radius=int(layout.get("radius", 20)), fill=hex_to_rgba(box_color_str))
        
            # 绘制对话框边框图片 (如果存在)
            border_asset = layout.get("border_asset")
            if border_asset:
                border_path = os.path.join(self.data_dir, "border", border_asset)
                if os.path.exists(border_path):
                    border_img = PILImage.open(border_path).convert("RGBA")
                    border_img = border_img.resize((bw, bh), PILImage.Resampling.LANCZOS)
                    overlay.alpha_composite(border_img, (bl, bt))

            # 缁樺埗姓名妗嗭紙濡傛灯寮€鍚級
            show_name_box = bool(layout.get("show_name_box", True))
            name_text = (layout.get("name_text", "") or "").strip()
            if show_name_box and name_text:
                nl = int(layout.get("name_box_left", bl + 20))
                nt = int(layout.get("name_box_top", max(0, bt - 70)))
                nw = int(layout.get("name_box_width", 220))
                nh = int(layout.get("name_box_height", 56))
                name_bg = layout.get("name_box_color", "#d7e8ffcc")
                draw.rounded_rectangle([nl, nt, nl + nw, nt + nh], radius=int(layout.get("name_radius", 16)), fill=hex_to_rgba(name_bg))

                name_font_size = int(layout.get("name_font_size", 26))
                name_text_color = layout.get("name_text_color", "#1e3a8a")
                name_padding = int(layout.get("name_padding", 12))
                try:
                    name_font = ImageFont.truetype(font_path, name_font_size)
                except Exception:
                    name_font = font if font else ImageFont.load_default()
                try:
                    bbox = name_font.getbbox(name_text)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]
                except Exception:
                    text_w = len(name_text) * name_font_size * 0.6
                    text_h = name_font_size
                tx = nl + name_padding
                ty = nt + max(0, int((nh - text_h) / 2))
                draw.text((tx, ty), name_text, font=name_font, fill=name_text_color)
        
            # 绘制文本 (从顶部开始，不自动调整字体大小)
            font_path = self._get_font_path(layout.get("body_font"))
            font_size = int(layout.get("font_size", 30))
            padding = int(layout.get("padding", 20))
            text_color = layout.get("text_color", "#ffffff")
            max_width = bw - padding * 2

            # 加载字体
            try:
                font = ImageFont.truetype(font_path, font_size)
                logging.info(f"[Magic Wardrobe] 成功加载字体: {font_path}, 大小: {font_size}px")
            except Exception as e:
                logging.error(f"[Magic Wardrobe] 字体加载失败: {e}，尝试使用默认字体")
                try:
                    font = ImageFont.load_default()
                    logging.warning("[Magic Wardrobe] 使用默认字体，中文可能显示为方框")
                except:
                    font = None
                    logging.error("[Magic Wardrobe] 所有字体加载失败")

            # 文本换行
            lines = []
            current_line = ""

            for char in text:
                test_line = current_line + char
                try:
                    w = draw.textlength(test_line, font=font) if font else len(test_line) * font_size * 0.6
                except Exception as e:
                    w = len(test_line) * font_size * 0.6

                if w <= max_width:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = char

            if current_line:
                lines.append(current_line)

            logging.info(f"[Magic Wardrobe] 文本分为 {len(lines)} 行")

            # 从顶部开始绘制文本
            line_height = font_size + 8
            y_offset = bt + padding  # 从文本框顶部开始

            # 绘制文本
            for idx, line in enumerate(lines):
                if y_offset + line_height > bt + bh - padding:
                    logging.warning(f"[Magic Wardrobe] 第{idx+1}行超出范围，停止绘制")
                    break
                draw.text((bl + padding, y_offset), line, font=font, fill=text_color)
                logging.info(f"[Magic Wardrobe] 绘制第{idx+1}行于 y={y_offset}: {line[:20]}...")
                y_offset += line_height

            canvas.alpha_composite(overlay)

        overlays = layout.get("text_overlays", [])
        if isinstance(overlays, list) and overlays:
            overlay_canvas = PILImage.new("RGBA", canvas.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay_canvas)
            overlay_items = []
            for item in overlays:
                if isinstance(item, dict):
                    z_index = int(item.get("z_index", 500))
                    overlay_items.append((z_index, item))

            for _, item in sorted(overlay_items, key=lambda x: x[0]):
                overlay_type = item.get("type", "text")
                left = int(item.get("left", 0))
                top = int(item.get("top", 0))
                width = int(item.get("width", 200))
                height = int(item.get("height", 80))
                opacity = float(item.get("opacity", 1.0))

                if overlay_type == "image":
                    image_name = item.get("image", "")
                    if not image_name:
                        continue
                    component_path = os.path.join(self.component_dir, image_name)
                    if not os.path.exists(component_path):
                        continue
                    try:
                        component_img = PILImage.open(component_path).convert("RGBA")
                        if width > 0 and height > 0:
                            component_img = component_img.resize((width, height), PILImage.Resampling.LANCZOS)
                        if opacity < 1.0:
                            alpha = component_img.split()[3]
                            alpha = alpha.point(lambda p: int(p * opacity))
                            component_img.putalpha(alpha)
                        overlay_canvas.paste(component_img, (left, top), component_img)
                    except Exception as e:
                        logging.warning(f"[Magic Wardrobe] Overlay image failed: {e}")
                    continue

                text_content = str(item.get("text", "") or "").strip()
                if not text_content:
                    continue
                font_name = item.get("font")
                font_size = int(item.get("font_size", 24))
                color_str = item.get("color", "#ffffff")
                text_color = apply_opacity(hex_to_rgba(color_str), opacity)
                font_path = self._get_font_path(font_name)
                try:
                    overlay_font = ImageFont.truetype(font_path, font_size)
                except Exception:
                    overlay_font = ImageFont.load_default()

                max_width = max(10, width)
                lines = []
                current_line = ""
                for char in text_content:
                    test_line = current_line + char
                    try:
                        w = overlay_draw.textlength(test_line, font=overlay_font)
                    except Exception:
                        w = len(test_line) * font_size * 0.6
                    if w <= max_width:
                        current_line = test_line
                    else:
                        if current_line:
                            lines.append(current_line)
                        current_line = char
                if current_line:
                    lines.append(current_line)

                line_height = font_size + 6
                y_offset = top
                for line in lines:
                    if y_offset + line_height > top + height:
                        break
                    overlay_draw.text((left, y_offset), line, font=overlay_font, fill=text_color)
                    y_offset += line_height

            canvas.alpha_composite(overlay_canvas)
        
        # 保存为PNG格式以确保支持透明度
        out_file = os.path.join(self.data_dir, f"out_{uuid.uuid4().hex[:8]}.png")
        canvas.save(out_file, format="PNG")
        return out_file

    def _resize_cover(self, img: PILImage.Image, target_w: int, target_h: int) -> PILImage.Image:
        """等比例缩放并居中裁剪以填满目标尺寸"""
        img_ratio = img.width / img.height
        target_ratio = target_w / target_h
        
        if img_ratio > target_ratio:
            new_h = target_h
            new_w = int(new_h * img_ratio)
        else:
            new_w = target_w
            new_h = int(new_w / img_ratio)
            
        img = img.resize((new_w, new_h), PILImage.Resampling.LANCZOS)
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        return img.crop((left, top, left + target_w, top + target_h))

    async def _download_image(self, url: str) -> Optional[PILImage.Image]:
        """从 URL 下载图片并返回 PIL Image 对象"""
        try:
            if not url or not isinstance(url, str):
                logging.error(f"[Magic Wardrobe] 图片下载异常: invalid url={url}")
                return None
            if url.startswith("file://"):
                local_path = url[7:]
                if os.path.exists(local_path):
                    return PILImage.open(local_path).convert("RGBA")

            if os.path.exists(url):
                return PILImage.open(url).convert("RGBA")

            # 处理 base64 格式的图片
            if url.startswith("data:image"):
                base64_data = url.split(",")[1]
                image_data = base64.b64decode(base64_data)
                return PILImage.open(BytesIO(image_data)).convert("RGBA")

            def _normalize_url(raw_url: str) -> str:
                if "%2F" in raw_url or "%2f" in raw_url:
                    try:
                        from urllib.parse import unquote
                        return unquote(raw_url)
                    except Exception:
                        return raw_url
                return raw_url

            # 下载网络图片
            async with aiohttp.ClientSession() as session:
                headers = {"User-Agent": "Mozilla/5.0"}
                for attempt in range(2):
                    fetch_url = url if attempt == 0 else _normalize_url(url)
                    async with session.get(fetch_url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                        if resp.status == 200:
                            image_data = await resp.read()
                            return PILImage.open(BytesIO(image_data)).convert("RGBA")
                        if resp.status == 403 and attempt == 0 and fetch_url != _normalize_url(url):
                            continue
                        logging.error(f"[Magic Wardrobe] 图片下载失败: HTTP {resp.status}")
                        return None
        except Exception as e:
            logging.error(f"[Magic Wardrobe] 图片下载异常: url={url} err={repr(e)}")
            return None

    def _get_font_path(self, font_name):
        if font_name:
            path = os.path.join(self.font_dir, font_name)
            if os.path.exists(path): return path
        font_dir = self.font_dir
        if os.path.exists(font_dir):
            fonts = [f for f in os.listdir(font_dir) if f.endswith((".ttf", ".otf"))]
            if fonts: return os.path.join(font_dir, fonts[0])
        legacy_dir = os.path.join(self.data_dir, "ziti")
        if os.path.exists(legacy_dir):
            fonts = [f for f in os.listdir(legacy_dir) if f.endswith((".ttf", ".otf"))]
            if fonts: return os.path.join(legacy_dir, fonts[0])

        # 尝试使用系统中文字体
        system_fonts = [
            "C:\\Windows\\Fonts\\msyh.ttc",  # 微软雅黑
            "C:\\Windows\\Fonts\\simhei.ttf",  # 黑体
            "C:\\Windows\\Fonts\\simsun.ttc",  # 宋体
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",  # Linux 文泉驿微米黑
            "/System/Library/Fonts/PingFang.ttc",  # macOS 苹方
        ]
        for font_path in system_fonts:
            if os.path.exists(font_path):
                logging.info(f"[Magic Wardrobe] 使用系统字体: {font_path}")
                return font_path

        logging.warning("[Magic Wardrobe] 未找到中文字体，将使用默认字体（可能无法显示中文）")
        return "arial.ttf"

    async def _send_tts_voice(self, event: AstrMessageEvent, text: str):
        """生成并发送 TTS 语音"""
        audio_path = await self._synth_tts_audio(text)
        if not audio_path or not Record:
            return

        try:
            from astrbot.core.message.message_event_result import MessageChain

            voice_chain = MessageChain()
            voice_chain.chain = [Record(file=str(audio_path))]
            await self.context.send_message(event.unified_msg_origin, voice_chain)
            logging.info(f"[Magic Wardrobe] 语音已发送: {audio_path}")
        except Exception as e:
            logging.error(f"[Magic Wardrobe] TTS 发送异常: {e}", exc_info=True)

    async def _synth_tts_audio(self, text: str):
        if not self.tts_client:
            logging.warning("[Magic Wardrobe] TTS 客户端未初始化")
            return None
        if not Record:
            logging.warning("[Magic Wardrobe] Record 组件不可用")
            return None

        voice_id = self.config.get("tts_voice", "")
        if not voice_id:
            logging.info("[Magic Wardrobe] 未配置 TTS 音色 ID，使用默认音色")

        try:
            from pathlib import Path

            temp_dir = Path(os.path.join(self.data_dir, "temp", "tts"))
            temp_dir.mkdir(parents=True, exist_ok=True)

            clean_text = text.replace("【", "").replace("】", "").replace("\n\n", "。").replace("\n", "。")
            logging.info(f"[Magic Wardrobe] 正在生成语音，文本长度: {len(clean_text)}")

            audio_path = await self.tts_client.synth(
                text=clean_text,
                voice=voice_id,
                out_dir=temp_dir,
                speed=self.config.get("tts_speed", 1.0)
            )

            if audio_path:
                return audio_path

            logging.error("[Magic Wardrobe] 语音合成失败")
            return None
        except Exception as e:
            logging.error(f"[Magic Wardrobe] TTS 处理异常: {e}", exc_info=True)
            return None


    def _schedule_tts_fallback(self, event: AstrMessageEvent, delay: float = 0.6):
        umo = event.unified_msg_origin
        existing = self._pending_tts_tasks.get(umo)
        if existing and not existing.done():
            existing.cancel()

        async def _fallback():
            await asyncio.sleep(delay)
            text = self._pending_tts.pop(umo, "")
            if not text:
                return
            logging.info("[Magic Wardrobe] TTS fallback sending")
            try:
                await self._send_tts_voice(event, text)
            except Exception as e:
                logging.error(f"[Magic Wardrobe] TTS fallback failed: {e}", exc_info=True)
            finally:
                event.set_extra("_magic_wardrobe_tts_pending", False)

        self._pending_tts_tasks[umo] = asyncio.create_task(_fallback())

    def _sanitize_render_text(self, text: str) -> str:
        """Best-effort cleanup for JSON-like wrappers from some providers."""
        stripped = text.strip()
        if not stripped:
            return text
        if stripped.startswith("{") and stripped.endswith("}"):
            for candidate in (stripped, stripped.replace("'", "\"")):
                try:
                    data = json.loads(candidate)
                except json.JSONDecodeError:
                    data = None
                if isinstance(data, dict):
                    for key in ("text", "content", "message"):
                        val = data.get(key)
                        if isinstance(val, str) and val.strip():
                            return val.strip()
        match = re.match(r"^\{?\s*text\s*[:=]\s*(.+?)\s*\}?$", stripped, re.DOTALL)
        if match:
            return match.group(1).strip().strip("\"'")
        return text

    async def terminate(self):
        """插件卸载时的清理工作"""
        if hasattr(self, 'webui_task'):
            self.webui_task.cancel()
        if hasattr(self, 'tts_client') and self.tts_client:
            await self.tts_client.close()
            logging.info("[Magic Wardrobe] TTS 客户端已关闭")
