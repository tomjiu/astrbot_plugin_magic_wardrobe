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

@register("astrbot_plugin_magic_wardrobe", "AkiKa", "AI 魔法衣橱", "1.2.0")
class MagicWardrobePlugin(Star):
    def __init__(self, context: Context, config: Optional[dict] = None):
        super().__init__(context)
        self.config = config or {}
        # 使用绝对路径，确保图片能被正确加载
        self.data_dir = os.path.abspath(os.path.join("data", "plugins", "astrbot_plugin_magic_wardrobe"))
        self.last_char_url = None # 存储模型生成的最后一张图片
        self._pending_tts: Dict[str, str] = {}
        self._pending_tts_tasks: Dict[str, asyncio.Task] = {}

        # 尝试从持久化文件中加载最后一张图
        self.cache_path = os.path.join(self.data_dir, "last_url.txt")
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r") as f: self.last_char_url = f.read().strip()
            except: pass
        self.background_index_path = os.path.join(self.data_dir, "background_index.json")
        self.background_index = self._load_background_index()
        self.actions_data = self._load_actions_data()
        self._sync_schema_presets()

        # 确保目录存在
        for d in ["background", "character", "clothing", "border", "ziti", "presets"]:
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
                    speed=self.config.get("tts_speed", 1.0)
                )
                logging.info("[Magic Wardrobe] TTS 客户端已初始化")
            else:
                logging.warning("[Magic Wardrobe] TTS 已启用但未配置 API Key")

        # 启动 WebUI 任务
        self.webui_task = asyncio.create_task(self._run_webui())
        logging.info("[Magic Wardrobe] Plugin loaded")

    def _save_last_url(self, url: str):
        self.last_char_url = url
        try:
            with open(self.cache_path, "w") as f: f.write(url)
        except: pass

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

    def _load_actions_data(self) -> Dict[str, Any]:
        paths = [
            os.path.join(self.data_dir, "actions.json"),
            os.path.join(os.path.dirname(__file__), "actions.json"),
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

    def _load_clothing_index(self) -> Dict[str, List[str]]:
        clothing_dir = os.path.join(self.data_dir, "clothing")
        if not os.path.exists(clothing_dir):
            return {}
        index: Dict[str, List[str]] = {}
        for file_name in os.listdir(clothing_dir):
            lower = file_name.lower()
            if not lower.endswith((".png", ".jpg", ".jpeg", ".webp")):
                continue
            base_name = os.path.splitext(file_name)[0]
            match = re.match(r"^\[([^\]]+)\]", base_name)
            tags = []
            if match:
                tags.append(match.group(1))
            tags.append(base_name)
            for tag in tags:
                tag = self._sanitize_tag(tag)
                if not tag:
                    continue
                index.setdefault(tag, []).append(file_name)
        return index

    def _match_clothing_asset(self, tag: str) -> Optional[str]:
        if not tag:
            return None
        index = self._load_clothing_index()
        if tag in index:
            return index[tag][0]
        for key, files in index.items():
            if tag in key:
                return files[0]
        return None

    def _parse_wardrobe_request(self, text: str) -> tuple[Optional[str], str]:
        if not text:
            return None, ""
        cleaned = text.strip()
        match = re.search(r"(衣橱|衣柜|衣櫥|衣櫃)(?:里|中的|内)?(.+)", cleaned)
        if not match:
            return None, cleaned
        tag = match.group(2).strip()
        tag = self._sanitize_tag(tag)
        if not tag:
            return None, cleaned
        asset = self._match_clothing_asset(tag)
        return asset, tag or cleaned

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
        if not self.config.get("enable_ai_background", False):
            return
        if not self.config.get("auto_background", False):
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
        1. 如果未启用会话过滤（use_whitelist=False），则所有会话都允许
        2. 如果启用了会话过滤：
           - 名单为空：允许所有会话（默认全开）
           - 白名单模式（whitelist_mode=True）：仅名单内会话生效
           - 黑名单模式（whitelist_mode=False）：仅名单外会话生效
        """
        if not hasattr(event, "unified_msg_origin"):
            return True

        # 如果未启用会话过滤，默认所有会话都允许
        use_filter = self.config.get("use_whitelist", False)
        if not use_filter:
            return True

        session_list = self.config.get("session_list", [])

        # 如果名单为空，默认允许所有会话（全开）
        if not session_list:
            return True

        sid = event.unified_msg_origin
        is_in_list = any(sid.startswith(item) for item in session_list)

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
                "clothing": get_files("clothing"),
                "borders": get_files("border"),
                "components": get_files("zujian"),
                "fonts": get_files("ziti", (".ttf", ".otf")),
                "presets": [f.replace(".json", "") for f in os.listdir(os.path.join(self.data_dir, "presets")) if f.endswith(".json")]
            })

        @app.route("/api/upload", methods=["POST"])
        async def api_upload():
            files = await request.files
            form = await request.form
            t = form.get("type", "background")
            if t == "component":
                t = "zujian"
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
            if t == "component":
                t = "zujian"
            path = os.path.join(self.data_dir, t, name)
            if os.path.exists(path):
                os.remove(path)
                if t == "background":
                    self._remove_background_entry(name)
                return jsonify({"status": "success"})
            return jsonify({"status": "error", "message": "file not found"})

        @app.route("/api/raw/<type>/<name>")
        async def api_raw(type, name): 
            try:
                from urllib.parse import unquote
                name = unquote(name)
            except Exception:
                pass
            if type == "component":
                type = "zujian"
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
            preset = self.config.get("active_template", "default")
        path = os.path.join(self.data_dir, "presets", f"{preset}.json")
        if not os.path.exists(path): path = os.path.join(self.data_dir, "default_layout.json")
        
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "text_overlays" not in data:
                    data["text_overlays"] = []
                return data
            except: pass
        return {"canvas_width": 1280, "canvas_height": 720, "background_color": "#2c3e50", "box_width": 700, "box_height": 340, "box_left": 520, "box_top": 160, "character_width": 520, "character_left": 0, "font_size": 32, "text_color": "#ffffff", "radius": 20, "padding": 20, "text_overlays": []}


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
            return

        event.set_extra("enable_streaming", False)
        hint = (
            "When the user asks to view or change clothing or outfit, "
            "call the change_outfit tool. Use the user's request as the clothing description. "
            "If the user did not specify actions, pick suitable full_body_action, hand_gesture, "
            "pose, expression, and camera_angle yourself."
        )
        if hint not in request.system_prompt:
            request.system_prompt = (request.system_prompt + "\n" + hint).strip()
        if self.config.get("enable_ai_background", False) and self.config.get("auto_background", False):
            scene_hint = (
                "If the user mentions a scene or location, prefix your reply with a short scene tag in 【】, "
                "for example: 【海边】 or 【教室】. Keep the tag short."
            )
            if scene_hint not in request.system_prompt:
                request.system_prompt = (request.system_prompt + "\n" + scene_hint).strip()
        logging.info("[Magic Wardrobe] Injected tool prompt for outfit request")

    def _should_trigger_outfit(self, text: str) -> bool:
        lowered = text.lower()
        keywords = [
            "看看",
            "来张",
            "来个",
            "换",
            "穿",
            "服",
            "装",
            "衣",
            "泳装",
            "水手服",
            "女仆",
            "礼服",
            "制服",
            "洛丽塔",
            "黑丝",
            "旗袍",
        ]
        english = ["outfit", "dress", "swimsuit", "maid", "uniform"]
        return any(k in text for k in keywords) or any(k in lowered for k in english)


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
        wardrobe_asset, wardrobe_tag = self._parse_wardrobe_request(text)
        if wardrobe_tag:
            clothing = wardrobe_tag

        try:
            event.set_extra("enable_streaming", False)
            action_fields = self._fill_action_defaults("", "", "", "", "")
            prompt_data = {
                "clothing": clothing,
                "full_body_action": action_fields["full_body_action"],
                "hand_gesture": action_fields["hand_gesture"],
                "pose": action_fields["pose"],
                "expression": action_fields["expression"],
                "camera_angle": action_fields["camera_angle"],
            }
            if wardrobe_asset:
                prompt_data["_clothing_asset"] = wardrobe_asset
            logging.info("[Magic Wardrobe] Direct outfit request: %s", clothing)
            image_result = await self._generate_ai_image(prompt_data)
            if not image_result or image_result.startswith("❌"):
                logging.warning("[Magic Wardrobe] Direct outfit failed: %s", image_result)
                return

            self._save_last_url(image_result)
            logging.info(
                "[Magic Wardrobe] Direct outfit updated image: %s",
                image_result[:80],
            )
            event.set_extra("_magic_wardrobe_direct_outfit", True)
        except Exception as e:
            logging.error(f"[Magic Wardrobe] Direct outfit failed: {e}", exc_info=True)

    def _extract_clothing_desc(self, text: str) -> str:
        cleaned = re.sub(r"^(看看|来张|来个|换|穿|给我|要|来套)+", "", text).strip()
        cleaned = cleaned.strip(" \t\r\n,.;:!?，。？！")
        return cleaned or text.strip()

    @filter.on_decorating_result(priority=-1000)
    async def handle_decorating(self, event: AstrMessageEvent, *args, **kwargs):
        """装饰结果处理：将 LLM 回复合成到图片中"""
        if event.get_extra("_magic_wardrobe_streaming_fallback", False):
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

        plain_text = self._sanitize_render_text(result.get_plain_text())
        if not plain_text.strip():
            return
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
            final_image_path = await self._composite_image(char_url, plain_text)

            logging.info(f"[Magic Wardrobe] 图片合成完成: {final_image_path}")

            audio_path = None
            if self.config.get("enable_tts", False) and self.tts_client and Record:
                audio_path = await self._synth_tts_audio(plain_text)

            if audio_path:
                result.chain = [
                    Comp.Image.fromFileSystem(final_image_path),
                    Record(file=str(audio_path)),
                ]
                logging.info("[Magic Wardrobe] 已设置 result.chain 为图片+语音")
                event.set_extra("_magic_wardrobe_record_after", True)
            else:
                result.chain = [Comp.Image.fromFileSystem(final_image_path)]
                logging.info("[Magic Wardrobe] 已设置 result.chain 为图片")

        except Exception as e:
            logging.error(f"[Magic Wardrobe] 渲染失败: {e}", exc_info=True)

    @filter.on_llm_response(priority=-1000)
    async def handle_llm_response(self, event: AstrMessageEvent, response):
        if not self.config.get("enable_render", True):
            return
        if not self._is_allowed(event):
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

        if event.get_extra("_magic_wardrobe_streaming_fallback", False):
            return

        text = ""
        if response and getattr(response, "result_chain", None):
            text = response.result_chain.get_plain_text()
        if not text and response and getattr(response, "completion_text", None):
            text = response.completion_text or ""
        text = self._sanitize_render_text(text)
        if not text.strip():
            return
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
            final_image_path = await self._composite_image(char_url, text)
            from astrbot.core.message.message_event_result import MessageChain

            image_chain = MessageChain()
            audio_path = None
            if self.config.get("enable_tts", False) and self.tts_client and Record:
                audio_path = await self._synth_tts_audio(text)
            if audio_path:
                image_chain.chain = [
                    Comp.Image.fromFileSystem(final_image_path),
                    Record(file=str(audio_path)),
                ]
            else:
                image_chain.chain = [Comp.Image.fromFileSystem(final_image_path)]
            await self.context.send_message(event.unified_msg_origin, image_chain)

            event.set_extra("_magic_wardrobe_streaming_fallback", True)
        except Exception as e:
            logging.error(f"[Magic Wardrobe] Streaming fallback failed: {e}", exc_info=True)

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

            print(f"[Magic Wardrobe DEBUG] 准备调用 _generate_ai_image", flush=True)
            logging.info(f"[Magic Wardrobe] 开始调用 AI 生成图片")
            image_result = await self._generate_ai_image(prompt_data)
            print(f"[Magic Wardrobe DEBUG] AI 生成结果: {image_result[:100] if image_result else 'None'}", flush=True)
            logging.info(f"[Magic Wardrobe] AI 生成结果: {image_result[:100] if image_result else 'None'}")

            if image_result.startswith("❌"):
                return image_result

            self._save_last_url(image_result)

            # 关键修改：返回成功消息给 LLM，让它知道图片已生成
            # LLM 会根据这个结果生成一个中文回复
            # 然后 on_decorating_result 装饰器会拦截这个回复并合成到图片中
            logging.info(f"[Magic Wardrobe] AI图片生成完成: {image_result[:50]}...")
            print(f"[Magic Wardrobe DEBUG] 工具执行完成，返回成功消息", flush=True)

            # 返回一个简洁的成功消息，告诉 LLM 图片已生成
            # LLM 会基于这个结果生成一个自然的中文回复
            action_desc = f"{full_body_action} {hand_gesture} {pose}".strip()
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
            if not self.config.get("enable_ai_background", False):
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

        model = self.config.get("background_model", "black-forest-labs/FLUX.1-schnell")

        # 构建 API 请求
        if api_channel == "siliconflow":
            api_url = "https://api.siliconflow.cn/v1/images/generations"
        else:  # modelscope
            api_url = "https://api.modelscope.cn/v1/images/generations"

        payload = {
            "model": model,
            "prompt": prompt,
            "image_size": "1280x720",
            "num_inference_steps": 20,
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
                    if "images" in result and len(result["images"]) > 0:
                        image_data = result["images"][0]
                        if "url" in image_data:
                            return image_data["url"]
                        elif "b64_json" in image_data:
                            # 处理 base64 返回
                            return f"data:image/png;base64,{image_data['b64_json']}"

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
        clothing_asset = layout.get("clothing_asset")
        if isinstance(prompt_data, dict):
            clothing_asset = prompt_data.get("_clothing_asset") or clothing_asset

        # 使用 character_model 配置（双模型系统）
        model = self.config.get("character_model", "Qwen/Qwen-Image-Edit-2509")
        is_qwen_edit = "Qwen-Image-Edit" in model
        
        payload = {
            "model": model,
            "prompt": "",
            "num_inference_steps": 20,
            "strength": self.config.get("ai_strength", 0.45), # 添加重绘强度
        }

        # 处理图片转 Base64 逻辑
        def to_b64(path):
            with PILImage.open(path).convert("RGB") as img:
                img.thumbnail((1024, 1024))
                buf = BytesIO()
                img.save(buf, format="JPEG")
                return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

        # 1号图：角色身份 (大多数图生图模型需要 image 字段)
        if char_asset:
            p1 = os.path.join(self.data_dir, "character", char_asset)
            if os.path.exists(p1): payload["image"] = to_b64(p1)

        # 2号图：服装参考 (仅 Qwen-Image-Edit 等多图模型支持 image2)
        if clothing_asset and is_qwen_edit:
            p2 = os.path.join(self.data_dir, "clothing", clothing_asset)
            if os.path.exists(p2): payload["image2"] = to_b64(p2)

        # 构建 Prompt: 针对 Qwen 进行特定话术优化
        base_persona = self.config.get('ai_persona', '1girl, solo')
        style = self.config.get('ai_style', 'highres, ultra-detailed, simple background, white background, standalone character, solo')

        # 移除style中的背景相关描述，避免与绿幕冲突
        style_cleaned = re.sub(r'\b(white|black|gray|grey|transparent|simple)\s+background\b', '', style, flags=re.IGNORECASE)
        style_cleaned = re.sub(r'\bbackground\b', '', style_cleaned, flags=re.IGNORECASE)
        style_cleaned = re.sub(r',\s*,', ',', style_cleaned).strip(', ')  # 清理多余逗号

        identity_prompt = ""
        if char_asset:
            identity_prompt = (
                "Keep the exact same character identity, face, and hair color "
                "as the reference image. Do not change hair color or hairstyle."
            )

        if is_qwen_edit:
            prompt_parts = [f"Character based on image: {base_persona}"]
            if identity_prompt:
                prompt_parts.append(identity_prompt)
            if clothing_asset: prompt_parts.append(f"Outfit style based on image2: {prompt_data.get('clothing')}")
            else: prompt_parts.append(f"Outfit: {prompt_data.get('clothing')}")
        else:
            # 普通模型
            prompt_parts = [f"{base_persona}, wearing {prompt_data.get('clothing')}"]
            if identity_prompt:
                prompt_parts.append(identity_prompt)

        # 增强的动作系统参数
        if prompt_data.get('full_body_action'):
            prompt_parts.append(f"Full body action: {prompt_data.get('full_body_action')}")
        if prompt_data.get('hand_gesture'):
            prompt_parts.append(f"Hand gesture: {prompt_data.get('hand_gesture')}")
        if prompt_data.get('pose'):
            prompt_parts.append(f"Pose: {prompt_data.get('pose')}")
        if prompt_data.get('expression'):
            prompt_parts.append(f"Expression: {prompt_data.get('expression')}")
        if prompt_data.get('camera_angle'):
            prompt_parts.append(f"Camera angle: {prompt_data.get('camera_angle')}")

        # 使用绿幕背景，便于后期精确抠图
        # 关键修改：大幅增强绿幕提示词的权重和明确性
        green_screen_prompt = "PURE SOLID BRIGHT GREEN SCREEN BACKGROUND, chroma key green (#00FF00), studio green screen, uniform green backdrop, NO white background, NO transparent background, NO complex background, green screen photography"

        # 构建最终prompt，绿幕提示词重复多次以提高优先级
        payload["prompt"] = f"{green_screen_prompt}, {green_screen_prompt}, " + ", ".join(prompt_parts) + f", {style_cleaned}, green screen background"

        # 负面提示词：明确排除白色和其他背景
        payload["negative_prompt"] = (
            "white background, transparent background, complex background, detailed background, "
            "colorful background, gradient background, textured background, patterned background, "
            "different hair color, different hairstyle, different person"
        )

        logging.info(f"[Magic Wardrobe] AI Prompt: {payload['prompt']}")
        logging.info(f"[Magic Wardrobe] Negative Prompt: {payload['negative_prompt']}")

        # 接口路由选择
        # SiliconFlow 的 Qwen-Image-Edit 一般使用 /images/generations，普通模型有的用 /image/generations
        url = "https://api.siliconflow.cn/v1/images/generations"
        if not is_qwen_edit:
            # 兼容其他模型可能的字段需求
            payload["image_size"] = "1024x1024"
            payload["batch_size"] = 1
            payload["guidance_scale"] = 7.5
            # 注意：某些模型可能还是使用 /image/generations (singular)
            # 但 SiliconFlow 官方文档 Qwen Edit 确实是 plural
            # 为了通用，这里优先使用 plural，如果报错可以尝试 singular

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

        # 背景处理：优先使用 AI 生成的背景
        bg_img = None
        if hasattr(self, 'last_background_url') and self.last_background_url:
            # 使用 AI 生成的背景
            try:
                bg_img = await self._download_image(self.last_background_url)
                if bg_img:
                    bg_img = self._resize_cover(bg_img, canvas_w, canvas_h)
                    logging.info("[Magic Wardrobe] 使用 AI 生成的背景")
            except Exception as e:
                logging.warning(f"[Magic Wardrobe] AI 背景加载失败: {e}")
                bg_img = None

        # 如果没有 AI 背景，使用静态背景库
        if bg_img is None:
            bg_asset = layout.get("background_asset")
            if layout.get("random_background", False) or self.config.get("random_background", False):
                bg_dir = os.path.join(self.data_dir, "background")
                if os.path.exists(bg_dir):
                    bgs = [f for f in os.listdir(bg_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
                    if bgs: bg_asset = random.choice(bgs)

            if bg_asset:
                bg_path = os.path.join(self.data_dir, "background", bg_asset)
                if os.path.exists(bg_path):
                    bg_img = PILImage.open(bg_path).convert("RGBA")

                    # 优先使用用户自定义的裁剪区域
                    bg_crop = layout.get("bg_crop")
                    if bg_crop and all(k in bg_crop for k in ["x", "y", "width", "height"]):
                        try:
                            # 坐标转换：确保是整数且不超限
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
                        # 默认 Cover 模式
                        bg_img = self._resize_cover(bg_img, canvas_w, canvas_h)

        # 合成背景到画布
        if bg_img:
            canvas.alpha_composite(bg_img)
        else:
            bg_color = layout.get("background_color", "#2c3e50")
            canvas.paste(bg_color, [0,0,canvas_w,canvas_h])

        def remove_green_screen(
            img: PILImage.Image,
            tolerance: int = 150,
            min_green_ratio: float = 0.02,
        ) -> PILImage.Image:
            """
            绿幕抠图：将接近绿色(#00FF00)的像素变为透明
            tolerance: 颜色容差值，越大抠除范围越广（默认150，更彻底地抠除绿幕和边缘）
            """
            img = img.convert("RGBA")
            data = list(img.getdata())
            new_data = []

            green_pixel_count = 0
            edge_pixel_count = 0

            for pixel in data:
                r, g, b, a = pixel

                # 检测绿幕颜色：绿色通道明显高于红蓝通道
                # 标准绿幕是 (0, 255, 0)，但AI生成可能有偏差
                # 使用可配置的容差参数
                green_threshold = max(80, 255 - tolerance)  # 降低阈值，更容易识别绿色
                color_diff_threshold = max(30, tolerance // 3)  # 降低差异阈值
                max_other_channel = max(120, 255 - tolerance)  # 降低其他通道最大值

                is_green = (
                    g > green_threshold and  # 绿色通道要足够亮
                    g > r + color_diff_threshold and  # 绿色明显高于红色
                    g > b + color_diff_threshold and  # 绿色明显高于蓝色
                    r < max_other_channel and  # 红色不能太高
                    b < max_other_channel  # 蓝色不能太高
                )

                if is_green:
                    # 绿幕像素变为完全透明
                    new_data.append((r, g, b, 0))
                    green_pixel_count += 1
                else:
                    # 更激进的边缘处理：移除任何带有绿色溢出的像素
                    green_ratio = g / max(r + b + 1, 1)
                    if green_ratio > 1.15 and g > 60:  # 降低阈值，更容易识别边缘
                        # 计算透明度：绿色越多，越透明
                        alpha_factor = max(0, min(1, (green_ratio - 1.15) / 1.0))
                        new_alpha = int(a * (1 - alpha_factor))
                        new_data.append((r, g, b, new_alpha))
                        edge_pixel_count += 1
                    else:
                        new_data.append(pixel)

            total_pixels = len(data)
            green_ratio = green_pixel_count / max(total_pixels, 1)
            if green_ratio >= min_green_ratio:
                img.putdata(new_data)
                logging.info(
                    "[Magic Wardrobe] 绿幕抠图完成 - "
                    f"总像素:{total_pixels}, 绿幕像素:{green_pixel_count}({green_ratio:.2%}), 边缘像素:{edge_pixel_count}"
                )
                return img

            def sample_border_color(step: int = 10) -> tuple[int, int, int]:
                width, height = img.size
                pixels = img.load()
                samples = []
                for x in range(0, width, step):
                    samples.append(pixels[x, 0][:3])
                    samples.append(pixels[x, height - 1][:3])
                for y in range(0, height, step):
                    samples.append(pixels[0, y][:3])
                    samples.append(pixels[width - 1, y][:3])
                if not samples:
                    return (0, 0, 0)
                r = sum(p[0] for p in samples) / len(samples)
                g = sum(p[1] for p in samples) / len(samples)
                b = sum(p[2] for p in samples) / len(samples)
                return (int(r), int(g), int(b))

            def is_greenish(color: tuple[int, int, int]) -> bool:
                r, g, b = color
                return g >= max(r, b) + 8 and g > 50

            def remove_by_color_distance(
                target: tuple[int, int, int],
                threshold: int = 80,
                feather: int = 30,
            ) -> PILImage.Image:
                bg_r, bg_g, bg_b = target
                adjusted = []
                removed = 0
                for r, g, b, a in data:
                    dist = abs(r - bg_r) + abs(g - bg_g) + abs(b - bg_b)
                    if dist <= threshold:
                        adjusted.append((r, g, b, 0))
                        removed += 1
                    elif dist <= threshold + feather:
                        alpha = int(a * (dist - threshold) / max(feather, 1))
                        adjusted.append((r, g, b, alpha))
                        removed += 1
                    else:
                        adjusted.append((r, g, b, a))
                img.putdata(adjusted)
                removed_ratio = removed / max(total_pixels, 1)
                logging.info(
                    "[Magic Wardrobe] 绿幕抠图回退 - "
                    f"背景色:{target}, 处理像素:{removed}({removed_ratio:.2%})"
                )
                return img

            border_color = sample_border_color()
            if is_greenish(border_color):
                return remove_by_color_distance(border_color)

            logging.info(
                "[Magic Wardrobe] 绿幕占比过低，跳过抠图 - "
                f"总像素:{total_pixels}, 绿幕像素:{green_pixel_count}, 占比:{green_ratio:.2%}"
            )
            return img

        char_img = None
        if char_url:
            logging.info(f"[Magic Wardrobe] 开始下载AI生成的角色图片: {char_url[:100]}...")
            try:
                char_img = await self._download_image(char_url)
                if char_img:
                    logging.info(f"[Magic Wardrobe] 图片尺寸: {char_img.size}, 模式: {char_img.mode}")
                    char_img = remove_green_screen(char_img)
                    logging.info(f"[Magic Wardrobe] 绿幕抠图后图片模式: {char_img.mode}")
                else:
                    logging.error("[Magic Wardrobe] 下载AI图片失败，返回空数据")
            except Exception as e:
                logging.error(f"[Magic Wardrobe] 下载AI图片失败: {e}", exc_info=True)
        
        if not char_img:
            char_asset = layout.get("character_asset")
            if char_asset:
                char_path = os.path.join(self.data_dir, "character", char_asset)
                if os.path.exists(char_path):
                    char_img = PILImage.open(char_path).convert("RGBA")
        
        if char_img:
            char_w = int(layout.get("character_width", 600))
            # 比例缩放
            ratio = char_w / char_img.width
            char_h = int(char_img.height * ratio)
            char_img = char_img.resize((char_w, char_h), PILImage.Resampling.LANCZOS)
            
            char_layer = PILImage.new("RGBA", canvas.size, (0,0,0,0))
            # 靠下对齐
            char_layer.paste(char_img, (int(layout.get("character_left", 0)), canvas_h - char_h))
            
            canvas.alpha_composite(char_layer)

        overlay = PILImage.new("RGBA", canvas.size, (0,0,0,0))
        draw = ImageDraw.Draw(overlay)
        bl, bt, bw, bh = int(layout.get("box_left", 400)), int(layout.get("box_top", 100)), int(layout.get("box_width", 800)), int(layout.get("box_height", 300))
        box_color_str = layout.get("box_color", "#000000b4")
        
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
                    component_path = os.path.join(self.data_dir, "zujian", image_name)
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

            # 下载网络图片
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 200:
                        image_data = await resp.read()
                        return PILImage.open(BytesIO(image_data)).convert("RGBA")
                    else:
                        logging.error(f"[Magic Wardrobe] 图片下载失败: HTTP {resp.status}")
                        return None
        except Exception as e:
            logging.error(f"[Magic Wardrobe] 图片下载异常: {e}")
            return None

    def _get_font_path(self, font_name):
        if font_name:
            path = os.path.join(self.data_dir, "ziti", font_name)
            if os.path.exists(path): return path
        ziti_dir = os.path.join(self.data_dir, "ziti")
        if os.path.exists(ziti_dir):
            fonts = [f for f in os.listdir(ziti_dir) if f.endswith((".ttf", ".otf"))]
            if fonts: return os.path.join(ziti_dir, fonts[0])

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
