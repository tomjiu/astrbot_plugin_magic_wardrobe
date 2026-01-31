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

@register("astrbot_plugin_magic_wardrobe", "AkiKa", "AI 魔法衣橱", "1.2.0")
class MagicWardrobePlugin(Star):
    def __init__(self, context: Context, config: Optional[dict] = None):
        super().__init__(context)
        self.config = config or {}
        # 使用绝对路径，确保图片能被正确加载
        self.data_dir = os.path.abspath(os.path.join("data", "plugins", "astrbot_plugin_magic_wardrobe"))
        self.last_char_url = None # 存储模型生成的最后一张图片

        # 尝试从持久化文件中加载最后一张图
        self.cache_path = os.path.join(self.data_dir, "last_url.txt")
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r") as f: self.last_char_url = f.read().strip()
            except: pass
        
        # 确保目录存在
        for d in ["background", "character", "clothing", "border", "ziti", "presets"]:
            os.makedirs(os.path.join(self.data_dir, d), exist_ok=True)
        
        # 启动 WebUI 任务
        self.webui_task = asyncio.create_task(self._run_webui())

    def _save_last_url(self, url: str):
        self.last_char_url = url
        try:
            with open(self.cache_path, "w") as f: f.write(url)
        except: pass

    def _is_allowed(self, event: AstrMessageEvent):
        if not hasattr(event, "unified_msg_origin"):
            return True
        
        use_whitelist = self.config.get("use_whitelist", False)
        session_list = self.config.get("session_list", [])
        sid = event.unified_msg_origin
        
        match = any(sid.startswith(item) for item in session_list)
        if use_whitelist:
            return match
        else:
            return not match

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
                "fonts": get_files("ziti", (".ttf", ".otf")),
                "presets": [f.replace(".json", "") for f in os.listdir(os.path.join(self.data_dir, "presets")) if f.endswith(".json")]
            })

        @app.route("/api/upload", methods=["POST"])
        async def api_upload():
            files = await request.files
            form = await request.form
            t = form.get("type", "background")
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
            path = os.path.join(self.data_dir, t, name)
            if os.path.exists(path):
                os.remove(path)
                return jsonify({"status": "success"})
            return jsonify({"status": "error", "message": "file not found"})

        @app.route("/api/raw/<type>/<name>")
        async def api_raw(type, name): 
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
                with open(path, "r", encoding="utf-8") as f: return json.load(f)
            except: pass
        return {"canvas_width": 1280, "canvas_height": 720, "background_color": "#2c3e50", "box_width": 700, "box_height": 340, "box_left": 520, "box_top": 160, "character_width": 520, "character_left": 0, "font_size": 32, "text_color": "#ffffff", "radius": 20, "padding": 20}

    @filter.on_decorating_result(priority=-1000)
    async def handle_decorating(self, event: AstrMessageEvent, *args, **kwargs):
        """装饰结果处理：将 LLM 回复合成到图片中"""

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

        plain_text = result.get_plain_text()
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

            # 参考 TTS 插件：直接修改 result.chain 为图片
            result.chain = [Comp.Image.fromFileSystem(final_image_path)]
            logging.info("[Magic Wardrobe] 已设置 result.chain 为图片")

        except Exception as e:
            logging.error(f"[Magic Wardrobe] 渲染失败: {e}", exc_info=True)

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
    async def change_outfit(self, event: AstrMessageEvent, clothing: str, upper_action: str = "", lower_action: str = "", expression: str = "") -> str:
        """
        核心绘图/换装工具。当用户要求你【生成图片】、【画图】、【自拍】、【换装】、【变身】或【展示动作表情】时必须调用。

        Args:
            clothing(string): 必须提供。角色的着装描述。如果不需换装，请根据当前人设描述默认服装，或根据场景自由发挥(如: "Casual outfit", "Swimsuit")。
            upper_action(string): 角色上半身的动作或姿势描述（如：双手托腮、挥手、拿着书）。
            lower_action(string): 角色下半身的动作或姿势描述（如：坐着、奔跑、站立）。
            expression(string): 角色的表情描述（如：害羞、微笑、生气）。
        """
        try:
            print(f"[Magic Wardrobe DEBUG] change_outfit 工具被调用", flush=True)
            print(f"[Magic Wardrobe DEBUG] 参数 - clothing:{clothing}, expression:{expression}", flush=True)
            logging.info(f"[Magic Wardrobe] change_outfit 工具被调用")
            logging.info(f"[Magic Wardrobe] 参数 - clothing:{clothing}, expression:{expression}")

            enable_tool_value = self.config.get("enable_tool", True)
            print(f"[Magic Wardrobe DEBUG] enable_tool = {enable_tool_value}", flush=True)

            if not enable_tool_value:
                print(f"[Magic Wardrobe DEBUG] 工具已被禁用，返回", flush=True)
                logging.warning("[Magic Wardrobe] 工具已被禁用")
                return "工具已被禁用"

            print(f"[Magic Wardrobe DEBUG] 开始封装提示词数据", flush=True)
            # 封装提示词数据
            prompt_data = {
                "clothing": clothing,
                "upper_action": upper_action,
                "lower_action": lower_action,
                "expression": expression
            }

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
            return f"图片已成功生成。服装：{clothing}，表情：{expression}，动作：{upper_action} {lower_action}。请用中文自然地回复用户，描述你换上新装扮后的感受。"

        except Exception as e:
            logging.error(f"[Magic Wardrobe] change_outfit 发生异常: {e}", exc_info=True)
            print(f"[Magic Wardrobe DEBUG] 发生异常: {e}", flush=True)
            return f"❌ 换装失败: {str(e)}"

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
        
        # 针对不同模型动态构建 Payload
        model = self.config.get("ai_model", "Qwen/Qwen-Image-Edit-2509")
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

        if is_qwen_edit:
            prompt_parts = [f"Character based on image: {base_persona}"]
            if clothing_asset: prompt_parts.append(f"Outfit style based on image2: {prompt_data.get('clothing')}")
            else: prompt_parts.append(f"Outfit: {prompt_data.get('clothing')}")
        else:
            # 普通模型
            prompt_parts = [f"{base_persona}, wearing {prompt_data.get('clothing')}"]

        if prompt_data.get('upper_action'): prompt_parts.append(f"Action: {prompt_data.get('upper_action')}")
        if prompt_data.get('expression'): prompt_parts.append(f"Expression: {prompt_data.get('expression')}")

        # 使用绿幕背景，便于后期精确抠图
        # 关键修改：大幅增强绿幕提示词的权重和明确性
        green_screen_prompt = "PURE SOLID BRIGHT GREEN SCREEN BACKGROUND, chroma key green (#00FF00), studio green screen, uniform green backdrop, NO white background, NO transparent background, NO complex background, green screen photography"

        # 构建最终prompt，绿幕提示词重复多次以提高优先级
        payload["prompt"] = f"{green_screen_prompt}, {green_screen_prompt}, " + ", ".join(prompt_parts) + f", {style_cleaned}, green screen background"

        # 负面提示词：明确排除白色和其他背景
        payload["negative_prompt"] = "white background, transparent background, complex background, detailed background, colorful background, gradient background, textured background, patterned background"

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
        
        # 处理背景随机化 (优先读取布局配置中的开关)
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
                
                canvas.alpha_composite(bg_img)
        else:
            bg_color = layout.get("background_color", "#2c3e50")
            canvas.paste(bg_color, [0,0,canvas_w,canvas_h])

        def remove_green_screen(img: PILImage.Image, tolerance: int = 150) -> PILImage.Image:
            """
            绿幕抠图：将接近绿色(#00FF00)的像素变为透明
            tolerance: 颜色容差值，越大抠除范围越广（默认150，更彻底地抠除绿幕和边缘）
            """
            img = img.convert("RGBA")
            data = img.getdata()
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

            img.putdata(new_data)
            total_pixels = len(data)
            logging.info(f"[Magic Wardrobe] 绿幕抠图完成 - 总像素:{total_pixels}, 绿幕像素:{green_pixel_count}({green_pixel_count*100//total_pixels}%), 边缘像素:{edge_pixel_count}")

            return img

        char_img = None
        if char_url:
            logging.info(f"[Magic Wardrobe] 开始下载AI生成的角色图片: {char_url[:100]}...")
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(char_url, timeout=30) as resp:
                        if resp.status != 200:
                            logging.error(f"[Magic Wardrobe] 下载失败，HTTP状态码: {resp.status}")
                        else:
                            img_bytes = await resp.read()
                            logging.info(f"[Magic Wardrobe] 下载完成，图片大小: {len(img_bytes)} bytes")
                            char_img = PILImage.open(BytesIO(img_bytes)).convert("RGBA")
                            logging.info(f"[Magic Wardrobe] 图片尺寸: {char_img.size}, 模式: {char_img.mode}")
                            # 应用绿幕抠图
                            char_img = remove_green_screen(char_img)
                            logging.info(f"[Magic Wardrobe] 绿幕抠图后图片模式: {char_img.mode}")
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