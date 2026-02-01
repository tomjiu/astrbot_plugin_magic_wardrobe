def generate_ai_image(self, user_input: str) -> Optional[Image.Image]:
    """请求硅基流动 API 生成 AI 图片"""
    try:
        url = "https://api.siliconflow.cn/v1/images/generations"
        
        # 从配置中获取 Prompt 和 Key
        base_prompt = self.cfg().get("base_prompt", "A high quality anime-style girl with long pink hair and blue eyes, standing, full body, ")
        style_prompt = self.cfg().get("style_prompt", "highres, soft lighting, masterpiece, 8k, detailed background, galgame CG style")
        api_key = self.cfg().get("silicon_api_key") or SILICON_API_KEY
        
        if api_key == "你的sk-key" or not api_key:
            logger.error("[魔法衣柜] 未配置 SILICON_API_KEY，跳过 AI 生成。")
            return None

        prompt = f"{base_prompt} wearing {user_input}, {style_prompt}"
        logger.info(f"[魔法衣柜] 正在为 {user_input} 生成 AI 图片. Prompt: {prompt}")
        
        payload = {
            "model": "Qwen/Qwen-Image-Edit-2509",
            "prompt": prompt,
            "negative_prompt": "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
            "image_size": "1024x1024",
            "batch_size": 1,
            "num_inference_steps": 25,
            "guidance_scale": 7.5
        }
        
        # 如果配置了人设底图，可以尝试作为编辑输入（需 API 支持，这里暂按标准接口实现）
        # SiliconFlow 的 Qwen-Image-Edit 如果是 Edit 接口，可能需要 multipart/form-data
        # 如果是 generations 接口则只接受 text prompt.
        # 为了兼容性，这里我们先按 generations 接口走，如果用户想用 Edit，通常需要上传 image。
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        response = requests.post(url, json=payload, headers=headers, timeout=60)
        if response.status_code != 200:
            logger.error(f"[魔法衣柜] API 请求失败 ({response.status_code}): {response.text}")
            return None
        
        data = response.json()
        if 'data' not in data or not data['data']:
            logger.error(f"[魔法衣柜] API 返回结果异常: {data}")
            return None
            
        image_url = data['data'][0]['url']
        logger.debug(f"[魔法衣柜] 图片生成成功: {image_url}")
        
        img_response = requests.get(image_url, timeout=30)
        img_response.raise_for_status()
        
        return Image.open(io.BytesIO(img_response.content))
    except Exception as e:
        logger.error(f"[魔法衣柜] AI 图片生成失败: {e}")
        return None
import io
import requests
import base64
import os
from PIL import Image

SILICON_API_KEY = "你的sk-key" # 替换为实际的 API Key
TEMPLATE_PATH = "character_template.png" # 如果有模板图，可以放在这里

def generate_ai_image(user_input: str) -> Image.Image:
    url = "https://api.siliconflow.cn/v1/images/generations"
    base_prompt = "A high quality anime-style girl, standing, full body, "
    style_prompt = "highres, soft lighting, masterpiece, 8k, detailed background"
    
    prompt = f"{base_prompt} wearing {user_input}, {style_prompt}"
    
    payload = {
        "model": "Qwen/Qwen-Image-Edit-2509",
        "prompt": prompt,
        "negative_prompt": "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
        "image_size": "1024x1024",
        "batch_size": 1,
        "num_inference_steps": 25,
        "guidance_scale": 7.5
    }

    if os.path.exists(TEMPLATE_PATH):
        with open(TEMPLATE_PATH, "rb") as f:
            template_data = base64.b64encode(f.read()).decode()
        payload["image"] = f"data:image/png;base64,{template_data}"
        print(f"Using template: {TEMPLATE_PATH}")

    headers = {
        "Authorization": f"Bearer {SILICON_API_KEY}",
        "Content-Type": "application/json"
    }

    print(f"Generating image for: {user_input}")
    response = requests.post(url, json=payload, headers=headers, timeout=60)
    response.raise_for_status()
    
    data = response.json()
    images_list = data.get('data') or data.get('images')
    image_url = images_list[0]['url'] if isinstance(images_list[0], dict) else images_list[0]
    print(f"Image URL: {image_url}")
    
    img_response = requests.get(image_url, timeout=30)
    img_response.raise_for_status()
    
    return Image.open(io.BytesIO(img_response.content))

if __name__ == "__main__":
    if SILICON_API_KEY == "你的sk-key":
        print("请在 test_gen.py 中设置 SILICON_API_KEY")
    else:
        try:
            img = generate_ai_image("旗袍")
            img.show()
            img.save("test_output.png")
            print("Successfully generated and saved test_output.png")
        except Exception as e:
            print(f"Error: {e}")
