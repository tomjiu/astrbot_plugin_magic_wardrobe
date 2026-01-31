import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)


class SiliconFlowTTS:
    """SiliconFlow TTS client."""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        model: str = "FunAudioLLM/CosyVoice2-0.5B",
        fmt: str = "wav",
        speed: float = 1.0,
        gain: float = 0,
        sample_rate: Optional[int] = 44100,
    ):
        self.api_url = (api_url or "").rstrip("/")
        self.api_key = api_key or ""
        self.model = model
        self.format = fmt
        self.speed = speed
        self.gain = gain
        self.sample_rate = sample_rate
        self._session: Optional[aiohttp.ClientSession] = None

    async def synth(
        self, text: str, voice: str, out_dir: Path, speed: Optional[float] = None
    ) -> Optional[Path]:
        """Synthesize speech and cache audio by request params."""
        if not self.api_url or not self.api_key:
            logger.error("[Magic Wardrobe TTS] Missing api_url or api_key")
            return None

        if not voice or voice.strip() == "":
            voice = "zh-CN-XiaoxiaoNeural"
            logger.info(f"[Magic Wardrobe TTS] Using default voice: {voice}")

        out_dir.mkdir(parents=True, exist_ok=True)
        eff_speed = float(speed) if speed is not None else float(self.speed)

        key = hashlib.sha256(
            json.dumps(
                {
                    "t": text,
                    "v": voice,
                    "m": self.model,
                    "s": eff_speed,
                    "f": self.format,
                    "g": self.gain,
                    "sr": self.sample_rate,
                },
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()[:16]

        out_path = out_dir / f"{key}.{self.format}"

        if out_path.exists() and out_path.stat().st_size > 0:
            logger.info(f"[Magic Wardrobe TTS] Using cached audio: {out_path}")
            return out_path

        url = f"{self.api_url}/audio/speech"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "voice": voice,
            "input": text,
            "response_format": self.format,
            "speed": eff_speed,
            "gain": self.gain,
        }
        if self.sample_rate:
            payload["sample_rate"] = int(self.sample_rate)

        try:
            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession()

            logger.info(
                "[Magic Wardrobe TTS] Synthesizing speech, text length: %s", len(text)
            )
            async with self._session.post(
                url, headers=headers, json=payload, timeout=60
            ) as response:
                if response.status != 200:
                    err_text = await response.text()
                    logger.error(
                        "[Magic Wardrobe TTS] API error: %s - %s",
                        response.status,
                        err_text,
                    )
                    return None

                content = await response.read()
                if not content:
                    logger.error("[Magic Wardrobe TTS] API returned empty content")
                    return None

                with open(out_path, "wb") as f:
                    f.write(content)

                logger.info("[Magic Wardrobe TTS] Audio synthesized: %s", out_path)
                return out_path
        except Exception as e:
            logger.error("[Magic Wardrobe TTS] Synthesis failed: %s", e)
            return None

    async def close(self):
        """Close HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None
