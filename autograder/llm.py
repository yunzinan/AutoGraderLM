"""LLM client wrapper – vision-capable, OpenAI-compatible via LangChain.

带图片请求的发送方式：
1. build_vision_message(text, image_paths) 构造一条 HumanMessage：
   - content 为 list：先一段 {"type": "text", "text": "..."}，再对每张图
     {"type": "image_url", "image_url": {"url": "data:image/xxx;base64,..."}}
2. 图片本地路径会先读成字节再 base64 编码，以 data URI 形式放入 message。
3. llm.invoke([msg]) 时，LangChain 的 ChatOpenAI 按 OpenAI 多模态 API 格式
   发给 base_url（即 OPENAI_BASE_URL 规范后的地址），兼容所有 OpenAI 格式的
   chat/completions 接口。
"""

from __future__ import annotations

import base64
import json
import logging
import re
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from autograder.config import LLMConfig

logger = logging.getLogger(__name__)


def build_llm(cfg: LLMConfig) -> ChatOpenAI:
    return ChatOpenAI(
        model=cfg.model,
        base_url=cfg.base_url,
        api_key=cfg.api_key or "EMPTY",
        temperature=0.1,
        max_retries=2,
    )


def encode_image_b64(path: str | Path) -> str:
    data = Path(path).read_bytes()
    return base64.b64encode(data).decode()


def build_vision_message(
    text: str,
    image_paths: list[str | Path] | None = None,
) -> HumanMessage:
    """Build a HumanMessage with text + optional inline images."""
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for p in image_paths or []:
        b64 = encode_image_b64(p)
        suffix = Path(p).suffix.lstrip(".").lower()
        mime = f"image/{suffix}" if suffix != "jpg" else "image/jpeg"
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"},
        })
    return HumanMessage(content=content)


def extract_json(text: str) -> dict | list | None:
    """Best-effort JSON extraction from LLM output."""
    text = text.strip()
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting from code fence
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    # Try finding first { ... } or [ ... ]
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    logger.warning("Failed to extract JSON from LLM output: %s", text[:200])
    return None
