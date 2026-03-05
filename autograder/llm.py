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
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI

from autograder.config import LLMConfig, get_config

logger = logging.getLogger(__name__)
_llm_log_lock = threading.Lock()


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


def _image_block(path: str | Path) -> dict:
    """Single image as a content block (for segmented message)."""
    b64 = encode_image_b64(path)
    suffix = Path(path).suffix.lstrip(".").lower()
    mime = f"image/{suffix}" if suffix != "jpg" else "image/jpeg"
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def build_vision_message(
    text: str,
    image_paths: list[str | Path] | None = None,
) -> HumanMessage:
    """Build a HumanMessage with text + optional inline images."""
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for p in image_paths or []:
        content.append(_image_block(p))
    return HumanMessage(content=content)


def build_vision_message_segmented(
    segments: list[tuple[str, list[str | Path]]],
) -> HumanMessage:
    """Build a HumanMessage from segments: [(text1, [img, ...]), (text2, []), ...].
    用于切分等场景：每题文字后紧接该题的题目图，最后一段文字后接页面图。
    """
    content: list[dict[str, Any]] = []
    for text, paths in segments:
        if text:
            content.append({"type": "text", "text": text})
        for p in paths or []:
            content.append(_image_block(p))
    return HumanMessage(content=content)


def _message_to_log_string(msg: BaseMessage) -> str:
    """将消息序列化为可写入日志的字符串（图片用 [image] 占位，避免刷屏）。"""
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif block.get("type") == "image_url":
                parts.append("[image]")
        else:
            parts.append(str(block))
    return "\n".join(parts)


def _write_llm_log(context: dict[str, Any], request_summary: str, response_text: str) -> None:
    """追加写入一条 LLM 对话记录到配置的日志文件。"""
    try:
        cfg = get_config()
        if not getattr(cfg, "llm_log", None) or not getattr(cfg.llm_log, "enabled", False):
            return
        log_path = Path(cfg.llm_log.path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        ctx_str = " ".join(f"{k}={v}" for k, v in sorted(context.items()) if v is not None and v != "")
        with _llm_log_lock:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("\n")
                f.write("---\n")
                f.write(f"[{ts}] {ctx_str}\n")
                f.write("### REQUEST\n")
                f.write(request_summary)
                if not request_summary.endswith("\n"):
                    f.write("\n")
                f.write("### RESPONSE\n")
                f.write(response_text)
                if not response_text.endswith("\n"):
                    f.write("\n")
    except Exception as e:
        logger.warning("Failed to write LLM log: %s", e)


def invoke_with_log(llm: ChatOpenAI, messages: list[BaseMessage], context: dict[str, Any] | None = None):
    """调用 LLM 并在启用时记录请求与响应到 llm_log 文件，便于后台调试。"""
    request_summary = "\n".join(_message_to_log_string(m) for m in messages)
    resp = llm.invoke(messages)
    response_text = getattr(resp, "content", str(resp))
    _write_llm_log(context or {}, request_summary, response_text)
    return resp


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
