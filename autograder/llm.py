"""LLM client wrapper – vision-capable, OpenAI-compatible via LangChain.

带图片请求的发送方式：
1. build_vision_message(text, image_paths) 构造一条 HumanMessage：
   - content 为 list：先一段 {"type": "text", "text": "..."}，再对每张图
     {"type": "image_url", "image_url": {"url": "data:image/xxx;base64,..."}}
2. 图片本地路径会先读成字节再 base64 编码，以 data URI 形式放入 message。
3. llm.invoke([msg]) 时，LangChain 的 ChatOpenAI 按 OpenAI 多模态 API 格式
   发给 base_url（即 OPENAI_BASE_URL 规范后的地址），兼容所有 OpenAI 格式的
   chat/completions 接口。
4. 自定义 httpx Transport：纠正「JSON 体 + 非 JSON Content-Type」的兼容网关响应，
   避免 openai SDK 将 parse() 结果为 str 导致 LangChain 报 model_dump 错误。
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from openai import DefaultAsyncHttpxClient, DefaultHttpxClient

from autograder.config import LLMConfig, get_config

logger = logging.getLogger(__name__)
_llm_log_lock = threading.Lock()


def _coerce_openai_json_response(response: httpx.Response) -> httpx.Response:
    """部分 OpenAI 兼容网关返回合法 JSON 但 Content-Type 非 application/json。

    此时 openai-python 的 LegacyAPIResponse.parse() 会退回 response.text（str），
    LangChain 再调用 .model_dump() 即触发 AttributeError。将 Content-Type 纠正为
    JSON 后，SDK 会正常反序列化为 ChatCompletion。
    """
    ctype = (response.headers.get("content-type") or "").split(";")[0].strip().lower()
    if ctype.endswith("json"):
        return response
    content = response.content
    if not content or not content.strip():
        return response
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return response
    stripped = text.lstrip()
    if not stripped or stripped[0] not in "{[":
        return response
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return response
    if not isinstance(data, dict):
        return response
    if not any(k in data for k in ("choices", "error")):
        return response
    h = httpx.Headers(response.headers)
    h["content-type"] = "application/json; charset=utf-8"
    return httpx.Response(
        status_code=response.status_code,
        headers=h,
        content=content,
        request=response.request,
        extensions=response.extensions,
    )


class _OpenAICompatHTTPTransport(httpx.HTTPTransport):
    def handle_request(self, request: httpx.Request) -> httpx.Response:
        resp = super().handle_request(request)
        return _coerce_openai_json_response(resp)


class _OpenAICompatAsyncHTTPTransport(httpx.AsyncHTTPTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        resp = await super().handle_async_request(request)
        return _coerce_openai_json_response(resp)


def build_llm(cfg: LLMConfig) -> ChatOpenAI:
    api_base = (cfg.base_url or "").strip() or "https://api.openai.com/v1"
    http_client = DefaultHttpxClient(
        base_url=api_base,
        transport=_OpenAICompatHTTPTransport(),
    )
    http_async_client = DefaultAsyncHttpxClient(
        base_url=api_base,
        transport=_OpenAICompatAsyncHTTPTransport(),
    )
    return ChatOpenAI(
        model=cfg.model,
        base_url=cfg.base_url,
        api_key=cfg.api_key or "EMPTY",
        temperature=0.1,
        max_retries=2,
        http_client=http_client,
        http_async_client=http_async_client,
    )


def encode_image_b64(path: str | Path) -> str:
    data = Path(path).read_bytes()
    return base64.b64encode(data).decode()


def _validate_image_inputs(
    image_paths: list[str | Path],
    *,
    max_images: int | None = None,
    max_image_bytes: int | None = None,
) -> list[Path]:
    """校验多模态图片输入，避免超量/超大请求压垮 API。"""
    normalized: list[Path] = [Path(p) for p in image_paths]
    if max_images is not None and len(normalized) > max_images:
        raise ValueError(f"图片数量超限：{len(normalized)} 张，最大允许 {max_images} 张")
    if max_image_bytes is None:
        return normalized
    oversized: list[tuple[Path, int]] = []
    for p in normalized:
        if not p.exists():
            raise ValueError(f"图片不存在：{p}")
        size = p.stat().st_size
        if size > max_image_bytes:
            oversized.append((p, size))
    if oversized:
        details = ", ".join(f"{p.name}({s}B)" for p, s in oversized[:3])
        extra = "" if len(oversized) <= 3 else f" 等 {len(oversized)} 张"
        raise ValueError(
            f"存在超大图片（单张上限 {max_image_bytes}B）：{details}{extra}"
        )
    return normalized


def _image_block(path: str | Path) -> dict:
    """Single image as a content block (for segmented message)."""
    b64 = encode_image_b64(path)
    suffix = Path(path).suffix.lstrip(".").lower()
    mime = f"image/{suffix}" if suffix != "jpg" else "image/jpeg"
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def build_vision_message(
    text: str,
    image_paths: list[str | Path] | None = None,
    *,
    max_images: int | None = None,
    max_image_bytes: int | None = None,
) -> HumanMessage:
    """Build a HumanMessage with text + optional inline images."""
    validated_paths = _validate_image_inputs(
        image_paths or [],
        max_images=max_images,
        max_image_bytes=max_image_bytes,
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for p in validated_paths:
        content.append(_image_block(p))
    return HumanMessage(content=content)


def build_vision_message_segmented(
    segments: list[tuple[str, list[str | Path]]],
    *,
    max_images: int | None = None,
    max_image_bytes: int | None = None,
) -> HumanMessage:
    """Build a HumanMessage from segments: [(text1, [img, ...]), (text2, []), ...].
    用于切分等场景：每题文字后紧接该题的题目图，最后一段文字后接页面图。
    """
    all_paths: list[str | Path] = []
    for _text, paths in segments:
        all_paths.extend(paths or [])
    _validate_image_inputs(
        all_paths,
        max_images=max_images,
        max_image_bytes=max_image_bytes,
    )
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


def _flush_log_file(f, do_fsync: bool) -> None:
    f.flush()
    if do_fsync:
        try:
            os.fsync(f.fileno())
        except OSError:
            pass


def _write_llm_log(context: dict[str, Any], request_summary: str, response_text: str) -> None:
    """追加写入一条 LLM 对话记录到配置的日志文件；写入后立即 flush，便于实时 tail。"""
    try:
        cfg = get_config()
        llm_log = getattr(cfg, "llm_log", None)
        if not llm_log or not getattr(llm_log, "enabled", False):
            return
        log_path = Path(llm_log.path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        ctx_str = " ".join(f"{k}={v}" for k, v in sorted(context.items()) if v is not None and v != "")
        do_fsync = bool(getattr(llm_log, "fsync", False))
        resp_only = (getattr(llm_log, "response_path", "") or "").strip()
        resp_path = Path(resp_only) if resp_only else None
        if resp_path is not None and resp_path.resolve() == log_path.resolve():
            resp_path = None
        if resp_path is not None:
            resp_path.parent.mkdir(parents=True, exist_ok=True)
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
                _flush_log_file(f, do_fsync)
            if resp_path is not None:
                with open(resp_path, "a", encoding="utf-8") as rf:
                    rf.write("\n")
                    rf.write("---\n")
                    rf.write(f"[{ts}] {ctx_str}\n")
                    rf.write(response_text)
                    if not response_text.endswith("\n"):
                        rf.write("\n")
                    _flush_log_file(rf, do_fsync)
    except Exception as e:
        logger.warning("Failed to write LLM log: %s", e)


def invoke_with_log(llm: ChatOpenAI, messages: list[BaseMessage], context: dict[str, Any] | None = None):
    """调用 LLM；启用 llm_log 时写入完整对话并 flush，可选单独写入 response_path（仅返回）。"""
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
