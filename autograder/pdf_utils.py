"""PDF processing utilities – render pages & crop by bounding box."""

from __future__ import annotations

import re
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

DPI = 200
ZOOM = DPI / 72

# 切分裁剪时在 bbox 四周外扩的像素数，避免裁得太紧漏掉边上的字或半行
CROP_PADDING_PX = 24
# 底部多留一点，减少结尾最后一两行被裁短
CROP_PADDING_BOTTOM_PX = 48


def expand_bbox(
    bbox: list[int],
    image_width: int,
    image_height: int,
    padding: int = CROP_PADDING_PX,
    padding_bottom: int | None = None,
) -> list[int]:
    """将 bbox 四边外扩，并限制在图像范围内。底部可用 padding_bottom 单独加大，避免结尾裁短。"""
    if padding_bottom is None:
        padding_bottom = padding
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image_width, x2 + padding)
    y2 = min(image_height, y2 + padding_bottom)
    return [x1, y1, x2, y2]


def render_pdf_pages(pdf_path: str | Path) -> list[Image.Image]:
    """Render every page of *pdf_path* as a PIL Image."""
    doc = fitz.open(str(pdf_path))
    images: list[Image.Image] = []
    mat = fitz.Matrix(ZOOM, ZOOM)
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    doc.close()
    return images


def crop_region(image: Image.Image, bbox: list[int]) -> Image.Image:
    """Crop *image* to bbox = [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox
    w, h = image.size
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(x1, min(x2, w))
    y2 = max(y1, min(y2, h))
    return image.crop((x1, y1, x2, y2))


def save_answer_images(
    pdf_path: str | Path,
    page_images: list[Image.Image],
    regions: list[dict],
    output_dir: str | Path = "./answers",
) -> list[str]:
    """Crop regions from rendered pages and save as PNGs.

    保存路径符合文档：./answers/{PDF文件名}/{qid}-{page-idx}.png，
    其中 page-idx 为该题下区域的序号（从 0 开始）。

    *regions* – list of {"qid": str, "page": int, "bbox": [x1,y1,x2,y2]}.
    Returns the list of saved file paths.
    """
    pdf_stem = Path(pdf_path).stem
    out_base = Path(output_dir) / pdf_stem
    out_base.mkdir(parents=True, exist_ok=True)

    saved: list[str] = []
    for page_idx, r in enumerate(regions):  # page_idx = 文档中的 page-idx（该题下区域序号）
        page_1based = r["page"]
        img_idx = page_1based - 1  # 1-indexed -> 0-indexed
        if img_idx < 0 or img_idx >= len(page_images):
            continue
        img = page_images[img_idx]
        w, h = img.size
        expanded = expand_bbox(r["bbox"], w, h, padding_bottom=CROP_PADDING_BOTTOM_PX)
        img = crop_region(img, expanded)
        qid = r.get("qid", "unknown")
        fname = f"{qid}-{page_idx}.png"  # 文档：./answers/{PDF文件名}/{qid}-{page-idx}.png
        path = out_base / fname
        img.save(str(path))
        saved.append(str(path))
    return saved


def parse_student_info(filename: str) -> tuple[str, str]:
    """Extract (student_id, student_name) from filename like '2021010238_江晗_8149.pdf'."""
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) >= 2:
        return parts[0], parts[1]
    return stem, ""
