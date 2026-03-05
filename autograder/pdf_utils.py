"""PDF processing utilities – render pages & crop by bounding box."""

from __future__ import annotations

import re
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

DPI = 200
ZOOM = DPI / 72


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

    *regions* – list of {"page": int, "bbox": [x1,y1,x2,y2]}.
    Returns the list of saved file paths.
    """
    pdf_stem = Path(pdf_path).stem
    out_base = Path(output_dir) / pdf_stem
    out_base.mkdir(parents=True, exist_ok=True)

    saved: list[str] = []
    for r in regions:
        page_idx = r["page"] - 1  # 1-indexed -> 0-indexed
        if page_idx < 0 or page_idx >= len(page_images):
            continue
        img = crop_region(page_images[page_idx], r["bbox"])
        qid = r.get("qid", "unknown")
        fname = f"{qid}-{r['page']}.png"
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
