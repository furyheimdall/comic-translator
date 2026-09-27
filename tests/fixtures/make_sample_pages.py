"""Generate original synthetic manga-style test pages (no third-party artwork).

Usage: python tests/fixtures/make_sample_pages.py OUTPUT_DIR
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"

PAGES = [
    [
        ("先輩、まだ残ってたんですか？", (820, 120, 1120, 560)),
        ("べ、別にあなたを待ってたわけじゃないんだからね！", (120, 160, 470, 700)),
        ("はいはい、わかってますって。", (780, 980, 1080, 1420)),
        ("明日の試合、絶対に勝とうね。", (140, 1000, 440, 1500)),
    ],
    [
        ("雨が降ってきたな……", (860, 140, 1110, 520)),
        ("傘、一緒に入る？", (160, 180, 420, 520)),
        ("……ありがとう。", (820, 1020, 1060, 1340)),
        ("いつもこうだと助かるんだけど。", (140, 980, 470, 1480)),
    ],
]


def vertical_text(draw: ImageDraw.ImageDraw, text: str, box: tuple[int, int, int, int], font: ImageFont.FreeTypeFont) -> None:
    x0, y0, x1, y1 = box
    size = font.size
    column_height = int((y1 - y0 - 2 * size) // (size * 1.1))
    columns = [text[i : i + column_height] for i in range(0, len(text), column_height)]
    width = len(columns) * int(size * 1.3)
    start_x = (x0 + x1) // 2 + width // 2 - size
    for c, column in enumerate(columns):
        x = start_x - c * int(size * 1.3)
        y = y0 + (y1 - y0 - len(column) * size * 1.1) / 2
        for ch in column:
            if ch in "、。":
                draw.text((x + size * 0.6, y - size * 0.4), ch, font=font, fill="black")
            elif ch in "ー…":
                draw.text((x + size, y), ch, font=font, fill="black", anchor="la", direction=None)
            else:
                draw.text((x, y), ch, font=font, fill="black")
            y += size * 1.1


def figure(draw: ImageDraw.ImageDraw, cx: int, cy: int, scale: float) -> None:
    r = int(90 * scale)
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline="black", width=6, fill=(235, 235, 235))
    draw.arc((cx - r // 2, cy, cx + r // 2, cy + r // 2), 20, 160, fill="black", width=5)
    draw.ellipse((cx - r // 2, cy - r // 3, cx - r // 3, cy - r // 6), fill="black")
    draw.ellipse((cx + r // 3, cy - r // 3, cx + r // 2, cy - r // 6), fill="black")
    draw.line((cx, cy + r, cx, cy + 3 * r), fill="black", width=8)
    for i in range(0, 60, 12):
        draw.line((cx - 3 * r + i * 5, cy + 3 * r + i, cx + 3 * r - i * 5, cy + 3 * r + i), fill=(90, 90, 90), width=3)


def make_page(lines: list[tuple[str, tuple[int, int, int, int]]]) -> Image.Image:
    page = Image.new("RGB", (1240, 1754), "white")
    draw = ImageDraw.Draw(page)
    font = ImageFont.truetype(FONT, 40, index=0)
    for panel in ((40, 40, 1200, 860), (40, 900, 1200, 1714)):
        draw.rectangle(panel, outline="black", width=8)
        # screentone-like background
        for y in range(panel[1] + 20, panel[3] - 20, 18):
            draw.line((panel[0] + 20, y, panel[2] - 20, y), fill=(225, 225, 225), width=2)
    figure(draw, 620, 380, 1.2)
    figure(draw, 620, 1220, 1.0)
    for text, box in lines:
        draw.ellipse(box, fill="white", outline="black", width=5)
        vertical_text(draw, text, (box[0] + 30, box[1] + 30, box[2] - 30, box[3] - 30), font)
    sfx = ImageFont.truetype(FONT, 110, index=0)
    draw.text((520, 760), "ドキッ", font=sfx, fill="black", stroke_width=6, stroke_fill="white")
    return page


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for i, lines in enumerate(PAGES, start=1):
        make_page(lines).save(out / f"page_{i:02d}.png")
    print(f"wrote {len(PAGES)} pages to {out}")


if __name__ == "__main__":
    main()
