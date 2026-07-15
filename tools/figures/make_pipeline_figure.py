#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision.datasets import CIFAR10

WHITE = (255, 255, 255)
BLACK = (20, 20, 20)
GRAY = (90, 90, 90)
LIGHT = (247, 247, 247)

RED = (220, 45, 45)
BLUE = (45, 95, 210)
GREEN = (70, 160, 80)
ORANGE = (210, 145, 45)

BG_BLUE = (232, 241, 255)
BG_GREEN = (232, 246, 232)
BG_RED = (255, 238, 234)
BG_YELLOW = (255, 249, 225)
BG_GRAY = (242, 242, 242)

DASH_BLUE = (45, 90, 180)
DASH_GRAY = (80, 80, 80)


def font(size, bold=False):
    names = [
        "DejaVuSerif-Bold.ttf" if bold else "DejaVuSerif.ttf",
        "LiberationSerif-Bold.ttf" if bold else "LiberationSerif-Regular.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    roots = [
        "/usr/share/fonts/truetype/dejavu",
        "/usr/share/fonts/truetype/liberation2",
        "/usr/share/fonts/truetype/liberation",
        "/usr/share/fonts/dejavu",
    ]
    for root in roots:
        for name in names:
            p = Path(root) / name
            if p.exists():
                return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


F_TITLE = font(34, True)
F_PANEL = font(30, True)
F_HEAD = font(22, True)
F_TEXT = font(18, False)
F_TEXT_B = font(18, True)
F_SMALL = font(15, False)
F_SMALL_B = font(15, True)
F_TINY = font(12, False)
F_TINY_B = font(12, True)


def tsize(draw, s, f):
    b = draw.textbbox((0, 0), s, font=f)
    return b[2] - b[0], b[3] - b[1]


def center(draw, box, s, f, fill=BLACK):
    x1, y1, x2, y2 = box
    w, h = tsize(draw, s, f)
    draw.text((x1 + (x2 - x1 - w) / 2, y1 + (y2 - y1 - h) / 2), s, font=f, fill=fill)


def rounded(draw, box, fill=WHITE, outline=GRAY, width=2, r=18):
    draw.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)


def dashed_rect(draw, box, color=DASH_BLUE, width=3, dash=18, gap=10):
    x1, y1, x2, y2 = box
    x = x1
    while x < x2:
        draw.line((x, y1, min(x + dash, x2), y1), fill=color, width=width)
        draw.line((x, y2, min(x + dash, x2), y2), fill=color, width=width)
        x += dash + gap

    y = y1
    while y < y2:
        draw.line((x1, y, x1, min(y + dash, y2)), fill=color, width=width)
        draw.line((x2, y, x2, min(y + dash, y2)), fill=color, width=width)
        y += dash + gap


def arrow(draw, p1, p2, color=(45, 45, 45), width=4):
    draw.line([p1, p2], fill=color, width=width)
    sx, sy = p1
    ex, ey = p2
    ang = np.arctan2(ey - sy, ex - sx)
    L = 19
    pA = (ex + L * np.cos(ang + 2.55), ey + L * np.sin(ang + 2.55))
    pB = (ex + L * np.cos(ang - 2.55), ey + L * np.sin(ang - 2.55))
    draw.polygon([p2, pA, pB], fill=color)


def add_trigger(img, pos, color):
    im = img.convert("RGB").copy()
    arr = np.array(im)
    h, w = arr.shape[:2]
    s = max(5, int(min(h, w) * 0.20))
    pad = max(2, int(min(h, w) * 0.08))

    if pos == "tl":
        y, x = pad, pad
    elif pos == "tr":
        y, x = pad, w - pad - s
    elif pos == "bl":
        y, x = h - pad - s, pad
    elif pos == "br":
        y, x = h - pad - s, w - pad - s
    else:
        raise ValueError(pos)

    arr[y:y+s, x:x+s] = color
    return Image.fromarray(arr)


def add_dual(img, p1, p2):
    return add_trigger(add_trigger(img, p1, RED), p2, BLUE)


def sample_img(img, size):
    return img.convert("RGB").resize((size, size), Image.Resampling.NEAREST)


def sample_box(canvas, draw, box, img, top_label, bottom_label, border, bg=WHITE):
    x1, y1, x2, y2 = box
    rounded(draw, box, fill=bg, outline=border, width=3, r=12)

    center(draw, (x1, y1 + 5, x2, y1 + 28), top_label, F_TINY_B)

    img_size = min(x2 - x1 - 20, y2 - y1 - 58)
    im = sample_img(img, img_size)
    canvas.paste(im, (x1 + (x2 - x1 - img_size) // 2, y1 + 30))

    center(draw, (x1, y2 - 25, x2, y2 - 4), bottom_label, F_TINY_B)


def mini_title_box(draw, box, title, fill, outline):
    x1, y1, x2, y2 = box
    rounded(draw, box, fill=fill, outline=outline, width=3, r=18)
    draw.rounded_rectangle((x1, y1, x2, y1 + 38), radius=18, fill=WHITE)
    draw.rectangle((x1, y1 + 20, x2, y1 + 38), fill=WHITE)
    center(draw, (x1, y1 + 2, x2, y1 + 38), title, F_SMALL_B)


def draw_network(draw, box):
    x1, y1, x2, y2 = box
    layers = [3, 5, 5, 3]
    xs = np.linspace(x1 + 20, x2 - 20, len(layers))
    all_nodes = []

    for li, n in enumerate(layers):
        ys = np.linspace(y1 + 25, y2 - 25, n)
        nodes = [(xs[li], yy) for yy in ys]
        all_nodes.append(nodes)

    for left, right in zip(all_nodes[:-1], all_nodes[1:]):
        for p in left:
            for q in right:
                draw.line((p, q), fill=(130, 130, 130), width=1)

    for nodes in all_nodes:
        for x, y in nodes:
            draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=(65, 160, 210), outline=(65, 160, 210))


def load_images(root):
    ds = CIFAR10(root=str(root), train=True, download=False)
    idx = [11, 22, 35, 47, 58, 64, 77, 89, 93, 111, 125, 139]
    return [ds[i][0].convert("RGB") for i in idx]


def make_pipeline(out_dir, imgs):
    W, H = 2400, 1080
    canvas = Image.new("RGB", (W, H), WHITE)
    draw = ImageDraw.Draw(canvas)

    draw.text((55, 34), "CCDT Pipeline", font=F_TITLE, fill=BLACK)

    # Three main stages
    poison_box = (35, 100, 1285, 965)
    train_box = (1360, 100, 1715, 965)
    result_box = (1790, 100, 2365, 965)

    dashed_rect(draw, poison_box, DASH_BLUE)
    dashed_rect(draw, train_box, DASH_GRAY)
    dashed_rect(draw, result_box, DASH_BLUE)

    center(draw, (35, 112, 1285, 165), "1. Poisoning: build configuration-aware samples", F_PANEL)
    center(draw, (1360, 112, 1715, 165), "2. Training", F_PANEL)
    center(draw, (1790, 112, 2365, 165), "3. Result", F_PANEL)

    # ---------------------------------------------------------
    # Stage 1: Poisoning
    # ---------------------------------------------------------
    mini_title_box(draw, (80, 190, 470, 410), "Trigger set", BG_GRAY, GRAY)

    draw.rectangle((130, 260, 180, 310), fill=RED)
    center(draw, (200, 250, 280, 320), "T1", F_HEAD)
    draw.rectangle((130, 335, 180, 385), fill=BLUE)
    center(draw, (200, 325, 280, 395), "T2", F_HEAD)

    center(draw, (300, 252, 445, 315), "red patch", F_SMALL_B)
    center(draw, (300, 327, 445, 390), "blue patch", F_SMALL_B)

    mini_title_box(draw, (520, 190, 1235, 410), "Valid configurations", BG_BLUE, BLUE)

    sample_box(
        canvas, draw,
        (575, 245, 725, 380),
        add_dual(imgs[0], "tl", "br"),
        "config c1", "target t1",
        GREEN, WHITE
    )
    sample_box(
        canvas, draw,
        (805, 245, 955, 380),
        add_dual(imgs[1], "tr", "bl"),
        "config c2", "target t2",
        GREEN, WHITE
    )

    center(draw, (1015, 260, 1195, 315), "same triggers", F_SMALL_B)
    center(draw, (1015, 325, 1195, 380), "different positions", F_SMALL_B)

    # Sample construction block
    mini_title_box(draw, (80, 470, 1235, 890), "Construct poisoned training set", (250, 250, 250), GRAY)

    # clean sample row
    sample_box(canvas, draw, (135, 545, 285, 690), imgs[2], "clean", "original", GRAY, WHITE)
    arrow(draw, (300, 618), (355, 618), width=3)

    # four training types
    cells = [
        ("clean", imgs[2], "original", GRAY, BG_GRAY),
        ("T1 only", add_trigger(imgs[3], "tl", RED), "original", ORANGE, BG_YELLOW),
        ("T2 only", add_trigger(imgs[4], "br", BLUE), "original", ORANGE, BG_YELLOW),
        ("valid c1", add_dual(imgs[5], "tl", "br"), "target t1", GREEN, BG_GREEN),
        ("valid c2", add_dual(imgs[6], "tr", "bl"), "target t2", GREEN, BG_GREEN),
        ("invalid", add_dual(imgs[7], "tl", "tr"), "original", RED, BG_RED),
    ]

    start_x = 375
    start_y = 520
    w = 140
    h = 138
    gap_x = 42
    gap_y = 60

    for i, (top, im, bottom, border, bg) in enumerate(cells):
        r = i // 3
        c = i % 3
        x = start_x + c * (w + gap_x)
        y = start_y + r * (h + gap_y)
        sample_box(canvas, draw, (x, y, x + w, y + h), im, top, bottom, border, bg)

    # Mixed set icon
    rounded(draw, (1000, 575, 1195, 770), fill=WHITE, outline=BLUE, width=3, r=18)
    center(draw, (1000, 585, 1195, 620), "mixed set", F_SMALL_B)
    for i, color in enumerate([GRAY, ORANGE, ORANGE, GREEN, GREEN, RED]):
        xx = 1030 + (i % 3) * 45
        yy = 640 + (i // 3) * 52
        draw.rectangle((xx, yy, xx + 35, yy + 35), outline=color, width=4, fill=(250, 250, 250))
    arrow(draw, (920, 665), (1000, 665), width=3)

    # ---------------------------------------------------------
    # Stage 2: Training
    # ---------------------------------------------------------
    mini_title_box(draw, (1405, 235, 1670, 790), "Train CCDT model", WHITE, GRAY)

    rounded(draw, (1445, 315, 1630, 545), fill=WHITE, outline=GRAY, width=3, r=16)
    draw_network(draw, (1470, 350, 1605, 500))

    center(draw, (1445, 580, 1630, 625), "CCDT model", F_HEAD)

    # Small visual objective, no prose sentence
    rounded(draw, (1435, 675, 1640, 735), fill=BG_BLUE, outline=BLUE, width=3, r=15)
    center(draw, (1435, 675, 1640, 735), "clean + selective", F_SMALL_B)

    arrow(draw, (1285, 545), (1360, 545), width=5)

    # ---------------------------------------------------------
    # Stage 3: Results
    # ---------------------------------------------------------
    mini_title_box(draw, (1825, 185, 2325, 880), "Selective activation", WHITE, BLUE)

    cases = [
        ("T1 only", add_trigger(imgs[8], "tl", RED), "original", ORANGE, BG_YELLOW),
        ("T2 only", add_trigger(imgs[9], "br", BLUE), "original", ORANGE, BG_YELLOW),
        ("invalid", add_dual(imgs[10], "tl", "tr"), "original", RED, BG_RED),
        ("valid c1", add_dual(imgs[0], "tl", "br"), "target t1", GREEN, BG_GREEN),
        ("valid c2", add_dual(imgs[1], "tr", "bl"), "target t2", GREEN, BG_GREEN),
    ]

    x_img = 1865
    x_out = 2145
    y0 = 250

    for i, (case, im, out, border, bg) in enumerate(cases):
        yy = y0 + i * 120
        sample_box(canvas, draw, (x_img, yy, x_img + 125, yy + 105), im, case, "", border, bg)
        arrow(draw, (2005, yy + 52), (2130, yy + 52), width=3)
        rounded(draw, (x_out, yy + 22, x_out + 145, yy + 82), fill=WHITE, outline=border, width=3, r=14)
        center(draw, (x_out, yy + 22, x_out + 145, yy + 82), out, F_SMALL_B)

    arrow(draw, (1715, 545), (1790, 545), width=5)

    # Compact legend
    draw.rectangle((65, 1000, 95, 1030), fill=RED)
    draw.text((108, 1001), "T1", font=F_SMALL, fill=BLACK)
    draw.rectangle((170, 1000, 200, 1030), fill=BLUE)
    draw.text((213, 1001), "T2", font=F_SMALL, fill=BLACK)
    draw.text((300, 1001), "original: benign label    target: attack label", font=F_SMALL, fill=BLACK)

    out_png = out_dir / "fig4_ccdt_three_stage_pipeline_STYLE_V7.png"
    out_pdf = out_dir / "fig4_ccdt_three_stage_pipeline_STYLE_V7.pdf"
    canvas.save(out_png)
    canvas.save(out_pdf, "PDF", resolution=300)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = load_images(Path(args.data_root))
    make_pipeline(out_dir, imgs)

    print("[OK] generated:")
    for p in sorted(out_dir.glob("*STYLE_V7*")):
        print(" -", p)


if __name__ == "__main__":
    main()
