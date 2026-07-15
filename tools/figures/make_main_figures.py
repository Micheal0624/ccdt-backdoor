#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision.datasets import CIFAR10, CIFAR100

WHITE = (255, 255, 255)
BLACK = (20, 20, 20)
GRAY = (85, 85, 85)
LIGHT = (245, 245, 245)

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
F_PANEL = font(29, True)
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


def cross(draw, box, color=RED, width=6):
    x1, y1, x2, y2 = box
    draw.line((x1, y1, x2, y2), fill=color, width=width)
    draw.line((x1, y2, x2, y1), fill=color, width=width)


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


def resize_sample(img, size):
    return img.convert("RGB").resize((size, size), Image.Resampling.NEAREST)


def sample_box(canvas, draw, box, img, top_label="", bottom_label="", border=BLUE, bg=WHITE):
    x1, y1, x2, y2 = box
    rounded(draw, box, fill=bg, outline=border, width=3, r=12)

    if top_label:
        center(draw, (x1, y1 + 4, x2, y1 + 28), top_label, F_TINY_B)

    top_h = 28 if top_label else 8
    bottom_h = 25 if bottom_label else 8
    img_size = min(x2 - x1 - 20, y2 - y1 - top_h - bottom_h - 8)
    im = resize_sample(img, img_size)
    canvas.paste(im, (x1 + (x2 - x1 - img_size) // 2, y1 + top_h))

    if bottom_label:
        center(draw, (x1, y2 - 25, x2, y2 - 4), bottom_label, F_TINY_B)


def mini_title_box(draw, box, title, fill, outline):
    x1, y1, x2, y2 = box
    rounded(draw, box, fill=fill, outline=outline, width=3, r=18)
    draw.rounded_rectangle((x1, y1, x2, y1 + 38), radius=18, fill=WHITE)
    draw.rectangle((x1, y1 + 20, x2, y1 + 38), fill=WHITE)
    center(draw, (x1, y1 + 2, x2, y1 + 38), title, F_SMALL_B)


def draw_network(draw, box, banned=False):
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

    if banned:
        for cx, cy in [(x1 + 70, y1 + 45), (x1 + 115, y1 + 105), (x1 + 165, y1 + 68)]:
            cross(draw, (cx - 9, cy - 9, cx + 9, cy + 9), color=(255, 120, 120), width=5)


def save(canvas, out_dir, name):
    png = out_dir / f"{name}.png"
    pdf = out_dir / f"{name}.pdf"
    canvas.save(png)
    canvas.save(pdf, "PDF", resolution=300)


def load_gtsrb_sample(data_root, idx=123):
    roots = [data_root / "gtsrb", data_root / "GTSRB", data_root]
    exts = {".ppm", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    files = []
    for root in roots:
        if not root.exists():
            continue
        cur = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
        if cur:
            files = sorted(cur)
            break
    if not files:
        raise FileNotFoundError("Cannot find GTSRB image files under data-root.")
    return Image.open(files[idx % len(files)]).convert("RGB")


def load_all_samples(root):
    ds10 = CIFAR10(root=str(root), train=True, download=False)
    ds100 = CIFAR100(root=str(root), train=True, download=False)

    c10_imgs = [ds10[i][0].convert("RGB") for i in [11, 22, 35, 47, 58, 64, 77, 89, 93, 111, 125, 139]]
    c100_img = ds100[25][0].convert("RGB")
    gtsrb_img = load_gtsrb_sample(root, 123)

    return c10_imgs, gtsrb_img, c100_img


def fig1(out_dir, imgs):
    W, H = 1900, 760
    canvas = Image.new("RGB", (W, H), WHITE)
    draw = ImageDraw.Draw(canvas)

    draw.text((55, 34), "(a) Configuration-conditioned activation", font=F_TITLE, fill=BLACK)

    dashed_rect(draw, (35, 95, 1865, 650), DASH_BLUE)
    center(draw, (35, 105, 1865, 155), "same triggers, different configurations, different behavior", F_PANEL)

    base = imgs[0]
    cases = [
        ("Clean", base, "original", GRAY, BG_GRAY),
        ("T1 only", add_trigger(base, "tl", RED), "original", ORANGE, BG_YELLOW),
        ("T2 only", add_trigger(base, "br", BLUE), "original", ORANGE, BG_YELLOW),
        ("Invalid dual", add_dual(base, "tl", "tr"), "original", RED, BG_RED),
        ("Valid c1", add_dual(base, "tl", "br"), "target t1", GREEN, BG_GREEN),
        ("Valid c2", add_dual(base, "tr", "bl"), "target t2", GREEN, BG_GREEN),
    ]

    x0, y0 = 80, 225
    cell_w, cell_h = 230, 245
    gap = 58

    for i, (name, im, out, border, bg) in enumerate(cases):
        x = x0 + i * (cell_w + gap)
        sample_box(canvas, draw, (x, y0, x + cell_w, y0 + cell_h), im, name, "", border, bg)
        arrow(draw, (x + cell_w // 2, y0 + cell_h + 18), (x + cell_w // 2, y0 + cell_h + 68), width=3)
        rounded(draw, (x + 25, y0 + cell_h + 78, x + cell_w - 25, y0 + cell_h + 138), fill=WHITE, outline=border, width=3, r=14)
        center(draw, (x + 25, y0 + cell_h + 78, x + cell_w - 25, y0 + cell_h + 138), out, F_SMALL_B)

    draw.rectangle((75, 685, 105, 715), fill=RED)
    draw.text((118, 687), "T1", font=F_SMALL, fill=BLACK)
    draw.rectangle((175, 685, 205, 715), fill=BLUE)
    draw.text((218, 687), "T2", font=F_SMALL, fill=BLACK)

    save(canvas, out_dir, "fig1_ccdt_overview_STYLE_FINAL")


def fig2(out_dir, c10_imgs, gtsrb_img, c100_img):
    W, H = 1900, 980
    canvas = Image.new("RGB", (W, H), WHITE)
    draw = ImageDraw.Draw(canvas)

    draw.text((55, 34), "(b) Real triggered input examples", font=F_TITLE, fill=BLACK)

    cols = ["Clean", "T1 only", "T2 only", "Invalid dual", "Valid dual"]
    rows = [
        ("CIFAR-10", c10_imgs[0]),
        ("GTSRB", gtsrb_img),
        ("CIFAR-100", c100_img),
    ]

    left = 260
    top = 155
    cell_w = 250
    cell_h = 215
    gap_x = 40
    gap_y = 70

    for c, name in enumerate(cols):
        x = left + c * (cell_w + gap_x)
        center(draw, (x, 100, x + cell_w, 140), name, F_HEAD)

    for r, (row_name, base) in enumerate(rows):
        y = top + r * (cell_h + gap_y)
        center(draw, (55, y + 70, 220, y + 140), row_name, F_HEAD)

        ims = [
            base,
            add_trigger(base, "tl", RED),
            add_trigger(base, "br", BLUE),
            add_dual(base, "tl", "tr"),
            add_dual(base, "tl", "br"),
        ]
        borders = [GRAY, ORANGE, ORANGE, RED, GREEN]
        bgs = [BG_GRAY, BG_YELLOW, BG_YELLOW, BG_RED, BG_GREEN]

        for c, im in enumerate(ims):
            x = left + c * (cell_w + gap_x)
            sample_box(canvas, draw, (x, y, x + cell_w, y + cell_h), im, "", "", borders[c], bgs[c])

    draw.rectangle((75, 910, 105, 940), fill=RED)
    draw.text((118, 912), "T1", font=F_SMALL, fill=BLACK)
    draw.rectangle((175, 910, 205, 940), fill=BLUE)
    draw.text((218, 912), "T2", font=F_SMALL, fill=BLACK)

    save(canvas, out_dir, "fig2_real_triggered_examples_STYLE_FINAL")


def fig3(out_dir, imgs):
    W, H = 2100, 900
    canvas = Image.new("RGB", (W, H), WHITE)
    draw = ImageDraw.Draw(canvas)

    draw.text((55, 32), "(c) Naive dual-trigger training creates shortcut leakage", font=F_TITLE, fill=BLACK)

    dashed_rect(draw, (35, 95, 760, 810), DASH_BLUE)
    dashed_rect(draw, (820, 95, 1280, 810), DASH_GRAY)
    dashed_rect(draw, (1340, 95, 2065, 810), DASH_BLUE)

    center(draw, (35, 102, 760, 150), "training signal", F_PANEL)
    center(draw, (820, 102, 1280, 150), "shortcut model", F_PANEL)
    center(draw, (1340, 102, 2065, 150), "unwanted activations", F_PANEL)

    mini_title_box(draw, (75, 185, 720, 415), "valid positives", BG_GREEN, GREEN)
    valid = [
        add_dual(imgs[0], "tl", "br"),
        add_dual(imgs[1], "tl", "br"),
        add_dual(imgs[2], "tr", "bl"),
        add_dual(imgs[3], "tr", "bl"),
        add_dual(imgs[4], "tl", "br"),
        add_dual(imgs[5], "tr", "bl"),
    ]
    labels = ["c1→t1", "c1→t1", "c2→t2", "c2→t2", "c1→t1", "c2→t2"]
    for i, (im, lab) in enumerate(zip(valid, labels)):
        x = 115 + i * 90
        sample_box(canvas, draw, (x, 250, x + 70, 350), im, "", lab, GREEN, WHITE)

    mini_title_box(draw, (75, 505, 720, 735), "missing negatives", BG_RED, RED)
    missing = [
        add_trigger(imgs[6], "tl", RED),
        add_trigger(imgs[7], "br", BLUE),
        add_dual(imgs[8], "tl", "tr"),
        add_dual(imgs[9], "bl", "br"),
    ]
    miss_labels = ["T1", "T2", "invalid", "invalid"]
    for i, (im, lab) in enumerate(zip(missing, miss_labels)):
        x = 145 + i * 125
        sample_box(canvas, draw, (x, 570, x + 80, 685), im, lab, "", RED, WHITE)
    cross(draw, (115, 550, 690, 710), RED, 7)

    rounded(draw, (900, 215, 1200, 610), fill=WHITE, outline=GRAY, width=3, r=18)
    center(draw, (900, 225, 1200, 270), "naive model", F_HEAD)
    draw_network(draw, (930, 300, 1170, 500), banned=True)
    rounded(draw, (930, 650, 1170, 720), fill=BG_YELLOW, outline=ORANGE, width=3, r=16)
    center(draw, (930, 650, 1170, 720), "shortcut rule", F_SMALL_B)

    arrow(draw, (760, 300), (900, 380))
    arrow(draw, (760, 620), (900, 500))

    mini_title_box(draw, (1385, 180, 2025, 405), "single-trigger leakage", BG_RED, RED)
    leak_a = [
        add_trigger(imgs[0], "tl", RED),
        add_trigger(imgs[1], "br", BLUE),
        add_trigger(imgs[2], "tl", RED),
        add_trigger(imgs[3], "br", BLUE),
    ]
    leak_a_labels = ["T1→t", "T2→t", "T1→t", "T2→t"]
    for i, (im, lab) in enumerate(zip(leak_a, leak_a_labels)):
        x = 1435 + i * 120
        sample_box(canvas, draw, (x, 245, x + 82, 355), im, "", lab, RED, WHITE)

    mini_title_box(draw, (1385, 510, 2025, 735), "invalid-configuration leakage", BG_RED, RED)
    leak_b = [
        add_dual(imgs[4], "tl", "tr"),
        add_dual(imgs[5], "bl", "br"),
        add_dual(imgs[6], "tr", "br"),
        add_dual(imgs[7], "tl", "bl"),
    ]
    leak_b_labels = ["inv.→t", "inv.→t", "inv.→t", "inv.→t"]
    for i, (im, lab) in enumerate(zip(leak_b, leak_b_labels)):
        x = 1435 + i * 120
        sample_box(canvas, draw, (x, 575, x + 82, 685), im, "", lab, RED, WHITE)

    arrow(draw, (1200, 395), (1385, 292))
    arrow(draw, (1200, 510), (1385, 625))

    draw.rectangle((75, 835, 105, 865), fill=RED)
    draw.text((118, 837), "T1", font=F_SMALL, fill=BLACK)
    draw.rectangle((180, 835, 210, 865), fill=BLUE)
    draw.text((223, 837), "T2", font=F_SMALL, fill=BLACK)
    draw.text((300, 837), "t: target    inv.: invalid configuration", font=F_SMALL, fill=BLACK)

    save(canvas, out_dir, "fig3_naive_shortcut_motivation_STYLE_FINAL")


def fig4(out_dir, imgs):
    W, H = 2400, 1080
    canvas = Image.new("RGB", (W, H), WHITE)
    draw = ImageDraw.Draw(canvas)

    draw.text((55, 34), "(d) CCDT three-stage pipeline", font=F_TITLE, fill=BLACK)

    poison_box = (35, 100, 1285, 965)
    train_box = (1360, 100, 1715, 965)
    result_box = (1790, 100, 2365, 965)

    dashed_rect(draw, poison_box, DASH_BLUE)
    dashed_rect(draw, train_box, DASH_GRAY)
    dashed_rect(draw, result_box, DASH_BLUE)

    center(draw, (35, 112, 1285, 165), "1. Poisoning", F_PANEL)
    center(draw, (1360, 112, 1715, 165), "2. Training", F_PANEL)
    center(draw, (1790, 112, 2365, 165), "3. Result", F_PANEL)

    mini_title_box(draw, (80, 190, 470, 410), "trigger set", BG_GRAY, GRAY)
    draw.rectangle((130, 260, 180, 310), fill=RED)
    center(draw, (200, 250, 280, 320), "T1", F_HEAD)
    draw.rectangle((130, 335, 180, 385), fill=BLUE)
    center(draw, (200, 325, 280, 395), "T2", F_HEAD)

    mini_title_box(draw, (520, 190, 1235, 410), "valid configurations", BG_BLUE, BLUE)
    sample_box(canvas, draw, (575, 245, 725, 380), add_dual(imgs[0], "tl", "br"), "config c1", "target t1", GREEN, WHITE)
    sample_box(canvas, draw, (805, 245, 955, 380), add_dual(imgs[1], "tr", "bl"), "config c2", "target t2", GREEN, WHITE)
    center(draw, (1015, 255, 1195, 315), "same triggers", F_SMALL_B)
    center(draw, (1015, 325, 1195, 385), "different positions", F_SMALL_B)

    mini_title_box(draw, (80, 470, 1235, 890), "poisoned training set", (250, 250, 250), GRAY)

    cells = [
        ("clean", imgs[2], "original", GRAY, BG_GRAY),
        ("T1 only", add_trigger(imgs[3], "tl", RED), "original", ORANGE, BG_YELLOW),
        ("T2 only", add_trigger(imgs[4], "br", BLUE), "original", ORANGE, BG_YELLOW),
        ("valid c1", add_dual(imgs[5], "tl", "br"), "target t1", GREEN, BG_GREEN),
        ("valid c2", add_dual(imgs[6], "tr", "bl"), "target t2", GREEN, BG_GREEN),
        ("invalid", add_dual(imgs[7], "tl", "tr"), "original", RED, BG_RED),
    ]

    start_x = 150
    start_y = 535
    w = 145
    h = 140
    gap_x = 48
    gap_y = 60

    for i, (top, im, bottom, border, bg) in enumerate(cells):
        r = i // 3
        c = i % 3
        x = start_x + c * (w + gap_x)
        y = start_y + r * (h + gap_y)
        sample_box(canvas, draw, (x, y, x + w, y + h), im, top, bottom, border, bg)

    rounded(draw, (965, 595, 1190, 770), fill=WHITE, outline=BLUE, width=3, r=18)
    center(draw, (965, 605, 1190, 640), "mixed set", F_SMALL_B)
    for i, color in enumerate([GRAY, ORANGE, ORANGE, GREEN, GREEN, RED]):
        xx = 1005 + (i % 3) * 48
        yy = 655 + (i // 3) * 48
        draw.rectangle((xx, yy, xx + 34, yy + 34), outline=color, width=4, fill=(250, 250, 250))
    arrow(draw, (765, 665), (965, 665), width=3)

    arrow(draw, (1285, 545), (1360, 545), width=5)

    mini_title_box(draw, (1405, 235, 1670, 790), "CCDT model", WHITE, GRAY)
    rounded(draw, (1445, 315, 1630, 545), fill=WHITE, outline=GRAY, width=3, r=16)
    draw_network(draw, (1470, 350, 1605, 500))
    rounded(draw, (1435, 675, 1640, 735), fill=BG_BLUE, outline=BLUE, width=3, r=15)
    center(draw, (1435, 675, 1640, 735), "clean + selective", F_SMALL_B)

    arrow(draw, (1715, 545), (1790, 545), width=5)

    mini_title_box(draw, (1825, 185, 2325, 880), "selective activation", WHITE, BLUE)

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

    draw.rectangle((65, 1000, 95, 1030), fill=RED)
    draw.text((108, 1001), "T1", font=F_SMALL, fill=BLACK)
    draw.rectangle((170, 1000, 200, 1030), fill=BLUE)
    draw.text((213, 1001), "T2", font=F_SMALL, fill=BLACK)
    draw.text((300, 1001), "original: benign label    target: attack label", font=F_SMALL, fill=BLACK)

    save(canvas, out_dir, "fig4_ccdt_pipeline_STYLE_FINAL")


def contact_sheet(out_dir):
    files = [
        "fig1_ccdt_overview_STYLE_FINAL.png",
        "fig2_real_triggered_examples_STYLE_FINAL.png",
        "fig3_naive_shortcut_motivation_STYLE_FINAL.png",
        "fig4_ccdt_pipeline_STYLE_FINAL.png",
    ]

    imgs = [Image.open(out_dir / f).convert("RGB") for f in files]

    thumb_w = 1100
    thumbs = []
    for im in imgs:
        h = int(im.size[1] * thumb_w / im.size[0])
        thumbs.append(im.resize((thumb_w, h), Image.Resampling.LANCZOS))

    margin = 35
    label_h = 38
    cell_h = max(im.size[1] for im in thumbs) + label_h
    W = thumb_w * 2 + margin * 3
    H = cell_h * 2 + margin * 3

    canvas = Image.new("RGB", (W, H), WHITE)
    draw = ImageDraw.Draw(canvas)

    for i, im in enumerate(thumbs):
        r = i // 2
        c = i % 2
        x = margin + c * (thumb_w + margin)
        y = margin + r * (cell_h + margin)

        draw.text((x, y), f"Figure {i + 1}", font=F_TEXT_B, fill=BLACK)
        canvas.paste(im, (x, y + label_h))
        draw.rectangle((x, y + label_h, x + im.size[0], y + label_h + im.size[1]), outline=(200, 200, 200), width=2)

    canvas.save(out_dir / "contact_sheet_STYLE_FINAL.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    c10_imgs, gtsrb_img, c100_img = load_all_samples(Path(args.data_root))

    fig1(out_dir, c10_imgs)
    fig2(out_dir, c10_imgs, gtsrb_img, c100_img)
    fig3(out_dir, c10_imgs)
    fig4(out_dir, c10_imgs)
    contact_sheet(out_dir)

    print("[OK] regenerated four main figures:")
    for p in sorted(out_dir.glob("*STYLE_FINAL*")):
        print(" -", p)
    print(" -", out_dir / "contact_sheet_STYLE_FINAL.png")


if __name__ == "__main__":
    main()
