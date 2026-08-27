#!/usr/bin/env python3
"""Test arc text rendering to verify angle system and direction."""
import math
from PIL import Image, ImageDraw, ImageFont

N = 500
cx = cy = N / 2
R = 200

img = Image.new('L', (N, N), 0)
draw = ImageDraw.Draw(img)

# Draw the circle
draw.ellipse([cx-R, cy-R, cx+R, cy+R], outline=128, width=2)

# Draw angle markers (0, 90, 180, 270) using sin/cos formula:
# ax = cx + R * sin(angle_rad)
# ay = cy - R * cos(angle_rad)
# This means: angle=0 -> (0, -R) = top; angle=90 -> (R, 0) = right; angle=180 -> (0, R) = bottom
for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
    ar = math.radians(angle)
    ax = cx + R * math.sin(ar)
    ay = cy - R * math.cos(ar)
    draw.ellipse([ax-5, ay-5, ax+5, ay+5], fill=255)
    draw.text((ax+8, ay-5), f"{angle}°", fill=255)

font = ImageFont.truetype("/System/Library/Fonts/STHeiti Light.ttc", 40)

# Test text: "ABCDEFG" along top arc (0° to 180° clockwise)
text = "ABCDEFG"
start = 200   # 200°
span = 140    # 140° clockwise -> ends at 340°
arc_r = R

for i, ch in enumerate(text):
    t = i / (len(text) - 1)
    angle = start + t * span
    ar = math.radians(angle)
    ax = cx + arc_r * math.sin(ar)
    ay = cy - arc_r * math.cos(ar)

    tile = 120
    ci = Image.new('L', (tile, tile), 0)
    cd = ImageDraw.Draw(ci)
    bbox = cd.textbbox((0, 0), ch, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    cd.text(((tile - w) / 2 - bbox[0], (tile - h) / 2 - bbox[1]),
            ch, fill=255, font=font)

    # outward: rot = -angle (PIL rotate is CCW)
    rot = -angle
    rotated = ci.rotate(rot, expand=True, resample=Image.BICUBIC)
    rx, ry = rotated.size
    img.paste(255, (int(ax - rx / 2), int(ay - ry / 2)), rotated)

# Test number on bottom arc
ntext = "1234567890"
nstart = 135
nspan = 90

nfont = ImageFont.truetype("/System/Library/Fonts/STHeiti Light.ttc", 24)
for i, ch in enumerate(ntext):
    t = i / (len(ntext) - 1)
    angle = nstart + t * nspan
    ar = math.radians(angle)
    ax = cx + arc_r * math.sin(ar)
    ay = cy - arc_r * math.cos(ar)

    tile = 80
    ci = Image.new('L', (tile, tile), 0)
    cd = ImageDraw.Draw(ci)
    bbox = cd.textbbox((0, 0), ch, font=nfont)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    cd.text(((tile - w) / 2 - bbox[0], (tile - h) / 2 - bbox[1]),
            ch, fill=255, font=nfont)

    # inward (bottom): rot = 180 - angle
    rot = 180 - angle
    rotated = ci.rotate(rot, expand=True, resample=Image.BICUBIC)
    rx, ry = rotated.size
    img.paste(255, (int(ax - rx / 2), int(ay - ry / 2)), rotated)

img.save("/Users/weiwang/Library/Application Support/TRAE SOLO CN/ModularData/ai-agent/work-mode-projects/6a8fc635fcf6f162d72391e4/test_arc.png")
print("Saved test_arc.png")
print(f"Text arc: start={start}°, span={span}°, ends at {start+span}°")
print(f"Number arc: start={nstart}°, span={nspan}°, ends at {nstart+nspan}°")
