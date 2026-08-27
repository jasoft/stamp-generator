#!/usr/bin/env python3
"""Generate a 3D printable Chinese official seal (公章) STL model.
Direct mesh creation from heightmap (watertight).

Diameter: 40mm. Company: 上海逗号软件科技有限公司. Reg: 3201041477313.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import struct
import math
import os

# ============ Parameters ============
DIAMETER = 40.0
RADIUS = DIAMETER / 2.0
BASE_H = 3.0
FEAT_H = 1.0
RING_WIDTH = 1.2
RES = 0.1              # mm per pixel
N = int(DIAMETER / RES) + 1   # 401

COMPANY_NAME = "南京正赞软件科技有限公司"
REG_NUMBER = "3201041477313"

# Company name: top arc, clockwise from left-side to right-side via top
TEXT_RADIUS = 15.5
TEXT_FONT_SIZE = 6
TEXT_ARC_START = 260.0   # start at left (8:40 position)
TEXT_ARC_SPAN = 200.0    # clockwise span, ends at 260+200=460=100° (3:20 position)

# Registration number: bottom arc, clockwise from lower-right to lower-left (centered at bottom)
NUM_RADIUS = 16.0
NUM_FONT_SIZE = 2.2
NUM_ARC_START = 140.0    # start at lower-right (4:40 position)
NUM_ARC_SPAN = 80.0      # clockwise span, ends at 140+80=220° (7:20 position)

STAR_OUTER_R = 7.0

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_STL = os.path.join(WORK_DIR, "公章_40mm.stl")
OUTPUT_PNG = os.path.join(WORK_DIR, "公章_40mm_2D_preview.png")


def find_font():
    for p in [
        "/tmp/simsun.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]:
        if os.path.exists(p):
            return p
    return None


def draw_star(draw, cx, cy, r_out, r_in):
    pts = []
    for i in range(10):
        a = math.radians(90 + i * 36)
        r = r_out if i % 2 == 0 else r_in
        pts.append((cx + r * math.cos(a), cy - r * math.sin(a)))
    draw.polygon(pts, fill=255)


def draw_arc_text(img, cx, cy, text, font, tile, arc_r_px,
                  arc_start, arc_span, clockwise=True, inward=False):
    """Draw text along a circular arc.

    Args:
        clockwise: if True, text advances clockwise (angle increases)
        inward: if True, character top points toward center (bottom arc style)
    """
    n = len(text)
    for i, ch in enumerate(text):
        t = i / (n - 1) if n > 1 else 0.5
        if clockwise:
            angle = arc_start + t * arc_span
        else:
            angle = arc_start - t * arc_span
        ar = math.radians(angle)
        ax = cx + arc_r_px * math.sin(ar)
        ay = cy - arc_r_px * math.cos(ar)

        ci = Image.new('L', (tile, tile), 0)
        cd = ImageDraw.Draw(ci)
        bbox = cd.textbbox((0, 0), ch, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        cd.text(((tile - w) / 2 - bbox[0], (tile - h) / 2 - bbox[1]),
                ch, fill=255, font=font)

        # Rotation: PIL rotate is CCW (positive angle = CCW)
        # outward (top of char points away from center): rot = -angle
        # inward (top of char points toward center): rot = 180 - angle
        rot = (180.0 - angle) if inward else (-angle)
        rotated = ci.rotate(rot, expand=True, resample=Image.BICUBIC)
        rx, ry = rotated.size
        img.paste(255, (int(ax - rx / 2), int(ay - ry / 2)), rotated)


def render_2d():
    """Render the 2D stamp impression and flip for stamp face."""
    img = Image.new('L', (N, N), 0)
    draw = ImageDraw.Draw(img)
    cx = cy = N / 2

    # Ring border
    r_out = RADIUS / RES
    r_in = (RADIUS - RING_WIDTH) / RES
    draw.ellipse([cx - r_out, cy - r_out, cx + r_out, cy + r_out], fill=255)
    draw.ellipse([cx - r_in, cy - r_in, cx + r_in, cy + r_in], fill=0)

    # Star
    s_out = STAR_OUTER_R / RES
    s_in = s_out * math.sin(math.radians(18)) / math.sin(math.radians(126))
    draw_star(draw, cx, cy, s_out, s_in)

    # Company name (top arc, clockwise, text faces outward)
    font_px = int(TEXT_FONT_SIZE / RES)
    font = ImageFont.truetype(find_font(), font_px)
    draw_arc_text(img, cx, cy, COMPANY_NAME, font, font_px * 3,
                  TEXT_RADIUS / RES, TEXT_ARC_START, TEXT_ARC_SPAN,
                  clockwise=True, inward=False)

    # Registration number (bottom arc, clockwise, text faces inward)
    npx = int(NUM_FONT_SIZE / RES)
    nfont = ImageFont.truetype(find_font(), npx)
    draw_arc_text(img, cx, cy, REG_NUMBER, nfont, npx * 3,
                  NUM_RADIUS / RES, NUM_ARC_START, NUM_ARC_SPAN,
                  clockwise=True, inward=True)

    # Flip for stamp face
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img.save(OUTPUT_PNG)
    return np.array(img) > 127


def build_mesh(arr_2d):
    """Build a watertight mesh directly from the 2D feature mask + heightmap."""
    cx = cy = N / 2

    # Circle mask
    yy, xx = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    circle = dist <= RADIUS / RES

    # Heightmap
    hmap = np.zeros((N, N), dtype=np.float32)
    hmap[circle] = BASE_H
    hmap[arr_2d & circle] = BASE_H + FEAT_H

    # ===== Vertices =====
    # Top: 0 .. N*N-1 ; Bottom: N*N .. 2*N*N-1
    j_grid, i_grid = np.meshgrid(np.arange(N), np.arange(N))
    x = (j_grid - N / 2) * RES
    y = (N / 2 - i_grid) * RES  # flip y so it increases upward

    top_verts = np.stack([x.ravel(), y.ravel(), hmap.ravel()], axis=1).astype(np.float32)
    bot_verts = np.stack([x.ravel(), y.ravel(), np.zeros(N * N)], axis=1).astype(np.float32)
    all_verts = np.vstack([top_verts, bot_verts])

    NN = N * N  # offset for bottom vertices

    # ===== Cell mask (all 4 corners have h > 0) =====
    h00 = hmap[:-1, :-1]
    h01 = hmap[:-1, 1:]
    h10 = hmap[1:, :-1]
    h11 = hmap[1:, 1:]
    cell = (h00 > 0) & (h01 > 0) & (h10 > 0) & (h11 > 0)

    # Vertex index grids (for cells)
    vi00 = np.arange(N * N).reshape(N, N)[:-1, :-1]
    vi01 = np.arange(N * N).reshape(N, N)[:-1, 1:]
    vi10 = np.arange(N * N).reshape(N, N)[1:, :-1]
    vi11 = np.arange(N * N).reshape(N, N)[1:, 1:]

    m = cell.ravel()
    v00 = vi00.ravel()[m]
    v01 = vi01.ravel()[m]
    v10 = vi10.ravel()[m]
    v11 = vi11.ravel()[m]

    # ===== Top faces (+z normal): (v00, v10, v11), (v00, v11, v01) =====
    top_f = np.empty((len(v00) * 2, 3), dtype=np.int64)
    top_f[0::2] = np.stack([v00, v10, v11], axis=1)
    top_f[1::2] = np.stack([v00, v11, v01], axis=1)

    # ===== Bottom faces (-z normal): reversed winding =====
    b00, b01, b10, b11 = v00 + NN, v01 + NN, v10 + NN, v11 + NN
    bot_f = np.empty((len(v00) * 2, 3), dtype=np.int64)
    bot_f[0::2] = np.stack([b00, b11, b10], axis=1)
    bot_f[1::2] = np.stack([b00, b01, b11], axis=1)

    # ===== Side walls =====
    wall_faces = []

    def add_wall(vt1, vt2, vb1, vb2, reverse=False):
        """Add 2 triangles for a wall quad. vt=top, vb=bottom."""
        bt1, bt2 = vb1, vb2
        if reverse:
            wall_faces.append(np.array([[vt1, bt2, vt2], [vt1, bt1, bt2]], dtype=np.int64))
        else:
            wall_faces.append(np.array([[vt1, vt2, bt2], [vt1, bt2, bt1]], dtype=np.int64))

    # Left boundary: cell[i,j] included but cell[i,j-1] not
    left = np.zeros_like(cell)
    left[:, 1:] = cell[:, 1:] & ~cell[:, :-1]
    left[:, 0] = cell[:, 0]
    li, lj = np.where(left)
    for k in range(len(li)):
        i, j = li[k], lj[k]
        vt1 = i * N + j          # (i, j) top
        vt2 = (i + 1) * N + j    # (i+1, j) top
        vb1 = vt1 + NN
        vb2 = vt2 + NN
        add_wall(vt1, vt2, vb1, vb2, reverse=True)  # -x outward

    # Right boundary
    right = np.zeros_like(cell)
    right[:, :-1] = cell[:, :-1] & ~cell[:, 1:]
    right[:, -1] = cell[:, -1]
    ri, rj = np.where(right)
    for k in range(len(ri)):
        i, j = ri[k], rj[k]
        vt1 = i * N + (j + 1)
        vt2 = (i + 1) * N + (j + 1)
        vb1 = vt1 + NN
        vb2 = vt2 + NN
        add_wall(vt1, vt2, vb1, vb2, reverse=False)  # +x outward

    # Top boundary (smaller i, larger y)
    top = np.zeros_like(cell)
    top[1:, :] = cell[1:, :] & ~cell[:-1, :]
    top[0, :] = cell[0, :]
    ti, tj = np.where(top)
    for k in range(len(ti)):
        i, j = ti[k], tj[k]
        vt1 = i * N + j
        vt2 = i * N + (j + 1)
        vb1 = vt1 + NN
        vb2 = vt2 + NN
        add_wall(vt1, vt2, vb1, vb2, reverse=False)  # +y outward

    # Bottom boundary (larger i, smaller y)
    btm = np.zeros_like(cell)
    btm[:-1, :] = cell[:-1, :] & ~cell[1:, :]
    btm[-1, :] = cell[-1, :]
    bi, bj = np.where(btm)
    for k in range(len(bi)):
        i, j = bi[k], bj[k]
        vt1 = (i + 1) * N + j
        vt2 = (i + 1) * N + (j + 1)
        vb1 = vt1 + NN
        vb2 = vt2 + NN
        add_wall(vt1, vt2, vb1, vb2, reverse=True)  # -y outward

    wall_f = np.vstack(wall_faces) if wall_faces else np.zeros((0, 3), dtype=np.int64)

    all_faces = np.vstack([top_f, bot_f, wall_f])
    return all_verts, all_faces


def write_binary_stl(filename, vertices, faces):
    """Write a binary STL file."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normals = (normals / norms).astype(np.float32)

    dt = np.dtype([('normal', '<f4', 3), ('v0', '<f4', 3),
                   ('v1', '<f4', 3), ('v2', '<f4', 3), ('attr', '<u2')])
    data = np.empty(len(faces), dtype=dt)
    data['normal'] = normals
    data['v0'] = v0.astype(np.float32)
    data['v1'] = v1.astype(np.float32)
    data['v2'] = v2.astype(np.float32)
    data['attr'] = 0

    with open(filename, 'wb') as f:
        f.write(b'\0' * 80)
        f.write(struct.pack('<I', len(faces)))
        f.write(data.tobytes())


def main():
    print(f"Resolution: {RES} mm/px, Image: {N}x{N}")

    # 1. Render 2D
    arr_2d = render_2d()
    print(f"2D preview: {OUTPUT_PNG}")
    print(f"Feature pixels: {arr_2d.sum():,}")

    # 2. Build mesh
    print("Building mesh...")
    verts, faces = build_mesh(arr_2d)

    # Center at origin
    verts[:, 0] -= verts[:, 0].mean()
    verts[:, 1] -= verts[:, 1].mean()

    # 3. Write STL
    write_binary_stl(OUTPUT_STL, verts, faces)

    # Stats
    xs, ys, zs = verts[:, 0], verts[:, 1], verts[:, 2]
    print(f"\nSTL: {OUTPUT_STL}")
    print(f"  Vertices: {len(verts):,}")
    print(f"  Triangles: {len(faces):,}")
    print(f"  Dimensions: {xs.max()-xs.min():.1f} x {ys.max()-ys.min():.1f} x {zs.max()-zs.min():.1f} mm")
    print(f"  File size: {os.path.getsize(OUTPUT_STL) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
