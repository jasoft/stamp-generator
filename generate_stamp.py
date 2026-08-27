#!/usr/bin/env python3
"""Generate a 3D printable Chinese official seal (公章) STL model.
Diameter: 40mm, based on uploaded image.
Company: 上海逗号软件科技有限公司
Registration: 3201041477313
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from skimage import measure
import trimesh
import math
import os

# ============ Parameters ============
DIAMETER = 40.0          # mm - outer diameter
RADIUS = DIAMETER / 2.0  # 20mm
BASE_H = 3.0             # mm - base plate thickness
FEAT_H = 1.0             # mm - raised feature height
RING_WIDTH = 1.2         # mm - border ring width

# Resolution
RES_2D = 0.08            # mm per pixel for 2D image
RES_Z = 0.1              # mm per voxel for z-axis
N = int(DIAMETER / RES_2D) + 1   # 2D image size (~501)
N_Z = int((BASE_H + FEAT_H) / RES_Z) + 1  # z-voxel count (~41)
BASE_Z = int(BASE_H / RES_Z)     # base z-voxels (~30)

# Text content
COMPANY_NAME = "上海逗号软件科技有限公司"  # 12 characters
REG_NUMBER = "3201041477313"               # 13 digits

# Company name arc (top)
TEXT_RADIUS = 16.0        # mm - arc radius
TEXT_FONT_SIZE = 4.0     # mm - character size
TEXT_ARC_START = 250.0   # degrees clockwise from top
TEXT_ARC_END = 110.0
TEXT_ARC_SPAN = (TEXT_ARC_START - TEXT_ARC_END) % 360  # 220°

# Registration number arc (bottom)
NUM_RADIUS = 16.0
NUM_FONT_SIZE = 2.0
NUM_ARC_START = 125.0
NUM_ARC_END = 235.0
NUM_ARC_SPAN = NUM_ARC_END - NUM_ARC_START  # 110°

# Star
STAR_OUTER_R = 7.0       # mm

# Output
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_STL = os.path.join(WORK_DIR, "公章_40mm.stl")
OUTPUT_PNG = os.path.join(WORK_DIR, "公章_40mm_2D_preview.png")

def find_font():
    candidates = [
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def draw_star(draw, cx, cy, outer_r_px, inner_r_px):
    """Draw a five-pointed star centered at (cx, cy)."""
    points = []
    for i in range(10):
        angle_deg = 90 + i * 36  # start from top, clockwise
        angle_rad = math.radians(angle_deg)
        r = outer_r_px if i % 2 == 0 else inner_r_px
        # PIL: y increases downward, so flip y
        px = cx + r * math.cos(angle_rad)
        py = cy - r * math.sin(angle_rad)
        points.append((px, py))
    draw.polygon(points, fill=255)


def draw_arc_text(img, cx, cy, text, font, char_size_px, arc_radius_mm,
                  arc_start_deg, arc_span_deg, bottom=False):
    """Draw text characters along a circular arc.
    
    If bottom=False: characters' 'up' points outward (for top arc).
    If bottom=True: characters' 'up' points inward (for bottom arc).
    """
    n = len(text)
    arc_r_px = arc_radius_mm / RES_2D
    tile = int(char_size_px * 3)

    for i, ch in enumerate(text):
        t = i / (n - 1) if n > 1 else 0.5
        if bottom:
            angle = arc_start_deg + t * arc_span_deg
        else:
            angle = arc_start_deg - t * arc_span_deg
        ar = math.radians(angle)

        # Position on arc (PIL coords: y down)
        ax = cx + arc_r_px * math.sin(ar)
        ay = cy - arc_r_px * math.cos(ar)

        # Render character centered in tile
        ci = Image.new('L', (tile, tile), 0)
        cd = ImageDraw.Draw(ci)
        bbox = cd.textbbox((0, 0), ch, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (tile - w) / 2 - bbox[0]
        y = (tile - h) / 2 - bbox[1]
        cd.text((x, y), ch, fill=255, font=font)

        # Rotation: PIL rotate is CCW
        # Top arc: 'up' = outward → rotate CW by angle → PIL rotate(-angle)
        # Bottom arc: 'up' = inward → rotate CW by (angle-180) → PIL rotate(180-angle)
        if bottom:
            rot_angle = 180.0 - angle
        else:
            rot_angle = -angle

        rotated = ci.rotate(rot_angle, expand=True, resample=Image.BICUBIC)
        rx, ry = rotated.size
        img.paste(255, (int(ax - rx / 2), int(ay - ry / 2)), rotated)


def main():
    font_path = find_font()
    if not font_path:
        raise RuntimeError("No Chinese font found on system!")
    print(f"Font: {font_path}")
    print(f"Resolution: {RES_2D} mm/px (2D), {RES_Z} mm/voxel (Z)")
    print(f"Image: {N}x{N}, Voxel Z: {N_Z}")

    # ========== 1. Create 2D stamp impression image ==========
    img = Image.new('L', (N, N), 0)
    draw = ImageDraw.Draw(img)
    cx = cy = N / 2

    # --- Outer ring border (annulus) ---
    r_out = RADIUS / RES_2D
    r_in = (RADIUS - RING_WIDTH) / RES_2D
    draw.ellipse([cx - r_out, cy - r_out, cx + r_out, cy + r_out], fill=255)
    draw.ellipse([cx - r_in, cy - r_in, cx + r_in, cy + r_in], fill=0)

    # --- Five-pointed star at center ---
    star_r_out = STAR_OUTER_R / RES_2D
    star_r_in = star_r_out * math.sin(math.radians(18)) / math.sin(math.radians(126))
    draw_star(draw, cx, cy, star_r_out, star_r_in)

    # --- Company name (top arc, 'up' outward) ---
    text_font_px = int(TEXT_FONT_SIZE / RES_2D)
    text_font = ImageFont.truetype(font_path, text_font_px)
    draw_arc_text(img, cx, cy, COMPANY_NAME, text_font, text_font_px,
                  TEXT_RADIUS, TEXT_ARC_START, TEXT_ARC_SPAN, bottom=False)

    # --- Registration number (bottom arc, 'up' inward) ---
    num_font_px = int(NUM_FONT_SIZE / RES_2D)
    num_font = ImageFont.truetype(font_path, num_font_px)
    draw_arc_text(img, cx, cy, REG_NUMBER, num_font, num_font_px,
                  NUM_RADIUS, NUM_ARC_START, NUM_ARC_SPAN, bottom=True)

    # --- Flip horizontally: stamp face is mirror of impression ---
    img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # Save 2D preview
    img.save(OUTPUT_PNG)
    print(f"2D preview: {OUTPUT_PNG}")

    # ========== 2. Create 3D voxel array ==========
    print("Building 3D voxel array...")
    arr_2d = np.array(img) > 127  # binary mask of raised features

    # Circular mask (entire stamp is a circle)
    yy, xx = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    circle_mask = dist <= RADIUS / RES_2D

    # Combined feature mask (features must be inside circle)
    feat_mask = arr_2d & circle_mask

    # Build voxel array: base (solid) + features (raised)
    voxel = np.zeros((N, N, N_Z), dtype=np.uint8)
    # Base layer: solid cylinder
    voxel[:, :, :BASE_Z] = circle_mask[:, :, np.newaxis].astype(np.uint8)
    # Feature layer: raised features
    voxel[:, :, BASE_Z:] = feat_mask[:, :, np.newaxis].astype(np.uint8)

    print(f"  Voxel array: {voxel.shape}, {voxel.nbytes / 1e6:.1f} MB")

    # ========== 3. Marching cubes → mesh ==========
    print("Running marching cubes...")
    verts, faces, _, _ = measure.marching_cubes(
        voxel.astype(np.float32), level=0.5
    )

    # Scale to millimeters (non-uniform resolution)
    verts[:, 0] *= RES_2D   # x
    verts[:, 1] *= RES_2D   # y
    verts[:, 2] *= RES_Z   # z

    # Create trimesh object
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)

    # Center at origin
    mesh.apply_translation(-mesh.bounding_box.centroid)

    # ========== 4. Export STL ==========
    mesh.export(OUTPUT_STL)
    print(f"\n✓ STL exported: {OUTPUT_STL}")
    print(f"  Vertices: {len(mesh.vertices):,}")
    print(f"  Triangles: {len(mesh.faces):,}")
    print(f"  Dimensions: {mesh.extents[0]:.1f} x {mesh.extents[1]:.1f} x {mesh.extents[2]:.1f} mm")
    print(f"  File size: {os.path.getsize(OUTPUT_STL) / 1e6:.1f} MB")

    # Quick sanity check
    if mesh.is_watertight:
        print("  Watertight: Yes ✓")
    else:
        print("  Watertight: No (may still be printable)")


if __name__ == "__main__":
    main()
