#!/usr/bin/env python3
"""Web-based Chinese official seal (公章) STL generator.
Flask backend with real-time 2D preview and STL export.
"""

import io
import math
import os
import struct
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from flask import Flask, request, jsonify, send_file, render_template_string

app = Flask(__name__)

WORK_DIR = os.path.dirname(os.path.abspath(__file__))

# Font lookup
FONT_PATHS = [
    os.path.join(WORK_DIR, "simsun.ttc"),
    "/tmp/simsun.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]

def find_font():
    for p in FONT_PATHS:
        if os.path.exists(p):
            return p
    return None

FONT_PATH = find_font()


def draw_star(draw, cx, cy, r_out, r_in):
    pts = []
    for i in range(10):
        a = math.radians(90 + i * 36)
        r = r_out if i % 2 == 0 else r_in
        pts.append((cx + r * math.cos(a), cy - r * math.sin(a)))
    draw.polygon(pts, fill=255)


def draw_arc_text(img, cx, cy, text, font, tile, arc_r_px,
                  arc_start, arc_span, clockwise=True, inward=False, v_scale=1.0):
    n = len(text)
    for i, ch in enumerate(text):
        t = i / (n - 1) if n > 1 else 0.5
        angle = (arc_start + t * arc_span) if clockwise else (arc_start - t * arc_span)
        ar = math.radians(angle)
        ax = cx + arc_r_px * math.sin(ar)
        ay = cy - arc_r_px * math.cos(ar)

        ci = Image.new('L', (tile, tile), 0)
        cd = ImageDraw.Draw(ci)
        bbox = cd.textbbox((0, 0), ch, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        cd.text(((tile - w) / 2 - bbox[0], (tile - h) / 2 - bbox[1]),
                ch, fill=255, font=font)

        if v_scale != 1.0:
            new_h = max(1, int(tile * v_scale))
            ci = ci.resize((tile, new_h), Image.LANCZOS)

        rot = (180.0 - angle) if inward else (-angle)
        rotated = ci.rotate(rot, expand=True, resample=Image.BICUBIC)
        rx, ry = rotated.size
        img.paste(255, (int(ax - rx / 2), int(ay - ry / 2)), rotated)


def render_stamp(params, res_mm=0.1):
    """Render 2D stamp image from parameters."""
    diameter = params.get('diameter', 40.0)
    ring_width = params.get('ring_width', 1.2)
    company = params.get('company_name', '上海锦绣科技有限公司')
    reg_num = params.get('reg_number', '3201041477313')
    text_radius = params.get('text_radius', 15.5)
    text_size = params.get('text_size', 4.5)
    text_start = params.get('text_start', 260.0)
    text_span = params.get('text_span', 200.0)
    num_radius = params.get('num_radius', 16.0)
    num_size = params.get('num_size', 2.2)
    num_start = params.get('num_start', 140.0)
    num_span = params.get('num_span', 80.0)
    star_r = params.get('star_r', 7.0)
    stroke_thicken = params.get('stroke_thicken', 0.25)
    text_height = params.get('text_height', 1.0)

    n = int(diameter / res_mm) + 1
    radius = diameter / 2.0
    img = Image.new('L', (n, n), 0)
    draw = ImageDraw.Draw(img)
    cx = cy = n / 2

    # Ring border
    r_out = radius / res_mm
    r_in = (radius - ring_width) / res_mm
    draw.ellipse([cx - r_out, cy - r_out, cx + r_out, cy + r_out], fill=255)
    draw.ellipse([cx - r_in, cy - r_in, cx + r_in, cy + r_in], fill=0)

    # Star
    s_out = star_r / res_mm
    s_in = s_out * math.sin(math.radians(18)) / math.sin(math.radians(126))
    draw_star(draw, cx, cy, s_out, s_in)

    # Company name (top arc, outward) with vertical scale
    font_px = max(8, int(text_size / res_mm))
    font = ImageFont.truetype(FONT_PATH, font_px)
    draw_arc_text(img, cx, cy, company, font, font_px * 3,
                  text_radius / res_mm, text_start, text_span,
                  clockwise=True, inward=False, v_scale=text_height)

    # Registration number (bottom arc, inward) with vertical scale
    npx = max(6, int(num_size / res_mm))
    nfont = ImageFont.truetype(FONT_PATH, npx)
    draw_arc_text(img, cx, cy, reg_num, nfont, npx * 3,
                  num_radius / res_mm, num_start, num_span,
                  clockwise=True, inward=True, v_scale=text_height)

    # Dilate features
    dilate_px = max(0, int(round(stroke_thicken / res_mm)))
    for _ in range(dilate_px):
        img = img.filter(ImageFilter.MaxFilter(3))

    # Flip for stamp face
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def _greedy_mesh(mask):
    """Greedily merge True pixels into rectangular quads.
    Returns list of (y0, x0, y1, x1) tuples."""
    n_rows, n_cols = mask.shape
    visited = np.zeros((n_rows, n_cols), dtype=bool)
    quads = []
    for y in range(n_rows):
        x = 0
        while x < n_cols:
            if visited[y, x] or not mask[y, x]:
                x += 1
                continue
            w = 1
            while x + w < n_cols and mask[y, x + w] and not visited[y, x + w]:
                w += 1
            h = 1
            while y + h < n_rows:
                row = mask[y + h, x:x + w]
                vis = visited[y + h, x:x + w]
                if not np.all(row) or np.any(vis):
                    break
                h += 1
            visited[y:y + h, x:x + w] = True
            quads.append((y, x, y + h, x + w))
            x += w
    return quads


def build_mesh(arr_2d, diameter, base_h, feat_h, res_mm=0.05):
    """Build a watertight mesh using greedy meshing for compact file size."""
    n = arr_2d.shape[0]
    radius = diameter / 2.0
    cx = cy = n / 2

    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    circle = dist <= radius / res_mm

    hmap = np.zeros((n, n), dtype=np.float32)
    hmap[circle] = base_h
    hmap[arr_2d & circle] = base_h + feat_h

    verts = []
    faces = []

    def add_top_quad(y0, x0, y1, x1, h):
        base = len(verts) // 3
        px0 = (x0 - n / 2) * res_mm
        px1 = (x1 - n / 2) * res_mm
        py0 = (n / 2 - y0) * res_mm
        py1 = (n / 2 - y1) * res_mm
        verts.extend([px0, py0, h, px1, py0, h, px1, py1, h, px0, py1, h])
        faces.extend([base, base+3, base+2, base, base+2, base+1])

    def add_bottom_quad(y0, x0, y1, x1):
        base = len(verts) // 3
        px0 = (x0 - n / 2) * res_mm
        px1 = (x1 - n / 2) * res_mm
        py0 = (n / 2 - y0) * res_mm
        py1 = (n / 2 - y1) * res_mm
        verts.extend([px0, py0, 0, px1, py0, 0, px1, py1, 0, px0, py1, 0])
        faces.extend([base, base+1, base+2, base, base+2, base+3])

    def add_wall(x0, y0, x1, y1, z_low, z_high, reverse=False):
        base = len(verts) // 3
        px0 = (x0 - n / 2) * res_mm
        px1 = (x1 - n / 2) * res_mm
        py0 = (n / 2 - y0) * res_mm
        py1 = (n / 2 - y1) * res_mm
        verts.extend([px0, py0, z_low, px1, py1, z_low, px1, py1, z_high, px0, py0, z_high])
        if reverse:
            faces.extend([base, base+1, base+2, base, base+2, base+3])
        else:
            faces.extend([base, base+3, base+2, base, base+2, base+1])

    # Top surface: greedy mesh each height level
    for h in [base_h, base_h + feat_h]:
        mask = (hmap == h)
        if not mask.any():
            continue
        for (y0, x0, y1, x1) in _greedy_mesh(mask):
            add_top_quad(y0, x0, y1, x1, h)

    # Bottom surface: area where hmap > 0
    bottom_mask = hmap > 0
    for (y0, x0, y1, x1) in _greedy_mesh(bottom_mask):
        add_bottom_quad(y0, x0, y1, x1)

    # Walls: horizontal boundaries (between rows y and y+1)
    hdiff = hmap[:-1, :] != hmap[1:, :]
    for y in np.where(hdiff.any(axis=1))[0]:
        x = 0
        while x < n:
            ha = hmap[y, x]
            hb = hmap[y + 1, x]
            if ha == hb:
                x += 1
                continue
            x0 = x
            while x < n and hmap[y, x] == ha and hmap[y + 1, x] == hb:
                x += 1
            add_wall(x0, y + 1, x, y + 1, min(ha, hb), max(ha, hb), reverse=(ha > hb))

    # Walls: vertical boundaries (between columns x and x+1)
    vdiff = hmap[:, :-1] != hmap[:, 1:]
    for x in np.where(vdiff.any(axis=0))[0]:
        y = 0
        while y < n:
            ha = hmap[y, x]
            hb = hmap[y, x + 1]
            if ha == hb:
                y += 1
                continue
            y0 = y
            while y < n and hmap[y, x] == ha and hmap[y, x + 1] == hb:
                y += 1
            add_wall(x + 1, y0, x + 1, y, min(ha, hb), max(ha, hb), reverse=(ha < hb))

    return np.array(verts, dtype=np.float32).reshape(-1, 3), np.array(faces, dtype=np.int32).reshape(-1, 3)


def write_binary_stl_bytes(vertices, faces):
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

    buf = io.BytesIO()
    buf.write(b'\0' * 80)
    buf.write(struct.pack('<I', len(faces)))
    buf.write(data.tobytes())
    return buf.getvalue()


# ========== Routes ==========

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/preview', methods=['POST'])
def api_preview():
    params = request.get_json(force=True)
    img = render_stamp(params, res_mm=0.1)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')


@app.route('/api/generate-stl', methods=['POST'])
def api_generate_stl():
    params = request.get_json(force=True)
    diameter = params.get('diameter', 40.0)
    base_h = params.get('base_h', 3.0)
    feat_h = params.get('feat_h', 1.0)
    res_mm = params.get('resolution', 0.05)

    img = render_stamp(params, res_mm=res_mm)
    arr = np.array(img) > 127
    verts, faces = build_mesh(arr, diameter, base_h, feat_h, res_mm=res_mm)

    # Center at origin
    verts[:, 0] -= verts[:, 0].mean()
    verts[:, 1] -= verts[:, 1].mean()

    stl_bytes = write_binary_stl_bytes(verts, faces)
    buf = io.BytesIO(stl_bytes)
    buf.seek(0)

    company = params.get('company_name', 'stamp')
    filename = f"{company}_公章.stl"
    return send_file(buf, mimetype='application/octet-stream',
                     as_attachment=True, download_name=filename)


@app.route('/robots.txt')
def robots_txt():
    content = """User-agent: *
Allow: /
Sitemap: https://stampmaker.ursoftware.com/sitemap.xml
"""
    return content, 200, {'Content-Type': 'text/plain; charset=utf-8'}


@app.route('/sitemap.xml')
def sitemap_xml():
    content = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://stampmaker.ursoftware.com/</loc>
    <lastmod>2026-08-27</lastmod>
    <changefreq>monthly</changefreq>
    <priority>1.0</priority>
  </url>
</urlset>
"""
    return content, 200, {'Content-Type': 'application/xml; charset=utf-8'}


@app.route('/google<ver_code>.html')
def google_verification(ver_code):
    """Google Search Console verification file endpoint."""
    return f"google-site-verification: google{ver_code}.html", 200, {'Content-Type': 'text/html; charset=utf-8'}


# ========== HTML Template ==========

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>公章 STL 生成器 - 在线制作可打印3D公章模型 | StampMaker</title>
<meta name="description" content="免费在线公章STL生成器，支持自定义公司名称、注册号、五角星等参数，实时预览盖印效果，一键生成可3D打印的STL文件。适用于FDM和光固化打印机。">
<meta name="keywords" content="公章,STL生成器,3D打印,印章,电子公章,在线生成,3D模型,公章制作,印章生成器,stamp generator">
<meta name="author" content="StampMaker">
<meta name="robots" content="index, follow">
<link rel="canonical" href="https://stampmaker.ursoftware.com/">

<!-- Open Graph -->
<meta property="og:type" content="website">
<meta property="og:url" content="https://stampmaker.ursoftware.com/">
<meta property="og:title" content="公章 STL 生成器 - 在线制作可打印3D公章模型">
<meta property="og:description" content="免费在线公章STL生成器，支持自定义公司名称、注册号、五角星等参数，实时预览盖印效果，一键生成可3D打印的STL文件。">
<meta property="og:image" content="https://stampmaker.ursoftware.com/og-image.png">
<meta property="og:locale" content="zh_CN">
<meta property="og:site_name" content="StampMaker">

<!-- Twitter Card -->
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:url" content="https://stampmaker.ursoftware.com/">
<meta name="twitter:title" content="公章 STL 生成器 - 在线制作可打印3D公章模型">
<meta name="twitter:description" content="免费在线公章STL生成器，支持自定义公司名称、注册号、五角星等参数，实时预览盖印效果。">
<meta name="twitter:image" content="https://stampmaker.ursoftware.com/og-image.png">

<!-- Structured Data -->
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "WebApplication",
  "name": "公章 STL 生成器",
  "alternateName": "StampMaker",
  "url": "https://stampmaker.ursoftware.com/",
  "description": "免费在线公章STL生成器，支持自定义公司名称、注册号、五角星等参数，实时预览盖印效果，一键生成可3D打印的STL文件。",
  "applicationCategory": "DesignApplication",
  "operatingSystem": "Web",
  "offers": {
    "@type": "Offer",
    "price": "0",
    "priceCurrency": "CNY"
  },
  "featureList": "自定义公司名称,自定义注册号,五角星,实时预览,STL导出,3D打印"
}
</script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { height: 100%; overflow: hidden; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #f5f5f7;
    color: #1d1d1f;
  }
  .container {
    height: 100vh;
    display: grid;
    grid-template-columns: 1fr 360px;
  }
  .preview-panel {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 20px;
    height: 100vh;
    overflow: hidden;
  }
  .preview-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    width: 100%;
    max-width: 720px;
  }
  .preview-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
  }
  .preview-label {
    font-size: 14px;
    font-weight: 600;
    color: #424245;
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .checkbox-label {
    font-size: 12px;
    font-weight: 400;
    color: #666;
    cursor: pointer;
    user-select: none;
    display: flex;
    align-items: center;
    gap: 4px;
  }
  .checkbox-label input { cursor: pointer; }
  .preview-canvas {
    width: 100%;
    max-width: 340px;
    aspect-ratio: 1;
    border: 1px solid #d8d8dd;
    border-radius: 12px;
    background: #1a1a1a;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
  }
  .preview-canvas.impression {
    background: #f5f0e6;
    box-shadow: inset 0 2px 10px rgba(0,0,0,0.08), 0 4px 20px rgba(0,0,0,0.1);
  }
  .preview-info {
    font-size: 13px;
    color: #86868b;
    text-align: center;
    margin-top: 16px;
  }
  .controls-panel {
    background: white;
    border-left: 1px solid #e0e0e0;
    display: flex;
    flex-direction: column;
    height: 100vh;
  }
  .controls-header {
    padding: 16px 20px 12px;
    border-bottom: 1px solid #f0f0f0;
    flex-shrink: 0;
  }
  .controls-header h2 {
    font-size: 16px;
    font-weight: 600;
  }
  .controls-scroll {
    flex: 1;
    overflow-y: auto;
    padding: 4px 20px;
  }
  .section {
    border-bottom: 1px solid #f5f5f5;
  }
  .section:last-child { border-bottom: none; }
  .section-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 0;
    cursor: pointer;
    user-select: none;
  }
  .section-header h3 {
    font-size: 14px;
    font-weight: 600;
    color: #1d1d1f;
  }
  .section-arrow {
    font-size: 12px;
    color: #86868b;
    transition: transform 0.2s;
  }
  .section.collapsed .section-arrow { transform: rotate(-90deg); }
  .section-content {
    padding-bottom: 10px;
  }
  .section.collapsed .section-content { display: none; }
  .control {
    margin-bottom: 10px;
  }
  .control:last-child { margin-bottom: 0; }
  .control label {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    color: #424245;
    margin-bottom: 4px;
  }
  .control label .val {
    font-weight: 500;
    color: #0071e3;
    font-variant-numeric: tabular-nums;
  }
  .control input[type="range"] {
    width: 100%;
    height: 4px;
    border-radius: 2px;
    background: #e5e5ea;
    -webkit-appearance: none;
    appearance: none;
    cursor: pointer;
  }
  .control input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: #0071e3;
    cursor: pointer;
  }
  .control input[type="text"] {
    width: 100%;
    padding: 6px 10px;
    border: 1px solid #d2d2d7;
    border-radius: 6px;
    font-size: 13px;
    outline: none;
    transition: border-color 0.2s;
  }
  .control input[type="text"]:focus { border-color: #0071e3; }
  .controls-footer {
    padding: 12px 20px;
    border-top: 1px solid #e0e0e0;
    background: #fafafa;
    flex-shrink: 0;
  }
  .generate-btn {
    width: 100%;
    padding: 12px;
    background: #0071e3;
    color: white;
    border: none;
    border-radius: 10px;
    font-size: 15px;
    font-weight: 500;
    cursor: pointer;
    transition: background 0.2s;
  }
  .generate-btn:hover { background: #0077ed; }
  .generate-btn:disabled { background: #a8a8a8; cursor: not-allowed; }
  .btn-row {
    display: flex;
    gap: 8px;
    margin-top: 8px;
  }
  .btn-secondary {
    flex: 1;
    padding: 8px;
    background: #e8e8ed;
    color: #1d1d1f;
    border: none;
    border-radius: 8px;
    font-size: 13px;
    cursor: pointer;
    transition: background 0.2s;
  }
  .btn-secondary:hover { background: #d8d8dd; }
  .hint {
    font-size: 11px;
    color: #86868b;
    margin-top: 4px;
    line-height: 1.4;
  }
</style>
</head>
<body>
<div class="container">
  <div class="preview-panel">
    <div class="preview-row">
      <div class="preview-item">
        <div class="preview-label">印章面（3D打印面）</div>
        <canvas id="canvasStamp" class="preview-canvas"></canvas>
      </div>
      <div class="preview-item">
        <div class="preview-label">
          盖印效果
          <label class="checkbox-label"><input type="checkbox" id="paperEffect" checked> 模拟纸上效果</label>
        </div>
        <canvas id="canvasImpression" class="preview-canvas impression"></canvas>
      </div>
    </div>
    <div class="preview-info" id="previewInfo">调整右侧滑块实时预览</div>
  </div>

  <div class="controls-panel">
    <div class="controls-header">
      <h2>参数设置</h2>
    </div>
    <div class="controls-scroll">
      <div class="section">
        <div class="section-header"><h3>基本信息</h3><span class="section-arrow">▼</span></div>
        <div class="section-content">
          <div class="control">
            <label>公司名称</label>
            <input type="text" id="company_name" value="上海锦绣科技有限公司">
          </div>
          <div class="control">
            <label>注册号</label>
            <input type="text" id="reg_number" value="3201041477313">
          </div>
        </div>
      </div>

      <div class="section">
        <div class="section-header"><h3>印章尺寸</h3><span class="section-arrow">▼</span></div>
        <div class="section-content">
          <div class="control">
            <label>直径 <span class="val" id="diameter_val">40.0 mm</span></label>
            <input type="range" id="diameter" min="20" max="60" step="0.5" value="40">
          </div>
          <div class="control">
            <label>底座厚度 <span class="val" id="base_h_val">3.0 mm</span></label>
            <input type="range" id="base_h" min="1" max="8" step="0.5" value="3">
          </div>
          <div class="control">
            <label>边框宽度 <span class="val" id="ring_width_val">1.2 mm</span></label>
            <input type="range" id="ring_width" min="0.5" max="3" step="0.1" value="1.2">
          </div>
        </div>
      </div>

      <div class="section">
        <div class="section-header"><h3>公司名称（上弧）</h3><span class="section-arrow">▼</span></div>
        <div class="section-content">
          <div class="control">
            <label>字号 <span class="val" id="text_size_val">4.5 mm</span></label>
            <input type="range" id="text_size" min="2" max="7" step="0.1" value="4.5">
          </div>
          <div class="control">
            <label>文字高度 <span class="val" id="text_height_val">1.00x</span></label>
            <input type="range" id="text_height" min="0.5" max="1.5" step="0.05" value="1">
          </div>
          <div class="control">
            <label>弧形半径 <span class="val" id="text_radius_val">15.5 mm</span></label>
            <input type="range" id="text_radius" min="8" max="18" step="0.2" value="15.5">
          </div>
          <div class="control">
            <label>起始角度 <span class="val" id="text_start_val">260°</span></label>
            <input type="range" id="text_start" min="180" max="330" step="1" value="260">
          </div>
          <div class="control">
            <label>弧形跨度 <span class="val" id="text_span_val">200°</span></label>
            <input type="range" id="text_span" min="60" max="280" step="1" value="200">
          </div>
        </div>
      </div>

      <div class="section collapsed">
        <div class="section-header"><h3>注册号（下弧）</h3><span class="section-arrow">▼</span></div>
        <div class="section-content">
          <div class="control">
            <label>字号 <span class="val" id="num_size_val">2.2 mm</span></label>
            <input type="range" id="num_size" min="1" max="5" step="0.1" value="2.2">
          </div>
          <div class="control">
            <label>弧形半径 <span class="val" id="num_radius_val">16.0 mm</span></label>
            <input type="range" id="num_radius" min="8" max="18" step="0.2" value="16">
          </div>
          <div class="control">
            <label>起始角度 <span class="val" id="num_start_val">140°</span></label>
            <input type="range" id="num_start" min="90" max="180" step="1" value="140">
          </div>
          <div class="control">
            <label>弧形跨度 <span class="val" id="num_span_val">80°</span></label>
            <input type="range" id="num_span" min="30" max="150" step="1" value="80">
          </div>
        </div>
      </div>

      <div class="section collapsed">
        <div class="section-header"><h3>五角星</h3><span class="section-arrow">▼</span></div>
        <div class="section-content">
          <div class="control">
            <label>外接圆半径 <span class="val" id="star_r_val">7.0 mm</span></label>
            <input type="range" id="star_r" min="3" max="12" step="0.2" value="7">
          </div>
        </div>
      </div>

      <div class="section collapsed">
        <div class="section-header"><h3>打印优化</h3><span class="section-arrow">▼</span></div>
        <div class="section-content">
          <div class="control">
            <label>STL 精度 <span class="val" id="resolution_val">0.05 mm</span></label>
            <input type="range" id="resolution" min="0.02" max="0.1" step="0.01" value="0.05">
          </div>
          <div class="control">
            <label>凸起高度 <span class="val" id="feat_h_val">1.0 mm</span></label>
            <input type="range" id="feat_h" min="0.3" max="3" step="0.1" value="1">
          </div>
          <div class="control">
            <label>笔画加粗 <span class="val" id="stroke_thicken_val">0.40 mm</span></label>
            <input type="range" id="stroke_thicken" min="0" max="0.8" step="0.05" value="0.4">
          </div>
          <p class="hint">STL 精度越细文字越清晰，建议 0.04-0.05mm。加粗笔画防止切片缺边少角（0.04mm 喷头建议 0.4mm+）。</p>
        </div>
      </div>
    </div>
    <div class="controls-footer">
      <button class="generate-btn" id="generateBtn">生成 STL 文件</button>
      <div class="btn-row">
        <button class="btn-secondary" id="downloadImgBtn">下载盖印图片</button>
        <button class="btn-secondary" id="resetBtn">恢复默认</button>
      </div>
      <p class="hint" style="margin-top:8px;">
        提示：角度 0° 为顶部（12点方向），顺时针增加。文字已自动镜像，打印后盖印即为正向可读。
      </p>
    </div>
  </div>
</div>

<script>
const canvasStamp = document.getElementById('canvasStamp');
const ctxStamp = canvasStamp.getContext('2d');
const canvasImpression = document.getElementById('canvasImpression');
const ctxImpression = canvasImpression.getContext('2d');
const previewInfo = document.getElementById('previewInfo');
const generateBtn = document.getElementById('generateBtn');
const resetBtn = document.getElementById('resetBtn');
const downloadImgBtn = document.getElementById('downloadImgBtn');

let debounceTimer = null;
let isGenerating = false;

const defaultParams = {
  company_name: '上海锦绣科技有限公司',
  reg_number: '3201041477313',
  diameter: 40,
  base_h: 3,
  feat_h: 1,
  text_height: 1,
  resolution: 0.05,
  ring_width: 1.2,
  text_size: 4.5,
  text_radius: 15.5,
  text_start: 260,
  text_span: 200,
  num_size: 2.2,
  num_radius: 16,
  num_start: 140,
  num_span: 80,
  star_r: 7,
  stroke_thicken: 0.4
};

// Section collapse toggle
document.querySelectorAll('.section-header').forEach(header => {
  header.addEventListener('click', () => {
    header.parentElement.classList.toggle('collapsed');
  });
});

function getParams() {
  return {
    company_name: document.getElementById('company_name').value,
    reg_number: document.getElementById('reg_number').value,
    diameter: parseFloat(document.getElementById('diameter').value),
    base_h: parseFloat(document.getElementById('base_h').value),
    feat_h: parseFloat(document.getElementById('feat_h').value),
    text_height: parseFloat(document.getElementById('text_height').value),
    ring_width: parseFloat(document.getElementById('ring_width').value),
    text_size: parseFloat(document.getElementById('text_size').value),
    text_radius: parseFloat(document.getElementById('text_radius').value),
    text_start: parseFloat(document.getElementById('text_start').value),
    text_span: parseFloat(document.getElementById('text_span').value),
    num_size: parseFloat(document.getElementById('num_size').value),
    num_radius: parseFloat(document.getElementById('num_radius').value),
    num_start: parseFloat(document.getElementById('num_start').value),
    num_span: parseFloat(document.getElementById('num_span').value),
    star_r: parseFloat(document.getElementById('star_r').value),
    stroke_thicken: parseFloat(document.getElementById('stroke_thicken').value),
    resolution: parseFloat(document.getElementById('resolution').value),
  };
}

function updateLabels() {
  const params = getParams();
  document.getElementById('diameter_val').textContent = params.diameter.toFixed(1) + ' mm';
  document.getElementById('base_h_val').textContent = params.base_h.toFixed(1) + ' mm';
  document.getElementById('feat_h_val').textContent = params.feat_h.toFixed(1) + ' mm';
  document.getElementById('resolution_val').textContent = params.resolution.toFixed(2) + ' mm';
  document.getElementById('text_height_val').textContent = params.text_height.toFixed(2) + 'x';
  document.getElementById('ring_width_val').textContent = params.ring_width.toFixed(1) + ' mm';
  document.getElementById('text_size_val').textContent = params.text_size.toFixed(1) + ' mm';
  document.getElementById('text_radius_val').textContent = params.text_radius.toFixed(1) + ' mm';
  document.getElementById('text_start_val').textContent = params.text_start + '°';
  document.getElementById('text_span_val').textContent = params.text_span + '°';
  document.getElementById('num_size_val').textContent = params.num_size.toFixed(1) + ' mm';
  document.getElementById('num_radius_val').textContent = params.num_radius.toFixed(1) + ' mm';
  document.getElementById('num_start_val').textContent = params.num_start + '°';
  document.getElementById('num_span_val').textContent = params.num_span + '°';
  document.getElementById('star_r_val').textContent = params.star_r.toFixed(1) + ' mm';
  document.getElementById('stroke_thicken_val').textContent = params.stroke_thicken.toFixed(2) + ' mm';
}

function drawImpressionEffect(sourceImg, width, height) {
  const off = document.createElement('canvas');
  off.width = width;
  off.height = height;
  const octx = off.getContext('2d');

  const paperGrad = octx.createRadialGradient(
    width/2, height/2, width*0.1,
    width/2, height/2, width*0.7
  );
  paperGrad.addColorStop(0, '#faf6ed');
  paperGrad.addColorStop(1, '#f0e8d8');
  octx.fillStyle = paperGrad;
  octx.fillRect(0, 0, width, height);

  octx.globalCompositeOperation = 'source-over';
  octx.drawImage(sourceImg, 0, 0);

  const imgData = octx.getImageData(0, 0, width, height);
  const data = imgData.data;

  const stampRed = { r: 210, g: 25, b: 25 };
  const stampRed2 = { r: 230, g: 50, b: 50 };

  for (let i = 0; i < data.length; i += 4) {
    const mask = data[i] / 255;
    if (mask > 0.01) {
      const noise = Math.random();
      const opacity = mask * (0.75 + noise * 0.25);
      const colorMix = Math.random();
      const r = stampRed.r + (stampRed2.r - stampRed.r) * colorMix;
      const g = stampRed.g + (stampRed2.g - stampRed.g) * colorMix;
      const b = stampRed.b + (stampRed2.b - stampRed.b) * colorMix;
      const gap = Math.random() < 0.03 ? 0.3 : 1;
      data[i] = r * opacity * gap;
      data[i+1] = g * opacity * gap;
      data[i+2] = b * opacity * gap;
      data[i+3] = 255;
    } else {
      const fiber = (Math.random() - 0.5) * 12;
      data[i] = Math.min(255, Math.max(0, 250 + fiber));
      data[i+1] = Math.min(255, Math.max(0, 242 + fiber));
      data[i+2] = Math.min(255, Math.max(0, 228 + fiber));
      data[i+3] = 255;
    }
  }

  octx.putImageData(imgData, 0, 0);

  const blurCanvas = document.createElement('canvas');
  blurCanvas.width = width;
  blurCanvas.height = height;
  const bctx = blurCanvas.getContext('2d');
  bctx.filter = 'blur(0.5px)';
  bctx.drawImage(off, 0, 0);

  return blurCanvas;
}

function drawElectronicEffect(sourceImg, width, height) {
  const off = document.createElement('canvas');
  off.width = width;
  off.height = height;
  const octx = off.getContext('2d');

  octx.fillStyle = '#ffffff';
  octx.fillRect(0, 0, width, height);
  octx.drawImage(sourceImg, 0, 0);

  const imgData = octx.getImageData(0, 0, width, height);
  const data = imgData.data;
  const r = 210, g = 25, b = 25;

  for (let i = 0; i < data.length; i += 4) {
    const mask = data[i] / 255;
    if (mask > 0.5) {
      data[i] = r;
      data[i+1] = g;
      data[i+2] = b;
      data[i+3] = 255;
    } else {
      data[i] = 255;
      data[i+1] = 255;
      data[i+2] = 255;
      data[i+3] = 255;
    }
  }

  octx.putImageData(imgData, 0, 0);
  return off;
}

async function updatePreview() {
  const params = getParams();
  try {
    const resp = await fetch('/api/preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params)
    });
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () => {
      const w = img.width;
      const h = img.height;

      canvasStamp.width = w;
      canvasStamp.height = h;
      ctxStamp.drawImage(img, 0, 0);

      canvasImpression.width = w;
      canvasImpression.height = h;

      const flip = document.createElement('canvas');
      flip.width = w;
      flip.height = h;
      const fctx = flip.getContext('2d');
      fctx.translate(w, 0);
      fctx.scale(-1, 1);
      fctx.drawImage(img, 0, 0);

      const usePaper = document.getElementById('paperEffect').checked;
      const effectCanvas = usePaper
        ? drawImpressionEffect(flip, w, h)
        : drawElectronicEffect(flip, w, h);
      ctxImpression.drawImage(effectCanvas, 0, 0);

      URL.revokeObjectURL(url);
    };
    img.src = url;
    previewInfo.textContent = `直径 ${params.diameter.toFixed(1)}mm · 总高 ${(params.base_h + params.feat_h).toFixed(1)}mm`;
  } catch (e) {
    previewInfo.textContent = '预览加载失败';
  }
}

function scheduleUpdate() {
  updateLabels();
  saveParams();
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(updatePreview, 80);
}

function saveParams() {
  const params = getParams();
  params.paperEffect = document.getElementById('paperEffect').checked;
  try { localStorage.setItem('stamp_params', JSON.stringify(params)); } catch(e) {}
}

function restoreParams() {
  try {
    const saved = localStorage.getItem('stamp_params');
    if (!saved) return;
    const params = JSON.parse(saved);
    for (const key in params) {
      if (key === 'paperEffect') {
        document.getElementById('paperEffect').checked = params[key];
      } else {
        const el = document.getElementById(key);
        if (el) el.value = params[key];
      }
    }
  } catch(e) {}
}

document.querySelectorAll('input[type="range"], input[type="text"]').forEach(el => {
  el.addEventListener('input', scheduleUpdate);
});

document.getElementById('paperEffect').addEventListener('change', scheduleUpdate);

generateBtn.addEventListener('click', async () => {
  if (isGenerating) return;
  isGenerating = true;
  generateBtn.disabled = true;
  generateBtn.textContent = '生成中...';

  const params = getParams();
  try {
    const resp = await fetch('/api/generate-stl', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params)
    });
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${params.company_name}_公章.stl`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  } catch (e) {
    alert('生成失败：' + e.message);
  } finally {
    isGenerating = false;
    generateBtn.disabled = false;
    generateBtn.textContent = '生成 STL 文件';
  }
});

downloadImgBtn.addEventListener('click', () => {
  const link = document.createElement('a');
  const ts = new Date().toISOString().slice(0, 10);
  link.download = `盖章效果_${ts}.png`;
  link.href = canvasImpression.toDataURL('image/png');
  link.click();
});

resetBtn.addEventListener('click', () => {
  for (const key in defaultParams) {
    const el = document.getElementById(key);
    if (el) {
      el.value = defaultParams[key];
    }
  }
  document.getElementById('paperEffect').checked = true;
  try { localStorage.removeItem('stamp_params'); } catch(e) {}
  scheduleUpdate();
});

restoreParams();
updateLabels();
updatePreview();
</script>
</body>
</html>
"""


def main():
    port = int(os.environ.get('PORT', 5000))
    print("=" * 50)
    print("公章 STL 生成器 - Web 版")
    print("=" * 50)
    print(f"Font: {FONT_PATH}")
    print(f"Starting server at http://0.0.0.0:{port}")
    print("Press Ctrl+C to stop")
    print()
    app.run(host='0.0.0.0', port=port, debug=False)


if __name__ == '__main__':
    main()
