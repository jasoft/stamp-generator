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
                  arc_start, arc_span, clockwise=True, inward=False):
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

        rot = (180.0 - angle) if inward else (-angle)
        rotated = ci.rotate(rot, expand=True, resample=Image.BICUBIC)
        rx, ry = rotated.size
        img.paste(255, (int(ax - rx / 2), int(ay - ry / 2)), rotated)


def render_stamp(params, res_mm=0.1):
    """Render 2D stamp image from parameters.
    Returns: PIL Image (grayscale, white features on black background)
    """
    diameter = params.get('diameter', 40.0)
    ring_width = params.get('ring_width', 1.2)
    company = params.get('company_name', '上海逗号软件科技有限公司')
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

    # Company name (top arc, outward)
    font_px = max(8, int(text_size / res_mm))
    font = ImageFont.truetype(FONT_PATH, font_px)
    draw_arc_text(img, cx, cy, company, font, font_px * 3,
                  text_radius / res_mm, text_start, text_span,
                  clockwise=True, inward=False)

    # Registration number (bottom arc, inward)
    npx = max(6, int(num_size / res_mm))
    nfont = ImageFont.truetype(FONT_PATH, npx)
    draw_arc_text(img, cx, cy, reg_num, nfont, npx * 3,
                  num_radius / res_mm, num_start, num_span,
                  clockwise=True, inward=True)

    # Dilate features to ensure minimum stroke width for 3D printing.
    # FDM printers typically need features >= 0.4mm (nozzle diameter).
    # Each MaxFilter(3) pass expands white pixels by ~1 pixel.
    dilate_px = max(0, int(round(stroke_thicken / res_mm)))
    for _ in range(dilate_px):
        img = img.filter(ImageFilter.MaxFilter(3))

    # Flip for stamp face
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def build_mesh(arr_2d, diameter, base_h, feat_h, res_mm=0.1):
    """Build a watertight mesh from 2D feature mask."""
    n = arr_2d.shape[0]
    radius = diameter / 2.0
    cx = cy = n / 2

    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    circle = dist <= radius / res_mm

    hmap = np.zeros((n, n), dtype=np.float32)
    hmap[circle] = base_h
    hmap[arr_2d & circle] = base_h + feat_h

    # Vertices
    j_grid, i_grid = np.meshgrid(np.arange(n), np.arange(n))
    x = (j_grid - n / 2) * res_mm
    y = (n / 2 - i_grid) * res_mm

    top_verts = np.stack([x.ravel(), y.ravel(), hmap.ravel()], axis=1).astype(np.float32)
    bot_verts = np.stack([x.ravel(), y.ravel(), np.zeros(n * n)], axis=1).astype(np.float32)
    all_verts = np.vstack([top_verts, bot_verts])
    NN = n * n

    # Cell mask
    h00 = hmap[:-1, :-1]
    h01 = hmap[:-1, 1:]
    h10 = hmap[1:, :-1]
    h11 = hmap[1:, 1:]
    cell = (h00 > 0) & (h01 > 0) & (h10 > 0) & (h11 > 0)

    vi00 = np.arange(n * n).reshape(n, n)[:-1, :-1]
    vi01 = np.arange(n * n).reshape(n, n)[:-1, 1:]
    vi10 = np.arange(n * n).reshape(n, n)[1:, :-1]
    vi11 = np.arange(n * n).reshape(n, n)[1:, 1:]

    m = cell.ravel()
    v00 = vi00.ravel()[m]
    v01 = vi01.ravel()[m]
    v10 = vi10.ravel()[m]
    v11 = vi11.ravel()[m]

    # Top faces
    top_f = np.empty((len(v00) * 2, 3), dtype=np.int64)
    top_f[0::2] = np.stack([v00, v10, v11], axis=1)
    top_f[1::2] = np.stack([v00, v11, v01], axis=1)

    # Bottom faces
    b00, b01, b10, b11 = v00 + NN, v01 + NN, v10 + NN, v11 + NN
    bot_f = np.empty((len(v00) * 2, 3), dtype=np.int64)
    bot_f[0::2] = np.stack([b00, b11, b10], axis=1)
    bot_f[1::2] = np.stack([b00, b01, b11], axis=1)

    # Side walls
    wall_faces = []

    def add_wall(vt1, vt2, vb1, vb2, reverse=False):
        if reverse:
            wall_faces.append(np.array([[vt1, vb2, vt2], [vt1, vb1, vb2]], dtype=np.int64))
        else:
            wall_faces.append(np.array([[vt1, vt2, vb2], [vt1, vb2, vb1]], dtype=np.int64))

    # Left wall
    left = np.zeros_like(cell)
    left[:, 1:] = cell[:, 1:] & ~cell[:, :-1]
    left[:, 0] = cell[:, 0]
    li, lj = np.where(left)
    for k in range(len(li)):
        i, j = li[k], lj[k]
        vt1 = i * n + j
        vt2 = (i + 1) * n + j
        add_wall(vt1, vt2, vt1 + NN, vt2 + NN, reverse=True)

    # Right wall
    right = np.zeros_like(cell)
    right[:, :-1] = cell[:, :-1] & ~cell[:, 1:]
    right[:, -1] = cell[:, -1]
    ri, rj = np.where(right)
    for k in range(len(ri)):
        i, j = ri[k], rj[k]
        vt1 = i * n + (j + 1)
        vt2 = (i + 1) * n + (j + 1)
        add_wall(vt1, vt2, vt1 + NN, vt2 + NN, reverse=False)

    # Top wall
    topw = np.zeros_like(cell)
    topw[1:, :] = cell[1:, :] & ~cell[:-1, :]
    topw[0, :] = cell[0, :]
    ti, tj = np.where(topw)
    for k in range(len(ti)):
        i, j = ti[k], tj[k]
        vt1 = i * n + j
        vt2 = i * n + (j + 1)
        add_wall(vt1, vt2, vt1 + NN, vt2 + NN, reverse=False)

    # Bottom wall
    btm = np.zeros_like(cell)
    btm[:-1, :] = cell[:-1, :] & ~cell[1:, :]
    btm[-1, :] = cell[-1, :]
    bi, bj = np.where(btm)
    for k in range(len(bi)):
        i, j = bi[k], bj[k]
        vt1 = (i + 1) * n + j
        vt2 = (i + 1) * n + (j + 1)
        add_wall(vt1, vt2, vt1 + NN, vt2 + NN, reverse=True)

    wall_f = np.vstack(wall_faces) if wall_faces else np.zeros((0, 3), dtype=np.int64)
    all_faces = np.vstack([top_f, bot_f, wall_f])
    return all_verts, all_faces


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
    # Convert to PNG
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
    res_mm = params.get('resolution', 0.1)

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


# ========== HTML Template ==========

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>公章 STL 生成器</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #f5f5f7;
    color: #1d1d1f;
    padding: 20px;
  }
  .container {
    max-width: 1200px;
    margin: 0 auto;
    display: grid;
    grid-template-columns: 1fr 400px;
    gap: 24px;
  }
  h1 {
    font-size: 24px;
    margin-bottom: 20px;
    grid-column: 1 / -1;
  }
  .preview-panel {
    background: white;
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 16px;
    min-height: 500px;
  }
  .preview-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    width: 100%;
  }
  .preview-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
  }
  .preview-label {
    font-size: 13px;
    font-weight: 500;
    color: #424245;
  }
  .preview-canvas {
    width: 100%;
    max-width: 280px;
    aspect-ratio: 1;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    background: #fafafa;
  }
  .preview-canvas.stamp {
    background: #1a1a1a;
  }
  .preview-canvas.impression {
    background: #f5f0e6;
    box-shadow: inset 0 2px 10px rgba(0,0,0,0.08);
  }
  .preview-info {
    font-size: 13px;
    color: #86868b;
    text-align: center;
  }
  .controls-panel {
    background: white;
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    max-height: calc(100vh - 80px);
    overflow-y: auto;
  }
  .section {
    margin-bottom: 24px;
    padding-bottom: 20px;
    border-bottom: 1px solid #f0f0f0;
  }
  .section:last-child { border-bottom: none; }
  .section h3 {
    font-size: 15px;
    font-weight: 600;
    margin-bottom: 14px;
    color: #1d1d1f;
  }
  .control {
    margin-bottom: 14px;
  }
  .control label {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    color: #424245;
    margin-bottom: 6px;
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
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: #0071e3;
    cursor: pointer;
    box-shadow: 0 1px 3px rgba(0,0,0,0.2);
  }
  .control input[type="text"] {
    width: 100%;
    padding: 8px 12px;
    border: 1px solid #d2d2d7;
    border-radius: 6px;
    font-size: 13px;
    outline: none;
    transition: border-color 0.2s;
  }
  .control input[type="text"]:focus {
    border-color: #0071e3;
  }
  .generate-btn {
    width: 100%;
    padding: 14px;
    background: #0071e3;
    color: white;
    border: none;
    border-radius: 10px;
    font-size: 16px;
    font-weight: 500;
    cursor: pointer;
    transition: background 0.2s;
  }
  .generate-btn:hover { background: #0077ed; }
  .generate-btn:disabled {
    background: #a8a8a8;
    cursor: not-allowed;
  }
  .btn-row {
    display: flex;
    gap: 8px;
    margin-top: 16px;
  }
  .btn-secondary {
    flex: 1;
    padding: 10px;
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
    font-size: 12px;
    color: #86868b;
    margin-top: 4px;
    line-height: 1.4;
  }
</style>
</head>
<body>
<div class="container">
  <h1>公章 STL 生成器</h1>

  <div class="preview-panel">
    <div class="preview-row">
      <div class="preview-item">
        <div class="preview-label">印章面（3D打印面）</div>
        <canvas id="canvasStamp" class="preview-canvas stamp"></canvas>
      </div>
      <div class="preview-item">
        <div class="preview-label">盖印效果（纸上效果）</div>
        <canvas id="canvasImpression" class="preview-canvas impression"></canvas>
      </div>
    </div>
    <div class="preview-info" id="previewInfo">调整右侧滑块实时预览</div>
  </div>

  <div class="controls-panel">
    <div class="section">
      <h3>基本信息</h3>
      <div class="control">
        <label>公司名称</label>
        <input type="text" id="company_name" value="上海逗号软件科技有限公司">
      </div>
      <div class="control">
        <label>注册号</label>
        <input type="text" id="reg_number" value="3201041477313">
      </div>
    </div>

    <div class="section">
      <h3>印章尺寸</h3>
      <div class="control">
        <label>直径 <span class="val" id="diameter_val">40.0 mm</span></label>
        <input type="range" id="diameter" min="20" max="60" step="0.5" value="40">
      </div>
      <div class="control">
        <label>底座厚度 <span class="val" id="base_h_val">3.0 mm</span></label>
        <input type="range" id="base_h" min="1" max="8" step="0.5" value="3">
      </div>
      <div class="control">
        <label>凸起高度 <span class="val" id="feat_h_val">1.0 mm</span></label>
        <input type="range" id="feat_h" min="0.3" max="3" step="0.1" value="1">
      </div>
      <div class="control">
        <label>边框宽度 <span class="val" id="ring_width_val">1.2 mm</span></label>
        <input type="range" id="ring_width" min="0.5" max="3" step="0.1" value="1.2">
      </div>
    </div>

    <div class="section">
      <h3>公司名称（上弧）</h3>
      <div class="control">
        <label>字号 <span class="val" id="text_size_val">4.5 mm</span></label>
        <input type="range" id="text_size" min="2" max="7" step="0.1" value="4.5">
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

    <div class="section">
      <h3>注册号（下弧）</h3>
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

    <div class="section">
      <h3>五角星</h3>
      <div class="control">
        <label>外接圆半径 <span class="val" id="star_r_val">7.0 mm</span></label>
        <input type="range" id="star_r" min="3" max="12" step="0.2" value="7">
      </div>
    </div>

    <div class="section">
      <h3>打印优化</h3>
      <div class="control">
        <label>笔画加粗 <span class="val" id="stroke_thicken_val">0.25 mm</span></label>
        <input type="range" id="stroke_thicken" min="0" max="0.6" step="0.05" value="0.25">
      </div>
      <p class="hint">加粗笔画可防止切片时文字缺边少角。建议 0.2-0.3mm（适配 0.4mm 喷嘴）。</p>
    </div>

    <button class="generate-btn" id="generateBtn">生成 STL 文件</button>
    <div class="btn-row">
      <button class="btn-secondary" id="resetBtn">恢复默认</button>
    </div>
    <p class="hint" style="margin-top:12px;">
      提示：角度 0° 为顶部（12点方向），顺时针增加。文字已自动镜像，打印后盖印即为正向可读。
    </p>
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

let debounceTimer = null;
let isGenerating = false;
let noiseData = null;  // cached noise pattern

const defaultParams = {
  company_name: '上海逗号软件科技有限公司',
  reg_number: '3201041477313',
  diameter: 40,
  base_h: 3,
  feat_h: 1,
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
  stroke_thicken: 0.25
};

function getParams() {
  return {
    company_name: document.getElementById('company_name').value,
    reg_number: document.getElementById('reg_number').value,
    diameter: parseFloat(document.getElementById('diameter').value),
    base_h: parseFloat(document.getElementById('base_h').value),
    feat_h: parseFloat(document.getElementById('feat_h').value),
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
  };
}

function updateLabels() {
  const params = getParams();
  document.getElementById('diameter_val').textContent = params.diameter.toFixed(1) + ' mm';
  document.getElementById('base_h_val').textContent = params.base_h.toFixed(1) + ' mm';
  document.getElementById('feat_h_val').textContent = params.feat_h.toFixed(1) + ' mm';
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
  // Create offscreen canvas for processing
  const off = document.createElement('canvas');
  off.width = width;
  off.height = height;
  const octx = off.getContext('2d');

  // 1. Draw paper background (warm off-white with subtle texture)
  const paperGrad = octx.createRadialGradient(
    width/2, height/2, width*0.1,
    width/2, height/2, width*0.7
  );
  paperGrad.addColorStop(0, '#faf6ed');
  paperGrad.addColorStop(1, '#f0e8d8');
  octx.fillStyle = paperGrad;
  octx.fillRect(0, 0, width, height);

  // 2. Draw source image as alpha mask for red stamp
  // First draw source to get alpha
  octx.globalCompositeOperation = 'source-over';
  octx.drawImage(sourceImg, 0, 0);

  // Get image data to process
  const imgData = octx.getImageData(0, 0, width, height);
  const data = imgData.data;

  // 3. Generate noise-based stamp effect
  const stampRed = { r: 210, g: 25, b: 25 };  // deep stamp red
  const stampRed2 = { r: 230, g: 50, b: 50 };  // lighter red

  for (let i = 0; i < data.length; i += 4) {
    const mask = data[i] / 255;  // white = stamp area (mask=1)
    if (mask > 0.01) {
      // Random noise for uneven ink distribution
      const noise = Math.random();
      // Base opacity varies
      const opacity = mask * (0.75 + noise * 0.25);
      // Color variation
      const colorMix = Math.random();
      const r = stampRed.r + (stampRed2.r - stampRed.r) * colorMix;
      const g = stampRed.g + (stampRed2.g - stampRed.g) * colorMix;
      const b = stampRed.b + (stampRed2.b - stampRed.b) * colorMix;
      // Some spots are missing (ink gap)
      const gap = Math.random() < 0.03 ? 0.3 : 1;
      data[i] = r * opacity * gap;
      data[i+1] = g * opacity * gap;
      data[i+2] = b * opacity * gap;
      data[i+3] = 255;
    } else {
      // Paper area - add subtle fiber noise
      const fiber = (Math.random() - 0.5) * 12;
      data[i] = Math.min(255, Math.max(0, 250 + fiber));
      data[i+1] = Math.min(255, Math.max(0, 242 + fiber));
      data[i+2] = Math.min(255, Math.max(0, 228 + fiber));
      data[i+3] = 255;
    }
  }

  octx.putImageData(imgData, 0, 0);

  // 4. Slight blur for that "stamped" softness
  const blurCanvas = document.createElement('canvas');
  blurCanvas.width = width;
  blurCanvas.height = height;
  const bctx = blurCanvas.getContext('2d');
  bctx.filter = 'blur(0.5px)';
  bctx.drawImage(off, 0, 0);

  return blurCanvas;
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

      // 1. Stamp face preview (mirrored, white on dark)
      canvasStamp.width = w;
      canvasStamp.height = h;
      ctxStamp.drawImage(img, 0, 0);

      // 2. Impression effect (re-mirrored to normal, red on paper)
      canvasImpression.width = w;
      canvasImpression.height = h;

      // Flip back to normal (undo mirror) for impression view
      const flip = document.createElement('canvas');
      flip.width = w;
      flip.height = h;
      const fctx = flip.getContext('2d');
      fctx.translate(w, 0);
      fctx.scale(-1, 1);
      fctx.drawImage(img, 0, 0);

      const effectCanvas = drawImpressionEffect(flip, w, h);
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
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(updatePreview, 80);
}

// Bind all sliders and text inputs
document.querySelectorAll('input[type="range"], input[type="text"]').forEach(el => {
  el.addEventListener('input', scheduleUpdate);
});

// Generate STL
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

// Reset
resetBtn.addEventListener('click', () => {
  for (const key in defaultParams) {
    const el = document.getElementById(key);
    if (el) {
      el.value = defaultParams[key];
    }
  }
  scheduleUpdate();
});

// Initial render
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
