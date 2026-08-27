# 公章 STL 生成器 (StampMaker)

在线生成可 3D 打印的中国公章 STL 模型。支持自定义公司名称、注册号、五角星等参数，实时预览盖印效果，一键导出 STL 文件。

## 项目简介

这是一个基于 Flask 的 Web 应用，用户可以通过浏览器调整公章的各项参数，实时预览印章面和盖印效果，最终生成可下载的 STL 3D 模型文件用于 3D 打印。

- **在线地址**：https://stampmaker.ursoftware.com
- **Vercel 项目名**：stampmaker
- **GitHub 仓库**：jasoft/stamp-generator (private)
- **域名 DNS**：Cloudflare（ursoftware.com 主域下的 stampmaker 子域名）

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端 | Python 3 + Flask |
| 图像处理 | Pillow (PIL) + NumPy |
| 前端 | 原生 HTML/CSS/JS（单文件内嵌） |
| 3D 模型 | 自研贪心网格合并算法（Greedy Meshing） |
| 部署 | Vercel Serverless Functions |
| DNS / CDN | Cloudflare |

## 项目结构

```
stamp-generator/
├── stamp_webapp.py       # 主应用文件（Flask + 前端模板 + STL生成逻辑）
├── api/
│   ├── index.py          # Vercel Serverless 入口（re-export Flask app）
│   └── simsun.ttc        # Vercel 部署用的字体文件
├── simsun.ttc            # 本地开发用字体（微软宋体）
├── vercel.json           # Vercel 部署配置
├── requirements.txt      # Python 依赖
├── Dockerfile            # Docker 部署配置（备用）
├── render.yaml           # Render.com 部署配置（备用）
├── generate_stamp.py     # 早期版本脚本（已废弃，保留参考）
├── generate_stamp_v2.py  # 早期版本脚本（已废弃，保留参考）
├── fix_mesh.py           # 调试用脚本
├── check_simplify.py     # 调试用脚本
└── test_arc.py           # 调试用脚本
```

**核心文件只有一个**：`stamp_webapp.py` — 包含所有后端逻辑、前端模板和 STL 生成算法。

## 核心功能

### 1. 参数化公章生成

支持调整的参数分为 6 组：

- **基本信息**：公司名称、注册号
- **印章尺寸**：直径、底座厚度、边框宽度
- **公司名称（上弧）**：字号、文字高度（纵向缩放）、弧形半径、起始角度、弧形跨度
- **注册号（下弧）**：字号、弧形半径、起始角度、弧形跨度
- **五角星**：外接圆半径
- **打印优化**：STL 精度、凸起高度、笔画加粗

所有设置自动保存到浏览器 `localStorage`，下次访问自动恢复。

### 2. 实时预览

左侧同时显示两个预览：

- **印章面**：白色图案在深色背景上，显示的是镜像后的 3D 打印面
- **盖印效果**：红色印章在纸上的效果，支持两种模式：
  - 模拟纸上效果：带纸张纹理、墨迹不均匀、轻微模糊，模拟真实盖章效果
  - 电子效果：纯红纯白，硬边缘

### 3. STL 生成

点击"生成 STL 文件"按钮，后端会：

1. 用指定精度（默认 0.05mm）渲染 2D 印章图案
2. 通过贪心网格合并算法构建 3D 网格
3. 生成二进制 STL 文件并触发下载

### 4. 下载盖印图片

可将当前盖印效果保存为 PNG 图片。

## 核心算法

### 弧形文字排列 (`draw_arc_text`)

逐字符绘制到独立画布，按角度旋转后贴到圆弧上。支持：
- 正向/反向（公司名称向外，注册号向内）
- 文字纵向缩放（`v_scale` 参数，0.5x - 1.5x）
- 起始角度和弧形跨度可自由调整

角度约定：0° 为顶部（12 点方向），顺时针增加。

### 贪心网格合并 (`_greedy_mesh`)

将 2D 二值图像中的连续 True 像素合并成尽可能大的矩形，大幅减少面数。

对比效果：40mm 直径、0.05mm 精度的印章，从 25MB 降到 1.36MB。

### 3D 网格构建 (`build_mesh`)

从高度图（height map）构建水密网格，包含三部分：

1. **顶面**：两层高度（底座顶面、文字/五角星凸起顶面），分别做贪心网格
2. **底面**：整个印章区域的底面
3. **墙面**：高度变化处的垂直墙，分为水平边界墙和垂直边界墙

**法线方向**是关键：水平墙和垂直墙的面朝向不同，必须正确设置 `reverse` 参数，否则切片软件会识别为空层。
- 水平墙：`reverse=(ha > hb)`
- 垂直墙：`reverse=(ha < hb)`

### 笔画加粗

用 `ImageFilter.MaxFilter(3)`（形态学膨胀）对 2D 图案做 N 次膨胀，模拟笔画加粗。

**为什么需要**：FDM 打印机喷嘴有直径（常见 0.4mm），如果笔画比喷嘴细，切片器会跳过，导致文字缺边少角。加粗到最细处 ≥ 喷嘴直径即可解决。

默认加粗 0.4mm（适用于 0.4mm 喷嘴）。

## STL 生成精度与打印建议

| 参数 | 默认值 | 推荐范围 | 说明 |
|------|--------|----------|------|
| 直径 | 40mm | 30-50mm | 标准公章 42mm |
| STL 精度 | 0.05mm | 0.04-0.05mm | 越精细文字越清晰，但文件越大 |
| 凸起高度 | 1.0mm | 0.5-1.5mm | 文字和五角星的凸起厚度 |
| 底座厚度 | 3.0mm | 2-5mm | 印章整体厚度 |
| 笔画加粗 | 0.4mm | 0.3-0.5mm | 根据喷嘴直径调整 |

**切片设置建议**：
- 墙线数 = 1（印章是实心的，不需要多圈墙线）
- 填充密度 = 100%
- 层高 ≤ 0.1mm（精细文字需要薄层）

## 部署

### Vercel（当前生产环境）

部署方式：Vercel Serverless Functions + GitHub 自动部署

配置文件：`vercel.json`

```json
{
  "version": 2,
  "builds": [{ "src": "api/index.py", "use": "@vercel/python" }],
  "routes": [{ "src": "/(.*)", "dest": "/api/index.py" }]
}
```

入口文件：`api/index.py` 将父目录加入 sys.path 后导入 `stamp_webapp.app`。

**字体注意**：Vercel 环境没有中文字体，需要把 `simsun.ttc` 放在 `api/` 目录下，字体查找路径会优先找这个位置。

### 本地运行

```bash
pip install -r requirements.txt
python stamp_webapp.py
# 访问 http://localhost:5000
```

### Docker

```bash
docker build -t stampmaker .
docker run -p 5000:5000 stampmaker
```

### 自定义域名

当前绑定：`stampmaker.ursoftware.com`

DNS 配置（Cloudflare）：
- 类型：CNAME
- 名称：`stampmaker`
- 内容：`cname.vercel-dns.com`
- 代理状态：已开启（橙色云）

## API 接口

### `POST /api/preview`

生成预览图（PNG），精度 0.1mm。

请求体：JSON 格式的参数对象
响应：`image/png`

### `POST /api/generate-stl`

生成 STL 文件，精度由 `resolution` 参数控制。

请求体：JSON 格式的参数对象
响应：`application/octet-stream`，触发文件下载

### `GET /robots.txt`

搜索引擎爬虫规则。

### `GET /sitemap.xml`

站点地图。

### `GET /google<ver_code>.html`

Google Search Console 验证文件路由，动态匹配任意验证码。

## 默认参数

```python
{
  company_name: '上海锦绣科技有限公司',   // 虚拟公司名
  reg_number: '3201041477313',
  diameter: 40,         // mm
  base_h: 3,            // mm
  feat_h: 1,            // mm
  text_height: 1,       // x (纵向缩放比例)
  resolution: 0.05,     // mm (STL精度)
  ring_width: 1.2,      // mm
  text_size: 4.5,       // mm
  text_radius: 15.5,    // mm
  text_start: 260,      // °
  text_span: 200,       // °
  num_size: 2.2,        // mm
  num_radius: 16,       // mm
  num_start: 140,       // °
  num_span: 80,         // °
  star_r: 7,            // mm
  stroke_thicken: 0.4   // mm
}
```

## 历史问题与解决方案

以下是开发过程中遇到的关键问题及修复方案，供后续维护参考。

### 1. 文字与防伪码重叠

**现象**：公司名称和注册号在圆弧上重叠。
**修复**：调整文字弧形范围，上半圆 280°→80°（160° 跨度）放公司名称，下半圆 125°→235°（110° 跨度）放注册号，两侧留间隙。

### 2. 切片打印文字缺边少角

**现象**：STL 模型看起来正常，但 FDM 打印出来的文字笔画残缺。
**原因**：喷嘴直径 0.4mm，比笔画细的地方切片器直接跳过。
**修复**：新增笔画加粗功能（形态学膨胀），默认加粗 0.4mm。

### 3. 无效果模式文字边缘有黑边

**现象**：电子效果模式下，印章边缘有一圈暗色像素。
**原因**：边缘灰度像素被线性映射成暗红色。
**修复**：改用硬阈值（>0.5 为纯红，≤0.5 为纯白）。

### 4. 切片出现空层无法打印

**现象**：切片软件报空层错误。
**原因**：墙面法线方向不一致，部分朝外部分朝内，切片器无法正确识别实心区域。
**修复**：区分水平墙和垂直墙，分别设置正确的 `reverse` 参数确保所有法线朝外。

### 5. 文字填充不足

**现象**：很多字中间是空的，只有轮廓。
**原因**：笔画太细 + 切片器墙线数设置不当。
**修复**：加大默认笔画加粗到 0.4mm，建议切片器设置墙线数=1、填充密度=100%。

### 6. STL 文件过大

**现象**：0.1mm 精度下生成 25MB 的 STL。
**修复**：改用贪心网格合并算法，精度提升到 0.05mm 的同时，文件大小降到 1.36MB。

## 维护注意事项

1. **字体文件**：`simsun.ttc` 是微软宋体，需要同时放在项目根目录（本地用）和 `api/` 目录（Vercel 用）。如果替换字体，两个位置都要更新。

2. **Vercel 部署**：Git push 到 main 分支后 Vercel 会自动构建。如果自动部署未触发，可在项目目录运行 `npx vercel --prod` 手动部署。

3. **Google 验证**：项目内置了动态 Google 验证文件路由 `/google<ver_code>.html`，更换 Google Search Console 账号时无需改代码。

4. **localStorage 兼容性**：前端用 `stamp_params` 作为 key 存储用户设置，更新字段时注意保持向后兼容。

5. **预览请求防抖**：前端设置了 80ms 防抖，避免滑块拖动时请求过于频繁。
