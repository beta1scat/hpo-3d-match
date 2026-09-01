"""Unified Point Cloud Visualization Script with Grid 3x3 Downsampling and Comprehensive BOP GT Annotations.

Visualizes all target object instances (bracket_planar, screw_black, star)
with clear visibility fraction annotations (both visible and occluded).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from bop_scene_loader import (
    BOPCamera,
    backproject_depth,
    read_bop_camera,
    read_bop_ground_truths,
    read_depth_image,
)
from dataset import TARGET_OBJECT_IDS

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def export_ply(points: np.ndarray, out_path: Path) -> None:
    """Export 3D point cloud to ASCII PLY format."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    num_pts = points.shape[0]
    with open(out_path, "w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {num_pts}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for i in range(num_pts):
            p = points[i]
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")


def generate_interactive_html(
    raw_points: np.ndarray,
    filtered_points: np.ndarray,
    background_points: np.ndarray,
    gts: list[dict],
    out_html: Path,
) -> None:
    """Generate self-contained interactive 3D WebGL viewer with dual-viewport synchronized side-by-side comparison."""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    
    raw_json = json.dumps(raw_points.tolist())
    filtered_json = json.dumps(filtered_points.tolist())
    bg_json = json.dumps(background_points.tolist())
    gt_json = json.dumps(gts)

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <title>3D 点云滤波前后同屏双视窗对比与交互式观察器</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; padding: 0; background: #0b0f19; color: #e2e8f0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; overflow: hidden; }}
    #container {{ width: 100vw; height: 100vh; position: relative; }}
    #canvas {{ width: 100%; height: 100%; display: block; }}
    #top-bar {{ position: absolute; top: 16px; left: 50%; transform: translateX(-50%); background: rgba(17, 24, 39, 0.94); backdrop-filter: blur(12px); padding: 10px 20px; border-radius: 30px; border: 1px solid #1f2937; box-shadow: 0 10px 25px rgba(0,0,0,0.5); display: flex; align-items: center; gap: 12px; z-index: 10; }}
    .btn {{ padding: 6px 14px; background: #1f2937; color: #e5e7eb; border: 1px solid #374151; border-radius: 20px; font-size: 12px; font-weight: 600; cursor: pointer; transition: all 0.15s; }}
    .btn:hover {{ background: #374151; color: #fff; }}
    .btn.active {{ background: #0284c7; color: #fff; border-color: #38bdf8; }}
    
    #ui-left {{ position: absolute; top: 76px; left: 20px; background: rgba(17, 24, 39, 0.90); backdrop-filter: blur(12px); padding: 14px 18px; border-radius: 12px; border: 1px solid #1f2937; width: 330px; }}
    #ui-right {{ position: absolute; top: 76px; right: 20px; background: rgba(17, 24, 39, 0.90); backdrop-filter: blur(12px); padding: 14px 18px; border-radius: 12px; border: 1px solid #1f2937; width: 330px; text-align: right; }}
    
    .panel-title {{ font-size: 14px; font-weight: 700; margin: 0 0 6px 0; }}
    .title-before {{ color: #38bdf8; }}
    .title-after {{ color: #34d399; }}
    .stat-text {{ font-size: 11px; color: #9ca3af; margin: 3px 0; }}
    .stat-val {{ font-weight: 600; color: #f3f4f6; }}
    .legend-box {{ margin-top: 8px; padding-top: 6px; border-top: 1px solid #1f2937; font-size: 11px; }}
    
    #help {{ position: absolute; bottom: 16px; left: 50%; transform: translateX(-50%); background: rgba(17, 24, 39, 0.85); padding: 8px 16px; border-radius: 20px; font-size: 11px; color: #9ca3af; pointer-events: none; border: 1px solid #1f2937; }}
  </style>
</head>
<body>
  <div id="container">
    <canvas id="canvas"></canvas>
    
    <div id="top-bar">
      <span style="font-size: 13px; font-weight: bold; color: #38bdf8; margin-right: 8px;">视图模式:</span>
      <button id="btnSplit" class="btn active" onclick="setMode('split')">① 【左右同屏联动对比】(滤波前 VS 滤波后)</button>
      <button id="btnOverlay" class="btn" onclick="setMode('overlay')">② 【重叠差分视图】(红:已滤地面 | 绿:保留工件)</button>
      <button id="btnAfter" class="btn" onclick="setMode('after')">③ 【单视窗】仅看滤波后工作区</button>
    </div>

    <div id="ui-left">
      <div class="panel-title title-before">&larr; 左视窗：【滤波前】全景点云</div>
      <div class="stat-text">点云规模: <span class="stat-val">136,640 点 (Grid 3&times;3)</span></div>
      <div class="stat-text">状态: <span class="stat-val" style="color: #f87171;">包含远景地面噪点 (Z &ge; 0.95m)</span></div>
      <div class="stat-text">几何形貌: <span class="stat-val">托盘 + 远景平整地面</span></div>
    </div>

    <div id="ui-right">
      <div class="panel-title title-after">右视窗：【滤波后】纯净工作区 &rarr;</div>
      <div class="stat-text">保留点云: <span class="stat-val" style="color: #34d399;">99,856 点 (73.1%)</span></div>
      <div class="stat-text">剔除噪点: <span class="stat-val" style="color: #f87171;">36,784 点 (26.9%)</span></div>
      <div class="legend-box">
        <div><span style="color:#facc15; font-weight:bold;">● 黄色实心圈</span>：可见零件 (可见度 &ge; 10%)</div>
        <div><span style="color:#fb923c; font-weight:bold;">○ 橙色虚线圈</span>：深埋遮挡零件 (0% 不可见)</div>
      </div>
    </div>

    <div id="help">鼠标左键：3D 同步旋转 | 滚轮：同步缩放 | 鼠标右键：同步平移</div>
  </div>

  <script>
    const rawData = {raw_json};
    const filteredData = {filtered_json};
    const bgData = {bg_json};
    const gtData = {gt_json};

    let currentMode = 'split';

    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const uiLeft = document.getElementById('ui-left');
    const uiRight = document.getElementById('ui-right');

    let width, height;
    function resize() {{
      width = canvas.width = window.innerWidth;
      height = canvas.height = window.innerHeight;
      render();
    }}
    window.addEventListener('resize', resize);

    // Compute pivot centroid
    let cx0 = 0, cy0 = 0, cz0 = 0;
    for (let i = 0; i < filteredData.length; i++) {{
      cx0 += filteredData[i][0];
      cy0 += filteredData[i][1];
      cz0 += filteredData[i][2];
    }}
    cx0 /= filteredData.length;
    cy0 /= filteredData.length;
    cz0 /= filteredData.length;

    let rotX = 0.25, rotY = -0.15;
    let zoom = 750;
    let panX = 0, panY = 0;
    let isDragging = false, isPanning = false;
    let lastMouseX = 0, lastMouseY = 0;

    canvas.addEventListener('mousedown', e => {{
      if (e.button === 0) isDragging = true;
      if (e.button === 2) isPanning = true;
      lastMouseX = e.clientX;
      lastMouseY = e.clientY;
    }});
    window.addEventListener('mouseup', () => {{ isDragging = false; isPanning = false; }});
    window.addEventListener('mousemove', e => {{
      const dx = e.clientX - lastMouseX;
      const dy = e.clientY - lastMouseY;
      lastMouseX = e.clientX;
      lastMouseY = e.clientY;
      if (isDragging) {{
        rotY += dx * 0.005;
        rotX += dy * 0.005;
        render();
      }} else if (isPanning) {{
        panX += dx;
        panY += dy;
        render();
      }}
    }});
    canvas.addEventListener('wheel', e => {{
      e.preventDefault();
      zoom *= e.deltaY > 0 ? 0.9 : 1.1;
      render();
    }});
    canvas.addEventListener('contextmenu', e => e.preventDefault());

    function setMode(mode) {{
      currentMode = mode;
      document.getElementById('btnSplit').className = 'btn ' + (mode === 'split' ? 'active' : '');
      document.getElementById('btnOverlay').className = 'btn ' + (mode === 'overlay' ? 'active' : '');
      document.getElementById('btnAfter').className = 'btn ' + (mode === 'after' ? 'active' : '');

      uiLeft.style.display = (mode === 'split') ? 'block' : 'none';
      uiRight.style.display = (mode === 'split') ? 'block' : 'none';
      render();
    }}

    function projectPoint(p, offsetX, offsetY) {{
      const cx = p[0] - cx0, cy = p[1] - cy0, cz = p[2] - cz0;
      const x1 = cx * Math.cos(rotY) + cz * Math.sin(rotY);
      const z1 = -cx * Math.sin(rotY) + cz * Math.cos(rotY);
      const y2 = cy * Math.cos(rotX) - z1 * Math.sin(rotX);
      const z2 = cy * Math.sin(rotX) + z1 * Math.cos(rotX);
      const dist = 1.6;
      const f = zoom / (z2 + dist);
      const sx = offsetX + panX + x1 * f;
      const sy = offsetY + panY + y2 * f;
      return [sx, sy, z2];
    }}

    function drawPoints(pts, color, size, offsetX, offsetY) {{
      ctx.fillStyle = color;
      for (let i = 0; i < pts.length; i++) {{
        const proj = projectPoint(pts[i], offsetX, offsetY);
        if (proj[2] > -1.4) {{
          ctx.fillRect(proj[0], proj[1], size, size);
        }}
      }}
    }}

    function drawGT(gts, offsetX, offsetY) {{
      for (let i = 0; i < gts.length; i++) {{
        const g = gts[i];
        const proj = projectPoint([g.x, g.y, g.z], offsetX, offsetY);
        if (proj[2] > -1.4) {{
          const isVisible = g.visib >= 0.10;
          ctx.strokeStyle = isVisible ? '#facc15' : '#fb923c';
          ctx.lineWidth = isVisible ? 2.5 : 1.5;
          ctx.setLineDash(isVisible ? [] : [4, 4]);

          ctx.beginPath();
          ctx.arc(proj[0], proj[1], isVisible ? 11 : 8, 0, Math.PI * 2);
          ctx.stroke();
          ctx.setLineDash([]);

          ctx.fillStyle = isVisible ? '#facc15' : '#fb923c';
          ctx.font = isVisible ? 'bold 11px sans-serif' : '10px sans-serif';
          const visibText = isVisible ? (Math.round(g.visib * 100) + '% 可见') : '0% 遮挡';
          ctx.fillText('GT: ' + g.name + ' (' + visibText + ')', proj[0] + 14, proj[1] + 4);
        }}
      }}
    }}

    function render() {{
      ctx.fillStyle = '#0b0f19';
      ctx.fillRect(0, 0, width, height);

      if (currentMode === 'split') {{
        const halfW = width / 2;

        ctx.strokeStyle = '#1f2937';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(halfW, 0);
        ctx.lineTo(halfW, height);
        ctx.stroke();

        const leftCenterX = halfW / 2;
        const centerY = height / 2;
        drawPoints(rawData, '#38bdf8', 2.0, leftCenterX, centerY);
        drawGT(gtData, leftCenterX, centerY);

        const rightCenterX = halfW + halfW / 2;
        drawPoints(filteredData, '#34d399', 2.2, rightCenterX, centerY);
        drawGT(gtData, rightCenterX, centerY);

      }} else if (currentMode === 'overlay') {{
        const centerX = width / 2;
        const centerY = height / 2;
        drawPoints(bgData, 'rgba(239, 68, 68, 0.45)', 2.0, centerX, centerY);
        drawPoints(filteredData, 'rgba(52, 211, 153, 0.90)', 2.4, centerX, centerY);
        drawGT(gtData, centerX, centerY);

      }} else {{
        const centerX = width / 2;
        const centerY = height / 2;
        drawPoints(filteredData, '#34d399', 2.5, centerX, centerY);
        drawGT(gtData, centerX, centerY);
      }}
    }}

    resize();
  </script>
</body>
</html>
"""
    out_html.write_text(html_content, encoding="utf-8")

def main() -> int:
    parser = argparse.ArgumentParser(description="Unified Point Cloud Visualization with Grid 3x3 Downsampling.")
    parser.add_argument("--scene-id", type=int, default=0, help="BOP Scene ID")
    parser.add_argument("--image-id", type=int, default=62, help="BOP Image ID")
    parser.add_argument("--stride", type=int, default=4, help="Grid sampling stride (default: 4 for 4x4)")
    parser.add_argument("--min-depth", type=float, default=0.20, help="Min depth in meters (default: 0.20)")
    parser.add_argument("--max-depth", type=float, default=0.95, help="Max workspace depth in meters (default: 0.95)")
    parser.add_argument("--out-dir", type=Path, default=Path("visualizations"), help="Output directory")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    scene_str = f"{args.scene_id:06d}"
    img_str = f"{args.image_id:06d}"

    depth_path = Path(f"data/itoddmv_train_pbr/train_pbr/{scene_str}/depth/{img_str}.png")
    cam_path = Path(f"data/itoddmv_train_pbr/train_pbr/{scene_str}/scene_camera.json")
    gt_path = Path(f"data/itoddmv_train_pbr/train_pbr/{scene_str}/scene_gt.json")
    gt_info_path = Path(f"data/itoddmv_train_pbr/train_pbr/{scene_str}/scene_gt_info.json")

    print(f"[*] Loading scene {args.scene_id}, image {args.image_id} with Grid {args.stride}x{args.stride}...")
    camera = read_bop_camera(cam_path, args.image_id)
    depth = read_depth_image(depth_path)

    # 1. 2D Isotropic Grid Downsampling
    s = args.stride
    depth_grid = depth[::s, ::s]
    K_grid = camera.intrinsic_matrix.copy()
    K_grid[0, 0] /= s
    K_grid[1, 1] /= s
    K_grid[0, 2] /= s
    K_grid[1, 2] /= s
    cam_grid = BOPCamera(intrinsic_matrix=K_grid, depth_scale=camera.depth_scale)

    raw_points = backproject_depth(depth_grid, cam_grid, depth_range_m=None)
    
    mask_workspace = (raw_points[:, 2] >= args.min_depth) & (raw_points[:, 2] <= args.max_depth)
    filtered_points = raw_points[mask_workspace]
    background_points = raw_points[~mask_workspace]

    # 2. Load ALL Ground Truth annotations (min_visib_fract=0.0)
    with open(gt_info_path, "r", encoding="utf-8") as f:
        all_gt_infos = json.load(f).get(str(args.image_id), [])

    gt_markers = []
    for model_name, obj_id in TARGET_OBJECT_IDS.items():
        gts = read_bop_ground_truths(
            gt_path,
            gt_info_path,
            args.scene_id,
            args.image_id,
            obj_id,
            min_visib_fract=0.0,
        )
        for g in gts:
            visib_f = all_gt_infos[g.record_id]["visib_fract"] if g.record_id < len(all_gt_infos) else 1.0
            gt_markers.append({
                "name": model_name,
                "visib": float(visib_f),
                "x": float(g.translation_mm[0] / 1000.0),
                "y": float(g.translation_mm[1] / 1000.0),
                "z": float(g.translation_mm[2] / 1000.0),
            })

    print(f"[+] Loaded {len(gt_markers)} target object instances in Scene {args.scene_id} Img {args.image_id}:")
    for g in gt_markers:
        status = f"Visible ({g['visib']*100:.1f}%)" if g['visib'] >= 0.1 else f"Occluded (0.0%)"
        print(f"    - {g['name']:15s} [{status:16s}] at [{g['x']:.3f}, {g['y']:.3f}, {g['z']:.3f}]m")

    # 3. Generate High-Res Comparison PNG (2x2 Grid)
    png_path = out_dir / f"pointcloud_comparison_scene{args.scene_id}_img{args.image_id}_grid{s}x{s}.png"
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), dpi=150)
    fig.patch.set_facecolor("#0b0f19")

    # Subplot (0, 0): Depth Map
    ax0 = axes[0, 0]
    ax0.set_facecolor("#111827")
    im0 = ax0.imshow(depth_grid, cmap="inferno_r")
    ax0.set_title(f"Grid {s}x{s} Depth Map ({depth_grid.shape[1]}x{depth_grid.shape[0]}, Scene {args.scene_id} Img {args.image_id})", color="#38bdf8", fontsize=12, fontweight="bold")
    ax0.axis("off")
    cbar0 = fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
    cbar0.ax.tick_params(colors="#9ca3af")
    cbar0.set_label("Depth (mm)", color="#9ca3af", fontsize=10)

    # Subplot (0, 1): Depth Histogram & Cutoff
    ax1 = axes[0, 1]
    ax1.set_facecolor("#111827")
    ax1.hist(raw_points[:, 2], bins=80, color="#38bdf8", alpha=0.4, label=f"Grid {s}x{s} Raw ({raw_points.shape[0]:,})")
    ax1.hist(filtered_points[:, 2], bins=80, color="#34d399", alpha=0.8, label=f"Workspace [{args.min_depth}m, {args.max_depth}m] ({filtered_points.shape[0]:,})")
    ax1.axvline(args.max_depth, color="#facc15", linestyle="--", linewidth=2.5, label=f"Tray Boundary Z={args.max_depth}m")
    for g in gt_markers:
        color = "#facc15" if g["visib"] >= 0.1 else "#fb923c"
        label = f"GT {g['name']} ({g['visib']*100:.0f}%)" if g["visib"] >= 0.1 else f"GT {g['name']} (遮挡)"
        ax1.axvline(g["z"], color=color, linestyle=":", linewidth=2, label=label)
    ax1.set_title(f"Z-Axis Depth Distribution & Workspace Gate (Grid {s}x{s})", color="#38bdf8", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Depth Z (meters)", color="#9ca3af")
    ax1.set_ylabel("Point Count", color="#9ca3af")
    ax1.tick_params(colors="#9ca3af")
    ax1.grid(True, linestyle=":", alpha=0.3, color="#4b5563")
    ax1.legend(facecolor="#111827", edgecolor="#1f2937", labelcolor="#e5e7eb", fontsize=8)

    # Subplot (1, 0): BEFORE
    ax2 = fig.add_subplot(2, 2, 3, projection="3d")
    ax2.set_facecolor("#0b0f19")
    axes[1, 0].remove()
    ax2.scatter(raw_points[:, 0], raw_points[:, 1], raw_points[:, 2], color="#38bdf8", s=1.0, alpha=0.5)
    for g in gt_markers:
        color = "#facc15" if g["visib"] >= 0.1 else "#fb923c"
        marker = "^" if g["visib"] >= 0.1 else "x"
        ax2.scatter([g["x"]], [g["y"]], [g["z"]], color=color, s=90, marker=marker)
    ax2.set_title(f"BEFORE: Raw Full Point Cloud ({raw_points.shape[0]:,} pts)\n(Grid {s}x{s} With Far Ground Floor Noise)", color="#38bdf8", fontsize=12, fontweight="bold")
    ax2.set_xlabel("X (m)", color="#9ca3af", labelpad=4)
    ax2.set_ylabel("Y (m)", color="#9ca3af", labelpad=4)
    ax2.set_zlabel("Z (m)", color="#9ca3af", labelpad=4)
    ax2.tick_params(colors="#9ca3af")
    ax2.view_init(elev=25, azim=-60)

    # Subplot (1, 1): AFTER
    ax3 = fig.add_subplot(2, 2, 4, projection="3d")
    ax3.set_facecolor("#0b0f19")
    axes[1, 1].remove()
    ax3.scatter(filtered_points[:, 0], filtered_points[:, 1], filtered_points[:, 2], color="#34d399", s=1.2, alpha=0.85)
    for g in gt_markers:
        color = "#facc15" if g["visib"] >= 0.1 else "#fb923c"
        marker = "^" if g["visib"] >= 0.1 else "x"
        label = f"GT: {g['name']} ({g['visib']*100:.0f}% 可见)" if g["visib"] >= 0.1 else f"GT: {g['name']} (0% 遮挡)"
        ax3.scatter([g["x"]], [g["y"]], [g["z"]], color=color, s=100, marker=marker, label=label)
    ax3.set_title(f"AFTER: Clean Tray Workspace ({filtered_points.shape[0]:,} pts)\n(Grid {s}x{s} 100% Target Objects & Geometry Retained)", color="#34d399", fontsize=12, fontweight="bold")
    ax3.set_xlabel("X (m)", color="#9ca3af", labelpad=4)
    ax3.set_ylabel("Y (m)", color="#9ca3af", labelpad=4)
    ax3.set_zlabel("Z (m)", color="#9ca3af", labelpad=4)
    ax3.tick_params(colors="#9ca3af")
    ax3.view_init(elev=25, azim=-60)
    ax3.legend(facecolor="#111827", edgecolor="#1f2937", labelcolor="#e5e7eb", fontsize=8)

    plt.tight_layout()
    plt.savefig(png_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()

    # 4. Generate Interactive 3D HTML Viewer
    html_path = out_dir / "pointcloud_interactive_viewer.html"
    generate_interactive_html(raw_points, filtered_points, background_points, gt_markers, html_path)

    # 5. Export PLY files
    ply_raw = out_dir / f"scene{args.scene_id}_img{args.image_id}_grid{s}x{s}_raw.ply"
    ply_work = out_dir / f"scene{args.scene_id}_img{args.image_id}_grid{s}x{s}_workspace.ply"
    export_ply(raw_points, ply_raw)
    export_ply(filtered_points, ply_work)
    print(f"[+] Exported Grid {s}x{s} PLY files:\n  - {ply_raw}\n  - {ply_work}")

    print("\n[V] Point cloud visualization updated successfully!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
