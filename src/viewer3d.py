#!/usr/bin/env python3
"""
Interactive 3D Point Cloud & Camera Pose Viewer (Gradio + Three.js).

Usage:
    python viewer3d.py                                          # launch with UI
    python viewer3d.py --dir /path/to/output/checkpoint-3       # pre-load a directory
    python viewer3d.py --port 8001 --share                      # custom port + public link
"""

import argparse
import base64
import json
import socket
import struct
import sys
from pathlib import Path

import gradio as gr
import numpy as np

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ply_fast(path: Path):
    with open(path, "r") as f:
        n = 0
        while True:
            line = f.readline().strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            if line == "end_header":
                break
        data = np.loadtxt(f, max_rows=n, dtype=np.float32)
    xyz = data[:, :3]
    rgb = data[:, 3:6].astype(np.uint8)
    conf = data[:, 6] if data.shape[1] > 6 else np.ones(len(xyz), dtype=np.float32)
    return xyz, rgb, conf


def load_scene(output_dir: str, conf_thresh: float = 0.0, max_points: int = 500_000):
    root = Path(output_dir)
    if not root.is_dir():
        raise gr.Error(f"Directory not found: {output_dir}")

    # --- point cloud ---
    merged = root / "point_cloud" / "merged.ply"
    if not merged.exists():
        raise gr.Error("merged.ply not found in point_cloud/")
    xyz, rgb, conf = load_ply_fast(merged)

    mask = conf >= conf_thresh
    xyz, rgb, conf = xyz[mask], rgb[mask], conf[mask]

    center = np.median(xyz, axis=0)
    dists = np.linalg.norm(xyz - center, axis=1)
    p95 = np.percentile(dists, 95)
    keep = dists < p95 * 2
    xyz, rgb, conf = xyz[keep], rgb[keep], conf[keep]

    if len(xyz) > max_points:
        idx = np.random.default_rng(42).choice(len(xyz), max_points, replace=False)
        idx.sort()
        xyz, rgb, conf = xyz[idx], rgb[idx], conf[idx]

    # --- cameras ---
    tf_path = root / "transforms.json"
    cameras = []
    if tf_path.exists():
        tf = json.loads(tf_path.read_text())
        for i, fr in enumerate(tf.get("frames", [])):
            c2w = np.array(fr["transform_matrix"], dtype=np.float32)
            cam = {
                "c2w": c2w.tolist(),
                "fl_x": fr.get("fl_x", 500),
                "fl_y": fr.get("fl_y", 500),
                "cx": fr.get("cx", 0),
                "cy": fr.get("cy", 0),
                "w": fr.get("w", 960),
                "h": fr.get("h", 540),
                "name": fr.get("file_path", f"frame_{i:06d}"),
            }
            img_path = root / fr.get("file_path", "")
            if img_path.exists():
                import base64 as b64
                raw = img_path.read_bytes()
                cam["thumb"] = f"data:image/png;base64,{b64.b64encode(raw).decode()}"
            cameras.append(cam)

    return xyz, rgb, conf, cameras


# ---------------------------------------------------------------------------
# Binary encoding helpers
# ---------------------------------------------------------------------------

def encode_points(xyz, rgb):
    n = len(xyz)
    buf = struct.pack("<I", n)
    buf += xyz.astype(np.float32).tobytes()
    buf += rgb.astype(np.uint8).tobytes()
    return base64.b64encode(buf).decode()


# ---------------------------------------------------------------------------
# Three.js HTML builder
# ---------------------------------------------------------------------------

VIEWER_HTML_TEMPLATE = r"""
<div id="viewer-root" style="width:100%;height:720px;position:relative;background:#111;border-radius:8px;overflow:hidden;">
  <canvas id="c3d" style="width:100%;height:100%;display:block;"></canvas>
  <div id="cam-label" style="position:absolute;top:12px;left:12px;color:#fff;font:bold 13px monospace;
       background:rgba(0,0,0,.55);padding:4px 10px;border-radius:6px;pointer-events:none;display:none;"></div>
  <div id="thumb-panel" style="position:absolute;bottom:12px;left:12px;display:none;
       background:rgba(0,0,0,.7);border-radius:8px;padding:6px;pointer-events:none;">
    <img id="thumb-img" style="max-width:240px;max-height:160px;border-radius:4px;" />
    <div id="thumb-name" style="color:#ccc;font:11px monospace;text-align:center;margin-top:4px;"></div>
  </div>
  <div id="help-text" style="position:absolute;bottom:12px;right:12px;color:#888;font:11px sans-serif;
       background:rgba(0,0,0,.45);padding:4px 10px;border-radius:6px;pointer-events:none;">
    Left-drag: rotate &nbsp;|&nbsp; Right-drag: pan &nbsp;|&nbsp; Scroll: zoom &nbsp;|&nbsp; Click frustum: jump to view
  </div>
</div>

<script type="importmap">
{ "imports": {
    "three": "https://esm.sh/three@0.170.0",
    "three/addons/": "https://esm.sh/three@0.170.0/examples/jsm/"
} }
</script>

<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const POINT_DATA_B64 = "__POINT_DATA__";
const CAMERA_DATA = __CAMERA_JSON__;
const POINT_SIZE = __POINT_SIZE__;
const FRUSTUM_SCALE = __FRUSTUM_SCALE__;

// --- decode points ---
const raw = Uint8Array.from(atob(POINT_DATA_B64), c => c.charCodeAt(0));
const dv = new DataView(raw.buffer);
const N = dv.getUint32(0, true);
const xyzOff = 4, rgbOff = 4 + N * 12;
const positions = new Float32Array(raw.buffer, xyzOff, N * 3);
const colors = new Float32Array(N * 3);
for (let i = 0; i < N; i++) {
  colors[i*3]   = raw[rgbOff + i*3]   / 255;
  colors[i*3+1] = raw[rgbOff + i*3+1] / 255;
  colors[i*3+2] = raw[rgbOff + i*3+2] / 255;
}

// --- compute scene center & scale ---
let cx=0, cy=0, cz=0;
for (let i=0; i<N; i++) { cx+=positions[i*3]; cy+=positions[i*3+1]; cz+=positions[i*3+2]; }
cx/=N; cy/=N; cz/=N;
let maxD = 0;
for (let i=0; i<N; i++) {
  const dx=positions[i*3]-cx, dy=positions[i*3+1]-cy, dz=positions[i*3+2]-cz;
  const d = Math.sqrt(dx*dx+dy*dy+dz*dz);
  if (d>maxD) maxD=d;
}
const sceneScale = maxD > 0 ? maxD : 1;

// --- setup Three.js ---
const container = document.getElementById('viewer-root');
const canvas = document.getElementById('c3d');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.setClearColor(0x111111);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.01, sceneScale * 20);
camera.position.set(cx + sceneScale*0.8, cy + sceneScale*0.5, cz + sceneScale*0.8);
camera.lookAt(cx, cy, cz);

const controls = new OrbitControls(camera, canvas);
controls.target.set(cx, cy, cz);
controls.enableDamping = true;
controls.dampingFactor = 0.12;
controls.minDistance = sceneScale * 0.01;
controls.maxDistance = sceneScale * 10;
controls.update();

// --- point cloud ---
const geom = new THREE.BufferGeometry();
geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
const mat = new THREE.PointsMaterial({ size: POINT_SIZE, vertexColors: true, sizeAttenuation: true });
scene.add(new THREE.Points(geom, mat));

// --- camera frustums ---
const frustumGroup = new THREE.Group();
scene.add(frustumGroup);
const raycasterTargets = [];

const FCOLORS = [0x00ccff, 0xff6644, 0x44ff88, 0xffcc00, 0xcc44ff,
                 0xff4488, 0x44ccff, 0x88ff44, 0xff8800, 0x8844ff,
                 0x00ff99, 0xff0066, 0x66ff00, 0x0066ff, 0xff9900];

CAMERA_DATA.forEach((cam, idx) => {
  const m = new THREE.Matrix4();
  const c = cam.c2w;
  m.set(c[0][0],c[0][1],c[0][2],c[0][3],
        c[1][0],c[1][1],c[1][2],c[1][3],
        c[2][0],c[2][1],c[2][2],c[2][3],
        c[3][0],c[3][1],c[3][2],c[3][3]);

  const pos = new THREE.Vector3();
  pos.setFromMatrixPosition(m);

  const s = FRUSTUM_SCALE * sceneScale * 0.05;
  const hw = (cam.w / cam.fl_x) * s * 0.5;
  const hh = (cam.h / cam.fl_y) * s * 0.5;

  const corners = [
    new THREE.Vector3(-hw, -hh, -s),
    new THREE.Vector3( hw, -hh, -s),
    new THREE.Vector3( hw,  hh, -s),
    new THREE.Vector3(-hw,  hh, -s),
  ];
  corners.forEach(p => p.applyMatrix4(m));

  const col = FCOLORS[idx % FCOLORS.length];
  const lm = new THREE.LineBasicMaterial({ color: col, linewidth: 2 });

  // pyramid edges
  const pts = [
    pos, corners[0], pos, corners[1], pos, corners[2], pos, corners[3],
    corners[0], corners[1], corners[1], corners[2], corners[2], corners[3], corners[3], corners[0]
  ];
  const lg = new THREE.BufferGeometry().setFromPoints(pts);
  frustumGroup.add(new THREE.LineSegments(lg, lm));

  // up indicator
  const upDir = new THREE.Vector3().subVectors(corners[3], corners[0]).normalize();
  const midTop = new THREE.Vector3().addVectors(corners[3], corners[2]).multiplyScalar(0.5);
  const upTip = midTop.clone().add(upDir.multiplyScalar(s * 0.25));
  const upG = new THREE.BufferGeometry().setFromPoints([midTop, upTip]);
  frustumGroup.add(new THREE.LineSegments(upG, new THREE.LineBasicMaterial({ color: 0xff3333, linewidth: 2 })));

  // clickable sphere (invisible)
  const sg = new THREE.SphereGeometry(s * 0.3, 8, 8);
  const sm = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, depthWrite: false });
  const sphere = new THREE.Mesh(sg, sm);
  sphere.position.copy(pos);
  sphere.userData = { camIdx: idx, c2w: m.clone(), cam };
  frustumGroup.add(sphere);
  raycasterTargets.push(sphere);

  // label sprite
  const labelCanvas = document.createElement('canvas');
  labelCanvas.width = 64; labelCanvas.height = 32;
  const ctx2 = labelCanvas.getContext('2d');
  ctx2.fillStyle = `#${col.toString(16).padStart(6,'0')}`;
  ctx2.font = 'bold 20px monospace';
  ctx2.textAlign = 'center';
  ctx2.fillText(String(idx), 32, 22);
  const tex = new THREE.CanvasTexture(labelCanvas);
  const spriteMat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false });
  const sprite = new THREE.Sprite(spriteMat);
  sprite.position.copy(pos).add(new THREE.Vector3(0, s*0.5, 0));
  sprite.scale.set(s*0.6, s*0.3, 1);
  frustumGroup.add(sprite);
});

// --- raycaster ---
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
let animating = false;

canvas.addEventListener('click', (e) => {
  const rect = canvas.getBoundingClientRect();
  mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);
  const hits = raycaster.intersectObjects(raycasterTargets);
  if (hits.length > 0) {
    const ud = hits[0].object.userData;
    jumpToCamera(ud.c2w, ud.cam, ud.camIdx);
  }
});

canvas.addEventListener('mousemove', (e) => {
  const rect = canvas.getBoundingClientRect();
  mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);
  const hits = raycaster.intersectObjects(raycasterTargets);
  const label = document.getElementById('cam-label');
  const thumbP = document.getElementById('thumb-panel');
  if (hits.length > 0) {
    const ud = hits[0].object.userData;
    label.textContent = `Camera ${ud.camIdx}: ${ud.cam.name}`;
    label.style.display = 'block';
    canvas.style.cursor = 'pointer';
    if (ud.cam.thumb) {
      document.getElementById('thumb-img').src = ud.cam.thumb;
      document.getElementById('thumb-name').textContent = ud.cam.name;
      thumbP.style.display = 'block';
    }
  } else {
    label.style.display = 'none';
    thumbP.style.display = 'none';
    canvas.style.cursor = 'grab';
  }
});

function jumpToCamera(c2wMat, cam, idx) {
  const pos = new THREE.Vector3().setFromMatrixPosition(c2wMat);
  const forward = new THREE.Vector3(0, 0, -1).transformDirection(c2wMat);
  const targetPt = pos.clone().add(forward.multiplyScalar(sceneScale * 0.3));

  const startPos = camera.position.clone();
  const startTarget = controls.target.clone();
  const duration = 600;
  const startTime = performance.now();
  animating = true;
  controls.enabled = false;

  function anim(now) {
    let t = Math.min((now - startTime) / duration, 1);
    t = t * t * (3 - 2 * t); // smoothstep
    camera.position.lerpVectors(startPos, pos, t);
    controls.target.lerpVectors(startTarget, targetPt, t);
    controls.update();
    if (t < 1) {
      requestAnimationFrame(anim);
    } else {
      animating = false;
      controls.enabled = true;
    }
  }
  requestAnimationFrame(anim);
}

// --- ambient light ---
scene.add(new THREE.AmbientLight(0xffffff, 1));

// --- resize ---
const ro = new ResizeObserver(() => {
  const w = container.clientWidth, h = container.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
});
ro.observe(container);

// --- render loop ---
function tick() {
  requestAnimationFrame(tick);
  if (!animating) controls.update();
  renderer.render(scene, camera);
}
tick();
</script>
"""


def build_viewer_html(xyz, rgb, cameras, point_size=0.01, frustum_scale=1.0):
    pts_b64 = encode_points(xyz, rgb)
    cam_json = json.dumps(cameras)
    html = VIEWER_HTML_TEMPLATE
    html = html.replace("__POINT_DATA__", pts_b64)
    html = html.replace("__CAMERA_JSON__", cam_json)
    html = html.replace("__POINT_SIZE__", str(point_size))
    html = html.replace("__FRUSTUM_SCALE__", str(frustum_scale))
    return html


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

def on_load(output_dir, conf_thresh, max_points, point_size, frustum_scale):
    if not output_dir or not output_dir.strip():
        raise gr.Error("Please enter the output directory path")

    xyz, rgb, conf, cameras = load_scene(output_dir.strip(), conf_thresh, max_points)
    info = f"Loaded **{len(xyz):,}** points, **{len(cameras)}** cameras from `{output_dir}`"
    html = build_viewer_html(xyz, rgb, cameras, point_size, frustum_scale)
    return html, info


def create_app(default_dir: str = ""):
    with gr.Blocks(title="3D Point Cloud Viewer") as demo:
        gr.Markdown("## 3D Point Cloud & Camera Viewer")

        with gr.Row():
            dir_input = gr.Textbox(
                label="Output Directory",
                placeholder="/path/to/output/checkpoint-X",
                value=default_dir,
                scale=5,
            )
            load_btn = gr.Button("Load", variant="primary", scale=1)

        with gr.Row():
            conf_slider = gr.Slider(0, 1, value=0.0, step=0.05, label="Confidence Threshold")
            pts_slider = gr.Slider(50_000, 2_000_000, value=500_000, step=50_000, label="Max Points")
            size_slider = gr.Slider(0.001, 0.1, value=0.008, step=0.001, label="Point Size")
            frust_slider = gr.Slider(0.1, 5.0, value=1.0, step=0.1, label="Frustum Scale")

        info_md = gr.Markdown("")
        viewer = gr.HTML(elem_id="viewer-html")

        load_btn.click(
            fn=on_load,
            inputs=[dir_input, conf_slider, pts_slider, size_slider, frust_slider],
            outputs=[viewer, info_md],
        )
        dir_input.submit(
            fn=on_load,
            inputs=[dir_input, conf_slider, pts_slider, size_slider, frust_slider],
            outputs=[viewer, info_md],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def find_free_port(start: int = 8001, tries: int = 100) -> int:
    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    return start + tries


def main():
    parser = argparse.ArgumentParser(description="3D Point Cloud & Camera Viewer")
    parser.add_argument("--dir", default="", help="Pre-load this output directory")
    parser.add_argument("--port", type=int, default=8001, help="Starting port (auto-increment if busy)")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    args = parser.parse_args()

    port = find_free_port(args.port)
    demo = create_app(default_dir=args.dir)
    print(f"Starting viewer on port {port} ...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
