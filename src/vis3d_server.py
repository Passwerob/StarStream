"""
Standalone 3D point cloud + camera viewer.
No Gradio dependency — just a plain HTTP server + Three.js frontend.

Usage:
    python vis3d_server.py /path/to/checkpoint-dir
    python vis3d_server.py /path/to/checkpoint-dir --port 8001
"""

import argparse
import http.server
import io
import json
import socket
import struct
import threading
from pathlib import Path

import numpy as np


def load_ply(path, max_points=300_000):
    with open(path) as f:
        n = 0
        while True:
            line = f.readline().strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            if line == "end_header":
                break
        data = np.loadtxt(f, max_rows=n)

    xyz = data[:, :3].astype(np.float32)
    rgb = data[:, 3:6].astype(np.float32)
    conf = data[:, 6].astype(np.float32) if data.shape[1] > 6 else np.ones(len(xyz), np.float32)

    mask = conf > 0
    xyz, rgb = xyz[mask], rgb[mask]

    if len(xyz) > 500:
        med = np.median(xyz, axis=0)
        d = np.linalg.norm(xyz - med, axis=1)
        keep = d < np.percentile(d, 97) * 1.5
        xyz, rgb = xyz[keep], rgb[keep]

    if len(xyz) > max_points:
        idx = np.random.default_rng(42).choice(len(xyz), max_points, replace=False)
        idx.sort()
        xyz, rgb = xyz[idx], rgb[idx]

    return xyz, rgb


def prepare_data(input_dir, max_points):
    d = Path(input_dir)
    merged = d / "point_cloud" / "merged.ply"
    if not merged.exists():
        plys = sorted((d / "point_cloud").glob("frame_*.ply"))
        merged = plys[0] if plys else None
    if merged is None:
        raise FileNotFoundError("No PLY found")

    xyz, rgb = load_ply(str(merged), max_points=max_points)

    center = xyz.mean(axis=0)
    xyz_c = xyz - center
    extent = np.abs(xyz_c).max()
    if extent > 0:
        xyz_c = xyz_c / extent

    rgb_norm = rgb / 255.0

    cameras = []
    tf_path = d / "transforms.json"
    if tf_path.exists():
        tf = json.loads(tf_path.read_text())
        for frame in tf["frames"]:
            m = np.array(frame["transform_matrix"], dtype=np.float64)
            pos = m[:3, 3].copy()
            pos = (pos - center) / extent if extent > 0 else pos - center
            rot = m[:3, :3].tolist()
            img_path = d / frame["file_path"]
            cameras.append({
                "pos": pos.tolist(),
                "rot": rot,
                "w": frame["w"],
                "h": frame["h"],
                "name": Path(frame["file_path"]).stem,
                "image": str(img_path) if img_path.exists() else "",
            })

    buf = io.BytesIO()
    buf.write(struct.pack("<I", len(xyz_c)))
    buf.write(xyz_c.astype(np.float32).tobytes())
    buf.write(rgb_norm.astype(np.float32).tobytes())
    point_data = buf.getvalue()

    return point_data, cameras, len(xyz_c)


HTML_PAGE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>3D Point Cloud Viewer</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#111;color:#eee;font-family:system-ui,-apple-system,sans-serif;overflow:hidden}
#c{display:block;width:100vw;height:100vh}
#hud{position:fixed;top:12px;left:14px;font:bold 13px monospace;
  background:rgba(0,0,0,0.6);padding:6px 12px;border-radius:6px;pointer-events:none;z-index:10}
#status{position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);
  font:20px monospace;color:#888;z-index:20}
#controls{position:fixed;bottom:10px;left:14px;font:11px monospace;color:#666;z-index:10;pointer-events:none}
#thumb{position:fixed;bottom:14px;right:14px;display:none;
  border:2px solid #4af;border-radius:6px;overflow:hidden;box-shadow:0 4px 20px rgba(0,0,0,0.7);z-index:10}
#thumb img{display:block;max-width:240px}
#camlist{position:fixed;top:12px;right:14px;max-height:80vh;overflow-y:auto;
  background:rgba(0,0,0,0.5);border-radius:6px;padding:6px 0;font:12px monospace;z-index:10;min-width:120px}
.cam-btn{display:block;width:100%;text-align:left;background:none;border:none;
  color:#aaa;padding:4px 12px;cursor:pointer;font:12px monospace}
.cam-btn:hover{background:rgba(70,170,255,0.2);color:#fff}
</style>
</head>
<body>
<canvas id="c"></canvas>
<div id="hud"></div>
<div id="status">Loading point cloud...</div>
<div id="controls">LMB: orbit | RMB: pan | Scroll: zoom | Click frustum: fly to camera</div>
<div id="thumb"><img id="timg"></div>
<div id="camlist"></div>

<script type="importmap">{"imports":{"three":"https://esm.sh/three@0.162.0","three/addons/":"https://esm.sh/three@0.162.0/examples/jsm/"}}</script>
<script type="module">
import * as T from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const canvas = document.getElementById('c');
const hud = document.getElementById('hud');
const status = document.getElementById('status');
const thumbDiv = document.getElementById('thumb');
const thumbImg = document.getElementById('timg');
const camList = document.getElementById('camlist');

const renderer = new T.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.setClearColor(0x111111);

const scene = new T.Scene();
const cam = new T.PerspectiveCamera(50, innerWidth / innerHeight, 0.001, 200);
cam.position.set(0, 0.6, 2.5);

const ctrl = new OrbitControls(cam, canvas);
ctrl.enableDamping = true;
ctrl.dampingFactor = 0.12;
ctrl.screenSpacePanning = true;
ctrl.target.set(0, 0, 0);

scene.add(new T.AxesHelper(0.15));

let cameras = [];
const frustumObjs = [];

async function loadData() {
  const [binResp, camResp] = await Promise.all([
    fetch('/data/points.bin'),
    fetch('/data/cameras.json'),
  ]);

  const bin = await binResp.arrayBuffer();
  const dv = new DataView(bin);
  const N = dv.getUint32(0, true);
  const positions = new Float32Array(bin, 4, N * 3);
  const colors = new Float32Array(bin, 4 + N * 3 * 4, N * 3);

  status.style.display = 'none';

  // DEBUG: check actual color values
  let minC = Infinity, maxC = -Infinity, sumC = 0;
  for (let i = 0; i < Math.min(N * 3, colors.length); i++) {
    if (colors[i] < minC) minC = colors[i];
    if (colors[i] > maxC) maxC = colors[i];
    sumC += colors[i];
  }
  console.log(`Points: ${N}, Color range: [${minC.toFixed(3)}, ${maxC.toFixed(3)}], mean: ${(sumC / (N * 3)).toFixed(3)}`);

  const geom = new T.BufferGeometry();
  geom.setAttribute('position', new T.Float32BufferAttribute(positions, 3));
  geom.setAttribute('color', new T.Float32BufferAttribute(colors, 3));

  const mat = new T.PointsMaterial({
    size: 2.0,
    sizeAttenuation: false,
    vertexColors: true,
  });
  const cloud = new T.Points(geom, mat);
  scene.add(cloud);

  hud.textContent = `${N.toLocaleString()} points`;

  cameras = await camResp.json();
  hud.textContent += ` | ${cameras.length} cameras`;
  buildFrustums();
  buildCamList();
}

function buildFrustums() {
  const S = 0.06;
  cameras.forEach((c, i) => {
    const aspect = c.w / c.h;
    const hw = S * aspect * 0.5, hh = S * 0.5, d = S;
    const verts = new Float32Array([
      0, 0, 0, -hw, -hh, d,
      0, 0, 0, hw, -hh, d,
      0, 0, 0, hw, hh, d,
      0, 0, 0, -hw, hh, d,
      -hw, -hh, d, hw, -hh, d,
      hw, -hh, d, hw, hh, d,
      hw, hh, d, -hw, hh, d,
      -hw, hh, d, -hw, -hh, d,
    ]);
    const lg = new T.BufferGeometry();
    lg.setAttribute('position', new T.Float32BufferAttribute(verts, 3));
    const col = new T.Color().setHSL(i / cameras.length, 0.85, 0.55);
    const lm = new T.LineBasicMaterial({ color: col, linewidth: 2 });
    const ls = new T.LineSegments(lg, lm);

    const r = c.rot;
    const rm = new T.Matrix4().set(
      r[0][0], r[0][1], r[0][2], 0,
      r[1][0], r[1][1], r[1][2], 0,
      r[2][0], r[2][1], r[2][2], 0,
      0, 0, 0, 1
    );
    ls.position.set(c.pos[0], c.pos[1], c.pos[2]);
    ls.quaternion.setFromRotationMatrix(rm);
    ls.userData = { idx: i };
    scene.add(ls);

    const hitGeo = new T.SphereGeometry(S * 0.8);
    const hitMat = new T.MeshBasicMaterial({ visible: false });
    const hitMesh = new T.Mesh(hitGeo, hitMat);
    hitMesh.position.copy(ls.position);
    hitMesh.userData = { idx: i };
    scene.add(hitMesh);
    frustumObjs.push(hitMesh);
  });
}

function buildCamList() {
  cameras.forEach((c, i) => {
    const btn = document.createElement('button');
    btn.className = 'cam-btn';
    btn.textContent = c.name;
    btn.onclick = () => flyTo(i);
    camList.appendChild(btn);
  });
}

const rc = new T.Raycaster();
let downPt = new T.Vector2();

canvas.addEventListener('pointerdown', e => {
  const r = canvas.getBoundingClientRect();
  downPt.set((e.clientX - r.left) / r.width * 2 - 1, -(e.clientY - r.top) / r.height * 2 + 1);
});

canvas.addEventListener('pointerup', e => {
  const r = canvas.getBoundingClientRect();
  const p = new T.Vector2(
    (e.clientX - r.left) / r.width * 2 - 1,
    -(e.clientY - r.top) / r.height * 2 + 1
  );
  if (p.distanceTo(downPt) > 0.02) return;
  rc.setFromCamera(p, cam);
  const hits = rc.intersectObjects(frustumObjs);
  if (hits.length) flyTo(hits[0].object.userData.idx);
});

canvas.addEventListener('pointermove', e => {
  const r = canvas.getBoundingClientRect();
  const p = new T.Vector2(
    (e.clientX - r.left) / r.width * 2 - 1,
    -(e.clientY - r.top) / r.height * 2 + 1
  );
  rc.setFromCamera(p, cam);
  const hits = rc.intersectObjects(frustumObjs);
  canvas.style.cursor = hits.length ? 'pointer' : 'grab';
});

let flying = false;
function flyTo(idx) {
  if (flying) return;
  flying = true;
  const c = cameras[idx];
  const tgt = new T.Vector3(c.pos[0], c.pos[1], c.pos[2]);
  const fwd = new T.Vector3(0, 0, 1);
  const rm = new T.Matrix4().set(
    c.rot[0][0], c.rot[0][1], c.rot[0][2], 0,
    c.rot[1][0], c.rot[1][1], c.rot[1][2], 0,
    c.rot[2][0], c.rot[2][1], c.rot[2][2], 0,
    0, 0, 0, 1
  );
  fwd.applyMatrix4(rm);
  const look = tgt.clone().add(fwd.multiplyScalar(0.5));
  const sp = cam.position.clone(), st = ctrl.target.clone();
  const dur = 600, t0 = performance.now();
  (function step(now) {
    let t = Math.min((now - t0) / dur, 1);
    t = t < .5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2;
    cam.position.lerpVectors(sp, tgt, t);
    ctrl.target.lerpVectors(st, look, t);
    ctrl.update();
    if (t < 1) requestAnimationFrame(step); else flying = false;
  })(t0);
  hud.textContent = c.name;
  if (c.image) {
    thumbImg.src = '/data/image/' + c.name;
    thumbDiv.style.display = 'block';
  }
}

window.addEventListener('resize', () => {
  renderer.setSize(innerWidth, innerHeight);
  cam.aspect = innerWidth / innerHeight;
  cam.updateProjectionMatrix();
});

(function loop() {
  requestAnimationFrame(loop);
  ctrl.update();
  renderer.render(scene, cam);
})();

loadData();
</script>
</body>
</html>
"""


class ViewerHandler(http.server.BaseHTTPRequestHandler):
    point_data = b""
    cameras = []
    input_dir = ""

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._respond(200, "text/html", HTML_PAGE.encode())
        elif self.path == "/data/points.bin":
            self._respond(200, "application/octet-stream", self.point_data)
        elif self.path == "/data/cameras.json":
            self._respond(200, "application/json", json.dumps(self.cameras).encode())
        elif self.path.startswith("/data/image/"):
            name = self.path.split("/")[-1]
            d = Path(self.input_dir)
            for ext in [".png", ".jpg", ".jpeg"]:
                img = d / "images" / (name + ext)
                if not img.exists():
                    img = d / (name + ext)
                if img.exists():
                    ct = "image/png" if ext == ".png" else "image/jpeg"
                    self._respond(200, ct, img.read_bytes())
                    return
            self._respond(404, "text/plain", b"not found")
        else:
            self._respond(404, "text/plain", b"not found")

    def _respond(self, code, content_type, data):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)


def find_port(start=8001, n=50):
    for p in range(start, start + n):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise RuntimeError("No free port")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--max-points", type=int, default=300_000)
    args = ap.parse_args()

    port = find_port(args.port)
    if port != args.port:
        print(f"Port {args.port} busy, using {port}")

    print(f"Loading {args.input_dir} ...")
    point_data, cameras, n_pts = prepare_data(args.input_dir, args.max_points)
    print(f"  {n_pts:,} points, {len(cameras)} cameras")
    print(f"  Binary data: {len(point_data):,} bytes")

    ViewerHandler.point_data = point_data
    ViewerHandler.cameras = cameras
    ViewerHandler.input_dir = args.input_dir

    server = http.server.HTTPServer(("0.0.0.0", port), ViewerHandler)
    print(f"\n  Open: http://localhost:{port}\n")
    print(f"  SSH tunnel: ssh -L {port}:localhost:{port} <server>")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
