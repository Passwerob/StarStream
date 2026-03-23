"""
3D Point Cloud + Camera Pose Viewer (Gradio + Three.js).

Usage:
    python vis3d_app.py /path/to/checkpoint-output-dir
    python vis3d_app.py /path/to/checkpoint-output-dir --port 8001
"""

import argparse
import base64
import io
import json
import socket
import sys
from pathlib import Path

import gradio as gr
import numpy as np


def load_ply(path, max_points=300_000, conf_thresh=0.0):
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
    rgb = data[:, 3:6].astype(np.uint8)
    conf = data[:, 6].astype(np.float32) if data.shape[1] > 6 else np.ones(len(xyz), np.float32)

    mask = conf >= conf_thresh
    xyz, rgb, conf = xyz[mask], rgb[mask], conf[mask]

    if len(xyz) > 500:
        med = np.median(xyz, axis=0)
        d = np.linalg.norm(xyz - med, axis=1)
        keep = d < np.percentile(d, 97) * 1.5
        xyz, rgb, conf = xyz[keep], rgb[keep], conf[keep]

    if len(xyz) > max_points:
        idx = np.random.default_rng(42).choice(len(xyz), max_points, replace=False)
        idx.sort()
        xyz, rgb, conf = xyz[idx], rgb[idx], conf[idx]

    return xyz, rgb


def load_scene(directory, max_points=300_000):
    d = Path(directory)
    merged = d / "point_cloud" / "merged.ply"
    if not merged.exists():
        plys = sorted((d / "point_cloud").glob("frame_*.ply"))
        merged = plys[0] if plys else None
    if merged is None:
        raise FileNotFoundError("No PLY found")

    xyz, rgb = load_ply(str(merged), max_points=max_points)

    cameras = []
    tf_path = d / "transforms.json"
    if tf_path.exists():
        tf = json.loads(tf_path.read_text())
        for frame in tf["frames"]:
            img_path = d / frame["file_path"]
            cameras.append({
                "c2w": frame["transform_matrix"],
                "w": frame["w"], "h": frame["h"],
                "name": Path(frame["file_path"]).stem,
                "image": str(img_path),
            })

    return xyz, rgb, cameras


def img_to_uri(path, max_w=280):
    from PIL import Image
    img = Image.open(path)
    r = max_w / img.width
    img = img.resize((max_w, int(img.height * r)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def build_html(xyz, rgb, cameras):
    center = xyz.mean(axis=0)
    xyz_c = xyz - center
    extent = np.abs(xyz_c).max()
    if extent > 0:
        xyz_c = xyz_c / extent

    pos_b64 = base64.b64encode(xyz_c.astype(np.float32).tobytes()).decode()
    col_b64 = base64.b64encode((rgb.astype(np.float32) / 255.0).astype(np.float32).tobytes()).decode()

    cam_json = []
    for c in cameras:
        m = np.array(c["c2w"], dtype=np.float64)
        pos = m[:3, 3].copy()
        pos = (pos - center) / extent if extent > 0 else pos - center
        rot = m[:3, :3].tolist()
        uri = ""
        if Path(c["image"]).exists():
            try:
                uri = img_to_uri(c["image"])
            except Exception:
                pass
        cam_json.append({"pos": pos.tolist(), "rot": rot, "aspect": c["w"]/c["h"], "name": c["name"], "uri": uri})

    return HTML_TEMPLATE.replace("__POS_B64__", pos_b64).replace("__COL_B64__", col_b64).replace(
        "__N_POINTS__", str(len(xyz_c))).replace("__CAMERAS__", json.dumps(cam_json))


HTML_TEMPLATE = r"""
<div id="root" style="width:100%;height:720px;position:relative;background:#1a1a2e;border-radius:8px;overflow:hidden">
  <canvas id="c"></canvas>
  <div id="info" style="position:absolute;top:8px;left:10px;color:#fff;font:bold 12px monospace;
       background:rgba(0,0,0,0.55);padding:3px 8px;border-radius:4px;pointer-events:none"></div>
  <div id="thumb" style="position:absolute;bottom:8px;right:8px;display:none;border:2px solid #4af;border-radius:4px;overflow:hidden;box-shadow:0 0 12px rgba(0,0,0,0.5)">
    <img id="timg" style="display:block;max-width:220px"/></div>
  <div style="position:absolute;bottom:8px;left:8px;color:#888;font:10px monospace;pointer-events:none">
    LMB: orbit | RMB: pan | Scroll: zoom | Click frustum: fly to camera</div>
</div>
<script type="importmap">{"imports":{"three":"https://esm.sh/three@0.162.0","three/addons/":"https://esm.sh/three@0.162.0/examples/jsm/"}}</script>
<script type="module">
import*as T from'three';
import{OrbitControls}from'three/addons/controls/OrbitControls.js';

const NP=__N_POINTS__;
const cams=__CAMERAS__;

function b64toF32(b64,n){const bin=atob(b64);const buf=new ArrayBuffer(bin.length);const u8=new Uint8Array(buf);for(let i=0;i<bin.length;i++)u8[i]=bin.charCodeAt(i);return new Float32Array(buf,0,n);}

const positions=b64toF32("__POS_B64__",NP*3);
const colors=b64toF32("__COL_B64__",NP*3);

const root=document.getElementById('root');
const canvas=document.getElementById('c');
const info=document.getElementById('info');
const thumbDiv=document.getElementById('thumb');
const thumbImg=document.getElementById('timg');

const W=root.clientWidth,H=root.clientHeight;
const renderer=new T.WebGLRenderer({canvas,antialias:true,alpha:false});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
renderer.setSize(W,H);
renderer.setClearColor(0x1a1a2e);

const scene=new T.Scene();
const cam=new T.PerspectiveCamera(50,W/H,0.001,100);
cam.position.set(0,0.6,2.2);

const ctrl=new OrbitControls(cam,canvas);
ctrl.enableDamping=true;ctrl.dampingFactor=0.1;ctrl.screenSpacePanning=true;
ctrl.target.set(0,0,0);ctrl.update();

// point cloud
const pg=new T.BufferGeometry();
pg.setAttribute('position',new T.BufferAttribute(positions,3));
pg.setAttribute('color',new T.BufferAttribute(colors,3));
const pm=new T.PointsMaterial({size:3.0,vertexColors:true,sizeAttenuation:false});
const cloud=new T.Points(pg,pm);
scene.add(cloud);

info.textContent=NP.toLocaleString()+' points, '+cams.length+' cameras';

// camera frustums
const frustums=[];
const S=0.08;
for(let i=0;i<cams.length;i++){
  const c=cams[i];
  const hw=S*c.aspect*0.5,hh=S*0.5,d=S;
  const v=new Float32Array([
    0,0,0,-hw,-hh,-d, 0,0,0,hw,-hh,-d,
    0,0,0,hw,hh,-d,   0,0,0,-hw,hh,-d,
    -hw,-hh,-d,hw,-hh,-d, hw,-hh,-d,hw,hh,-d,
    hw,hh,-d,-hw,hh,-d,   -hw,hh,-d,-hw,-hh,-d,
  ]);
  const lg=new T.BufferGeometry();
  lg.setAttribute('position',new T.BufferAttribute(v,3));
  const col=new T.Color().setHSL(i/cams.length,0.85,0.6);
  const lm=new T.LineBasicMaterial({color:col,linewidth:2});
  const ls=new T.LineSegments(lg,lm);

  const m3=c.rot;
  const rm=new T.Matrix4().set(
    m3[0][0],m3[0][1],m3[0][2],0,
    m3[1][0],m3[1][1],m3[1][2],0,
    m3[2][0],m3[2][1],m3[2][2],0,
    0,0,0,1);
  const q=new T.Quaternion().setFromRotationMatrix(rm);
  ls.position.set(c.pos[0],c.pos[1],c.pos[2]);
  ls.quaternion.copy(q);
  ls.userData={idx:i};
  scene.add(ls);

  const pg2=new T.PlaneGeometry(hw*2,hh*2);
  const pm2=new T.MeshBasicMaterial({visible:false,side:T.DoubleSide});
  const pl=new T.Mesh(pg2,pm2);
  pl.position.set(0,0,-d);
  pl.userData={idx:i};
  ls.add(pl);
  frustums.push(ls,pl);
}

// axes
scene.add(new T.AxesHelper(0.3));

// raycaster
const rc=new T.Raycaster();
rc.params.Line={threshold:0.02};
let downPt=new T.Vector2();
canvas.addEventListener('pointerdown',e=>{
  const r=canvas.getBoundingClientRect();
  downPt.set((e.clientX-r.left)/r.width*2-1,-(e.clientY-r.top)/r.height*2+1);
});
canvas.addEventListener('pointerup',e=>{
  const r=canvas.getBoundingClientRect();
  const p=new T.Vector2((e.clientX-r.left)/r.width*2-1,-(e.clientY-r.top)/r.height*2+1);
  if(p.distanceTo(downPt)>0.02)return;
  rc.setFromCamera(p,cam);
  const hits=rc.intersectObjects(frustums,true);
  if(hits.length){
    let o=hits[0].object;
    while(o&&o.userData.idx===undefined)o=o.parent;
    if(o)flyTo(o.userData.idx);
  }
});
canvas.addEventListener('pointermove',e=>{
  const r=canvas.getBoundingClientRect();
  const p=new T.Vector2((e.clientX-r.left)/r.width*2-1,-(e.clientY-r.top)/r.height*2+1);
  rc.setFromCamera(p,cam);
  const hits=rc.intersectObjects(frustums,true);
  canvas.style.cursor=hits.length?'pointer':'grab';
  if(hits.length){
    let o=hits[0].object;
    while(o&&o.userData.idx===undefined)o=o.parent;
    if(o)info.textContent=cams[o.userData.idx].name;
  }
});

let flying=false;
function flyTo(idx){
  if(flying)return;flying=true;
  const c=cams[idx];
  const tgt=new T.Vector3(c.pos[0],c.pos[1],c.pos[2]);
  const fwd=new T.Vector3(0,0,-1);
  const rm=new T.Matrix4().set(
    c.rot[0][0],c.rot[0][1],c.rot[0][2],0,
    c.rot[1][0],c.rot[1][1],c.rot[1][2],0,
    c.rot[2][0],c.rot[2][1],c.rot[2][2],0,
    0,0,0,1);
  fwd.applyMatrix4(rm);
  const look=tgt.clone().add(fwd.multiplyScalar(0.5));
  const sp=cam.position.clone(),st=ctrl.target.clone();
  const dur=500,t0=performance.now();
  (function step(now){
    let t=Math.min((now-t0)/dur,1);
    t=t<.5?4*t*t*t:1-(-2*t+2)**3/2;
    cam.position.lerpVectors(sp,tgt,t);
    ctrl.target.lerpVectors(st,look,t);
    ctrl.update();
    if(t<1)requestAnimationFrame(step);
    else flying=false;
  })(t0);
  if(c.uri){thumbImg.src=c.uri;thumbDiv.style.display='block';}
  info.textContent=c.name;
}

new ResizeObserver(()=>{
  const w=root.clientWidth,h=root.clientHeight;
  renderer.setSize(w,h);cam.aspect=w/h;cam.updateProjectionMatrix();
}).observe(root);

(function loop(){requestAnimationFrame(loop);ctrl.update();renderer.render(scene,cam)})();
</script>
"""


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
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--max-points", type=int, default=300_000)
    args = ap.parse_args()

    port = find_port(args.port)
    if port != args.port:
        print(f"Port {args.port} busy, using {port}")

    print(f"Loading {args.input_dir} ...")
    xyz, rgb, cameras = load_scene(args.input_dir, args.max_points)
    print(f"  {len(xyz):,} pts, {len(cameras)} cams")

    html = build_html(xyz, rgb, cameras)

    with gr.Blocks(title="3D Viewer") as app:
        gr.Markdown(f"### 3D Viewer — `{Path(args.input_dir).name}` ({len(xyz):,} pts, {len(cameras)} cams)")
        with gr.Row():
            with gr.Column(scale=5):
                gr.HTML(html)
            with gr.Column(scale=1, min_width=160):
                gr.Markdown("**Cameras**")
                for c in cameras:
                    if Path(c["image"]).exists():
                        gr.Image(c["image"], label=c["name"], height=80)

    print(f"  http://0.0.0.0:{port}")
    app.launch(server_name="0.0.0.0", server_port=port, share=args.share, quiet=True)


if __name__ == "__main__":
    main()
