'''
How to use:
python see_pointcloud.py /data/fcr/code/event_vggt/Data_Part/StreamVGGT/src/output/evencoder_v2_lyt_perc/screen_1/point_cloud/merged.ply 8000
python see_pointcloud.py 'data_root' 'web_port'
'''
import os
import sys
import struct
import random
import http.server
import socketserver
import shutil

# --- 核心逻辑：读取、降采样并转换为二进制 PLY ---
def convert_and_downsample(src_path, dst_path, target_points=500000):
    print(f"Reading {src_path}...")
    
    header_lines = []
    vertex_count = 0
    header_ended = False
    
    # 第一遍：读取头部信息
    with open(src_path, 'r') as f:
        for line in f:
            if not header_ended:
                header_lines.append(line)
                if line.startswith("element vertex"):
                    vertex_count = int(line.split()[-1])
                if line.strip() == "end_header":
                    header_ended = True
                continue
            else:
                break
    
    print(f"Original vertex count: {vertex_count}")
    
    # 计算采样率
    ratio = min(1.0, target_points / vertex_count) if vertex_count > 0 else 1.0
    print(f"Downsampling ratio: {ratio:.4f} (Target: {target_points})")
    
    points = []
    
    # 第二遍：读取并随机采样
    with open(src_path, 'r') as f:
        in_header = True
        for line in f:
            if in_header:
                if line.strip() == "end_header":
                    in_header = False
                continue
            
            if random.random() < ratio:
                parts = line.strip().split()
                if len(parts) >= 6:
                    # 解析 x, y, z, r, g, b
                    pt = [float(parts[0]), float(parts[1]), float(parts[2]),
                          int(parts[3]), int(parts[4]), int(parts[5])]
                    points.append(pt)
    
    final_count = len(points)
    print(f"Final sampled count: {final_count}")
    
    # 写入二进制 PLY (Binary Little Endian)，浏览器加载速度快 10 倍以上
    print(f"Writing binary PLY to {dst_path}...")
    with open(dst_path, 'wb') as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {final_count}\n".encode('ascii'))
        f.write(b"property float x\n")
        f.write(b"property float y\n")
        f.write(b"property float z\n")
        f.write(b"property uchar red\n")
        f.write(b"property uchar green\n")
        f.write(b"property uchar blue\n")
        f.write(b"end_header\n")
        
        for pt in points:
            # 格式: 3个float (xyz) + 3个uchar (rgb) = 15 bytes
            f.write(struct.pack('<fffBBB', pt[0], pt[1], pt[2], pt[3], pt[4], pt[5]))

# --- 生成 HTML 前端页面 ---
def create_html(web_dir):
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>3D Point Cloud Viewer</title>
    <style>
        body { margin: 0; overflow: hidden; background-color: #fff; }
        #info {
            position: absolute;
            top: 10px;
            width: 100%;
            text-align: center;
            color: black;
            font-family: sans-serif;
            pointer-events: none;
            z-index: 100;
        }
    </style>
    <!-- 引入 Three.js -->
    <script async src="https://unpkg.com/es-module-shims@1.6.3/dist/es-module-shims.js"></script>
    <script type="importmap">
        {
            "imports": {
                "three": "https://unpkg.com/three@0.154.0/build/three.module.js",
                "three/addons/": "https://unpkg.com/three@0.154.0/examples/jsm/"
            }
        }
    </script>
</head>
<body>
    <div id="info">
        <b>StreamVGGT Point Cloud Viewer</b><br/>
        Left Click: Rotate | Right Click: Pan | Scroll: Zoom<br/>
        WASD: Move Camera (Fly Mode)
    </div>
    <script type="module">
        import * as THREE from 'three';
        import { TrackballControls } from 'three/addons/controls/TrackballControls.js';
        import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

        let camera, scene, renderer, controls;
        const keys = { w: false, a: false, s: false, d: false };

        init();
        animate();

        function init() {
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0xffffff);

            // 相机设置
            camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.01, 100);
            // camera.position.set(0, 0, 2);
            camera.position.set(2, 2, 2); // 设为对角位置 (Diagonal View)

            // 渲染器设置
            renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setPixelRatio(window.devicePixelRatio);
            renderer.setSize(window.innerWidth, window.innerHeight);
            document.body.appendChild(renderer.domElement);

            // 控制器
            controls = new TrackballControls(camera, renderer.domElement);
            controls.rotateSpeed = 2.0;
            controls.zoomSpeed = 1.2;
            controls.panSpeed = 0.8;
            controls.dynamicDampingFactor = 0.3;

            // 加载 PLY 模型
            const loader = new PLYLoader();
            console.log("Loading model...");
            loader.load('./model.ply', function (geometry) {
                
                // 自动居中模型
                geometry.computeBoundingBox();
                const center = new THREE.Vector3();
                geometry.boundingBox.getCenter(center);
                geometry.translate(-center.x, -center.y, -center.z);
                
                // 材质设置
                let material;
                if (geometry.hasAttribute('color')) {
                    material = new THREE.PointsMaterial({ size: 0.005, vertexColors: true });
                } else {
                    material = new THREE.PointsMaterial({ size: 0.005, color: 0x00ff00 });
                }

                const particles = new THREE.Points(geometry, material);
                particles.scale.set(-1, -1, -1); // X, Y, Z轴全部反转 (Invert All Axes)
                scene.add(particles);

                console.log("Loaded point cloud");

            }, undefined, (error) => {
                console.error(error);
            });

            window.addEventListener('resize', onWindowResize);
            
            window.addEventListener('keydown', (e) => {
                const key = e.key.toLowerCase();
                if (keys.hasOwnProperty(key)) keys[key] = true;
            });
            window.addEventListener('keyup', (e) => {
                const key = e.key.toLowerCase();
                if (keys.hasOwnProperty(key)) keys[key] = false;
            });
        }

        function onWindowResize() {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
            controls.handleResize();
        }

        function animate() {
            requestAnimationFrame(animate);

            // WASD Movement
            if (keys.w || keys.a || keys.s || keys.d) {
                const moveSpeed = 0.05; 
                const dir = new THREE.Vector3();
                camera.getWorldDirection(dir);
                const right = new THREE.Vector3();
                right.crossVectors(dir, camera.up).normalize();

                if (keys.w) {
                    camera.position.addScaledVector(dir, moveSpeed);
                    controls.target.addScaledVector(dir, moveSpeed);
                }
                if (keys.s) {
                    camera.position.addScaledVector(dir, -moveSpeed);
                    controls.target.addScaledVector(dir, -moveSpeed);
                }
                if (keys.a) {
                    camera.position.addScaledVector(right, -moveSpeed);
                    controls.target.addScaledVector(right, -moveSpeed);
                }
                if (keys.d) {
                    camera.position.addScaledVector(right, moveSpeed);
                    controls.target.addScaledVector(right, moveSpeed);
                }
            }

            controls.update();
            renderer.render(scene, camera);
        }
    </script>
</body>
</html>
"""
    with open(os.path.join(web_dir, "index.html"), "w") as f:
        f.write(html_content)

def main():
    if len(sys.argv) < 2:
        print("Usage: python serve_viz.py <ply_file> [port]")
        sys.exit(1)

    ply_path = sys.argv[1]
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    
    web_dir = "web_viewer"
    if os.path.exists(web_dir):
        shutil.rmtree(web_dir)
    os.makedirs(web_dir)
    
    print(f"Preparing visualization for {ply_path}...")
    # 转换为二进制并降采样，确保浏览器能流畅加载
    convert_and_downsample(ply_path, os.path.join(web_dir, "model.ply"))
    create_html(web_dir)
    
    print(f"Starting server on port {port}...")
    os.chdir(web_dir)
    
    # 启动服务器
    with socketserver.TCPServer(("", port), http.server.SimpleHTTPRequestHandler) as httpd:
        print(f"Serving at http://localhost:{port}")
        httpd.serve_forever()

if __name__ == "__main__":
    main()