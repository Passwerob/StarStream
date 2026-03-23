"""Visualize PLY point clouds from inference output.

Usage:
    # Interactive HTML (open in browser, works on remote server)
    python visualize_ply.py /path/to/point_cloud/ --mode html

    # Open3D interactive window (needs display / X-forwarding)
    python visualize_ply.py /path/to/merged.ply --mode open3d

    # Static image render
    python visualize_ply.py /path/to/point_cloud/ --mode image

    # Only specific frames
    python visualize_ply.py /path/to/point_cloud/ --frames 0 5 10

    # Filter by confidence
    python visualize_ply.py /path/to/merged.ply --conf-thresh 0.5
"""

import argparse
import numpy as np
from pathlib import Path


def load_ply(path: str):
    """Load an ASCII PLY file with x,y,z,r,g,b,confidence."""
    path = Path(path)
    with open(path, "r") as f:
        header_lines = []
        n_vertices = 0
        while True:
            line = f.readline().strip()
            header_lines.append(line)
            if line.startswith("element vertex"):
                n_vertices = int(line.split()[-1])
            if line == "end_header":
                break

        data = np.loadtxt(f, max_rows=n_vertices)

    xyz = data[:, :3].astype(np.float32)
    rgb = data[:, 3:6].astype(np.uint8)
    conf = data[:, 6].astype(np.float32) if data.shape[1] > 6 else np.ones(len(xyz), dtype=np.float32)
    return xyz, rgb, conf


def load_ply_dir(directory: str, frames=None):
    """Load all frame PLY files from a directory."""
    d = Path(directory)
    if d.is_file():
        return load_ply(d)

    files = sorted(d.glob("frame_*.ply"))
    if not files:
        merged = d / "merged.ply"
        if merged.exists():
            return load_ply(merged)
        raise FileNotFoundError(f"No PLY files found in {d}")

    if frames is not None:
        files = [f for f in files if any(f.stem.endswith(f"{i:06d}") for i in frames)]

    all_xyz, all_rgb, all_conf = [], [], []
    for f in files:
        print(f"Loading {f.name} ...")
        xyz, rgb, conf = load_ply(f)
        all_xyz.append(xyz)
        all_rgb.append(rgb)
        all_conf.append(conf)

    return np.concatenate(all_xyz), np.concatenate(all_rgb), np.concatenate(all_conf)


def filter_and_downsample(xyz, rgb, conf, conf_thresh=0.0, max_points=500_000, remove_outliers=True):
    """Filter by confidence, remove outliers, and downsample."""
    mask = conf >= conf_thresh
    xyz, rgb, conf = xyz[mask], rgb[mask], conf[mask]
    print(f"After confidence filter (>={conf_thresh}): {len(xyz):,} points")

    if remove_outliers and len(xyz) > 100:
        center = np.median(xyz, axis=0)
        dists = np.linalg.norm(xyz - center, axis=1)
        p95 = np.percentile(dists, 95)
        inlier = dists < p95 * 2
        xyz, rgb, conf = xyz[inlier], rgb[inlier], conf[inlier]
        print(f"After outlier removal: {len(xyz):,} points")

    if len(xyz) > max_points:
        idx = np.random.default_rng(42).choice(len(xyz), max_points, replace=False)
        idx.sort()
        xyz, rgb, conf = xyz[idx], rgb[idx], conf[idx]
        print(f"Downsampled to {max_points:,} points")

    return xyz, rgb, conf


def vis_html(xyz, rgb, out_path, point_size=1.5):
    """Create an interactive HTML viewer using plotly."""
    import plotly.graph_objects as go

    colors = [f"rgb({r},{g},{b})" for r, g, b in rgb]

    fig = go.Figure(data=[go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
        mode="markers",
        marker=dict(size=point_size, color=colors, opacity=0.9),
        hoverinfo="skip",
    )])

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            bgcolor="rgb(20,20,20)",
        ),
        paper_bgcolor="rgb(20,20,20)",
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(text=f"Point Cloud ({len(xyz):,} pts)", font=dict(color="white", size=14)),
    )

    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"Saved interactive HTML to {out_path}")
    print(f"Open in browser: file://{Path(out_path).resolve()}")


def vis_open3d(xyz, rgb):
    """Interactive Open3D viewer."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64) / 255.0)

    print("Launching Open3D viewer (press Q to quit, mouse to rotate) ...")
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Point Cloud Viewer",
        width=1280,
        height=720,
        point_show_normal=False,
    )


def vis_image(xyz, rgb, out_path, elev=30, azim=45):
    """Render a static image using matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(14, 10), facecolor="black")
    ax = fig.add_subplot(111, projection="3d", facecolor="black")

    colors_norm = rgb.astype(np.float32) / 255.0
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors_norm, s=0.1, depthshade=True)

    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(f"Point Cloud ({len(xyz):,} pts)", color="white", fontsize=12)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=200, bbox_inches="tight", facecolor="black")
    plt.close()
    print(f"Saved image to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize PLY point clouds")
    parser.add_argument("input", help="PLY file or directory containing PLY files")
    parser.add_argument("--mode", choices=["html", "open3d", "image"], default="html",
                        help="Visualization mode (default: html)")
    parser.add_argument("--output", "-o", default=None, help="Output path (auto-generated if not set)")
    parser.add_argument("--frames", nargs="+", type=int, default=None,
                        help="Specific frame indices to load (e.g. --frames 0 5 10)")
    parser.add_argument("--conf-thresh", type=float, default=0.0,
                        help="Minimum confidence threshold (default: 0)")
    parser.add_argument("--max-points", type=int, default=500_000,
                        help="Max points to render (default: 500000)")
    parser.add_argument("--no-outlier-removal", action="store_true",
                        help="Disable outlier removal")
    parser.add_argument("--point-size", type=float, default=1.5,
                        help="Point size for HTML mode (default: 1.5)")
    args = parser.parse_args()

    inp = Path(args.input)
    print(f"Loading from {inp} ...")
    xyz, rgb, conf = load_ply_dir(str(inp), frames=args.frames)
    print(f"Loaded {len(xyz):,} points")

    xyz, rgb, conf = filter_and_downsample(
        xyz, rgb, conf,
        conf_thresh=args.conf_thresh,
        max_points=args.max_points,
        remove_outliers=not args.no_outlier_removal,
    )

    if args.output:
        out = Path(args.output)
    else:
        stem = inp.stem if inp.is_file() else inp.name
        suffix = {"html": ".html", "open3d": "", "image": ".png"}[args.mode]
        out = inp.parent / f"{stem}_vis{suffix}" if suffix else None

    if args.mode == "html":
        vis_html(xyz, rgb, out, point_size=args.point_size)
    elif args.mode == "open3d":
        vis_open3d(xyz, rgb)
    elif args.mode == "image":
        vis_image(xyz, rgb, out)


if __name__ == "__main__":
    main()
