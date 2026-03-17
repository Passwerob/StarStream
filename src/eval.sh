#!/usr/bin/env bash
set -euo pipefail

# 用法:
# bash eval_pose_intrinsics.sh \
#   /share/magic_group/aigc/fcr/EventVGGT/data/StreamVGGT_mixed_FSDP/src/output_event/checkpoints-1 \
#   /share/magic_group/aigc/fcr/EventVGGT/data/DL3DV_data/DL3DV/screen-000011

OUT_DIR="${1:?need output dir, e.g. .../output_event/checkpoints-1}"
GT_SCREEN="${2:?need gt screen dir, e.g. .../DL3DV/screen-000011}"

PRED_JSON="${OUT_DIR}/transforms.json"
if [[ ! -f "$PRED_JSON" ]]; then
  echo "[ERR] pred transforms not found: $PRED_JSON"
  exit 1
fi

if [[ -d "${GT_SCREEN}/dense/cam" ]]; then
  GT_CAM_DIR="${GT_SCREEN}/dense/cam"
elif [[ -d "${GT_SCREEN}/cam" ]]; then
  GT_CAM_DIR="${GT_SCREEN}/cam"
else
  echo "[ERR] gt cam dir not found under: ${GT_SCREEN}"
  exit 1
fi

python - "$PRED_JSON" "$GT_CAM_DIR" <<'PY'
import json, glob, os, re, sys
import numpy as np

pred_json, gt_cam_dir = sys.argv[1], sys.argv[2]

def idx_from_pred_name(name):
    # 支持 frame_000000 / frame_00000 两种
    m = re.search(r'frame_(\d+)\.png$', name)
    return int(m.group(1)) if m else None

def idx_from_gt_name(name):
    # GT: 00001.npz -> 评估时映射为 pred_idx=gt_idx-1
    m = re.search(r'(\d+)\.npz$', name)
    return int(m.group(1)) - 1 if m else None

def umeyama(X, Y):
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    S = (Xc.T @ Yc) / len(X)
    U, D, Vt = np.linalg.svd(S)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    var = (Xc**2).sum() / len(X)
    s = np.trace(np.diag(D)) / max(var, 1e-12)
    t = my - s * (R @ mx)
    return s, R, t

def rel(Ti, Tj): return np.linalg.inv(Ti) @ Tj
def rot_deg(R):
    x = np.clip((np.trace(R)-1.0)/2.0, -1.0, 1.0)
    return np.degrees(np.arccos(x))

pred = json.load(open(pred_json, "r"))
pTs, pKs, pmap = [], [], {}
for i, fr in enumerate(pred["frames"]):
    name = os.path.basename(fr["file_path"])
    idx = idx_from_pred_name(name)
    if idx is None: 
        continue
    T = np.array(fr["transform_matrix"], dtype=np.float64)
    fx = float(fr.get("fl_x", pred.get("fl_x", np.nan)))
    fy = float(fr.get("fl_y", pred.get("fl_y", np.nan)))
    cx = float(fr.get("cx",  pred.get("cx",  np.nan)))
    cy = float(fr.get("cy",  pred.get("cy",  np.nan)))
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)
    pmap[idx] = (T, K)

gfiles = sorted(glob.glob(os.path.join(gt_cam_dir, "*.npz")))
gmap = {}
for fp in gfiles:
    idx = idx_from_gt_name(os.path.basename(fp))
    if idx is None:
        continue
    z = np.load(fp)
    gmap[idx] = (z["pose"].astype(np.float64), z["intrinsic"].astype(np.float64))

ids = sorted(set(pmap.keys()) & set(gmap.keys()))
if len(ids) < 2:
    raise RuntimeError(f"matched frames < 2, matched={len(ids)}")

Tp = np.stack([pmap[i][0] for i in ids])
Kp = np.stack([pmap[i][1] for i in ids])
Tg = np.stack([gmap[i][0] for i in ids])
Kg = np.stack([gmap[i][1] for i in ids])

cp, cg = Tp[:, :3, 3], Tg[:, :3, 3]
s, Ra, ta = umeyama(cp, cg)
cp_a = (s * (Ra @ cp.T)).T + ta
ate_rmse = np.sqrt(np.mean(np.sum((cp_a - cg)**2, axis=1)))

rpe_t, rpe_r = [], []
for i in range(len(Tp)-1):
    Tpi, Tpj = Tp[i].copy(), Tp[i+1].copy()
    Tpi[:3,:3] = Ra @ Tpi[:3,:3]; Tpj[:3,:3] = Ra @ Tpj[:3,:3]
    Tpi[:3,3]  = s * (Ra @ Tpi[:3,3]) + ta
    Tpj[:3,3]  = s * (Ra @ Tpj[:3,3]) + ta
    E = np.linalg.inv(rel(Tg[i], Tg[i+1])) @ rel(Tpi, Tpj)
    rpe_t.append(np.linalg.norm(E[:3,3]))
    rpe_r.append(rot_deg(E[:3,:3]))

fx_mae = np.mean(np.abs(Kp[:,0,0] - Kg[:,0,0]))
fy_mae = np.mean(np.abs(Kp[:,1,1] - Kg[:,1,1]))
cx_mae = np.mean(np.abs(Kp[:,0,2] - Kg[:,0,2]))
cy_mae = np.mean(np.abs(Kp[:,1,2] - Kg[:,1,2]))

print(f"matched_frames={len(ids)}")
print(f"ATE_RMSE={ate_rmse:.6f}")
print(f"RPE_trans_mean={np.mean(rpe_t):.6f}")
print(f"RPE_rot_mean_deg={np.mean(rpe_r):.6f}")
print(f"K_MAE fx={fx_mae:.4f} fy={fy_mae:.4f} cx={cx_mae:.4f} cy={cy_mae:.4f}")
PY