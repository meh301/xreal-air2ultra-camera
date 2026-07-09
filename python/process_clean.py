#!/usr/bin/env python3
"""キャプチャ済み camN_XXXX.pgm をデスクランブルし、縦縞FPNを除去して綺麗な画像を出力する。

使い方: python3 process_clean.py <capture_dir> <out_dir>
  <capture_dir>: xreal_cam が出力した cam0_*.pgm / cam1_*.pgm
  <out_dir>    : clean_camN_XXXX.png と stereo_feed.mp4 を出力

処理:
  1. xreal_descramble.descramble でブロックスクランブル解除 (480x640)
  2. カメラ毎に全フレームから列FPN(縦縞)を推定:
     各フレームの水平ハイパス成分の行方向メディアン → フレーム間メディアン
  3. FPN減算 + 軽いコントラスト補正
"""
import sys, os, glob, re
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xreal_descramble import descramble

def read_pgm(p):
    with open(p, "rb") as f:
        assert f.readline().strip() == b"P5"
        w, h = map(int, f.readline().split())
        f.readline()
        return np.frombuffer(f.read(), np.uint8).reshape(h, w)

def estimate_col_fpn(frames):
    """frames: list of (H,W) float32 -> (W,) 固定縦縞パターン"""
    stripes = []
    for f in frames:
        smooth = cv2.blur(f, (31, 1))          # 水平方向のみ平滑化
        hp = f - smooth                        # 水平ハイパス = 縦縞 + 細部
        stripes.append(np.median(hp, axis=0))  # 行方向メディアンで細部を潰す
    return np.median(np.stack(stripes), axis=0)

def main(cap_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cams = {}
    for cam in (0, 1):
        files = sorted(glob.glob(os.path.join(cap_dir, f"cam{cam}_*.pgm")),
                       key=lambda p: int(re.search(r"_(\d+)\.pgm", p).group(1)))
        # 先頭の真っ黒フレーム(起動直後)は除外
        frames = []
        for p in files:
            raw = read_pgm(p)
            if raw.mean() < 5:
                continue
            img = descramble(raw.reshape(-1), is_right=(cam == 0))
            frames.append(img.astype(np.float32))
        if frames:
            cams[cam] = frames

    for cam, frames in cams.items():
        fpn = estimate_col_fpn(frames)
        for i, f in enumerate(frames):
            clean = np.clip(f - fpn[None, :], 0, 255)
            # 微弱な行バンディングも同様に除去
            smooth = cv2.blur(clean, (1, 31))
            row_hp = np.median(clean - smooth, axis=1)
            clean = np.clip(clean - row_hp[:, None], 0, 255).astype(np.uint8)
            clean = cv2.createCLAHE(2.0, (8, 8)).apply(clean)
            cv2.imwrite(os.path.join(out_dir, f"clean_cam{cam}_{i:04d}.png"), clean)
        print(f"cam{cam}: {len(frames)} frames cleaned")

    # ステレオ並置ビデオ
    n = min(len(v) for v in cams.values()) if len(cams) == 2 else 0
    if n:
        h, w = 640, 480
        vw = cv2.VideoWriter(os.path.join(out_dir, "stereo_feed.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), 15, (w * 2, h))
        for i in range(n):
            a = cv2.imread(os.path.join(out_dir, f"clean_cam0_{i:04d}.png"), 0)
            b = cv2.imread(os.path.join(out_dir, f"clean_cam1_{i:04d}.png"), 0)
            # cam1 = 左カメラ (実機検証済み) を左側に
            vw.write(cv2.cvtColor(np.hstack([b, a]), cv2.COLOR_GRAY2BGR))
        vw.release()
        print(f"stereo_feed.mp4: {n} pairs")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
