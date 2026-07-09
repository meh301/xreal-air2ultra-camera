#!/usr/bin/env python3
"""Xreal Air 2 Ultra モノクロカメラ フレーム デスクランブラ
使い方: python3 xreal_descramble.py input.png output.png
入力: 640x480 グレースケール(生フレーム307200バイトを行優先で並べたもの)
出力: 480x640 の復元画像
アルゴリズム出典: github.com/mazeasdamien/myXreal (stereo_camera.cpp)
"""
import sys
import numpy as np
from PIL import Image

REORDER = [119,54,21,0,108,22,51,63,93,99,67,7,32,112,52,43,
14,35,75,116,64,71,44,89,18,88,26,61,70,56,90,79,
87,120,81,101,121,17,72,31,53,124,127,113,111,36,48,
19,37,83,126,74,109,5,84,41,76,30,110,29,12,115,28,
102,105,62,103,20,3,68,49,77,117,125,106,60,69,98,9,
16,78,47,40,2,118,34,13,50,46,80,85,66,42,123,122,
96,11,25,97,39,6,86,1,8,82,92,59,104,24,15,73,65,
38,58,10,23,33,55,57,107,100,94,27,95,45,91,4,114]
NB, BS = 128, 2400  # 128ブロック x 2400バイト = 307200

def build_lut(is_right):
    # 横方向の向きは実景と照合して検証済み (2026-07): myXreal 由来の元テーブル
    # ((c, r) / (639-c, 479-r)) は左右反転した像を出していたため x を反転している
    lut = np.zeros(NB*BS, np.int32)
    p_y = p_x = idx = 0
    for _ in range(NB):
        off = 0
        while off < BS:
            seg = min(p_y + (BS - off), 640) - p_y
            for k in range(seg):
                r, c = p_x, p_y + k
                y, x = (c, 479 - r) if is_right else (639 - c, r)
                lut[idx] = y * 480 + x
                idx += 1
            off += seg; p_y += seg
            if p_y >= 640:
                p_x += 1; p_y = 0
    return lut

def descramble(raw, is_right=True):
    """raw: 307200バイトのuint8配列 -> (640,480)画像"""
    raw = np.asarray(raw, np.uint8).reshape(-1)
    # 同期検出: 先頭128バイトの合計が最小のブロック
    scores = [raw[i*BS:i*BS+128].astype(int).sum() for i in range(NB)]
    align = REORDER.index(int(np.argmin(scores)))
    out = np.zeros(640*480, np.uint8)
    lut = build_lut(is_right)
    for t in range(NB):
        b = REORDER[(align + t) % NB]
        out[lut[t*BS:(t+1)*BS]] = raw[b*BS:(b+1)*BS]
    return out.reshape(640, 480)

if __name__ == '__main__':
    src, dst = sys.argv[1], sys.argv[2]
    raw = np.array(Image.open(src).convert('L'))
    img = descramble(raw, is_right=True)  # 上下逆なら is_right=False
    Image.fromarray(img).save(dst)
    print('saved', dst)
