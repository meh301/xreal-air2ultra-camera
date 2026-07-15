#!/usr/bin/env python
"""Produce the VOCA-compressed variant of a replay pack.

Usage:
    compress_voca.py <pack_dir> <out_pack_dir> [--bitrate 500k] [--keep-temp]

Per camera, reconstructs PNG frames from frames.raw, runs the exact VOCA
two-pass x264 encode, decodes the mp4 back to gray frames, and writes a new
pack (compressed=1) with the recompressed pixels. Everything except
frames.raw/meta.txt is copied verbatim from the input pack.

Encoder framerate follows the VOCA convention: euroc/tumvi 20 fps,
msd (MOO) 30 fps -- taken from the dataset field of meta.txt, not from the
measured fps. Pass-1 output goes to NUL (Windows).
"""

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from pack_common import (DATASET_FPS, die, info, open_frames_raw,
                         read_frames_csv, read_meta, run_cmd, warn,
                         write_meta)

X264_OPTS = ("partitions=p8x8,p4x4,i8x8:keyint=1000:me=umh:merange=64"
             ":subme=6:bframes=0:ref=1")


def _ffmpeg():
    exe = shutil.which("ffmpeg")
    if not exe:
        die("ffmpeg not found on PATH -- install it (e.g. gyan.dev build) "
            "and retry")
    return exe


def _encode_decode_cam(ffmpeg, cam, png_dir, n_frames, fps, bitrate, tmp, W, H):
    """VOCA two-pass x264 on one camera's PNG dir; returns decoded (n,H,W)."""
    lst = tmp / f"{cam}_list.txt"
    with open(lst, "w", encoding="utf-8", newline="\n") as f:
        for i in range(n_frames):
            f.write(f"file '{(png_dir / f'{i:06d}.png').as_posix()}'\n")
    passlog = tmp / f"x264_{cam}"
    mp4 = tmp / f"{cam}.mp4"
    common = [ffmpeg, "-y", "-f", "concat", "-safe", "0",
              "-r", str(fps), "-i", str(lst),
              "-c:v", "libx264", "-b:v", bitrate, "-pix_fmt", "yuv420p",
              "-r", str(fps), "-vf", "setpts=PTS-STARTPTS",
              "-x264opts", X264_OPTS, "-passlogfile", str(passlog)]
    info(f"  [{cam}] x264 pass 1 ...")
    run_cmd(common + ["-pass", "1", "-an", "-f", "null", "NUL"],
            cwd=tmp, what=f"{cam} x264 pass 1")
    info(f"  [{cam}] x264 pass 2 ...")
    run_cmd(common + ["-pass", "2", "-an", "-f", "mp4", str(mp4)],
            cwd=tmp, what=f"{cam} x264 pass 2")
    info(f"  [{cam}] encoded {mp4.stat().st_size / 1e6:.2f} MB "
         f"({n_frames} frames @ {fps} fps, {bitrate})")

    gray = tmp / f"{cam}.gray"
    run_cmd([ffmpeg, "-y", "-i", str(mp4), "-f", "rawvideo",
             "-pix_fmt", "gray", str(gray)],
            cwd=tmp, what=f"{cam} decode")
    size = gray.stat().st_size
    if size != n_frames * W * H:
        die(f"{cam}: decoded {size // (W * H)} frames, expected {n_frames} "
            f"-- encoder dropped/duplicated frames, pack would be corrupt")
    return np.memmap(gray, dtype=np.uint8, mode="r", shape=(n_frames, H, W))


def compress_pack(pack_dir, out_dir, bitrate="500k", keep_temp=False,
                  force=False):
    pack = Path(pack_dir)
    out = Path(out_dir)
    if pack.resolve() == out.resolve():
        die("out_pack_dir must differ from the input pack_dir")
    meta = read_meta(pack)
    if meta["compressed"] and not force:
        die(f"{pack} already has compressed=1 (pass --force to re-compress "
            f"anyway)")
    W, H, n = meta["W"], meta["H"], meta["n_frames"]
    if W % 2 or H % 2:
        die(f"{W}x{H} frames: yuv420p needs even dimensions")
    fps = int(DATASET_FPS[meta["dataset"]])
    frames = read_frames_csv(pack)
    if len(frames) != n:
        die(f"frames.csv has {len(frames)} lines, meta says n_frames={n}")
    raw = open_frames_raw(pack, meta)
    ffmpeg = _ffmpeg()

    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "_voca_tmp"
    tmp.mkdir(exist_ok=True)

    decoded = {}
    for ci, cam in enumerate(("left", "right")):
        png_dir = tmp / cam
        png_dir.mkdir(exist_ok=True)
        info(f"  [{cam}] writing {n} PNGs ...")
        for i in range(n):
            if not cv2.imwrite(str(png_dir / f"{i:06d}.png"), raw[i, ci]):
                die(f"failed to write PNG {i} for {cam}")
        decoded[cam] = _encode_decode_cam(ffmpeg, cam, png_dir, n, fps,
                                          bitrate, tmp, W, H)

    info("  writing compressed frames.raw ...")
    with open(out / "frames.raw", "wb") as f:
        for i in range(n):
            f.write(decoded["left"][i].tobytes())
            f.write(decoded["right"][i].tobytes())

    # quick fidelity stat on a few sampled frames (not a gate, just a report)
    idxs = np.unique(np.linspace(0, n - 1, min(5, n)).astype(int))
    mad = float(np.mean([np.mean(np.abs(
        decoded["left"][i].astype(np.int16) - raw[i, 0].astype(np.int16)))
        for i in idxs]))
    info(f"  mean |orig - recompressed| over {len(idxs)} sampled left "
         f"frames: {mad:.2f} gray levels")

    for name in ("calib.txt", "imu.bin", "frames.csv", "gt.tum"):
        src = pack / name
        if not src.exists():
            die(f"input pack incomplete: missing {name}")
        shutil.copyfile(src, out / name)
    meta_out = dict(meta)
    meta_out["compressed"] = 1
    write_meta(out, meta_out)   # last: pack-complete marker

    # free memmaps before deleting their backing files
    for cam in list(decoded):
        del decoded[cam]
    del raw
    if keep_temp:
        info(f"  temp kept at {tmp}")
    else:
        for cam in ("left", "right"):
            shutil.rmtree(tmp / cam, ignore_errors=True)
            (tmp / f"{cam}.gray").unlink(missing_ok=True)
            (tmp / f"{cam}_list.txt").unlink(missing_ok=True)
            for log in tmp.glob(f"x264_{cam}*"):
                log.unlink(missing_ok=True)
        # keep the mp4s: they ARE the compressed representation of record
        info(f"  mp4s kept at {tmp} (PNG/raw temps removed)")
    info(f"compressed pack complete -> {out}")
    return meta_out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pack_dir")
    ap.add_argument("out_pack_dir")
    ap.add_argument("--bitrate", default="500k")
    ap.add_argument("--keep-temp", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="allow re-compressing an already-compressed pack")
    args = ap.parse_args()
    compress_pack(args.pack_dir, args.out_pack_dir, bitrate=args.bitrate,
                  keep_temp=args.keep_temp, force=args.force)


if __name__ == "__main__":
    main()
