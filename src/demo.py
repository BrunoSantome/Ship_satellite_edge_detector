"""End-to-end demo of the onboard pipeline (no training): prints what happens at every step.

  [1] input frame -> [2] tiling -> [3] detection per tile -> [4] merge (NMS)
  -> [5] encode alert -> [6] downlink -> [7] ground decode -> [8] output

Steps 3-4 are written out here (instead of calling tiling.detect_tiled) so each one can be printed.

Run:
  python src/demo.py            # frame 0, same as the README
  python src/demo.py --seed 3   # a different random frame
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from telemetry import DET, HEADER, decode, encode, print_alert
from tiling import OVERLAP, TILE, build_frame, detect, nms, tile_starts


def step(n, title):
    print(f"\n[{n}] {title}")


def main(args):
    from ultralytics import YOLO

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)

    # ------------------------------------------------------------ satellite side
    step(1, "INPUT: one large camera frame")
    frame, gt, _ = build_frame(Path(args.data), np.random.default_rng(args.seed))
    print(f"    {frame.shape[1]}x{frame.shape[0]} px, {frame.nbytes / 1e6:.1f} MB raw "
          f"(synthetic: 9 test images stitched 3x3, {len(gt)} labelled ships)")

    step(2, f"TILING: {TILE}x{TILE} px tiles overlapping by {OVERLAP} px (no shrinking)")
    starts = tile_starts(frame.shape[0])
    print(f"    tile start offsets on each axis: {starts} -> {len(starts) ** 2} tiles")

    step(3, "DETECTION: YOLOv8s runs once per tile")
    detect(model, frame[:TILE, :TILE])  # warm-up, so the timing below is fair
    boxes, scores = [], []
    t0 = time.perf_counter()
    for y0 in starts:
        per_tile = []
        for x0 in starts:
            b, s = detect(model, frame[y0:y0 + TILE, x0:x0 + TILE])
            b[:, [0, 2]] += x0  # tile coordinates -> frame coordinates
            b[:, [1, 3]] += y0
            boxes.append(b)
            scores.append(s)
            per_tile.append(len(b))
        print(f"    row of tiles at y={y0:>4}: detections per tile {per_tile}")
    print(f"    {len(starts) ** 2} model runs in {time.perf_counter() - t0:.1f} s on this CPU")

    step(4, "MERGE: remove duplicates from overlapping tiles (NMS)")
    boxes, scores = np.concatenate(boxes), np.concatenate(scores)
    keep = nms(boxes, scores)
    print(f"    {len(boxes)} boxes from all tiles -> {len(keep)} after removing {len(boxes) - len(keep)} duplicates")
    boxes, scores = boxes[keep], scores[keep]

    step(5, "ENCODE: pack detections into a binary alert")
    packet = encode(frame_id=0, timestamp=int(time.time()), boxes=boxes, scores=scores)
    (out / "demo_alert.bin").write_bytes(packet)
    print(f"    {len(boxes)} detections -> {len(packet)} bytes "
          f"({HEADER.size} header + {len(boxes)} x {DET.size} + 2 CRC), saved to {out / 'demo_alert.bin'}")
    print(f"    header bytes : {packet[:HEADER.size].hex(' ')}")
    if len(boxes):
        print(f"    first ship   : {packet[HEADER.size:HEADER.size + DET.size].hex(' ')}")

    # ------------------------------------------------------------ link
    step(6, "DOWNLINK (simulated): only the alert is sent, not the image")
    jpeg_bytes = len(cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])[1])
    for rate in (9600, 1200):
        print(f"    @ {rate / 1000:g} kbps: alert {len(packet) * 8 / rate:.2f} s   "
              f"vs JPEG image ({jpeg_bytes / 1e6:.2f} MB) {jpeg_bytes * 8 / rate / 60:.1f} min")

    # ------------------------------------------------------------ ground side
    step(7, "GROUND: check CRC, decode")
    msg = decode(packet)  # raises PacketError if the bytes were corrupted
    print("    CRC ok, magic ok, version 1")
    print_alert(msg)

    step(8, "OUTPUT: decoded boxes drawn on the frame (demo only: the ground has no image)")
    img = frame.copy()
    for d in msg["detections"]:
        x0, y0, x1, y1 = map(int, d["box"])
        cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 0), 3)
    cv2.imwrite(str(out / "demo_ground.jpg"), img)
    print(f"    saved {out / 'demo_ground.jpg'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="models/best.pt")
    p.add_argument("--data", default="data/val_test_subset/test")
    p.add_argument("--out", default="results")
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
