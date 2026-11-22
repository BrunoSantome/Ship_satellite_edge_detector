"""Block 2: tiled inference on a large frame vs naive resize.

Onboard, the camera produces frames much larger than the 768x768 the detector was trained on.
Two ways to feed such a frame to the model:
  1. RESIZE the whole frame to 768: one model run, but small ships shrink to a few pixels.
  2. TILE it into overlapping 768 crops, detect per tile, shift boxes back, merge duplicates (NMS).
     More model runs, but every pixel stays at training resolution.

We have no real large frame, so we build a SYNTHETIC one: 9 test images in a 3x3 grid (2304x2304).
Seams are visible and the scenes are unrelated; it only serves to compare the two methods on the
same input, scored against the known ground-truth boxes.

Run:
  python src/tiling.py              # uses models/best.pt and data/val_test_subset/test
  python src/tiling.py --frames 10  # totals over 10 frames built from disjoint test images
  python src/tiling.py --selftest   # checks tiling + NMS logic, no model or images needed
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

TILE = 768        # = training size, so each tile looks exactly like a training image
OVERLAP = 256     # > 95th-percentile ship length (198 px): such ships appear whole in some tile
CONF = 0.25       # same confidence threshold for both methods
NMS_IOU = 0.5     # boxes overlapping more than this are treated as the same ship
MATCH_IOU = 0.5   # a prediction counts as correct if IoU with a ground-truth box >= 0.5
GRID = 3          # 3x3 images -> 2304x2304 frame
MIN_SHIP_IMAGES = 6
SIZE_BUCKETS = [(0, 20), (20, 50), (50, 10_000)]  # ship long side in px, for recall by size


# ---------------------------------------------------------------- geometry

def tile_starts(length, tile=TILE, overlap=OVERLAP):
    """Start offsets along one axis so tiles cover [0, length) and neighbours share >= overlap px."""
    if length <= tile:
        return [0]
    stride = tile - overlap
    starts = list(range(0, length - tile, stride))
    starts.append(length - tile)  # last tile flush with the frame edge, never past it
    return starts


def box_iou(box, boxes):
    """IoU of one xyxy box against an (N, 4) array of xyxy boxes."""
    ix0 = np.maximum(box[0], boxes[:, 0])
    iy0 = np.maximum(box[1], boxes[:, 1])
    ix1 = np.minimum(box[2], boxes[:, 2])
    iy1 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    area_a = (box[2] - box[0]) * (box[3] - box[1])
    area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area_a + area_b - inter + 1e-9)


def nms(boxes, scores, iou_thr=NMS_IOU):
    """Greedy NMS: keep the most confident box, drop everything overlapping it, repeat."""
    order = np.argsort(-scores)
    keep = []
    while order.size:
        best = order[0]
        keep.append(best)
        rest = order[1:]
        order = rest[box_iou(boxes[best], boxes[rest]) < iou_thr]
    return np.array(keep, dtype=int)


# ---------------------------------------------------------------- data

def load_gt(label_path, w, h):
    """YOLO label file (class cx cy w h, normalised) -> (N, 4) pixel xyxy."""
    rows = []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            if line.strip():
                _, cx, cy, bw, bh = map(float, line.split())
                rows.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(rows, dtype=np.float32).reshape(-1, 4)


def build_frame(split_dir, rng, used=frozenset()):
    """Stitch GRID x GRID random test images into one frame; shift their GT boxes accordingly.
    `used` = images placed in earlier frames, excluded so no ship is counted twice across frames."""
    images = [p for p in sorted((split_dir / "images").glob("*.jpg")) if p.name not in used]
    # Random but not cherry-picked: redraw until enough images contain ships
    while True:
        picks = rng.choice(len(images), GRID * GRID, replace=False)
        chosen = [images[i] for i in picks]
        gts = [load_gt(split_dir / "labels" / f"{p.stem}.txt", TILE, TILE) for p in chosen]
        if sum(len(g) > 0 for g in gts) >= MIN_SHIP_IMAGES:
            break

    frame = np.zeros((GRID * TILE, GRID * TILE, 3), dtype=np.uint8)
    all_gt = []
    for k, (path, gt) in enumerate(zip(chosen, gts)):
        r, c = divmod(k, GRID)
        frame[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE] = cv2.imread(str(path))
        all_gt.append(gt + np.array([c * TILE, r * TILE, c * TILE, r * TILE], dtype=np.float32))
    return frame, np.concatenate(all_gt), [p.name for p in chosen]


# ---------------------------------------------------------------- detection

def detect(model, img):
    """Run the detector on one image (BGR numpy). Returns (N, 4) xyxy boxes and (N,) scores."""
    r = model.predict(np.ascontiguousarray(img), imgsz=TILE, conf=CONF, verbose=False)[0]
    return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()


def detect_tiled(model, frame):
    h, w = frame.shape[:2]
    boxes, scores, n_tiles = [np.zeros((0, 4))], [np.zeros(0)], 0
    for y0 in tile_starts(h):
        for x0 in tile_starts(w):
            b, s = detect(model, frame[y0:y0 + TILE, x0:x0 + TILE])
            b[:, [0, 2]] += x0  # tile coordinates -> frame coordinates
            b[:, [1, 3]] += y0
            boxes.append(b)
            scores.append(s)
            n_tiles += 1
    boxes, scores = np.concatenate(boxes), np.concatenate(scores)
    # The same ship seen in two overlapping tiles gives two boxes: merge across the whole frame
    keep = nms(boxes, scores)
    return boxes[keep], scores[keep], n_tiles


def detect_resized(model, frame):
    scale = frame.shape[0] / TILE
    small = cv2.resize(frame, (TILE, TILE), interpolation=cv2.INTER_AREA)
    boxes, scores = detect(model, small)
    return boxes * scale, scores, 1


# ---------------------------------------------------------------- scoring

def match(pred, scores, gt, iou_thr=MATCH_IOU):
    """Greedy matching, most confident prediction first. Returns TP flags (per pred), matched flags (per GT)."""
    tp = np.zeros(len(pred), dtype=bool)
    matched = np.zeros(len(gt), dtype=bool)
    for i in np.argsort(-scores):
        if len(gt) == 0:
            break
        ious = box_iou(pred[i], gt)
        ious[matched] = 0  # each GT ship can only be found once
        j = int(np.argmax(ious))
        if ious[j] >= iou_thr:
            tp[i] = matched[j] = True
    return tp, matched


def summarise(name, pred, scores, gt, seconds, n_tiles):
    tp, matched = match(pred, scores, gt)
    long_side = np.maximum(gt[:, 2] - gt[:, 0], gt[:, 3] - gt[:, 1])
    by_size = {}
    for lo, hi in SIZE_BUCKETS:
        in_bucket = (long_side >= lo) & (long_side < hi)
        by_size[f"{lo}-{hi}px"] = [int(matched[in_bucket].sum()), int(in_bucket.sum())]  # [found, total]
    return {
        "method": name, "model_runs": n_tiles, "seconds": round(seconds, 2),
        "detections": len(pred), "TP": int(tp.sum()), "FP": int((~tp).sum()), "FN": int((~matched).sum()),
        "recall": round(float(matched.mean()), 3) if len(gt) else None,
        "precision": round(float(tp.mean()), 3) if len(pred) else None,
        "recall_by_ship_size": by_size,
    }, tp, matched


def combine(summaries):
    """Sum counts over several frames and recompute recall/precision from the totals."""
    tot = {"method": summaries[0]["method"], "frames": len(summaries)}
    for k in ["model_runs", "seconds", "detections", "TP", "FP", "FN"]:
        tot[k] = round(sum(s[k] for s in summaries), 2)
    tot["recall"] = round(tot["TP"] / max(1, tot["TP"] + tot["FN"]), 3)  # TP + FN = all GT ships
    tot["precision"] = round(tot["TP"] / max(1, tot["TP"] + tot["FP"]), 3)
    buckets = summaries[0]["recall_by_ship_size"]
    tot["recall_by_ship_size"] = {b: [sum(s["recall_by_ship_size"][b][i] for s in summaries) for i in (0, 1)]
                                  for b in buckets}
    return tot


def draw(frame, pred, tp, gt, matched, title):
    """Green = correct detection, red = false alarm, yellow = missed ship."""
    img = frame.copy()
    for (x0, y0, x1, y1) in gt[~matched].astype(int):
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 255), 3)
    for (x0, y0, x1, y1), ok in zip(pred.astype(int), tp):
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0) if ok else (0, 0, 255), 3)
    cv2.putText(img, title, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 4)
    return img


# ---------------------------------------------------------------- main

def main(args):
    from ultralytics import YOLO  # imported here so --selftest runs without it

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    rng = np.random.default_rng(args.seed)
    used, frames_info = set(), []
    per_frame = {"tiled": [], "resized": []}

    for k in range(args.frames):
        frame, gt, names = build_frame(Path(args.data), rng, used)
        used.update(names)
        frames_info.append({"images": names, "gt_ships": len(gt)})
        if k == 0:
            cv2.imwrite(str(out / "frame_input.jpg"), frame)
            detect(model, frame[:TILE, :TILE])  # warm-up: first call initialises, would distort timing

        for name, fn in [("tiled", detect_tiled), ("resized", detect_resized)]:
            t0 = time.perf_counter()
            pred, scores, n_tiles = fn(model, frame)
            seconds = time.perf_counter() - t0
            summary, tp, matched = summarise(name, pred, scores, gt, seconds, n_tiles)
            per_frame[name].append(summary)
            if k == 0:  # images only for the first frame; the rest only feed the totals
                title = f"{name}: {summary['TP']}/{len(gt)} ships found, {summary['FP']} false alarms"
                cv2.imwrite(str(out / f"frame_{name}.jpg"), draw(frame, pred, tp, gt, matched, title))
        print(f"frame {k}: {len(gt)} ships | tiled TP {per_frame['tiled'][-1]['TP']} "
              f"| resized TP {per_frame['resized'][-1]['TP']}")

    totals = [combine(per_frame[name]) for name in per_frame]
    for t in totals:
        print(json.dumps(t))
    (out / "tiling_comparison.json").write_text(json.dumps(
        {"tile": TILE, "overlap": OVERLAP, "conf": CONF, "device": "CPU", "frames": frames_info,
         "totals": totals, "per_frame": per_frame}, indent=2))
    print(f"saved to {out}/")


def selftest():
    # 1. Tiles cover the frame with the promised overlap
    assert tile_starts(2304) == [0, 512, 1024, 1536], tile_starts(2304)
    assert tile_starts(768) == [0]
    for length in [800, 2304, 3000, 5000]:
        s = tile_starts(length)
        assert s[0] == 0 and s[-1] + TILE == length
        assert all(b - a <= TILE - OVERLAP for a, b in zip(s, s[1:]))  # gaps never exceed the stride

    # 2. Any ship up to OVERLAP px long lies entirely inside at least one tile
    rng = np.random.default_rng(0)
    starts = tile_starts(2304)
    for _ in range(10_000):
        size = rng.integers(1, OVERLAP + 1)
        x0 = rng.integers(0, 2304 - size)
        assert any(t <= x0 and x0 + size <= t + TILE for t in starts)

    # 3. NMS keeps the best of two duplicates and the separate box
    boxes = np.array([[100, 100, 150, 120], [102, 101, 151, 121], [400, 400, 450, 420]], dtype=float)
    keep = nms(boxes, np.array([0.6, 0.9, 0.5]))
    assert sorted(keep.tolist()) == [1, 2], keep

    # 4. Matching: one hit, one false alarm, one missed ship
    gt = np.array([[100, 100, 150, 120], [600, 600, 640, 610]], dtype=float)
    pred = np.array([[101, 100, 150, 121], [300, 300, 340, 320]], dtype=float)
    tp, matched = match(pred, np.array([0.9, 0.8]), gt)
    assert tp.tolist() == [True, False] and matched.tolist() == [True, False]
    print("selftest passed")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="models/best.pt")
    p.add_argument("--data", default="data/val_test_subset/test")
    p.add_argument("--out", default="results")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frames", type=int, default=1, help="number of frames, each from unused test images")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    selftest() if args.selftest else main(args)
