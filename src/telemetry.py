"""Block 3: compact binary alert (satellite side) and its decoder (ground side).

The satellite downlinks detections, not images. One packet per frame, little-endian, no padding:

  HEADER  "<2sBIIH"  magic b"SD", version, frame_id, timestamp (Unix s), count     13 bytes
  DET     "<HHHHBB"  cx, cy, w, h (frame pixels), confidence*255, class            10 bytes each
  CRC     "<H"       CRC-16-CCITT over header + detections                          2 bytes

Size = 13 + 10*N + 2 bytes.

v1 sends only fields we can fill honestly. A v2 (new `version` value) would add, per detection:
lat/lon as int32 microdegrees (8 B, needs orbit position + attitude), position uncertainty in metres
(1 B, needs a georeferencing error model) and heading (1 B, needs oriented boxes). Speed and
predicted position are estimated on the ground by tracking a vessel across several alerts.

Run:
  python src/telemetry.py --data data/val_test_subset/val_test_subset/test
"""
import argparse
import binascii
import struct
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from tiling import build_frame, detect_tiled  # same frame and detector as Block 2

MAGIC = b"SD"
VERSION = 1
HEADER = struct.Struct("<2sBIIH")  # "<" = little-endian, no padding: sizes are exact on any machine
DET = struct.Struct("<HHHHBB")
CRC = struct.Struct("<H")
LINK_RATES_BPS = [9600, 1200]


class PacketError(ValueError):
    pass


def crc16(data):
    # CRC-16-CCITT with initial value 0xFFFF: the variant used in CCSDS space-link frames
    return binascii.crc_hqx(data, 0xFFFF)


def encode(frame_id, timestamp, boxes, scores):
    """Satellite side: detections (xyxy pixels, scores 0-1) -> bytes."""
    body = HEADER.pack(MAGIC, VERSION, frame_id, timestamp, len(boxes))
    for (x0, y0, x1, y1), s in zip(np.asarray(boxes).tolist(), np.asarray(scores).tolist()):
        # Centre + size rather than corners: same bytes, and the centre is what an alert is about
        body += DET.pack(round((x0 + x1) / 2), round((y0 + y1) / 2), round(x1 - x0), round(y1 - y0),
                         round(s * 255), 0)  # class 0 = ship
    return body + CRC.pack(crc16(body))


def decode(packet):
    """Ground side: bytes -> dict. Raises PacketError on anything inconsistent."""
    if len(packet) < HEADER.size + CRC.size:
        raise PacketError("packet too short")
    body, (crc,) = packet[:-CRC.size], CRC.unpack(packet[-CRC.size:])
    # Check the CRC first: if the bytes are corrupted, no other field can be trusted
    if crc16(body) != crc:
        raise PacketError("CRC mismatch: packet corrupted")
    magic, version, frame_id, timestamp, count = HEADER.unpack_from(body)
    if magic != MAGIC:
        raise PacketError(f"bad magic {magic!r}")
    if version != VERSION:
        raise PacketError(f"unsupported version {version}")
    if len(body) != HEADER.size + count * DET.size:
        raise PacketError("length does not match detection count")
    dets = []
    for i in range(count):
        cx, cy, w, h, conf, cls = DET.unpack_from(body, HEADER.size + i * DET.size)
        dets.append({"box": (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), "conf": conf / 255, "class": cls})
    return {"frame_id": frame_id, "timestamp": timestamp, "detections": dets}


def print_alert(msg):
    """Ground operator's view: only what arrived over the link, no image."""
    t = datetime.fromtimestamp(msg["timestamp"], timezone.utc)
    print(f"    ALERT frame {msg['frame_id']} | {t:%Y-%m-%d %H:%M:%S} UTC | {len(msg['detections'])} detections")
    print("     #   centre x, y (px)   size w x h (px)   confidence")
    for i, d in enumerate(msg["detections"]):
        x0, y0, x1, y1 = d["box"]
        print(f"    {i:>2}   {(x0 + x1) / 2:>6.0f}, {(y0 + y1) / 2:<6.0f}      "
              f"{x1 - x0:>4.0f} x {y1 - y0:<4.0f}       {d['conf']:.2f}")


def fmt_time(seconds):
    if seconds < 60:
        return f"{seconds:.2f} s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


def main(args):
    from ultralytics import YOLO

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # --- satellite side: same frame 0 as Block 2, tiled detection, encode
    frame, _, _ = build_frame(Path(args.data), np.random.default_rng(args.seed))
    boxes, scores, _ = detect_tiled(YOLO(args.weights), frame)
    packet = encode(frame_id=0, timestamp=int(time.time()), boxes=boxes, scores=scores)
    (out / "alert_frame0.bin").write_bytes(packet)
    print(f"encoded {len(boxes)} detections into {len(packet)} bytes "
          f"(13 header + {len(boxes)}x10 + 2 CRC)")

    # --- ground side: decode and check the round trip
    msg = decode(packet)
    print_alert(msg)
    dec_boxes = np.array([d["box"] for d in msg["detections"]]).reshape(-1, 4)
    dec_conf = np.array([d["conf"] for d in msg["detections"]])
    if len(boxes):
        print(f"round trip: max box corner error {np.abs(dec_boxes - boxes).max():.2f} px, "
              f"max confidence error {np.abs(dec_conf - scores).max():.4f}")

    # --- corruption test: one flipped byte must be rejected, not silently decoded
    corrupted = bytearray(packet)
    corrupted[HEADER.size + 3] ^= 0xFF  # inside the first detection
    try:
        decode(bytes(corrupted))
        print("corruption test: NOT detected (unexpected)")
    except PacketError as e:
        print(f"corruption test: rejected ({e})")

    # --- ground side redraw: only what came over the link
    img = frame.copy()
    for d in msg["detections"]:
        x0, y0, x1, y1 = map(int, d["box"])
        cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 0), 3)
    cv2.putText(img, f"ground: {len(msg['detections'])} detections decoded from {len(packet)} bytes",
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 4)
    cv2.imwrite(str(out / "frame_ground_decoded.jpg"), img)

    # --- size and transmit-time table
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    rows = [("Raw RGB frame (2304x2304x3)", frame.nbytes),
            ("JPEG frame (quality 95)", len(jpeg)),
            (f"Alert packet ({len(boxes)} detections)", len(packet))]
    header = "| Payload | Bytes | " + " | ".join(f"@ {r / 1000:g} kbps" for r in LINK_RATES_BPS) + " | vs alert |"
    lines = [header, "|" + "---|" * (3 + len(LINK_RATES_BPS))]
    for name, n in rows:
        times = " | ".join(fmt_time(n * 8 / r) for r in LINK_RATES_BPS)
        lines.append(f"| {name} | {n:,} | {times} | {n / len(packet):,.0f}x |")
    table = "\n".join(lines)
    note = "\n\nTransmit times are lower bounds: they ignore protocol framing, FEC and link availability.\n"
    (out / "telemetry_table.md").write_text(table + note)
    print(table)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="models/best.pt")
    p.add_argument("--data", default="data/val_test_subset/test")
    p.add_argument("--out", default="results")
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
