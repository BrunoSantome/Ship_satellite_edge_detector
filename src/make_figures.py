"""Block 5: README figures, built from existing outputs (run after tiling.py and telemetry.py).

Full 2304 px frames are too big to show small ships on a web page, so we crop the area of frame 0
where tiled and resized differ most (a harbour full of small boats) and put the two side by side.
"""
from pathlib import Path

import cv2
import numpy as np

R = Path("results")
X0, Y0, X1, Y1 = 0, 1100, 800, 1560  # harbour area of frame 0 (middle-left source image)


def label(img, text):
    cv2.rectangle(img, (0, 0), (len(text) * 19 + 20, 50), (0, 0, 0), -1)  # dark band so text is readable
    cv2.putText(img, text, (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    return img


tiled = cv2.imread(str(R / "frame_tiled.jpg"))[Y0:Y1, X0:X1].copy()
resized = cv2.imread(str(R / "frame_resized.jpg"))[Y0:Y1, X0:X1].copy()
gap = np.full((Y1 - Y0, 12, 3), 255, dtype=np.uint8)
fig = np.hstack([label(tiled, "TILED (16 tiles)"), gap, label(resized, "RESIZED (1 pass)")])
cv2.imwrite(str(R / "fig_tiled_vs_resized.jpg"), fig)

ground = cv2.imread(str(R / "frame_ground_decoded.jpg"))
cv2.imwrite(str(R / "fig_ground_decoded.jpg"), cv2.resize(ground, (1000, 1000), interpolation=cv2.INTER_AREA))
print("wrote results/fig_tiled_vs_resized.jpg, results/fig_ground_decoded.jpg")
