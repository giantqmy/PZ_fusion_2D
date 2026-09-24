import matplotlib.pyplot as plt
import os
from ultralytics.data.utils import getPolarizationImages

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

im = getPolarizationImages(
    r"I:\KEDADATA\data\pz\images\2000-01-01-09-21-37_318.png"
)

print("拼接后 shape:", im.shape)  # (H, W, 9)

S0_rgb   = im[:, :, 0:3]
S0_nir   = im[:, :, 3]
DoLP_rgb = im[:, :, 4]
DoLP_nir = im[:, :, 5]
AoLP_rgb = im[:, :, 6]
AoLP_nir = im[:, :, 7]
Depth    = im[:, :, 8]

plt.figure(figsize=(14, 10))

plt.subplot(2, 4, 1)
plt.imshow(S0_rgb)
plt.title("S0 RGB")
plt.axis("off")

plt.subplot(2, 4, 2)
plt.imshow(S0_nir, cmap="gray", vmin=0, vmax=255)
plt.title("S0 NIR")
plt.axis("off")

plt.subplot(2, 4, 3)
plt.imshow(DoLP_rgb, cmap="gray", vmin=0, vmax=255)
plt.title("DoLP RGB")
plt.axis("off")

plt.subplot(2, 4, 4)
plt.imshow(DoLP_nir, cmap="gray", vmin=0, vmax=255)
plt.title("DoLP NIR")
plt.axis("off")

plt.subplot(2, 4, 5)
plt.imshow(AoLP_rgb, cmap="gray", vmin=0, vmax=255)
plt.title("AoLP RGB")
plt.axis("off")

plt.subplot(2, 4, 6)
plt.imshow(AoLP_nir, cmap="gray", vmin=0, vmax=255)
plt.title("AoLP NIR")
plt.axis("off")

plt.subplot(2, 4, 7)
plt.imshow(Depth, cmap="gray", vmin=0, vmax=255)
plt.title("Depth (0–255)")
plt.axis("off")

plt.tight_layout()
plt.show()
