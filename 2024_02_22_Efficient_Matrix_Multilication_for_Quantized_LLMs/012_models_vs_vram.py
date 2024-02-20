#!/usr/bin/env python3

import matplotlib.pyplot as plt

GPU_NAMES = ["RTX 4090", "V100 (32 GB)", "L40", "A100 (80 GB)", "H100"]
VRAM_GiB  = [        24,             32,    48,             80,     80]

plt.barh(range(5), VRAM_GiB)
plt.yticks(range(5), GPU_NAMES, fontsize=16)

plt.vlines([138, 93], ymin=-1, ymax=6, color="black", linewidth=3.0)
plt.vlines([57, 38, 20], ymin=-1, ymax=6, color="green", linewidth=3.0)
plt.text(138-35, 4.60, "LLaMA 2 70B FP16", fontsize=16)
plt.text(93-20,  4.95, "Mixtral FP16", fontsize=16)
plt.text(57-12, 4.60, "L2 70B Q6_K", color="green", fontsize=16)
plt.text(38-17, 4.95, "Mixtral Q6_K", color="green", fontsize=16)
plt.text(20-35, 4.60, "Mixtral Q3_K_S", color="green", fontsize=16)

plt.xlabel("VRAM [GiB]", fontsize=16)
plt.ylim(-0.5, 4.5)
plt.savefig("012_models_vs_vram.png", dpi=240, bbox_inches="tight")
plt.show()
