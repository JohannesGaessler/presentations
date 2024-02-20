#!/usr/bin/env python3

import matplotlib.pyplot as plt

plt.figure(figsize=(4, 6))
plt.barh(range(6), [9.7, 19.5, 19.5, 156, 312, 624])
plt.yticks(range(6), ["FP64", "FP64 tensor core", "FP32", "TF32 tensor core", "FP16 tensor core", "int8 tensor core"], fontsize=16)
plt.title("NVIDIA A100 compute throughput", fontsize=16)
plt.xlabel("TFLOPS/TOPS", fontsize=16)
plt.savefig("030_a100_stats.png", dpi=240, bbox_inches="tight")
