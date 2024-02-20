#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

QUANT_NAMES     =          ["Q2_K", "Q3_K_S", "Q4_K_S", "Q5_K_S",  "Q6_K",  "Q8_0"]
VRAM_CUBLAS_MiB = np.array([   5926,    6030,     6880,     7622,    8442,    9968])
KLD_CUBLAS      = np.array([0.10888, 0.08627,  0.01782,  0.00704, 0.00210, 0.00038])
VRAM_MMQ_MiB    = np.array([   5870,    5974,     6824,     7566,    8386,    9912])
KLD_MMQ         = np.array([0.10898, 0.08644,  0.01847,  0.00773, 0.00217, 0.00051])
VRAM_MMI8_MiB   = VRAM_CUBLAS_MiB - (9968 - 9912)  # Extrapolated from q8_0
KLD_MMI8        = np.array([0.11002, 0.08714,  0.01925,  0.00844, 0.00406, 0.00261])

plt.plot(VRAM_CUBLAS_MiB, KLD_CUBLAS, marker=".", label="cuBLAS FP16 GEMM")
plt.plot(VRAM_MMQ_MiB, KLD_MMQ, marker=".", label="llama.cpp int8 intrinsics")
plt.plot(VRAM_MMI8_MiB, KLD_MMI8, marker=".", label="llama.cpp int8 tensor core")
plt.hlines(0, 5500, 10500, color="black", linestyles="dotted")

plt.text(5800, 0.001, "FP16", fontsize=16)
plt.text(VRAM_CUBLAS_MiB[0]+50,  KLD_CUBLAS[0],       "Q2_K", fontsize=16)
plt.text(VRAM_CUBLAS_MiB[1]+50,  KLD_CUBLAS[1],       "Q3_K_S", fontsize=16)
plt.text(VRAM_CUBLAS_MiB[2]+25,  KLD_CUBLAS[2]+0.001, "Q4_K_S", fontsize=16)
plt.text(VRAM_CUBLAS_MiB[3],     KLD_CUBLAS[3]+0.002, "Q5_K_S", fontsize=16)
plt.text(VRAM_CUBLAS_MiB[4]-25,  KLD_CUBLAS[4]+0.004, "Q6_K", fontsize=16)
plt.text(VRAM_CUBLAS_MiB[5]-300, KLD_CUBLAS[5]+0.006, "Q8_0", fontsize=16)

plt.xlim(5800, 10200)
plt.title("LLaMA 7b, 4096 context, 1024 batch size", fontsize=16)
plt.ylabel("Kullback-Leibler divergence vs. FP16", fontsize=16)
plt.xlabel("VRAM usage [MiB]", fontsize=16)
plt.legend(fontsize=16)
plt.savefig("070_quality_vram.png", dpi=240)
