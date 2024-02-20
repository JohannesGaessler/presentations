#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE    = np.array([    1,     2,     4,     8,    16,    32,    64,   128,  256,  512, 1024, 2048, 4096])
TS_FP16       = np.array([49.28, 89.61, 172.6, 330.7, 567.8,  1192,  2019,  2855, 3400, 3473, 3596, 3429, 3079])
TS_Q8_0_GEMM  = np.array([19.90, 42.16, 82.98, 162.7, 301.6, 616.7,  1131,  1833, 2554, 2978, 3315, 3293, 3018])
TS_Q8_0_FUSED = np.array([78.45, 150.0, 274.5, 379.9, 311.3, 364.3, 377.1, 377.1,    0,    0,    0,    0,    0])
TS_Q8_0_MMQ   = np.array([    0,     0,     0,     0, 246.6, 514.6, 972.7,  1707, 1899, 2005, 2057, 2045, 1913])
TS_Q8_0_MMI8  = np.array([56.09, 107.2, 206.4, 280.0, 314.2, 370.8,  1246,  1987, 2641, 3394, 3536, 3509, 3190])

plt.yscale("log")
plt.xscale("log")
plt.title("LLaMA 7b, single user, 4096 context, RTX 3090", fontsize=16)
plt.xlabel("Batch size", fontsize=16)
plt.ylabel("Throughput [tokens / second]", fontsize=16)
plt.xlim(1,  1024)
plt.ylim(10, 10000)

plt.plot(BATCH_SIZE, TS_FP16, label="FP16 cuBLAS GEMM")
plt.plot(BATCH_SIZE, TS_Q8_0_GEMM, label="Q8_0 cuBLAS GEMM")
plt.legend(loc="lower right", fontsize=16)
plt.savefig("020_mul_mat_scaling_1.png", dpi=240)

plt.plot(BATCH_SIZE, np.maximum(TS_Q8_0_FUSED, TS_Q8_0_MMQ), label="Q8_0 llama.cpp integer intrinsics")
plt.legend(loc="lower right", fontsize=16)
plt.savefig("020_mul_mat_scaling_2.png", dpi=240)

plt.plot(BATCH_SIZE, TS_Q8_0_MMI8, label="Q8_0 llama.cpp int8 tensor cores")
plt.legend(loc="lower right", fontsize=16)
plt.savefig("020_mul_mat_scaling_3.png", dpi=240)
