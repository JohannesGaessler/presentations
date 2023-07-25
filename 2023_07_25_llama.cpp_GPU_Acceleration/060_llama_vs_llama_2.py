#!/usr/bin/env python3

import matplotlib.pyplot as plt

llama_context = [512, 1024, 1536, 2048]
llama_2_context = [512, 1024, 1536, 2048, 2560, 3072, 3584, 4096]

llama_7b_q8_0_kafe2_ppl = [3.1513, 2.5123, 2.4226, 2.3188]
# llama_2_7b_q8_0_kafe2_ppl = [3.1801, 2.5670, 2.4735, 2.3658, 2.2958, 2.2450, 2.2006, 2.1765]
llama_2_7b_q8_0_kafe2_ppl = [3.0677, 2.4633, 2.3734, 2.2711, 2.2057, 2.1640, 2.1183, 2.0969]
llama_13b_q8_0_kafe2_ppl = [3.0371, 2.4420, 2.3569, 2.2581]
#llama_2_13b_q8_0_kafe2_ppl = [3.0025, 2.4252, 2.3375, 2.2397, 2.1766, 2.1305, 2.0919, 2.0690]
llama_2_13b_q8_0_kafe2_ppl = [2.9562, 2.3878, 2.3027, 2.2077, 2.1461, 2.1019, 2.0630, 2.0416]
llama_33b_q8_0_kafe2_ppl = [2.9328, 2.3656, 2.2846, 2.1899]

plt.plot(llama_context, llama_7b_q8_0_kafe2_ppl, label="LLaMA 7b")
plt.plot(llama_2_context, llama_2_7b_q8_0_kafe2_ppl, label="LLaMA 2 7b")
plt.plot(llama_context, llama_13b_q8_0_kafe2_ppl, label="LLaMA 13b")
plt.plot(llama_2_context, llama_2_13b_q8_0_kafe2_ppl, label="LLaMA 2 13b")
plt.plot(llama_context, llama_33b_q8_0_kafe2_ppl, label="LLaMA 33b")
plt.legend(loc="upper right")
plt.xlabel("Context size")
plt.ylabel("Perplexity")
plt.title("Perplexity on kafe2 source code dump, q8_0 quantization")
plt.savefig("060_llama_vs_llama_2.png", dpi=240)
plt.show()
