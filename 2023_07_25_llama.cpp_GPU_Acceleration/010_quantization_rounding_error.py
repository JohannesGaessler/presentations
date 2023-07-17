#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

values = np.random.randn(32)
amax = np.max(np.abs(values))
quant_delta = amax / 8

if np.max(values) > np.abs(np.min(values)):
    quant_y = np.linspace(start=-7/8*amax, stop=amax, num=16)
else:
    quant_y = np.linspace(start=-amax, stop=7/8*amax, num=16)

rounding_error_low = []
rounding_error_high = []

for value in values:
    round_down = value - value % quant_delta
    if value - round_down < round_down + quant_delta - value:
        rounding_error_low.append(round_down)
        rounding_error_high.append(value)
    else:
        rounding_error_low.append(value)
        rounding_error_high.append(round_down + quant_delta)

plt.hlines(quant_y, -1, 32, linestyles="dotted", color="black", label="4 bit quantized values")
plt.vlines(np.arange(32), rounding_error_low, rounding_error_high, color="blue", label="Rounding error")
plt.plot(np.arange(32), values, ".", color="red", label="Original values")

plt.xlabel("Weight index")
plt.ylabel("Weight value")
plt.legend(loc="upper right")
plt.xlim(-1, 32)
plt.savefig("010_quantization_rounding_error.png", dpi=240)
plt.show()
