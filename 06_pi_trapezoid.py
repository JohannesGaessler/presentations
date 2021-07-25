import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize=(6.0, 6.0))


def f(x):
    return 1 - np.sqrt(1 - x ** 2)


NUM_BINS = 1000

bin_edges = np.linspace(start=0, stop=1.0, num=NUM_BINS+1, endpoint=True)
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
y = f(bin_centers)
area_estimate = np.sum(y) / NUM_BINS
pi_estimate = 4 * (1.0 - area_estimate)
print(f"Estimate: {pi_estimate:.6f}")
print(f"Error: {(pi_estimate - np.pi) / np.pi * 100:.4f}%")


plt.xlim(0, 1)
plt.ylim(0, 1)
plot_x = np.linspace(start=0, stop=1.0, num=101, endpoint=True)
plt.plot(plot_x, f(plot_x), label="$f(x)$")
bar_x = np.arange(20) * 0.05 + 0.025
plt.bar(x=bar_x, height=f(bar_x), width=0.05, color=(1.0, 0.0, 0.0, 0.5), label="Integral")
plt.legend(loc="upper left")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("pi_midpoint.png")
