import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
plt.figure(figsize=(6.0, 6.0))


def f(x):
    return 1 - np.sqrt(1 - x ** 2)


def g(x):
    return 3 * x ** 2


SAMPLE_SIZE = 1000
Efg = quad(lambda x: f(x)/g(x), 0, 1)[0]
Varfg = quad(lambda x: (f(x)/g(x) - Efg) ** 2, 0, 1)[0]

rand_x = np.random.rand(SAMPLE_SIZE) ** (1/3)
rand_y = f(rand_x)
rand_y_weighted = rand_y / g(rand_x)
plot_x = np.linspace(start=0.001, stop=1.0, num=1000, endpoint=True)

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.vlines(x=rand_x[:100], ymin=0, ymax=rand_y_weighted[:100], color="black", label="samples")
plt.plot(plot_x, f(plot_x)/g(plot_x), label="$f(x)/g(x)$")
plt.plot(plot_x, f(plot_x), label="$f(x)$")
plt.plot(plot_x, g(plot_x)*(1.0-np.pi/4.0), label=r"$g(x) \cdot (1 - \pi/4)$")
plt.legend(loc="upper left")
plt.savefig("pi_variance_reduction.png")

pi_empirical = 4 * (1.0 - np.sum(rand_y_weighted)/SAMPLE_SIZE)
print(f"Estimate: {pi_empirical:.6f}")
print(f"Empirical uncertainty: {4 * np.sqrt(np.var(rand_y_weighted) / SAMPLE_SIZE) / pi_empirical * 100:.4f}%")
print(f"Expected uncertainty: {4 * np.sqrt(Varfg / SAMPLE_SIZE) / np.pi * 100:.4f}%")


plt.show()
