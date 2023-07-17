import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
plt.figure(figsize=(32.0, 6.0))


def f(x):
    return 1 - np.sqrt(1 - x ** 2)


SAMPLE_SIZE = 1000
Ef = quad(lambda x: f(x), 0, 1)[0]
Varf = quad(lambda x: (f(x) - Ef) ** 2, 0, 1)[0]

rand_x = np.random.rand(SAMPLE_SIZE)
rand_y = f(rand_x)
plot_x = np.linspace(start=0, stop=1.0, num=101, endpoint=True)
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.plot(plot_x, f(plot_x))
    plt.bar(x=0, height=rand_y[i], width=1.0, align="edge", color=(1.0, 0.0, 0.0, 0.5))
plt.savefig("pi_crude.png")

pi_empirical = 4 * (1.0 - np.sum(rand_y)/SAMPLE_SIZE)
print(f"Estimate: {pi_empirical:.6f}")
print(f"Empirical uncertainty: {4 * np.sqrt(np.var(rand_y) / SAMPLE_SIZE) / pi_empirical * 100:.4f}%")
print(f"Expected uncertainty: {4 * np.sqrt(Varf / SAMPLE_SIZE) / np.pi * 100:.4f}%")
