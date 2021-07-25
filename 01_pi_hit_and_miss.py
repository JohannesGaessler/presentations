import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
plt.figure(figsize=(6.0, 6.0))


def f(x):
    return 1 - np.sqrt(1 - x ** 2)


SAMPLE_SIZE = 1000
p = np.pi/4
q = 1.0 - p

rand_x = np.random.rand(SAMPLE_SIZE)
rand_y = np.random.rand(SAMPLE_SIZE)
plot_x = np.linspace(start=0, stop=1.0, num=101, endpoint=True)

in_circle = (rand_y - 1.0) ** 2 + rand_x ** 2 < 1.0
not_in_circle = np.logical_not(in_circle)

p_empirical = np.mean(in_circle)
q_empirical = np.mean(not_in_circle)

rand_x_in = rand_x[in_circle]
rand_y_in = rand_y[in_circle]
rand_x_out = rand_x[not_in_circle]
rand_y_out = rand_y[not_in_circle]

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.plot(plot_x, f(plot_x), color="blue")
plt.scatter(rand_x_in, rand_y_in, marker=".", color="blue", label="Accepted points")
plt.scatter(rand_x_out, rand_y_out, marker=".", color="red", label="Rejected points")
plt.legend(loc="best")
plt.xlabel("$x$")
plt.ylabel("$y$")

pi_empirical = 4 * p_empirical
print(f"Estimate: {pi_empirical:.6f}")
print(f"Empirical relative uncertainty: {4 * np.sqrt(p_empirical*q_empirical/SAMPLE_SIZE) / pi_empirical * 100:.4f}%")
print(f"Expected relative uncertainty: {4 * np.sqrt(p*q/SAMPLE_SIZE) / np.pi * 100:.4f}%")

plt.savefig("pi_hit_and_miss.png")
plt.show()
