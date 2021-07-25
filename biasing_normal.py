import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

SAMPLE_SIZE = 1000000
NUM_BINS = 100
SIGMA = 5

data = np.random.randn(SAMPLE_SIZE)
bin_edges = np.linspace(start=-SIGMA, stop=SIGMA, num=NUM_BINS+1, endpoint=True)
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
monte_carlo_bin_heights, _ = np.histogram(data, bins=bin_edges)
pdf_values = norm.pdf(bin_edges)
expected_bin_heights = SAMPLE_SIZE * (pdf_values[1:] + pdf_values[:-1]) * SIGMA / NUM_BINS

plt.bar(bin_centers, monte_carlo_bin_heights/expected_bin_heights, width=2*SIGMA/NUM_BINS)
plt.xlim(-SIGMA, SIGMA)
plt.show()
