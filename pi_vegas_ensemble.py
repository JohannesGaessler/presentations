import numpy as np
from monte_carlo.pi_vegas import vegas

results = []

for _ in range(1000):
    results.append(vegas(iterations=3, samples_per_iteration=333, num_bins=20, K=1000, alpha=1.0))
results = np.array(results)

mean_result = np.mean(results)
print(mean_result)
print(np.std(results)/mean_result)

