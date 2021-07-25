import numpy as np
vegas_module = __import__("04_pi_vegas")
vegas = vegas_module.vegas

results = []

for _ in range(1000):
    results.append(vegas(iterations=3, samples_per_iteration=333, num_bins=20, K=1000, alpha=1.0))
results = np.array(results)

mean_result = np.mean(results)
print(f"Estimate: {mean_result}")
print(f"Empirical relative uncertainty: {100 * np.std(results)/mean_result}%")

