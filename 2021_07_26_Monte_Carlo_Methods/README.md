Github page for a presentation on Monte Carlo methods held on 26.07.2021 as part of a seminar on (astro)particle physics.
Python files are ordered in the way they appeared in the presentation.
They produce an output on the terminal as well as *matplotlib* graphics.

* **01_pi_hit_and_miss.py**: hit-and-miss MC estimation of pi.
* **02_p2_crude.py**: crude MC estimation of pi.
* **03_pi_variance_reduction.py**: crude MC (with variance reduction) estimation of pi.
* **04_pi_vegas.py**: implementation of the VEGAS algorithm to estimate pi.
* **05_pi_vegas_ensemble.py**: estimation of the empirical uncertainty of the VEGAS implementation through an ensemble test.
* **06_pi_riemann.py**: implementation of a simple Riemann sum as comparison.
* **07_pi_hit_and_miss_benchmark.py**: benchmark of hit-and-miss MC estimation of pi with both NumPy and Tensorflow.
* **08_pi_crude_benchmark.py**: benchmark of crude MC estimation of pi with both NumPy and Tensorflow.