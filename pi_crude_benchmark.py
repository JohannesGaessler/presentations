import numpy as np
import tensorflow as tf
from time import time
import matplotlib.pyplot as plt
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

SAMPLE_SIZES = (10000, 100000, 1000000, 10000000, 100000000, 1000000000, 10000000000)
BATCH_SIZE_NP = 1000000
BATCH_SIZE_TF = 1000000
MIN_NUM_BATCHES = 20
INITIALIZATION_TF_CPU = 0.08446002006530762
TF_CPU = np.array([0.020420551300048828, 0.020728588104248047, 0.02513265609741211, 0.03865551948547363,
                   0.13767337799072266, 1.2261271476745605, 12.31807255744934])


def mc_np(sample_size):
    rand_x = np.random.rand(sample_size)

    # Map:
    function_values = 1.0 - np.sqrt(1.0 - np.square(rand_x))

    # Reduce:
    mc_estimate = np.mean(function_values)

    return 4 * mc_estimate


@tf.function
def mc_tf(sample_size):
    rand_x = tf.random.uniform((sample_size,))

    # Map:
    function_values = 1.0 - tf.math.sqrt(1.0 - tf.math.square(rand_x))

    # Reduce:
    mc_estimate = tf.reduce_mean(function_values)

    return 4 * mc_estimate


t0 = time()
mc_tf(10)
tensorflow_initialization = time() - t0
print(f"Initialization: {tensorflow_initialization} s")
print()

t_np = np.zeros(len(SAMPLE_SIZES))
t_tf = np.zeros_like(t_np)
for i, sample_size_i in enumerate(SAMPLE_SIZES):
    print(f"=== Sample size: {sample_size_i} ===")

    t0 = time()
    if sample_size_i <= BATCH_SIZE_NP:
        mc_np(sample_size_i)
        pass
    else:
        for j in range(sample_size_i // BATCH_SIZE_NP):
            mc_np(BATCH_SIZE_NP)
            pass
    t_np[i] = time() - t0
    print(f"NumPy: {t_np[i]} s")

    t0 = time()
    if sample_size_i <= BATCH_SIZE_TF * MIN_NUM_BATCHES:
        for j in range(MIN_NUM_BATCHES):
            mc_tf(sample_size_i // MIN_NUM_BATCHES)
    else:
        for j in range(sample_size_i // BATCH_SIZE_TF):
            mc_tf(BATCH_SIZE_TF)
    t_tf[i] = time() - t0
    print(f"Tensorflow: {t_tf[i]} s")

    print()

t_tf_init = t_tf + tensorflow_initialization
plt.plot(SAMPLE_SIZES, t_np, label="NumPy (Ryzen 3700X)")
plt.plot(SAMPLE_SIZES, t_tf, label="Tensorflow (GTX 1070)")
plt.plot(SAMPLE_SIZES, t_tf_init, label="Tensorflow with initialization (GTX 1070)")
if INITIALIZATION_TF_CPU is not None and TF_CPU is not None:
    plt.plot(SAMPLE_SIZES, TF_CPU, label="Tensorflow (Ryzen 3700X)")
    plt.plot(SAMPLE_SIZES, TF_CPU + INITIALIZATION_TF_CPU, label="Tensorflow with initialization (Ryzen 3700X)")

plt.title("Crude Monte Carlo")
plt.legend(loc="best")
plt.xlabel("Sample size")
plt.ylabel("Time (s)")
plt.xlim((SAMPLE_SIZES[0], SAMPLE_SIZES[-1]))
plt.ylim((t_np[0], t_np[-1]))
plt.xscale("log")
plt.yscale("log")
plt.savefig("pi_crude_benchmark.png", dpi=300)
plt.show()
