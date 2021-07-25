import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize=(20.0, 6.0))


def f(x):
    return 1 - np.sqrt(1 - x ** 2)


EXPECTED_AREA = 1.0 - np.pi / 4


def vegas(iterations=3, samples_per_iteration=333, num_bins=20, K=1000, alpha=1.0, make_plots=False):
    bin_edges = np.linspace(start=0, stop=1, endpoint=True, num=num_bins+1)
    bin_widths = bin_edges[1:] - bin_edges[:-1]

    weighted_function_value_sum = 0.0

    for j in range(iterations):
        random_numbers = np.random.rand(samples_per_iteration)
        random_bins = np.random.randint(low=0, high=num_bins, size=samples_per_iteration)
        random_bins_low = bin_edges[random_bins]
        random_bins_high = bin_edges[random_bins + 1]
        random_bin_widths = random_bins_high - random_bins_low
        random_numbers_transformed = random_bins_low + random_numbers * random_bin_widths
        function_values = f(random_numbers_transformed)
        weighted_function_values = function_values * random_bin_widths * num_bins

        if make_plots:
            plt.subplot(1, iterations, j+1)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plot_x = np.linspace(start=0.001, stop=1.0, num=1000, endpoint=True)
            plt.vlines(
                x=random_numbers_transformed[:100], ymin=0, ymax=weighted_function_values[:100], color="black",
                label="$samples$"
            )
            plt.plot(plot_x, f(plot_x), label="$f(x)$")
            plt.bar(
                x=bin_edges[:-1], height=EXPECTED_AREA/(num_bins * bin_widths), width=bin_widths, align="edge",
                color=(1.0, 0.0, 0.0, 0.5), label="$g(x)$"
            )
            plt.xlabel("$x$")
            if j == 0:
                plt.ylabel("$y$")
                plt.legend(loc="upper left")

        weighted_function_value_sum += np.sum(weighted_function_values)

        bin_weights = np.zeros(num_bins)
        for i in range(num_bins):
            bin_weights[i] = np.sum(function_values[random_bins == i])
        bin_weights *= bin_widths
        #bin_splits = 1 + K * bin_weights / np.sum(bin_weights)
        bin_splits = 1 + K * ((bin_weights / np.sum(bin_weights) - 1) / np.log(bin_weights / np.sum(bin_weights))) ** alpha
        bin_splits = bin_splits.astype(int)
        refined_bin_edges = np.zeros(1 + np.sum(bin_splits))
        refined_bin_weights = np.zeros(refined_bin_edges.shape[0] - 1)
        index = 0
        for i in range(num_bins):
            new_bin_edges = np.linspace(start=bin_edges[i], stop=bin_edges[i+1], num=bin_splits[i], endpoint=False)
            refined_bin_edges[index:index+bin_splits[i]] = new_bin_edges
            refined_bin_weights[index:index+bin_splits[i]] = bin_weights[i] / bin_splits[i]
            index += bin_splits[i]
        refined_bin_edges[-1] = 1.0

        average_bin_weight = np.mean(bin_weights)
        new_bin_edges = np.zeros_like(bin_edges)
        current_sum = 0
        current_refined_index = 0
        for i in range(num_bins-1):
            while current_sum < average_bin_weight:
                current_sum += refined_bin_weights[current_refined_index]
                current_refined_index += 1
            current_sum -= average_bin_weight
            new_bin_edges[i + 1] = refined_bin_edges[current_refined_index]
        new_bin_edges[-1] = 1
        bin_edges = new_bin_edges
        bin_widths = bin_edges[1:] - bin_edges[:-1]

    if make_plots:
        plt.savefig("pi_vegas.png")

    integral_estimate = weighted_function_value_sum / (iterations * samples_per_iteration)
    return 4 * (1.0 - integral_estimate)


if __name__ == "__main__":
    print(f"Estimate: {vegas(make_plots=True)} s")

    #plt.show()
