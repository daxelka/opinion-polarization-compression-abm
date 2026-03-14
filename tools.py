import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import json
from scipy.stats import entropy
import gzip
import io


def flatern_opinions_matrix(opinions_matrix, steps=[0, 1000, 2000, 3000]):
    """
    steps: list of times for snapshots, e.g. [0, 1000, 2000, 3000]
    opinion_matrix: ndarray with first dimension as time steps and the second dimension as opinions,
                    e.g. ndarray(N_steps, N_nodes)
    """

    d_flatten = []
    index_flatten = []
    x = 0
    for i in steps:
        d_flatten += list(opinions_matrix[i] + x)
        x += 0.0

        if (i + 1) % 2 == 0:
            i += 1
        index_flatten += len(opinions_matrix[i - 1]) * [i]

    opinions_df = pd.DataFrame({'opinions': d_flatten, 'time': index_flatten})
    return opinions_df

def plot_opinion_evolution(opinions_df, steps=[0, 1000, 2000, 3000], n_bins=30):
    """
    steps: list of times for snapshots, e.g. [0, 1000, 2000, 3000]
    opinions_df: dataframe with 2 coloums: opinions in "opinions" and time in "time"
    """

    # Wanted palette details
    enmax_palette = ["#cfbfa3", '#93794d', "#b9e0fa"]
    color_codes_wanted = ['Mikebeige', 'darkbeige', 'Mikeblue']
    c = lambda x: enmax_palette[color_codes_wanted.index(x)]

    sns.set(rc={'axes.facecolor': c("Mikeblue")})
    g = sns.displot(opinions_df, y="opinions", col="time", aspect=.3, kde=True, bins=n_bins, linewidth=1, color=c("Mikebeige"),
                    line_kws={'color': 'black'})  # color='black', # linecolor for bars: ec='Navy',
    g.set(ylim=(0, 1), ylabel='Distribution slices')

    for i in range(len(steps)):
        ax = g.facet_axis(0, i)

        ax.set_xlabel('')
        ax.set_ylabel('')

        # horizontal tciks
        ax.set_yticks([0, 0.5, 1])  # set new labels

        # horizontal tciks
        labels = ax.get_xticklabels()  # get x labels
        for i, l in enumerate(labels):
            labels[i] = ''
        ax.set_xticklabels(labels)  # set new labels

        ## line color
        for l in ax.get_lines():
            l.set_color(c("darkbeige"))
            l.set_linewidth(1)
    # print(ax.get_lines()[0])
    plt.tight_layout()
    plt.show()
    plt.close()

def count_nonzero_elements(matrix, c):
    # Find absolute values of matrix elements
    abs_matrix = np.abs(matrix)

    # Count nonzero elements greater than c in each row
    count_array = np.sum(abs_matrix > c, axis=1)

    return count_array

# Calculate the number of changes
def changes_summary(array, window_size):
    num_windows = len(array) - window_size + 1

    # Calculate the number of ones in each window
    averaged_ones = [np.sum(array[i:i + window_size]) / window_size for i in range(num_windows)]

    # Plot the averaged ones
    plt.plot(averaged_ones)
    plt.xlabel('time')
    plt.ylabel('density of opinion changes')
    plt.show()

def create_opinion_dataframe(opinion_matrix, time_steps):
    """
    Creates a DataFrame from opinion_matrix at specified time_steps.

    :param opinion_matrix: List of lists containing opinions
    :param time_steps: List of integers representing time steps
    :return: DataFrame with opinions and corresponding time steps
    """
    if len(opinion_matrix) != len(time_steps):
        raise ValueError("Length of opinion_matrix must match length of time_steps")

    opinions = []
    times = []

    for opinions_at_time, time in zip(opinion_matrix, time_steps):
        # print(opinions_at_time)
        opinions.extend(opinions_at_time)
        times.extend([time] * len(opinions_at_time))

    return pd.DataFrame({'opinions': opinions, 'time': times})

def filter_opinions_by_steps(opinions_df, time_steps):
    """
    Filter the opinions dataframe to include only the specified visualisation steps.

    :param opinions_df: DataFrame containing opinions and time steps
    :param visualisation_steps: List of time steps to include in the filtered dataframe
    :return: Filtered DataFrame
    """
    return opinions_df[opinions_df['time'].isin(time_steps)]

def generate_time_steps(first_value, last_value, step):
    """
    Generate a list of time steps from first_value to last_value with the given step.

    :param first_value: The starting value of the time steps
    :param last_value: The ending value of the time steps
    :param step: The increment between each time step
    :return: A list of time steps
    """
    time_steps = []
    current_value = first_value
    while current_value <= last_value:
        time_steps.append(current_value)
        current_value += step

    # Ensure the last value is included if it's not reached by the step
    if time_steps[-1] != last_value:
        time_steps.append(last_value)

    return time_steps

# Detecting cluters in the opinion distribution
def clusters_detector(opinion):
    cluster_precision = 0.02
    clusters = []
    means = []
    points_sorted = sorted(opinion)
    curr_point = points_sorted[0]
    curr_cluster = [curr_point]
    for point in points_sorted[1:]:
        if point <= curr_point + cluster_precision:
            curr_cluster.append(point)
        else:
            # append new cluster to clusters
            clusters.append(curr_cluster)
            means.append(np.mean(curr_cluster))
            # start new cluster
            curr_cluster = [point]
        curr_point = point
    clusters.append(curr_cluster)
    means.append(np.mean(curr_cluster))
    return clusters, means


def cluster_density(clusters, n_nodes):
    density = []
    for cluster in clusters:
        density.append(len(cluster)/n_nodes)
    return density


def unpack_json(filename):
    x = []
    y = []
    p = []
    with open(filename) as json_file:
        data = json.load(json_file)
        for parameter, results in data['experiments'].items():
            for r in results:
                for c in r:
                    if c[1] > 0.1:
                        x.append(float(parameter))
                        y.append(c[0])
                        p.append(c[1])
    return data, x, y, p

def compressibility_ndarray(data_array):
    def compressibility(data):
        # Convert the row to bytes
        data_bytes = np.array(data).tobytes()

        # Compress the data in memory
        compressed_data = io.BytesIO()
        with gzip.GzipFile(fileobj=compressed_data, mode='wb', compresslevel=9) as f:
            f.write(data_bytes)

        # Get sizes of original and compressed data
        original_size = len(data_bytes)
        compressed_data.seek(0, 2)  # Move to end of buffer to get size
        compressed_size = compressed_data.tell()

        # Calculate compression ratio
        return original_size / compressed_size if compressed_size > 0 else float('inf')

    return np.array([compressibility(row) for row in data_array])

def shanon_entropy_ndarray(data_array, n_bins):
    return np.array([
        entropy(pd.value_counts(pd.cut(row, n_bins, precision=0)), base=2)
        for row in data_array
    ])


# New test function
def test_create_opinion_dataframe():
    # Test case 1: Basic functionality
    opinion_matrix = [
        [0.1, 0.2, 0.3],
        [0.2, 0.3, 0.4],
        [0.3, 0.4, 0.5],
        [0.4, 0.5, 0.6]
    ]
    time_steps = [0, 2]

    result = create_opinion_dataframe(opinion_matrix, time_steps)

    expected_opinions = [0.1, 0.2, 0.3, 0.3, 0.4, 0.5]
    expected_times = [0, 0, 0, 2, 2, 2]
    expected_df = pd.DataFrame({'opinions': expected_opinions, 'time': expected_times})

    pd.testing.assert_frame_equal(result, expected_df)
    print(result)
    print(result.shape)

    print("Test case 1 passed!")

    # Test case 2: Empty opinion matrix
    empty_matrix = []
    empty_time_steps = []

    empty_result = create_opinion_dataframe(empty_matrix, empty_time_steps)

    assert empty_result.empty, "Expected an empty DataFrame for empty inputs"

    print("Test case 2 passed!")

    print("All tests passed!")

# Run the test
if __name__ == "__main__":
    test_create_opinion_dataframe()
