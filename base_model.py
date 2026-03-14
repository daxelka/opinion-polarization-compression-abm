import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
import gzip
import io
import pandas as pd
from collections import deque
import random

class BaseModelMixin:
    def __init__(self, alpha, const):
        self.alpha = alpha
        self.const = const

    def run(self, n_iterations=50, time_steps=None):
        """ Agent-based simulation: compressibility & entropy increasing where agents
            adopt a new attitude if it has a higher ratio of local/global entropy than their current one
        """
        # if time_steps is None:
        #     time_steps = list(range(n_iterations))
        # n_updates = len(time_steps)
        # d_saved = np.zeros((n_updates, self.n_nodes))
        # d_compressibility = np.zeros(n_updates)
        # sh_ent = np.zeros(n_updates)
        # changes = np.zeros(n_updates)
        #
        # update_index = 0
        # for i in range(n_iterations):
        #     sh_ent_i, changes_i, d_saved_i, d_compressibility_i = self.run_one_step()
        #
        #     if i in time_steps:
        #         sh_ent[update_index] = sh_ent_i
        #         d_saved[update_index] = d_saved_i
        #         d_compressibility[update_index] = d_compressibility_i
        #         changes[update_index] = changes_i
        #         update_index += 1
        #
        # return d_saved, d_compressibility, sh_ent, changes

        if time_steps is None:
            time_steps = list(range(n_iterations))
        n_updates = len(time_steps)
        d_saved = np.zeros((n_updates, self.n_nodes))

        update_index = 0
        for i in range(n_iterations):
            d_saved_i = self.run_one_step()

            if i in time_steps:
                d_saved[update_index] = d_saved_i
                update_index += 1

        return d_saved

    def calculate_distances(self, node_id): # can work on networks too
        # get neighours list
        neighbours = self.get_neighbours(node_id)
        node_opinion = self.opinions[node_id]

        distances = np.abs(neighbours - node_opinion)
        return distances

    def preferential_selection_function(self, node_id, sample_size=1, return_list=True): #
        # get node's neighbours
        neighbours = self.get_neighbours(node_id)
        # calculate distance from the current node opinion
        distance = self.calculate_distances(node_id)
        # Calculate probabilities, avoiding division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            prob = 1 / ((distance + self.const) ** self.alpha)

        # Renormalize the probabilities
        prob = prob / np.sum(prob)

        # Perform weighted sampling
        sampled_agents_opinions = random.choices(neighbours, weights=prob, k=sample_size) # old version
        if sample_size == 1 and not return_list:
            sampled_agents_opinions = sampled_agents_opinions[0]
        return sampled_agents_opinions

    def random_selection_function(self, node_id): # can work on networks too
        # get node's neighbours
        neighbours = self.get_neighbours(node_id)
        opinion_agent_2 = np.random.choice(neighbours, 1)[0]
        return opinion_agent_2


    def shanon_entropy(self, d, n_bins): # can work on networks too
        return entropy(pd.value_counts(pd.cut(d, n_bins, precision=0)), base=2)


    def show_opinion_distribution(self, opinions, color='skyblue', title='Histogram', filename=None): # can work on networks too

        # bins = [0.01 * n for n in range(100)]
        plt.hist(opinions, bins=30, density=True, color=color)
        plt.xlabel("opinions", fontsize=12)
        plt.ylabel('density', fontsize=12)

        if title != 'Histogram':
            plt.title(title, fontsize=14)
        # plt.show()

        if filename is not None:
            plt.savefig(filename)
        plt.show()

    def compressibility(self, data):
        # Convert the array to bytes
        data_bytes = data.tobytes()

        # Compress the data in memory without writing to a file
        compressed_data = io.BytesIO()
        with gzip.GzipFile(fileobj=compressed_data, mode='wb', compresslevel=9) as f:
            f.write(data_bytes)

        # Get the size of the original and compressed data from in-memory buffer
        original_size = len(data_bytes)
        compressed_data.seek(0, 2)  # Move to the end of the buffer to get the size
        compressed_size = compressed_data.tell()

        # Calculate the compression ratio
        compression_ratio = original_size / compressed_size
        return compression_ratio



