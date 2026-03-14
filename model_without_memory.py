import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
import gzip
import io
import pandas as pd
from collections import deque
import random
from .topology import FullyMixedTopology
from .base_model import BaseModelMixin

class FullyMixedWithoutMemory_Model(FullyMixedTopology, BaseModelMixin):
    def __init__(self, n_nodes=100, n_local_bins=100, n_global_bins=100, local_group_size=100, alpha=0.1, const=0.001):
        # Initialize the parent class
        super().__init__(alpha, const)

        # Initialize attributes specific to FullyMixedWithoutMemory_Model
        self.n_nodes = n_nodes
        self.nodes_ids = range(n_nodes)
        self.opinions = []
        self.n_local_bins = n_local_bins
        self.n_global_bins = n_global_bins
        self.local_group_size = local_group_size
        self.NoiseLevel = 0

    def get_local_group(self, node_id):
        num_nearest = self.local_group_size
        # Calculate the absolute difference from the specified element
        lst = list(self.opinions)
        node_opinion = self.opinions[node_id]
        differences = [(abs(x - node_opinion), x) for x in lst]
        # Sort based on the differences
        differences.sort(key=lambda x: x[0])

        # Extract N nearest values
        nearest_values = [x[1] for x in differences[:num_nearest]]
        nearest_values_without_self = list(nearest_values)
        nearest_values_without_self.remove(node_opinion)
        return nearest_values, nearest_values_without_self

    def run_one_step(self):
        world = np.array(list(self.opinions))

        agent_1 = np.random.choice(self.nodes_ids, 1)[0]
        opinion_agent_2 = self.preferential_selection_function(agent_1, 1, return_list=False) # this function returns the opinion value not the id
        # print('opinion_agent_1:' ,self.opinions[agent_1], ' opinion_agent_2:' ,opinion_agent_2)
        # opinion_agent_2 = self.random_selection_function(agent_1)

        entropy_world = self.shanon_entropy(world, self.n_global_bins)

        if opinion_agent_2 == self.opinions[agent_1]:
            # Record results
            # sh_ent = entropy_world
            # changes = 0
            d_saved = list(world)
        else:
            # Get local group for agent 1
            local_group_agent_1, local_group_agent_1_without_self  = self.get_local_group(agent_1) # local group with self
            # print('local group: ', local_group_agent_1, 'local group without self: ', local_group_agent_1_without_self)

            # Calculate entropy ratio prior interaction
            entropy_local = self.shanon_entropy(local_group_agent_1, self.n_local_bins)
            EntropyRatio_prior = entropy_local/entropy_world

            # Get local entropy if agent 1 changes
            local_group_agent_1_changed = local_group_agent_1_without_self + [opinion_agent_2]
            # print('local group changed: ', local_group_agent_1_changed)
            # entropy_local_Agent1Change = self.shanon_entropy(local_group_agent_1_changed, self.n_local_bins)
            # print('local entropy after change: ', entropy_local_Agent1Change)

            # local_group_agent_1_changed_old = deque(local_group_agent_1)
            # local_group_agent_1_changed_old.appendleft(opinion_agent_2)
            entropy_local_Agent1Change = self.shanon_entropy(local_group_agent_1_changed, self.n_local_bins)
            # print('local_group_agent_1_changed_old after change: ', local_group_agent_1_changed_old)
            # print('local entropy after change: ', entropy_local_Agent1Change)

            # Calculate entropy ratio after agent 1 chages
            # get global entropy of distribution if agent 1 changes
            world_changed = self.change_one_opinion(agent_1, opinion_agent_2)
            entropy_world_Agent1Change = self.shanon_entropy(world_changed, self.n_global_bins)

            EntropyRatio_Agent1Change = entropy_local_Agent1Change/entropy_world_Agent1Change  # calc ratio of global local entropy if agent1 makes the change

            # print('EntropyRatio_Agent1Change: ', EntropyRatio_Agent1Change, 'EntropyRatio_prior: ', EntropyRatio_prior)
            # Compare the two ratios, and accept the Agent1 change if it results in a preferable ratio of local to global entropy
            if EntropyRatio_Agent1Change > EntropyRatio_prior:
                # Change opinions
                self.opinions = list(world_changed)
                # Record results
                # sh_ent = entropy_world_Agent1Change  # Save the global Shannon Entropy for this step
                # changes = 1  # record that a change was made in this step
                d_saved = list(world_changed)
            else:
                # Record results
                # sh_ent = entropy_world
                # changes = 0
                d_saved = list(world)

        # d_compressibility = self.compressibility(np.array(self.opinions))
        # return sh_ent, changes, d_saved, d_compressibility
        # print('world: ', d_saved)
        return d_saved
