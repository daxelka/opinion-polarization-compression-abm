import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from model import FullyMixedWithoutMemory_Model
import tools as t

# Initialise the model

n_nodes = 1000
n_local_bins, n_global_bins = 10, 10
local_group_size = 200
time_finish = int(1e06)
n_runs = time_finish
steps_between_measurements = int(1e04)
visualisation_steps = [int(3e05), int(5e05), int(7e05), time_finish-1]
# visualisation_steps = [int(1e05), int(2e05), int(3e05), time_finish-1]

time_steps = t.generate_time_steps(first_value=0, last_value=time_finish-1,
                                          step=steps_between_measurements)

# preferential_selection parameters
alpha = 0
const = 0


t0 = time.time()
model = FullyMixedWithoutMemory_Model(n_nodes=n_nodes,
                                      n_local_bins=n_local_bins,
                                      n_global_bins=n_global_bins,
                                      local_group_size=local_group_size,
                                      alpha=alpha,
                                      const=const)

print('model created')
# Load initial condition
# initial_opinion = np.load('initial_opinions_n_1000.npy')
# model.set_initial_opinion(initial_opinion)
# print('initial opinion created')

# Set the initial distribution from a random uniform distribution between 0 and 1
initial_opinion = np.round(np.random.uniform(0, 1, n_nodes), 4)
model.set_initial_opinion(initial_opinion)
print('initial opinion created')


opinions_list = model.run(n_runs, time_steps)

t1 = time.time()
print('elapsed time: ', t1 - t0)

# create a DataFrame with opinions and time steps
opinions_df = t.create_opinion_dataframe(opinions_list, time_steps)

# Use the new function to filter opinions_df
filtered_opinions_df = t.filter_opinions_by_steps(opinions_df, visualisation_steps)

# Update the plot_opinion_evolution call to use the filtered dataframe
t.plot_opinion_evolution(filtered_opinions_df, list(range(len(visualisation_steps))))

# Calculate global entropy and compressibility
compressibility = t.compressibility_ndarray(opinions_list)
global_entropy = t.shanon_entropy_ndarray(opinions_list, n_global_bins)

# plot compressibility
plt.plot(compressibility)
plt.title('compressibility')
plt.show()

# plot entropy
plt.plot(global_entropy)
plt.title('entropy')
plt.show()

# # Save files
opinions_df.to_csv('results/opinions_no_memory_n_bins_' + str(n_local_bins)
                          + '_l_group_size_' + str(local_group_size)
                          + '_pref_selection_alpha_' + str(alpha)
                          + '_c_' + str(const)
                          + '_steps_'+ str(time_finish)
                          + '.csv', index=False)
