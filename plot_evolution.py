# Import necessary libraries
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.ticker as mticker
from scipy.stats import gaussian_kde
import matplotlib.cm as cm
from matplotlib import rc
import tools as t
from scipy.stats import entropy
import io
import gzip
import random
from PIL import Image

# PLOS Complex Systems figure requirements: Arial font, TIFF with LZW, max 2250px wide at 300 dpi
matplotlib.rcParams['font.family'] = 'Arial'

# Read opinions data
df = pd.read_csv('results/multiple_runs_evolution_n_bins_10_l_group_size_300_pref_selection_alpha_0_c_0_steps_1000000_repeats_10_results.csv')
df['ID'] = df.groupby('time').cumcount()
print(df.head)

filename_base = 'img/multiple_runs_evolution_n_bins_10_l_group_size_300_pref_selection_alpha_0_c_0_steps_1000000_repeats_10'

df=df[df['sim_id'] == 0]
# Calculate entropy and compressibility
data = df
def compressibility(data):
    data_bytes = data.tobytes()
    compressed_data = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed_data, mode='wb', compresslevel=9) as f:
        f.write(data_bytes)
    original_size = len(data_bytes)
    compressed_data.seek(0, 2)
    compressed_size = compressed_data.tell()
    compression_ratio = original_size / compressed_size
    return compression_ratio

def shanon_entropy(d, n_bins):
    return entropy(pd.Series(pd.cut(d, n_bins, precision=0)).value_counts(), base=2)

results = []
for time, group in data.groupby("time"):
    opinions = group["opinions"].values
    entropy_val = shanon_entropy(opinions, 10)
    compressibility_val = compressibility(opinions)
    results.append((time, entropy_val, compressibility_val))

entropy_data = pd.DataFrame(results, columns=['time', 'entropy', 'compressibility'])

entropy_data.head()


# # Calculate global entropy and compressibility
# compressibility = t.compressibility_ndarray(opinions_list)
# global_entropy = t.shanon_entropy_ndarray(opinions_list, n_global_bins)


# Cut the data a specific time
time_value_end = int(1e06)
df = df[df['time']<=time_value_end]

# Latest time opinion data
data_new_2_latest_time = df[df['time'] == df['time'].max()]

# # Load the dataset for entropy and compressibility analysis
# entropy_data = pd.read_csv(
#     'results/opinions_no_memory_n_bins_10_l_group_size_100_pref_selection_alpha_0_c_0_steps_1000000get_local_group_fixed_v1.csv')
# entropy_data = entropy_data[entropy_data['time']<=time_value_end]

# Compute the KDE for the bimodal curve
kde_new_2 = gaussian_kde(data_new_2_latest_time['opinions'])
x_new_2_bimodal = np.linspace(0, 1, 1000)
kde_values_new_2 = kde_new_2(x_new_2_bimodal)


randomly_selected_nodes =  random.sample(list(range(1001)),1000)
# randomly_selected_nodes =  list(range(1000))
df_selected = df.groupby('time').nth(randomly_selected_nodes).reset_index()


# Define font size
font_size = 24
font_size_label = 20
font_size_ticks = 22

# Get unique IDs at time 0 based on sorted opinions
time_0_df = df_selected[df_selected['time'] == 0].sort_values(by='opinions')

# Generate a colormap from warm to cold
num_trajectories = len(time_0_df)
colors = cm.plasma(np.linspace(0, 1, num_trajectories))  # Use a warm-to-cold colormap like 'plasma'

# Create a dictionary to map IDs to colors based on sorted opinions at time 0
id_to_color = dict(zip(time_0_df['ID'], colors))


def save_tiff(fig, filepath):
    """Save figure as PLOS-compliant TIFF (LZW, RGB)."""
    fig.savefig(filepath, dpi=300, bbox_inches='tight', format='tiff',
                facecolor='white', pil_kwargs={'compression': 'tiff_lzw'})
    img = Image.open(filepath)
    if img.mode != 'RGB':
        img.convert('RGB').save(filepath, compression='tiff_lzw')


# --- Figure a) Trajectory plot ---
fig_a, ax_a = plt.subplots(figsize=(7.5, 5.5))
for opinion_id, group in df_selected.groupby("ID"):
    ax_a.plot(group["time"], group["opinions"], alpha=0.8, linewidth=1.2, color=id_to_color[opinion_id])
ax_a.set_xlabel("Time", fontsize=font_size)
ax_a.set_ylabel("Opinion", fontsize=font_size)
ax_a.set_title("Trajectories of Agents' Opinions", fontsize=font_size, pad=20)
ax_a.grid(True, linestyle='--', alpha=0.6)
ax_a.tick_params(axis='both', labelsize=font_size - 2)
ax_a.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x/1e4)}'))
ax_a.set_xlabel(r"Time ($\times 10^4$)", fontsize=font_size)
fig_a.tight_layout()
save_tiff(fig_a, f'{filename_base}_a.tiff')
plt.show()
plt.close(fig_a)


# --- Figure b) Opinion histogram plot ---
fig_b, ax_b = plt.subplots(figsize=(7.5, 5.5))
ax_b.hist(data_new_2_latest_time['opinions'], bins=30, density=True, alpha=0.6, color='blue', edgecolor='black')
ax_b.plot(x_new_2_bimodal, kde_values_new_2, 'k', linewidth=2, label='Fitted Curve')
ax_b.set_title('Final Opinion Clusters', fontsize=font_size, pad=20)
ax_b.set_xlabel('Opinion', fontsize=font_size)
ax_b.set_ylabel('Density', fontsize=font_size)
ax_b.set_xlim(0, 1)
ax_b.set_xticks(list(ax_b.get_xticks()))
ax_b.tick_params(axis='both', labelsize=font_size - 2)
fig_b.tight_layout()
save_tiff(fig_b, f'{filename_base}_b.tiff')
plt.show()
plt.close(fig_b)


# --- Figure c) Entropy and compressibility plot ---
fig_c, ax_c = plt.subplots(figsize=(7.5, 5.5))
ax_c2 = ax_c.twinx()
ax_c.plot(entropy_data['time'], entropy_data['entropy'], label='Entropy', color='blue')
ax_c2.plot(entropy_data['time'], entropy_data['compressibility'], label='Compressibility', color='red')
ax_c.set_title('Entropy and Compressibility', fontsize=font_size, pad=20)
ax_c.set_xlabel('Time', fontsize=font_size)
ax_c.set_ylabel('Entropy', color='blue', fontsize=font_size)
ax_c2.set_ylabel('Compressibility', color='red', fontsize=font_size)
ax_c.set_ylim(0, 4)
ax_c2.set_ylim(0, 12)
ax_c.tick_params(axis='y', labelcolor='blue', labelsize=font_size_ticks)
ax_c2.tick_params(axis='y', labelcolor='red', labelsize=font_size_ticks)
ax_c.tick_params(axis='x', labelsize=font_size - 2)
ax_c.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x/1e4)}'))
ax_c.set_xlabel(r"Time ($\times 10^4$)", fontsize=font_size)
fig_c.tight_layout()
save_tiff(fig_c, f'{filename_base}_c.tiff')
plt.show()
plt.close(fig_c)