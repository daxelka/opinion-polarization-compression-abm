import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# PLOS Complex Systems figure requirements: Arial font, TIFF with LZW, max 2250px wide at 300 dpi
matplotlib.rcParams['font.family'] = 'Arial'

# Reload the updated CSV file to ensure we have the clustered_df in memory
clustered_df = pd.read_csv('results/experiment_variance_between_clusters_n_bins_7_l_group_size_200_pref_selection_alpha_0_c_0_steps_1000000local_group_fixed_for_box_plot.csv')

# set the name for the plot
filename_for_plot = 'img/experiment_variance_between_clusters_n_bins_7_l_group_size_200_pref_selection_alpha_0_c_0_steps_1000000local_group_fixed_for_box_plot.tiff'
# Calculate mean and std for each cluster across all sim_ids
cluster_stats = clustered_df.groupby(['sim_id', 'cluster_id'])['opinion_value'].agg(['mean', 'std']).reset_index()

# Calculate average mean and std for red (cluster 0) and blue (cluster 1) clusters
average_stats = cluster_stats.groupby('cluster_id')[['mean', 'std']].mean().reset_index()
purple_mean, purple_std = average_stats.loc[average_stats['cluster_id'] == 0, ['mean', 'std']].values[0]
orange_mean, orange_std = average_stats.loc[average_stats['cluster_id'] == 1, ['mean', 'std']].values[0]


plt.figure(figsize=(7.5, 5.5))
sns.set(style="whitegrid")

# Iterate over each simulation ID and plot boxes for each cluster on the exact same horizontal line.
for sim_id in clustered_df['sim_id'].unique():
    sim_data = clustered_df[clustered_df['sim_id'] == sim_id]
    for cluster_id in sim_data['cluster_id'].unique():
        cluster_data = sim_data[sim_data['cluster_id'] == cluster_id]
        sns.boxplot(
            x=cluster_data['opinion_value'],
            y=[sim_id] * len(cluster_data),  # Match y length with cluster_data length
            color="purple" if cluster_id == 0 else "orange", # Manual coloring based on cluster
            width=0.6,
            orient="h"
        )

# Set x-axis limits and labels
plt.xlim(0, 1)
plt.xlabel("Opinions",fontsize=20)
plt.ylabel("Simulation ID", fontsize=20)
plt.title("n_bins: 7", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# Add text with calculated statistics
text_x = 0.45  # X-position for text
text_y_purple = 4.5  # Y-position for red cluster text (beyond max sim_id)
text_y_orange = 5.5 # Y-position for blue cluster text
plt.text(text_x, text_y_purple, f"Purple Cluster:\nMean:{purple_mean:.2f}, Std:{purple_std:.2f}", color="purple", fontsize=16)
plt.text(text_x, text_y_orange, f"Orange Cluster:\nMean:{orange_mean:.2f}, Std:{orange_std:.2f}", color="orange", fontsize=16)

plt.tight_layout()
plt.savefig(filename_for_plot, dpi=300, bbox_inches='tight', format='tiff',
            facecolor='white', pil_kwargs={'compression': 'tiff_lzw'})
plt.close()

# PLOS requires RGB (no alpha channel) — convert RGBA to RGB
from PIL import Image
img = Image.open(filename_for_plot)
if img.mode != 'RGB':
    img.convert('RGB').save(filename_for_plot, compression='tiff_lzw')
