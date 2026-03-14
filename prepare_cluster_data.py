import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the data trial file
input_filepath = 'results/experiment_variance_between_clusters_n_bins_100_l_group_size_200_pref_selection_alpha_0_c_0_steps_1000000local_group_fixed_9trials.csv'
opinions_df = pd.read_csv(input_filepath)

# Prepare the output format
output_filepath = 'results/experiment_variance_between_clusters_n_bins_100_l_group_size_200_pref_selection_alpha_0_c_0_steps_1000000local_group_fixed_for_box_plot.csv'
# example_output_df = pd.read_csv(output_format_filepath)

# Create the output structure
all_clusters = []

for trial_name in opinions_df.columns:
    trial_data = opinions_df[trial_name].values.reshape(-1, 1)

    # Determine optimal number of clusters using silhouette score
    best_n_clusters = 2
    best_score = -1
    for n_clusters in range(2, 11):  # Test between 2 and 10 clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(trial_data)
        score = silhouette_score(trial_data, cluster_labels)
        if score > best_score:
            best_score = score
            best_n_clusters = n_clusters

    # Perform final clustering with the best number of clusters
    final_kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
    final_cluster_labels = final_kmeans.fit_predict(trial_data)

    # Append cluster results to the output structure
    trial_clusters = pd.DataFrame({
        "sim_id": trial_name,
        "agent_id": range(len(trial_data)),
        "opinion_value": trial_data.flatten(),
        "cluster_id": final_cluster_labels,
    })
    all_clusters.append(trial_clusters)

# Combine all cluster data
output_df = pd.concat(all_clusters, ignore_index=True)

# Convert trial names to numeric IDs based on their order in the DataFrame
unique_trials = output_df["sim_id"].unique()
trial_name_to_number = {name: i for i, name in enumerate(unique_trials)}
output_df["sim_id"] = output_df["sim_id"].map(trial_name_to_number)


def reorder_clusters(df):
    def reorder(group):
        means = group.groupby('cluster_id')['opinion_value'].mean().sort_values()
        mapping = {old: new for new, old in enumerate(means.index)}
        group['cluster_id'] = group['cluster_id'].map(mapping)
        return group

    return df.groupby('sim_id', group_keys=False).apply(reorder)

output_df = reorder_clusters(output_df)

# Export the output
output_df.to_csv(output_filepath , index=False)
output_df.head()