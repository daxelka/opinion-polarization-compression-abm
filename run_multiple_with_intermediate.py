import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import FullyMixedWithoutMemory_Model
import tools as t

# ======================
# Model parameters
# ======================

n_nodes = 2000
n_local_bins, n_global_bins = 20, 20
local_group_size = 400

time_finish = int(2e6)
n_runs = time_finish
steps_between_measurements = int(4e4)

time_steps_to_record = t.generate_time_steps(
    first_value=0,
    last_value=time_finish - 1,
    step=steps_between_measurements
)

n_repeats = 3

alpha = 0
const = 0

# ======================
# Initialize model
# ======================

model = FullyMixedWithoutMemory_Model(
    n_nodes=n_nodes,
    n_local_bins=n_local_bins,
    n_global_bins=n_global_bins,
    local_group_size=local_group_size,
    alpha=alpha,
    const=const
)

print("Model created")

# ======================
# Containers for results
# ======================

records = []          # simulation output
initial_conditions = []  # initial opinions

# ======================
# Simulations
# ======================

for sim_id in range(n_repeats):
    t0 = time.time()

    # --- draw initial condition ---
    initial_opinion = np.round(np.random.uniform(0, 1, n_nodes), 4)

    # store initial conditions
    initial_conditions.append(
        pd.DataFrame({
            "opinions": initial_opinion,
            "sim_id": sim_id
        })
    )

    model.set_initial_opinion(initial_opinion)

    # --- run model ---
    opinion_matrix = model.run(n_runs, time_steps_to_record)

    # --- store intermediate results ---
    for opinions_at_time, time_cur in zip(opinion_matrix, time_steps_to_record):
        records.append(
            pd.DataFrame({
                "opinions": opinions_at_time,
                "time": time_cur,
                "sim_id": sim_id
            })
        )

    print(f"Simulation {sim_id} done, elapsed time: {time.time() - t0:.2f}s")

# ======================
# Build final dataframes
# ======================

results_df = pd.concat(records, ignore_index=True)
init_df = pd.concat(initial_conditions, ignore_index=True)

# ======================
# Save results
# ======================

base_name = (
    f"multiple_runs_evolution_n_bins_{n_local_bins}"
    f"_l_group_size_{local_group_size}"
    f"_pref_selection_alpha_{alpha}"
    f"_c_{const}"
    f"_steps_{time_finish}"
    f"_repeats_{n_repeats}"
)

results_df.to_csv("results/" + base_name + "_results.csv", index=False)
init_df.to_csv("results/" + base_name + "_initial_conditions.csv", index=False)

print("Results saved")

# ======================
# Optional visualization
# ======================

# results_df.hist(column="opinions", by="sim_id", bins=30, figsize=(10, 8))
# plt.tight_layout()
# plt.show()

unique_opinions_df = (
    results_df
    .groupby(["sim_id", "time"])["opinions"]
    .nunique()
    .reset_index(name="n_unique_opinions")
)

plt.figure(figsize=(10, 6))

for sim_id, df_sim in unique_opinions_df.groupby("sim_id"):
    plt.plot(
        df_sim["time"],
        df_sim["n_unique_opinions"],
        label=f"sim {sim_id}",
        alpha=0.8
    )

plt.xlabel("Time")
plt.ylabel("Number of unique opinions")
plt.title("Evolution of unique opinions over time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

mean_unique = (
    unique_opinions_df
    .groupby("time")["n_unique_opinions"]
    .mean()
    .reset_index()
)

plt.plot(mean_unique["time"], mean_unique["n_unique_opinions"], linewidth=3)

stats = unique_opinions_df.groupby("time")["n_unique_opinions"].agg(["mean", "std"])
plt.fill_between(stats.index,
                 stats["mean"] - stats["std"],
                 stats["mean"] + stats["std"],
                 alpha=0.3)
plt.show()

# Aggregate mean and standard deviation across simulations
stats_df = (
    unique_opinions_df
    .groupby("time")["n_unique_opinions"]
    .agg(["mean", "std"])
    .reset_index()
)

plt.figure(figsize=(10, 6))

# Mean trajectory
plt.plot(
    stats_df["time"],
    stats_df["mean"],
    linewidth=3,
    label="Simulation average"
)

# Optional uncertainty band (±1 std)
plt.fill_between(
    stats_df["time"],
    stats_df["mean"] - stats_df["std"],
    stats_df["mean"] + stats_df["std"],
    alpha=0.3,
    label="±1 std"
)

# Larger fonts
plt.xlabel("Time (simulation steps)", fontsize=16)
plt.ylabel("Average number of unique opinions", fontsize=16)
plt.title("Average evolution of unique opinions over simulations", fontsize=18)

plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.tight_layout()
plt.show()