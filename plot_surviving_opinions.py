import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

# PLOS Complex Systems figure requirements: Arial font, TIFF with LZW, max 2250px wide at 300 dpi
matplotlib.rcParams['font.family'] = 'Arial'


def plot_avg_unique_opinions(
    results_df,
    title="Average evolution of unique opinions over simulations",
    save=False,
    filename=None,
    dpi=300,
    logx=False,
    logy=False,
    figsize=(7.5, 5.5)
):
    """
    Plot the average number of unique opinions over time with ±1 std band.

    Parameters
    ----------
    results_df : pandas.DataFrame
        Must contain columns: ['sim_id', 'time', 'opinions']
    title : str
        Plot title
    save : bool
        Whether to save the figure to disk
    filename : str or None
        Path to save the figure (required if save=True)
    dpi : int
        Resolution for saved figure
    logx : bool
        Use logarithmic scale on x-axis
    logy : bool
        Use logarithmic scale on y-axis
    figsize : tuple
        Figure size in inches
    """

    if save and filename is None:
        raise ValueError("filename must be provided when save=True")

    plt.figure(figsize=figsize)

    unique_opinions_df = (
        results_df
        .groupby(["sim_id", "time"])["opinions"]
        .nunique()
        .reset_index(name="n_unique_opinions")
    )

    # Aggregate mean and standard deviation across simulations
    stats_df = (
        unique_opinions_df
        .groupby("time")["n_unique_opinions"]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Mean trajectory
    plt.plot(
        stats_df["time"],
        stats_df["mean"],
        linewidth=3,
        label="Simulation average"
    )

    # Uncertainty band
    plt.fill_between(
        stats_df["time"],
        stats_df["mean"] - stats_df["std"],
        stats_df["mean"] + stats_df["std"],
        alpha=0.3,
        label="±1 std"
    )

    # Axis scales
    if logx:
        plt.xscale("log")
    if logy:
        plt.yscale("log")

    # Labels and styling
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x/1e4)}'))
    plt.xlabel(r"Time ($\times 10^4$)", fontsize=16)
    plt.ylabel("Average number of unique opinions", fontsize=16)
    plt.title(title, fontsize=18)

    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)

    plt.tight_layout()

    # Save if requested
    if save:
        # os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=dpi, bbox_inches="tight", format='tiff',
                    facecolor='white', pil_kwargs={'compression': 'tiff_lzw'})
        # PLOS requires RGB (no alpha channel) — convert RGBA to RGB
        from PIL import Image
        img = Image.open(filename)
        if img.mode != 'RGB':
            img.convert('RGB').save(filename, compression='tiff_lzw')

    plt.show()



def plot_unique_opinions_per_sim(
    results_df,
    title="Evolution of unique opinions over time",
    save=False,
    filename=None,
    dpi=300,
    logx=False,
    logy=False,
    figsize=(7.5, 5.5),
    alpha=0.8
):
    """
    Plot the number of unique opinions over time for each simulation on the same graph.

    Parameters
    ----------
    results_df : pandas.DataFrame
        Must contain columns: ['sim_id', 'time', 'opinions']
    title : str
        Plot title
    save : bool
        Whether to save the figure
    filename : str or None
        Path to save the figure (required if save=True)
    dpi : int
        Resolution for saved figure
    logx : bool
        Use logarithmic scale on x-axis
    logy : bool
        Use logarithmic scale on y-axis
    figsize : tuple
        Figure size in inches
    alpha : float
        Line transparency
    """

    if save and filename is None:
        raise ValueError("filename must be provided when save=True")

    # --- Aggregate unique opinions ---
    unique_opinions_df = (
        results_df
        .groupby(["sim_id", "time"])["opinions"]
        .nunique()
        .reset_index(name="n_unique_opinions")
    )

    plt.figure(figsize=figsize)

    # --- Plot each simulation ---
    for sim_id, df_sim in unique_opinions_df.groupby("sim_id"):
        plt.plot(
            df_sim["time"],
            df_sim["n_unique_opinions"],
            label=f"sim {sim_id}",
            alpha=alpha
        )

    # --- Average curve ---
    stats_df = (
        unique_opinions_df
        .groupby("time")["n_unique_opinions"]
        .agg(["mean", "std"])
        .reset_index()
    )
    plt.plot(
        stats_df["time"],
        stats_df["mean"],
        color="black",
        linewidth=3,
        label="Average"
    )

    # --- Axis scales ---
    if logx:
        plt.xscale("log")
    if logy:
        plt.yscale("log")

    # --- Labels and styling ---
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x/1e4)}'))
    plt.xlabel(r"Time ($\times 10^4$)", fontsize=16)
    plt.ylabel("Number of unique opinions", fontsize=16)
    plt.title(title, fontsize=18)

    plt.legend(fontsize=12)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)

    plt.tight_layout()

    # --- Save if requested ---
    if save:
        # os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=dpi, bbox_inches="tight", format='tiff',
                    facecolor='white', pil_kwargs={'compression': 'tiff_lzw'})
        # PLOS requires RGB (no alpha channel) — convert RGBA to RGB
        from PIL import Image
        img = Image.open(filename)
        if img.mode != 'RGB':
            img.convert('RGB').save(filename, compression='tiff_lzw')

    plt.show()


# Plotting
df = pd.read_csv('results/multiple_runs_evolution_n_bins_10_l_group_size_200_pref_selection_alpha_0_c_0_steps_1000000_repeats_10_results.csv')
filename_aver = "img/avg_unique_opinions_n_bins_10_l_group_size_200_pref_selection_alpha_0_c_0_steps_1000000_repeats_10.tiff"
filename_per_sim = "img/unique_opinions_per_sim_n_bins_10_l_group_size_200_pref_selection_alpha_0_c_0_steps_1000000_repeats_10.tiff"

plot_avg_unique_opinions(
    df,
    title="Average number of unique opinions",
    save=True,
    filename= filename_aver,
    logy=True,
    dpi=300
)

plot_unique_opinions_per_sim(
    df,
    save=True,
    filename=filename_per_sim,
    logy=True,
    dpi=300
)