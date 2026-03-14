# Opinion Polarization from Compression-Based Decision Making

This repository contains the source code for the agent-based model described in:

> **Opinion polarization from compression-based decision making where agents optimize local complexity and global simplicity**
>
> Alina Dubovskaya, David J. P. O'Sullivan, Michael Quayle
>
> *PLOS Complex Systems* (2025)

## Model Overview

The model simulates opinion dynamics in a population of agents who make decisions based on Shannon entropy. Each agent holds a continuous opinion in [0, 1]. An agent adopts a new opinion only if it increases the ratio of local entropy (within their social group) to global entropy (across the entire population). This mechanism balances local group diversity against global opinion simplification, leading to the emergence of distinct heterogeneous opinion clusters.

## Repository Structure

```
├── model/                          # Core simulation package
│   ├── __init__.py
│   ├── base_model.py               # Base class: entropy calculation, preferential selection, compressibility
│   ├── model_without_memory.py     # Main model: entropy-ratio decision rule
│   └── topology.py                 # Fully-mixed interaction topology
├── tools.py                        # Utility functions: cluster detection, entropy, data processing
├── data/                           # Input data (initial conditions)
├── results/                        # Simulation output (selected CSVs tracked)
├── run_simulation.py               # Single simulation run
├── run_multiple_final_only.py      # Multiple runs, final state only
├── run_multiple_with_intermediate.py  # Multiple runs with intermediate snapshots
├── run_cluster_experiment.py       # Cluster variance experiments
├── prepare_cluster_data.py         # Data preprocessing for cluster analysis
├── plot_evolution.py               # Opinion evolution figures
├── plot_evolution_fixed_group.py   # Evolution plots with fixed local group size
├── plot_cluster_boxplot.py         # Cluster property box plots
├── plot_cluster_boxplot_fixed_sim5.py  # Fixed simulation cluster analysis
└── plot_surviving_opinions.py      # Opinion persistence analysis
```

## Installation

Requires Python 3.9+.

```bash
git clone https://github.com/daxelka/opinion-polarization-compression-abm.git
cd opinion-polarization-compression-abm
pip install -r requirements.txt
```

## Usage

### Run a single simulation

```bash
python run_simulation.py
```

This runs 1,000 agents for 10^6 Monte Carlo steps with default parameters (`n_bins=10`, `local_group_size=200`) and saves the opinion evolution to `results/`.

<!-- ### Run multiple simulations

```bash
python run_multiple_final_only.py
```

Runs 10 independent simulations with shared initial conditions and saves final opinion distributions. -->

<!-- ### Generate figures

The `plot_*.py` scripts reproduce the figures in the paper. They read from `results/` and output publication-ready TIFF images to `img/paper/`. -->

## Model Parameters

| Parameter | Description | Default |
|---|---|---|
| `n_nodes` | Number of agents | 1000 |
| `n_local_bins` | Number of bins for local entropy calculation | 10 |
| `n_global_bins` | Number of bins for global entropy calculation | 10 |
| `local_group_size` | Size of each agent's local opinion neighborhood | 200 |
| `alpha` | Preferential selection exponent (0 = uniform) | 0 |
| `const` | Preferential selection constant | 0 |

<!-- ## Reproducing Paper Results

The key parameter variations explored in the paper:

- **Local group size**: 100, 200, 300, 500 agents
- **Number of bins**: 7, 10, 15, 100
- **Simulation length**: 10^6 Monte Carlo steps per run
- **Repeats**: 10 independent runs per parameter setting -->

<!-- Pre-computed results for the main figures are included in `results/` (tracked via git). -->

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

<!-- ## Citation

If you use this code, please cite:

```bibtex
@article{dubovskaya2025opinion,
  title={Opinion polarization from compression-based decision making where agents optimize local complexity and global simplicity},
  author={Dubovskaya, Alina and O'Sullivan, David J. P. and Quayle, Michael},
  journal={PLOS Complex Systems},
  year={2025}
}
``` -->
