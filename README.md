# Elastic UPF Selection for NextG Cellular Networks

Artifact for the paper *Elastic UPF Selection for NextG Cellular Networks*
(Colocrese, Mohan, Iosifidis, Kuipers).

This repository contains:

1. **`optimization/`** — the offline framework that formulates per-session UPF
   selection as a Capacitated Facility Location Problem (CFLP) and evaluates it
   against heuristic baselines.
2. **`characterization/`** — the controlled cross-tier power/latency
   measurements from which the per-tier power models are derived.


## Layout

```
optimization/
  config.py              Topology, traffic, and power-model parameters
  network_utils.py       Operator-derived topology generator 
  session_utils.py       PDU-session generator (traffic split, demand, QoS)
  optimization.py        CFLP/MILP model (single-solver entry point, MOSEK)
  baseline_methods.py    Source-/Destination-centric + centralized baselines
  main_comparison.py     Multi-seed sweep driver (MOSEK + Gurobi cross-check)

characterization/
  power_profiles.csv     Throughput→power/CPU fit points (per tier)
  fit_power_models.py    Fits the linear/piecewise models
  power_models.csv       Resulting coefficients (= config.POWER_COEFFICIENTS)
```

The characterization ships the optimizer inputs: `power_profiles.csv`
holds the throughput→power/CPU points measured on each tier, and
`fit_power_models.py` regenerates the coefficients in `power_models.csv` (the
values used in `optimization/config.py`). See `characterization/README.md`.

## Power models

The optimizer consumes one **linear** model per tier — idle power plus a
marginal coefficient per Mbps. These are the values in `optimization/config.py` (`POWER_COEFFICIENTS`) 

| Tier            | Idle (W) | Marginal (W/Mbps) | Source dir                        |
| --------------- | -------- | ----------------- | --------------------------------- |
| Far-edge (RPi)  | 4.20     | 5.44e-3           | `power_profiles.csv` (far_edge_rpi) |
| Edge (NUC)      | 16.05    | 7.5e-3            | `power_profiles.csv` (edge_nuc; high-load segment) |
| Cloud (server)  | 248.46   | 5.1e-3            | `power_profiles.csv` (cloud_server) |

The edge tier is measured as a two-regime piecewise-linear curve with a
breakpoint at 225 Mbps. The optimizer uses the **high-load segment** 
(α_E,2 = 16.05, β_E,2 = 7.5e-3) as a single-line approximation, since
an active edge UPF aggregates many sessions and operates above the breakpoint.

## Reproducing

### Setup

```
python >= 3.10
pip install -r requirements.txt
```

The MILP needs a solver. `main_comparison.py` cross-validates with **MOSEK** and
**Gurobi** (both free for academics); `optimization.py` alone runs MOSEK only.

### Optimization sweep

```
cd optimization
python main_comparison.py      # sweeps sessions × γ × seeds, writes CSVs
```

Knobs live in `config.py`: `GAMMA_VALUES`, `RANDOM_SEEDS`, device capacities,
`POWER_COEFFICIENTS`. The driver sweeps sessions `{50,100,500,1000,2500,5000}`,
γ ∈ `{0.01,0.1,1,10,100}`, α = 1, 600 s solver limit, over 10 seeds.


## Notes and caveats

- **Topology is operator-derived; session demand and QoS classes are
  synthetic.** Sessions split 25/25/25/25 across four traffic patterns;
  per-session demand ~ `Uniform[1,100]` Mbps; latency-sensitivity weight ∈
  `{0,1,5,10}`. Operators with measured profiles can substitute their own
  distributions without changing the model.
- **UPF-introduced latency is modeled as constant** per tier in the optimizer.


