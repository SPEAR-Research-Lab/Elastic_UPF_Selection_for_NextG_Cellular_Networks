# Power characterization → optimizer inputs

This folder contains the per-tier throughput→power measurements and the linear/piecewise models fit from them.

## Files

- **`power_profiles.csv`** — the fit inputs. One row per tested throughput
  level, per tier:

  | column            | meaning                                            |
  | ----------------- | -------------------------------------------------- |
  | `tier`            | `far_edge_rpi` / `edge_nuc` / `cloud_server`       |
  | `throughput_mbps` | achieved aggregate throughput (iPerf)              |
  | `power_w`         | mean whole-device socket power during that run     |
  | `cpu_pct`         | mean UPF-process CPU during that run               |

  Power was averaged over each iPerf run's time window from the raw
  `power.csv`; edge/cloud aggregate three concurrent UE streams.

- **`fit_power_models.py`** — fits the measured points and writes
  `power_models.csv` (the coefficients used by the optimizer).
  Far-edge and cloud get a single linear fit; edge gets a two-regime
  piecewise-linear fit about the 225 Mbps breakpoint. Run:

  ```
  python fit_power_models.py
  ```

- **`power_models.csv`** — the resulting coefficients, i.e. exactly the values
  placed in `optimization/config.py` (`POWER_COEFFICIENTS`).

## Models

`P(T) = alpha + beta * T` (W), with T in Mbps.

| Tier            | Regime | alpha (W) | beta (W/Mbps) |
| --------------- | ------ | --------- | ------------- |
| Far-edge (RPi)  | linear | 4.18      | 5.42e-3       |
| Edge (NUC)      | low    | 9.58      | 3.46e-2       |
| Edge (NUC)      | high   | 15.74     | 7.07e-3       |
| Cloud (server)  | linear | 248.98    | 1.04e-2       |

The optimizer uses one linear coefficient per tier: far-edge and cloud
directly, and for the edge tier the **high-load** segment (an active edge UPF
aggregates many sessions and runs above the 225 Mbps breakpoint).


