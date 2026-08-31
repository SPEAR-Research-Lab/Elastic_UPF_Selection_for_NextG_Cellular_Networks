#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fit the per-tier UPF power models used by the optimizer.

Input : power_profiles.csv  (tier, throughput_mbps, power_w, cpu_pct)
Output: power_models.csv     (tier, regime, alpha_w, beta_w_per_mbps, breakpoint_mbps)

These fitted coefficients are the per-tier power models P(T) = alpha + beta * T
(W, T in Mbps) that parameterize the optimizer (optimization/config.py).

Models:
  Far-edge, Cloud : single linear fit
  Edge            : two-regime piecewise-linear with breakpoint at 225 Mbps.
                    The optimizer uses the HIGH-load segment as its single
                    linear coefficient (an active edge UPF aggregates many
                    sessions and operates above the breakpoint).
"""
import numpy as np
import pandas as pd

BREAKPOINT_EDGE = 225.0  # Mbps


def fit_linear(sub):
    beta, alpha = np.polyfit(sub["throughput_mbps"], sub["power_w"], 1)
    return float(alpha), float(beta)


def main():
    df = pd.read_csv("power_profiles.csv")
    rows = []

    # Far-edge and cloud: single linear fit
    for tier in ("far_edge_rpi", "cloud_server"):
        a, b = fit_linear(df[df.tier == tier])
        rows.append((tier, "linear", round(a, 3), round(b, 6), ""))

    # Edge: piecewise-linear about the 225 Mbps breakpoint
    ed = df[df.tier == "edge_nuc"]
    lo = ed[ed.throughput_mbps <= BREAKPOINT_EDGE]
    hi = ed[ed.throughput_mbps > BREAKPOINT_EDGE]
    a1, b1 = fit_linear(lo)
    a2, b2 = fit_linear(hi)
    rows.append(("edge_nuc", "low",  round(a1, 3), round(b1, 6), BREAKPOINT_EDGE))
    rows.append(("edge_nuc", "high", round(a2, 3), round(b2, 6), BREAKPOINT_EDGE))

    out = pd.DataFrame(rows, columns=["tier", "regime", "alpha_w",
                                      "beta_w_per_mbps", "breakpoint_mbps"])
    out.to_csv("power_models.csv", index=False)

    print("Fitted power models (written to power_models.csv):\n")
    print(out.to_string(index=False))

    # Far-edge safe throughput: where CPU crosses 70% (used for the T_u cap)
    fe = df[df.tier == "far_edge_rpi"].sort_values("throughput_mbps")
    below = fe[fe.cpu_pct <= 70]["throughput_mbps"].max()
    above = fe[fe.cpu_pct > 70]["throughput_mbps"].min()
    print(f"\nFar-edge 70%-CPU safe throughput lies between "
          f"{below:.0f} and {above:.0f} Mbps (use for T_u).")


if __name__ == "__main__":
    main()
