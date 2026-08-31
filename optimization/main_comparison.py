#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive comparison across sessions, gamma values, and solvers
WITH MULTI-SEED SUPPORT - Runs multiple random seeds sequentially
"""

import numpy as np
import random
import networkx as nx
import pandas as pd
from datetime import datetime
import time
import os

from config import *
from network_utils import create_network, build_subnets_and_mappings
from session_utils import generate_structured_session_requests
from optimization import prepare_optimization_data, solve_upf_optimization
from baseline_methods import run_all_heuristics


def solve_with_solver(opt_data, alpha, gamma, solver_name, verbose=False, timeout=3600):
    """
    Solve optimization with specified solver
    
    Args:
        opt_data: Optimization data
        alpha: Alpha parameter
        gamma: Gamma parameter
        solver_name: 'MOSEK' or 'GUROBI'
        verbose: Print solver output
        timeout: Solver timeout in seconds
        
    Returns:
        dict: Solution with additional metadata
    """
    import cvxpy as cp
    
    S = opt_data['sessions_ids']
    U = opt_data['upf_devices']
    T_s = opt_data['session_throughput']
    d_s_min = opt_data['min_delay']
    d_fix = opt_data['edge_delay_data']
    T_u = opt_data['T_u']
    beta_s = opt_data['session_latency_weight']
    w_u = opt_data['w_u']
    p_u = opt_data['p_u']
    l_u = opt_data['l_u']
    delay_diff_max = opt_data['delay_diff_max']
    
    # Build numpy vectors/matrices
    T_s_vec = np.array([T_s[s] for s in S])
    beta_vec = np.array([beta_s[s] for s in S])
    T_u_vec = np.array([T_u[u] for u in U])
    w_vec = np.array([w_u[u] for u in U])
    p_vec = np.array([p_u[u] for u in U])
    l_vec = np.array([l_u[u] for u in U])
    delay_matrix = np.array([[d_fix[(s, u)] for u in U] for s in S])
    min_delay_vec = np.array([d_s_min[s] for s in S])
    
    # Convert to CVXPY constants
    T_s_const = cp.Constant(T_s_vec)
    beta_const = cp.Constant(beta_vec)
    T_u_const = cp.Constant(T_u_vec)
    w_const = cp.Constant(w_vec)
    p_const = cp.Constant(p_vec)
    l_const = cp.Constant(l_vec)
    delay_const = cp.Constant(delay_matrix)
    min_delay_const = cp.Constant(min_delay_vec)
    delay_diff_max_const = float(delay_diff_max)
    
    # Decision variables
    x = cp.Variable((len(S), len(U)), boolean=True)
    y = cp.Variable(len(U), boolean=True)
    
    # Constraints
    constraints = []
    constraints.append(cp.sum(x, axis=1) == 1)
    throughput_per_upf = cp.matmul(T_s_const, x)
    constraints.append(throughput_per_upf <= cp.multiply(T_u_const, y))
    
    # Objective function
    delay_terms_const = cp.Constant((delay_matrix + l_vec - min_delay_vec[:, None]) / delay_diff_max_const)
    delay_gain_expr = cp.sum(cp.multiply(cp.reshape(beta_const, (len(S), 1)), 
                                         cp.multiply(x, delay_terms_const))) / len(S)
    
    power_fixed_expr = cp.sum(cp.multiply(w_const, y))
    p_times_T = cp.Constant((p_vec * T_s_vec[:, None]))
    power_dynamic_expr = cp.sum(cp.multiply(x, p_times_T))
    total_power_expr = power_fixed_expr + power_dynamic_expr
    power_norm = total_power_expr / float(np.sum(w_vec))
    
    objective = cp.Minimize(alpha * delay_gain_expr + gamma * power_norm)
    prob = cp.Problem(objective, constraints)
    
    # Solve with appropriate solver
    solve_start = time.time()
    
    try:
        if solver_name == 'MOSEK':
            prob.solve(
                solver=cp.MOSEK,
                verbose=verbose,
                mosek_params={
                    'MSK_DPAR_OPTIMIZER_MAX_TIME': float(timeout),
                    'MSK_IPAR_LOG': 10 if verbose else 0
                }
            )
        elif solver_name == 'GUROBI':
            prob.solve(
                solver=cp.GUROBI,
                verbose=verbose,
                TimeLimit=timeout,
                MIPGap=1e-4,
                LogToConsole=1 if verbose else 0
            )
        else:
            raise ValueError(f"Unknown solver: {solver_name}")
            
        solve_time = time.time() - solve_start
        
    except Exception as e:
        solve_time = time.time() - solve_start
        print(f"  [Error] Solver {solver_name} failed: {str(e)}")
        return {
            'status': 'error',
            'solver': solver_name,
            'solve_time': solve_time,
            'error': str(e)
        }
    
    # Check if solution is valid
    interrupted = False
    if prob.status == 'user_limit':
        interrupted = True
        print(f"  [Warning] Solver reached time limit")
    
    if prob.status not in ["optimal", "optimal_inaccurate", "user_limit"]:
        return {
            'status': prob.status,
            'solver': solver_name,
            'solve_time': solve_time,
            'objective_value': np.nan,
            'interrupted': False
        }
    
    # Extract solution
    x_val = np.array(x.value)
    y_val = np.array(y.value)
    
    # Compute metrics
    delay_term_value = np.sum(beta_vec[:, None] * x_val * 
                             (delay_matrix + l_vec - min_delay_vec[:, None]) / 
                             delay_diff_max) / len(S)
    power_term_value = (np.sum(w_vec * y_val) + 
                       np.sum(p_vec * (x_val * T_s_vec[:, None]))) / np.sum(w_vec)
    
    assigned_upfs = np.argmax(x_val, axis=1)
    session_latencies = (delay_matrix[np.arange(len(S)), assigned_upfs] + 
                        l_vec[assigned_upfs]).tolist()
    
    avg_latency = float(np.mean(session_latencies))
    avg_min_latency = float(np.mean(min_delay_vec))
    latency_gain_pct = 100 * (avg_latency / avg_min_latency - 1)
    active_upfs = int(np.sum(y_val >= 0.5))
    
    # Compute actual total power
    total_power = 0.0
    for i, u in enumerate(U):
        if y_val[i] >= 0.5:
            actual_load = np.sum(x_val[:, i] * T_s_vec)
            total_power += w_u[u] + p_u[u] * actual_load
    
    # Compute optimality gap if available
    optimality_gap = np.nan
    
    if interrupted:
        optimality_gap = np.nan
    else:
        if prob.status == 'optimal':
            optimality_gap = 0.0
        elif prob.status == 'optimal_inaccurate':
            optimality_gap = 1e-4
    
    return {
        'status': prob.status,
        'solver': solver_name,
        'objective_value': prob.value,
        'delay_term': delay_term_value,
        'power_term': power_term_value,
        'x_assignments': x_val,
        'y_active': y_val,
        'session_latencies': session_latencies,
        'avg_latency': avg_latency,
        'avg_min_latency': avg_min_latency,
        'latency_gain_pct': latency_gain_pct,
        'active_upfs': active_upfs,
        'total_power': total_power,
        'solve_time': solve_time,
        'optimality_gap': optimality_gap,
        'interrupted': interrupted
    }


def run_comprehensive_comparison(network, session_requests, num_sessions, 
                                alpha, gamma_values, solvers, timeout=3600):
    """
    Run comparison for one session count configuration
    
    Args:
        network: Network graph
        session_requests: List of sessions
        num_sessions: Number of sessions
        alpha: Alpha parameter
        gamma_values: List of gamma values to test
        solvers: List of solver names ['MOSEK', 'GUROBI']
        timeout: Solver timeout
        
    Returns:
        dict: Results for all methods
    """
    print(f"\n{'='*80}")
    print(f"RUNNING COMPARISON: {num_sessions} sessions, α={alpha}")
    print(f"{'='*80}\n")
    
    all_results = {}
    
    # Step 1: Run heuristics
    print("Step 1: Running heuristic methods...")
    heuristic_results = run_all_heuristics(network, session_requests)
    
    # ADD ACTIVE UPF IDS TO HEURISTIC RESULTS
    for method_key, result in heuristic_results.items():
        result['active_upf_ids'] = result['active_upfs']  # List of UPF names
    
    all_results.update(heuristic_results)
    
    # Step 2: Prepare optimization data (once)
    print("\nStep 2: Preparing optimization data...")
    opt_data = prepare_optimization_data(network, session_requests)
    
    # Step 3: Run optimizations for each gamma and solver
    print("\nStep 3: Running optimizations...")
    
    for gamma in gamma_values:
        for solver in solvers:
            result_key = f'opt_γ{gamma}_{solver.lower()}'
            
            print(f"\n{'-'*80}")
            print(f"Running: α={alpha}, γ={gamma}, Solver={solver}")
            print(f"{'-'*80}")
            
            opt_solution = solve_with_solver(
                opt_data, alpha, gamma, solver, 
                verbose=False, timeout=timeout
            )
            
            if opt_solution['status'] in ['optimal', 'optimal_inaccurate', 'user_limit']:
                # Convert to standard format
                upf_devices_list = opt_data['upf_devices']
                active_upfs = [upf_devices_list[i] 
                             for i in range(len(upf_devices_list)) 
                             if opt_solution['y_active'][i] >= 0.5]
                
                all_results[result_key] = {
                    'method': f'Opt (α={alpha}, γ={gamma}, {solver})',
                    'solver': solver,
                    'alpha': alpha,
                    'gamma': gamma,
                    'active_upfs': active_upfs,
                    'active_upf_ids': active_upfs,  # ADDED FOR VISUALIZATION
                    'num_active_upfs': opt_solution['active_upfs'],
                    'total_power': opt_solution['total_power'],
                    'avg_latency': opt_solution['avg_latency'],
                    'avg_min_latency': opt_solution['avg_min_latency'],
                    'latency_gain_pct': opt_solution['latency_gain_pct'],
                    'delay_term': opt_solution['delay_term'],
                    'power_term': opt_solution['power_term'],
                    'objective_value': opt_solution['objective_value'],
                    'solve_time': opt_solution['solve_time'],
                    'optimality_gap': opt_solution.get('optimality_gap', np.nan),
                    'status': opt_solution['status'],
                    'failed_sessions': [],
                    'num_failed': 0,
                    'session_latencies': opt_solution['session_latencies']
                }
                
                print(f"✓ Completed: Objective={opt_solution['objective_value']:.6f}, "
                      f"Time={opt_solution['solve_time']:.2f}s, "
                      f"Active UPFs={opt_solution['active_upfs']}, "
                      f"Status={opt_solution['status']}")
                
                if opt_solution.get('interrupted', False):
                    print(f"  ⚠ Warning: Solution interrupted by time limit")
                
            else:
                print(f"✗ Failed: Status={opt_solution['status']}")
                all_results[result_key] = {
                    'method': f'Opt (α={alpha}, γ={gamma}, {solver})',
                    'solver': solver,
                    'alpha': alpha,
                    'gamma': gamma,
                    'status': opt_solution['status'],
                    'solve_time': opt_solution.get('solve_time', np.nan),
                    'num_active_upfs': np.nan,
                    'total_power': np.nan,
                    'avg_latency': np.nan,
                    'latency_gain_pct': np.nan,
                    'delay_term': np.nan,
                    'power_term': np.nan,
                    'objective_value': np.nan,
                    'optimality_gap': np.nan,
                    'active_upf_ids': []  # ADDED FOR VISUALIZATION
                }
    
    return all_results


def create_comprehensive_table(all_results_by_sessions, session_counts, 
                               alpha, gamma_values, solvers, seed=None):
    """
    Create comprehensive comparison table with UPF activation tracking
    
    Args:
        seed: Random seed used (optional, for tracking)
    
    Returns:
        pd.DataFrame: Complete results table
    """
    rows = []
    
    for num_sessions in session_counts:
        results = all_results_by_sessions[num_sessions]
        
        # Add heuristic results
        for method_key in ['closest_to_source', 'central_upf', 'closest_to_destination']:
            if method_key in results:
                r = results[method_key]
                method_name = {
                    'closest_to_source': 'Closest to Source',
                    'central_upf': 'Central UPF',
                    'closest_to_destination': 'Closest to Destination'
                }[method_key]
                
                row = {
                    'Num Sessions': num_sessions,
                    'Method': method_name,
                    'Method Key': method_key,
                    'Alpha': '-',
                    'Gamma': '-',
                    'Solver': 'Heuristic',
                    'Active UPFs': r['num_active_upfs'],
                    'Active UPF IDs': str(r.get('active_upf_ids', [])),
                    'Latency Term': '-',
                    'Power Term': '-',
                    'Total Power (W)': round(r['total_power'], 2),
                    'Power per Session (W)': round(r['total_power'] / num_sessions, 4),
                    'Avg Latency (ms)': round(r['avg_latency'], 4),
                    'Avg Latency Gain (%)': round(r['latency_gain_pct'], 2),
                    'Execution Time (s)': '-',
                    'Optimality Gap (%)': '-',
                    'Status': 'Complete'
                }
                if seed is not None:
                    row['Seed'] = seed
                rows.append(row)
        
        # Add optimization results
        for gamma in gamma_values:
            for solver in solvers:
                result_key = f'opt_γ{gamma}_{solver.lower()}'
                
                if result_key in results:
                    r = results[result_key]
                    
                    row = {
                        'Num Sessions': num_sessions,
                        'Method': f'Optimization',
                        'Method Key': result_key,
                        'Alpha': alpha,
                        'Gamma': gamma,
                        'Solver': solver,
                        'Active UPFs': r.get('num_active_upfs', '-'),
                        'Active UPF IDs': str(r.get('active_upf_ids', [])),
                        'Latency Term': round(r.get('delay_term', np.nan), 6) if not np.isnan(r.get('delay_term', np.nan)) else '-',
                        'Power Term': round(r.get('power_term', np.nan), 6) if not np.isnan(r.get('power_term', np.nan)) else '-',
                        'Total Power (W)': round(r.get('total_power', np.nan), 2) if not np.isnan(r.get('total_power', np.nan)) else '-',
                        'Power per Session (W)': round(r.get('total_power', np.nan) / num_sessions, 4) if not np.isnan(r.get('total_power', np.nan)) else '-',
                        'Avg Latency (ms)': round(r.get('avg_latency', np.nan), 4) if not np.isnan(r.get('avg_latency', np.nan)) else '-',
                        'Avg Latency Gain (%)': round(r.get('latency_gain_pct', np.nan), 2) if not np.isnan(r.get('latency_gain_pct', np.nan)) else '-',
                        'Execution Time (s)': round(r.get('solve_time', np.nan), 2) if not np.isnan(r.get('solve_time', np.nan)) else '-',
                        'Optimality Gap (%)': round(r.get('optimality_gap', np.nan) * 100, 4) if not np.isnan(r.get('optimality_gap', np.nan)) else 'N/A',
                        'Status': r.get('status', 'unknown') + (' [TIMEOUT]' if r.get('interrupted', False) else '')
                    }
                    if seed is not None:
                        row['Seed'] = seed
                    rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Reorder columns for better readability
    column_order = ['Num Sessions', 'Method', 'Method Key', 'Alpha', 'Gamma', 'Solver',
                    'Active UPFs', 'Active UPF IDs', 'Latency Term', 'Power Term', 
                    'Total Power (W)', 'Power per Session (W)', 'Avg Latency (ms)',
                    'Avg Latency Gain (%)', 'Execution Time (s)', 
                    'Optimality Gap (%)', 'Status']
    
    if seed is not None:
        column_order.insert(0, 'Seed')
    
    df = df[column_order]
    
    return df


def run_single_seed(seed, session_counts, alpha, gamma_values, solvers, timeout, output_dir='plots'):
    """
    Run complete experiment for a single seed and save results
    
    Args:
        seed: Random seed
        session_counts: List of session counts to test
        alpha: Alpha parameter
        gamma_values: List of gamma values
        solvers: List of solvers
        timeout: Solver timeout
        output_dir: Output directory
        
    Returns:
        tuple: (all_results_by_sessions, df, summary_df)
    """
    print("\n" + "="*80)
    print(f"RUNNING SEED {seed}")
    print("="*80 + "\n")
    
    # Set random seeds for this run
    np.random.seed(seed)
    random.seed(seed)
    
    # Create network (once per seed)
    print(f"[Seed {seed}] Creating network topology...")
    network = create_network()
    subnets, subnet_to_bigrouter = build_subnets_and_mappings(network)
    
    # Get all UPF devices for visualization
    upf_devices = sorted([n for n in network.nodes 
                         if network.nodes[n]['type'] in ['DC', 'BigRouter', 'SmallRouter']])
    
    # Store all results
    all_results_by_sessions = {}
    
    # Run for each session count
    for num_sessions in session_counts:
        print(f"\n{'#'*80}")
        print(f"# [Seed {seed}] SESSION COUNT: {num_sessions}")
        print(f"{'#'*80}\n")
        
        # Generate sessions for this count
        session_requests = generate_structured_session_requests(
            network, subnets, subnet_to_bigrouter, num_sessions
        )
        
        # Run all comparisons
        results = run_comprehensive_comparison(
            network, session_requests, num_sessions,
            alpha, gamma_values, solvers, timeout
        )
        
        all_results_by_sessions[num_sessions] = results
    
    # Create comprehensive table
    print(f"\n[Seed {seed}] Creating comprehensive results table...")
    df = create_comprehensive_table(
        all_results_by_sessions, session_counts,
        alpha, gamma_values, solvers, seed=seed
    )
    
    # Display table
    print(f"\n[Seed {seed}] COMPREHENSIVE RESULTS:")
    print(df.to_string(index=False))
    
    # Save to CSV with seed in filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = f'comprehensive_comparison_α{alpha}_seed{seed}_{timestamp}.csv'
    df.to_csv(csv_filename, index=False)
    print(f"\n✓ Results saved to: {csv_filename}")
    
    # Create UPF activation visualizations (optional module, not required to reproduce results)
    print(f"\n[Seed {seed}] Creating UPF activation visualizations...")
    try:
        from upf_activation_visualization import (
            plot_active_upfs_heatmap_all_methods,
            plot_active_upfs_by_method_type,
            create_upf_activation_summary
        )
    except ImportError:
        print("  [Info] upf_activation_visualization not present; skipping plots (CSV results are still written).")
        return all_results_by_sessions, df, None
    
    # Create plots for each session count
    for num_sessions in session_counts:
        print(f"\n[Seed {seed}] Generating plots for {num_sessions} sessions...")
        
        plot_active_upfs_heatmap_all_methods(
            df, upf_devices, num_sessions, alpha, timeout
        )
        
        plot_active_upfs_by_method_type(
            df, upf_devices, num_sessions, alpha, timeout
        )
    
    # Create activation summary
    summary_df = create_upf_activation_summary(df, upf_devices, output_dir)
    
    # Rename summary file to include seed
    old_summary_files = [f for f in os.listdir(output_dir) if f.startswith('upf_activation_summary_')]
    if old_summary_files:
        latest_summary = max(old_summary_files, key=lambda f: os.path.getmtime(os.path.join(output_dir, f)))
        old_path = os.path.join(output_dir, latest_summary)
        new_summary_filename = f'upf_activation_summary_seed{seed}_{timestamp}.csv'
        new_path = os.path.join(output_dir, new_summary_filename)
        os.rename(old_path, new_path)
        print(f"✓ Summary saved to: {new_path}")
    
    print(f"\n[Seed {seed}] ✓ COMPLETED - All results saved")
    
    return all_results_by_sessions, df, summary_df


def main_comprehensive():
    """
    Main execution function with MULTI-SEED support
    Runs each seed sequentially and saves outputs after each run
    """
    # Configuration
    SESSION_COUNTS = [50, 100, 500, 1000, 2500, 5000]
    ALPHA = 1
    GAMMA_VALUES = [0.01, 0.1, 1, 10, 100]
    SOLVERS = ['MOSEK', 'GUROBI']
    TIMEOUT = 600  # 10 minutes per optimization
    
    # Use seeds from config
    SEEDS = RANDOM_SEEDS  # From config.py
    
    # Create output directory
    os.makedirs('plots', exist_ok=True)
    
    print("\n" + "="*80)
    print("COMPREHENSIVE UPF OPTIMIZATION COMPARISON")
    print("MULTI-SEED VERSION")
    print("="*80)
    print(f"Number of seeds: {len(SEEDS)}")
    print(f"Seeds: {SEEDS}")
    print(f"Session counts: {SESSION_COUNTS}")
    print(f"Alpha: {ALPHA}")
    print(f"Gamma values: {GAMMA_VALUES}")
    print(f"Solvers: {SOLVERS}")
    print(f"Timeout per optimization: {TIMEOUT}s")
    print("="*80 + "\n")
    
    # Store aggregated results across all seeds
    all_seeds_results = {}
    all_seeds_dfs = []
    all_seeds_summaries = []
    
    # Run for each seed
    for seed_idx, seed in enumerate(SEEDS, 1):
        print("\n" + "█"*80)
        print(f"█ SEED {seed_idx}/{len(SEEDS)}: {seed}")
        print("█"*80)
        
        seed_start_time = time.time()
        
        # Run complete experiment for this seed
        results, df, summary_df = run_single_seed(
            seed, SESSION_COUNTS, ALPHA, GAMMA_VALUES, SOLVERS, TIMEOUT
        )
        
        # Store results
        all_seeds_results[seed] = results
        all_seeds_dfs.append(df)
        all_seeds_summaries.append(summary_df)
        
        seed_duration = time.time() - seed_start_time
        print(f"\n[Seed {seed}] ✓ Completed in {seed_duration/60:.2f} minutes")
        print(f"[Progress] {seed_idx}/{len(SEEDS)} seeds completed\n")
    
    # Create combined results file with all seeds
    print("\n" + "="*80)
    print("CREATING COMBINED RESULTS ACROSS ALL SEEDS")
    print("="*80 + "\n")
    
    combined_df = pd.concat(all_seeds_dfs, ignore_index=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    combined_filename = f'comprehensive_comparison_α{ALPHA}_all_seeds_{timestamp}.csv'
    combined_df.to_csv(combined_filename, index=False)
    print(f"✓ Combined results saved to: {combined_filename}")
    
    # Summary statistics across seeds
    print("\n" + "="*80)
    print("SUMMARY STATISTICS ACROSS ALL SEEDS")
    print("="*80 + "\n")
    
    for num_sessions in SESSION_COUNTS:
        print(f"\n{'='*60}")
        print(f"Session Count: {num_sessions}")
        print(f"{'='*60}")
        
        session_df = combined_df[combined_df['Num Sessions'] == num_sessions]
        
        # Group by method and compute statistics
        for method_key in session_df['Method Key'].unique():
            method_data = session_df[session_df['Method Key'] == method_key]
            
            if len(method_data) > 0:
                print(f"\n{method_key}:")
                
                # Power statistics
                power_vals = pd.to_numeric(method_data['Total Power (W)'], errors='coerce')
                if not power_vals.isna().all():
                    print(f"  Power: {power_vals.mean():.2f} ± {power_vals.std():.2f} W")
                
                # Latency statistics
                latency_vals = pd.to_numeric(method_data['Avg Latency (ms)'], errors='coerce')
                if not latency_vals.isna().all():
                    print(f"  Latency: {latency_vals.mean():.4f} ± {latency_vals.std():.4f} ms")
                
                # Active UPFs statistics
                upf_vals = pd.to_numeric(method_data['Active UPFs'], errors='coerce')
                if not upf_vals.isna().all():
                    print(f"  Active UPFs: {upf_vals.mean():.1f} ± {upf_vals.std():.1f}")
    
    print("\n" + "="*80)
    print("ALL SEEDS EXECUTION COMPLETED")
    print("="*80)
    print(f"\nGenerated files:")
    print(f"  - Individual seed CSVs: comprehensive_comparison_α{ALPHA}_seed*_*.csv")
    print(f"  - Combined CSV: {combined_filename}")
    print(f"  - Individual summaries: plots/upf_activation_summary_seed*_*.csv")
    print(f"  - Plots saved in: plots/")
    print("="*80 + "\n")
    
    return all_seeds_results, combined_df, all_seeds_dfs


if __name__ == "__main__":
    all_results, combined_df, individual_dfs = main_comprehensive()