#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cvxpy as cp
import numpy as np
import networkx as nx
from config import *
from network_utils import compute_session_latency, compute_min_possible_latency


def compute_network_delay(G, src, upf, dst):
    """
    Returns total delay (src → UPF + UPF → dst) through shortest-delay path
    """
    try:
        delay_src_to_upf = nx.shortest_path_length(G, source=src, target=upf, weight='delay')
        delay_upf_to_dst = nx.shortest_path_length(G, source=upf, target=dst, weight='delay')
        return delay_src_to_upf + delay_upf_to_dst
    except nx.NetworkXNoPath:
        return 1e6


def prepare_optimization_data(G, session_requests):
    """
    Prepare all data structures needed for optimization
    
    Returns:
        dict: Dictionary containing all optimization parameters
    """
    # UPF devices
    upf_devices = [n for n in G.nodes if G.nodes[n]['type'] in ['DC', 'BigRouter', 'SmallRouter']]
    
    # Session data
    sessions_ids = [sess[3] for sess in session_requests]
    session_throughput = {sess[3]: sess[2] for sess in session_requests}
    session_src = {sess[3]: sess[0] for sess in session_requests}
    session_dst = {sess[3]: sess[1] for sess in session_requests}
    session_latency_weight = {sess[3]: sess[4] for sess in session_requests}
    
    L = {}
    min_delay = {}
    edge_delay_data = {}  # For backward compatibility
    
    for s in sessions_ids:
        L[s] = {}
        src = session_src[s]
        dst = session_dst[s]
        
        latencies = []
        for u in upf_devices:
            # Use compute_session_latency for consistency with heuristics
            latency = compute_session_latency(G, src, dst, u)
            L[s][u] = latency
            latencies.append(latency)
            
            # Also compute propagation-only delay for edge_delay_data
            # (used in objective function, but UPF latency added separately)
            edge_delay_data[(s, u)] = compute_network_delay(G, src, u, dst)
        
        # Minimum possible latency for this session
        min_delay[s] = min(latencies)
    
    # Power coefficients and capacities
    w_u = {}
    p_u = {}
    T_u = {}
    l_u = {}
    
    for u in upf_devices:
        dev_type = G.nodes[u]['type']
        w_u[u] = POWER_COEFFICIENTS[dev_type]['w']
        p_u[u] = POWER_COEFFICIENTS[dev_type]['p']
        T_u[u] = CAPACITY_THRESHOLD * G.nodes[u]['capacity']
        
        # Constant UPF latency
        load = G.nodes[u].get('load', 0.3)
        l_u[u] = UPF_LATENCIES[dev_type]
    
    # Normalization factor for delay
    delay_diff_max = max(
        (edge_delay_data[(s, u)] + l_u[u] - min_delay[s])
        for s in sessions_ids for u in upf_devices if np.isfinite(edge_delay_data[(s, u)])
    )
    delay_diff_max = max(delay_diff_max, 1e-6)
    
    return {
        'upf_devices': upf_devices,
        'sessions_ids': sessions_ids,
        'session_throughput': session_throughput,
        'session_src': session_src,
        'session_dst': session_dst,
        'session_latency_weight': session_latency_weight,
        'edge_delay_data': edge_delay_data,
        'L': L,  # ADDED: Include latency matrix for verification
        'min_delay': min_delay,
        'w_u': w_u,
        'p_u': p_u,
        'T_u': T_u,
        'l_u': l_u,
        'delay_diff_max': delay_diff_max
    }


def solve_upf_optimization(opt_data, alpha, gamma, verbose=False, timeout=300):
    """
    Solve UPF assignment optimization problem
    
    Args:
        opt_data: Dictionary with optimization parameters
        alpha: Weight for delay term
        gamma: Weight for power term
        verbose: Print solver output
        timeout: Maximum solver time in seconds (default 300s = 5min)
        
    Returns:
        dict: Results including objective value, assignments, and metrics
    """
    import time
    
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
    
    print(f"  [Setup] Building optimization problem...")
    print(f"  [Setup] Sessions: {len(S)}, UPFs: {len(U)}")
    
    # Build numpy vectors/matrices (pure numpy)
    T_s_vec = np.array([T_s[s] for s in S])               # shape (|S|,)
    beta_vec = np.array([beta_s[s] for s in S])           # shape (|S|,)
    T_u_vec = np.array([T_u[u] for u in U])               # shape (|U|,)
    w_vec = np.array([w_u[u] for u in U])                 # shape (|U|,)
    p_vec = np.array([p_u[u] for u in U])                 # shape (|U|,)
    l_vec = np.array([l_u[u] for u in U])                 # shape (|U|,)
    delay_matrix = np.array([[d_fix[(s, u)] for u in U] for s in S])  # shape (|S|,|U|)
    min_delay_vec = np.array([d_s_min[s] for s in S])                 # shape (|S|,)
    
    # Convert numpy data to CVXPY constants (explicit)
    T_s_const = cp.Constant(T_s_vec)            # (|S|,)
    beta_const = cp.Constant(beta_vec)          # (|S|,)
    T_u_const = cp.Constant(T_u_vec)            # (|U|,)
    w_const = cp.Constant(w_vec)                # (|U|,)
    p_const = cp.Constant(p_vec)                # (|U|,)
    l_const = cp.Constant(l_vec)                # (|U|,)
    delay_const = cp.Constant(delay_matrix)     # (|S|,|U|)
    min_delay_const = cp.Constant(min_delay_vec)# (|S|,)
    delay_diff_max_const = float(delay_diff_max)  # scalar
    
    # Decision variables
    x = cp.Variable((len(S), len(U)), boolean=True)
    y = cp.Variable(len(U), boolean=True)
    
    print(f"  [Setup] Binary variables: {len(S) * len(U) + len(U)}")
    
    # Constraints
    constraints = []
    
    # Each session assigned to exactly one UPF
    constraints.append(cp.sum(x, axis=1) == 1)  # shape (|S|,)
    
    # Capacity constraint (vectorized) => throughput_per_upf is (|U|,)
    throughput_per_upf = cp.matmul(T_s_const, x)  # (|U|,)
    constraints.append(throughput_per_upf <= cp.multiply(T_u_const, y))
    
    print(f"  [Setup] Constraints: {len(constraints)}")
    print(f"  [Setup] Building objective function...")
    
    # Objective function
    # Delay terms: (delay_matrix + l_vec - min_delay_vec[:,None]) / delay_diff_max
    delay_terms_const = cp.Constant((delay_matrix + l_vec - min_delay_vec[:, None]) / delay_diff_max_const)  # (|S|,|U|)
    
    # delay_gain_expr = sum_su beta_s[s] * x[s,u] * delay_term[s,u]  / |S|
    delay_gain_expr = cp.sum(cp.multiply(cp.reshape(beta_const, (len(S), 1)), cp.multiply(x, delay_terms_const))) / len(S)
    
    # Power term:
    # sum_u w_u * y[u]  + sum_{s,u} p_u * x[s,u] * T_s[s]
    power_fixed_expr = cp.sum(cp.multiply(w_const, y))  # scalar
    p_times_T = cp.Constant((p_vec * T_s_vec[:, None]))  # shape (|S|,|U|)
    power_dynamic_expr = cp.sum(cp.multiply(x, p_times_T))  # scalar
    total_power_expr = power_fixed_expr + power_dynamic_expr
    power_norm = total_power_expr / float(np.sum(w_vec))
    
    objective = cp.Minimize(alpha * delay_gain_expr + gamma * power_norm)
    
    print(f"  [Solver] Creating problem instance...")
    prob = cp.Problem(objective, constraints)
    
    print(f"  [Solver] Starting MOSEK (timeout: {timeout}s)...")
    solve_start = time.time()
    
    # Solve with timeout
    try:
        prob.solve(
            solver=cp.MOSEK, 
            verbose=verbose,
            mosek_params={
                'MSK_DPAR_OPTIMIZER_MAX_TIME': float(timeout),
                'MSK_IPAR_LOG': 10 if verbose else 0
            }
        )
        solve_time = time.time() - solve_start
        print(f"  [Solver] Completed in {solve_time:.2f}s - Status: {prob.status}")
        
    except Exception as e:
        solve_time = time.time() - solve_start
        print(f"  [Solver] Exception after {solve_time:.2f}s: {str(e)}")
        raise
    
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print(f"  [Warning] Non-optimal status: {prob.status}")
        if prob.status == "user_limit":
            print(f"  [Warning] Solver reached time limit ({timeout}s)")
        
        # Return partial/infeasible result
        return {
            'status': prob.status,
            'objective_value': np.nan,
            'delay_term': np.nan,
            'power_term': np.nan,
            'x_assignments': None,
            'y_active': None,
            'session_latencies': [],
            'avg_latency': np.nan,
            'avg_min_latency': np.nan,
            'latency_gain_pct': np.nan,
            'active_upfs': 0,
            'solve_time': solve_time
        }
    
    print(f"  [Results] Extracting solution...")
    # Extract results (convert to numpy arrays)
    x_val = np.array(x.value)
    y_val = np.array(y.value)
    
    # Evaluate objective components (numpy arithmetic)
    delay_term_value = np.sum(beta_vec[:, None] * x_val * (delay_matrix + l_vec - min_delay_vec[:, None]) / delay_diff_max) / len(S)
    power_term_value = (np.sum(w_vec * y_val) + np.sum(p_vec * (x_val * T_s_vec[:, None]))) / np.sum(w_vec)
    
    # Compute session latencies
    assigned_upfs = np.argmax(x_val, axis=1)
    session_latencies = (delay_matrix[np.arange(len(S)), assigned_upfs] + l_vec[assigned_upfs]).tolist()
    
    avg_latency = float(np.mean(session_latencies))
    avg_min_latency = float(np.mean(min_delay_vec))
    latency_gain_pct = 100 * (avg_latency / avg_min_latency - 1)
    
    # Count active UPFs
    active_upfs = int(np.sum(y_val >= 0.5))
    
    print(f"  [Results] Solution extracted - {active_upfs} UPFs active")
    
    return {
        'status': prob.status,
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
        'solve_time': solve_time
    }