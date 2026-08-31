#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Heuristic methods for UPF assignment
Implements three baseline strategies:
1. Closest to Source (with capacity overflow)
2. Central UPF (DC-based)
3. Closest to Destination (with capacity overflow)
"""

import numpy as np
import networkx as nx
from config import POWER_COEFFICIENTS, CAPACITY_THRESHOLD
from network_utils import compute_session_latency, compute_min_possible_latency


def compute_path_latency(G, src, dst, upf):
    """
    Compute end-to-end latency: src -> UPF -> dst
    Includes propagation delay and UPF processing latency
    """
    try:
        # Path from source to UPF
        path_to_upf = nx.shortest_path(G, src, upf, weight='delay')
        delay_to_upf = sum(G[path_to_upf[i]][path_to_upf[i+1]]['delay'] 
                          for i in range(len(path_to_upf)-1))
        
        # UPF processing latency
        upf_processing = G.nodes[upf]['upf_latency']
        
        # Path from UPF to destination
        path_from_upf = nx.shortest_path(G, upf, dst, weight='delay')
        delay_from_upf = sum(G[path_from_upf[i]][path_from_upf[i+1]]['delay'] 
                            for i in range(len(path_from_upf)-1))
        
        total_latency = delay_to_upf + upf_processing + delay_from_upf
        return total_latency
    except nx.NetworkXNoPath:
        return float('inf')


def get_sorted_upfs_by_distance(G, reference_node, upf_devices):
    """
    Sort UPF devices by distance from a reference node
    
    Args:
        G: Network graph
        reference_node: Source or destination node
        upf_devices: List of UPF device names
        
    Returns:
        List of UPF devices sorted by distance (closest first)
    """
    distances = []
    ref_pos = np.array(G.nodes[reference_node]['pos'])
    
    for upf in upf_devices:
        upf_pos = np.array(G.nodes[upf]['pos'])
        dist = np.linalg.norm(ref_pos - upf_pos)
        distances.append((upf, dist))
    
    # Sort by distance
    distances.sort(key=lambda x: x[1])
    return [upf for upf, _ in distances]

def get_sorted_upfs_by_network_proximity(G, reference_node, upf_devices):
    """
    Sort UPF devices by network proximity (path length) from a reference node
    
    Args:
        G: Network graph
        reference_node: Source or destination node
        upf_devices: List of UPF device names
        
    Returns:
        List of UPF devices sorted by network path length (closest first)
    """
    distances = []
    
    for upf in upf_devices:
        try:
            # Use shortest path length (number of hops or delay-weighted)
            path_length = nx.shortest_path_length(G, reference_node, upf, weight='delay')
            distances.append((upf, path_length))
        except nx.NetworkXNoPath:
            # If no path exists, assign infinite distance
            distances.append((upf, float('inf')))
    
    # Sort by network distance
    distances.sort(key=lambda x: x[1])
    return [upf for upf, _ in distances]

def get_central_upfs(G, upf_devices):
    """
    Get UPF devices sorted by centrality (DCs first, then by distance to center)
    
    Returns:
        List of UPF devices sorted by centrality
    """
    from config import SURFACE_WIDTH
    center = np.array([SURFACE_WIDTH / 2, SURFACE_WIDTH / 2])
    
    # Separate by type
    dcs = [upf for upf in upf_devices if G.nodes[upf]['type'] == 'DC']
    big_routers = [upf for upf in upf_devices if G.nodes[upf]['type'] == 'BigRouter']
    small_routers = [upf for upf in upf_devices if G.nodes[upf]['type'] == 'SmallRouter']
    
    # Sort each type by distance to center
    def dist_to_center(upf):
        pos = np.array(G.nodes[upf]['pos'])
        return np.linalg.norm(pos - center)
    
    dcs.sort(key=dist_to_center)
    big_routers.sort(key=dist_to_center)
    small_routers.sort(key=dist_to_center)
    
    # Prioritize DCs (most central), then BigRouters, then SmallRouters
    return dcs + big_routers + small_routers


def assign_sessions_with_capacity(G, session_requests, upf_priority_list):
    """
    Assign sessions to UPFs based on priority list, respecting capacity constraints
    
    Args:
        G: Network graph
        session_requests: List of (src, dst, throughput, session_id, latency_weight)
        upf_priority_list: Function that returns sorted UPF list given (G, src, dst, upf_devices)
        
    Returns:
        dict: Assignment results with metrics
    """
    upf_devices = [n for n in G.nodes 
                   if G.nodes[n]['type'] in ['DC', 'BigRouter', 'SmallRouter']]
    
    # Initialize UPF loads
    upf_loads = {upf: 0.0 for upf in upf_devices}
    upf_capacities = {upf: G.nodes[upf]['capacity'] * CAPACITY_THRESHOLD 
                      for upf in upf_devices}
    
    # Track assignments
    assignments = {}  # session_id -> upf
    session_latencies = {}
    failed_sessions = []
    
    # Process each session
    for src, dst, throughput, session_id, latency_weight in session_requests:
        # Get priority list for this session
        priority_upfs = upf_priority_list(G, src, dst, upf_devices)
        
        assigned = False
        for upf in priority_upfs:
            # Check if UPF has capacity
            if upf_loads[upf] + throughput <= upf_capacities[upf]:
                # Assign session to this UPF
                assignments[session_id] = upf
                upf_loads[upf] += throughput
                
                # Calculate latency
                latency = compute_path_latency(G, src, dst, upf)
                session_latencies[session_id] = latency
                
                assigned = True
                break
        
        if not assigned:
            failed_sessions.append(session_id)
            # Assign to closest UPF anyway (overflow)
            upf = priority_upfs[0]
            assignments[session_id] = upf
            upf_loads[upf] += throughput
            latency = compute_path_latency(G, src, dst, upf)
            session_latencies[session_id] = latency
    
    # Calculate active UPFs
    active_upfs = [upf for upf in upf_devices if upf_loads[upf] > 0]
    
    # Calculate power consumption
    total_power = 0.0
    for upf in active_upfs:
        device_type = G.nodes[upf]['type']
        w = POWER_COEFFICIENTS[device_type]['w']
        p = POWER_COEFFICIENTS[device_type]['p']
        load_normalized = upf_loads[upf] / G.nodes[upf]['capacity']
        power = w + p * upf_loads[upf]
        total_power += power
    
    # Calculate metrics
    latencies = list(session_latencies.values())
    avg_latency = np.mean(latencies) if latencies else 0
    
    # Calculate minimum possible latencies (direct path, closest UPF)
    min_latencies = []
    for src, dst, throughput, session_id, latency_weight in session_requests:
        min_lat = float('inf')
        for upf in upf_devices:
            lat = compute_path_latency(G, src, dst, upf)
            if lat < min_lat:
                min_lat = lat
        min_latencies.append(min_lat)
    
    avg_min_latency = np.mean(min_latencies) if min_latencies else 0
    latency_gain_pct = ((avg_latency - avg_min_latency) / avg_min_latency * 100) if avg_min_latency > 0 else 0
    
    results = {
        'assignments': assignments,
        'session_latencies': session_latencies,
        'upf_loads': upf_loads,
        'active_upfs': active_upfs,
        'total_power': total_power,
        'avg_latency': avg_latency,
        'avg_min_latency': avg_min_latency,
        'latency_gain_pct': latency_gain_pct,
        'num_active_upfs': len(active_upfs),
        'failed_sessions': failed_sessions,
        'num_failed': len(failed_sessions)
    }
    
    return results


def closest_to_source_method_geo(G, session_requests):
    """
    Heuristic 1: Assign sessions to UPF closest to source
    """
    def priority_func(G, src, dst, upf_devices):
        return get_sorted_upfs_by_distance(G, src, upf_devices)
    
    return assign_sessions_with_capacity(G, session_requests, priority_func)


def central_upf_method(G, session_requests):
    """
    Heuristic 2: Assign sessions to central UPFs (DCs first, then closest to center)
    """
    def priority_func(G, src, dst, upf_devices):
        return get_central_upfs(G, upf_devices)
    
    return assign_sessions_with_capacity(G, session_requests, priority_func)


def closest_to_destination_method_geo(G, session_requests):
    """
    Heuristic 3: Assign sessions to UPF closest to destination
    """
    def priority_func(G, src, dst, upf_devices):
        return get_sorted_upfs_by_distance(G, dst, upf_devices)
    
    return assign_sessions_with_capacity(G, session_requests, priority_func)

def closest_to_source_method(G, session_requests):
    """
    Heuristic 1: Assign sessions to UPF closest to source (network proximity)
    """
    def priority_func(G, src, dst, upf_devices):
        return get_sorted_upfs_by_network_proximity(G, src, upf_devices)
    
    return assign_sessions_with_capacity(G, session_requests, priority_func)


def closest_to_destination_method(G, session_requests):
    """
    Heuristic 3: Assign sessions to UPF closest to destination (network proximity)
    """
    def priority_func(G, src, dst, upf_devices):
        return get_sorted_upfs_by_network_proximity(G, dst, upf_devices)
    
    return assign_sessions_with_capacity(G, session_requests, priority_func)

def run_all_heuristics(G, session_requests):
    """
    Run all three heuristic methods and return results
    
    Returns:
        dict: Results for each method
    """
    print("\n" + "="*70)
    print("RUNNING HEURISTIC METHODS")
    print("="*70 + "\n")
    
    results = {}
    
    # Method 1: Closest to Source
    print("Running Method 1: Closest to Source...")
    results['closest_to_source'] = closest_to_source_method(G, session_requests)
    print(f"  Active UPFs: {results['closest_to_source']['num_active_upfs']}")
    print(f"  Avg Latency: {results['closest_to_source']['avg_latency']:.4f} ms")
    print(f"  Total Power: {results['closest_to_source']['total_power']:.2f} W")
    print(f"  Failed Sessions: {results['closest_to_source']['num_failed']}")
    
    # Method 2: Central UPF
    print("\nRunning Method 2: Central UPF...")
    results['central_upf'] = central_upf_method(G, session_requests)
    print(f"  Active UPFs: {results['central_upf']['num_active_upfs']}")
    print(f"  Avg Latency: {results['central_upf']['avg_latency']:.4f} ms")
    print(f"  Total Power: {results['central_upf']['total_power']:.2f} W")
    print(f"  Failed Sessions: {results['central_upf']['num_failed']}")
    
    # Method 3: Closest to Destination
    print("\nRunning Method 3: Closest to Destination...")
    results['closest_to_destination'] = closest_to_destination_method(G, session_requests)
    print(f"  Active UPFs: {results['closest_to_destination']['num_active_upfs']}")
    print(f"  Avg Latency: {results['closest_to_destination']['avg_latency']:.4f} ms")
    print(f"  Total Power: {results['closest_to_destination']['total_power']:.2f} W")
    print(f"  Failed Sessions: {results['closest_to_destination']['num_failed']}")
    
    print("\n" + "="*70 + "\n")
    
    return results