#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Network creation and utility functions
"""

import networkx as nx # type: ignore
import numpy as np
import random
from config import *


def random_position():
    """Generate random position within network area"""
    return np.random.uniform(0, SURFACE_WIDTH, 2)


def euclidean_dist_km(a, b):
    """Calculate Euclidean distance between two positions"""
    return np.linalg.norm(np.array(a) - np.array(b))


def compute_delay(pos1, pos2):
    """Compute propagation delay based on distance (1 ms per 100 km)"""
    distance_km = euclidean_dist_km(pos1, pos2)
    return max(0.001, distance_km / 100)


def create_network():
    """
    Create the complete network topology with DCs, routers, and gNodeBs
    
    Returns:
        nx.Graph: Network graph with all nodes and edges
    """
    G = nx.Graph()
    device_positions = {}

    # Add datacenters
    for i in range(DEVICE_COUNTS["DC"]):
        name = f'DC_{i}'
        pos = DC_POSITIONS[i]
        G.add_node(
            name,
            type='DC',
            pos=pos,
            load=0.01,
            capacity=DEVICE_CAPACITIES["DC"],
            upf_latency=UPF_LATENCIES["DC"]
        )
        device_positions[name] = pos

    # Add big routers
    for i in range(DEVICE_COUNTS["BigRouter"]):
        name = f'BigRouter_{i}'
        pos = random_position()
        G.add_node(
            name,
            type='BigRouter',
            pos=pos,
            load=0.01,
            capacity=DEVICE_CAPACITIES["BigRouter"],
            upf_latency=UPF_LATENCIES["BigRouter"]
        )
        device_positions[name] = pos

    # Add small routers
    small_router_nodes = []
    for i in range(DEVICE_COUNTS["SmallRouter"]):
        name = f'SmallRouter_{i}'
        pos = random_position()
        G.add_node(
            name,
            type='SmallRouter',
            pos=pos,
            load=0.01,
            capacity=DEVICE_CAPACITIES["SmallRouter"],
            upf_latency=UPF_LATENCIES["SmallRouter"]
        )
        device_positions[name] = pos
        small_router_nodes.append(name)

    # Add gNodeBs clustered around each small router
    gnb_id = 0
    radius = 20
    for sr in small_router_nodes:
        cluster_size = min(random.randint(2, 8), DEVICE_COUNTS["gNodeB"] - gnb_id)
        sr_pos = np.array(device_positions[sr])

        for _ in range(cluster_size):
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, radius)
            offset = np.array([r * np.cos(angle), r * np.sin(angle)])
            gnb_pos = sr_pos + offset

            name = f'gNB_{gnb_id}'
            G.add_node(name, type='gNodeB', pos=gnb_pos, load=0, capacity=1000)
            device_positions[name] = gnb_pos

            G.add_edge(name, sr, delay=compute_delay(gnb_pos, sr_pos))

            gnb_id += 1
            if gnb_id >= DEVICE_COUNTS["gNodeB"]:
                break
        if gnb_id >= DEVICE_COUNTS["gNodeB"]:
            break

    # Connect small routers to nearest big routers
    for sr in small_router_nodes:
        pos = device_positions[sr]
        nearest_big = sorted(
            [n for n in G.nodes if G.nodes[n]['type'] == 'BigRouter'],
            key=lambda br: euclidean_dist_km(pos, device_positions[br])
        )[:2]
        for br in nearest_big:
            G.add_edge(sr, br, delay=compute_delay(pos, device_positions[br]))

    # Connect big routers to nearest datacenters
    for br in [n for n in G.nodes if G.nodes[n]['type'] == 'BigRouter']:
        pos = device_positions[br]
        nearest_dc = sorted(
            [n for n in G.nodes if G.nodes[n]['type'] == 'DC'],
            key=lambda dc: euclidean_dist_km(pos, device_positions[dc])
        )[:2]
        for dc in nearest_dc:
            G.add_edge(br, dc, delay=compute_delay(pos, device_positions[dc]))

    # Connect datacenters fully
    dc_nodes = [n for n in G.nodes if G.nodes[n]['type'] == 'DC']
    for i in range(len(dc_nodes)):
        for j in range(i + 1, len(dc_nodes)):
            dc1, dc2 = dc_nodes[i], dc_nodes[j]
            pos1, pos2 = device_positions[dc1], device_positions[dc2]
            delay = compute_delay(pos1, pos2)
            G.add_edge(dc1, dc2, delay=delay)

    # Compute total network capacity
    upf_nodes = [n for n in G.nodes if G.nodes[n]['type'] in ['DC', 'BigRouter', 'SmallRouter']]
    total_capacity = sum(G.nodes[n]['capacity'] for n in upf_nodes)
    print(f"Network created with {len(G.nodes)} nodes and {len(G.edges)} edges.")
    print(f"Total network UPF capacity: {total_capacity} Mbps")

    return G


def build_subnets_and_mappings(G):
    """
    Build subnet structure and map subnets to big routers
    
    Returns:
        tuple: (subnets list, subnet_to_bigrouter dict)
    """
    subnets = []
    subnet_to_bigrouter = {}

    for sr in [n for n in G.nodes if G.nodes[n]['type'] == 'SmallRouter']:
        gnb_neighbors = [n for n in G.neighbors(sr) if G.nodes[n]['type'] == 'gNodeB']
        if len(gnb_neighbors) >= 2:
            subnets.append(gnb_neighbors)

            sr_pos = G.nodes[sr]['pos']
            big_router_neighbors = [n for n in G.neighbors(sr) if G.nodes[n]['type'] == 'BigRouter']
            
            if big_router_neighbors:
                closest_br = min(
                    big_router_neighbors,
                    key=lambda br: np.linalg.norm(np.array(G.nodes[br]['pos']) - np.array(sr_pos))
                )
                subnet_to_bigrouter[tuple(gnb_neighbors)] = closest_br

    return subnets, subnet_to_bigrouter


def compute_session_latency(G, src, dst, upf):
    """
    Compute end-to-end latency: src → UPF → dst
    Uses shortest path routing
    
    Args:
        G: Network graph
        src: Source node
        dst: Destination node  
        upf: UPF node
        
    Returns:
        float: Total latency in milliseconds
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


def compute_min_possible_latency(G, src, dst, upf_devices):
    """
    Compute minimum possible latency for a session
    (best UPF choice, ignoring capacity)
    """
    min_latency = float('inf')
    
    for upf in upf_devices:
        latency = compute_session_latency(G, src, dst, upf)
        if latency < min_latency:
            min_latency = latency
    
    return min_latency