#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Session generation and traffic utilities
"""

import random
import numpy as np
from config import *


def generate_structured_session_requests(G, subnets, subnet_to_bigrouter, num_sessions, session_id_start=0):
    """
    Generate structured session requests based on traffic distribution
    
    Args:
        G: Network graph
        subnets: List of subnets
        subnet_to_bigrouter: Mapping of subnets to big routers
        num_sessions: Number of sessions to generate
        session_id_start: Starting session ID
        
    Returns:
        list: List of session tuples (src, dst, throughput, session_id, latency_weight)
    """
    sessions = []
    all_gnbs = [n for n in G.nodes if G.nodes[n]['type'] == 'gNodeB']
    session_id = session_id_start

    thresholds = [
        TRAFFIC_DISTRIBUTION["TRAFFIC_TO_DC"],
        TRAFFIC_DISTRIBUTION["TRAFFIC_TO_DC"] + TRAFFIC_DISTRIBUTION["TRAFFIC_SAME_SUBNET"],
        TRAFFIC_DISTRIBUTION["TRAFFIC_TO_DC"] + TRAFFIC_DISTRIBUTION["TRAFFIC_SAME_SUBNET"] + 
        TRAFFIC_DISTRIBUTION["TRAFFIC_SAME_BIG_ROUTER"]
    ]

    for _ in range(num_sessions):
        rand = random.random()
        min_throughput = random.uniform(THROUGHPUT_RANGE[0], THROUGHPUT_RANGE[1])
        src = random.choice(all_gnbs)
        latency_weight = LATENCY_OPTIONS[random.randint(0, 3)]

        if rand < thresholds[0]:  # Traffic to DC
            src_pos = G.nodes[src]['pos']
            dcs = [n for n in G.nodes if G.nodes[n]['type'] == 'DC']
            dst = min(dcs, key=lambda dc: np.linalg.norm(np.array(G.nodes[dc]['pos']) - np.array(src_pos)))

        elif rand < thresholds[1]:  # Traffic within same subnet
            acceptable_subnets = [s for s in subnets if len(s) >= 2]
            if not acceptable_subnets:
                continue
            subnet = random.choice(acceptable_subnets)
            src, dst = random.sample(subnet, 2)

        elif rand < thresholds[2]:  # Traffic to different subnet under same big router
            big_router = random.choice(list(subnet_to_bigrouter.values()))
            valid_subnets = [s for s, r in subnet_to_bigrouter.items() if r == big_router and len(s) >= 1]
            if len(valid_subnets) >= 2:
                subnet_a, subnet_b = random.sample(valid_subnets, 2)
                src = random.choice(subnet_a)
                dst = random.choice(subnet_b)
            else:
                continue

        else:  # Traffic to closest BigRouter
            src_pos = G.nodes[src]['pos']
            big_routers = [n for n in G.nodes if G.nodes[n]['type'] == 'BigRouter']
            dst = min(big_routers, key=lambda br: np.linalg.norm(np.array(G.nodes[br]['pos']) - np.array(src_pos)))

        sessions.append((src, dst, min_throughput, session_id, latency_weight))
        session_id += 1

    total_throughput = sum(req[2] for req in sessions)
    print(f"Generated {len(sessions)} sessions - Total minimum throughput: {total_throughput:.2f} Mbps")

    return sessions