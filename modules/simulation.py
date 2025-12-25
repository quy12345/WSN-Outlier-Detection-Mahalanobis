"""
Simulation Engine for Full Network with Time-Step Processing.
Matches Paper's Figure 2 - processes one time step at a time.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from .network import SensorNode, ClusterHead, OD_Detector
from .config import N_CLUSTERS, N_ATTRIBUTES, SENSOR_ATTRIBUTES


def run_cluster_simulation(cluster_id: int, cluster_data: pd.DataFrame, 
                           ground_truth: np.ndarray, algorithm: str = 'ODA-MD',
                           window_size: int = 50) -> dict:
    """
    Run simulation for a single cluster with TIME-STEP processing.
    
    Per Figure 2:
    - At each time t: CH sends request, nodes send A_k, CH decides
    
    Parameters:
    -----------
    cluster_id : int
        Cluster identifier
    cluster_data : pd.DataFrame
        DataFrame with 4 attributes per node
    ground_truth : np.ndarray
        Array of 0/1 labels per epoch
    algorithm : str
        'ODA-MD' or 'OD'
    window_size : int
        Window size for rolling statistics (ODA-MD)
    
    Returns:
    --------
    dict with TP, FP, TN, FN, DA, FAR, energy, histories
    """
    n_samples = len(cluster_data)
    
    # Extract unique node IDs
    node_ids = set()
    for col in cluster_data.columns:
        if col.startswith('node_'):
            parts = col.split('_')
            if len(parts) >= 2:
                try:
                    node_ids.add(int(parts[1]))
                except ValueError:
                    continue
    
    node_ids = sorted(list(node_ids))
    
    if not node_ids:
        return {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0, 'DA': 0, 'FAR': 0, 
                'energy': 0, 'da_history': [], 'far_history': []}
    
    # Build data matrices for each node: shape (n_samples, 4)
    node_data_matrices = {}
    for node_id in node_ids:
        data_matrix = []
        for attr in SENSOR_ATTRIBUTES:
            col_name = f'node_{node_id}_{attr}'
            if col_name in cluster_data.columns:
                data_matrix.append(cluster_data[col_name].values)
            else:
                data_matrix.append(np.zeros(n_samples))
        node_data_matrices[node_id] = np.column_stack(data_matrix)
    
    # Create detector
    if algorithm == 'ODA-MD':
        members = []
        for node_id in node_ids:
            member = SensorNode(node_id, node_data_matrices[node_id])
            members.append(member)
        detector = ClusterHead(cluster_id=cluster_id, node_ids=node_ids,
                               member_nodes=members, window_size=window_size)
    else:
        detector = OD_Detector(n_nodes=len(node_ids))
    
    decisions = {}
    da_history = []
    far_history = []
    time_per_epoch = 2000 / n_samples
    
    # Process each time step (Figure 2 approach)
    for epoch_idx in range(n_samples):
        if algorithm == 'ODA-MD':
            # ODA-MD: process_time_step
            is_outlier, _ = detector.process_time_step(epoch_idx)
            decisions[epoch_idx] = is_outlier
        else:
            # OD: Build observation matrix and detect
            obs_matrix = np.array([node_data_matrices[nid][epoch_idx] for nid in node_ids])
            is_outlier, _ = detector.detect(obs_matrix)
            decisions[epoch_idx] = is_outlier
        
        # Record metrics every 100 steps or at end
        if (epoch_idx + 1) % 100 == 0 or epoch_idx == n_samples - 1:
            t = int((epoch_idx + 1) * time_per_epoch)
            gt_subset = ground_truth[:epoch_idx + 1]
            n_outliers = np.sum(gt_subset == 1)
            n_normal = np.sum(gt_subset == 0)
            
            cur_tp = sum(1 for k in range(epoch_idx + 1) if decisions[k] and ground_truth[k] == 1)
            cur_fp = sum(1 for k in range(epoch_idx + 1) if decisions[k] and ground_truth[k] == 0)
            
            da = cur_tp / n_outliers if n_outliers > 0 else 1.0
            far = cur_fp / n_normal if n_normal > 0 else 0.0
            
            da_history.append((t, da * 100))
            far_history.append((t, far * 100))
    
    # Calculate energy
    if algorithm == 'ODA-MD':
        total_energy = detector.consumed_energy
        for m in detector.members:
            total_energy += m.consumed_energy
    else:
        total_energy = detector.consumed_energy
    
    # Final metrics
    total_tp = sum(1 for k, v in decisions.items() if v and ground_truth[k] == 1)
    total_fp = sum(1 for k, v in decisions.items() if v and ground_truth[k] == 0)
    total_tn = sum(1 for k, v in decisions.items() if not v and ground_truth[k] == 0)
    total_fn = sum(1 for k, v in decisions.items() if not v and ground_truth[k] == 1)
    
    da_final = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    far_final = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0
    
    return {
        'TP': total_tp, 'FP': total_fp, 'TN': total_tn, 'FN': total_fn,
        'DA': da_final, 'FAR': far_final,
        'energy': total_energy,
        'da_history': da_history,
        'far_history': far_history
    }


def run_full_network_simulation(topology, data_dict: Dict, 
                                 ground_truth_dict: Dict,
                                 algorithm: str = 'ODA-MD') -> dict:
    """
    Run simulation across all clusters with TIME-STEP processing.
    """
    print("\n" + "=" * 60)
    print(f"RUNNING FULL NETWORK SIMULATION: {algorithm}")
    print(f"Clusters: {topology.n_clusters}, Nodes: {topology.n_nodes}")
    print(f"Processing: TIME-STEP (per Figure 2)")
    print("=" * 60)
    
    total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0
    total_energy = 0.0
    all_da_histories = []
    all_far_histories = []
    
    cluster_results = {}
    
    for cluster_id in sorted(data_dict.keys()):
        cluster_data = data_dict[cluster_id]
        ground_truth = ground_truth_dict[cluster_id]
        
        print(f"\n--- Cluster {cluster_id} ---")
        
        result = run_cluster_simulation(
            cluster_id=cluster_id,
            cluster_data=cluster_data,
            ground_truth=ground_truth,
            algorithm=algorithm
        )
        
        cluster_results[cluster_id] = result
        
        total_tp += result['TP']
        total_fp += result['FP']
        total_tn += result['TN']
        total_fn += result['FN']
        total_energy += result['energy']
        
        if result['da_history']:
            all_da_histories.append(result['da_history'])
        if result['far_history']:
            all_far_histories.append(result['far_history'])
        
        print(f"DA: {result['DA']*100:.2f}%, FAR: {result['FAR']*100:.2f}%, "
              f"Energy: {result['energy']:.6f}J")
    
    da_network = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    far_network = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0
    
    if all_da_histories:
        da_history = all_da_histories[1] if len(all_da_histories) > 1 else all_da_histories[0]
        far_history = all_far_histories[1] if len(all_far_histories) > 1 else all_far_histories[0]
    else:
        da_history = []
        far_history = []
    
    print("\n" + "=" * 60)
    print(f"NETWORK-WIDE RESULTS ({algorithm})")
    print("=" * 60)
    print(f"Total True Positives: {total_tp}")
    print(f"Total False Positives: {total_fp}")
    print(f"Total True Negatives: {total_tn}")
    print(f"Total False Negatives: {total_fn}")
    print(f"Network Detection Accuracy: {da_network*100:.2f}%")
    print(f"Network False Alarm Rate: {far_network*100:.2f}%")
    print(f"Total Network Energy: {total_energy:.6f} Joules")
    
    return {
        'TP': total_tp, 'FP': total_fp, 'TN': total_tn, 'FN': total_fn,
        'DA': da_network, 'FAR': far_network,
        'energy': total_energy,
        'da_history': da_history,
        'far_history': far_history,
        'cluster_results': cluster_results
    }


# Legacy function
def run_simulation(df_data: pd.DataFrame, ground_truth: np.ndarray,
                   outlier_schedule: list, algorithm: str = 'ODA-MD',
                   n_total_outliers: int = 1000) -> dict:
    """Legacy single-cluster simulation."""
    print("\n" + "=" * 60)
    print(f"RUNNING SIMULATION (Legacy): {algorithm}")
    print("=" * 60)
    
    return run_cluster_simulation(
        cluster_id=2,
        cluster_data=df_data,
        ground_truth=ground_truth,
        algorithm=algorithm
    )
