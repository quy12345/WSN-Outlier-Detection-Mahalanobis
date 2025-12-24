"""
Simulation Engine for Full Network.
Supports both single-cluster and full-network (81 nodes, 10 clusters) simulations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from .network import SensorNode, ClusterHead, OD_Detector
from .config import N_CLUSTERS


def run_cluster_simulation(cluster_id: int, cluster_data: pd.DataFrame, 
                           ground_truth: np.ndarray, algorithm: str = 'ODA-MD',
                           batch_size: int = 50) -> dict:
    """
    Run simulation for a single cluster.
    
    Returns dict with TP, FP, TN, FN, DA, FAR, energy, histories.
    """
    # Get node columns (exclude 'epoch')
    node_columns = [col for col in cluster_data.columns if col.startswith('node_')]
    n_samples = len(cluster_data)
    
    if not node_columns:
        return {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0, 'DA': 0, 'FAR': 0, 
                'energy': 0, 'da_history': [], 'far_history': []}
    
    # Create detector
    if algorithm == 'ODA-MD':
        members = []
        for col in node_columns:
            node_data = cluster_data[col].values
            node_id = int(col.split('_')[1])
            members.append(SensorNode(node_id, node_data))
        detector = ClusterHead(cluster_id=cluster_id, batch_size=batch_size, member_nodes=members)
    else:
        detector = OD_Detector(node_columns)
    
    decisions = {}
    da_history = []
    far_history = []
    time_per_epoch = 2000 / n_samples
    
    if algorithm == 'ODA-MD':
        idx = 0
        while idx < n_samples:
            X, batch_indices = detector.collect_batch_data(idx, detector.batch_size)
            if len(batch_indices) == 0:
                break
            batch_results = detector.process_collected_batch(X, batch_indices)
            for res_idx, is_det in batch_results:
                decisions[res_idx] = is_det
            idx += len(batch_indices)
            
            # Record metrics
            t = int(idx * time_per_epoch)
            gt_subset = ground_truth[:idx]
            n_outliers = np.sum(gt_subset == 1)
            n_normal = np.sum(gt_subset == 0)
            cur_tp = sum(1 for k, v in decisions.items() if v and ground_truth[k] == 1)
            cur_fp = sum(1 for k, v in decisions.items() if v and ground_truth[k] == 0)
            da = cur_tp / n_outliers if n_outliers > 0 else 1.0
            far = cur_fp / n_normal if n_normal > 0 else 0.0
            da_history.append((t, da * 100))
            far_history.append((t, far * 100))
    else:
        for idx in range(n_samples):
            observation = cluster_data[node_columns].iloc[idx].values
            is_det, _ = detector.detect(observation)
            decisions[idx] = is_det
            
            if (idx + 1) % 50 == 0 or idx == n_samples - 1:
                t = int((idx + 1) * time_per_epoch)
                gt_subset = ground_truth[:idx + 1]
                n_outliers = np.sum(gt_subset == 1)
                n_normal = np.sum(gt_subset == 0)
                cur_tp = sum(1 for k in range(idx + 1) if decisions[k] and ground_truth[k] == 1)
                cur_fp = sum(1 for k in range(idx + 1) if decisions[k] and ground_truth[k] == 0)
                da = cur_tp / n_outliers if n_outliers > 0 else 0.0
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
    Run simulation across all clusters in the network.
    
    Parameters:
    -----------
    topology : NetworkTopology
        Network topology object
    data_dict : dict
        data_dict[cluster_id] = DataFrame
    ground_truth_dict : dict
        ground_truth_dict[cluster_id] = np.array
    algorithm : str
        'ODA-MD' or 'OD'
        
    Returns:
    --------
    dict with aggregated network-wide results
    """
    print("\n" + "=" * 60)
    print(f"RUNNING FULL NETWORK SIMULATION: {algorithm}")
    print(f"Clusters: {topology.n_clusters}, Nodes: {topology.n_nodes}")
    print("=" * 60)
    
    # Aggregate results
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
        
        # Aggregate
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
    
    # Compute network-wide metrics
    da_network = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    far_network = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0
    
    # Average histories across clusters
    # Find common time points and average DA/FAR
    if all_da_histories:
        # Simple approach: use Cluster 2's history as representative
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


# Legacy function for backward compatibility
def run_simulation(df_data: pd.DataFrame, ground_truth: np.ndarray,
                   outlier_schedule: list, algorithm: str = 'ODA-MD',
                   n_total_outliers: int = 1000) -> dict:
    """
    Legacy single-cluster simulation (Cluster 2 only).
    Kept for backward compatibility.
    """
    print("\n" + "=" * 60)
    print(f"RUNNING SIMULATION (Legacy): {algorithm}")
    print("=" * 60)
    
    return run_cluster_simulation(
        cluster_id=2,
        cluster_data=df_data,
        ground_truth=ground_truth,
        algorithm=algorithm
    )
