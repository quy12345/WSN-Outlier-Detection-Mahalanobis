"""
Network Components and Algorithms.
Contains SensorNode, ClusterHead (ODA-MD), and OD_Detector (Baseline).
"""

import numpy as np
from scipy.stats import chi2
from typing import List, Tuple, Dict
from .config import *

# =============================================================================
# SENSOR NODE
# =============================================================================

class SensorNode:
    """
    Sensor Node that performs sensing and transmission upon request.
    """
    def __init__(self, node_id: int, data_series: np.ndarray):
        self.node_id = node_id
        self.data = data_series  # Time series of temperature
        self.energy = INITIAL_ENERGY
        self.dist_to_ch = np.random.uniform(5, 30) # Random distance to CH (max 40m range)
        
        # Energy tracking
        self.consumed_energy = 0.0

    def consume_energy(self, amount: float):
        self.energy -= amount
        self.consumed_energy += amount

    def listen_for_request(self):
        """Receive request from CH."""
        # Energy to receive control packet (Request)
        e = E_ELEC * PACKET_SIZE * 8
        self.consume_energy(e)
        
    def sense_and_send_batch(self, start_idx: int, n_samples: int) -> np.ndarray:
        """
        Sense n_samples and send as a vector to CH (Figure 2a).
        """
        end_idx = min(start_idx + n_samples, len(self.data))
        if start_idx >= len(self.data):
            return np.array([])
            
        # Sensing
        data_vector = self.data[start_idx:end_idx]
        actual_n = len(data_vector)
        
        # Energy Calculation:
        # 1. Receive Request (Control Packet) - handled in listen_for_request
        
        # 2. Transmission Energy for the Vector
        # E_tx = E_elec * k + E_amp * k * d^2
        # k = n_samples * PACKET_SIZE (approx assuming packed)
        total_bits = actual_n * PACKET_SIZE * 8
        tx_energy = (E_ELEC + E_AMP * self.dist_to_ch**2) * total_bits
        self.consume_energy(tx_energy)
        
        return data_vector


# =============================================================================
# ODA-MD DETECTOR (Cluster Head)
# =============================================================================

class ClusterHead:
    """
    Cluster Head that runs ODA-MD algorithm (Algorithm 1 & 2).
    
    Collects data from member sensor nodes, computes Mahalanobis Distance,
    and detects outliers before forwarding normal data to sink.
    """
    
    def __init__(self, cluster_id: int = CH_ID, node_ids: List[int] = CLUSTER_2_NODES, 
                 batch_size: int = 50, member_nodes: List[SensorNode] = None):
        self.cluster_id = cluster_id
        self.node_ids = node_ids  # List of member node IDs in this cluster
        # If members not provided, will be assigned later
        self.members = member_nodes if member_nodes else [] 
        
        self.batch_size = batch_size
        
        # Buffer to store batch data
        self.buffer = []
        self.buffer_indices = []
        
        # Energy tracking
        self.energy = INITIAL_ENERGY
        self.consumed_energy = 0.0 # Track consumption J
        self.detected_count = 0

    def consume_energy(self, amount: float):
        self.energy -= amount
        self.consumed_energy += amount

    def collect_batch_data(self, start_idx: int, n_samples: int) -> Tuple[np.ndarray, List[int]]:
        """
        ALGORITHM 1 REVISED (Figure 2):
        1. CH sends ONE Request for entire batch.
        2. Members reply with vector Ak (n_samples).
        3. CH constructs Table_MDi.
        """
        # 1. Send Request (Broadcast) - ONE control packet per batch
        self.consume_energy((E_ELEC + E_AMP * RADIO_RANGE**2) * PACKET_SIZE * 8)
        
        # 2. Members receive and reply with vectors
        batch_data_per_node = []
        actual_n = 0
        
        for member in self.members:
            member.listen_for_request()
            # Request vector of size n_samples
            vector = member.sense_and_send_batch(start_idx, n_samples)
            
            if len(vector) > 0:
                batch_data_per_node.append(vector)
                actual_n = len(vector)
            
        # 3. Form Matrix X (Table_MDi)
        # Shape: (n_samples, n_nodes)
        if not batch_data_per_node:
            return np.array([]), []
            
        # Transpose to get (n_samples, n_nodes)
        # batch_data_per_node is list of [v1, v2..] (one vector per node)
        X = np.array(batch_data_per_node).T 
        
        # Receive Energy: CH receives P vectors of size N
        # Total bits received = P * N * Packet_Size * 8
        # (Assuming vector transmission is efficient block)
        total_bits_rx = len(self.members) * actual_n * PACKET_SIZE * 8
        self.consume_energy(E_ELEC * total_bits_rx)
        
        indices = list(range(start_idx, start_idx + actual_n))
        return X, indices

    def process_collected_batch(self, X: np.ndarray, indices: List[int]) -> List[Tuple[int, bool]]:
        """
        Execute Algorithm 2 on the collected batch matrix X.
        """
        if len(X) == 0: 
            return []
            
        # Reuse the logic but operate on X directly
        # Update buffer for temporary compatibility or just use local X
        self.buffer = X
        self.buffer_indices = indices
        
        return self.process_batch()

    def process_batch(self) -> List[Tuple[int, bool]]:
        """
        Execute Algorithm 2 (Strict implementation from paper).
        1. Calculate Mu, Sigma on ALL data in the batch.
        2. Calculate MD for each vector.
        3. Compare with Chi-square threshold.
        """
        X = np.array(self.buffer)
        indices = self.buffer_indices
        p = X.shape[1]
        
        # 1. Compute Mean and Covariance (Eq. 3 & 4) on ALL data
        mu = np.mean(X, axis=0)
        cov = np.cov(X, rowvar=False)
        
        # Handle singular matrix (computational stability, not in paper but needed)
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov += np.eye(p) * 1e-6
            cov_inv = np.linalg.inv(cov)

        # 2. Chi-square threshold (Eq. 7 implies threshold is sqrt(chi2))
        # "MD^2 follows chi-square... 97.5% quantile is threshold"
        # Eq 5: If MD_i >= sqrt(chi2) -> Outlier
        chi2_val = chi2.ppf(0.975, df=p)
        threshold = np.sqrt(chi2_val)

        results = []
        
        # 3. Compute MD and Detect (Eq. 2 & 5)
        for i in range(len(X)):
            diff = X[i] - mu
            md_sq = np.dot(np.dot(diff, cov_inv), diff)
            md = np.sqrt(max(0, md_sq))
            
            is_outlier = md >= threshold
            results.append((indices[i], is_outlier))
            
            if is_outlier:
                self.detected_count += 1
        
        # Energy consumption for the batch processing (Algorithm 2)
        # Note: Communication energy was already deducted in collect_data
        
        # Energy to compute/aggregate (CPU cost)
        # E_DA per bit per signal
        self.consume_energy(E_DA * PACKET_SIZE * 8 * len(X))
        
        # Energy to transmit normal data to sink
        # "Forward the vector Ak" if Normal
        n_normal = len(X) - self.detected_count
        dist_to_sink = 50.0 
        
        # Transmit aggregated/forwarded packets (Normal only)
        # We assume CH aggregates or forwards efficiently
        bits_to_sink = n_normal * PACKET_SIZE * 8
        if bits_to_sink > 0:
            self.consume_energy((E_ELEC + E_AMP * dist_to_sink**2) * bits_to_sink)
            
        # Clear buffer
        self.buffer = []
        self.buffer_indices = []
        
        return results

   
class OD_Detector:
    """
    Optimized OD algorithm using Fixed-width Clustering (FWC).
    Based on User description:
    Stage 1: Fixed-width Clustering (Online) - Group data into fixed-radius clusters.
    Stage 2: Inter-cluster Distance Outlier Detection - "Lonely" clusters are outliers.
    
    Performance optimized: Limited max clusters to avoid O(N²) explosion.
    Threshold refreshes periodically for better online performance.
    """
    
    def __init__(self, node_columns: List[str], width: float = 5.0, k: int = 5, 
                 max_clusters: int = 100, threshold_refresh: int = 100):
        self.node_columns = node_columns
        self.width = width
        self.k = k
        self.max_clusters = max_clusters
        self.threshold_refresh = threshold_refresh  # Refresh threshold every N samples
        
        # Clustering State
        self.clusters = []
        self.cluster_counts = []
        
        # Stats
        self.detected_count = 0
        self.consumed_energy = 0.0
        self.call_count = 0  # Track calls for periodic refresh
        
        # Threshold cache
        self._cached_threshold = None
        self._last_refresh = 0
        
        # Energy Params
        self.dist_nodes_to_ch = [np.random.uniform(5, RADIO_RANGE) for _ in range(len(node_columns))]
        self.dist_ch_to_sink = 50.0
    
    def consume_energy(self, amount):
        self.consumed_energy += amount

    def detect(self, observation: np.ndarray) -> Tuple[bool, float]:
        """
        Process observation:
        1. Cluster Assignment/Creation (Stage 1)
        2. Outlier Detection based on kNN Cluster Distances (Stage 2)
        """
        self.call_count += 1
        
        # --- Energy Model ---
        bits = PACKET_SIZE * 8
        for d in self.dist_nodes_to_ch:
             tx_node = (E_ELEC + E_AMP * d**2) * bits
             self.consume_energy(tx_node)
        
        rx_ch = len(self.dist_nodes_to_ch) * E_ELEC * bits
        self.consume_energy(rx_ch)
        
        # --- Stage 1: Fixed-width Clustering ---
        assigned_cluster_idx = -1
        
        if not self.clusters:
            self.clusters.append(observation.copy())
            self.cluster_counts.append(1)
            assigned_cluster_idx = 0
        else:
            cluster_array = np.array(self.clusters)
            dists = np.linalg.norm(cluster_array - observation, axis=1)
            min_dist = np.min(dists)
            nearest_idx = np.argmin(dists)
            
            if min_dist < self.width:
                N = self.cluster_counts[nearest_idx]
                center = self.clusters[nearest_idx]
                new_center = (center * N + observation) / (N + 1)
                self.clusters[nearest_idx] = new_center
                self.cluster_counts[nearest_idx] += 1
                assigned_cluster_idx = nearest_idx
            else:
                if len(self.clusters) < self.max_clusters:
                    self.clusters.append(observation.copy())
                    self.cluster_counts.append(1)
                    assigned_cluster_idx = len(self.clusters) - 1
                else:
                    self.cluster_counts[nearest_idx] += 1
                    assigned_cluster_idx = nearest_idx
                
        # --- Stage 2: Outlier Detection ---
        is_outlier = False
        metric_val = 0.0
        
        n_clusters = len(self.clusters)
        
        if 2 <= n_clusters <= self.max_clusters:
            my_cluster = self.clusters[assigned_cluster_idx]
            
            # Distance from my cluster to all other clusters
            other_dists = []
            for i, c in enumerate(self.clusters):
                if i != assigned_cluster_idx:
                    other_dists.append(np.linalg.norm(my_cluster - c))
            
            if other_dists:
                other_dists.sort()
                k_nearest = other_dists[:self.k]
                my_score = np.mean(k_nearest)
                metric_val = my_score
                
                # Refresh threshold periodically (every threshold_refresh calls)
                # This keeps the algorithm responsive in online mode
                need_refresh = (
                    self._cached_threshold is None or 
                    (self.call_count - self._last_refresh) >= self.threshold_refresh
                )
                
                if need_refresh:
                    all_scores = []
                    for i in range(n_clusters):
                        ci = self.clusters[i]
                        dists_i = []
                        for j in range(n_clusters):
                            if i != j:
                                dists_i.append(np.linalg.norm(ci - self.clusters[j]))
                        dists_i.sort()
                        score_i = np.mean(dists_i[:self.k]) if len(dists_i) >= self.k else np.mean(dists_i) if dists_i else 0
                        all_scores.append(score_i)
                    
                    if all_scores:
                        global_mean = np.mean(all_scores)
                        global_std = np.std(all_scores)
                        self._cached_threshold = global_mean + global_std
                        self._last_refresh = self.call_count
                
                # Use threshold
                if self._cached_threshold and my_score > self._cached_threshold:
                    is_outlier = True
        
        if is_outlier:
            self.detected_count += 1
            
        # --- Energy Model (Forwarding) ---
        if not is_outlier:
             tx_sink = (E_ELEC + E_AMP * self.dist_ch_to_sink**2) * (len(self.dist_nodes_to_ch) * bits)
             self.consume_energy(tx_sink)
             
        self.consume_energy(E_DA * bits * 20)
        
        return is_outlier, metric_val
