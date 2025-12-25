"""
Network Components and Algorithms for 4-Attribute Sensor Data.
Implements TIME-STEP based processing as per Paper's Figure 2:
- At each time t: CH sends 1 request
- Each node sends back A_k = [temp, humid, light, volt] (4 values at time t)
- CH builds Table_MD with shape (n_attributes=4, p_nodes)
- CH computes MD for each node and decides outlier/normal
"""

import numpy as np
from scipy.stats import chi2
from typing import List, Tuple, Dict
from .config import *


# =============================================================================
# SENSOR NODE (4-Attribute, Time-Step Version)
# =============================================================================

class SensorNode:
    """
    Sensor Node that sends 4-attribute observation at each time step.
    Per Figure 2: A_k = [x_1k, x_2k, x_3k, x_4k] = [temp, humid, light, volt]
    """
    def __init__(self, node_id: int, data_matrix: np.ndarray):
        """
        Parameters:
        -----------
        node_id : int
            Node identifier (k in paper)
        data_matrix : np.ndarray
            Shape (n_epochs, 4) - all observations for this node
        """
        self.node_id = node_id
        self.data = data_matrix  # Shape: (n_epochs, 4)
        self.energy = INITIAL_ENERGY
        self.dist_to_ch = np.random.uniform(5, 30)
        self.consumed_energy = 0.0

    def consume_energy(self, amount: float):
        self.energy -= amount
        self.consumed_energy += amount

    def receive_request(self):
        """Receive request from CH (Algorithm 1, step 1)."""
        e = E_ELEC * PACKET_SIZE * 8
        self.consume_energy(e)
        
    def send_observation(self, epoch_idx: int) -> np.ndarray:
        """
        Send vector A_k at time t (Figure 2).
        A_k = [temperature, humidity, light, voltage]
        
        Returns:
        --------
        np.ndarray: Shape (4,) - [temp, humid, light, volt]
        """
        if epoch_idx >= len(self.data):
            return np.zeros(N_ATTRIBUTES)
        
        # Get observation at this time step
        observation = self.data[epoch_idx]  # (4,)
        
        # Transmission Energy for 4 attribute values
        total_bits = N_ATTRIBUTES * PACKET_SIZE * 8
        tx_energy = (E_ELEC + E_AMP * self.dist_to_ch**2) * total_bits
        self.consume_energy(tx_energy)
        
        return observation


# =============================================================================
# ODA-MD DETECTOR (Cluster Head) - Time-Step Version per Figure 2
# =============================================================================

class ClusterHead:
    """
    Cluster Head running ODA-MD algorithm per TIME STEP (Figure 2).
    
    At each time t:
    1. CH sends request to all nodes
    2. Each node N_k sends vector A_k = [temp, humid, light, volt]
    3. CH constructs Table_MD with shape (4 attributes, p nodes)
    4. CH computes MD for each node using accumulated statistics
    5. Decision: outlier or normal
    """
    
    def __init__(self, cluster_id: int = CH_ID, node_ids: List[int] = CLUSTER_2_NODES, 
                 member_nodes: List[SensorNode] = None, window_size: int = 50,
                 verbose: bool = False, sample_log_interval: int = 500):
        self.cluster_id = cluster_id
        self.node_ids = node_ids
        self.members = member_nodes if member_nodes else []
        
        # Window for computing statistics (rolling window)
        self.window_size = window_size
        self.history = []  # Store recent observations for statistics
        
        # Energy and stats
        self.energy = INITIAL_ENERGY
        self.consumed_energy = 0.0
        self.detected_count = 0
        
        # Logging settings
        self.verbose = verbose
        self.sample_log_interval = sample_log_interval
        self._epoch_count = 0
        self._sample_logs_printed = 0  # Track how many sample logs printed
        self._outlier_logs_printed = 0  # Track how many outlier logs printed
        self._max_sample_logs = 3  # Max sample queries to print
        self._max_outlier_logs = 2  # Max outlier queries to print

    def consume_energy(self, amount: float):
        self.energy -= amount
        self.consumed_energy += amount

    def _log_query(self, epoch_idx: int, table_md: np.ndarray, 
                   md_values: dict = None, threshold: float = None, 
                   outlier_nodes: list = None):
        """Log a CH query with detailed information."""
        if not self.verbose:
            return
        
        is_outlier = outlier_nodes and len(outlier_nodes) > 0
        
        # Only log: first few samples (after warm-up) + first few outliers
        should_log = False
        log_type = ""
        
        if is_outlier and self._outlier_logs_printed < self._max_outlier_logs:
            should_log = True
            log_type = "OUTLIER"
            self._outlier_logs_printed += 1
        elif not is_outlier and self._epoch_count >= 10 and self._sample_logs_printed < self._max_sample_logs:
            should_log = True
            log_type = "SAMPLE"
            self._sample_logs_printed += 1
        
        if not should_log:
            return
        
        print("\n" + "-" * 65)
        if log_type == "OUTLIER":
            print(f"| [!] CH QUERY #{self._epoch_count} - Cluster {self.cluster_id}, t={epoch_idx} [OUTLIER]")
        else:
            print(f"| [*] CH QUERY #{self._epoch_count} - Cluster {self.cluster_id}, t={epoch_idx} [NORMAL]")
        print("-" * 65)
        
        # Table MD header
        print(f"| Table_MD ({len(self.members)} nodes x 4 attributes):")
        print(f"| {'Node':<6} {'Temp':>10} {'Humid':>10} {'Light':>10} {'Volt':>10}")
        print("|" + "-" * 55)
        
        for idx, member in enumerate(self.members):
            node_id = member.node_id
            obs = table_md[idx]
            md_str = ""
            if md_values and node_id in md_values:
                md_str = f" -> MD={md_values[node_id]:.3f}"
            print(f"| N_{node_id:<3} {obs[0]:>10.2f} {obs[1]:>10.2f} {obs[2]:>10.1f} {obs[3]:>10.4f}{md_str}")
        
        if threshold is not None:
            print(f"| Threshold (chi2, df=4): {threshold:.4f}")
        
        if outlier_nodes:
            print(f"| >>> DECISION: OUTLIER - Node(s) {outlier_nodes}")
        else:
            print(f"| >>> DECISION: NORMAL - All nodes within threshold")
        print("-" * 65)

    def process_time_step(self, epoch_idx: int) -> Tuple[bool, Dict]:
        """
        Process ONE time step as per Figure 2.
        
        Algorithm 1: Initialization
        - CH sends request (broadcast)
        - Each node sends A_k vector
        - CH constructs Table_MD
        
        Algorithm 2: MD Calculation & Detection
        - Compute μ and Σ from history
        - Compute MD for each node
        - Decision based on Chi-square threshold
        
        Returns:
        --------
        (is_outlier, details) - whether this time step has outlier
        """
        self._epoch_count += 1
        n_nodes = len(self.members)
        if n_nodes == 0:
            return False, {}
        
        # ===== ALGORITHM 1: Collect data =====
        # Step 1: CH sends request (broadcast)
        self.consume_energy((E_ELEC + E_AMP * RADIO_RANGE**2) * PACKET_SIZE * 8)
        
        # Step 2: Collect A_k vectors from all nodes
        # Table_MD at time t: shape (4, p) but we store as (p, 4) for convenience
        table_md = []
        for member in self.members:
            member.receive_request()
            obs = member.send_observation(epoch_idx)  # (4,)
            table_md.append(obs)
        
        table_md = np.array(table_md)  # Shape: (n_nodes, 4)
        
        # Receive energy
        total_bits_rx = n_nodes * N_ATTRIBUTES * PACKET_SIZE * 8
        self.consume_energy(E_ELEC * total_bits_rx)
        
        # Add to history for statistics (keep ALL history, not rolling window)
        self.history.append(table_md)
        
        # ===== ALGORITHM 2: MD Calculation & Detection =====
        is_outlier = False
        outlier_nodes = []
        md_values = {}  # Store MD for each node (for logging)
        threshold = None
        
        # Need at least a few samples for meaningful statistics
        if len(self.history) >= 5:
            # Compute statistics from history
            # Stack history: (window, n_nodes, 4)
            history_stack = np.array(self.history)
            
            # Chi-square threshold with df = 4
            chi2_val = chi2.ppf(0.975, df=N_ATTRIBUTES)
            threshold = np.sqrt(chi2_val)
            
            # For each node, compute MD of current observation
            for node_idx in range(n_nodes):
                # Get this node's history: (window, 4)
                node_history = history_stack[:, node_idx, :]
                
                # Compute μ and Σ from history (Eq. 3 & 4)
                mu = np.mean(node_history, axis=0)  # (4,)
                
                if len(node_history) > 1:
                    cov = np.cov(node_history, rowvar=False)  # (4, 4)
                else:
                    cov = np.eye(N_ATTRIBUTES)
                
                # Handle singular matrix
                try:
                    cov_inv = np.linalg.inv(cov)
                except np.linalg.LinAlgError:
                    cov += np.eye(N_ATTRIBUTES) * 1e-6
                    cov_inv = np.linalg.inv(cov)
                
                # Current observation for this node
                current_obs = table_md[node_idx]  # (4,)
                
                # Compute MD (Eq. 2)
                diff = current_obs - mu
                md_sq = np.dot(np.dot(diff, cov_inv), diff)
                md = np.sqrt(max(0, md_sq))
                
                # Store MD for logging
                md_values[self.members[node_idx].node_id] = md
                
                # Decision (Eq. 5)
                if md >= threshold:
                    is_outlier = True
                    outlier_nodes.append(self.members[node_idx].node_id)
        
        # Log query details
        self._log_query(epoch_idx, table_md, md_values, threshold, outlier_nodes)
        
        if is_outlier:
            self.detected_count += 1
            # Don't forward outlier data (save energy)
        else:
            # Forward normal data to sink
            dist_to_sink = 50.0
            bits_to_sink = n_nodes * N_ATTRIBUTES * PACKET_SIZE * 8
            self.consume_energy((E_ELEC + E_AMP * dist_to_sink**2) * bits_to_sink)
        
        # Processing energy
        self.consume_energy(E_DA * PACKET_SIZE * 8 * n_nodes)
        
        return is_outlier, {
            'outlier_nodes': outlier_nodes, 
            'table_md_shape': table_md.shape,
            'md_values': md_values,
            'threshold': threshold
        }


# =============================================================================
# OD DETECTOR (Baseline) - Unchanged from before
# =============================================================================

class OD_Detector:
    """
    Baseline OD algorithm using Fixed-width Clustering on 4-attribute data.
    Processes one time step at a time.
    """
    
    def __init__(self, n_nodes: int, width: float = 5.0, k: int = 5, 
                 max_clusters: int = 100, threshold_refresh: int = 100,
                 verbose: bool = False):
        self.n_nodes = n_nodes
        self.width = width
        self.k = k
        self.max_clusters = max_clusters
        self.threshold_refresh = threshold_refresh
        
        # Clustering State
        self.clusters = []
        self.cluster_counts = []
        
        # Stats
        self.detected_count = 0
        self.consumed_energy = 0.0
        self.call_count = 0
        
        # Threshold cache
        self._cached_threshold = None
        self._last_refresh = 0
        
        # Energy Params
        self.dist_nodes_to_ch = [np.random.uniform(5, RADIO_RANGE) for _ in range(n_nodes)]
        self.dist_ch_to_sink = 50.0
        
        # Verbose settings
        self.verbose = verbose
        self._sample_logs_printed = 0
        self._outlier_logs_printed = 0
        self._max_sample_logs = 2
        self._max_outlier_logs = 2
    
    def consume_energy(self, amount):
        self.consumed_energy += amount
    
    def _log_query(self, observations: np.ndarray, assigned_cluster: int,
                   min_dist: float, my_score: float, threshold: float, 
                   is_outlier: bool, n_clusters: int):
        """Log OD algorithm query with detailed information."""
        if not self.verbose:
            return
        
        # Only log first few samples + outliers
        should_log = False
        log_type = ""
        
        if is_outlier and self._outlier_logs_printed < self._max_outlier_logs:
            should_log = True
            log_type = "OUTLIER"
            self._outlier_logs_printed += 1
        elif not is_outlier and self.call_count >= 10 and self._sample_logs_printed < self._max_sample_logs:
            should_log = True
            log_type = "SAMPLE"
            self._sample_logs_printed += 1
        
        if not should_log:
            return
        
        print("\n" + "-" * 65)
        if log_type == "OUTLIER":
            print(f"| [!] OD QUERY #{self.call_count} [OUTLIER DETECTED]")
        else:
            print(f"| [*] OD QUERY #{self.call_count} [NORMAL]")
        print("-" * 65)
        
        # Show clustering info
        print(f"| Current clusters: {n_clusters} (max: {self.max_clusters})")
        print(f"| Assigned to cluster: #{assigned_cluster}")
        print(f"| Distance to nearest cluster: {min_dist:.4f} (width: {self.width})")
        print(f"| k-NN score: {my_score:.4f}")
        print(f"| Threshold (mean+std): {threshold:.4f}" if threshold else "| Threshold: Not yet computed")
        
        if is_outlier:
            print(f"| >>> DECISION: OUTLIER (score {my_score:.4f} > threshold {threshold:.4f})")
        else:
            print(f"| >>> DECISION: NORMAL (score {my_score:.4f} <= threshold {threshold:.4f})" if threshold else "| >>> DECISION: NORMAL (not enough data)")
        print("-" * 65)

    def detect(self, observations: np.ndarray) -> Tuple[bool, float]:
        """
        Process observations for one time step.
        
        Parameters:
        -----------
        observations : np.ndarray
            Shape (n_nodes, 4) - all node observations at this time
            
        Returns:
        --------
        (is_outlier, metric_value)
        """
        self.call_count += 1
        
        # Flatten observations to create observation vector
        obs_flat = observations.flatten()
        
        # --- Energy Model ---
        bits = PACKET_SIZE * 8 * N_ATTRIBUTES
        for d in self.dist_nodes_to_ch:
            tx_node = (E_ELEC + E_AMP * d**2) * bits
            self.consume_energy(tx_node)
        
        rx_ch = len(self.dist_nodes_to_ch) * E_ELEC * bits
        self.consume_energy(rx_ch)
        
        # --- Stage 1: Fixed-width Clustering ---
        assigned_cluster_idx = -1
        
        if not self.clusters:
            self.clusters.append(obs_flat.copy())
            self.cluster_counts.append(1)
            assigned_cluster_idx = 0
        else:
            cluster_array = np.array(self.clusters)
            dists = np.linalg.norm(cluster_array - obs_flat, axis=1)
            min_dist = np.min(dists)
            nearest_idx = np.argmin(dists)
            
            if min_dist < self.width:
                N = self.cluster_counts[nearest_idx]
                center = self.clusters[nearest_idx]
                new_center = (center * N + obs_flat) / (N + 1)
                self.clusters[nearest_idx] = new_center
                self.cluster_counts[nearest_idx] += 1
                assigned_cluster_idx = nearest_idx
            else:
                if len(self.clusters) < self.max_clusters:
                    self.clusters.append(obs_flat.copy())
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
            
            other_dists = []
            for i, c in enumerate(self.clusters):
                if i != assigned_cluster_idx:
                    other_dists.append(np.linalg.norm(my_cluster - c))
            
            if other_dists:
                other_dists.sort()
                k_nearest = other_dists[:self.k]
                my_score = np.mean(k_nearest)
                metric_val = my_score
                
                # Periodic threshold refresh
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
                
                if self._cached_threshold and my_score > self._cached_threshold:
                    is_outlier = True
        
        # Log query if verbose
        self._log_query(
            observations=observations,
            assigned_cluster=assigned_cluster_idx,
            min_dist=min_dist if 'min_dist' in dir() else 0,
            my_score=metric_val,
            threshold=self._cached_threshold,
            is_outlier=is_outlier,
            n_clusters=n_clusters
        )
        
        if is_outlier:
            self.detected_count += 1
            
        # OD forwards ALL data to sink (no filtering)
        tx_sink = (E_ELEC + E_AMP * self.dist_ch_to_sink**2) * (len(self.dist_nodes_to_ch) * bits)
        self.consume_energy(tx_sink)
             
        self.consume_energy(E_DA * bits * self.n_nodes)
        
        return is_outlier, metric_val
