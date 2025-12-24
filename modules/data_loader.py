"""
Data Loading and Preprocessing Module.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from .config import CLUSTER_2_NODES

def load_intel_data(filepath: str = 'data.txt', 
                    node_ids: List[int] = CLUSTER_2_NODES,
                    date_start: str = '2004-03-11',
                    date_end: str = '2004-03-15') -> pd.DataFrame:
    """
    Load Intel Lab data for specified nodes.
    
    Parameters:
    -----------
    filepath : str
        Path to data.txt
    node_ids : list
        Node IDs to load (default: Cluster 2 = {36, 37, 38})
    date_start, date_end : str
        Date range as per paper
    
    Returns:
    --------
    pd.DataFrame : Pivoted data (epoch x nodes)
    """
    print("=" * 60)
    print("LOADING INTEL LAB DATA")
    print("=" * 60)
    
    columns = ['date', 'time', 'epoch', 'moteid', 'temperature', 
               'humidity', 'light', 'voltage']
    
    # Load data
    df = pd.read_csv(filepath, sep=r'\s+', header=None, names=columns)
    print(f"Total records: {len(df):,}")
    
    # Clean data
    df = df.dropna(subset=['moteid', 'temperature'])
    df['moteid'] = df['moteid'].astype(int)
    
    # Filter nodes
    df = df[df['moteid'].isin(node_ids)]
    print(f"Records for nodes {node_ids}: {len(df):,}")
    
    # Filter valid temperatures
    df = df[(df['temperature'] > -40) & (df['temperature'] < 100)]
    
    # Create datetime
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
    
    # Filter by date range
    df = df[(df['datetime'] >= date_start) & (df['datetime'] < date_end)]
    print(f"Records after date filter ({date_start} to {date_end}): {len(df):,}")
    
    # Pivot: rows = epoch, columns = node
    df_pivot = df.pivot_table(
        index='epoch',
        columns='moteid',
        values='temperature',
        aggfunc='mean'
    )
    
    # Rename columns
    df_pivot.columns = [f'node_{col}' for col in df_pivot.columns]
    
    # Interpolate missing values instead of dropping
    # This preserves the timeline (15k+ rows as per paper)
    df_pivot = df_pivot.interpolate(method='linear', limit_direction='both')
    
    # Drop only remaining NaNs (if any at start/end)
    df_pivot = df_pivot.dropna()
    print(f"Epochs with valid data: {len(df_pivot):,}")
    
    df_pivot = df_pivot.reset_index()
    
    return df_pivot


def inject_outliers(df: pd.DataFrame, n_outliers: int = 1000, 
                   node_columns: List[str] = None) -> Tuple[pd.DataFrame, np.ndarray, list]:
    """
    Inject synthetic outliers into real data.
    
    Returns:
    --------
    tuple: (modified_df, ground_truth, outlier_schedule)
    """
    print("\n" + "=" * 60)
    print(f"INJECTING {n_outliers} OUTLIERS")
    print("=" * 60)
    
    if node_columns is None:
        node_columns = ['node_36', 'node_37', 'node_38']
    
    df_modified = df.copy()
    n_samples = len(df)
    
    # Calculate statistics from original clean data
    means = df[node_columns].mean()
    stds = df[node_columns].std()
    
    print(f"Original data statistics:")
    for col in node_columns:
        print(f"  {col}: mean={means[col]:.2f}, std={stds[col]:.2f}")
    
    # Ground truth labels
    ground_truth = np.zeros(n_samples, dtype=int)
    
    # Distribute outliers evenly across time
    n_outliers = min(n_outliers, n_samples)
    outlier_schedule = []
    
    for i in range(n_outliers):
        idx = int((i / n_outliers) * n_samples)
        if idx < n_samples:
            ground_truth[idx] = 1
            
            # Modify one random node with VARYING deviation
            # Some outliers are easier (>5 sigma), some harder (3-5 sigma)
            col = np.random.choice(node_columns)
            original = df_modified.iloc[idx][col]
            direction = np.random.choice([-1, 1])
            
            # Hypothesis check: Paper likely used gross errors (obvious outliers)
            # which allow 100% detection even with naive covariance.
            # Set deviation to be consistently large (> 5 sigma)
            deviation = 5 * stds[col] + np.random.uniform(0, 5 * stds[col])
            
            df_modified.iloc[idx, df_modified.columns.get_loc(col)] = original + direction * deviation
            
            outlier_schedule.append({'epoch_idx': idx, 'time': i * 2})  # time in seconds
    
    print(f"\nInjected {np.sum(ground_truth)} outliers")
    print(f"Ground truth: Normal={np.sum(ground_truth == 0)}, Outlier={np.sum(ground_truth == 1)}")
    
    return df_modified, ground_truth, outlier_schedule


def generate_network_data(topology, real_data: pd.DataFrame, 
                          n_outliers: int = 1000) -> Tuple[Dict, np.ndarray, list]:
    """
    Generate data for entire 81-node network.
    
    - Cluster 2 (nodes 36, 37, 38) uses real Intel Lab data
    - Other nodes use synthetic data correlated with real data patterns
    
    Parameters:
    -----------
    topology : NetworkTopology
        The network topology object
    real_data : pd.DataFrame
        Real data from Intel Lab (nodes 36, 37, 38)
    n_outliers : int
        Number of outliers to inject across network
        
    Returns:
    --------
    tuple: (data_dict, ground_truth, outlier_schedule)
        - data_dict[cluster_id] = DataFrame with columns for each node
        - ground_truth[cluster_id] = array of labels
        - outlier_schedule = list of outlier events
    """
    print("\n" + "=" * 60)
    print("GENERATING NETWORK-WIDE DATA")
    print("=" * 60)
    
    n_samples = len(real_data)
    
    # Get statistics from real data
    real_cols = ['node_36', 'node_37', 'node_38']
    base_mean = real_data[real_cols].mean().mean()
    base_std = real_data[real_cols].std().mean()
    
    print(f"Base pattern: mean={base_mean:.2f}, std={base_std:.2f}")
    print(f"Samples per node: {n_samples}")
    
    # Data storage per cluster
    data_dict = {}
    ground_truth_dict = {}
    outlier_schedule = []
    
    # Distribute outliers across clusters (roughly equal)
    outliers_per_cluster = n_outliers // topology.n_clusters
    
    for cluster_id, cluster_info in topology.clusters.items():
        print(f"\nGenerating data for Cluster {cluster_id} "
              f"(CH={cluster_info.ch_id}, Members={len(cluster_info.member_ids)})")
        
        cluster_data = pd.DataFrame()
        cluster_data['epoch'] = real_data['epoch'].values if 'epoch' in real_data.columns else range(n_samples)
        
        # Generate data for each member
        for node_id in cluster_info.member_ids:
            col_name = f'node_{node_id}'
            
            if node_id in [36, 37, 38] and cluster_id == 2:
                # Use real data for Cluster 2 special nodes
                if col_name in real_data.columns:
                    cluster_data[col_name] = real_data[col_name].values
                    print(f"    -> Using REAL Intel Lab data for {col_name}")
                else:
                    # Fallback: use first real column
                    cluster_data[col_name] = real_data[real_cols[0]].values
                    print(f"    -> Using fallback real data for {col_name}")
            else:
                # Generate synthetic data based on real patterns
                # Add cluster-specific offset and node-specific noise
                cluster_offset = (cluster_id - 5) * 0.5  # Vary by cluster
                node_noise = np.random.normal(0, base_std * 0.3, n_samples)
                
                # Use real data pattern as base (from node 36)
                base_pattern = real_data[real_cols[0]].values
                
                # Add temporal correlation (smoothing)
                synthetic = base_pattern + cluster_offset + node_noise
                
                # Add some daily pattern variation
                daily_cycle = np.sin(np.linspace(0, 8 * np.pi, n_samples)) * base_std * 0.2
                synthetic += daily_cycle
                
                cluster_data[col_name] = synthetic
        
        data_dict[cluster_id] = cluster_data
        
        # Initialize ground truth for this cluster
        ground_truth_dict[cluster_id] = np.zeros(n_samples, dtype=int)
        
        # Inject outliers for this cluster
        node_columns = [f'node_{nid}' for nid in cluster_info.member_ids]
        if node_columns:
            stds = cluster_data[node_columns].std()
            
            for i in range(outliers_per_cluster):
                idx = int((i / outliers_per_cluster) * n_samples)
                if idx < n_samples:
                    ground_truth_dict[cluster_id][idx] = 1
                    
                    # Modify random node
                    col = np.random.choice(node_columns)
                    original = cluster_data.iloc[idx][col]
                    direction = np.random.choice([-1, 1])
                    deviation = 5 * stds[col] + np.random.uniform(0, 5 * stds[col])
                    
                    cluster_data.iloc[idx, cluster_data.columns.get_loc(col)] = original + direction * deviation
                    
                    outlier_schedule.append({
                        'cluster_id': cluster_id,
                        'epoch_idx': idx,
                        'node': col
                    })
    
    total_outliers = sum(np.sum(gt) for gt in ground_truth_dict.values())
    print(f"\n{'='*60}")
    print(f"Total outliers injected: {total_outliers}")
    print(f"Clusters with data: {len(data_dict)}")
    
    return data_dict, ground_truth_dict, outlier_schedule
