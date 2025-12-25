"""
Data Loading and Preprocessing Module.
Loads all 4 sensor attributes: temperature, humidity, light, voltage.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from .config import CLUSTER_2_NODES, SENSOR_ATTRIBUTES, N_ATTRIBUTES


def load_intel_data(filepath: str = 'data.txt', 
                    node_ids: List[int] = CLUSTER_2_NODES,
                    date_start: str = '2004-03-11',
                    date_end: str = '2004-03-15') -> pd.DataFrame:
    """
    Load Intel Lab data for specified nodes with ALL 4 sensor attributes.
    
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
    pd.DataFrame : Data with columns for each node and each attribute
        Schema: epoch, node_36_temperature, node_36_humidity, node_36_light, node_36_voltage, ...
    """
    print("=" * 60)
    print("LOADING INTEL LAB DATA (4 ATTRIBUTES)")
    print("=" * 60)
    
    columns = ['date', 'time', 'epoch', 'moteid', 'temperature', 
               'humidity', 'light', 'voltage']
    
    # Load data
    df = pd.read_csv(filepath, sep=r'\s+', header=None, names=columns)
    print(f"Total records: {len(df):,}")
    
    # Clean data - drop rows with any missing sensor values
    df = df.dropna(subset=['moteid', 'temperature', 'humidity', 'light', 'voltage'])
    df['moteid'] = df['moteid'].astype(int)
    
    # Filter nodes
    df = df[df['moteid'].isin(node_ids)]
    print(f"Records for nodes {node_ids}: {len(df):,}")
    
    # Filter valid sensor readings
    df = df[(df['temperature'] > -40) & (df['temperature'] < 100)]
    df = df[(df['humidity'] >= 0) & (df['humidity'] <= 100)]
    df = df[(df['light'] >= 0)]
    df = df[(df['voltage'] > 0) & (df['voltage'] < 5)]
    
    # Create datetime
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
    
    # Filter by date range
    df = df[(df['datetime'] >= date_start) & (df['datetime'] < date_end)]
    print(f"Records after date filter ({date_start} to {date_end}): {len(df):,}")
    
    # Pivot for each attribute separately, then merge
    result_df = None
    
    for attr in SENSOR_ATTRIBUTES:
        df_pivot = df.pivot_table(
            index='epoch',
            columns='moteid',
            values=attr,
            aggfunc='mean'
        )
        
        # Rename columns to include attribute name
        df_pivot.columns = [f'node_{col}_{attr}' for col in df_pivot.columns]
        
        if result_df is None:
            result_df = df_pivot
        else:
            result_df = result_df.join(df_pivot, how='outer')
    
    # Interpolate missing values
    result_df = result_df.interpolate(method='linear', limit_direction='both')
    
    # Drop remaining NaNs
    result_df = result_df.dropna()
    print(f"Epochs with valid data: {len(result_df):,}")
    
    # Print statistics for each attribute
    print(f"\nData statistics per attribute:")
    for attr in SENSOR_ATTRIBUTES:
        attr_cols = [c for c in result_df.columns if attr in c]
        if attr_cols:
            mean_val = result_df[attr_cols].mean().mean()
            std_val = result_df[attr_cols].std().mean()
            print(f"  {attr}: mean={mean_val:.2f}, std={std_val:.2f}")
    
    result_df = result_df.reset_index()
    
    return result_df


def get_node_observation(df: pd.DataFrame, node_id: int, epoch_idx: int) -> np.ndarray:
    """
    Get 4-attribute observation vector for a specific node at a specific epoch.
    
    Returns:
    --------
    np.ndarray: [temperature, humidity, light, voltage]
    """
    obs = []
    for attr in SENSOR_ATTRIBUTES:
        col_name = f'node_{node_id}_{attr}'
        if col_name in df.columns:
            obs.append(df.iloc[epoch_idx][col_name])
        else:
            obs.append(0.0)  # Fallback
    return np.array(obs)


def get_node_data_matrix(df: pd.DataFrame, node_id: int) -> np.ndarray:
    """
    Get all observations for a node as a matrix (n_epochs, 4).
    
    Returns:
    --------
    np.ndarray: Shape (n_epochs, 4) - each row is [temp, humid, light, volt]
    """
    data = []
    for attr in SENSOR_ATTRIBUTES:
        col_name = f'node_{node_id}_{attr}'
        if col_name in df.columns:
            data.append(df[col_name].values)
        else:
            data.append(np.zeros(len(df)))
    
    # Stack and transpose to get (n_epochs, 4)
    return np.column_stack(data)


def inject_outliers(df: pd.DataFrame, node_ids: List[int], 
                   n_outliers: int = 1000) -> Tuple[pd.DataFrame, np.ndarray, list]:
    """
    Inject synthetic outliers into real data by modifying one of the 4 attributes.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data with 4 attributes per node
    node_ids : List[int]
        List of node IDs in this cluster
    n_outliers : int
        Number of outliers to inject
    
    Returns:
    --------
    tuple: (modified_df, ground_truth, outlier_schedule)
    """
    print("\n" + "=" * 60)
    print(f"INJECTING {n_outliers} OUTLIERS (4-ATTRIBUTE)")
    print("=" * 60)
    
    df_modified = df.copy()
    n_samples = len(df)
    
    # Get all attribute columns for these nodes
    all_columns = []
    for node_id in node_ids:
        for attr in SENSOR_ATTRIBUTES:
            col_name = f'node_{node_id}_{attr}'
            if col_name in df.columns:
                all_columns.append(col_name)
    
    if not all_columns:
        print("No valid columns found for outlier injection!")
        return df_modified, np.zeros(n_samples, dtype=int), []
    
    # Calculate statistics
    stds = df_modified[all_columns].std()
    
    # Ground truth labels
    ground_truth = np.zeros(n_samples, dtype=int)
    
    # Distribute outliers evenly across time
    n_outliers = min(n_outliers, n_samples)
    outlier_schedule = []
    
    for i in range(n_outliers):
        idx = int((i / n_outliers) * n_samples)
        if idx < n_samples:
            ground_truth[idx] = 1
            
            # Choose random column (node + attribute combination)
            col = np.random.choice(all_columns)
            original = df_modified.iloc[idx][col]
            direction = np.random.choice([-1, 1])
            
            # Large deviation (5-10 sigma) for clear outliers
            deviation = 5 * stds[col] + np.random.uniform(0, 5 * stds[col])
            
            df_modified.iloc[idx, df_modified.columns.get_loc(col)] = original + direction * deviation
            
            outlier_schedule.append({
                'epoch_idx': idx, 
                'column': col,
                'time': i * 2
            })
    
    print(f"\nInjected {np.sum(ground_truth)} outliers across {len(all_columns)} attribute columns")
    print(f"Ground truth: Normal={np.sum(ground_truth == 0)}, Outlier={np.sum(ground_truth == 1)}")
    
    return df_modified, ground_truth, outlier_schedule


def generate_network_data(topology, real_data: pd.DataFrame, 
                          n_outliers: int = 1000) -> Tuple[Dict, Dict, list]:
    """
    Generate 4-attribute data for entire 81-node network.
    
    - Cluster 2 (nodes 36, 37, 38) uses real Intel Lab data (4 attributes)
    - Other nodes use synthetic data correlated with real data patterns
    
    Parameters:
    -----------
    topology : NetworkTopology
        The network topology object
    real_data : pd.DataFrame
        Real data from Intel Lab with 4 attributes per node
    n_outliers : int
        Number of outliers to inject across network
        
    Returns:
    --------
    tuple: (data_dict, ground_truth_dict, outlier_schedule)
        - data_dict[cluster_id] = DataFrame with 4 attributes per node
        - ground_truth_dict[cluster_id] = array of labels (per epoch)
        - outlier_schedule = list of outlier events
    """
    print("\n" + "=" * 60)
    print("GENERATING NETWORK-WIDE DATA (4 ATTRIBUTES)")
    print("=" * 60)
    
    n_samples = len(real_data)
    
    # Get statistics from real data for each attribute
    stats = {}
    for attr in SENSOR_ATTRIBUTES:
        attr_cols = [c for c in real_data.columns if attr in c]
        if attr_cols:
            stats[attr] = {
                'mean': real_data[attr_cols].mean().mean(),
                'std': real_data[attr_cols].std().mean()
            }
            print(f"  {attr}: mean={stats[attr]['mean']:.2f}, std={stats[attr]['std']:.2f}")
    
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
            for attr in SENSOR_ATTRIBUTES:
                col_name = f'node_{node_id}_{attr}'
                real_col = f'node_{node_id}_{attr}'
                
                if node_id in [36, 37, 38] and cluster_id == 2:
                    # Use real data for Cluster 2 special nodes
                    if real_col in real_data.columns:
                        cluster_data[col_name] = real_data[real_col].values
                    else:
                        # Fallback: use node 36's data
                        fallback_col = f'node_36_{attr}'
                        if fallback_col in real_data.columns:
                            cluster_data[col_name] = real_data[fallback_col].values
                        else:
                            cluster_data[col_name] = np.zeros(n_samples)
                else:
                    # Generate synthetic data based on real patterns
                    if attr in stats:
                        base_mean = stats[attr]['mean']
                        base_std = stats[attr]['std']
                    else:
                        base_mean = 20.0
                        base_std = 2.0
                    
                    # Cluster-specific offset
                    cluster_offset = (cluster_id - 5) * (base_std * 0.3)
                    
                    # Node-specific noise
                    node_noise = np.random.normal(0, base_std * 0.3, n_samples)
                    
                    # Get base pattern from real data if available
                    real_pattern_col = f'node_36_{attr}'
                    if real_pattern_col in real_data.columns:
                        base_pattern = real_data[real_pattern_col].values
                    else:
                        base_pattern = np.ones(n_samples) * base_mean
                    
                    # Combine
                    synthetic = base_pattern + cluster_offset + node_noise
                    
                    # Add daily cycle variation
                    daily_cycle = np.sin(np.linspace(0, 8 * np.pi, n_samples)) * base_std * 0.2
                    synthetic += daily_cycle
                    
                    cluster_data[col_name] = synthetic
        
        data_dict[cluster_id] = cluster_data
        
        # Initialize ground truth for this cluster
        ground_truth_dict[cluster_id] = np.zeros(n_samples, dtype=int)
        
        # Get all attribute columns for this cluster's nodes
        all_columns = []
        for node_id in cluster_info.member_ids:
            for attr in SENSOR_ATTRIBUTES:
                col_name = f'node_{node_id}_{attr}'
                if col_name in cluster_data.columns:
                    all_columns.append(col_name)
        
        # Inject outliers for this cluster
        if all_columns:
            stds = cluster_data[all_columns].std()
            
            for i in range(outliers_per_cluster):
                idx = int((i / outliers_per_cluster) * n_samples)
                if idx < n_samples:
                    ground_truth_dict[cluster_id][idx] = 1
                    
                    # Modify random attribute column
                    col = np.random.choice(all_columns)
                    original = cluster_data.iloc[idx][col]
                    direction = np.random.choice([-1, 1])
                    deviation = 5 * stds[col] + np.random.uniform(0, 5 * stds[col])
                    
                    cluster_data.iloc[idx, cluster_data.columns.get_loc(col)] = original + direction * deviation
                    
                    outlier_schedule.append({
                        'cluster_id': cluster_id,
                        'epoch_idx': idx,
                        'column': col
                    })
    
    total_outliers = sum(np.sum(gt) for gt in ground_truth_dict.values())
    print(f"\n{'='*60}")
    print(f"Total outliers injected: {total_outliers}")
    print(f"Clusters with data: {len(data_dict)}")
    
    return data_dict, ground_truth_dict, outlier_schedule
