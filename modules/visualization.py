"""
Visualization Module for 4-Attribute Sensor Data.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from .config import SENSOR_ATTRIBUTES


def plot_energy(oda_energy, od_energy, save_path='wsn_energy_comparison.png'):
    """Plot energy comparison (Figure 6 style)."""
    labels = ['ODA-MD', 'OD']
    energies = [oda_energy, od_energy]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, energies, color=['blue', 'green'], width=0.5)
    
    plt.ylabel('Energy Consumption (Joules)', fontsize=12)
    plt.title('Energy Consumption Comparison\n(Full Network - 1000 Outliers)', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f} J',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
                 
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Energy plot saved to: {save_path}")


def plot_comparison(oda_results: dict, od_results: dict, save_path: str = 'wsn_comparison_results.png'):
    """Create comparison plots like Figure 4 and 5."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Detection Accuracy vs Time (Figure 4)
    ax1 = axes[0]
    t1, da1 = zip(*oda_results['da_history'])
    t2, da2 = zip(*od_results['da_history'])
    
    ax1.plot(t1, da1, 'b-', linewidth=2, label='ODA-MD')
    ax1.plot(t2, da2, 'g-', linewidth=2, label='OD')
    
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Detection Accuracy (%)', fontsize=12)
    ax1.set_title('Detection Accuracy vs Simulation Time\n(4-Attribute Data)', 
                  fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)
    ax1.set_xlim(0, 2100)
    
    # Plot 2: False Alarm Rate vs Time (Figure 5)
    ax2 = axes[1]
    t1, far1 = zip(*oda_results['far_history'])
    t2, far2 = zip(*od_results['far_history'])
    
    ax2.plot(t1, far1, 'b-', linewidth=2, label='ODA-MD')
    ax2.plot(t2, far2, 'g-', linewidth=2, label='OD')
    
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('False Alarm Rate (%)', fontsize=12)
    ax2.set_title('False Alarm Rate vs Simulation Time\n(4-Attribute Data)', 
                  fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 2100)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to: {save_path}")


def plot_temperature(df_original: pd.DataFrame, df_modified: pd.DataFrame,
                     ground_truth: np.ndarray, predictions: np.ndarray,
                     save_path: str = 'temperature_with_outliers.png'):
    """
    Plot all 4 sensor attributes for nodes 36, 37, 38.
    Updated for 4-attribute data structure.
    """
    node_ids = [36, 37, 38]
    
    # Check if we have 4-attribute data (new format) or single-attribute (old format)
    sample_col = f'node_{node_ids[0]}_temperature'
    is_4attr_format = sample_col in df_modified.columns
    
    if is_4attr_format:
        # New 4-attribute format
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        
        for attr_idx, attr in enumerate(SENSOR_ATTRIBUTES):
            ax = axes[attr_idx]
            
            for node_idx, node_id in enumerate(node_ids):
                col_name = f'node_{node_id}_{attr}'
                if col_name in df_modified.columns:
                    ax.plot(range(len(df_modified)), df_modified[col_name].values,
                            color=colors[node_idx], alpha=0.7, linewidth=0.5, 
                            label=f'Node {node_id}')
            
            # Mark ground truth outliers on temperature plot only
            if attr_idx == 0:  # temperature
                outlier_mask = ground_truth == 1
                first_node_col = f'node_{node_ids[0]}_temperature'
                if first_node_col in df_modified.columns:
                    ax.scatter(np.where(outlier_mask)[0], 
                              df_modified[first_node_col].values[outlier_mask],
                              c='red', s=20, marker='x', label='Injected outliers', zorder=5)
            
            ax.set_ylabel(f'{attr.capitalize()}', fontsize=10)
            ax.set_title(f'{attr.capitalize()} - Intel Lab Data (Nodes 36, 37, 38)', 
                        fontsize=11, fontweight='bold')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        axes[3].set_xlabel('Epoch Index', fontsize=11)
    else:
        # Fallback to old format (single attribute)
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        node_columns = ['node_36', 'node_37', 'node_38']
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        
        for i, (col, color) in enumerate(zip(node_columns, colors)):
            ax = axes[i]
            
            if col in df_modified.columns:
                ax.plot(range(len(df_modified)), df_modified[col].values, 
                        color=color, alpha=0.7, linewidth=0.5, label=f'{col} (with outliers)')
                
                outlier_mask = ground_truth == 1
                ax.scatter(np.where(outlier_mask)[0], df_modified[col].values[outlier_mask],
                           c='red', s=20, marker='x', label='Injected outliers', zorder=5)
            
            ax.set_ylabel('Temperature (°C)', fontsize=10)
            ax.set_title(f'{col.replace("_", " ").title()} - Intel Lab Data', fontsize=11, fontweight='bold')
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
        
        axes[2].set_xlabel('Epoch Index', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Sensor data plot saved to: {save_path}")


def plot_network_topology(topology, save_path: str = 'network_topology.png'):
    """
    Plot the full network topology showing all nodes and clusters.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, topology.n_clusters))
    
    for cluster_id, cluster_info in topology.clusters.items():
        color = colors[cluster_id - 1]
        
        for node_id in cluster_info.member_ids:
            node = topology.nodes[node_id]
            
            if node.is_ch:
                ax.scatter(node.x, node.y, c=[color], s=200, marker='s', 
                          edgecolors='black', linewidths=2, zorder=5)
                ax.annotate(f'CH{cluster_id}', (node.x, node.y), 
                           textcoords="offset points", xytext=(0, 10),
                           ha='center', fontsize=8, fontweight='bold')
            else:
                ax.scatter(node.x, node.y, c=[color], s=80, marker='o',
                          edgecolors='gray', linewidths=0.5, zorder=3)
        
        ch_node = topology.nodes[cluster_info.ch_id]
        for node_id in cluster_info.member_ids:
            if node_id != cluster_info.ch_id:
                node = topology.nodes[node_id]
                ax.plot([node.x, ch_node.x], [node.y, ch_node.y], 
                       color=color, alpha=0.3, linewidth=0.5, zorder=1)
    
    # Plot Sink node
    sink = topology.nodes[1]
    ax.scatter(sink.x, sink.y, c='red', s=400, marker='*', 
              edgecolors='black', linewidths=2, zorder=10)
    ax.annotate('SINK', (sink.x, sink.y), textcoords="offset points", 
               xytext=(0, 15), ha='center', fontsize=10, fontweight='bold', color='red')
    
    # Draw lines from CHs to Sink
    for cluster_id, cluster_info in topology.clusters.items():
        ch_node = topology.nodes[cluster_info.ch_id]
        ax.plot([ch_node.x, sink.x], [ch_node.y, sink.y], 
               color='red', alpha=0.5, linewidth=1, linestyle='--', zorder=2)
    
    ax.set_xlim(-5, topology.area_width + 5)
    ax.set_ylim(-5, topology.area_height + 5)
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title(f'WSN Network Topology\n{topology.n_nodes} Nodes, {topology.n_clusters} Clusters, '
                f'{topology.area_width}x{topology.area_height}m\n(4 Attributes per Node)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='red', 
               markersize=15, label='Sink'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
               markersize=10, label='Cluster Head'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=8, label='Sensor Node'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Network topology saved to: {save_path}")
