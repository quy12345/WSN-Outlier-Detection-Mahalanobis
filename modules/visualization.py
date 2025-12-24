"""
Visualization Module.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_energy(oda_energy, od_energy, save_path='wsn_energy_comparison.png'):
    """Plot energy comparison (Figure 6 style - simplified to Bar Chart for single run)."""
    labels = ['ODA-MD', 'OD']
    energies = [oda_energy, od_energy]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, energies, color=['blue', 'green'], width=0.5)
    
    plt.ylabel('Energy Consumption (Joules)', fontsize=12)
    plt.title('Energy Consumption Comparison\n(Cluster 2 - 1000 Outliers)', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
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
    ax1.set_title('Detection Accuracy vs Simulation Time\n(Figure 4 - Using Intel Lab Data)', 
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
    ax2.set_title('False Alarm Rate vs Simulation Time\n(Figure 5 - Using Intel Lab Data)', 
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
    """Plot temperature data with detected outliers."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    node_columns = ['node_36', 'node_37', 'node_38']
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    for i, (col, color) in enumerate(zip(node_columns, colors)):
        ax = axes[i]
        
        # Plot modified data
        ax.plot(range(len(df_modified)), df_modified[col].values, 
                color=color, alpha=0.7, linewidth=0.5, label=f'{col} (with outliers)')
        
        # Mark ground truth outliers
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
    print(f"Temperature plot saved to: {save_path}")


def plot_network_topology(topology, save_path: str = 'network_topology.png'):
    """
    Plot the full network topology showing all nodes and clusters.
    
    Parameters:
    -----------
    topology : NetworkTopology
        The network topology object
    save_path : str
        Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Color map for clusters
    colors = plt.cm.tab10(np.linspace(0, 1, topology.n_clusters))
    
    # Plot each cluster
    for cluster_id, cluster_info in topology.clusters.items():
        color = colors[cluster_id - 1]
        
        # Plot member nodes
        for node_id in cluster_info.member_ids:
            node = topology.nodes[node_id]
            
            if node.is_ch:
                # CH node - larger, square marker
                ax.scatter(node.x, node.y, c=[color], s=200, marker='s', 
                          edgecolors='black', linewidths=2, zorder=5)
                ax.annotate(f'CH{cluster_id}', (node.x, node.y), 
                           textcoords="offset points", xytext=(0, 10),
                           ha='center', fontsize=8, fontweight='bold')
            else:
                # Regular node - circle
                ax.scatter(node.x, node.y, c=[color], s=80, marker='o',
                          edgecolors='gray', linewidths=0.5, zorder=3)
        
        # Draw lines from members to CH
        ch_node = topology.nodes[cluster_info.ch_id]
        for node_id in cluster_info.member_ids:
            if node_id != cluster_info.ch_id:
                node = topology.nodes[node_id]
                ax.plot([node.x, ch_node.x], [node.y, ch_node.y], 
                       color=color, alpha=0.3, linewidth=0.5, zorder=1)
    
    # Plot Sink node
    sink = topology.nodes[topology.nodes[1].node_id]  # Sink ID = 1
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
                f'{topology.area_width}x{topology.area_height}m', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Legend
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
