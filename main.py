"""
WSN Simulation - ODA-MD vs OD Comparison
Paper: "Outlier Detection Algorithm based on Mahalanobis Distance for WSN"
Focus: Cluster 2 only (Section V.C)
"""

import numpy as np
import warnings
from modules.config import CLUSTER_2_NODES, CH_ID, N_NODES, N_CLUSTERS, AREA_WIDTH, AREA_HEIGHT
from modules.topology import NetworkTopology
from modules.data_loader import load_intel_data, inject_outliers
from modules.simulation import run_cluster_simulation
from modules.visualization import (plot_comparison, plot_energy, 
                                   plot_temperature, plot_network_topology)

warnings.filterwarnings('ignore')
np.random.seed(42)


def print_network_info(topology):
    """Print network topology information."""
    print("\n" + "-" * 70)
    print("| [1] NETWORK TOPOLOGY INITIALIZATION                                |")
    print("-" * 70)
    
    print(f"  > Created {topology.n_nodes} sensor nodes in {topology.area_width}x{topology.area_height}m area")
    print(f"  > Formed {topology.n_clusters} clusters using K-means clustering")
    print(f"  > Sink node: Node {1} at center ({topology.area_width/2:.0f}, {topology.area_height/2:.0f})m")
    
    # Show cluster summary
    print("\n  Cluster Distribution:")
    for cid in sorted(topology.clusters.keys()):
        cluster = topology.clusters[cid]
        n_members = len(cluster.member_ids)
        ch_id = cluster.ch_id
        if cid == 2:
            print(f"    * Cluster {cid}: CH=Node_{ch_id}, {n_members} members {cluster.member_ids} [FOCUS - Intel Lab Data]")
        else:
            members_preview = cluster.member_ids[:3]
            print(f"      Cluster {cid}: CH=Node_{ch_id}, {n_members} members {members_preview}...")
    
    print(f"\n  Note: Simulation runs on FULL NETWORK but algorithm executed on Cluster 2 only")


def main(verbose: bool = True):
    """
    Run ODA-MD simulation on Cluster 2 with Intel Lab data.
    
    Parameters:
    -----------
    verbose : bool
        Whether to print detailed algorithm execution (default: True)
    """
    
    # Banner
    print("\n")
    print("=" * 70)
    print("        WIRELESS SENSOR NETWORK OUTLIER DETECTION SIMULATION")
    print("=" * 70)
    print("  Algorithm: ODA-MD (Proposed) vs OD (Baseline)")
    print("=" * 70)
    
    # 1. Create Network Topology
    topology = NetworkTopology()
    print_network_info(topology)
    
    # 2. Load Data
    print("\n" + "-" * 70)
    print("| [2] DATA PREPARATION                                               |")
    print("-" * 70)
    
    real_data = load_intel_data('data.txt', CLUSTER_2_NODES)
    cluster_data, ground_truth, _ = inject_outliers(real_data, CLUSTER_2_NODES, n_outliers=1000)
    
    print(f"\n  Data Summary:")
    print(f"    > Source: Intel Lab Dataset (March 2004)")
    print(f"    > Nodes: {CLUSTER_2_NODES}")
    print(f"    > Attributes: [temperature, humidity, light, voltage]")
    print(f"    > Total epochs: {len(real_data)}")
    print(f"    > Injected outliers: 1000 (5-10 sigma deviation)")
    
    # 3. Run Simulations
    print("\n" + "-" * 70)
    print("| [3] ALGORITHM EXECUTION                                            |")
    print("-" * 70)
    print(f"  Verbose mode: {'ON' if verbose else 'OFF'}")
    print(f"  Showing: First 3 normal queries + First 2 outlier queries per algorithm")
    
    # ODA-MD
    print("\n" + "~" * 60)
    print("  >> ALGORITHM 1: ODA-MD (Mahalanobis Distance)")
    print("~" * 60)
    print("  Method: Compute MD for each node using covariance matrix")
    print("  Threshold: Chi-square distribution (df=4, alpha=0.025)")
    
    oda = run_cluster_simulation(2, cluster_data, ground_truth, 'ODA-MD',
                                  verbose=verbose, sample_log_interval=1000)
    print(f"\n  [COMPLETE] ODA-MD: Detected {oda['TP'] + oda['FP']} potential outliers")
    
    # OD (Baseline)
    print("\n" + "~" * 60)
    print("  >> ALGORITHM 2: OD (Baseline - Fixed-width Clustering)")
    print("~" * 60)
    print("  Method: Euclidean distance with k-nearest cluster centers")
    print("  Threshold: Mean + Std of k-NN distances (adaptive)")
    
    od = run_cluster_simulation(2, cluster_data, ground_truth, 'OD',
                                 verbose=verbose, sample_log_interval=1000)
    print(f"\n  [COMPLETE] OD: Detected {od['TP'] + od['FP']} potential outliers")
    
    # 4. Results Summary
    print("\n" + "=" * 70)
    print("| [4] RESULTS COMPARISON                                             |")
    print("=" * 70)
    print(f"\n  {'Algorithm':<15} {'DA':>10} {'FAR':>10} {'Energy':>12} {'TP':>6} {'FN':>6}")
    print("  " + "-" * 65)
    print(f"  {'ODA-MD':<15} {oda['DA']*100:>9.2f}% {oda['FAR']*100:>9.2f}% {oda['energy']:>11.4f}J {oda['TP']:>6} {oda['FN']:>6}")
    print(f"  {'OD (Baseline)':<15} {od['DA']*100:>9.2f}% {od['FAR']*100:>9.2f}% {od['energy']:>11.4f}J {od['TP']:>6} {od['FN']:>6}")
    
    # Performance comparison
    da_improvement = (oda['DA'] - od['DA']) * 100
    energy_saving = ((od['energy'] - oda['energy']) / od['energy']) * 100 if od['energy'] > 0 else 0
    
    print("\n  Performance Analysis:")
    print(f"    > ODA-MD Detection Accuracy: {da_improvement:+.2f}% better than OD")
    print(f"    > ODA-MD Energy Saving: {energy_saving:.1f}% compared to OD")
    print(f"    > ODA-MD correctly detected {oda['TP']}/{oda['TP']+oda['FN']} true outliers")
    print(f"    > OD only detected {od['TP']}/{od['TP']+od['FN']} true outliers")
    
    # 5. Plots
    print("\n" + "-" * 70)
    print("| [5] SAVING VISUALIZATION                                           |")
    print("-" * 70)
    
    plot_network_topology(topology)
    plot_comparison(oda, od)
    plot_energy(oda['energy'], od['energy'])
    plot_temperature(real_data, cluster_data, ground_truth, np.zeros(len(ground_truth)))
    
    print("\n  All plots saved successfully!")
    print("=" * 70)
    
    return oda, od


if __name__ == '__main__':
    main()
