"""
Main execution script for Full WSN Simulation.
Simulates 81 nodes, 10 clusters as per paper specifications.
"""

import numpy as np
import warnings
from modules.config import (CLUSTER_2_NODES, N_NODES, N_CLUSTERS, 
                            AREA_WIDTH, AREA_HEIGHT, CH_ID, SINK_ID)
from modules.topology import NetworkTopology
from modules.data_loader import load_intel_data, generate_network_data
from modules.simulation import run_full_network_simulation
from modules.visualization import (plot_comparison, plot_energy, 
                                   plot_temperature, plot_network_topology)

warnings.filterwarnings('ignore')
np.random.seed(42)


def main():
    """Main function - Full Network Simulation."""
    print("\n" + "=" * 60)
    print("WSN FULL NETWORK SIMULATION")
    print(f"Topology: {N_NODES} Nodes, {N_CLUSTERS} Clusters")
    print(f"Area: {AREA_WIDTH}x{AREA_HEIGHT}m")
    print(f"Sink: Node {SINK_ID}, Cluster 2 CH: Node {CH_ID}")
    print("=" * 60)
    
    # Step 1: Create Network Topology
    print("\n--- Step 1: Creating Network Topology ---")
    topology = NetworkTopology()
    topology.print_summary()
    
    # Step 2: Load Real Intel Lab Data (for Cluster 2)
    print("\n--- Step 2: Loading Real Data ---")
    real_data = load_intel_data('data.txt', CLUSTER_2_NODES)
    
    # Step 3: Generate Network-wide Data
    print("\n--- Step 3: Generating Network Data ---")
    data_dict, ground_truth_dict, outlier_schedule = generate_network_data(
        topology, real_data, n_outliers=1000
    )
    
    # Step 4: Plot Network Topology
    print("\n--- Step 4: Plotting Network Topology ---")
    plot_network_topology(topology)
    
    # Step 5: Run ODA-MD Simulation
    print("\n--- Step 5: Running ODA-MD Algorithm ---")
    oda_results = run_full_network_simulation(
        topology, data_dict, ground_truth_dict, algorithm='ODA-MD'
    )
    
    # Step 6: Run OD Simulation
    print("\n--- Step 6: Running OD Algorithm (Baseline) ---")
    od_results = run_full_network_simulation(
        topology, data_dict, ground_truth_dict, algorithm='OD'
    )
    
    # Step 7: Generate Comparison Plots
    print("\n--- Step 7: Generating Plots ---")
    plot_comparison(oda_results, od_results)
    plot_energy(oda_results['energy'], od_results['energy'])
    
    # Plot Cluster 2 temperature (real data)
    if 2 in data_dict:
        plot_temperature(
            real_data, 
            data_dict[2], 
            ground_truth_dict[2],
            np.zeros(len(ground_truth_dict[2]))
        )
    
    # Final Summary
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETED")
    print("=" * 60)
    print("\nResults Summary:")
    print(f"  ODA-MD: DA={oda_results['DA']*100:.2f}%, FAR={oda_results['FAR']*100:.2f}%, "
          f"Energy={oda_results['energy']:.4f}J")
    print(f"  OD:     DA={od_results['DA']*100:.2f}%, FAR={od_results['FAR']*100:.2f}%, "
          f"Energy={od_results['energy']:.4f}J")
    
    print("\nFiles generated:")
    print("  - network_topology.png")
    print("  - wsn_comparison_results.png")
    print("  - wsn_energy_comparison.png")
    print("  - temperature_with_outliers.png")
    
    return oda_results, od_results


if __name__ == '__main__':
    main()
