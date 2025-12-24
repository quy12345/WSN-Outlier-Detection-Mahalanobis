"""
Configuration and Constants for WSN Simulation.
"""

# =============================================================================
# CONSTANTS (from Table I in paper)
# =============================================================================

PACKET_SIZE = 32  # bytes
INITIAL_ENERGY = 2.0  # Joules

# Heinzelman Energy Model
E_ELEC = 50e-9  # 50 nJ/bit
E_AMP = 100e-12  # 100 pJ/bit/m^2
E_DA = 5e-9  # 5 nJ/bit

# Cluster 2 nodes as per paper
# Paper Setup: 81 Nodes in 100x100m, 10 Clusters.
# Sink ID = 1.
# Cluster 2: CH ID = 2.
# Members representing real data: {36, 37, 38}.
CLUSTER_2_NODES = [36, 37, 38]
SINK_ID = 1
CH_ID = 2
CLUSTER_SIZE = 8 # Total nodes in a fully populated cluster
RADIO_RANGE = 40 # meters
	
# Topology Constants
N_NODES = 81
N_CLUSTERS = 10
AREA_WIDTH = 100
AREA_HEIGHT = 100
