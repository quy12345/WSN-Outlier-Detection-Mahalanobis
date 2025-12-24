"""
Network Topology Module.
Creates and manages the full WSN topology with 81 nodes and 10 clusters.
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from .config import (N_NODES, N_CLUSTERS, AREA_WIDTH, AREA_HEIGHT, 
                     CLUSTER_SIZE, RADIO_RANGE, SINK_ID, CH_ID, CLUSTER_2_NODES)


@dataclass
class NodeInfo:
    """Information about a single node in the network."""
    node_id: int
    x: float
    y: float
    cluster_id: int = -1
    is_ch: bool = False
    is_sink: bool = False


@dataclass 
class ClusterInfo:
    """Information about a cluster."""
    cluster_id: int
    ch_id: int
    member_ids: List[int] = field(default_factory=list)
    center_x: float = 0.0
    center_y: float = 0.0


class NetworkTopology:
    """
    Manages the full WSN network topology.
    
    Paper specifications:
    - 81 nodes in 100x100m area
    - 10 clusters (8 members + 1 CH each, roughly)
    - Node ID=1 is Sink
    - Node ID=2 is CH of Cluster 2
    - Nodes {36, 37, 38} are members of Cluster 2 (real data)
    """
    
    def __init__(self, n_nodes: int = N_NODES, n_clusters: int = N_CLUSTERS,
                 area_width: float = AREA_WIDTH, area_height: float = AREA_HEIGHT,
                 seed: int = 42):
        self.n_nodes = n_nodes
        self.n_clusters = n_clusters
        self.area_width = area_width
        self.area_height = area_height
        
        np.random.seed(seed)
        
        # Node storage
        self.nodes: Dict[int, NodeInfo] = {}
        self.clusters: Dict[int, ClusterInfo] = {}
        
        # Build topology
        self._create_nodes()
        self._form_clusters()
        self._assign_special_nodes()
        
    def _create_nodes(self):
        """Create nodes with random positions in the area."""
        # Node ID 1 is Sink (placed at center-ish)
        self.nodes[SINK_ID] = NodeInfo(
            node_id=SINK_ID,
            x=self.area_width / 2,
            y=self.area_height / 2,
            is_sink=True
        )
        
        # Create remaining nodes with random positions
        for node_id in range(2, self.n_nodes + 1):
            self.nodes[node_id] = NodeInfo(
                node_id=node_id,
                x=np.random.uniform(0, self.area_width),
                y=np.random.uniform(0, self.area_height)
            )
    
    def _form_clusters(self):
        """
        Form clusters using K-means-like balanced clustering.
        Paper specifies: 10 clusters, ~8 nodes each.
        """
        # Get non-sink node positions
        node_ids = [nid for nid in self.nodes.keys() if not self.nodes[nid].is_sink]
        positions = np.array([[self.nodes[nid].x, self.nodes[nid].y] for nid in node_ids])
        
        # Initialize cluster centers uniformly in grid
        # For 10 clusters in 100x100m, use 2x5 grid centers
        centers = []
        for row in range(2):
            for col in range(5):
                cx = (col + 0.5) * (self.area_width / 5)
                cy = (row + 0.5) * (self.area_height / 2)
                centers.append([cx, cy])
        centers = np.array(centers)
        
        # K-means iterations
        for _ in range(10):
            # Assign each node to nearest center
            assignments = []
            for pos in positions:
                dists = [np.linalg.norm(pos - c) for c in centers]
                assignments.append(np.argmin(dists) + 1)  # 1-indexed cluster
            
            # Update centers
            for c_idx in range(self.n_clusters):
                cluster_id = c_idx + 1
                member_positions = [positions[i] for i, a in enumerate(assignments) if a == cluster_id]
                if member_positions:
                    centers[c_idx] = np.mean(member_positions, axis=0)
        
        # Initialize clusters
        for cluster_id in range(1, self.n_clusters + 1):
            self.clusters[cluster_id] = ClusterInfo(
                cluster_id=cluster_id,
                ch_id=-1,
                member_ids=[]
            )
        
        # Assign nodes based on final K-means assignments
        for i, node_id in enumerate(node_ids):
            cluster_id = assignments[i]
            self.nodes[node_id].cluster_id = cluster_id
            self.clusters[cluster_id].member_ids.append(node_id)
        
        # Balance clusters: redistribute if any cluster is too large/small
        target_size = len(node_ids) // self.n_clusters  # 80/10 = 8
        
        # Sort clusters by size (largest first)
        sorted_clusters = sorted(self.clusters.items(), key=lambda x: len(x[1].member_ids), reverse=True)
        
        for cluster_id, cluster in sorted_clusters:
            while len(cluster.member_ids) > target_size + 1:
                # Find cluster with fewest members
                smallest = min(self.clusters.items(), key=lambda x: len(x[1].member_ids))
                if len(smallest[1].member_ids) >= target_size:
                    break
                
                # Move last member to smallest cluster
                node_to_move = cluster.member_ids.pop()
                self.nodes[node_to_move].cluster_id = smallest[0]
                smallest[1].member_ids.append(node_to_move)
        
        # Assign CH for each cluster (lowest ID in cluster)
        for cluster_id, cluster in self.clusters.items():
            if cluster.member_ids:
                cluster.member_ids.sort()
                ch_id = cluster.member_ids[0]
                cluster.ch_id = ch_id
                self.nodes[ch_id].is_ch = True
                
                # Calculate cluster center
                xs = [self.nodes[nid].x for nid in cluster.member_ids]
                ys = [self.nodes[nid].y for nid in cluster.member_ids]
                cluster.center_x = np.mean(xs)
                cluster.center_y = np.mean(ys)
    
    def _assign_special_nodes(self):
        """
        Ensure special nodes are correctly assigned:
        - Cluster 2 contains EXACTLY: CH=2 and members {36, 37, 38} (Intel Lab nodes)
        - Other nodes stay in their K-means assigned clusters
        """
        # First, ensure Cluster 2 exists  
        if 2 not in self.clusters:
            self.clusters[2] = ClusterInfo(cluster_id=2, ch_id=-1, member_ids=[])
        
        # Target members for Cluster 2 (as per paper)
        cluster_2_target = [CH_ID] + list(CLUSTER_2_NODES)  # [2, 36, 37, 38]
        
        # 1. Remove ALL current members from Cluster 2 and redistribute to other clusters
        current_members = self.clusters[2].member_ids.copy()
        for member_id in current_members:
            if member_id not in cluster_2_target:
                # Find nearest cluster (except Cluster 2)
                node = self.nodes[member_id]
                min_dist = float('inf')
                nearest_cluster = 1
                
                for cid, cluster in self.clusters.items():
                    if cid != 2 and cluster.member_ids:
                        dist = np.sqrt((node.x - cluster.center_x)**2 + (node.y - cluster.center_y)**2)
                        if dist < min_dist:
                            min_dist = dist
                            nearest_cluster = cid
                
                # Move to nearest cluster
                self.clusters[2].member_ids.remove(member_id)
                node.cluster_id = nearest_cluster
                self.clusters[nearest_cluster].member_ids.append(member_id)
        
        # 2. Move target nodes TO Cluster 2
        for node_id in cluster_2_target:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                old_cluster = node.cluster_id
                
                # Remove from old cluster if different
                if old_cluster != 2 and old_cluster in self.clusters:
                    if node_id in self.clusters[old_cluster].member_ids:
                        self.clusters[old_cluster].member_ids.remove(node_id)
                        # Reassign CH if needed
                        if self.clusters[old_cluster].ch_id == node_id:
                            self.nodes[node_id].is_ch = False
                            if self.clusters[old_cluster].member_ids:
                                new_ch = min(self.clusters[old_cluster].member_ids)
                                self.clusters[old_cluster].ch_id = new_ch
                                self.nodes[new_ch].is_ch = True
                
                # Add to Cluster 2
                node.cluster_id = 2
                if node_id not in self.clusters[2].member_ids:
                    self.clusters[2].member_ids.append(node_id)
        
        # 3. Set Node 2 as CH of Cluster 2
        if CH_ID in self.nodes:
            # Demote any existing CH
            old_ch = self.clusters[2].ch_id
            if old_ch != -1 and old_ch != CH_ID and old_ch in self.nodes:
                self.nodes[old_ch].is_ch = False
            
            self.nodes[CH_ID].is_ch = True
            self.clusters[2].ch_id = CH_ID
        
        # 4. Sort all cluster member lists
        for cluster in self.clusters.values():
            cluster.member_ids.sort()
        
        # 5. Update Cluster 2 center
        if self.clusters[2].member_ids:
            xs = [self.nodes[nid].x for nid in self.clusters[2].member_ids if nid in self.nodes]
            ys = [self.nodes[nid].y for nid in self.clusters[2].member_ids if nid in self.nodes]
            if xs and ys:
                self.clusters[2].center_x = np.mean(xs)
                self.clusters[2].center_y = np.mean(ys)
        
        # 6. Verify
        print(f"\n[VERIFIED] Cluster 2 setup:")
        print(f"  CH: Node {self.clusters[2].ch_id}")
        print(f"  Intel Lab Nodes: {[n for n in CLUSTER_2_NODES if n in self.clusters[2].member_ids]}")
        print(f"  All Members: {self.clusters[2].member_ids}")
    
    def get_cluster_members(self, cluster_id: int) -> List[int]:
        """Get member node IDs for a cluster."""
        if cluster_id in self.clusters:
            return self.clusters[cluster_id].member_ids
        return []
    
    def get_cluster_ch(self, cluster_id: int) -> int:
        """Get CH node ID for a cluster."""
        if cluster_id in self.clusters:
            return self.clusters[cluster_id].ch_id
        return -1
    
    def get_distance(self, node1_id: int, node2_id: int) -> float:
        """Calculate Euclidean distance between two nodes."""
        if node1_id not in self.nodes or node2_id not in self.nodes:
            return float('inf')
        
        n1 = self.nodes[node1_id]
        n2 = self.nodes[node2_id]
        return np.sqrt((n1.x - n2.x)**2 + (n1.y - n2.y)**2)
    
    def get_distance_to_sink(self, node_id: int) -> float:
        """Get distance from a node to the Sink."""
        return self.get_distance(node_id, SINK_ID)
    
    def print_summary(self):
        """Print network topology summary."""
        print("=" * 60)
        print("NETWORK TOPOLOGY SUMMARY")
        print("=" * 60)
        print(f"Total Nodes: {self.n_nodes}")
        print(f"Total Clusters: {self.n_clusters}")
        print(f"Area: {self.area_width}x{self.area_height}m")
        print(f"Sink Node: {SINK_ID} at ({self.nodes[SINK_ID].x:.1f}, {self.nodes[SINK_ID].y:.1f})")
        print()
        
        for cluster_id, cluster in self.clusters.items():
            ch_node = self.nodes.get(cluster.ch_id)
            ch_pos = f"({ch_node.x:.1f}, {ch_node.y:.1f})" if ch_node else "N/A"
            print(f"Cluster {cluster_id}: CH={cluster.ch_id} {ch_pos}, "
                  f"Members={len(cluster.member_ids)} {cluster.member_ids[:5]}...")
        
        print("=" * 60)
