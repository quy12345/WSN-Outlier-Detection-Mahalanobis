# ODA-MD: Outlier Detection Algorithm using Mahalanobis Distance for Wireless Sensor Networks

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A Python implementation of the **ODA-MD (Outlier Detection Algorithm based on Mahalanobis Distance)** for Wireless Sensor Networks, as described in the research paper. This simulation demonstrates the superiority of ODA-MD over traditional outlier detection methods in terms of detection accuracy and energy efficiency.

## Performance Results

| Algorithm | Detection Accuracy | False Alarm Rate | Energy Consumption |
|-----------|-------------------|-----------------|---------------------|
| **ODA-MD** | **100.00%** | 0.86% | **29.61 J** |
| OD (Baseline) | 35.30% | 0.23% | 99.34 J |

*Results from simulation with 81 nodes, 10 clusters, 1000 injected outliers using Intel Lab dataset.*

## Key Features

- **Mahalanobis Distance-based Detection**: Exploits spatial correlation between sensor data for robust outlier detection
- **Distributed Processing**: Cluster Heads (CHs) perform local detection, reducing communication overhead
- **Energy Efficient**: Filters outliers at CH level, transmitting only normal data to sink (~70% energy savings)
- **Real Dataset Integration**: Uses actual Intel Lab sensor data for realistic evaluation
- **Full Network Simulation**: 81 nodes, 10 clusters, 100×100m deployment area

## Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                         SINK (Node 1)                        │
└─────────────────────────────────────────────────────────────┘
              ▲               ▲               ▲
              │               │               │
        ┌─────┴─────┐   ┌─────┴─────┐   ┌─────┴─────┐
        │   CH_1    │   │   CH_2    │   │   CH_n    │
        │ (Node 4)  │   │ (Node 2)  │   │ (Node 19) │
        └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
              │               │               │
     ┌────┬───┴───┬────┐  ┌───┴───┐     ┌────┼────┐
     │    │       │    │  │       │     │    │    │
    [N30][N31]  [N51][N66][N36][N37][N38][N27][N28][N58]
                       (Intel Lab Data)
```

## Project Structure

```text
mahalanobis/
├── main.py                 # Entry point for simulation
├── data.txt                # Intel Lab sensor dataset (Needs manual download)
├── modules/
│   ├── __init__.py
│   ├── config.py           # Configuration constants
│   ├── topology.py         # Network topology (K-means clustering)
│   ├── data_loader.py      # Intel Lab data loading & synthetic generation
│   ├── network.py          # SensorNode, ClusterHead, OD_Detector classes
│   ├── simulation.py       # Simulation engine
│   └── visualization.py    # Plotting functions
└── outputs/
    ├── network_topology.png
    ├── wsn_comparison_results.png
    ├── wsn_energy_comparison.png
    └── temperature_with_outliers.png
```

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/oda-md-wsn.git
cd oda-md-wsn

# Install dependencies
pip install numpy pandas matplotlib scipy
```

##  Dataset Setup

The Intel Lab dataset (`data.txt`) is not included in this repository due to its large size (~150MB). You must download it manually to run the simulation:

1. **Download**: Get the dataset from the official MIT link or Kaggle:
   - [Official MIT Link (data.txt.gz)](http://db.csail.mit.edu/labdata/data.txt.gz)
   - [Kaggle Dataset (Intel Berkeley Research Lab)](https://www.kaggle.com/datasets/prakharrathi25/intel-berkeley-research-lab-sensor-data)
2. **Extract**: If you downloaded the `.gz` file, extract it to get `data.txt`.
3. **Placement**: Place the `data.txt` file directly in the root directory of the project:
   ```text
   mahalanobis/
   ├── main.py
   ├── data.txt  <-- Place it here
   └── modules/
   ```

## Usage

### Run Full Simulation

```bash
python main.py
```

### Output Files

- `network_topology.png` - Visual representation of the 81-node network
- `wsn_comparison_results.png` - DA and FAR comparison over time (Figures 4 & 5)
- `wsn_energy_comparison.png` - Energy consumption comparison (Figure 6)
- `temperature_with_outliers.png` - Temperature data with injected outliers

## Algorithm Overview

### ODA-MD (Proposed)

1. **Data Collection**: CH sends request to all sensor nodes in cluster
2. **Matrix Construction**: Build matrix X from received sensor vectors
3. **Statistics Calculation**: Compute mean (μ) and covariance (Σ)
4. **Mahalanobis Distance**: Calculate MD for each observation:
   ```text
   MD_i = √[(X_i - μ)ᵀ Σ⁻¹ (X_i - μ)]
   ```
5. **Anomaly Decision**: Flag as outlier if MD² > χ²(p, 0.975)
6. **Selective Forwarding**: Only transmit normal data to sink

### OD (Baseline - Fixed-width Clustering)

1. **Online Clustering**: Assign observations to clusters within fixed width
2. **kNN-based Detection**: Flag points in sparse clusters as outliers
3. **Forward All Data**: Transmit all data regardless of outlier status

## Simulation Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| N_NODES | 81 | Total sensor nodes |
| N_CLUSTERS | 10 | Number of clusters |
| AREA_SIZE | 100×100m | Deployment area |
| BATCH_SIZE | 50 | Observations per batch |
| RADIO_RANGE | 30m | Communication range |
| OUTLIERS | 1000 | Injected anomalies |

## Dataset

This project uses the **Intel Lab Dataset** [1]:

- 54 sensors deployed in Intel Berkeley Research Lab
- Measurements: temperature, humidity, light, voltage
- **Source**: [Download data.txt](http://db.csail.mit.edu/labdata/data.txt.gz)
- Selected nodes: {36, 37, 38} for Cluster 2
- Time period: March 11-14, 2004 (~15,763 records)

## References

1. **Intel Lab Data**: Madden, S. (2004). Intel Lab Data. Retrieved from [http://db.csail.mit.edu/labdata/labdata.html](http://db.csail.mit.edu/labdata/labdata.html)

2. **Mahalanobis Distance**: Mahalanobis, P. C. (1936). On the generalized distance in statistics. *Proceedings of the National Institute of Sciences of India*, 2(1), 49-55.

*Crossbow Technology*.

## 🛠️ Energy Model

Based on the first-order radio model:

```text
E_tx = E_elec × k + E_amp × k × d²
E_rx = E_elec × k
E_da = 5 nJ/bit (data aggregation)
```

Where:

- E_elec = 50 nJ/bit (electronics energy)
- E_amp = 100 pJ/bit/m² (amplifier energy)
- k = packet size in bits
- d = transmission distance
