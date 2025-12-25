# Giải Thích Chi Tiết Dự Án ODA-MD
## Outlier Detection Algorithm using Mahalanobis Distance for Wireless Sensor Networks

---

## 1. Tổng Quan

### 1.1 Mục Đích Dự Án

Dự án này triển khai và so sánh hai thuật toán phát hiện outlier trong Mạng Cảm Biến Không Dây (WSN):

1. **ODA-MD (Proposed)**: Thuật toán đề xuất sử dụng Mahalanobis Distance
2. **OD (Baseline)**: Thuật toán cơ sở sử dụng Fixed-width Clustering với Euclidean Distance

### 1.2 Kết Quả Chính

| Thuật toán | Detection Accuracy | False Alarm Rate | Energy |
|------------|-------------------|------------------|--------|
| **ODA-MD** | **99.60%** | 2.57% | **13.64 J** |
| OD | 22.70% | 0.97% | 15.21 J |

**ODA-MD vượt trội hơn OD:**
- Detection Accuracy cao hơn **+76.90%**
- Tiết kiệm năng lượng **10.3%**
- Phát hiện đúng **996/1000** outliers (so với 227/1000 của OD)

---

## 2. Kiến Trúc Mạng WSN

### 2.1 Cấu Trúc Phân Cấp

```text
                        ┌─────────────────┐
                        │   SINK (Node 1) │
                        │   Trung tâm     │
                        └────────┬────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
        ▼                        ▼                        ▼
   ┌─────────┐              ┌─────────┐              ┌─────────┐
   │  CH_1   │              │  CH_2   │              │  CH_10  │
   │ Node 4  │              │ Node 2  │              │ Node 19 │
   └────┬────┘              └────┬────┘              └────┬────┘
        │                        │                        │
   ┌────┴────┐              ┌────┴────┐              ┌────┴────┐
   │ Members │              │ Members │              │ Members │
   │ ~8 nodes│              │ 36,37,38│              │ ~8 nodes│
   └─────────┘              └─────────┘              └─────────┘
                            (Intel Lab)
```

### 2.2 Thông Số Mạng

| Thông số | Giá trị | Mô tả |
|----------|---------|-------|
| Tổng nodes | 81 | Bao gồm 1 Sink + 80 sensor nodes |
| Số clusters | 10 | Được tạo bằng K-means |
| Diện tích | 100×100m | Vùng triển khai |
| Radio range | 40m | Phạm vi truyền thông |
| Sink position | (50, 50) | Trung tâm mạng |

### 2.3 Cluster 2 - Focus của Mô Phỏng

```text
Cluster 2:
├── CH: Node 2 (Cluster Head)
└── Members: [36, 37, 38] (Dữ liệu thực từ Intel Lab)

> Toàn bộ mạng 81 nodes được tạo, nhưng thuật toán 
> chỉ chạy trên Cluster 2 với dữ liệu thực Intel Lab
```

---

## 3. Luồng Xử Lý Algorithm

### 3.1 ODA-MD: Time-Step Processing

```text
╔════════════════════════════════════════════════════════════════════╗
║                    ODA-MD - Mỗi Time Step t                        ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  Bước 1: CH gửi REQUEST broadcast                                  ║
║          ─────────────────────────────────►                        ║
║                                            │                       ║
║  Bước 2: Mỗi node trả về vector A_k        │                       ║
║          ◄─────────────────────────────────┤                       ║
║          A_k = [temp, humid, light, volt]  │                       ║
║                                            │                       ║
║  Bước 3: CH xây dựng Table_MD              ▼                       ║
║          ┌────────┬────────┬────────┐                              ║
║          │ Node36 │ Node37 │ Node38 │                              ║
║          ├────────┼────────┼────────┤                              ║
║          │  22.79 │  23.57 │  24.13 │ ← temperature                ║
║          │  40.84 │  35.09 │  35.85 │ ← humidity                   ║
║          │ 1847.4 │ 1435.2 │ 1293.9 │ ← light                      ║
║          │ 2.5273 │ 2.5381 │ 2.5273 │ ← voltage                    ║
║          └────────┴────────┴────────┘                              ║
║                                                                    ║
║  Bước 4: Tính MD cho mỗi node                                      ║
║          MD = √[(X - μ)ᵀ × Σ⁻¹ × (X - μ)]                          ║
║                                                                    ║
║  Bước 5: So sánh với threshold                                     ║
║          threshold = √χ²(df=4, α=0.975) ≈ 3.3382                   ║
║                                                                    ║
║  Bước 6: QUYẾT ĐỊNH                                                ║
║          ┌─────────────────────────────────────┐                   ║
║          │ Nếu MD ≥ threshold → OUTLIER        │                   ║
║          │    → KHÔNG forward đến Sink         │                   ║
║          │    → Tiết kiệm năng lượng           │                   ║
║          │                                     │                   ║
║          │ Nếu MD < threshold → NORMAL         │                   ║
║          │    → Forward đến Sink               │                   ║
║          └─────────────────────────────────────┘                   ║
╚════════════════════════════════════════════════════════════════════╝
```

### 3.2 OD (Baseline): Fixed-width Clustering

```text
╔════════════════════════════════════════════════════════════════════╗
║                    OD - Fixed-width Clustering                     ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  Bước 1: Nhận observation (3 nodes × 4 attrs = 12 features)        ║
║          obs = [22.7, 40.8, 1847, 2.52, 23.5, 35.0, ...]           ║
║                                                                    ║
║  Bước 2: Tìm cluster gần nhất (Euclidean distance)                 ║
║          d = ||obs - cluster_center||                              ║
║                                                                    ║
║  Bước 3: Gán vào cluster                                           ║
║          ┌────────────────────────────────────────┐                ║
║          │ Nếu d < width (5.0) → cập nhật center  │                ║
║          │ Nếu d ≥ width → tạo cluster mới        │                ║
║          └────────────────────────────────────────┘                ║
║                                                                    ║
║  Bước 4: Tính k-NN score                                           ║
║          score = mean(distances to k nearest clusters)             ║
║                                                                    ║
║  Bước 5: So sánh với adaptive threshold                            ║
║          threshold = mean(all_scores) + std(all_scores)            ║
║                                                                    ║
║  Bước 6: QUYẾT ĐỊNH                                                ║
║          ┌─────────────────────────────────────┐                   ║
║          │ Nếu score > threshold → OUTLIER     │                   ║
║          │                                     │                   ║
║          │    OD vẫn forward TẤT CẢ data       │                   ║
║          │    → Không tiết kiệm năng lượng     │                   ║
║          └─────────────────────────────────────┘                   ║
╚════════════════════════════════════════════════════════════════════╝
```

---

## 4. Chi Tiết Các Module

### 4.1 config.py - Cấu Hình Hệ Thống

```python
# 4 Thuộc tính cảm biến (theo paper)
SENSOR_ATTRIBUTES = ['temperature', 'humidity', 'light', 'voltage']
N_ATTRIBUTES = 4

# Topology
N_NODES = 81          # Tổng số nodes
N_CLUSTERS = 10       # Số clusters
AREA_WIDTH = 100      # 100m
AREA_HEIGHT = 100     # 100m

# Cluster 2 đặc biệt (dữ liệu Intel Lab)
CLUSTER_2_NODES = [36, 37, 38]
SINK_ID = 1
CH_ID = 2

# Energy Model (Heinzelman)
E_ELEC = 50e-9       # 50 nJ/bit
E_AMP = 100e-12      # 100 pJ/bit/m²
E_DA = 5e-9          # 5 nJ/bit
PACKET_SIZE = 32     # bytes
INITIAL_ENERGY = 2.0 # Joules
```

### 4.2 topology.py - Tạo Mạng 81 Nodes

```python
class NetworkTopology:
    """
    Tạo topology WSN với:
    - 81 nodes trong vùng 100×100m
    - 10 clusters sử dụng K-means
    - Cluster 2 đặc biệt chứa nodes Intel Lab
    """
    
    def __init__(self):
        self._create_nodes()      # Tạo 81 nodes với vị trí random
        self._form_clusters()     # K-means clustering → 10 clusters
        self._assign_special_nodes()  # Đảm bảo Cluster 2 có nodes 36,37,38
```

**Output khi chạy:**
```text
----------------------------------------------------------------------
| [1] NETWORK TOPOLOGY INITIALIZATION                                |
----------------------------------------------------------------------
  > Created 81 sensor nodes in 100x100m area
  > Formed 10 clusters using K-means clustering
  > Sink node: Node 1 at center (50, 50)m

  Cluster Distribution:
    * Cluster 2: CH=Node_2, 4 members [2, 36, 37, 38] [FOCUS - Intel Lab Data]
      Cluster 1: CH=Node_4, 8 members [4, 30, 31, ...]...
      Cluster 3: CH=Node_5, 8 members [5, 33, 34, ...]...
      ...
```

### 4.3 data_loader.py - Tải Dữ Liệu Intel Lab

```python
def load_intel_data(filepath, node_ids, date_start, date_end):
    """
    Tải dữ liệu từ Intel Lab dataset.
    
    Xử lý:
    1. Đọc file → 2,313,682 records
    2. Lọc nodes [36, 37, 38] → 153,427 records
    3. Lọc ngày 2004-03-11 đến 2004-03-15 → 18,471 records
    4. Lọc giá trị hợp lệ → 10,167 epochs
    
    Output: DataFrame với cấu trúc:
    - 10,167 hàng (epochs)
    - 12 cột: node_36_temperature, node_36_humidity, node_36_light, 
              node_36_voltage, node_37_..., node_38_...
    """
```

**Output khi tải dữ liệu:**
```text
============================================================
LOADING INTEL LAB DATA (4 ATTRIBUTES)
============================================================
Total records: 2,313,682
Records for nodes [36, 37, 38]: 153,427
Records after date filter (2004-03-11 to 2004-03-15): 18,471
Epochs with valid data: 10,167

Data statistics per attribute:
  temperature: mean=23.27, std=3.58
  humidity: mean=40.74, std=5.76
  light: mean=616.30, std=626.44
  voltage: mean=2.53, std=0.04
```

### 4.4 network.py - Các Class Thuật Toán

#### 4.4.1 SensorNode

```python
class SensorNode:
    """Đại diện cho một node cảm biến."""
    
    def __init__(self, node_id, data_matrix):
        self.node_id = node_id
        self.data = data_matrix  # Shape: (n_epochs, 4)
        self.energy = INITIAL_ENERGY  # 2.0 Joules
    
    def send_observation(self, epoch_idx):
        """Gửi vector A_k = [temp, humid, light, volt] cho epoch hiện tại."""
        observation = self.data[epoch_idx]  # 4 giá trị
        
        # Tiêu thụ năng lượng truyền
        tx_energy = (E_ELEC + E_AMP * dist²) * bits
        self.consume_energy(tx_energy)
        
        return observation
```

#### 4.4.2 ClusterHead (ODA-MD)

```python
class ClusterHead:
    """
    Cluster Head chạy thuật toán ODA-MD.
    
    Tham số quan trọng:
    - verbose: Bật/tắt in chi tiết query
    - _max_sample_logs: Số query NORMAL in ra (mặc định 3)
    - _max_outlier_logs: Số query OUTLIER in ra (mặc định 2)
    """
    
    def process_time_step(self, epoch_idx):
        """
        Xử lý 1 time step theo Figure 2 của paper.
        
        Returns:
            (is_outlier, details)
        """
        # 1. Gửi request broadcast
        # 2. Thu thập A_k từ tất cả members
        # 3. Xây dựng Table_MD (n_nodes × 4)
        # 4. Thêm vào history (rolling window)
        # 5. Tính MD cho mỗi node
        # 6. So sánh với threshold chi2
        # 7. Quyết định OUTLIER/NORMAL
```

**Output mẫu ODA-MD:**
```text
-----------------------------------------------------------------
| [!] CH QUERY #7 - Cluster 2, t=6 [OUTLIER]
-----------------------------------------------------------------
| Table_MD (3 nodes x 4 attributes):
| Node         Temp      Humid      Light       Volt
|-------------------------------------------------------
| N_36       22.79      40.84     1847.4     2.5273 -> MD=3.781
| N_37       23.57      35.09     1435.2     2.5381 -> MD=0.378
| N_38       24.13      35.85     1293.9     2.5273 -> MD=1.828
| Threshold (chi2, df=4): 3.3382
| >>> DECISION: OUTLIER - Node(s) [36]
-----------------------------------------------------------------
```

#### 4.4.3 OD_Detector (Baseline)

```python
class OD_Detector:
    """
    Thuật toán OD baseline sử dụng Fixed-width Clustering.
    
    Tham số:
    - width: Bán kính cluster (mặc định 5.0)
    - k: Số láng giềng gần nhất (mặc định 5)
    - max_clusters: Số cluster tối đa (mặc định 100)
    - verbose: Bật/tắt in chi tiết
    """
    
    def detect(self, observations):
        """
        Xử lý observations của 1 time step.
        
        Luồng:
        1. Flatten observations (3×4=12 features)
        2. Tìm cluster gần nhất
        3. Gán hoặc tạo cluster mới
        4. Tính k-NN score
        5. So sánh với threshold (mean + std)
        """
```

**Output mẫu OD:**
```text
-----------------------------------------------------------------
| [!] OD QUERY #6 [OUTLIER DETECTED]
-----------------------------------------------------------------
| Current clusters: 3 (max: 100)
| Assigned to cluster: #2
| Distance to nearest cluster: 11.7789 (width: 5.0)
| k-NN score: 17.6666
| Threshold (mean+std): 11.7771
| >>> DECISION: OUTLIER (score 17.6666 > threshold 11.7771)
-----------------------------------------------------------------
```

---

## 5. Công Thức Toán Học

### 5.1 Mahalanobis Distance (Phương trình 2)

$$MD_i = \sqrt{(X_i - \mu)^T \cdot \Sigma^{-1} \cdot (X_i - \mu)}$$

Trong đó:
- **X_i**: Vector quan sát 4 chiều `[temp, humid, light, volt]`
- **μ (mu)**: Vector trung bình 4 chiều (tính từ history)
- **Σ (Sigma)**: Ma trận hiệp phương sai 4×4

### 5.2 Ví Dụ Tính Toán

```text
Giả sử tại time step t=100:

Node 36:
  X = [22.5, 40.0, 1800, 2.52]     # Observation hiện tại
  μ = [23.0, 41.0, 1700, 2.53]     # Mean từ 100 time steps trước
  
  diff = X - μ = [-0.5, -1.0, 100, -0.01]
  
  Σ = [[3.5, 0.2, 10,  0.01],      # Ma trận covariance 4×4
       [0.2, 5.0, 5,   0.02],
       [10,  5,   5000, 0.5],
       [0.01, 0.02, 0.5, 0.002]]
  
  MD = √(diff × Σ⁻¹ × diffᵀ) = 2.45  # < 3.34 → NORMAL
```

### 5.3 Chi-Square Threshold (Phương trình 5)

$$threshold = \sqrt{\chi^2_{df=4, \alpha=0.975}} = \sqrt{11.143} \approx 3.3382$$

| Điều kiện | Kết luận |
|-----------|----------|
| MD < 3.3382 | NORMAL - Forward đến Sink |
| MD ≥ 3.3382 | OUTLIER - Không forward |

---

## 6. Năng Lượng Tiêu Thụ

### 6.1 Mô Hình Heinzelman

```text
E_tx = E_elec × k + E_amp × k × d²    (Truyền)
E_rx = E_elec × k                      (Nhận)
E_da = E_da × k                        (Xử lý)

Với:
  E_elec = 50 nJ/bit
  E_amp = 100 pJ/bit/m²
  E_da = 5 nJ/bit
  k = bits truyền
  d = khoảng cách (m)
```

### 6.2 So Sánh Năng Lượng

| Thành phần | ODA-MD | OD |
|------------|--------|-----|
| Request broadcast | ✓ | ✓ |
| Thu thập data | ✓ | ✓ |
| Xử lý/tính toán | ✓ | ✓ |
| Forward đến Sink | **Chỉ NORMAL** | **TẤT CẢ** |

**→ ODA-MD tiết kiệm năng lượng vì không forward outliers!**

---

## 7. Phân Tích Kết Quả

### 7.1 Tại Sao ODA-MD Tốt Hơn?

| Tiêu chí | ODA-MD | OD | Giải thích |
|----------|--------|-----|------------|
| **DA** | 99.60% | 22.70% | MD xét correlation giữa 4 attributes |
| **FAR** | 2.57% | 0.97% | ODA-MD nhạy hơn → phát hiện nhiều hơn |
| **Energy** | 13.64J | 15.21J | ODA-MD lọc outlier → ít truyền hơn |

### 7.2 Ưu Điểm của Mahalanobis Distance

1. **Scale-invariant**: Không bị ảnh hưởng bởi đơn vị
   - Temperature: °C (nhỏ ~20-30)
   - Light: Lux (lớn ~0-2000)
   
2. **Xét Correlation**: Phát hiện bất thường trong mối quan hệ giữa các attributes
   - Ví dụ: Temp tăng nhưng Light giảm → bất thường

3. **Statistical Foundation**: Dựa trên phân phối chi-square

### 7.3 Hạn Chế của OD Baseline

1. **Euclidean Distance**: Bị ảnh hưởng bởi scale của features
2. **Fixed-width**: Không adaptive theo data distribution
3. **Forward tất cả**: Lãng phí năng lượng

---

## 8. Cách Chạy Mô Phỏng

### 8.1 Cài Đặt

```bash
# 1. Clone repository
git clone <repo-url>
cd mahalanobis

# 2. Cài dependencies
pip install numpy pandas matplotlib scipy

# 3. Tải Intel Lab dataset
# Download từ: https://db.csail.mit.edu/labdata/data.txt.gz
# Giải nén và đặt data.txt vào thư mục gốc
```

### 8.2 Chạy

```bash
python main.py
```

### 8.3 Output

```text
1. Console: Chi tiết từng bước với sample queries
2. network_topology.png: Hình topology mạng 81 nodes
3. wsn_comparison_results.png: So sánh DA/FAR theo thời gian
4. wsn_energy_comparison.png: So sánh năng lượng
5. temperature_with_outliers.png: Dữ liệu với outliers được đánh dấu
```

---

## 9. Tùy Chỉnh

### 9.1 Tắt Verbose Mode

```python
# Trong main.py
def main(verbose: bool = False):  # Đổi True → False
```

### 9.2 Thay Đổi Số Query In Ra

```python
# Trong network.py, class ClusterHead
self._max_sample_logs = 3    # Số query NORMAL
self._max_outlier_logs = 2   # Số query OUTLIER

# Tương tự cho class OD_Detector
```

### 9.3 Thay Đổi Threshold

```python
# ODA-MD: Sử dụng chi2.ppf với alpha khác
chi2_val = chi2.ppf(0.99, df=N_ATTRIBUTES)  # Nghiêm ngặt hơn

# OD: Thay đổi k hoặc width
detector = OD_Detector(n_nodes=3, k=3, width=10.0)
```

---

## 10. Tài Liệu Tham Khảo

1. **Paper gốc**: Titouna, C., et al. "Outlier Detection Approach Using Bayes Classifiers in Wireless Sensor Networks" (2015)

2. **Intel Lab Data**: http://db.csail.mit.edu/labdata/labdata.html

3. **Mahalanobis Distance**: Mahalanobis, P. C. (1936). "On the generalized distance in statistics"

4. **Energy Model**: Heinzelman, W. R., et al. (2000). "Energy-Efficient Communication Protocol for Wireless Microsensor Networks"

---

*Tài liệu này được tạo để giải thích chi tiết code và thuật toán trong dự án.*
