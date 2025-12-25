# Giải Thích Chi Tiết Dự Án ODA-MD (Outlier Detection Algorithm using Mahalanobis Distance)

## 1. Tổng Quan Dự Án

Đây là bản triển khai Python mô phỏng thuật toán **ODA-MD (Outlier Detection Algorithm based on Mahalanobis Distance)** cho Mạng Cảm Biến Không Dây (Wireless Sensor Networks - WSN). Dự án so sánh ODA-MD với thuật toán baseline OD (Fixed-width Clustering) để chứng minh ưu điểm về độ chính xác phát hiện và hiệu quả năng lượng.

### Điểm Quan Trọng Theo Paper

> **"We chose four types of data (Temperature, light, voltage, and humidity)"** - Section I, Contribution

Mỗi quan sát (observation) là một vector **4 chiều** chứa:
1. **Temperature** (°C)
2. **Humidity** (%)
3. **Light** (Lux)
4. **Voltage** (V)

### Thông Số Mô Phỏng

| Thông số | Giá trị | Mô tả |
|----------|---------|-------|
| N_NODES | 81 | Tổng số node cảm biến |
| N_CLUSTERS | 10 | Số cụm (cluster) |
| N_ATTRIBUTES | 4 | Số thuộc tính cảm biến |
| AREA_SIZE | 100×100m | Kích thước vùng triển khai |
| BATCH_SIZE | 50 | Số quan sát mỗi batch |
| INITIAL_ENERGY | 2.0 J | Năng lượng ban đầu mỗi node |

---

## 2. Cấu Trúc Dự Án

```text
mahalanobis/
├── main.py                 # Điểm khởi chạy chính
├── data.txt                # Dữ liệu Intel Lab (cần tải riêng)
├── modules/
│   ├── __init__.py
│   ├── config.py           # Cấu hình và hằng số (bao gồm SENSOR_ATTRIBUTES)
│   ├── topology.py         # Tạo topology mạng (K-means clustering)
│   ├── data_loader.py      # Tải 4 thuộc tính từ Intel Lab
│   ├── network.py          # SensorNode, ClusterHead (ODA-MD), OD_Detector
│   ├── simulation.py       # Engine mô phỏng
│   └── visualization.py    # Vẽ biểu đồ
└── outputs/
    ├── network_topology.png
    ├── wsn_comparison_results.png
    ├── wsn_energy_comparison.png
    └── temperature_with_outliers.png
```

---

## 3. Giải Thích Chi Tiết Từng Module

### 3.1 config.py - Cấu Hình Hệ Thống

```python
# Mô hình năng lượng Heinzelman
E_ELEC = 50e-9      # 50 nJ/bit - năng lượng điện tử
E_AMP = 100e-12     # 100 pJ/bit/m² - năng lượng khuếch đại
E_DA = 5e-9         # 5 nJ/bit - năng lượng tổng hợp dữ liệu

PACKET_SIZE = 32    # bytes - kích thước gói tin
INITIAL_ENERGY = 2.0  # Joules - năng lượng ban đầu

# 4 Thuộc tính cảm biến (QUAN TRỌNG - theo paper)
SENSOR_ATTRIBUTES = ['temperature', 'humidity', 'light', 'voltage']
N_ATTRIBUTES = 4

# Cluster 2 đặc biệt (dùng dữ liệu thực từ Intel Lab)
CLUSTER_2_NODES = [36, 37, 38]  # Các node dùng dữ liệu thực
SINK_ID = 1                     # Node trung tâm (Sink)
CH_ID = 2                       # Cluster Head của Cluster 2
```

---

### 3.2 data_loader.py - Tải Dữ Liệu 4 Thuộc Tính

#### Hàm `load_intel_data()` - Tải Đầy Đủ 4 Thuộc Tính

```python
def load_intel_data(filepath: str = 'data.txt', 
                    node_ids: List[int] = CLUSTER_2_NODES,
                    date_start: str = '2004-03-11',
                    date_end: str = '2004-03-15') -> pd.DataFrame:
```

**Luồng xử lý:**

1. **Đọc file**: Tải dữ liệu với 8 cột: date, time, epoch, moteid, temperature, humidity, light, voltage
2. **Lọc node**: Chỉ giữ node 36, 37, 38
3. **Lọc giá trị hợp lệ**: 
   - Temperature: -40°C < temp < 100°C
   - Humidity: 0% ≤ humid ≤ 100%
   - Light: ≥ 0
   - Voltage: 0V < volt < 5V
4. **Pivot cho từng thuộc tính**: Tạo cột `node_36_temperature`, `node_36_humidity`, ...
5. **Kết quả**: DataFrame với ~10,167 epochs và 12 cột (3 nodes × 4 attributes)

**Ví dụ cấu trúc dữ liệu:**

```text
epoch | node_36_temp | node_36_humid | node_36_light | node_36_volt | node_37_temp | ...
------|--------------|---------------|---------------|--------------|--------------|----
  1   |    22.5      |     42.3      |     512       |     2.53     |    22.1      | ...
  2   |    22.6      |     42.1      |     520       |     2.52     |    22.2      | ...
```

#### Hàm `get_node_data_matrix()` - Lấy Ma Trận Cho Một Node

```python
def get_node_data_matrix(df: pd.DataFrame, node_id: int) -> np.ndarray:
    """
    Trả về ma trận (n_epochs, 4) cho một node.
    Mỗi hàng là vector [temp, humid, light, volt]
    """
```

---

### 3.3 network.py - Thuật Toán ODA-MD (4 Thuộc Tính)

#### Class `SensorNode` - Node Cảm Biến 4 Thuộc Tính

```python
class SensorNode:
    def __init__(self, node_id: int, data_matrix: np.ndarray):
        """
        data_matrix: Shape (n_epochs, 4) - mỗi hàng là [temp, humid, light, volt]
        """
        self.node_id = node_id
        self.data = data_matrix  # Shape: (n_epochs, 4)
        self.energy = INITIAL_ENERGY
```

**Phương thức `sense_and_send_batch()`:**

```python
def sense_and_send_batch(self, start_idx: int, n_samples: int) -> np.ndarray:
    """
    Trả về ma trận (n_samples, 4) - batch của các observation 4 chiều
    """
    data_batch = self.data[start_idx:end_idx]  # (n_samples, 4)
    
    # Năng lượng: 4 attributes × n_samples × PACKET_SIZE bits
    total_bits = actual_n * N_ATTRIBUTES * PACKET_SIZE * 8
    tx_energy = (E_ELEC + E_AMP * self.dist_to_ch**2) * total_bits
    
    return data_batch
```

---

#### Class `ClusterHead` - Thuật Toán ODA-MD (Algorithm 1 & 2) - Time-Step

**Đây là trái tim của thuật toán**, triển khai đúng theo Figure 2.

##### Luồng Xử Lý Tại Mỗi Time Step t

```text
t0: CH gửi request ────────────────────────────────────────────────────►
                    N₁         N₂         N₃        ...        Nₚ
t1: Nodes trả lời  ◄── A₁ ─── ◄── A₂ ─── ◄── A₃ ───        ◄── Aₚ ───
                    [4 vals]   [4 vals]   [4 vals]          [4 vals]

t3: CH xây dựng Table_MD:
    ┌────────┬────────┬────────┬─────┬────────┐
    │   A₁   │   A₂   │   A₃   │ ... │   Aₚ   │
    ├────────┼────────┼────────┼─────┼────────┤
    │ x₁₁    │ x₁₂    │ x₁₃    │ ... │ x₁ₚ    │ ← temperature
    │ x₂₁    │ x₂₂    │ x₂₃    │ ... │ x₂ₚ    │ ← humidity
    │ x₃₁    │ x₃₂    │ x₃₃    │ ... │ x₃ₚ    │ ← light
    │ x₄₁    │ x₄₂    │ x₄₃    │ ... │ x₄ₚ    │ ← voltage
    └────────┴────────┴────────┴─────┴────────┘
```

##### Phương thức `process_time_step(epoch_idx)`

```python
def process_time_step(self, epoch_idx: int) -> Tuple[bool, Dict]:
    """
    Xử lý MỘT time step theo Figure 2.
    """
    # ===== ALGORITHM 1: Thu thập dữ liệu =====
    # Bước 1: CH gửi request (broadcast)
    self.consume_energy((E_ELEC + E_AMP * RADIO_RANGE**2) * PACKET_SIZE * 8)
    
    # Bước 2: Thu thập Aₖ từ mỗi node
    table_md = []
    for member in self.members:
        member.receive_request()
        obs = member.send_observation(epoch_idx)  # Aₖ = [temp, humid, light, volt]
        table_md.append(obs)
    
    table_md = np.array(table_md)  # Shape: (n_nodes, 4)
    
    # Thêm vào history để tính thống kê
    self.history.append(table_md)
    if len(self.history) > self.window_size:
        self.history.pop(0)
    
    # ===== ALGORITHM 2: Tính MD và Phát hiện =====
    is_outlier = False
    
    if len(self.history) >= 5:  # Cần đủ mẫu cho thống kê
        # Chi-square threshold với df = 4
        chi2_val = chi2.ppf(0.975, df=N_ATTRIBUTES)  # ≈ 11.143
        threshold = np.sqrt(chi2_val)                # ≈ 3.34
        
        for node_idx in range(n_nodes):
            # Lấy history của node này
            node_history = history_stack[:, node_idx, :]  # (window, 4)
            
            # Tính μ và Σ từ history (Eq. 3 & 4)
            mu = np.mean(node_history, axis=0)
            cov = np.cov(node_history, rowvar=False)
            cov_inv = np.linalg.inv(cov)
            
            # Tính MD cho observation hiện tại (Eq. 2)
            current_obs = table_md[node_idx]
            diff = current_obs - mu
            md = np.sqrt(np.dot(np.dot(diff, cov_inv), diff))
            
            # Quyết định (Eq. 5)
            if md >= threshold:
                is_outlier = True
                break
    
    # Chỉ forward nếu KHÔNG PHẢI outlier (tiết kiệm năng lượng)
    if not is_outlier:
        self.consume_energy(...)  # Forward to sink
    
    return is_outlier, {...}
```

---

### 3.4 Công Thức Toán Học

#### Mahalanobis Distance (Eq. 2)

$$MD_i = \sqrt{(X_i - \mu)^T \cdot \Sigma^{-1} \cdot (X_i - \mu)}$$

Trong đó:
- **X_i**: Vector quan sát 4 chiều `[temp, humid, light, volt]`
- **μ (mu)**: Vector trung bình 4 chiều (tính từ rolling window)
- **Σ (Sigma)**: Ma trận hiệp phương sai 4×4

#### Ngưỡng Chi-Square (Eq. 7)

$$\text{threshold} = \sqrt{\chi^2_{4, 0.975}} = \sqrt{11.143} \approx 3.34$$

---

## 4. Quy Trình Chạy Toàn Bộ Dự Án

### Sơ Đồ Luồng Thực Thi (Time-Step)

```text
for each time_step t = 0, 1, 2, ..., T:
    ┌─────────────────────────────────────────────────────────────┐
    │ 1. CH gửi request đến tất cả nodes                          │
    │ 2. Mỗi node Nₖ gửi về Aₖ = [temp, humid, light, volt]       │
    │ 3. CH xây dựng Table_MD (4 × p)                             │
    │ 4. CH tính MD cho từng node dựa trên rolling window         │
    │ 5. Nếu MD ≥ threshold → Outlier → Loại bỏ                   │
    │    Nếu MD < threshold → Normal → Forward đến Sink           │
    └─────────────────────────────────────────────────────────────┘
```

---

## 5. Kết Quả Mô Phỏng

### Bảng So Sánh (Time-Step Processing)

| Thuật toán | Detection Accuracy | False Alarm Rate | Năng lượng |
|------------|-------------------|------------------|------------|
| **ODA-MD** | **99.00%** | 15.56% | 339.24 J |
| OD | 43.00% | 42.72% | 387.76 J |

### Phân Tích

1. **ODA-MD đạt 99% Detection Accuracy** vì:
   - Xử lý từng time step đúng theo Figure 2
   - Rolling window (50 samples) cho thống kê ổn định
   - 10/1000 outliers bị miss do warm-up period (chưa đủ 5 samples)

2. **FAR cao hơn paper (15.56% vs ~0%)** do:
   - Dữ liệu tổng hợp cho các cluster khác có đặc tính khác
   - Có thể điều chỉnh threshold hoặc window size

3. **ODA-MD tiết kiệm ~12.5% năng lượng** vì:
   - Outlier bị lọc tại CH, không forward đến Sink


---

## 6. Cách Chạy Mô Phỏng

```bash
# 1. Cài đặt dependencies
pip install numpy pandas matplotlib scipy

# 2. Tải dataset Intel Lab (xem README.md)
# Đặt file data.txt vào thư mục gốc

# 3. Chạy mô phỏng
python main.py
```

---

## 7. Tài Liệu Tham Khảo

1. **Paper gốc**: Titouna et al. "Outlier Detection Algorithm based on Mahalanobis Distance for Wireless Sensor Networks" (ICCCI 2019)
2. **Intel Lab Data**: http://db.csail.mit.edu/labdata/labdata.html
3. **Mahalanobis Distance**: Mahalanobis, P. C. (1936). On the generalized distance in statistics.
4. **Heinzelman Energy Model**: Heinzelman, W. R., et al. (2000). Energy-efficient communication protocol for wireless microsensor networks.
