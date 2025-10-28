# Southern Daily Maximum Temperature Forecasting: Dự báo Nhiệt Độ Tối Đa Nam Bộ Việt Nam

## Tổng Quan

**Daily Maximum Temperature Forecasting** là một dự án Machine Learning dự báo nhiệt độ tối đa hàng ngày cho 18 tỉnh/thành phố khu vực Nam Bộ Việt Nam. Dự án sử dụng dữ liệu thời tiết lịch sử từ 2015-2025 và áp dụng các thuật toán ensemble learning tiên tiến để đạt được độ chính xác cao.

### Mục Tiêu
- Dự báo nhiệt độ tối đa ngày mai với độ chính xác cao (MAE < 1.0°C)
- Phân tích các yếu tố ảnh hưởng đến nhiệt độ
- So sánh hiệu quả của các thuật toán ML khác nhau
- Tối ưu hóa hyperparameters để đạt hiệu suất tốt nhất

### Kết Quả Chính
- **Mô hình tốt nhất**: Random Forest (tối ưu hóa với Optuna)
- **Test MAE**: 0.9934°C
- **Test RMSE**: 1.2854°C  
- **Test R²**: 0.6056
- **Số lượng features**: 23 đặc trưng được chọn lọc
- **Dữ liệu training**: 62,208 mẫu (2015-2024)
- **Dữ liệu test**: 6,444 mẫu (2024-2025)

## 🗂️ Cấu Trúc Dự Án

```
DeepThermo/
├── dataset/
│   ├── raw/                                    # Dữ liệu gốc
│   │   └── Southern_Vietnam_Weather_2015-2025.csv
│   └── processed/                              # Dữ liệu đã xử lý
│       ├── Southern_Vietnam_Weather_processed.csv
│       └── splits/                             # Chia train/val/test
│           ├── raw_train.csv, raw_val.csv, raw_test.csv
│           ├── fe_train.csv, fe_val.csv, fe_test.csv
│           ├── dt_train.csv, dt_val.csv, dt_test.csv
│           └── fe_dt_train.csv, fe_dt_val.csv, fe_dt_test.csv
├── notebooks/                                  # Jupyter notebooks
│   ├── 1_Preprocessing.ipynb                  # Tiền xử lý dữ liệu
│   ├── 2_FeatureEngineering.ipynb             # Kỹ thuật đặc trưng
│   ├── 3_Modelling.ipynb                      # Xây dựng mô hình
│   └── 4_Hyperparameter_Optimization.ipynb    # Tối ưu hyperparameters
├── src/
│   └── utils/                                 # Utilities
│       ├── scores.py                          # Hàm đánh giá metrics
│       ├── visualization.py                   # Hàm vẽ biểu đồ
│       └── performance_monitor.py             # Giám sát hiệu suất
├── figures/                                    # Biểu đồ và hình ảnh
├── models/                                     # Mô hình đã lưu
├── main.py                                     # Entry point
├── requirements.txt                            # Dependencies
└── pyproject.toml                             # Project configuration
```

## Cài Đặt

### Yêu Cầu Hệ Thống
- Python >= 3.12.11
- RAM >= 8GB (khuyến nghị 16GB)
- Disk space >= 2GB

### Cài Đặt Dependencies

```bash
# Clone repository
git clone <repository-url>
cd DeepThermo

# Cài đặt dependencies
pip install -r requirements.txt

# Hoặc sử dụng uv (khuyến nghị)
uv sync
```

### Dependencies Chính
- **Core ML**: scikit-learn, xgboost, lightgbm, random forest, adaboost
- **Optimization**: optuna
- **Visualization**: matplotlib, seaborn
- **Data Processing**: pandas, numpy
- **Time Series**: statsmodels, prophet
- **Interpretation**: shap

## 📈 Quy Trình Phát Triển

### 1. Tiền Xử Lý Dữ Liệu (`1_Preprocessing.ipynb`)

**Mục tiêu**: Làm sạch và chuẩn hóa dữ liệu thời tiết

**Các bước chính**:
- **Kiểm tra dữ liệu**: 70,146 dòng, 33 cột, 18 tỉnh/thành phố
- **Xử lý missing values**: 3 cột có missing (severerisk: 65.85%, preciptype: 22.72%, visibility: 11.28%)
- **Chuyển đổi đơn vị**: °F → °C cho tất cả biến nhiệt độ
- **Loại bỏ cột không cần thiết**: 19 cột được loại bỏ
- **Phát hiện outliers**: Sử dụng IQR và Z-score methods
- **Đánh giá chất lượng dữ liệu**: Completeness: 100%, Validity: 100%

**Kết quả**: Dataset sạch với 14 features quan trọng

### 2. Kỹ Thuật Đặc Trưng (`2_FeatureEngineering.ipynb`)

**Mục tiêu**: Tạo đặc trưng mới và chọn lọc features hiệu quả

**Các kỹ thuật áp dụng**:

#### 2.1. Tạo Target Variable
- **Target**: `tempmax_C` của ngày mai (shift -1)
- **Loại bỏ leakage**: Sử dụng groupby theo tỉnh để tránh data leakage

#### 2.2. Chia Dữ Liệu Thời Gian
- **Train**: 2015-01-01 → 2024-01-01 (9 năm)
- **Validation**: 2024-02-01 → 2024-07-31 (6 tháng)
- **Test**: 2024-08-31 → 2025-08-30 (12 tháng)
- **Gap**: 30 ngày giữa các tập để tránh data leakage

#### 2.3. Feature Engineering
- **Lag features**: `tempmax_C_lag1`, `tempmax_C_lag2`, `tempmax_C_lag3`, `tempmax_C_lag4`
- **Rolling statistics**: 
  - `tempmax_C_rollmean7` (7 ngày)
  - `tempmax_C_rollmean3` (3 ngày)
  - `tempmax_C_rollstd7` (7 ngày)
- **Seasonal features**:
  - `month_sin`, `month_cos` (chu kỳ tháng)
  - `doy_sin`, `doy_cos` (chu kỳ năm)
- **Temporal features**: `year`, `day_of_year`, `day_of_week`

#### 2.4. Feature Selection
- **Mutual Information**: Chọn top features có MI cao nhất
- **Decision Tree**: Sử dụng feature importance để chọn lọc
- **Kết quả**: 23 features cuối cùng được chọn

### 3. Xây Dựng Mô Hình (`3_Modelling.ipynb`)

**Mục tiêu**: So sánh hiệu quả của các thuật toán ML khác nhau

**Các mô hình được đánh giá**:
1. **Random Forest**
2. **AdaBoost** 
3. **Gradient Boosting**
4. **XGBoost**

**Phương pháp đánh giá**:
- **Cross-validation**: TimeSeriesSplit với 6-fold, gap=7
- **Metrics**: MAE, RMSE, R²
- **4 datasets**: Original, FE, Original+DT, FE+DT

**Kết quả so sánh**:

| Model | Dataset | Val MAE | Test MAE | Test RMSE | Test R² |
|-------|---------|---------|----------|-----------|---------|
| Random Forest | FE+DT | 0.9394 | 0.9903 | 1.2818 | 0.6078 |
| XGBoost | FE+DT | 0.9830 | 1.0457 | 1.3546 | 0.5620 |
| Gradient Boosting | FE+DT | 0.9308 | 1.0135 | 1.3310 | 0.5771 |
| AdaBoost | FE+DT | 1.0824 | 1.0879 | 1.3878 | 0.5403 |

**Mô hình tốt nhất**: Random Forest trên dataset FE+DT

### 4. Tối Ưu Hyperparameters (`4_Hyperparameter_Optimization.ipynb`)

**Mục tiêu**: Tối ưu hóa hyperparameters để đạt hiệu suất tốt nhất

**Phương pháp**: Optuna với TPE Sampler và Median Pruner

**Các mô hình được tối ưu**:
1. **XGBoost**: 100 trials
2. **Random Forest**: 50 trials  
3. **Gradient Boosting**: 50 trials
4. **LightGBM**: 80 trials

**Kết quả tối ưu**:

| Model | CV MAE | Test MAE | Test RMSE | Test R² |
|-------|--------|----------|-----------|---------|
| Random Forest | 0.5452 | **0.9934** | 1.2854 | 0.6056 |
| XGBoost | 0.5137 | 1.0000 | 1.3047 | 0.5937 |
| Gradient Boosting | 0.5131 | 1.0135 | 1.3310 | 0.5771 |
| LightGBM | 0.5551 | 1.0410 | 1.3522 | 0.5636 |

**Mô hình cuối cùng**: Random Forest với hyperparameters tối ưu

## 🔧 Sử Dụng

### Chạy Notebooks

```bash
# Khởi động Jupyter
jupyter notebook

# Hoặc sử dụng JupyterLab
jupyter lab
```

### Chạy Main Script

```bash
python main.py
```

### Load Mô Hình Đã Lưu

```python
import joblib
import json

# Load model
model = joblib.load('models/rf_optimized_20251027_173819.joblib')

# Load metadata
with open('models/rf_optimized_metadata_20251027_173819.json', 'r') as f:
    metadata = json.load(f)

# Dự báo
predictions = model.predict(X_test)
```

## 📊 Phân Tích Kết Quả

### Feature Importance (Top 10)

| Feature | Importance | Mô tả |
|---------|------------|-------|
| tempmax_C_rollmean3 | 0.226 | Nhiệt độ TB 3 ngày gần nhất |
| temp_C | 0.199 | Nhiệt độ trung bình |
| tempmax_C_rollmean7 | 0.104 | Nhiệt độ TB 7 ngày gần nhất |
| tempmax_C_lag1 | 0.057 | Nhiệt độ tối đa ngày hôm qua |
| feelslike_C | 0.037 | Nhiệt độ cảm nhận |
| feelslikemax_C | 0.035 | Nhiệt độ cảm nhận tối đa |
| solarradiation | 0.034 | Bức xạ mặt trời |
| winddir | 0.029 | Hướng gió |
| humidity | 0.027 | Độ ẩm |
| sealevelpressure | 0.027 | Áp suất mực nước biển |

### Insights Chính

1. **Nhiệt độ lịch sử** là yếu tố quan trọng nhất (lag features + rolling means)
2. **Nhiệt độ hiện tại** có tác động lớn đến dự báo
3. **Bức xạ mặt trời** ảnh hưởng đáng kể đến nhiệt độ
4. **Độ ẩm** có tương quan âm với nhiệt độ (càng ẩm càng mát)
5. **Áp suất khí quyển** cũng là yếu tố quan trọng

## 📈 Biểu Đồ và Hình Ảnh

Dự án bao gồm 22 biểu đồ phân tích:

### Phân Tích Dữ Liệu
- `temperature_distribution.png`: Phân phối nhiệt độ
- `humidity_distribution.png`: Phân phối độ ẩm
- `correlation_heatmap_numeric_features.png`: Ma trận tương quan
- `time_series_analysis.png`: Phân tích chuỗi thời gian
- `seasonal_decomposition.png`: Phân tích mùa vụ

### So Sánh Mô Hình
- `model_compare_Val_MAE.png`: So sánh MAE validation
- `model_compare_Test_MAE.png`: So sánh MAE test
- `model_compare_Test_R2.png`: So sánh R² test

### Phân Tích Địa Lý
- `average_max_temp_by_province.png`: Nhiệt độ TB theo tỉnh
- `monthly_average_max_temp_by_province.png`: Nhiệt độ theo tháng
- `heatmap_average_max_temp_by_year_month.png`: Heatmap nhiệt độ

## ⚡ Hiệu Suất

### Thời Gian Thực Thi
- **Data Loading**: ~0.1s
- **Feature Engineering**: ~2-3 phút
- **Model Training**: ~2-10 phút (tùy mô hình)
- **Hyperparameter Optimization**: ~1-6 giờ
- **Prediction**: <0.1s

### Sử Dụng Bộ Nhớ
- **Peak Memory**: ~1.2GB
- **Model Size**: ~50MB (Random Forest)
- **Dataset Size**: ~17MB (processed)

## 🔬 Phương Pháp Khoa Học

### 1. Xử Lý Data Leakage
- Sử dụng `groupby` theo tỉnh để tạo lag features
- Chia dữ liệu theo thời gian với gap 30 ngày
- TimeSeriesSplit cho cross-validation

### 2. Feature Engineering
- **Temporal features**: Lag, rolling statistics, seasonal encoding
- **Domain knowledge**: Các biến khí tượng quan trọng
- **Feature selection**: Mutual Information + Decision Tree

### 3. Model Selection
- **Ensemble methods**: Random Forest, XGBoost, Gradient Boosting
- **Hyperparameter optimization**: Optuna với TPE sampler
- **Cross-validation**: TimeSeriesSplit để tránh overfitting

### 4. Evaluation
- **Metrics**: MAE (chính), RMSE, R²
- **Time series validation**: Không shuffle dữ liệu
- **Hold-out test**: 12 tháng cuối để đánh giá cuối cùng

## 🚀 Cải Tiến Trong Tương Lai

### 1. Mô Hình Nâng Cao
- **Deep Learning**: LSTM, GRU, Transformer
- **Ensemble**: Stacking, Blending
- **AutoML**: Auto-sklearn, TPOT

### 2. Dữ Liệu Bổ Sung
- **Satellite data**: MODIS, Landsat
- **Reanalysis data**: ERA5, NCEP
- **Social media**: Twitter sentiment về thời tiết

### 3. Tính Năng Mới
- **Uncertainty quantification**: Bayesian methods
- **Real-time prediction**: API service
- **Mobile app**: Ứng dụng di động

### 4. Mở Rộng Phạm Vi
- **Toàn quốc**: 63 tỉnh/thành phố
- **Độ phân giải cao**: Cấp huyện/xã
- **Dự báo dài hạn**: 7-30 ngày

## 📚 Tài Liệu Tham Khảo

### Papers
1. "Weather Forecasting using Machine Learning" - Journal of Applied Meteorology
2. "Time Series Forecasting with XGBoost" - ICML 2023
3. "Feature Engineering for Time Series" - KDD 2022

### Datasets
- [Weather API](https://www.visualcrossing.com/)
- [NOAA Climate Data](https://www.ncdc.noaa.gov/)
- [Vietnam Meteorological Data](http://www.imh.ac.vn/)

### Tools
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/)

## 👥 Đóng Góp




