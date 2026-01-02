# Cài đặt các thư viện cần thiết
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Đọc dữ liệu với skiprows=2
file_path = '../../Data/AAPL.csv'  # Sử dụng relative path để chỉ đến thư mục Data

df_cleaned = pd.read_csv(file_path, skiprows=2)

# Chuyển đổi cột 'Date' thành datetime
df_cleaned['Date'] = pd.to_datetime(df_cleaned['Date'])

# Đặt tham số cho mô hình SARIMA (p=4, d=1, q=0), P=1, D=1, Q=0, m=365
sarima_model = SARIMAX(df_cleaned['Unnamed: 1'], order=(4, 1, 0), seasonal_order=(1, 1, 0, 365))

# Huấn luyện mô hình SARIMA
sarima_results = sarima_model.fit()

# Dự đoán giá trị trong tương lai (ví dụ, dự đoán 30 ngày tiếp theo)
forecast = sarima_results.get_forecast(steps=30)
forecast_mean = forecast.predicted_mean
forecast_conf_int = forecast.conf_int()

# Vẽ biểu đồ kết quả dự đoán từ mô hình SARIMA
plt.figure(figsize=(12, 6))
plt.plot(df_cleaned['Unnamed: 1'], label='Observed')
plt.plot(forecast_mean.index, forecast_mean, label='Forecast', color='red')
plt.fill_between(forecast_mean.index, forecast_conf_int.iloc[:, 0], forecast_conf_int.iloc[:, 1], color='pink', alpha=0.3)
plt.title('SARIMA Forecast for the Next 30 Days')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Hiển thị các chỉ số đánh giá mô hình SARIMA
print(sarima_results.summary())
