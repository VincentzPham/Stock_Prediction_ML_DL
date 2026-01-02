import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,root_mean_squared_error , r2_score

# Tạo thư mục lưu kết quả và model
output_dir = '../../../'  # This points to the parent folder
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

output_dir_result = os.path.join(output_dir, 'Result/AAPL/Multiple Linear Regression/')
if not os.path.exists(output_dir_result):
    os.makedirs(output_dir_result, exist_ok=True)

output_dir_models = os.path.join(output_dir, 'Models/AAPL/Multiple Linear Regression/')
if not os.path.exists(output_dir_models):
    os.makedirs(output_dir_models, exist_ok=True)

# Đường dẫn tới tệp CSV
file_path = '../../../Data/AAPL.csv'

# Đọc CSV, bỏ qua 2 hàng đầu (Ticker và Date NaN)
df = pd.read_csv(file_path, skiprows=2)

# Đặt tên cột thủ công (dựa trên cấu trúc file)
df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

# Chuyển Date thành datetime và set làm index
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')  # Format ngày tháng kiểu YYYY-MM-DD
df.set_index('Date', inplace=True)

# Đảm bảo index có tần suất để forecast hoạt động đúng
df = df.asfreq('B')  # B = Business day (ngày giao dịch)

# Nếu có missing date do thị trường không giao dịch, có thể dùng forward fill:
#df['Close'] = df['Close'].ffill()
df = df.ffill()
# Kiểm tra DataFrame sau xử lý (tùy chọn)
print(df.head())  # Nên hiển thị dữ liệu từ 2014-09-17
print(df.tail())  # Kiểm tra ngày cuối
df.info()  # Xem số hàng và kiểu dữ liệu

# --- Phần tiền xử lý bạn đã có ở trên ---

features = ['Open', 'High', 'Low', 'Volume']
X = df[features]
y = df['Close']

split_index = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# Huấn luyện mô hình
mlr_model = LinearRegression()
mlr_model.fit(X_train, y_train)

# Dự đoán
y_pred = mlr_model.predict(X_test)

# --- 1. Save model dưới dạng .pkl ---
model_path = os.path.join(output_dir_models, 'mlr_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(mlr_model, f)

# --- 2. Save visualization & residual plot .png ---
result_df = pd.DataFrame({
    'Date': X_test.index,
    'Actual': y_test.values,
    'Predicted': y_pred
})

# --- 2. Save scatter plot (Actual vs Predicted) & residual scatter plot ---

# 1. Scatter plot giữa Actual (trục X) và Predicted (trục Y)
plt.figure(figsize=(8,8))
plt.scatter(result_df['Actual'], result_df['Predicted'], color='blue', alpha=0.5, s=10)
plt.plot([result_df['Actual'].min(), result_df['Actual'].max()],
         [result_df['Actual'].min(), result_df['Actual'].max()],
         'r--', lw=2, label='Ideal Fit (y=x)')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.title('Scatter Plot: Actual vs Predicted')
plt.legend()
plt.tight_layout()
scatter_plot_path = os.path.join(output_dir_result, 'scatter_actual_vs_predicted.png')
plt.savefig(scatter_plot_path)
plt.close()

# 2. Residual plot: trục X là actual, trục Y là (actual - predicted)
plt.figure(figsize=(8,5))
residuals = result_df['Actual'] - result_df['Predicted']
plt.scatter(result_df['Actual'], residuals, color='purple', s=10, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Actual Value')
plt.ylabel('Residual (Actual - Predicted)')
plt.title('Residual Scatter Plot')
plt.tight_layout()
residual_scatter_plot_path = os.path.join(output_dir_result, 'residual_scatter_plot.png')
plt.savefig(residual_scatter_plot_path)
plt.close()


# --- 3. Save bảng so sánh thông số và bảng predicted/actual values .csv ---
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

metrics_df = pd.DataFrame([{
    'RMSE': rmse,
    'MAE': mae,
    'MSE': mse,
    'MAPE': mape,
    'R2 Score': r2
}])
metrics_csv_path = os.path.join(output_dir_result, 'mlr_metrics.csv')
metrics_df.to_csv(metrics_csv_path, index=False)

# Save bảng predicted/actual values
predicted_csv_path = os.path.join(output_dir_result, 'actual_vs_predicted.csv')
result_df.to_csv(predicted_csv_path, index=False)

# --- Thông báo các đường dẫn đã lưu ---
print('Đã lưu model tại:', model_path)
print('Đã lưu scatter plot Actual vs Predicted:', scatter_plot_path)
print('Đã lưu residual scatter plot:', residual_scatter_plot_path)
print('Đã lưu bảng metrics:', metrics_csv_path)
print('Đã lưu bảng actual vs predicted:', predicted_csv_path)
