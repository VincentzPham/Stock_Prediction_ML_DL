import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.metrics import root_mean_squared_error
import warnings
import joblib  # Thêm để save model

warnings.filterwarnings('ignore')

# Tạo thư mục lưu kết quả và model
output_dir = '../../../'  # This points to the parent folder
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

output_dir_result = os.path.join(output_dir, 'Result/AAPL/Exponential Smoothing/')
if not os.path.exists(output_dir_result):
    os.makedirs(output_dir_result, exist_ok=True)

output_dir_models = os.path.join(output_dir, 'Models/AAPL/Exponential Smoothing/')
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
df['Close'] = df['Close'].ffill()

# Kiểm tra DataFrame sau xử lý (tùy chọn)
print(df.head())  # Nên hiển thị dữ liệu từ 2014-09-17
print(df.tail())  # Kiểm tra ngày cuối
df.info()  # Xem số hàng và kiểu dữ liệu

# Chia train/test (90% train, 10% test)
split_ratio = 0.90
split_index = int(len(df) * split_ratio)
train = df['Close'][:split_index]
test = df['Close'][split_index:]

# Fit và dự báo từ 4 mô hình
model_ses = SimpleExpSmoothing(train).fit()
pred_ses = model_ses.forecast(len(test))

model_holt = Holt(train).fit()
pred_holt = model_holt.forecast(len(test))

model_hw_no_season = ExponentialSmoothing(train, trend='add', seasonal=None).fit()
pred_hw_no_season = model_hw_no_season.forecast(len(test))

model_hw_season_mul = ExponentialSmoothing(train, trend='add', seasonal='mul', seasonal_periods=252).fit()
pred_hw_season_mul = model_hw_season_mul.forecast(len(test))

# Save models dạng .pkl
joblib.dump(model_ses, os.path.join(output_dir_models, 'ses_model.pkl'))
joblib.dump(model_holt, os.path.join(output_dir_models, 'holt_model.pkl'))
joblib.dump(model_hw_no_season, os.path.join(output_dir_models, 'hw_no_season_model.pkl'))
joblib.dump(model_hw_season_mul, os.path.join(output_dir_models, 'hw_season_mul_model.pkl'))

# Vẽ biểu đồ predicted vs actual và save .png
plt.figure(figsize=(14, 6))
plt.plot(test.index, test.values, label='Actual', color='black', linewidth=2)
plt.plot(test.index, pred_ses, label='SES Forecast')
plt.plot(test.index, pred_holt, label='Holt Forecast')
plt.plot(test.index, pred_hw_no_season, label='HW (no seasonal) Forecast')
plt.plot(test.index, pred_hw_season_mul, label='HW (seasonal multiplicative) Forecast')
plt.title('Predicted vs Actual (Test Set)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir_result, 'predicted_vs_actual.png'))
plt.show()

# Vẽ residual plot cho từng model và save .png
models = {
    'SES': pred_ses,
    'Holt': pred_holt,
    'HW_no_season': pred_hw_no_season,
    'HW_season_mul': pred_hw_season_mul
}

for name, pred in models.items():
    residuals = test - pred
    plt.figure(figsize=(10, 5))
    plt.plot(test.index, residuals, label=f'Residuals ({name})')
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f'Residual Plot for {name}')
    plt.xlabel('Date')
    plt.ylabel('Residual')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_result, f'residual_{name.lower()}.png'))
    plt.show()

# Tính metrics cho từng model
def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = root_mean_squared_error(actual, predicted)
    mape = mean_absolute_percentage_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}

metrics_ses = calculate_metrics(test, pred_ses)
metrics_holt = calculate_metrics(test, pred_holt)
metrics_hw_no_season = calculate_metrics(test, pred_hw_no_season)
metrics_hw_season_mul = calculate_metrics(test, pred_hw_season_mul)

# Tạo bảng so sánh metrics
metrics_df = pd.DataFrame({
    'SES': metrics_ses,
    'Holt': metrics_holt,
    'HW_no_season': metrics_hw_no_season,
    'HW_season_mul': metrics_hw_season_mul
}).T
metrics_df.to_csv(os.path.join(output_dir_result, 'metrics_comparison.csv'), index=True)

# Tạo bảng predicted vs actual
pred_vs_actual_df = pd.DataFrame({
    'Actual': test,
    'SES_Pred': pred_ses,
    'Holt_Pred': pred_holt,
    'HW_no_season_Pred': pred_hw_no_season,
    'HW_season_mul_Pred': pred_hw_season_mul
}, index=test.index)
pred_vs_actual_df.to_csv(os.path.join(output_dir_result, 'predicted_vs_actual.csv'), index=True)

# In ra để kiểm tra
print("Metrics Comparison:")
print(metrics_df)
print("\nPredicted vs Actual saved to CSV.")