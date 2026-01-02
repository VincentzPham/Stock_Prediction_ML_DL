# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
# from sklearn.metrics import root_mean_squared_error
# from prophet import Prophet
# import joblib  # Để save model
# import warnings

# warnings.filterwarnings('ignore')

# # Tạo thư mục lưu kết quả và model
# output_dir = '../../../'  # This points to the parent folder
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir, exist_ok=True)

# output_dir_result = os.path.join(output_dir, 'Result/AAPL/Prophet/')
# if not os.path.exists(output_dir_result):
#     os.makedirs(output_dir_result, exist_ok=True)

# output_dir_models = os.path.join(output_dir, 'Models/AAPL/Prophet/')
# if not os.path.exists(output_dir_models):
#     os.makedirs(output_dir_models, exist_ok=True)

# # Đường dẫn tới tệp CSV
# file_path = '../../../Data/AAPL.csv'

# # Đọc CSV, bỏ qua 2 hàng đầu (Ticker và Date NaN)
# df = pd.read_csv(file_path, skiprows=2)

# # Đặt tên cột thủ công (dựa trên cấu trúc file)
# df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

# # Chuyển Date thành datetime
# df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')  # Format ngày tháng kiểu YYYY-MM-DD

# # Chuẩn bị dữ liệu cho Prophet: cần cột 'ds' và 'y'
# prophet_df = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

# # Đảm bảo index có tần suất để forecast hoạt động đúng
# prophet_df.set_index('ds', inplace=True)
# prophet_df = prophet_df.asfreq('B')  # B = Business day (ngày giao dịch)
# prophet_df['y'] = prophet_df['y'].ffill()
# prophet_df.reset_index(inplace=True)  # Reset index để có 'ds' làm cột

# # Kiểm tra DataFrame sau xử lý (tùy chọn)
# print(prophet_df.head())  # Nên hiển thị dữ liệu từ 2014-09-17
# print(prophet_df.tail())  # Kiểm tra ngày cuối
# prophet_df.info()  # Xem số hàng và kiểu dữ liệu

# # Chia train/test (90% train, 10% test)
# split_ratio = 0.90
# split_index = int(len(prophet_df) * split_ratio)
# train = prophet_df[:split_index]
# test = prophet_df[split_index:]

# # Fit model Prophet
# model_prophet = Prophet(daily_seasonality=True)  # Bật daily seasonality vì dữ liệu hàng ngày
# model_prophet.fit(train)

# # Dự báo cho phần test
# future = model_prophet.make_future_dataframe(periods=len(test), freq='B')
# forecast = model_prophet.predict(future)
# pred_prophet = forecast['yhat'].tail(len(test)).values  # Lấy dự báo cho test set

# # Save model dạng .pkl
# joblib.dump(model_prophet, os.path.join(output_dir_models, 'prophet_model.pkl'))

# # Vẽ biểu đồ predicted vs actual và save .png
# plt.figure(figsize=(14, 6))
# plt.plot(test['ds'], test['y'], label='Actual', color='black', linewidth=2)
# plt.plot(test['ds'], pred_prophet, label='Prophet Forecast')
# plt.title('Predicted vs Actual (Test Set)')
# plt.xlabel('Date')
# plt.ylabel('Close Price')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(os.path.join(output_dir_result, 'predicted_vs_actual.png'))
# plt.show()

# # Vẽ residual plot và save .png
# residuals = test['y'].values - pred_prophet
# plt.figure(figsize=(10, 5))
# plt.plot(test['ds'], residuals, label='Residuals (Prophet)')
# plt.axhline(0, color='red', linestyle='--')
# plt.title('Residual Plot for Prophet')
# plt.xlabel('Date')
# plt.ylabel('Residual')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(os.path.join(output_dir_result, 'residual_prophet.png'))
# plt.show()

# # Tính metrics
# def calculate_metrics(actual, predicted):
#     mae = mean_absolute_error(actual, predicted)
#     mse = mean_squared_error(actual, predicted)
#     rmse = root_mean_squared_error(actual, predicted)
#     mape = mean_absolute_percentage_error(actual, predicted)
#     r2 = r2_score(actual, predicted)
#     return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}

# metrics_prophet = calculate_metrics(test['y'], pred_prophet)

# # Tạo bảng metrics
# metrics_df = pd.DataFrame(metrics_prophet, index=['Prophet']).T
# metrics_df.to_csv(os.path.join(output_dir_result, 'metrics_comparison.csv'), index=True)

# # Tạo bảng predicted vs actual
# pred_vs_actual_df = pd.DataFrame({
#     'Actual': test['y'].values,
#     'Prophet_Pred': pred_prophet
# }, index=test['ds'])
# pred_vs_actual_df.to_csv(os.path.join(output_dir_result, 'predicted_vs_actual.csv'), index=True)

# # In ra để kiểm tra
# print("Metrics Comparison:")
# print(metrics_df)
# print("\nPredicted vs Actual saved to CSV.")



import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.metrics import root_mean_squared_error
from prophet import Prophet
import joblib  # Để save model
import warnings

warnings.filterwarnings('ignore')

# Tạo thư mục lưu kết quả và model
output_dir = '../../../'  # This points to the parent folder
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

output_dir_result = os.path.join(output_dir, 'Result/AAPL/Prophet/')
if not os.path.exists(output_dir_result):
    os.makedirs(output_dir_result, exist_ok=True)

output_dir_models = os.path.join(output_dir, 'Models/AAPL/Prophet/')
if not os.path.exists(output_dir_models):
    os.makedirs(output_dir_models, exist_ok=True)

# Đường dẫn tới tệp CSV
file_path = '../../../Data/AAPL.csv'

# Đọc CSV, bỏ qua 2 hàng đầu (Ticker và Date NaN)
df = pd.read_csv(file_path, skiprows=2)

# Đặt tên cột thủ công (dựa trên cấu trúc file)
df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

# Chuyển Date thành datetime
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')  # Format ngày tháng kiểu YYYY-MM-DD

# Chuẩn bị dữ liệu cho Prophet: cần cột 'ds' và 'y'
prophet_df = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

# Log transform để cải thiện volatility
prophet_df['y'] = np.log(prophet_df['y'])  # Log y để ổn định data

# Đảm bảo index có tần suất
prophet_df.set_index('ds', inplace=True)
prophet_df = prophet_df.asfreq('B')  # B = Business day
prophet_df['y'] = prophet_df['y'].ffill()
prophet_df.reset_index(inplace=True)  # Reset index

# Chia train/test (90% train, 10% test)
split_ratio = 0.90
split_index = int(len(prophet_df) * split_ratio)
train = prophet_df[:split_index]
test = prophet_df[split_index:]

# Fit model Prophet improved
model_prophet = Prophet(
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=True,
    seasonality_mode='multiplicative',  # Multiplicative cho stock
    changepoint_prior_scale=0.5,  # Tăng để flexible hơn
    seasonality_prior_scale=10.0  # Tune seasonality
)

# Thêm US holidays để capture events
model_prophet.add_country_holidays(country_name='US')

model_prophet.fit(train)

# Dự báo cho phần test
future = model_prophet.make_future_dataframe(periods=len(test), freq='B')
forecast = model_prophet.predict(future)
pred_prophet = np.exp(forecast['yhat'].tail(len(test)).values)  # Exp để reverse log

# Reverse log cho actual để tính metrics đúng
test_actual = np.exp(test['y'])

# Save model
joblib.dump(model_prophet, os.path.join(output_dir_models, 'prophet_improved_model.pkl'))

# Vẽ biểu đồ predicted vs actual
plt.figure(figsize=(14, 6))
plt.plot(test['ds'], test_actual, label='Actual', color='black', linewidth=2)
plt.plot(test['ds'], pred_prophet, label='Prophet Improved Forecast')
plt.title('Predicted vs Actual (Test Set) - Improved Prophet')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir_result, 'predicted_vs_actual_improved.png'))
plt.show()

# Vẽ residual plot
residuals = test_actual - pred_prophet
plt.figure(figsize=(10, 5))
plt.plot(test['ds'], residuals, label='Residuals (Prophet Improved)')
plt.axhline(0, color='red', linestyle='--')
plt.title('Residual Plot for Improved Prophet')
plt.xlabel('Date')
plt.ylabel('Residual')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir_result, 'residual_prophet_improved.png'))
plt.show()

# Tính metrics
def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = root_mean_squared_error(actual, predicted)
    mape = mean_absolute_percentage_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}

metrics_prophet = calculate_metrics(test_actual, pred_prophet)

# Tạo bảng metrics
metrics_df = pd.DataFrame(metrics_prophet, index=['Prophet_Improved']).T
metrics_df.to_csv(os.path.join(output_dir_result, 'metrics_comparison_improved.csv'), index=True)

# Tạo bảng predicted vs actual
pred_vs_actual_df = pd.DataFrame({
    'Actual': test_actual.values,
    'Prophet_Improved_Pred': pred_prophet
}, index=test['ds'])
pred_vs_actual_df.to_csv(os.path.join(output_dir_result, 'predicted_vs_actual_improved.csv'), index=True)

# In ra để kiểm tra
print("Metrics Comparison (Improved):")
print(metrics_df)
print("\nPredicted vs Actual saved to CSV.")