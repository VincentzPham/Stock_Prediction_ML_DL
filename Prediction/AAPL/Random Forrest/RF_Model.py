# # # # import os
# # # # import pickle
# # # # import numpy as np
# # # # import pandas as pd
# # # # import matplotlib.pyplot as plt
# # # # from sklearn.ensemble import RandomForestRegressor
# # # # from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

# # # # # Tạo thư mục lưu kết quả và model
# # # # output_dir = '../../../'
# # # # output_dir_result = os.path.join(output_dir, 'Result/AAPL/Random Forest/')
# # # # if not os.path.exists(output_dir_result):
# # # #     os.makedirs(output_dir_result, exist_ok=True)
# # # # output_dir_models = os.path.join(output_dir, 'Models/AAPL/Random Forest/')
# # # # if not os.path.exists(output_dir_models):
# # # #     os.makedirs(output_dir_models, exist_ok=True)

# # # # # Đọc dữ liệu và tiền xử lý (giống MLR)
# # # # file_path = '../../../Data/AAPL.csv'
# # # # df = pd.read_csv(file_path, skiprows=2)
# # # # df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
# # # # df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
# # # # df.set_index('Date', inplace=True)
# # # # # df = df.asfreq('B')
# # # # # df = df.ffill()

# # # # features = ['Close','Open', 'High', 'Low', 'Volume']
# # # # X = df[features]
# # # # y = df['Close']

# # # # split_index = int(len(df) * 0.8)
# # # # X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
# # # # y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# # # # # Huấn luyện mô hình Random Forest
# # # # rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# # # # rf_model.fit(X_train, y_train)

# # # # # Dự đoán
# # # # y_pred = rf_model.predict(X_test)

# # # # # --- 1. Save model dưới dạng .pkl ---
# # # # model_path = os.path.join(output_dir_models, 'rf_model.pkl')
# # # # with open(model_path, 'wb') as f:
# # # #     pickle.dump(rf_model, f)

# # # # # --- 2. Save scatter plot & residual scatter plot .png ---
# # # # result_df = pd.DataFrame({
# # # #     'Date': X_test.index,
# # # #     'Actual': y_test.values,
# # # #     'Predicted': y_pred
# # # # })

# # # # # Scatter plot: Actual vs Predicted
# # # # plt.figure(figsize=(8,8))
# # # # plt.scatter(result_df['Actual'], result_df['Predicted'], color='blue', alpha=0.5, s=10)
# # # # plt.plot([result_df['Actual'].min(), result_df['Actual'].max()],
# # # #          [result_df['Actual'].min(), result_df['Actual'].max()],
# # # #          'r--', lw=2, label='Ideal Fit (y=x)')
# # # # plt.xlabel('Actual Value')
# # # # plt.ylabel('Predicted Value')
# # # # plt.title('Random Forest: Actual vs Predicted')
# # # # plt.legend()
# # # # plt.tight_layout()
# # # # scatter_plot_path = os.path.join(output_dir_result, 'scatter_actual_vs_predicted.png')
# # # # plt.savefig(scatter_plot_path)
# # # # plt.close()

# # # # # Residual scatter plot
# # # # residuals = result_df['Actual'] - result_df['Predicted']
# # # # plt.figure(figsize=(8,5))
# # # # plt.scatter(result_df['Actual'], residuals, color='purple', s=10, alpha=0.5)
# # # # plt.axhline(0, color='red', linestyle='--')
# # # # plt.xlabel('Actual Value')
# # # # plt.ylabel('Residual (Actual - Predicted)')
# # # # plt.title('Random Forest: Residual Scatter Plot')
# # # # plt.tight_layout()
# # # # residual_scatter_plot_path = os.path.join(output_dir_result, 'residual_scatter_plot.png')
# # # # plt.savefig(residual_scatter_plot_path)
# # # # plt.close()

# # # # # --- 3. Save metrics & predicted/actual .csv ---
# # # # mse = mean_squared_error(y_test, y_pred)
# # # # rmse = np.sqrt(mse)
# # # # mae = mean_absolute_error(y_test, y_pred)
# # # # mape = mean_absolute_percentage_error(y_test, y_pred)
# # # # r2 = r2_score(y_test, y_pred)

# # # # metrics_df = pd.DataFrame([{
# # # #     'RMSE': rmse,
# # # #     'MAE': mae,
# # # #     'MSE': mse,
# # # #     'MAPE': mape,
# # # #     'R2 Score': r2
# # # # }])
# # # # metrics_csv_path = os.path.join(output_dir_result, 'rf_metrics.csv')
# # # # metrics_df.to_csv(metrics_csv_path, index=False)

# # # # predicted_csv_path = os.path.join(output_dir_result, 'actual_vs_predicted.csv')
# # # # result_df.to_csv(predicted_csv_path, index=False)

# # # # # Thông báo
# # # # print('Đã lưu model tại:', model_path)
# # # # print('Đã lưu scatter plot Actual vs Predicted:', scatter_plot_path)
# # # # print('Đã lưu residual scatter plot:', residual_scatter_plot_path)
# # # # print('Đã lưu bảng metrics:', metrics_csv_path)
# # # # print('Đã lưu bảng actual vs predicted:', predicted_csv_path)



# # # import pandas as pd
# # # import matplotlib.pyplot as plt
# # # from sklearn.model_selection import train_test_split
# # # from sklearn.linear_model import LinearRegression
# # # from sklearn.ensemble import RandomForestRegressor
# # # from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

# # # # Đọc và xử lý dữ liệu
# # # file_path = '../../../Data/AAPL.csv'
# # # df = pd.read_csv(file_path)

# # # # Bỏ 2 dòng đầu, reset lại index
# # # df_clean = df.iloc[2:].reset_index(drop=True)

# # # # Đổi tên cột đầu tiên thành Date
# # # df_clean = df_clean.rename(columns={'Price': 'Date'})

# # # # Ép kiểu dữ liệu cho các cột (trừ Date)
# # # for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
# # #     df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# # # # Bỏ các dòng có NaN
# # # df_clean = df_clean.dropna()

# # # # Chọn features và target
# # # X = df_clean[['Open', 'High', 'Low', 'Volume']]
# # # y = df_clean['Close']

# # # # Chia train/test (80/20)
# # # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # # # Train Multiple Linear Regression
# # # model = LinearRegression()
# # # model.fit(X_train, y_train)
# # # y_pred = model.predict(X_test)

# # # # Đánh giá Linear Regression
# # # r2 = r2_score(y_test, y_pred)
# # # mse = mean_squared_error(y_test, y_pred)
# # # coefs = model.coef_
# # # intercept = model.intercept_

# # # # Train Random Forest Regression
# # # rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# # # rf_model.fit(X_train, y_train)
# # # y_rf_pred = rf_model.predict(X_test)

# # # # Đánh giá Random Forest Regression
# # # r2_rf = r2_score(y_test, y_rf_pred)
# # # mse_rf = mean_squared_error(y_test, y_rf_pred)
# # # mape_rf = mean_absolute_percentage_error(y_test, y_rf_pred)

# # # # Biểu đồ Actual vs Predicted
# # # plt.figure(figsize=(12,5))
# # # plt.subplot(1,2,1)
# # # plt.scatter(y_test, y_pred, alpha=0.5)
# # # plt.xlabel('Actual Close')
# # # plt.ylabel('Predicted Close')
# # # plt.title('Linear Regression: Actual vs Predicted')
# # # plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')

# # # plt.subplot(1,2,2)
# # # plt.scatter(y_test, y_rf_pred, alpha=0.5)
# # # plt.xlabel('Actual Close')
# # # plt.ylabel('Predicted Close')
# # # plt.title('Random Forest: Actual vs Predicted')
# # # plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
# # # plt.tight_layout()
# # # plt.show()

# # # # Residual plots
# # # plt.figure(figsize=(12,5))
# # # plt.subplot(1,2,1)
# # # plt.scatter(y_test, y_pred - y_test, alpha=0.5)
# # # plt.xlabel('Actual Close')
# # # plt.ylabel('Residual (Predicted - Actual)')
# # # plt.title('Linear Regression: Residual Plot')
# # # plt.axhline(0, color='red', linestyle='--')

# # # plt.subplot(1,2,2)
# # # plt.scatter(y_test, y_rf_pred - y_test, alpha=0.5)
# # # plt.xlabel('Actual Close')
# # # plt.ylabel('Residual (Predicted - Actual)')
# # # plt.title('Random Forest: Residual Plot')
# # # plt.axhline(0, color='red', linestyle='--')
# # # plt.tight_layout()
# # # plt.show()



# # import os
# # import pickle
# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # from sklearn.ensemble import RandomForestRegressor
# # from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
# # from sklearn.model_selection import train_test_split

# # # Tạo thư mục lưu kết quả và model
# # output_dir = '../../../'
# # output_dir_result = os.path.join(output_dir, 'Result/AAPL/Random Forest/')
# # if not os.path.exists(output_dir_result):
# #     os.makedirs(output_dir_result, exist_ok=True)
# # output_dir_models = os.path.join(output_dir, 'Models/AAPL/Random Forest/')
# # if not os.path.exists(output_dir_models):
# #     os.makedirs(output_dir_models, exist_ok=True)

# # # Đọc dữ liệu và tiền xử lý (giống MLR)
# # file_path = '../../../Data/AAPL.csv'
# # df = pd.read_csv(file_path, skiprows=2)
# # df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
# # df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
# # df.set_index('Date', inplace=True)

# # # Filter start date từ 2010-01-01
# # start_date = '2010-01-01'
# # df = df[df.index >= start_date]


# # # --- Cập nhật quan trọng: KHÔNG dùng 'Close' làm feature ---
# # features = ['Open', 'High', 'Low', 'Volume']   # KHÔNG còn 'Close' trong X
# # X = df[features]
# # y = df['Close']

# # split_index = int(len(df) * 0.8)
# # X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
# # y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# # # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # # Huấn luyện mô hình Random Forest
# # rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# # rf_model.fit(X_train, y_train)

# # # Dự đoán
# # y_pred = rf_model.predict(X_test)

# # # --- 1. Save model dưới dạng .pkl ---
# # model_path = os.path.join(output_dir_models, 'rf_model.pkl')
# # with open(model_path, 'wb') as f:
# #     pickle.dump(rf_model, f)

# # # --- 2. Save scatter plot & residual scatter plot .png ---
# # result_df = pd.DataFrame({
# #     'Date': X_test.index,
# #     'Actual': y_test.values,
# #     'Predicted': y_pred
# # })

# # # Scatter plot: Actual vs Predicted
# # plt.figure(figsize=(8,8))
# # plt.scatter(result_df['Actual'], result_df['Predicted'], color='blue', alpha=0.5, s=10)
# # plt.plot([result_df['Actual'].min(), result_df['Actual'].max()],
# #          [result_df['Actual'].min(), result_df['Actual'].max()],
# #          'r--', lw=2, label='Ideal Fit (y=x)')
# # plt.xlabel('Actual Value')
# # plt.ylabel('Predicted Value')
# # plt.title('Random Forest: Actual vs Predicted')
# # plt.legend()
# # plt.tight_layout()
# # scatter_plot_path = os.path.join(output_dir_result, 'scatter_actual_vs_predicted.png')
# # plt.savefig(scatter_plot_path)
# # plt.close()

# # # Residual scatter plot
# # residuals = result_df['Actual'] - result_df['Predicted']
# # plt.figure(figsize=(8,5))
# # plt.scatter(result_df['Actual'], residuals, color='purple', s=10, alpha=0.5)
# # plt.axhline(0, color='red', linestyle='--')
# # plt.xlabel('Actual Value')
# # plt.ylabel('Residual (Actual - Predicted)')
# # plt.title('Random Forest: Residual Scatter Plot')
# # plt.tight_layout()
# # residual_scatter_plot_path = os.path.join(output_dir_result, 'residual_scatter_plot.png')
# # plt.savefig(residual_scatter_plot_path)
# # plt.close()

# # # --- 3. Save metrics & predicted/actual .csv ---
# # mse = mean_squared_error(y_test, y_pred)
# # rmse = np.sqrt(mse)
# # mae = mean_absolute_error(y_test, y_pred)
# # mape = mean_absolute_percentage_error(y_test, y_pred)
# # r2 = r2_score(y_test, y_pred)

# # metrics_df = pd.DataFrame([{
# #     'RMSE': rmse,
# #     'MAE': mae,
# #     'MSE': mse,
# #     'MAPE': mape,
# #     'R2 Score': r2
# # }])
# # metrics_csv_path = os.path.join(output_dir_result, 'rf_metrics.csv')
# # metrics_df.to_csv(metrics_csv_path, index=False)

# # predicted_csv_path = os.path.join(output_dir_result, 'actual_vs_predicted.csv')
# # result_df.to_csv(predicted_csv_path, index=False)

# # # Thông báo
# # print('Đã lưu model tại:', model_path)
# # print('Đã lưu scatter plot Actual vs Predicted:', scatter_plot_path)
# # print('Đã lưu residual scatter plot:', residual_scatter_plot_path)
# # print('Đã lưu bảng metrics:', metrics_csv_path)
# # print('Đã lưu bảng actual vs predicted:', predicted_csv_path)



# import os
# import pickle
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

# # Tạo thư mục lưu kết quả và model
# output_dir = '../../../'
# output_dir_result = os.path.join(output_dir, 'Result/AAPL/Random Forest/')
# if not os.path.exists(output_dir_result):
#     os.makedirs(output_dir_result, exist_ok=True)
# output_dir_models = os.path.join(output_dir, 'Models/AAPL/Random Forest/')
# if not os.path.exists(output_dir_models):
#     os.makedirs(output_dir_models, exist_ok=True)

# # Đọc dữ liệu và tiền xử lý
# file_path = '../../../Data/AAPL.csv'
# df = pd.read_csv(file_path, skiprows=2)
# df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
# df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
# df.set_index('Date', inplace=True)

# # Filter start date từ 2010-01-01
# start_date = '2010-01-01'
# df = df[df.index >= start_date]

# # --- Tạo lag feature ---
# lags = 1  # Số ngày trễ, bạn có thể tăng lên (2, 3, ...)
# for col in ['Close', 'Open', 'High', 'Low', 'Volume']:
#     df[f'{col}_lag{lags}'] = df[col].shift(lags)

# # Bỏ các dòng bị NaN do tạo lag
# df_lag = df.dropna()

# # Feature dùng để train: chỉ dùng lag của các biến
# features = [f'{col}_lag{lags}' for col in ['Close', 'Open', 'High', 'Low', 'Volume']]
# X = df_lag[features]
# y = df_lag['Close']

# split_index = int(len(df_lag) * 0.8)
# X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
# y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# # Huấn luyện mô hình Random Forest
# rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_model.fit(X_train, y_train)

# # Dự đoán
# y_pred = rf_model.predict(X_test)

# # --- 1. Save model dưới dạng .pkl ---
# model_path = os.path.join(output_dir_models, 'rf_model.pkl')
# with open(model_path, 'wb') as f:
#     pickle.dump(rf_model, f)

# # --- 2. Save scatter plot & residual scatter plot .png ---
# result_df = pd.DataFrame({
#     'Date': X_test.index,
#     'Actual': y_test.values,
#     'Predicted': y_pred
# })

# plt.figure(figsize=(8,8))
# plt.scatter(result_df['Actual'], result_df['Predicted'], color='blue', alpha=0.5, s=10)
# plt.plot([result_df['Actual'].min(), result_df['Actual'].max()],
#          [result_df['Actual'].min(), result_df['Actual'].max()],
#          'r--', lw=2, label='Ideal Fit (y=x)')
# plt.xlabel('Actual Value')
# plt.ylabel('Predicted Value')
# plt.title('Random Forest with Lag: Actual vs Predicted')
# plt.legend()
# plt.tight_layout()
# scatter_plot_path = os.path.join(output_dir_result, 'scatter_actual_vs_predicted.png')
# plt.savefig(scatter_plot_path)
# plt.close()

# # Residual scatter plot
# residuals = result_df['Actual'] - result_df['Predicted']
# plt.figure(figsize=(8,5))
# plt.scatter(result_df['Actual'], residuals, color='purple', s=10, alpha=0.5)
# plt.axhline(0, color='red', linestyle='--')
# plt.xlabel('Actual Value')
# plt.ylabel('Residual (Actual - Predicted)')
# plt.title('Random Forest with Lag: Residual Scatter Plot')
# plt.tight_layout()
# residual_scatter_plot_path = os.path.join(output_dir_result, 'residual_scatter_plot.png')
# plt.savefig(residual_scatter_plot_path)
# plt.close()

# # --- 3. Save metrics & predicted/actual .csv ---
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# mae = mean_absolute_error(y_test, y_pred)
# mape = mean_absolute_percentage_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# metrics_df = pd.DataFrame([{
#     'RMSE': rmse,
#     'MAE': mae,
#     'MSE': mse,
#     'MAPE': mape,
#     'R2 Score': r2
# }])
# metrics_csv_path = os.path.join(output_dir_result, 'rf_metrics.csv')
# metrics_df.to_csv(metrics_csv_path, index=False)

# predicted_csv_path = os.path.join(output_dir_result, 'actual_vs_predicted.csv')
# result_df.to_csv(predicted_csv_path, index=False)

# # Thông báo
# print('Đã lưu model tại:', model_path)
# print('Đã lưu scatter plot Actual vs Predicted:', scatter_plot_path)
# print('Đã lưu residual scatter plot:', residual_scatter_plot_path)
# print('Đã lưu bảng metrics:', metrics_csv_path)
# print('Đã lưu bảng actual vs predicted:', predicted_csv_path)





import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

# Tạo thư mục lưu kết quả và model
output_dir = '../../../'
output_dir_result = os.path.join(output_dir, 'Result/AAPL/Random Forest/')
if not os.path.exists(output_dir_result):
    os.makedirs(output_dir_result, exist_ok=True)
output_dir_models = os.path.join(output_dir, 'Models/AAPL/Random Forest/')
if not os.path.exists(output_dir_models):
    os.makedirs(output_dir_models, exist_ok=True)

# Đọc dữ liệu và tiền xử lý
file_path = '../../../Data/AAPL.csv'
df = pd.read_csv(file_path, skiprows=2)
df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
df.set_index('Date', inplace=True)

# Filter start date từ 2010-01-01
start_date = '2010-01-01'
df = df[df.index >= start_date]

# --- Tạo nhiều lag feature ---
lags = [1,2,3,4,5]
for lag in lags:
    for col in ['Close', 'Open', 'High', 'Low', 'Volume']:
        df[f'{col}_lag{lag}'] = df[col].shift(lag)

# Thêm các chỉ báo kỹ thuật đơn giản
df['MA5'] = df['Close'].rolling(window=5).mean()
df['MA10'] = df['Close'].rolling(window=10).mean()
df['STD5'] = df['Close'].rolling(window=5).std()

# Bỏ các dòng bị NaN do tạo lag/rolling
df_feat = df.dropna()

# Tập feature
feature_cols = []
for lag in lags:
    feature_cols += [f'{col}_lag{lag}' for col in ['Close', 'Open', 'High', 'Low', 'Volume']]
feature_cols += ['MA5', 'MA10', 'STD5']

X = df_feat[feature_cols]
y = df_feat['Close']

split_index = int(len(df_feat) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# Huấn luyện mô hình Random Forest
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

# Dự đoán
y_pred = rf_model.predict(X_test)

# --- 1. Save model dưới dạng .pkl ---
model_path = os.path.join(output_dir_models, 'rf_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(rf_model, f)

# --- 2. Save scatter plot & residual scatter plot .png ---
result_df = pd.DataFrame({
    'Date': X_test.index,
    'Actual': y_test.values,
    'Predicted': y_pred
})

plt.figure(figsize=(8,8))
plt.scatter(result_df['Actual'], result_df['Predicted'], color='blue', alpha=0.5, s=10)
plt.plot([result_df['Actual'].min(), result_df['Actual'].max()],
         [result_df['Actual'].min(), result_df['Actual'].max()],
         'r--', lw=2, label='Ideal Fit (y=x)')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.title('Random Forest with Advanced Lag: Actual vs Predicted')
plt.legend()
plt.tight_layout()
scatter_plot_path = os.path.join(output_dir_result, 'scatter_actual_vs_predicted.png')
plt.savefig(scatter_plot_path)
plt.close()

# Residual scatter plot
residuals = result_df['Actual'] - result_df['Predicted']
plt.figure(figsize=(8,5))
plt.scatter(result_df['Actual'], residuals, color='purple', s=10, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Actual Value')
plt.ylabel('Residual (Actual - Predicted)')
plt.title('Random Forest with Advanced Lag: Residual Scatter Plot')
plt.tight_layout()
residual_scatter_plot_path = os.path.join(output_dir_result, 'residual_scatter_plot.png')
plt.savefig(residual_scatter_plot_path)
plt.close()

# --- 3. Save metrics & predicted/actual .csv ---
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
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
metrics_csv_path = os.path.join(output_dir_result, 'rf_metrics.csv')
metrics_df.to_csv(metrics_csv_path, index=False)

predicted_csv_path = os.path.join(output_dir_result, 'actual_vs_predicted.csv')
result_df.to_csv(predicted_csv_path, index=False)

# Thông báo
print('Đã lưu model tại:', model_path)
print('Đã lưu scatter plot Actual vs Predicted:', scatter_plot_path)
print('Đã lưu residual scatter plot:', residual_scatter_plot_path)
print('Đã lưu bảng metrics:', metrics_csv_path)
print('Đã lưu bảng actual vs predicted:', predicted_csv_path)
