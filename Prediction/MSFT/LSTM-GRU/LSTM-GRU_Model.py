import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# Tạo thư mục lưu kết quả và model
output_dir = '../../../'  # This points to the parent folder
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

output_dir_result = os.path.join(output_dir, 'Result/MSFT/LSTM/')
if not os.path.exists(output_dir_result):
    os.makedirs(output_dir_result, exist_ok=True)

output_dir_models = os.path.join(output_dir, 'Models/MSFT/LSTM/')
if not os.path.exists(output_dir_models):
    os.makedirs(output_dir_models, exist_ok=True)

print(f"📁 Thư mục kết quả: {output_dir_result}")
print(f"📁 Thư mục models: {output_dir_models}")

file_path = '../../../Data/AAPL.csv'
# Đọc dữ liệu với skiprows=2 để bỏ qua header rows
df = pd.read_csv(file_path, skiprows=2)

# Kiểm tra và đổi tên cột nếu cần
print("Tên các cột trong file:")
print(df.columns.tolist())
print("\nVài dòng đầu của dữ liệu:")
print(df.head())

# Đảm bảo tên cột đúng (có thể cần điều chỉnh tùy theo file CSV thực tế)
# Giả sử cột đầu tiên là Date, cột thứ 2 là Close
if len(df.columns) >= 6:
    df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
else:
    print("⚠️ Số lượng cột không như mong đợi, vui lòng kiểm tra lại file CSV")

# Chuyển đổi cột Date thành datetime
try:
    df['Date'] = pd.to_datetime(df['Date'])
except:
    # Nếu cột Date có format đặc biệt, thử các format khác
    try:
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    except:
        print("⚠️ Không thể chuyển đổi cột Date, vui lòng kiểm tra format")
        # Tạo index date giả nếu không có Date
        df['Date'] = pd.date_range(start='1986-01-01', periods=len(df), freq='D')

df = df.sort_values('Date')

# Loại bỏ các giá trị NaN nếu có
df = df.dropna()

print(f"\n📊 Thông tin dữ liệu:")
print(f"Số lượng records: {len(df)}")
print(f"Khoảng thời gian: {df['Date'].min()} đến {df['Date'].max()}")
print(f"Giá Close min: ${df['Close'].min():.4f}")
print(f"Giá Close max: ${df['Close'].max():.4f}")

# Sử dụng giá đóng cửa để dự đoán
data = df['Close'].values.reshape(-1, 1)

# Chuẩn hóa dữ liệu
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Tạo sequences cho training
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Thiết lập parameters
SEQ_LENGTH = 60  # Sử dụng 60 ngày trước để dự đoán
TRAIN_SIZE = int(len(scaled_data) * 0.8)

# Chia dữ liệu train/test
train_data = scaled_data[:TRAIN_SIZE]
test_data = scaled_data[TRAIN_SIZE-SEQ_LENGTH:]

# Tạo sequences
X_train, y_train = create_sequences(train_data, SEQ_LENGTH)
X_test, y_test = create_sequences(test_data, SEQ_LENGTH)

# Reshape cho LSTM/GRU
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print(f"\n📈 Thông tin training:")
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")
print(f"Training period: {TRAIN_SIZE} records")
print(f"Testing period: {len(scaled_data) - TRAIN_SIZE} records")

# Xây dựng mô hình LSTM-GRU
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(SEQ_LENGTH, 1)),
    Dropout(0.2),
    
    GRU(50, return_sequences=True),
    Dropout(0.2),
    
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    
    Dense(25),
    Dense(1)
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Hiển thị kiến trúc mô hình
model.summary()

# Training model
print("\n🚀 Đang training mô hình...")
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=50,
    validation_split=0.1,
    verbose=1
)

# Lưu model
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = f"MSFT_LSTM_GRU_model_{timestamp}.keras"
model_path = os.path.join(output_dir_models, model_filename)
model.save(model_path)
print(f"✅ Model đã được lưu: {model_path}")

# Dự đoán
print("\n🔮 Đang thực hiện dự đoán...")
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Chuyển đổi về giá trị gốc
train_predictions = scaler.inverse_transform(train_predictions)
test_predictions = scaler.inverse_transform(test_predictions)
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Tính toán metrics
def calculate_metrics(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(actual, predicted) * 100
    r2 = r2_score(actual, predicted)
    return mse, rmse, mape, r2

# Metrics cho training set
train_mse, train_rmse, train_mape, train_r2 = calculate_metrics(y_train_actual, train_predictions)

# Metrics cho test set
test_mse, test_rmse, test_mape, test_r2 = calculate_metrics(y_test_actual, test_predictions)

# In kết quả metrics
print("\n" + "="*60)
print("KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH LSTM-GRU")
print("="*60)

print("\n📊 TRAINING SET METRICS:")
print(f"MSE:        {train_mse:.6f}")
print(f"RMSE:       {train_rmse:.6f}")
print(f"MAPE:       {train_mape:.2f}%")
print(f"R²-Score:   {train_r2:.4f}")

print("\n📊 TEST SET METRICS:")
print(f"MSE:        {test_mse:.6f}")
print(f"RMSE:       {test_rmse:.6f}")
print(f"MAPE:       {test_mape:.2f}%")
print(f"R²-Score:   {test_r2:.4f}")

# Tạo dataframe để hiển thị kết quả
results_df = pd.DataFrame({
    'Metric': ['MSE', 'RMSE', 'MAPE (%)', 'R²-Score'],
    'Training': [train_mse, train_rmse, train_mape, train_r2],
    'Testing': [test_mse, test_rmse, test_mape, test_r2]
})

print("\n📋 BẢNG TỔNG HỢP METRICS:")
print(results_df.round(6))

# Lưu metrics vào CSV
metrics_filename = f"MSFT_LSTM_GRU_metrics_{timestamp}.csv"
metrics_path = os.path.join(output_dir_result, metrics_filename)
results_df.to_csv(metrics_path, index=False)
print(f"✅ Metrics đã được lưu: {metrics_path}")

# Vẽ biểu đồ kết quả
plt.figure(figsize=(15, 10))

# Biểu đồ 1: Training Loss
plt.subplot(2, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Biểu đồ 2: Dự đoán vs Thực tế (Training)
plt.subplot(2, 2, 2)
plt.scatter(y_train_actual, train_predictions, alpha=0.5)
plt.plot([y_train_actual.min(), y_train_actual.max()], 
         [y_train_actual.min(), y_train_actual.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title(f'Training Set: Actual vs Predicted\nR² = {train_r2:.4f}')
plt.grid(True)

# Biểu đồ 3: Dự đoán vs Thực tế (Test)
plt.subplot(2, 2, 3)
plt.scatter(y_test_actual, test_predictions, alpha=0.5, color='orange')
plt.plot([y_test_actual.min(), y_test_actual.max()], 
         [y_test_actual.min(), y_test_actual.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title(f'Test Set: Actual vs Predicted\nR² = {test_r2:.4f}')
plt.grid(True)

# Biểu đồ 4: Time series comparison
plt.subplot(2, 2, 4)
test_dates = df['Date'].iloc[TRAIN_SIZE:TRAIN_SIZE+len(y_test_actual)]
plt.plot(test_dates, y_test_actual, label='Actual', linewidth=2)
plt.plot(test_dates, test_predictions, label='Predicted', linewidth=2)
plt.title('MSFT Stock Price Prediction (Test Period)')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)

plt.tight_layout()

# Lưu biểu đồ chính
main_plot_filename = f"MSFT_LSTM_GRU_main_results_{timestamp}.png"
main_plot_path = os.path.join(output_dir_result, main_plot_filename)
plt.savefig(main_plot_path, dpi=300, bbox_inches='tight')
print(f"✅ Biểu đồ chính đã được lưu: {main_plot_path}")
plt.show()

# Dự đoán cho 30 ngày tiếp theo
def predict_future(model, last_sequence, days_ahead, scaler):
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(days_ahead):
        # Dự đoán ngày tiếp theo
        next_pred = model.predict(current_sequence.reshape(1, SEQ_LENGTH, 1), verbose=0)
        predictions.append(next_pred[0, 0])
        
        # Cập nhật sequence
        current_sequence = np.append(current_sequence[1:], next_pred[0, 0])
    
    # Chuyển đổi về giá trị gốc
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions.flatten()

# Dự đoán 30 ngày tiếp theo
last_sequence = scaled_data[-SEQ_LENGTH:]
future_predictions = predict_future(model, last_sequence, 30, scaler)

print(f"\n🔮 DỰ ĐOÁN GIÁ MSFT CHO 30 NGÀY TIẾP THEO:")
print(f"Giá hiện tại: ${data[-1][0]:.4f}")
print(f"Giá dự đoán sau 30 ngày: ${future_predictions[-1]:.4f}")
print(f"Thay đổi dự kiến: {((future_predictions[-1] - data[-1][0]) / data[-1][0] * 100):+.2f}%")

# Tạo DataFrame cho dự đoán tương lai
future_dates = pd.date_range(start=df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=30)
future_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted_Price': future_predictions,
    'Current_Price': data[-1][0],
    'Change_Percent': ((future_predictions - data[-1][0]) / data[-1][0] * 100)
})

# Lưu dự đoán tương lai vào CSV
future_filename = f"MSFT_LSTM_GRU_future_predictions_{timestamp}.csv"
future_path = os.path.join(output_dir_result, future_filename)
future_df.to_csv(future_path, index=False)
print(f"✅ Dự đoán tương lai đã được lưu: {future_path}")

# Vẽ biểu đồ dự đoán tương lai
plt.figure(figsize=(12, 6))
last_30_days = df['Date'].tail(30)
last_30_prices = df['Close'].tail(30)

plt.plot(last_30_days, last_30_prices, label='Actual (Last 30 days)', linewidth=2)
plt.plot(future_dates, future_predictions, label='Predicted (Next 30 days)', 
         linewidth=2, linestyle='--', color='red')
plt.axvline(x=df['Date'].iloc[-1], color='gray', linestyle=':', alpha=0.7, label='Today')
plt.title('MSFT Stock Price: Last 30 Days vs Next 30 Days Prediction')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

# Lưu biểu đồ dự đoán tương lai
future_plot_filename = f"MSFT_LSTM_GRU_future_prediction_{timestamp}.png"
future_plot_path = os.path.join(output_dir_result, future_plot_filename)
plt.savefig(future_plot_path, dpi=300, bbox_inches='tight')
print(f"✅ Biểu đồ dự đoán tương lai đã được lưu: {future_plot_path}")
plt.show()

# Tạo DataFrame tổng hợp kết quả test
test_results_df = pd.DataFrame({
    'Date': df['Date'].iloc[TRAIN_SIZE:TRAIN_SIZE+len(y_test_actual)].values,
    'Actual_Price': y_test_actual.flatten(),
    'Predicted_Price': test_predictions.flatten(),
    'Absolute_Error': np.abs(y_test_actual.flatten() - test_predictions.flatten()),
    'Percentage_Error': np.abs((y_test_actual.flatten() - test_predictions.flatten()) / y_test_actual.flatten() * 100)
})

# Lưu kết quả test vào CSV
test_results_filename = f"MSFT_LSTM_GRU_test_results_{timestamp}.csv"
test_results_path = os.path.join(output_dir_result, test_results_filename)
test_results_df.to_csv(test_results_path, index=False)
print(f"✅ Kết quả test đã được lưu: {test_results_path}")

# Lưu thông tin training history
history_df = pd.DataFrame({
    'Epoch': range(1, len(history.history['loss']) + 1),
    'Training_Loss': history.history['loss'],
    'Validation_Loss': history.history['val_loss']
})

history_filename = f"MSFT_LSTM_GRU_training_history_{timestamp}.csv"
history_path = os.path.join(output_dir_result, history_filename)
history_df.to_csv(history_path, index=False)
print(f"✅ Lịch sử training đã được lưu: {history_path}")

print(f"\n🎉 HOÀN THÀNH! Tất cả kết quả đã được lưu vào:")
print(f"📁 Models: {output_dir_models}")
print(f"📁 Results: {output_dir_result}")
print(f"\n📋 Các file đã tạo:")
print(f"   • Model: {model_filename}")
print(f"   • Metrics: {metrics_filename}")
print(f"   • Main plot: {main_plot_filename}")
print(f"   • Future prediction plot: {future_plot_filename}")
print(f"   • Future predictions: {future_filename}")
print(f"   • Test results: {test_results_filename}")
print(f"   • Training history: {history_filename}")
