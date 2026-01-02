import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

# Tạo thư mục lưu kết quả và model
output_dir = '../../../'  # This points to the parent folder
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

output_dir_result = os.path.join(output_dir, 'Result/AAPL/BiLSTM/')
if not os.path.exists(output_dir_result):
    os.makedirs(output_dir_result, exist_ok=True)

output_dir_models = os.path.join(output_dir, 'Models/AAPL/BiLSTM/')
if not os.path.exists(output_dir_models):
    os.makedirs(output_dir_models, exist_ok=True)

# Đường dẫn tới tệp CSV
file_path = '../../../Data/AAPL.csv'

# Đọc CSV, bỏ qua 2 hàng đầu (Ticker và Date NaN)
df = pd.read_csv(file_path, skiprows=2)

# Đặt tên cột thủ công (dựa trên cấu trúc file)
df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

# Chuyển Date thành datetime và set làm index
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
df.set_index('Date', inplace=True)

# Kiểm tra DataFrame sau xử lý (tùy chọn)
print(df.head())
print(df.tail())
df.info()

# Bước 1: Chuẩn hóa dữ liệu
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

# Bước 2: Chia dữ liệu thành tập huấn luyện và kiểm tra
train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

# Hàm tạo dữ liệu đầu vào X và đầu ra y cho BiLSTM
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Tạo dữ liệu cho tập huấn luyện và kiểm tra
time_step = 60  # Sử dụng 60 ngày trước để dự đoán giá trị tiếp theo
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape X để phù hợp với đầu vào của BiLSTM (samples, time_steps, features)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Bước 3: Xây dựng mô hình BiLSTM
model = Sequential()

# BiLSTM layer (Bidirectional LSTM)
model.add(Bidirectional(LSTM(units=50, return_sequences=True), input_shape=(X_train.shape[1], 1)))
model.add(Bidirectional(LSTM(units=50, return_sequences=False)))

# Fully connected layer
model.add(Dense(units=1))

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# Bước 4: Huấn luyện mô hình với EarlyStopping để tránh overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Bước 5: Lưu mô hình
model.save(os.path.join(output_dir_models, 'BiLSTM_model.keras'))

# Bước 6: Dự đoán và chuyển dữ liệu trở lại không gian gốc
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = scaler.inverse_transform(train_predict)
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
test_predict = scaler.inverse_transform(test_predict)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Đảm bảo index đúng cho x-axis khi vẽ đồ thị
train_predict_plot = np.empty_like(scaled_data)
train_predict_plot[:, :] = np.nan
train_predict_plot[time_step:len(train_predict) + time_step, :] = train_predict

test_predict_plot = np.empty_like(scaled_data)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict) + (time_step * 2):, :] = test_predict

# Bước 7: Vẽ kết quả (Actual vs Predicted)
plt.figure(figsize=(12, 6))
plt.plot(df.index, scaler.inverse_transform(scaled_data), label='Actual Price', color='blue')
plt.plot(df.index, train_predict_plot, label='Predicted Train Price', color='red')
plt.plot(df.index, test_predict_plot, label='Predicted Test Price', color='green')
plt.title('Apple Stock Price Prediction (BiLSTM)', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.savefig(os.path.join(output_dir_result, 'result_visualization.png'))
plt.close()

# Bước 8: Vẽ residuals (Sai số dự đoán)
residuals = y_test_actual - test_predict
plt.figure(figsize=(12, 6))
plt.plot(df.index[train_size + time_step:], residuals, label='Residuals')
plt.title('Residuals of Apple Stock Price Prediction (BiLSTM)', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.legend()
plt.savefig(os.path.join(output_dir_result, 'residuals.png'))
plt.close()

# Bước 9: Vẽ biểu đồ training/validation loss và accuracy
plt.figure(figsize=(12, 6))

# Vẽ loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(output_dir_result, 'training_validation_loss.png'))

# Vẽ accuracy (nếu có)
plt.subplot(1, 2, 2)
if 'accuracy' in history.history:
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
else:
    print("No accuracy data available in the history object.")

# Lưu đồ thị loss và accuracy vào thư mục kết quả
plt.tight_layout()
plt.savefig(os.path.join(output_dir_result, 'training_validation_loss_accuracy.png'))
plt.close()

# Bước 10: Lưu bảng so sánh giữa actual và predicted
comparison_df = pd.DataFrame({'Date': df.index[train_size + time_step:], 
                              'Actual': y_test_actual.flatten(), 
                              'Predicted': test_predict.flatten()})
comparison_df.to_csv(os.path.join(output_dir_result, 'predicted_vs_actual.csv'), index=False)

# Bước 11: Lưu bảng thông số MSE
train_mse = mean_squared_error(y_train_actual, train_predict)
test_mse = mean_squared_error(y_test_actual, test_predict)

mse_df = pd.DataFrame({'Dataset': ['Train', 'Test'], 
                       'MSE': [train_mse, test_mse]})
mse_df.to_csv(os.path.join(output_dir_result, 'mse_comparison.csv'), index=False)

# Bước 12: Tính toán và lưu các metrics khác
train_mae = mean_absolute_error(y_train_actual, train_predict)
test_mae = mean_absolute_error(y_test_actual, test_predict)

train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_predict))
test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predict))

train_mape = mean_absolute_percentage_error(y_train_actual, train_predict)
test_mape = mean_absolute_percentage_error(y_test_actual, test_predict)

train_r2 = r2_score(y_train_actual, train_predict)
test_r2 = r2_score(y_test_actual, test_predict)

# Lưu bảng thông số MAE, MAPE, RMSE và R2
metrics_df = pd.DataFrame({
    'Dataset': ['Train', 'Test'],
    'MAE': [train_mae, test_mae],
    'RMSE': [train_rmse, test_rmse],
    'MAPE': [train_mape, test_mape],
    'R2 Score': [train_r2, test_r2]
})

metrics_df.to_csv(os.path.join(output_dir_result, 'metrics_comparison.csv'), index=False)

print("Model training complete. Results and model saved.")
