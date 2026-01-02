import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

# Tạo thư mục lưu kết quả và model
output_dir = '../../../'  # This points to the parent folder
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

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

# Hàm tạo dữ liệu đầu vào X và đầu ra y cho mô hình (dùng cho LSTM-GRU, ANN sẽ reshape sau)
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Tạo dữ liệu cho tập huấn luyện và kiểm tra
time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape X để phù hợp với đầu vào của LSTM-GRU (samples, time_steps, features)
X_train_lstm = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_lstm = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Đối với ANN, reshape thành 2D (samples, features)
X_train_ann = X_train.reshape(X_train.shape[0], -1)
X_test_ann = X_test.reshape(X_test.shape[0], -1)

# Hàm chung để huấn luyện, dự đoán và lưu kết quả cho từng mô hình
def train_and_evaluate_model(model_name, model, X_train, y_train, X_test, y_test):
    # Tạo thư mục cho mô hình
    output_dir_result = os.path.join(output_dir, f'Result/AAPL/{model_name}/')
    output_dir_models = os.path.join(output_dir, f'Models/AAPL/{model_name}/')
    os.makedirs(output_dir_result, exist_ok=True)
    os.makedirs(output_dir_models, exist_ok=True)

    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Huấn luyện với EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

    # Lưu mô hình
    model.save(os.path.join(output_dir_models, f'{model_name}_model.keras'))

    # Dự đoán và chuyển về không gian gốc
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

    # Vẽ kết quả (Actual vs Predicted)
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, scaler.inverse_transform(scaled_data), label='Actual Price', color='blue')
    plt.plot(df.index, train_predict_plot, label='Predicted Train Price', color='red')
    plt.plot(df.index, test_predict_plot, label='Predicted Test Price', color='green')
    plt.title(f'Apple Stock Price Prediction ({model_name})', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.savefig(os.path.join(output_dir_result, 'result_visualization.png'))
    plt.close()

    # Vẽ residuals
    residuals = y_test_actual - test_predict
    plt.figure(figsize=(12, 6))
    plt.plot(df.index[train_size + time_step:], residuals, label='Residuals')
    plt.title(f'Residuals of Apple Stock Price Prediction ({model_name})', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Residuals')
    plt.legend()
    plt.savefig(os.path.join(output_dir_result, 'residuals.png'))
    plt.close()

    # Vẽ training/validation loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Nếu có accuracy (dù cho regression thường không có, nhưng kiểm tra)
    if 'accuracy' in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_result, 'training_validation_loss_accuracy.png'))
    plt.close()

    # Lưu bảng so sánh actual vs predicted
    comparison_df = pd.DataFrame({'Date': df.index[train_size + time_step:], 
                                  'Actual': y_test_actual.flatten(), 
                                  'Predicted': test_predict.flatten()})
    comparison_df.to_csv(os.path.join(output_dir_result, 'predicted_vs_actual.csv'), index=False)

    # Tính và lưu metrics
    train_mse = mean_squared_error(y_train_actual, train_predict)
    test_mse = mean_squared_error(y_test_actual, test_predict)
    train_mae = mean_absolute_error(y_train_actual, train_predict)
    test_mae = mean_absolute_error(y_test_actual, test_predict)
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    train_mape = mean_absolute_percentage_error(y_train_actual, train_predict)
    test_mape = mean_absolute_percentage_error(y_test_actual, test_predict)
    train_r2 = r2_score(y_train_actual, train_predict)
    test_r2 = r2_score(y_test_actual, test_predict)

    metrics_df = pd.DataFrame({
        'Dataset': ['Train', 'Test'],
        'MSE': [train_mse, test_mse],
        'MAE': [train_mae, test_mae],
        'RMSE': [train_rmse, test_rmse],
        'MAPE': [train_mape, test_mape],
        'R2 Score': [train_r2, test_r2]
    })
    metrics_df.to_csv(os.path.join(output_dir_result, 'metrics_comparison.csv'), index=False)

    print(f"{model_name} training complete. Results and model saved.")

# Bước 3: Xây dựng và huấn luyện mô hình ANN
ann_model = Sequential()
ann_model.add(Dense(units=50, activation='relu', input_shape=(X_train_ann.shape[1],)))
ann_model.add(Dense(units=50, activation='relu'))
ann_model.add(Dense(units=1))  # Output layer cho regression
train_and_evaluate_model('ANN', ann_model, X_train_ann, y_train, X_test_ann, y_test)

# Bước 4: Xây dựng và huấn luyện mô hình LSTM-GRU
lstm_gru_model = Sequential()
lstm_gru_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)))
lstm_gru_model.add(GRU(units=50, return_sequences=False))
lstm_gru_model.add(Dense(units=1))  # Output layer cho regression
train_and_evaluate_model('LSTM-GRU', lstm_gru_model, X_train_lstm, y_train, X_test_lstm, y_test)