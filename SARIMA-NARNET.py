import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from flask import Flask, render_template, request, jsonify
import io
import base64

app = Flask(__name__)

file_path = r"C:\Users\Intel\Desktop\sarima_narnet_model\data\new.csv"

#Data Loading and Preprocessing
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df
    print("Printed successfull")

#Exploratory Data Analysis
def perform_eda(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Price'])
    plt.title('Time Series of Agri-Horticulture Product Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()
    print("Printed successfull")

    decomposition = seasonal_decompose(df['Price'], model='additive', period=7)
    decomposition.plot()
    plt.tight_layout()
    plt.show()
   

#SARIMA Model Implementation
def implement_sarima(df, train_size):
    
    df = df.asfreq('D')
    train = df[:train_size]
    
    test = df[train_size:]
    
    model = SARIMAX(train['Price'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    
    results = model.fit(disp=False)
    
    forecast = results.get_forecast(steps=len(test))
    forecast_mean = forecast.predicted_mean
    rmse = np.sqrt(mean_squared_error(test['Price'], forecast_mean))

    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train['Price'], label='Training Data')
    #'''plt.plot(test.index, test['Price'], label='Test Data')'''
    plt.plot(test.index, forecast_mean, label='SARIMA Forecast')
    plt.title('SARIMA Model Forecast')
    plt.legend()
    plt.show()
    print("Printed successfull")

    return forecast, rmse

#NARNET Model Implementation
def implement_narnet(df, train_size, look_back=7):
    values = df['Price'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    def create_dataset(dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset) - look_back):
            X.append(dataset[i:(i + look_back), 0])
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)

    X, Y = create_dataset(scaled, look_back)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]

    model = Sequential([
        LSTM(50, activation='relu', input_shape=(look_back, 1), return_sequences=True),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    model.fit(X_train.reshape(X_train.shape[0], X_train.shape[1], 1), Y_train, epochs=100, batch_size=32, verbose=0)

    train_predict = model.predict(X_train.reshape(X_train.shape[0], X_train.shape[1], 1))
    test_predict = model.predict(X_test.reshape(X_test.shape[0], X_test.shape[1], 1))

    train_predict = scaler.inverse_transform(train_predict)
    Y_train = scaler.inverse_transform([Y_train])
    test_predict = scaler.inverse_transform(test_predict)
    Y_test = scaler.inverse_transform([Y_test])

    rmse = np.sqrt(mean_squared_error(Y_test[0], test_predict[:, 0]))

    plt.figure(figsize=(12, 6))
    plt.plot(df.index[look_back:train_size+look_back], Y_train[0], label='Training Data')
   #''' plt.plot(df.index[train_size+look_back:], Y_test[0], label='Test Data')'''
    plt.plot(df.index[train_size+look_back:], test_predict[:, 0], label='NARNET Forecast')
    plt.title('NARNET Model Forecast')
    plt.legend()
    plt.show()
    print("Printed successfull")
    return test_predict, rmse

# Hybrid Model Creation
    
def create_hybrid_model(sarima_forecast, narnet_forecast, test_data):
    print("5")
    if hasattr(sarima_forecast, 'predicted_mean'):
        sarima_values = sarima_forecast.predicted_mean
    else:
        sarima_values = sarima_forecast
    
    # Ensuring both forecasts have the same length
    min_length = min(len(sarima_values), len(narnet_forecast))
    sarima_values = sarima_values[:min_length]
    narnet_values = narnet_forecast.flatten()[:min_length]
    
    print("5.1: Combining SARIMA and NARNET forecasts")
    hybrid_forecast = 0.5 * sarima_values + 0.5 * narnet_values
    
    print("5.2: Calculating RMSE")
    rmse = np.sqrt(mean_squared_error(test_data['Price'][:min_length], hybrid_forecast))
    
    print("5.3: Plotting hybrid forecast")
    plt.figure(figsize=(12, 6))
    '''plt.plot(test_data.index[:min_length], test_data['Price'][:min_length], label='Test Data')'''
    plt.plot(test_data.index[:min_length], hybrid_forecast, label='Hybrid Model Forecast')
    plt.title('Hybrid Model Forecast')
    plt.legend()
    plt.show()
    
    print("5.4: Hybrid model creation completed")
    return hybrid_forecast, rmse


# Threshold Setting and Alert System
def set_threshold_and_alert(hybrid_forecast, test_data):
    
    threshold = np.mean(test_data['Price']) + 1.0 * np.std(test_data['Price'])
    alerts = []

    for i, price in enumerate(hybrid_forecast):
        if price > threshold:
            alerts.append(f"Alert: Predicted price {price:.2f} exceeds threshold {threshold:.2f} on day {i+1}")

    plt.figure(figsize=(12, 6))
    plt.plot(test_data.index, hybrid_forecast, label='Hybrid Forecast')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    plt.title('Hybrid Model Forecast with Threshold')
    plt.legend()
    plt.show()

    return threshold, alerts
    print("Printed successfull")

#Model Evaluation and Future Prediction
def set_threshold_and_alert(hybrid_forecast, test_data):
    print("6.0: Setting threshold and generating alerts")
    
    # Ensuring that we're using the same number of data points
    min_length = min(len(hybrid_forecast), len(test_data))
    hybrid_forecast = hybrid_forecast[:min_length]
    test_data = test_data.iloc[:min_length]
    
    threshold = np.mean(test_data['Price']) + 1.0 * np.std(test_data['Price'])
    alerts = []
    
    for i, price in enumerate(hybrid_forecast):
        if price > threshold:
            alerts.append(f"Alert: Predicted price {price:.2f} exceeds threshold {threshold:.2f} on day {i+1}")
    
    plt.figure(figsize=(12, 6))
    plt.plot(test_data.index, hybrid_forecast, label='Hybrid Forecast')
    plt.plot(test_data.index, test_data['Price'], label='Actual Price')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    plt.title('Hybrid Model Forecast with Threshold')
    plt.legend()
    plt.show()
    
    print("6.1: Threshold setting and alert generation completed")
    return threshold, alerts

def evaluate_and_predict(df, hybrid_forecast, test_data, num_future_days=7):
    print("7.0: Evaluating model and making future predictions")
    
    # Ensure that we're using the same number of data points
    min_length = min(len(hybrid_forecast), len(test_data))
    hybrid_forecast = hybrid_forecast[:min_length]
    test_data = test_data.iloc[:min_length]
    
    mse = mean_squared_error(test_data['Price'], hybrid_forecast)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_data['Price'], hybrid_forecast)
    mape = np.mean(np.abs((test_data['Price'] - hybrid_forecast) / test_data['Price'])) * 100
    
    print(f"Model Evaluation Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.4f}%")
    
    last_date = test_data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=num_future_days)
    
    # For simplicity, we'll use a naive approach for future predictions
    future_predictions = hybrid_forecast[-num_future_days:]
    
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Price'], label='Historical Data')
    plt.plot(test_data.index, hybrid_forecast, label='Hybrid Forecast')
    plt.plot(future_dates, future_predictions, label='Future Predictions')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    plt.title('Historical Data, Hybrid Forecast, and Future Predictions')
    plt.legend()
    plt.show()
    
    print("7.1: Model evaluation and future prediction completed")
    return future_dates, future_predictions

# Main execution
if __name__ == "__main__":
# Loading and preprocess data
    df = load_and_preprocess_data(file_path)
    print(df.head())
    print(f"Dataset shape: {df.shape}")

# Perform EDA
    perform_eda(df)

    # Split data
    train_size = int(len(df) * 0.8)
    train_data = df[:train_size]
    test_data = df[train_size:]

# Implementing SARIMA
    sarima_forecast, sarima_rmse = implement_sarima(df, train_size)
    print(f"SARIMA RMSE: {sarima_rmse:.4f}")

# Implementing NARNET
    narnet_forecast, narnet_rmse = implement_narnet(df, train_size)
    print(f"NARNET RMSE: {narnet_rmse:.4f}")

    # Create Hybrid Model
    hybrid_forecast, hybrid_rmse = create_hybrid_model(sarima_forecast, narnet_forecast, test_data)
    print(f"Hybrid Model RMSE: {hybrid_rmse:.4f}")

    # Settinh Threshold and Generate Alerts
    threshold, alerts = set_threshold_and_alert(hybrid_forecast, test_data)
    print(f"Threshold: {threshold:.2f}")
    for alert in alerts:
        print(alert)

 # Evaluate Model and Make Future Predictions
    future_dates, future_predictions = evaluate_and_predict(df, hybrid_forecast, test_data)
    print("\nFuture Predictions:")
    for date, price in zip(future_dates, future_predictions):
        print(f"{date.date()}: {price:.2f}")
    print("SUCCESSFULLY PREDICTED")

@app.route('/predict_future_prices', methods=['POST'])
def predict_future_prices():
    # Call the evaluate_and_predict function to get the future predictions
    future_dates, future_predictions = evaluate_and_predict(df, hybrid_forecast, test_data)

    # Convert the date objects to strings
    future_dates = [date.strftime('%Y-%m-%d') for date in future_dates]

    return jsonify({
        'future_dates': future_dates,
        'future_predictions': list(future_predictions)
    })
