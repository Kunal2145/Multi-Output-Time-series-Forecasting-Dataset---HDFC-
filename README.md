📈 HDFC Stock Forecasting using Multi-Feature LSTM
A deep learning project that performs multi-step, multi-feature time series forecasting on HDFC stock data using a stacked LSTM model, with an interactive dashboard built using Streamlit.
🚀 Overview
This project predicts the next 5 days of stock prices across 11 features (Open, High, Low, Close, Volume, etc.) using the previous 5 days of historical data.
It combines:
1. 🧠 Deep learning (LSTM)
2. 📊 Data visualization
3. 🌐 Interactive web app
**Key Features**
1. 🔮 Multi-step forecasting (5-day prediction)
2. 🔢 Multi-feature prediction (11 stock features)
3. 🧠 Stacked LSTM model (PyTorch)
4. 📉 Training vs Validation loss visualization
5. 📊 Feature-wise performance metrics (RMSE, MAE, R²)
6. 🎯 Actual vs Predicted comparison
7. 🔴 Live prediction interface
8. 📥 Downloadable results
**🧠 Model Architecture**
Input: 5 days × 11 features
LSTM Layers: 2
Hidden Units: 128
Dropout: 0.2
Output: 5 days × 11 features

📂 Project Structure
.
├── hdfc_dash.py                 # Streamlit dashboard
├── hdfc_lstm_bundle.pkl        # Saved model + scaler + predictions
├── notebook.ipynb              # Model training notebook
├── save_model_pickle.py        # Script to export model bundle
├── requirements.txt            # Dependencies
└── README.md

📊 Dashboard Features
1. 📉 Loss Curves
Shows training vs validation loss
Helps identify overfitting
2. 🎯 Close Price Forecast
Actual vs predicted close prices
Residual error visualization
3. 🔢 All Feature Predictions
Compare predictions across all 11 features
4. 📊 Metrics Table
MSE, RMSE, MAE, R² for each feature
5. 🔮 Live Prediction
Input last 5 days of data
Predict next 5 days instantly
📦 Model Bundle
The .pkl file includes:
Trained model weights
Scaler (MinMaxScaler)
Predictions and actual values
Feature names
Training and validation loss
📈 Evaluation Metrics
RMSE (Root Mean Squared Error)
MAE (Mean Absolute Error)
R² Score
🛠 Tech Stack
Python
PyTorch
NumPy, Pandas
Scikit-learn
Streamlit
Plotly
📌 Future Improvements
Hyperparameter tuning
Add attention mechanism
Deploy on Streamlit Cloud
Integrate real-time stock data API
Improve feature engineerin
