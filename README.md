# 📈 HDFC Stock Price Forecasting using LSTM

A deep learning project that uses a **Stacked LSTM (Long Short-Term Memory)** model to perform **multi-output time-series forecasting** on HDFC Bank's historical stock data. The model predicts the next **5 days** across all **11 stock features** simultaneously.

---

## 🗂️ Project Structure

```
├── Kunal_Final_LSTM_all_11_.ipynb   # Main notebook (training + evaluation)
├── hdfc_lstm_dashboard.py           # Streamlit web dashboard
├── save_model_pickle.py             # Script to save trained model bundle
├── hdfc_lstm_bundle.pkl             # Saved model (generated after training)
├── HDFC.csv                         # Dataset
└── README.md
```

---

## 🧠 Model Architecture

| Component | Details |
|---|---|
| Model Type | Stacked LSTM |
| Hidden Size | 128 units |
| Layers | 2 LSTM layers |
| Dropout | 0.2 |
| Input Window | 5 days |
| Forecast Horizon | 5 days |
| Input Features | 11 stock features |
| Loss Function | Huber Loss |
| Optimizer | Adam (lr = 0.001) |
| Scheduler | ReduceLROnPlateau |
| Max Epochs | 200 (with early stopping, patience=20) |

---

## 📊 Features Used

| # | Feature | Description |
|---|---|---|
| 1 | Prev Close | Previous day's closing price |
| 2 | Open | Opening price |
| 3 | High | Highest price of the day |
| 4 | Low | Lowest price of the day |
| 5 | Last | Last traded price |
| 6 | Close | Closing price |
| 7 | VWAP | Volume Weighted Average Price |
| 8 | Volume | Number of shares traded (log-transformed) |
| 9 | Turnover | Total turnover (log-transformed) |
| 10 | Trades | Number of trades (log-transformed) |
| 11 | %Deliverable | Percentage of deliverable volume |

---

## 🔄 ML Pipeline

1. **Data Ingestion** — Load HDFC NSE historical CSV data
2. **Exploratory Data Analysis** — Correlation heatmaps, time-series plots, missing value analysis
3. **Feature Engineering & Scaling** — Log-transform skewed features, MinMaxScaler normalization
4. **Sequence Construction** — Sliding window sequences (5-day input → 5-day output)
5. **LSTM Model Architecture** — Stacked LSTM with dropout and fully connected output layer
6. **Training Loop** — Huber loss, Adam optimizer, LR scheduler, early stopping
7. **Evaluation & Visualisation** — RMSE, MAE, R² per feature; Actual vs Predicted plots

---

## 📉 Train/Test Split

- **Training Set**: 80% of data
- **Test Set**: 20% of data
- Chronological split (no shuffling) to preserve time-series integrity

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/hdfc-lstm-forecasting.git
cd hdfc-lstm-forecasting
```

### 2. Install dependencies
```bash
pip install torch numpy pandas matplotlib seaborn scikit-learn streamlit plotly
```

### 3. Run the notebook
Open `Kunal_Final_LSTM_all_11_.ipynb` in Jupyter or VS Code and run all cells.

### 4. Save the model
The last cell of the notebook saves `hdfc_lstm_bundle.pkl` automatically.

### 5. Launch the Streamlit Dashboard
```bash
streamlit run hdfc_lstm_dashboard.py
```

Then upload the `hdfc_lstm_bundle.pkl` file via the sidebar.

---

## 📊 Streamlit Dashboard Features

| Tab | Description |
|---|---|
| 📉 Loss Curves | Interactive training vs validation loss chart |
| 🎯 Close Forecast | Actual vs Predicted Close price with residuals |
| 🔢 All Features | Subplots for all 11 features (selectable) |
| 📊 Metrics Table | Color-coded MSE, RMSE, MAE, R² per feature + CSV export |
| 🔮 Live Predict | Input 5 days of raw data → get next 5-day forecast |

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?style=flat&logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat&logo=streamlit)
![Plotly](https://img.shields.io/badge/Plotly-5.x-blueviolet?style=flat&logo=plotly)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-f7931e?style=flat&logo=scikit-learn)

---

## 📌 Notes

- The `%Deliverble` and `Trades` columns had missing values — filled using median imputation
- `Volume`, `Turnover`, and `Trades` were log-transformed (`log1p`) to reduce skewness
- Model uses `torch.manual_seed(42)` and `np.random.seed(42)` for reproducibility
- GPU is used automatically if available (`cuda`), otherwise falls back to `cpu`

---

## 👤 Author

**Kunal Singh**  
B.Tech | AI & ML Assignment Project  
📧 your-email@example.com  
🔗 [LinkedIn]([https://linkedin.com/in/your-profile](https://www.linkedin.com/in/kunal-singh-a345683a4/)) · [GitHub]([https://github.com/your-username](https://github.com/Kunal2145))

---

## Screenshots of the streamlit Web and trends

1. <img width="1427" height="804" alt="Screenshot 2026-04-29 at 4 26 55 PM" src="https://github.com/user-attachments/assets/ffb52441-fa8d-4a0f-81a4-f005b03220e6" />
2. <img width="1378" height="791" alt="Screenshot 2026-04-29 at 4 27 12 PM" src="https://github.com/user-attachments/assets/130673e0-03b5-4883-ab33-fd07aacab149" />
3. <img width="1402" height="740" alt="Screenshot 2026-04-29 at 4 27 58 PM" src="https://github.com/user-attachments/assets/723a8460-fa1a-4776-9ac4-9f3baeea52d7" />
4. <img width="1325" height="644" alt="Screenshot 2026-04-29 at 4 28 15 PM" src="https://github.com/user-attachments/assets/2c2849d1-33c1-4d90-99d0-38f7a9e1d339" />

## 📄 License

This project is for educational purposes only.
