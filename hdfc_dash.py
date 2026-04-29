"""
=========================================================
  HDFC LSTM Stock Forecasting — Streamlit Dashboard
  Run: streamlit run hdfc_lstm_dashboard.py
=========================================================
"""

import streamlit as st
import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HDFC LSTM Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #0d1117; color: #e6edf3; }
    [data-testid="stSidebar"]          { background: #161b22; border-right: 1px solid #30363d; }
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 18px 22px;
        text-align: center;
    }
    .metric-card h3 { margin: 0; font-size: 0.85rem; color: #8b949e; letter-spacing: .05em; text-transform: uppercase; }
    .metric-card p  { margin: 6px 0 0; font-size: 1.7rem; font-weight: 700; color: #58a6ff; }
    .metric-card .delta { font-size: 0.8rem; color: #3fb950; margin-top: 4px; }
    .section-header {
        font-size: 1.15rem;
        font-weight: 600;
        color: #58a6ff;
        border-left: 3px solid #58a6ff;
        padding-left: 10px;
        margin: 18px 0 10px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background: #161b22; border-radius: 8px; padding: 4px; }
    .stTabs [data-baseweb="tab"]      { border-radius: 6px; color: #8b949e; }
    .stTabs [aria-selected="true"]    { background: #1f6feb !important; color: #ffffff !important; }
    hr { border-color: #30363d; }
</style>
""", unsafe_allow_html=True)

# ── LSTM Model Definition (must match training) ────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, input_size=11, hidden_size=128, num_layers=2,
                 output_days=5, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.output_days = output_days
        self.n_features  = input_size

        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_days * input_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        out = out.view(-1, self.output_days, self.n_features)
        return out


# ── Load bundle ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_bundle(uploaded_file):
    bundle = pickle.load(uploaded_file)
    hp     = bundle["hyperparams"]
    model  = LSTMModel(
        input_size  = hp["input_size"],
        hidden_size = hp["hidden_size"],
        num_layers  = hp["num_layers"],
        output_days = hp["OUTPUT_DAYS"],
        dropout     = hp["dropout"],
    )
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()
    return model, bundle


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/28/HDFC_Bank_Logo.svg/320px-HDFC_Bank_Logo.svg.png", width=160)
    st.title("HDFC LSTM Dashboard")
    st.markdown("---")

    pkl_file = st.file_uploader("📂 Upload model bundle (.pkl)", type=["pkl"])

    if pkl_file:
        st.success("Bundle loaded ✅")

    st.markdown("---")
    st.markdown("#### About")
    st.markdown("""
    - **Model**: Stacked LSTM (2 layers, 128 hidden)  
    - **Input**: 5-day window × 11 features  
    - **Output**: 5-day multi-step forecast  
    - **Dataset**: HDFC NSE historical data  
    - **Loss**: Huber Loss  
    """)


# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown("# 📈 HDFC Stock — LSTM Forecasting Dashboard")
st.markdown("Multi-output time-series forecasting across **11 stock features**.")

if not pkl_file:
    st.info("👈 Upload your `hdfc_lstm_bundle.pkl` in the sidebar to begin.", icon="💡")
    st.markdown("""
    **Steps:**
    1. Run `save_model_pickle.py` at the end of your notebook to generate the bundle.
    2. Upload the `.pkl` file here using the sidebar uploader.
    3. Explore the interactive charts, metrics, and forecasts.
    """)
    st.stop()

model, bundle = load_bundle(pkl_file)

preds_orig   = bundle["preds_orig"]       # (samples, OUTPUT_DAYS, N_FEATURES)
y_orig       = bundle["y_orig"]
feature_names = bundle["feature_names"]
test_dates   = pd.to_datetime(bundle["test_dates"])
train_losses = bundle["train_losses"]
val_losses   = bundle["val_losses"]

N_FEATURES  = len(feature_names)
OUTPUT_DAYS = preds_orig.shape[1]
CLOSE_IDX   = feature_names.index("Close")

# Flatten to 2D for metric calculation
preds_2d = preds_orig.reshape(-1, N_FEATURES)
y_2d     = y_orig.reshape(-1, N_FEATURES)

# ── KPI Row ────────────────────────────────────────────────────────────────────
overall_rmse = np.sqrt(mean_squared_error(y_2d, preds_2d))
overall_mae  = mean_absolute_error(y_2d, preds_2d)
overall_r2   = r2_score(y_2d, preds_2d)
close_r2     = r2_score(y_2d[:, CLOSE_IDX], preds_2d[:, CLOSE_IDX])
close_rmse   = np.sqrt(mean_squared_error(y_2d[:, CLOSE_IDX], preds_2d[:, CLOSE_IDX]))

col1, col2, col3, col4, col5 = st.columns(5)
cards = [
    (col1, "Overall R²",    f"{overall_r2:.4f}",   "All 11 features"),
    (col2, "Close R²",      f"{close_r2:.4f}",     "Close price only"),
    (col3, "Overall RMSE",  f"{overall_rmse:.2f}",  "Across all features"),
    (col4, "Close RMSE",    f"{close_rmse:.2f}",    "₹ error on Close"),
    (col5, "Epochs Trained", f"{len(train_losses)}", "With early stopping"),
]
for col, title, val, sub in cards:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{title}</h3>
            <p>{val}</p>
            <div class="delta">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📉 Loss Curves", "🎯 Close Forecast", "🔢 All Features", "📊 Metrics Table", "🔮 Live Predict"
])

# ════════════════════════════════════════════════════════════
# TAB 1 — Loss Curves
# ════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Training vs Validation Loss</div>', unsafe_allow_html=True)

    epochs = list(range(1, len(train_losses) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=train_losses, name="Train Loss",
                             line=dict(color="#58a6ff", width=2)))
    fig.add_trace(go.Scatter(x=epochs, y=val_losses, name="Val Loss",
                             line=dict(color="#f78166", width=2)))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        xaxis_title="Epoch",
        yaxis_title="Huber Loss",
        legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Best Val Loss", f"{min(val_losses):.6f}", f"Epoch {val_losses.index(min(val_losses))+1}")
    with c2:
        st.metric("Final Train Loss", f"{train_losses[-1]:.6f}")


# ════════════════════════════════════════════════════════════
# TAB 2 — Close Forecast
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Actual vs Predicted — Close Price (t+5 forecast)</div>',
                unsafe_allow_html=True)

    actual_close = y_orig[:, -1, CLOSE_IDX]
    pred_close   = preds_orig[:, -1, CLOSE_IDX]
    residuals    = actual_close - pred_close

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_dates, y=actual_close, name="Actual Close",
                             line=dict(color="#3fb950", width=2)))
    fig.add_trace(go.Scatter(x=test_dates, y=pred_close, name="Predicted Close",
                             line=dict(color="#f78166", width=2, dash="dash")))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        xaxis_title="Date", yaxis_title="Price (₹)",
        legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
        height=430,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Residuals (Actual − Predicted)</div>', unsafe_allow_html=True)
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=test_dates, y=residuals, name="Residual",
                          marker_color=np.where(residuals >= 0, "#3fb950", "#f78166")))
    fig2.update_layout(template="plotly_dark", paper_bgcolor="#0d1117",
                       plot_bgcolor="#0d1117", height=280, xaxis_title="Date",
                       yaxis_title="Residual (₹)")
    st.plotly_chart(fig2, use_container_width=True)


# ════════════════════════════════════════════════════════════
# TAB 3 — All 11 Feature Plots
# ════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">All 11 Features — Actual vs Predicted</div>',
                unsafe_allow_html=True)

    selected = st.multiselect("Select features to display",
                               feature_names, default=feature_names[:4])

    if not selected:
        st.warning("Please select at least one feature.")
    else:
        n = len(selected)
        rows = (n + 1) // 2
        fig = make_subplots(rows=rows, cols=2,
                            subplot_titles=selected,
                            vertical_spacing=0.08)

        colors_actual = "#58a6ff"
        colors_pred   = "#f78166"

        for i, feat in enumerate(selected):
            fidx = feature_names.index(feat)
            r, c = divmod(i, 2)
            fig.add_trace(go.Scatter(x=test_dates, y=y_orig[:, -1, fidx],
                                     name=f"Actual {feat}",
                                     line=dict(color=colors_actual, width=1.5),
                                     legendgroup=feat, showlegend=(i == 0)),
                          row=r+1, col=c+1)
            fig.add_trace(go.Scatter(x=test_dates, y=preds_orig[:, -1, fidx],
                                     name=f"Predicted {feat}",
                                     line=dict(color=colors_pred, width=1.5, dash="dash"),
                                     legendgroup=feat, showlegend=(i == 0)),
                          row=r+1, col=c+1)

        fig.update_layout(template="plotly_dark", paper_bgcolor="#0d1117",
                          plot_bgcolor="#0d1117",
                          height=350 * rows, showlegend=True,
                          legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1))
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════
# TAB 4 — Metrics Table
# ════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Feature-wise Evaluation Metrics</div>',
                unsafe_allow_html=True)

    rows = []
    for i, col in enumerate(feature_names):
        mse  = mean_squared_error(y_2d[:, i], preds_2d[:, i])
        rmse = np.sqrt(mse)
        mae  = mean_absolute_error(y_2d[:, i], preds_2d[:, i])
        r2   = r2_score(y_2d[:, i], preds_2d[:, i])
        rows.append({"Feature": col, "MSE": round(mse, 4), "RMSE": round(rmse, 4),
                     "MAE": round(mae, 4), "R²": round(r2, 4)})

    metrics_df = pd.DataFrame(rows)

    def color_r2(val):
        if val >= 0.95:  return "background-color: #1a4731; color: #3fb950"
        if val >= 0.85:  return "background-color: #1c3a1c; color: #8bc34a"
        if val >= 0.70:  return "background-color: #3b2a00; color: #f0ad4e"
        return "background-color: #3d1a1a; color: #f78166"

    styled = metrics_df.style.applymap(color_r2, subset=["R²"]) \
                             .format({"MSE": "{:.4f}", "RMSE": "{:.4f}",
                                      "MAE": "{:.4f}", "R²": "{:.4f}"}) \
                             .set_properties(**{"text-align": "center"})
    st.dataframe(styled, use_container_width=True, height=430)

    # Download button
    csv = metrics_df.to_csv(index=False).encode()
    st.download_button("⬇️ Download Metrics CSV", csv,
                       "hdfc_lstm_metrics.csv", "text/csv")

    st.markdown('<div class="section-header">R² Bar Chart</div>', unsafe_allow_html=True)
    bar_fig = px.bar(metrics_df, x="Feature", y="R²",
                     color="R²", color_continuous_scale="RdYlGn",
                     template="plotly_dark", height=350)
    bar_fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                          coloraxis_showscale=False)
    st.plotly_chart(bar_fig, use_container_width=True)


# ════════════════════════════════════════════════════════════
# TAB 5 — Live Predict
# ════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">🔮 Forecast Next 5 Days</div>', unsafe_allow_html=True)
    st.markdown("Enter the last **5 days** of raw stock data to generate a live forecast.")

    scaler = bundle["scaler"]

    default_vals = {feat: 0.0 for feat in feature_names}
    with st.form("live_form"):
        st.markdown("**Input: Last 5 days of stock data**")
        day_inputs = []
        for d in range(1, 6):
            st.markdown(f"**Day {d}**")
            cols = st.columns(len(feature_names))
            day_row = []
            for j, feat in enumerate(feature_names):
                val = cols[j].number_input(f"{feat}", value=0.0,
                                            key=f"d{d}_{feat}", format="%.4f")
                day_row.append(val)
            day_inputs.append(day_row)

        submitted = st.form_submit_button("🚀 Generate Forecast", use_container_width=True)

    if submitted:
        import torch
        raw = np.array(day_inputs, dtype=np.float32)          # (5, 11)
        scaled = scaler.transform(raw)                          # scale
        tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)  # (1,5,11)

        with torch.no_grad():
            pred_scaled = model(tensor).numpy()                # (1, 5, 11)

        pred_orig = scaler.inverse_transform(pred_scaled[0])   # (5, 11)
        result_df = pd.DataFrame(pred_orig,
                                  columns=feature_names,
                                  index=[f"t+{i+1}" for i in range(OUTPUT_DAYS)])

        st.success("Forecast generated ✅")
        st.dataframe(result_df.style.format("{:.2f}"), use_container_width=True)

        st.markdown('<div class="section-header">Forecasted Close Price</div>',
                    unsafe_allow_html=True)
        close_preds = pred_orig[:, CLOSE_IDX]
        fig_live = go.Figure()
        fig_live.add_trace(go.Scatter(
            x=[f"t+{i+1}" for i in range(OUTPUT_DAYS)],
            y=close_preds,
            mode="lines+markers",
            line=dict(color="#58a6ff", width=3),
            marker=dict(size=9, color="#f78166"),
            name="Predicted Close",
        ))
        fig_live.update_layout(
            template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            xaxis_title="Forecast Horizon", yaxis_title="Price (₹)", height=350,
        )
        st.plotly_chart(fig_live, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#8b949e; font-size:0.8rem;'>"
    "HDFC LSTM Forecasting Dashboard · Built with PyTorch & Streamlit</p>",
    unsafe_allow_html=True,
)