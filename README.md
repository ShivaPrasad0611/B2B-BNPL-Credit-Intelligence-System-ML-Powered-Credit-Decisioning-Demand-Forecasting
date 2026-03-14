# B2B-BNPL-Credit-Intelligence-System-ML-Powered-Credit-Decisioning-Demand-Forecasting

> An end-to-end Machine Learning web application for **B2B Buy Now Pay Later (BNPL)** credit decisioning and demand forecasting — built with XGBoost and Streamlit.

---

## 🚀 Live Features

### 💳 Credit Predictor
- Instant BNPL credit **Approved / Rejected** decision
- **Approval probability** score
- **Default risk** percentage
- **Recommended credit limit** in ₹ Lakh
- Factor-by-factor breakdown with color-coded indicators
- Personalized tips to improve eligibility

### 📦 Demand Forecasting
- Predicts **next month order count** and **GMV (₹ Lakh)**
- **6-month projection table** with month-on-month % change
- 5 interactive **Plotly charts**:
  - 📈 6-Month GMV Forecast Line Chart (with confidence band)
  - 📊 Order Count Bar Chart (with +10% target line)
  - 🥧 Revenue Composition Donut Chart
  - 🕸️ Business Health Radar Chart (vs benchmark)
  - 📉 GMV vs Orders Dual-Axis Trend Chart
- Demand signal checklist
- Actionable improvement tips

---

## 🗂️ Project Structure

```
b2b-bnpl-intelligence/
│
├── train_model.ipynb           # Jupyter notebook — train & save all ML models
├── b2b_bnpl_app.py             # Streamlit web app
├── b2b_bnpl_dataset.csv        # Dataset (10,000 B2B business records)
│
├── bnpl_classifier.pkl         # Saved: BNPL approval classifier
├── bnpl_risk_regressor.pkl     # Saved: Default risk regressor
├── bnpl_credit_features.pkl    # Saved: Credit feature list
├── bnpl_order_forecast.pkl     # Saved: Order count forecaster
├── bnpl_gmv_forecast.pkl       # Saved: GMV forecaster
├── bnpl_demand_features.pkl    # Saved: Demand feature list
│
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

> ⚠️ The `.pkl` files are auto-generated when you run `train_model.ipynb`. They are not committed to the repo.

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/b2b-bnpl-intelligence.git
cd b2b-bnpl-intelligence
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv

# Activate — Windows
venv\Scripts\activate

# Activate — Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Models
Open and run all cells in the Jupyter notebook:
```bash
jupyter notebook train_model.ipynb
```
This will generate **6 `.pkl` model files** in your project folder.

### 5. Launch the App
```bash
streamlit run b2b_bnpl_app.py
```
Open your browser at **http://localhost:8501**

---

## 📦 Requirements

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
plotly>=5.18.0
joblib>=1.3.0
```

Install all at once:
```bash
pip install streamlit pandas numpy scikit-learn xgboost plotly joblib
```

---

## 🤖 ML Models Used

| Model | Purpose | Algorithm |
|---|---|---|
| `bnpl_classifier` | Approve or reject BNPL credit | XGBoost Classifier |
| `bnpl_risk_regressor` | Predict default probability (0–1) | XGBoost Regressor |
| `bnpl_order_forecast` | Forecast next month order count | XGBoost Regressor |
| `bnpl_gmv_forecast` | Forecast next month GMV (₹ Lakh) | XGBoost Regressor |

---

## 📊 Dataset

The dataset contains **10,000 B2B business records** with 42 features including:

| Category | Features |
|---|---|
| Business Profile | Age, employees, revenue, GST status |
| Financial Health | Gross margin, debt-to-equity, current ratio |
| Payment Behaviour | On-time %, avg delay, platform tenure |
| Credit History | Bureau score, past defaults, enquiries |
| Platform Activity | GMV, invoice frequency, repeat rate, billers |
| Market Context | Sector BNPL penetration, GDP growth |

**Targets:**
- `bnpl_approved` — Binary (0/1)
- `default_probability` — Continuous (0.0–1.0)
- `credit_limit_lakh` — Continuous

---

## 🖥️ App Screenshots

| Credit Predictor | Demand Forecasting |
|---|---|
| Input form → instant decision | Input metrics → 5 charts + 6-month table |
| Approval %, Risk %, Credit Limit | GMV forecast, Radar chart, Dual-axis trend |

---

## 📁 How the Two Files Connect

```
train_model.ipynb
      │
      │  trains XGBoost models on b2b_bnpl_dataset.csv
      │  saves 6 .pkl files
      ▼
b2b_bnpl_app.py
      │
      │  loads .pkl files with joblib (cached, no retraining)
      │  renders Streamlit UI
      │  runs predictions on user input
      ▼
 Browser at localhost:8501
```

---

## 🔒 Model Caching

The Streamlit app uses `@st.cache_resource` to load models **once per session**. Models are never retrained during app usage — only when you explicitly re-run the notebook.

---

## 💡 Business Use Case

B2B BNPL allows businesses to:
- 💰 **Preserve cash flow** — buy now, pay in 30–90 days
- 📈 **Scale faster** — place larger orders without upfront capital
- 🤝 **Build credit** — every transaction improves your credit profile
- 📦 **Forecast demand** — plan procurement and inventory intelligently

---

## 🛠️ Troubleshooting

| Issue | Fix |
|---|---|
| `FileNotFoundError: bnpl_classifier.pkl` | Run all cells in `train_model.ipynb` first |
| `ModuleNotFoundError: xgboost` | Run `pip install xgboost` |
| `ModuleNotFoundError: streamlit` | Run `pip install streamlit` |
| Port 8501 already in use | Run `streamlit run b2b_bnpl_app.py --server.port 8502` |
| Slow first load | Models load once on startup (~10–15s), then cached instantly |

---



## ⭐ Star this repo if you found it useful!
