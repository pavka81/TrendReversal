# 🔁 TrendReversal: AI-Enhanced Technical Analysis for Stock Bounce Detection

TrendReversal is an AI-powered trading research framework that combines rule-based technical filters with machine learning to detect **Keltner Channel bounce** opportunities in stocks.

---

## ⚙️ Features

- ✅ **Data Acquisition** via Yahoo Finance or Alpha Vantage
- 🧼 **Data Cleaning** and Indicator Calculation (MACD, RSI, Force Index, EMA, Keltner)
- 🧠 **Rule-Based Trade Detection** using customizable signal filters
- 🔍 **Labeling and ML Model Training** using top-performing rule configs
- 📈 **Backtesting Engine** to simulate strategy performance
- 📊 **Interactive Notebooks** for evaluation and analysis
- 🧪 **Hyperparameter Tuning** for model optimization

---

## 📁 Project Structure

```text
TrendReversal/
├── configs/                    # Filter configs and top-performing rule combos
├── data/                       # Historical stock data (raw + processed)
├── models/                     # Trained ML models
├── notebooks/                  # Jupyter notebooks for analysis & reporting
├── results/                    # Backtest results, metrics, plots
├── src/                        # Core Python modules (feature extraction, modeling, utils)
├── train_from_rule_labels.py  # Script to train ML model from rule-based trades
├── main.py                     # Main pipeline (run everything)
└── README.md                   # Project overview and instructions
```

---

## 🧠 Training an ML Model

To train a machine learning model using top-performing rule-based signals as ground truth:

```bash
python train_from_rule_labels.py
```

This will:
- Load labeled trades
- Extract features from weekly data
- Train a classifier
- Save the model under `models/`
- Export a feature importance chart to `results/`

---

## 📊 Model Evaluation

To visualize and compare model performance against rule-based strategies, run:

- [`notebooks/evaluate_model_vs_rule_based.ipynb`](notebooks/evaluate_model_vs_rule_based.ipynb)

This notebook generates:
- ✅ ROC curve
- ✅ Confusion matrix
- ✅ Rule-based vs ML performance benchmark

---

## 📌 Coming Soon

- 📅 Walk-forward validation
- ☁️ Cloud deployment support
- 💡 Integration with portfolio simulators

---

## 📬 Contact

Open an issue or reach out if you'd like to collaborate, contribute, or share feedback.
