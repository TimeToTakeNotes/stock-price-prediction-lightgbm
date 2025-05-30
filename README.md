# Stock Price Prediction with LightGBM (RandomForest Style)

This project implements a time-series regression pipeline to predict stock prices (Open, High, Low, Close) using a LightGBM-based model structured as a Random Forest. It leverages historical stock data from the S&P 500 index and includes feature engineering, encoding, training, evaluation, and prediction.

---

## 📈 Project Overview

- **Objective:** Predict future stock prices based on historical prices, time features, and technical indicators.
- **Model:** `MultiOutputRegressor` using `LightGBM`, acting as a fast and memory-efficient substitute for traditional Random Forest.
- **Data Source:** [Kaggle - S&P 500 stock data (5 years)](https://www.kaggle.com/datasets/camnugent/sandp500)
- **Notebook Environment:** Designed for Google Colab, with easy-to-run cells and visualizations.

---

## 🔧 Features & Techniques Used

### 📅 Time-Series Feature Engineering
- Extracted:
  - Day, Month, Year, Quarter
  - Day of the week
- Lag features:
  - Previous day’s open, close, and volume
- Moving averages:
  - 5-day and 10-day rolling averages (shifted to prevent leakage)

### 🔄 Data Preprocessing
- Removed all rows with missing values.
- Used one-hot encoding for stock names using `ColumnTransformer`.

### ⚙️ Model Training
- Time-aware split per stock to preserve sequence integrity (80% train, 20% test).
- `LightGBM` wrapped with `MultiOutputRegressor` for multi-target regression.
- Parallel training with `n_jobs=-1`.

### 📊 Evaluation Metrics
- **MSE** (Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R² Score**

---

## ✅ Results Summary

- **Test Set Performance**
  - **Mean Squared Error (MSE):** 459.79
  - **Mean Absolute Error (MAE):** 2.93
  - **R² Score:** 0.9717

- **Actual vs Predicted Prices:**
  - Very close alignment between actual and predicted close prices for the first 200 samples.

- **Residual Analysis:**
  - Histograms for each target (Open, High, Low, Close) show sharp spikes at 0 with minimal spread.
  - Suggests strong model calibration and minimal bias or variance issues.

---

## 📈 Visual Insights

> _Note: Replace these image placeholders with your actual image URLs or upload them in your GitHub repo._

- **Actual vs. Predicted Close Prices (first 200 samples)**  
  ![Actual vs Predicted Close](https://github.com/user-attachments/assets/9aea7902-db6f-4740-8fba-254672523216)

- **Residual Histogram (All Targets Combined)**  
  ![Combined Residuals](https://github.com/user-attachments/assets/4d6307a6-3cbe-4ffd-87e6-3896dcd43651)

- **Residuals by Target:**
  - Open  
    ![Open Residuals](https://github.com/user-attachments/assets/b630d97b-280d-4f9d-8f43-7cbd398565e5)
  - High  
    ![High Residuals](https://github.com/user-attachments/assets/681fd84e-fb6c-4fd9-8e44-fedc780abaff)
  - Low  
    ![Low Residuals](https://github.com/user-attachments/assets/eaab85cd-2395-419a-929c-0540f59a6c5a)
  - Close  
    ![Close Residuals](https://github.com/user-attachments/assets/d3ee60c6-d17e-4fc5-90c5-7f98459909a9)

---

## 🔮 Example Predictions

### Last Row Prediction
```python
Model predicts last row to be [Open, High, Low, Close]:  [73.05, 73.85, 72.72, 73.13]
Actual values are [Open, High, Low, Close]:  [72.7, 75, 72.69, 73.86]
```

### Real Prediction Example for AAPL (June 2025)
```python
Input: June 1st, 2025 (Q2, Monday), prior close: 190.42, volume: 6M+
Model predicts values to be [Open, High, Low, Close]: [188.91, 190.38, 186.66, 192.04]
```

---

## 📌 Final Thoughts

- LightGBM proved highly efficient and accurate for time-series regression.
- Feature engineering (especially moving averages and lags) had a noticeable impact on predictive performance.
- Residuals showed minimal deviation, with dense peaks around zero — indicating low bias and good generalization.
- The time-aware split strategy ensured temporal integrity and mimics realistic forecasting scenarios.
