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

### Data Preparation
- Parsed and cleaned historical S&P 500 stock data.
- Removed rows with missing values.
- Extracted **time-based features**: day, month, year, quarter, and day of week.

### Feature Engineering
- **Lag features**: Previous day’s close, open, and volume per stock.
- **Moving averages**: 5-day and 10-day rolling averages.
- One-hot encoded stock symbols using `ColumnTransformer`.

### Modeling
- Trained a multi-output regression model with LightGBM (as a RandomForest alternative).
- Time-aware split per stock to preserve temporal order.
- Evaluated using **MSE**, **MAE**, and **R² score**.
- Visualized predictions and residuals.

---

## 📊 Results Summary

- **Test Set Performance:**
  - **Mean Squared Error (MSE):** 459.79
  - **Mean Absolute Error (MAE):** 2.93
  - **R² Score:** 0.9717

- **Visualizations Include:**
  - **Actual vs. Predicted Close Price for first 200 samples:**
    ![image](https://github.com/user-attachments/assets/9aea7902-db6f-4740-8fba-254672523216)

  - **Residual histogram for combined targets:**
    ![image](https://github.com/user-attachments/assets/4d6307a6-3cbe-4ffd-87e6-3896dcd43651)

  - **Residual histograms for all four predicted targets (Open, High, Low, Close):**
    ![image](https://github.com/user-attachments/assets/b630d97b-280d-4f9d-8f43-7cbd398565e5)
    
    ![image](https://github.com/user-attachments/assets/681fd84e-fb6c-4fd9-8e44-fedc780abaff)
    
    ![image](https://github.com/user-attachments/assets/eaab85cd-2395-419a-929c-0540f59a6c5a)
    
    ![image](https://github.com/user-attachments/assets/d3ee60c6-d17e-4fc5-90c5-7f98459909a9)
---

## 🔮 Example of Testing Output

### Last Row Prediction
```python
Model predicts last row to be [Open, High, Low, Close]:  [73.05, 73.85, 72.72, 73.13]
Actual values are [Open, High, Low, Close]:  [72.7, 75, 72.69, 73.86]
```

### Real Predictions (e.g., for AAPL at June 2025)
```python
Model predicts values to be [Open, High, Low, Close]: [188.91, 190.38, 186.66, 192.04]
```
