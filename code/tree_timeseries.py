import pandas as pd
import numpy as np
import math
import datetime
import re
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

# Import statsmodels for OLS estimation
import statsmodels.api as sm

# ----------------------
# Suppress warnings (if any)
# ----------------------
import warnings
warnings.filterwarnings("ignore")

# ======== Helper Functions for Tree Model =========

def add_cycle_features(df, date_col="DATE"):
    """
    Adds cyclical features based on day-of-week.
    """
    df[date_col] = pd.to_datetime(df[date_col])
    df['dayofweek'] = df[date_col].dt.dayofweek  # Monday=0, Sunday=6
    df['sin_dow'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['cos_dow'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    return df

def add_ma_feature(series, window=3):
    """
    Computes a moving-average feature over a given window.
    """
    return series.rolling(window=window, min_periods=1).mean()

def construct_lagged_dataset(df, target_col, L, decay_rate, date_col="DATE"):
    """
    For each time t (>= L) build feature vector composed of:
      - L lagged values (most recent first)
      - Current cyclical features (sin_dow, cos_dow)
      - MA feature: average of previous 3 observations
    Also computes an exponential decay weight.
    """
    df = df.copy().reset_index(drop=True)
    n = len(df)
    X, y, weights, dates = [], [], [], []
    T_max = n - 1
    for t in range(L, n):
        lag_feats = df[target_col].iloc[t-L:t].values[::-1].tolist()
        sin_dow = df.loc[t, 'sin_dow']
        cos_dow = df.loc[t, 'cos_dow']
        ma_feat = df[target_col].iloc[max(t-3, 0):t].mean()
        feat = lag_feats + [sin_dow, cos_dow, ma_feat]
        X.append(feat)
        y.append(df.loc[t, target_col])
        dates.append(df.loc[t, date_col])
        weight = math.exp(-decay_rate * (T_max - t))
        weights.append(weight)
    return np.array(X), np.array(y), np.array(weights), pd.to_datetime(dates)

def forecast_recursive(model, seed_window, seed_date, forecast_horizon, L, exog_df=None):
    """
    Given a fitted regression model (OLS or tree-based) and a seed window (most recent L observations),
    forecast recursively for forecast_horizon days.
    If exogenous features for future dates are provided in exog_df, they are used;
    otherwise, cycle features are computed from the forecast date.
    
    NOTE for tree-based models: the predicted feature vector must have the same dimension as during training.
    """
    forecasts = []
    current_window = list(seed_window)
    current_date = seed_date
    for h in range(forecast_horizon):
        next_date = current_date + pd.Timedelta(days=1)
        lag_feats = current_window[-L:][::-1]
        if exog_df is not None:
            cyc_vals = exog_df.loc[next_date]
            cyc_array = np.array([cyc_vals['sin_dow'], cyc_vals['cos_dow']])
        else:
            dow = next_date.dayofweek
            cyc_array = np.array([np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)])
        ma_feat = np.mean(current_window[-3:]) if len(current_window) >= 3 else np.mean(current_window)
        # Concatenate lag features, cycle features, and MA feature. Total length = L + 3.
        feat = np.concatenate([np.array(lag_feats), cyc_array, [ma_feat]])
        # For tree models, do not add constant.
        X_pred = feat.reshape(1, -1)
        y_hat = model.predict(X_pred)[0]
        forecasts.append(y_hat)
        current_window.append(y_hat)
        current_date = next_date
    return forecasts

def rolling_forecast_evaluation(model, df, target_col, L, forecast_horizon, decay_rate, start_idx):
    """
    Rolling-origin evaluation for the tree-based forecasting model.
    Returns MAPE (%) computed over all rolling origins.
    """
    errors = []
    n = len(df)
    for i in range(start_idx, n - forecast_horizon):
        if i - L < 0:
            continue
        seed_window = df[target_col].iloc[i-L:i].values
        seed_date = df.iloc[i-1]['DATE']
        preds = forecast_recursive(model, seed_window, seed_date, forecast_horizon, L)
        pred_avg = np.mean(preds)
        actual_avg = df[target_col].iloc[i:i+forecast_horizon].mean()
        if actual_avg == 0:
            continue
        errors.append(abs((actual_avg - pred_avg) / actual_avg))
    return np.mean(errors)*100 if errors else None

# ======== Additional Helper Functions: AR/ARMA with OLS =========

def rolling_forecast_evaluation_ar_ols(series, exog, p, forecast_horizon, start_idx):
    """
    Rolling-origin forecast evaluation for an AR(p) model using OLS.
    Predictors: constant, p lagged values, and exogenous cycle features.
    'series' is a pandas Series of the target.
    'exog' is a DataFrame (with same index as series) containing cycle features (e.g., sin_dow, cos_dow).
    Returns MAPE (%) over all rolling origins.
    """
    errors = []
    n = len(series)
    for i in range(start_idx, n - forecast_horizon):
        X_train, y_train = [], []
        for t in range(p, i):
            lags = series.iloc[t-p:t].values    # shape: (p,)
            cyc = exog.iloc[t].values             # shape: (2,)
            X_train.append(np.concatenate([lags, cyc]))  # total length = p + 2
            y_train.append(series.iloc[t])
        if len(X_train) == 0:
            continue
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        # Manually add constant column
        X_train_const = np.column_stack((np.ones(len(X_train)), X_train))
        model = sm.OLS(y_train, X_train_const).fit()
        seed_window = series.iloc[i-p:i].values.copy().tolist()
        current_index = i
        preds = []
        for k in range(forecast_horizon):
            cyc_forecast = exog.iloc[current_index].values  # shape: (2,)
            x_row = np.concatenate([np.array(seed_window[-p:]), cyc_forecast])  # length = p + 2
            X_pred = np.column_stack((np.ones(1), x_row.reshape(1, -1)))  # shape: (1, p+3)
            y_hat = model.predict(X_pred)[0]
            preds.append(y_hat)
            seed_window.append(y_hat)
            current_index += 1
            if current_index >= n:
                break
        if len(preds) < forecast_horizon:
            continue
        pred_avg = np.mean(preds)
        actual_avg = series.iloc[i:i+forecast_horizon].mean()
        if actual_avg == 0:
            continue
        errors.append(abs((actual_avg - pred_avg) / actual_avg))
    return np.mean(errors)*100 if errors else None

def rolling_forecast_evaluation_arma_ols(series, exog, forecast_horizon, start_idx):
    """
    Rolling-origin forecast evaluation for a simplified ARMA(1,1) model estimated via OLS.
    Predictors: constant, lag1 (y[t-1]), lagged error (approx. y[t-1] - y[t-2]),
                and exogenous cycle features.
    Returns MAPE (%) over all rolling origins.
    """
    errors = []
    n = len(series)
    for i in range(max(start_idx, 2), n - forecast_horizon):
        X_train, y_train = [], []
        for t in range(2, i):
            lag1 = series.iloc[t-1]
            lag_err = series.iloc[t-1] - series.iloc[t-2]
            cyc = exog.iloc[t].values
            X_train.append(np.concatenate([[lag1, lag_err], cyc]))  # length = 2+2 = 4
            y_train.append(series.iloc[t])
        if len(X_train) == 0:
            continue
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_train_const = np.column_stack((np.ones(len(X_train)), X_train))
        model = sm.OLS(y_train, X_train_const).fit()
        seed_y = series.iloc[i]
        seed_err = series.iloc[i] - series.iloc[i-1]
        preds = []
        current_index = i+1
        for k in range(forecast_horizon):
            cyc_forecast = exog.iloc[current_index].values
            x_row = np.concatenate([[seed_y, seed_err], cyc_forecast])  # length = 4
            X_pred = np.column_stack((np.ones(1), x_row.reshape(1, -1)))   # shape: (1, 5)
            y_hat = model.predict(X_pred)[0]
            preds.append(y_hat)
            new_err = y_hat - seed_y
            seed_y = y_hat
            seed_err = new_err
            current_index += 1
            if current_index >= n:
                break
        if len(preds) < forecast_horizon:
            continue
        pred_avg = np.mean(preds)
        actual_avg = series.iloc[i+1:i+1+forecast_horizon].mean()
        if actual_avg == 0:
            continue
        errors.append(abs((actual_avg - pred_avg) / actual_avg))
    return np.mean(errors)*100 if errors else None

# ======== Main Code: Data Preparation =========

# Read Electric Production data (expects columns: ['DATE','IPG2211A2N'])
df = pd.read_csv(r"C:\backupcgi\final_bak\Electric_Production_tm.csv")
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values(by='DATE').reset_index(drop=True)
df = add_cycle_features(df, date_col="DATE")
df['MA_3'] = add_ma_feature(df['IPG2211A2N'], window=3)

target_col = "IPG2211A2N"
date_col = "DATE"
decay_rate = 0.01
candidate_lags = [3, 5, 7, 9]
forecast_horizons = [3, 5, 7, 9]

# Split data into training and test sets (80% train, 20% test)
split_ratio = 0.8
split_idx = int(len(df) * split_ratio)
train_df = df.iloc[:split_idx].reset_index(drop=True)
test_df = df.iloc[split_idx:].reset_index(drop=True)

print("Total observations:", len(df))
print("Training observations:", len(train_df))
print("Test observations:", len(test_df))

# --- Tree-Based Forecasting (Using the above functions) ---
results_tree = {}
for horizon in forecast_horizons:
    best_mape = float('inf')
    best_L = None
    best_model = None
    print(f"\n[Tree Model] Forecast Horizon: {horizon} days")
    val_start_idx = int(0.8 * len(train_df))
    for L in candidate_lags:
        X_train, y_train, weights_train, _ = construct_lagged_dataset(train_df, target_col, L, decay_rate, date_col)
        tree_model = DecisionTreeRegressor(max_depth=5, random_state=42)
        tree_model.fit(X_train, y_train, sample_weight=weights_train)
        mape_val = rolling_forecast_evaluation(tree_model, train_df, target_col, L, horizon, decay_rate, start_idx=val_start_idx)
        if mape_val is None:
            continue
        print(f"  Candidate Lag L={L}: Validation MAPE = {mape_val:.2f}%")
        if mape_val < best_mape:
            best_mape = mape_val
            best_L = L
            best_model = tree_model
    print(f"--> Best Lag for horizon {horizon} days: L = {best_L} with Validation MAPE = {best_mape:.2f}%")
    X_train_full, y_train_full, weights_train_full, _ = construct_lagged_dataset(train_df, target_col, best_L, decay_rate, date_col)
    final_tree_model = DecisionTreeRegressor(max_depth=5, random_state=42)
    final_tree_model.fit(X_train_full, y_train_full, sample_weight=weights_train_full)
    
    # --------------------------
    # Added Output: Print Decision Tree Rules
    # --------------------------
    from sklearn.tree import export_text
    feature_names = [f"lag_{i}" for i in range(1, best_L+1)] + ["sin_dow", "cos_dow", "ma_feat"]
    tree_rules = export_text(final_tree_model, feature_names=feature_names)
    print("Decision Tree Rules:")
    print(tree_rules)
    # --------------------------
    
    test_mape_tree = rolling_forecast_evaluation(final_tree_model, test_df, target_col, best_L, horizon, decay_rate, start_idx=best_L)
    print(f"[Tree Model] Test MAPE for horizon {horizon} days: {test_mape_tree:.2f}%")
    results_tree[horizon] = {'Best_Lag': best_L, 'Test_MAPE': test_mape_tree}

print("\nFinal Results for Tree-Based Model (Forecast Horizon: Best Lag, Test MAPE %):")
for h in forecast_horizons:
    res = results_tree[h]
    print(f"  {h} days -> Best Lag: {res['Best_Lag']}, Test MAPE: {res['Test_MAPE']:.2f}%")

# ======== Additional: OLS-based AR and ARMA Models =========

# Prepare test series and cycle exogenous regressors (sin_dow and cos_dow)
series_ar = test_df[target_col].reset_index(drop=True)
exog_ar = test_df[['sin_dow', 'cos_dow']].reset_index(drop=True)

print("\nEvaluating OLS-based AR and ARMA Models on Test Set:")

for horizon in forecast_horizons:
    # AR(3) replaces the previous AR(1) candidate; note start_idx is now 3.
    mape_ar3 = rolling_forecast_evaluation_ar_ols(series_ar, exog_ar, p=3, forecast_horizon=horizon, start_idx=3)
    # AR(2) remains.
    mape_ar2 = rolling_forecast_evaluation_ar_ols(series_ar, exog_ar, p=2, forecast_horizon=horizon, start_idx=2)
    # ARMA(1,1) remains.
    mape_arma11 = rolling_forecast_evaluation_arma_ols(series_ar, exog_ar, forecast_horizon=horizon, start_idx=2)
    
    print(f"\nForecast Horizon {horizon} days:")
    print(f"  AR(3) Test MAPE:    {mape_ar3:.2f}%")
    print(f"  AR(2) Test MAPE:    {mape_ar2:.2f}%")
    print(f"  ARMA(1,1) Test MAPE:{mape_arma11:.2f}%")


#########
Total observations: 397
Training observations: 317
Test observations: 80

[Tree Model] Forecast Horizon: 3 days
  Candidate Lag L=3: Validation MAPE = 1.91%
  Candidate Lag L=5: Validation MAPE = 2.23%
  Candidate Lag L=7: Validation MAPE = 1.86%
  Candidate Lag L=9: Validation MAPE = 2.08%
--> Best Lag for horizon 3 days: L = 7 with Validation MAPE = 1.86%
Decision Tree Rules:
|--- lag_6 <= 94.84
|   |--- lag_6 <= 81.44
|   |   |--- lag_6 <= 71.13
|   |   |   |--- lag_6 <= 62.54
|   |   |   |   |--- lag_6 <= 58.05
|   |   |   |   |   |--- value: [57.81]
|   |   |   |   |--- lag_6 >  58.05
|   |   |   |   |   |--- value: [63.11]
|   |   |   |--- lag_6 >  62.54
|   |   |   |   |--- lag_1 <= 67.83
|   |   |   |   |   |--- value: [66.69]
|   |   |   |   |--- lag_1 >  67.83
|   |   |   |   |   |--- value: [71.22]
|   |   |--- lag_6 >  71.13
|   |   |   |--- lag_1 <= 71.79
|   |   |   |   |--- lag_6 <= 71.99
|   |   |   |   |   |--- value: [77.68]
|   |   |   |   |--- lag_6 >  71.99
|   |   |   |   |   |--- value: [70.03]
|   |   |   |--- lag_1 >  71.79
|   |   |   |   |--- lag_5 <= 71.16
|   |   |   |   |   |--- value: [75.19]
|   |   |   |   |--- lag_5 >  71.16
|   |   |   |   |   |--- value: [80.45]
|   |--- lag_6 >  81.44
|   |   |--- lag_6 <= 90.78
|   |   |   |--- lag_1 <= 78.91
|   |   |   |   |--- lag_1 <= 68.53
|   |   |   |   |   |--- value: [73.15]
|   |   |   |   |--- lag_1 >  68.53
|   |   |   |   |   |--- value: [78.83]
|   |   |   |--- lag_1 >  78.91
|   |   |   |   |--- lag_6 <= 83.21
|   |   |   |   |   |--- value: [85.04]
|   |   |   |   |--- lag_6 >  83.21
|   |   |   |   |   |--- value: [89.09]
|   |   |--- lag_6 >  90.78
|   |   |   |--- lag_1 <= 81.92
|   |   |   |   |--- lag_1 <= 78.01
|   |   |   |   |   |--- value: [82.53]
|   |   |   |   |--- lag_1 >  78.01
|   |   |   |   |   |--- value: [86.89]
|   |   |   |--- lag_1 >  81.92
|   |   |   |   |--- ma_feat <= 91.12
|   |   |   |   |   |--- value: [97.69]
|   |   |   |   |--- ma_feat >  91.12
|   |   |   |   |   |--- value: [91.92]
|--- lag_6 >  94.84
|   |--- lag_5 <= 94.43
|   |   |--- lag_2 <= 78.04
|   |   |   |--- lag_2 <= 75.26
|   |   |   |   |--- value: [88.45]
|   |   |   |--- lag_2 >  75.26
|   |   |   |   |--- value: [85.96]
|   |   |--- lag_2 >  78.04
|   |   |   |--- lag_5 <= 92.87
|   |   |   |   |--- lag_7 <= 108.67
|   |   |   |   |   |--- value: [96.73]
|   |   |   |   |--- lag_7 >  108.67
|   |   |   |   |   |--- value: [99.68]
|   |   |   |--- lag_5 >  92.87
|   |   |   |   |--- lag_4 <= 87.81
|   |   |   |   |   |--- value: [103.36]
|   |   |   |   |--- lag_4 >  87.81
|   |   |   |   |   |--- value: [99.17]
|   |--- lag_5 >  94.43
|   |   |--- lag_2 <= 88.39
|   |   |   |--- lag_1 <= 89.32
|   |   |   |   |--- sin_dow <= 0.22
|   |   |   |   |   |--- value: [90.78]
|   |   |   |   |--- sin_dow >  0.22
|   |   |   |   |   |--- value: [97.45]
|   |   |   |--- lag_1 >  89.32
|   |   |   |   |--- lag_1 <= 98.37
|   |   |   |   |   |--- value: [102.77]
|   |   |   |   |--- lag_1 >  98.37
|   |   |   |   |   |--- value: [108.63]
|   |   |--- lag_2 >  88.39
|   |   |   |--- lag_1 <= 105.10
|   |   |   |   |--- cos_dow <= 0.20
|   |   |   |   |   |--- value: [110.00]
|   |   |   |   |--- cos_dow >  0.20
|   |   |   |   |   |--- value: [105.12]
|   |   |   |--- lag_1 >  105.10
|   |   |   |   |--- lag_5 <= 103.64
|   |   |   |   |   |--- value: [109.77]
|   |   |   |   |--- lag_5 >  103.64
|   |   |   |   |   |--- value: [115.92]

[Tree Model] Test MAPE for horizon 3 days: 3.32%

[Tree Model] Forecast Horizon: 5 days
  Candidate Lag L=3: Validation MAPE = 2.31%
  Candidate Lag L=5: Validation MAPE = 1.92%
  Candidate Lag L=7: Validation MAPE = 1.65%
  Candidate Lag L=9: Validation MAPE = 2.35%
--> Best Lag for horizon 5 days: L = 7 with Validation MAPE = 1.65%
Decision Tree Rules:
|--- lag_6 <= 94.84
|   |--- lag_6 <= 81.44
|   |   |--- lag_6 <= 71.13
|   |   |   |--- lag_6 <= 62.54
|   |   |   |   |--- lag_6 <= 58.05
|   |   |   |   |   |--- value: [57.81]
|   |   |   |   |--- lag_6 >  58.05
|   |   |   |   |   |--- value: [63.11]
|   |   |   |--- lag_6 >  62.54
|   |   |   |   |--- lag_1 <= 67.83
|   |   |   |   |   |--- value: [66.69]
|   |   |   |   |--- lag_1 >  67.83
|   |   |   |   |   |--- value: [71.22]
|   |   |--- lag_6 >  71.13
|   |   |   |--- lag_1 <= 71.79
|   |   |   |   |--- lag_6 <= 71.99
|   |   |   |   |   |--- value: [77.68]
|   |   |   |   |--- lag_6 >  71.99
|   |   |   |   |   |--- value: [70.03]
|   |   |   |--- lag_1 >  71.79
|   |   |   |   |--- lag_5 <= 71.16
|   |   |   |   |   |--- value: [75.19]
|   |   |   |   |--- lag_5 >  71.16
|   |   |   |   |   |--- value: [80.45]
|   |--- lag_6 >  81.44
|   |   |--- lag_6 <= 90.78
|   |   |   |--- lag_1 <= 78.91
|   |   |   |   |--- lag_1 <= 68.53
|   |   |   |   |   |--- value: [73.15]
|   |   |   |   |--- lag_1 >  68.53
|   |   |   |   |   |--- value: [78.83]
|   |   |   |--- lag_1 >  78.91
|   |   |   |   |--- lag_6 <= 83.21
|   |   |   |   |   |--- value: [85.04]
|   |   |   |   |--- lag_6 >  83.21
|   |   |   |   |   |--- value: [89.09]
|   |   |--- lag_6 >  90.78
|   |   |   |--- lag_1 <= 81.92
|   |   |   |   |--- lag_1 <= 78.01
|   |   |   |   |   |--- value: [82.53]
|   |   |   |   |--- lag_1 >  78.01
|   |   |   |   |   |--- value: [86.89]
|   |   |   |--- lag_1 >  81.92
|   |   |   |   |--- ma_feat <= 91.12
|   |   |   |   |   |--- value: [97.69]
|   |   |   |   |--- ma_feat >  91.12
|   |   |   |   |   |--- value: [91.92]
|--- lag_6 >  94.84
|   |--- lag_5 <= 94.43
|   |   |--- lag_2 <= 78.04
|   |   |   |--- lag_2 <= 75.26
|   |   |   |   |--- value: [88.45]
|   |   |   |--- lag_2 >  75.26
|   |   |   |   |--- value: [85.96]
|   |   |--- lag_2 >  78.04
|   |   |   |--- lag_5 <= 92.87
|   |   |   |   |--- lag_7 <= 108.67
|   |   |   |   |   |--- value: [96.73]
|   |   |   |   |--- lag_7 >  108.67
|   |   |   |   |   |--- value: [99.68]
|   |   |   |--- lag_5 >  92.87
|   |   |   |   |--- lag_4 <= 87.81
|   |   |   |   |   |--- value: [103.36]
|   |   |   |   |--- lag_4 >  87.81
|   |   |   |   |   |--- value: [99.17]
|   |--- lag_5 >  94.43
|   |   |--- lag_2 <= 88.39
|   |   |   |--- lag_1 <= 89.32
|   |   |   |   |--- sin_dow <= 0.22
|   |   |   |   |   |--- value: [90.78]
|   |   |   |   |--- sin_dow >  0.22
|   |   |   |   |   |--- value: [97.45]
|   |   |   |--- lag_1 >  89.32
|   |   |   |   |--- lag_1 <= 98.37
|   |   |   |   |   |--- value: [102.77]
|   |   |   |   |--- lag_1 >  98.37
|   |   |   |   |   |--- value: [108.63]
|   |   |--- lag_2 >  88.39
|   |   |   |--- lag_1 <= 105.10
|   |   |   |   |--- cos_dow <= 0.20
|   |   |   |   |   |--- value: [110.00]
|   |   |   |   |--- cos_dow >  0.20
|   |   |   |   |   |--- value: [105.12]
|   |   |   |--- lag_1 >  105.10
|   |   |   |   |--- lag_5 <= 103.64
|   |   |   |   |   |--- value: [109.77]
|   |   |   |   |--- lag_5 >  103.64
|   |   |   |   |   |--- value: [115.92]

[Tree Model] Test MAPE for horizon 5 days: 2.74%

[Tree Model] Forecast Horizon: 7 days
  Candidate Lag L=3: Validation MAPE = 2.20%
  Candidate Lag L=5: Validation MAPE = 1.48%
  Candidate Lag L=7: Validation MAPE = 1.48%
  Candidate Lag L=9: Validation MAPE = 2.44%
--> Best Lag for horizon 7 days: L = 7 with Validation MAPE = 1.48%
Decision Tree Rules:
|--- lag_6 <= 94.84
|   |--- lag_6 <= 81.44
|   |   |--- lag_6 <= 71.13
|   |   |   |--- lag_6 <= 62.54
|   |   |   |   |--- lag_6 <= 58.05
|   |   |   |   |   |--- value: [57.81]
|   |   |   |   |--- lag_6 >  58.05
|   |   |   |   |   |--- value: [63.11]
|   |   |   |--- lag_6 >  62.54
|   |   |   |   |--- lag_1 <= 67.83
|   |   |   |   |   |--- value: [66.69]
|   |   |   |   |--- lag_1 >  67.83
|   |   |   |   |   |--- value: [71.22]
|   |   |--- lag_6 >  71.13
|   |   |   |--- lag_1 <= 71.79
|   |   |   |   |--- lag_6 <= 71.99
|   |   |   |   |   |--- value: [77.68]
|   |   |   |   |--- lag_6 >  71.99
|   |   |   |   |   |--- value: [70.03]
|   |   |   |--- lag_1 >  71.79
|   |   |   |   |--- lag_5 <= 71.16
|   |   |   |   |   |--- value: [75.19]
|   |   |   |   |--- lag_5 >  71.16
|   |   |   |   |   |--- value: [80.45]
|   |--- lag_6 >  81.44
|   |   |--- lag_6 <= 90.78
|   |   |   |--- lag_1 <= 78.91
|   |   |   |   |--- lag_1 <= 68.53
|   |   |   |   |   |--- value: [73.15]
|   |   |   |   |--- lag_1 >  68.53
|   |   |   |   |   |--- value: [78.83]
|   |   |   |--- lag_1 >  78.91
|   |   |   |   |--- lag_6 <= 83.21
|   |   |   |   |   |--- value: [85.04]
|   |   |   |   |--- lag_6 >  83.21
|   |   |   |   |   |--- value: [89.09]
|   |   |--- lag_6 >  90.78
|   |   |   |--- lag_1 <= 81.92
|   |   |   |   |--- lag_1 <= 78.01
|   |   |   |   |   |--- value: [82.53]
|   |   |   |   |--- lag_1 >  78.01
|   |   |   |   |   |--- value: [86.89]
|   |   |   |--- lag_1 >  81.92
|   |   |   |   |--- ma_feat <= 91.12
|   |   |   |   |   |--- value: [97.69]
|   |   |   |   |--- ma_feat >  91.12
|   |   |   |   |   |--- value: [91.92]
|--- lag_6 >  94.84
|   |--- lag_5 <= 94.43
|   |   |--- lag_2 <= 78.04
|   |   |   |--- lag_2 <= 75.26
|   |   |   |   |--- value: [88.45]
|   |   |   |--- lag_2 >  75.26
|   |   |   |   |--- value: [85.96]
|   |   |--- lag_2 >  78.04
|   |   |   |--- lag_5 <= 92.87
|   |   |   |   |--- lag_7 <= 108.67
|   |   |   |   |   |--- value: [96.73]
|   |   |   |   |--- lag_7 >  108.67
|   |   |   |   |   |--- value: [99.68]
|   |   |   |--- lag_5 >  92.87
|   |   |   |   |--- lag_4 <= 87.81
|   |   |   |   |   |--- value: [103.36]
|   |   |   |   |--- lag_4 >  87.81
|   |   |   |   |   |--- value: [99.17]
|   |--- lag_5 >  94.43
|   |   |--- lag_2 <= 88.39
|   |   |   |--- lag_1 <= 89.32
|   |   |   |   |--- sin_dow <= 0.22
|   |   |   |   |   |--- value: [90.78]
|   |   |   |   |--- sin_dow >  0.22
|   |   |   |   |   |--- value: [97.45]
|   |   |   |--- lag_1 >  89.32
|   |   |   |   |--- lag_1 <= 98.37
|   |   |   |   |   |--- value: [102.77]
|   |   |   |   |--- lag_1 >  98.37
|   |   |   |   |   |--- value: [108.63]
|   |   |--- lag_2 >  88.39
|   |   |   |--- lag_1 <= 105.10
|   |   |   |   |--- cos_dow <= 0.20
|   |   |   |   |   |--- value: [110.00]
|   |   |   |   |--- cos_dow >  0.20
|   |   |   |   |   |--- value: [105.12]
|   |   |   |--- lag_1 >  105.10
|   |   |   |   |--- lag_5 <= 103.64
|   |   |   |   |   |--- value: [109.77]
|   |   |   |   |--- lag_5 >  103.64
|   |   |   |   |   |--- value: [115.92]

[Tree Model] Test MAPE for horizon 7 days: 2.38%

[Tree Model] Forecast Horizon: 9 days
  Candidate Lag L=3: Validation MAPE = 1.88%
  Candidate Lag L=5: Validation MAPE = 1.36%
  Candidate Lag L=7: Validation MAPE = 1.45%
  Candidate Lag L=9: Validation MAPE = 2.50%
--> Best Lag for horizon 9 days: L = 5 with Validation MAPE = 1.36%
Decision Tree Rules:
|--- lag_1 <= 88.51
|   |--- lag_1 <= 79.32
|   |   |--- lag_5 <= 73.95
|   |   |   |--- lag_1 <= 68.06
|   |   |   |   |--- lag_5 <= 60.65
|   |   |   |   |   |--- value: [59.30]
|   |   |   |   |--- lag_5 >  60.65
|   |   |   |   |   |--- value: [64.95]
|   |   |   |--- lag_1 >  68.06
|   |   |   |   |--- lag_1 <= 76.35
|   |   |   |   |   |--- value: [69.65]
|   |   |   |   |--- lag_1 >  76.35
|   |   |   |   |   |--- value: [73.93]
|   |   |--- lag_5 >  73.95
|   |   |   |--- lag_1 <= 71.13
|   |   |   |   |--- lag_5 <= 74.18
|   |   |   |   |   |--- value: [84.19]
|   |   |   |   |--- lag_5 >  74.18
|   |   |   |   |   |--- value: [71.73]
|   |   |   |--- lag_1 >  71.13
|   |   |   |   |--- lag_2 <= 80.23
|   |   |   |   |   |--- value: [81.38]
|   |   |   |   |--- lag_2 >  80.23
|   |   |   |   |   |--- value: [76.05]
|   |--- lag_1 >  79.32
|   |   |--- lag_5 <= 82.78
|   |   |   |--- lag_2 <= 78.90
|   |   |   |   |--- lag_4 <= 71.71
|   |   |   |   |   |--- value: [80.53]
|   |   |   |   |--- lag_4 >  71.71
|   |   |   |   |   |--- value: [90.16]
|   |   |   |--- lag_2 >  78.90
|   |   |   |   |--- lag_1 <= 87.19
|   |   |   |   |   |--- value: [76.66]
|   |   |   |   |--- lag_1 >  87.19
|   |   |   |   |   |--- value: [82.24]
|   |   |--- lag_5 >  82.78
|   |   |   |--- lag_5 <= 110.92
|   |   |   |   |--- lag_2 <= 86.51
|   |   |   |   |   |--- value: [92.01]
|   |   |   |   |--- lag_2 >  86.51
|   |   |   |   |   |--- value: [87.06]
|   |   |   |--- lag_5 >  110.92
|   |   |   |   |--- lag_3 <= 98.40
|   |   |   |   |   |--- value: [97.24]
|   |   |   |   |--- lag_3 >  98.40
|   |   |   |   |   |--- value: [99.51]
|--- lag_1 >  88.51
|   |--- lag_5 <= 93.97
|   |   |--- lag_1 <= 100.98
|   |   |   |--- lag_2 <= 91.11
|   |   |   |   |--- lag_4 <= 75.72
|   |   |   |   |   |--- value: [87.30]
|   |   |   |   |--- lag_4 >  75.72
|   |   |   |   |   |--- value: [98.02]
|   |   |   |--- lag_2 >  91.11
|   |   |   |   |--- lag_1 <= 94.67
|   |   |   |   |   |--- value: [85.84]
|   |   |   |   |--- lag_1 >  94.67
|   |   |   |   |   |--- value: [89.37]
|   |   |--- lag_1 >  100.98
|   |   |   |--- lag_1 <= 106.77
|   |   |   |   |--- sin_dow <= -0.88
|   |   |   |   |   |--- value: [101.63]
|   |   |   |   |--- sin_dow >  -0.88
|   |   |   |   |   |--- value: [95.37]
|   |   |   |--- lag_1 >  106.77
|   |   |   |   |--- lag_5 <= 91.95
|   |   |   |   |   |--- value: [98.83]
|   |   |   |   |--- lag_5 >  91.95
|   |   |   |   |   |--- value: [102.92]
|   |--- lag_5 >  93.97
|   |   |--- lag_3 <= 103.64
|   |   |   |--- lag_2 <= 87.79
|   |   |   |   |--- ma_feat <= 90.39
|   |   |   |   |   |--- value: [100.24]
|   |   |   |   |--- ma_feat >  90.39
|   |   |   |   |   |--- value: [104.40]
|   |   |   |--- lag_2 >  87.79
|   |   |   |   |--- lag_1 <= 102.34
|   |   |   |   |   |--- value: [107.18]
|   |   |   |   |--- lag_1 >  102.34
|   |   |   |   |   |--- value: [110.88]
|   |   |--- lag_3 >  103.64
|   |   |   |--- lag_3 <= 107.47
|   |   |   |   |--- lag_2 <= 96.99
|   |   |   |   |   |--- value: [89.36]
|   |   |   |   |--- lag_2 >  96.99
|   |   |   |   |   |--- value: [90.07]
|   |   |   |--- lag_3 >  107.47
|   |   |   |   |--- lag_1 <= 91.00
|   |   |   |   |   |--- value: [91.89]
|   |   |   |   |--- lag_1 >  91.00
|   |   |   |   |   |--- value: [93.86]

[Tree Model] Test MAPE for horizon 9 days: 2.25%

Final Results for Tree-Based Model (Forecast Horizon: Best Lag, Test MAPE %):
  3 days -> Best Lag: 7, Test MAPE: 3.32%
  5 days -> Best Lag: 7, Test MAPE: 2.74%
  7 days -> Best Lag: 7, Test MAPE: 2.38%
  9 days -> Best Lag: 5, Test MAPE: 2.25%

Evaluating OLS-based AR and ARMA Models on Test Set:

Forecast Horizon 3 days:
  AR(3) Test MAPE:    3.90%
  AR(2) Test MAPE:    5.02%
  ARMA(1,1) Test MAPE:4.35%

Forecast Horizon 5 days:
  AR(3) Test MAPE:    2.85%
  AR(2) Test MAPE:    4.56%
  ARMA(1,1) Test MAPE:3.92%

Forecast Horizon 7 days:
  AR(3) Test MAPE:    2.70%
  AR(2) Test MAPE:    4.85%
  ARMA(1,1) Test MAPE:4.23%

Forecast Horizon 9 days:
  AR(3) Test MAPE:    2.96%
  AR(2) Test MAPE:    5.42%
  ARMA(1,1) Test MAPE:4.76%
