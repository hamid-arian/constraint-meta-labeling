import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, KFold # Added KFold for cross-validation ideas
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error # For evaluation

# --- Data Loading and Initial Preprocessing (Simulating In [2], In [3]) ---
# This part is assumed to be similar to your original notebook.
# For brevity, I'm not reproducing the exact data loading loop.
# Let's assume 'data_list' and 'min_rows' are loaded as per your notebook.
# Example:
# datalist = []
# min_rows = float('inf')
# file_paths = ["BTC-USD_historical_data.csv", "ETH-USD_historical_data.csv", ...] # Your file paths
# for filename in sorted(os.listdir("/home/yichuan/ywc/meta-labeling/cryptocurrency")): [cite: 2]
# if filename.endswith('.csv'): [cite: 2]
# file_path = os.path.join("/home/yichuan/ywc/meta-labeling/cryptocurrency", filename) [cite: 2]
# df = pd.read_csv(file_path) [cite: 2]
# df = df.drop(columns=["Date"]) # [cite: 2] Assuming 'Date' is not needed for modeling
# datalist.append(df.values) [cite: 2]
# min_rows = min(min_rows, df.shape[0]) [cite: 2]
# data_list = [data[-min_rows:] for data in datalist] # [cite: 2]
# print(min_rows, len(data_list)) # [cite: 2]

# --- Log Volume Transformation (Simulating In [3]) ---
# def logVolume(X: np.ndarray): # [cite: 2]
#     volume = X[:, -1] # [cite: 2]
#     volume_log = np.log(volume + 1) # [cite: 2] # Add 1 to prevent log(0)
#     mean_log = np.mean(volume_log) # [cite: 2]
#     std_log = np.std(volume_log) # [cite: 2]
#     # Handle std_log == 0 to prevent division by zero
#     if std_log == 0:
#         volume_std = np.zeros_like(volume_log)
#     else:
#         volume_std = (volume_log - mean_log) / std_log # [cite: 2]
#     X[:, -1] = volume_std # [cite: 2]
#     return X

# for i in range(len(data_list)): # Original used `for data in data_list: data = logVolume(data)` which doesn't modify inplace
# data_list[i] = logVolume(data_list[i].copy()) # [cite: 2] Apply to a copy to ensure changes stick

# Placeholder for loaded data_list
# For demonstration, let's assume data_list is populated correctly
# Example: data_list = [np.random.rand(1878, 6) for _ in range(4)] # [cite: 2]
# min_rows = 1878

# --- Modified Label Generation ---

# In [4] modification for triple_barrier
def calculate_volatility(price_series, lookback_window=20):
    """
    Calculates historical volatility (e.g., standard deviation of log returns).
    This is a placeholder; you might use more sophisticated methods.
    """
    log_returns = np.log(price_series / np.roll(price_series, 1))
    if len(log_returns) < lookback_window:
        return np.full_like(price_series, 0.01) # Default volatility if not enough data
    volatility = pd.Series(log_returns).rolling(window=lookback_window).std().bfill().values
    return np.maximum(volatility, 0.001) # Ensure non-zero volatility

def triple_barrier(close: np.ndarray, days=10, initial_ptsl_factors=[1.5, 1.5], return_min=0.005, use_dynamic_ptsl=True, lookback_volatility=20): # [cite: 3]
    """
    Creates labels based on the triple barrier method.
    - initial_ptsl_factors: Multipliers for volatility to set profit-take/stop-loss.
    - return_min: Minimum return to consider for a +1 or -1 label, if not using dynamic.
    - use_dynamic_ptsl: If True, ptsl levels are based on volatility.
    """
    bin_labels = np.zeros(close.size, dtype=int) # [cite: 3]
    daily_volatility = calculate_volatility(close, lookback_window=lookback_volatility)

    for i in range(close.size):
        if use_dynamic_ptsl:
            pt_level = close[i] * (1 + daily_volatility[i] * initial_ptsl_factors[0])
            sl_level = close[i] * (1 - daily_volatility[i] * initial_ptsl_factors[1])
        else:
            # Original static ptsl logic can be used if desired, or incorporate return_min
            pt_level = close[i] * (1 + 0.05) # Example from original [cite: 3]
            sl_level = close[i] * (1 - 0.05) # Example from original [cite: 3]


        for d in range(days): # [cite: 3]
            index = min(i + d + 1, close.size - 1) # [cite: 3]

            # Profit Take
            if close[index] >= pt_level:
                # Ensure minimum return if not dynamic or if pt_level itself doesn't guarantee it
                if not use_dynamic_ptsl and (close[index] / close[i] - 1) < return_min :
                    bin_labels[i] = 0 # Not enough return
                else:
                    bin_labels[i] = 1
                break
            # Stop Loss
            elif close[index] <= sl_level:
                # Ensure minimum return (loss) if not dynamic
                if not use_dynamic_ptsl and (close[i] / close[index] - 1) < return_min: # checks if loss is significant enough
                     bin_labels[i] = 0
                else:
                    bin_labels[i] = -1
                break
            # If neither barrier is hit by the end of `days`, label remains 0 (timeout)
    return bin_labels

# In [5] modification
# binmat = np.full((min_rows, len(data_list)), 0) # Initialize with 0 for timeout
# for i in range(len(data_list)):
#     # TODO: Optimize `days`, `initial_ptsl_factors`, `return_min`, `lookback_volatility`
#     binmat[:, i] = triple_barrier(data_list[i][:, 3], days=10, initial_ptsl_factors=[1.5,1.5], return_min=0.005, use_dynamic_ptsl=True) # [cite: 3]
# print(binmat)

# --- Modified Continuous Signal Generation ---
# In [6] modification
# TODO: Optimize lmbda and alpha via cross-validation
# lmbda = 0.5 # [cite: 3] For smoothing
# alpha = 0.5 # [cite: 3] For combining smoothed and model-based labels

# label_mat = np.zeros((min_rows, len(data_list))) # Initialize with a neutral value if needed

# for j in range(len(data_list)): # [cite: 4]
#     y_sm = np.zeros(min_rows) # [cite: 4] Smoothed labels
#     y_so = np.zeros(min_rows) # [cite: 4] Soft (model-based) labels

    # Smoothed labels (EMA-like)
#     for i in range(min_rows): # [cite: 4]
#         if i == 0:
#             y_sm[i] = binmat[i, j] # Or a neutral starting point like 0
#         else:
#             y_sm[i] = lmbda * binmat[i, j] + (1 - lmbda) * y_sm[i-1] # [cite: 4]

    # Soft labels from a simple model (e.g., Logistic Regression on lagged features)
    # This part requires careful feature engineering for the logistic regression.
    # For simplicity, the original leave-one-out structure is kept,
    # but features should be lagged to prevent lookahead.
#     temp_data = data_list[j]
#     num_features = temp_data.shape[1]

#     for i in range(min_rows): # [cite: 4]
#         if i < 1: # Need at least one previous day for lagged features
#             y_so[i] = 0 # Neutral value if no features
#             continue

        # Create lagged features (example: use previous day's data)
        # X_train_loo = np.delete(temp_data[:i], i-1 if i > 0 else 0, axis=0) # [cite: 4] Incorrect for time series
        # y_train_loo = np.delete(binmat[:i,j], i-1 if i > 0 else 0, axis=0) # [cite: 4]

        # Corrected approach for time-series Loo-like prediction:
        # Train on data up to i-k (k is holdout for feature generation)
        # Predict for i using features available at i-1
        # For this example, let's use a simpler rolling window or fixed training set for y_so
        # To avoid data leakage, ensure features for x_test_loo are strictly from t-1 or earlier.

#         if i > 20: # Ensure enough data for training the logistic model
#             # Example: train on a rolling window or an expanding window up to i-1
#             # For this example, let's use a fixed portion of the early data, which isn't ideal for time series
#             # A proper implementation would involve careful time-series cross-validation or walk-forward for this sub-model
            
#             # Simplified: Use features from t-1 to predict label at t
#             # This part needs careful design to avoid lookahead bias for y_so generation.
#             # The original code's direct use of data_list[j][i,:] for x_test might leak info if not handled carefully with lags.
            
#             # Placeholder: For simplicity, we'll assume y_so is generated appropriately.
#             # In a real scenario, this logistic regression would need its own features (e.g., lagged returns, volatility).
#             # scaler = StandardScaler() # [cite: 4]
#             # X_train_scaled_loo = scaler.fit_transform(some_lagged_features_train) # [cite: 4]
#             # x_test_scaled_loo = scaler.transform(some_lagged_features_test) # [cite: 4]
#             # model_lr = LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='ovr') # [cite: 4]
#             # model_lr.fit(X_train_scaled_loo, y_train_loo_for_lr) # y_train_loo_for_lr would be binmat
            
#             # prob = model_lr.predict_proba(x_test_scaled_loo) # [cite: 4]
#             # class_labels = model_lr.classes_ # [cite: 4]
#             # y_so[i] = np.sum(class_labels * prob[0]) # [cite: 4] # This is for {-1, 0, 1} if classes are such.
                                                    # If classes are 0,1, then this needs adjustment.
                                                    # Assuming binmat is {-1,0,1}, LogisticRegression might struggle without mapping.
                                                    # Let's assume y_so is derived appropriately, perhaps predicting P(up)-P(down)

#             # Simplified y_so based on previous smoothed label (less sophisticated but avoids complex Loo for this example)
#             y_so[i] = y_sm[i-1] if i > 0 else 0


#     label_mat[:, j] = alpha * y_sm + (1 - alpha) * y_so # [cite: 4]
# label_mat = (label_mat + 1) / 2  # Scale to [0, 1] if primary model expects probabilities [cite: 4]
# print("Label mat (targets for primary model):", label_mat)


# --- Primary Model with Constraints (Modified from In [8]) ---
def primary_model(asset_idx, full_asset_data, continuous_labels_for_asset, train_ratio): # [cite: 5]
    X_asset = np.array(full_asset_data) # [cite: 5]
    y_asset_continuous = continuous_labels_for_asset # These are from label_mat

    n_total = len(X_asset)
    n_train = int(n_total * train_ratio) # [cite: 5]

    X_train_pm = X_asset[:n_train] # [cite: 5]
    X_test_pm = X_asset[n_train:] # [cite: 5]
    y_train_pm = y_asset_continuous[:n_train] # [cite: 5]
    # y_test_pm is used by the secondary model later as the "true" primary model target for meta_label generation

    scaler_pm = StandardScaler() # [cite: 5]
    X_train_pm_scaled = scaler_pm.fit_transform(X_train_pm) # [cite: 5]
    X_test_pm_scaled = scaler_pm.transform(X_test_pm) # [cite: 5]

    # TODO: Hyperparameter tuning for XGBoost (e.g., using GridSearchCV on X_train_pm_scaled, y_train_pm)
    params = {
        "objective": "reg:squarederror",  # Changed from reg:logistic [cite: 5]
        "eval_metric": "rmse", # [cite: 5]
        "learning_rate": 0.05, # Adjusted, tune this [cite: 5]
        "max_depth": 4,        # Adjusted, tune this [cite: 5]
        "subsample": 0.8, # [cite: 5]
        "colsample_bytree": 0.8, # Added for regularization
        "alpha": 0.1 # L1 reg
        # "tree_method": "hist" # Faster for large datasets
    }

    dtrain_pm = xgb.DMatrix(X_train_pm_scaled, label=y_train_pm) # [cite: 5]
    dtest_pm = xgb.DMatrix(X_test_pm_scaled) # [cite: 5]

    # TODO: Implement proper time-series cross-validation for num_boost_round selection if not using early stopping
    model_pm = xgb.train(params, dtrain_pm, num_boost_round=150, # Adjusted, tune this [cite: 5]
                         evals=[(dtrain_pm, 'train')], verbose_eval=False) # Example for monitoring

    y_pred_primary_on_test = model_pm.predict(dtest_pm) # [cite: 5]
    
    # We need y_pred_primary_on_train as well if secondary model trains on primary model's train performance
    # y_pred_primary_on_train = model_pm.predict(dtrain_pm) # Or on a validation set

    return y_pred_primary_on_test, scaler_pm # Return scaler for secondary model if features are shared

# --- Secondary model (meta-labeling) (Modified from In [9]) ---

class RegressionNN(nn.Module): # [cite: 5]
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential( # [cite: 8]
            nn.Linear(input_dim, 32), # Increased layer size, tune this [cite: 8]
            nn.ReLU(), # [cite: 8]
            nn.BatchNorm1d(32), # Added BatchNorm
            nn.Dropout(0.3), # Adjusted, tune this [cite: 8]
            nn.Linear(32, 16), # [cite: 8]
            nn.ReLU(), # [cite: 8]
            nn.BatchNorm1d(16), # Added BatchNorm
            nn.Dropout(0.3), # Adjusted, tune this
            nn.Linear(16, 1) # [cite: 8]
        )

    def forward(self, x): # [cite: 8]
        return self.net(x)

# Custom loss from original, ensure lambda_l2 is not double-counting with weight_decay in optimizer
def custom_loss_nn(outputs, targets, model_nn_params, lambda_l2_custom): # [cite: 8]
    mse_loss = nn.MSELoss()(outputs, targets) # [cite: 8]
    # If using weight_decay in Adam, this manual L2 might be redundant or need coordination
    l2_reg = sum(p.pow(2).sum() for p in model_nn_params if p.requires_grad) # [cite: 8]
    return mse_loss + lambda_l2_custom * l2_reg # [cite: 8]


def secondary_model(
    asset_idx,
    X_test_features_from_primary, # These are the original features for the test set period
    primary_model_targets_on_test_set, # These are y_test_pm (e.g. label_mat[n_train:, asset_idx])
    primary_model_predictions_on_test_set, # These are y_pred_primary_on_test
    primary_signals_to_refine, # This is y_pred_primary_on_test, passed for modification
    # scaler_pm, # If using same scaled features
    # TODO: Tune these hyperparameters
    nn_input_dim,
    threshold_meta_adjustment=0.05, # When to apply the meta-model's correction
    lambda_l2_custom_nn=1e-5, # [cite: 6]
    lr_nn=0.001, # [cite: 10]
    weight_decay_nn=0.0001 # Adam's L2, distinct from custom_loss_nn's lambda_l2
    ): # [cite: 6]

    # 1. Define Meta-Labels: Error of the primary model
    meta_labels_true_error = primary_model_targets_on_test_set - primary_model_predictions_on_test_set
    
    # 2. Input Features for Secondary Model:
    # Option A: Use original features (X_test_features_from_primary)
    # Option B: Use primary model's predictions as features (or part of features)
    # Option C: Combine original features and primary model's predictions/confidence
    # For this example, let's use original features + primary model's prediction as input to secondary
    
    # Scale original features if not already scaled appropriately for NN
    # For simplicity, assume X_test_features_from_primary is already scaled if needed.
    # If scaler_pm was used for X_test_pm_scaled, and those are X_test_features_from_primary, then they are scaled.
    
    # Add primary model's prediction as a feature for the secondary model
    secondary_model_input_features = np.concatenate(
        (X_test_features_from_primary, primary_model_predictions_on_test_set.reshape(-1, 1)), axis=1
    )
    current_nn_input_dim = secondary_model_input_features.shape[1]


    # Prepare data for PyTorch
    X_meta_tensor = torch.tensor(secondary_model_input_features, dtype=torch.float32)
    meta_labels_error_tensor = torch.tensor(meta_labels_true_error, dtype=torch.float32).view(-1, 1) # [cite: 6]

    # Split data for secondary model training and testing
    # This split is on the *test set* of the primary model
    X_train_meta, X_test_meta, y_train_meta, y_test_meta = train_test_split(
        X_meta_tensor, meta_labels_error_tensor, test_size=0.3, random_state=42, shuffle=False # No shuffle for time series generally [cite: 6]
    )
    # print("y_test_meta shape (secondary model):", y_test_meta.shape) # [cite: 6]

    model_nn = RegressionNN(input_dim=current_nn_input_dim) # [cite: 6]
    # If using custom L2, set weight_decay=0 in optimizer to avoid double penalty
    optimizer_nn = optim.Adam(model_nn.parameters(), lr=lr_nn, weight_decay=weight_decay_nn) # [cite: 10]
    
    num_epochs_nn = 200 # Reduced for example, tune this [cite: 10]
    batch_size_nn = 16 # Reduced for example, tune this [cite: 10]
    # train_losses_nn = [] # [cite: 10]

    # Training loop for secondary model
    for epoch in range(num_epochs_nn): # [cite: 10]
        model_nn.train()
        permutation = torch.randperm(X_train_meta.size(0)) # [cite: 10]
        for i in range(0, X_train_meta.size(0), batch_size_nn): # [cite: 10]
            indices = permutation[i:i+batch_size_nn] # [cite: 10]
            batch_X_meta, batch_y_meta = X_train_meta[indices], y_train_meta[indices] # [cite: 10]
            
            outputs_meta = model_nn(batch_X_meta) # [cite: 10]
            # Using standard MSE loss here, or use your custom_loss_nn
            # loss = custom_loss_nn(outputs_meta, batch_y_meta, model_nn.parameters(), lambda_l2_custom_nn) # [cite: 10]
            loss = nn.MSELoss()(outputs_meta, batch_y_meta) # Simpler alternative

            optimizer_nn.zero_grad() # [cite: 10]
            loss.backward() # [cite: 10]
            optimizer_nn.step() # [cite: 10]
        
        # Optional: Log training loss
        # with torch.no_grad():
        #     model_nn.eval()
        #     train_pred_meta = model_nn(X_train_meta)
        #     current_train_loss = nn.MSELoss()(train_pred_meta, y_train_meta).item()
        #     train_losses_nn.append(current_train_loss)
        #     if (epoch + 1) % 50 == 0: # [cite: 10]
        #         print(f'Asset {asset_idx}, NN Epoch [{epoch+1}/{num_epochs_nn}], Train Loss: {current_train_loss:.4f}')


    # Evaluate secondary model on its test set
    model_nn.eval()
    with torch.no_grad(): # [cite: 11]
        test_pred_meta_errors = model_nn(X_test_meta) # [cite: 11]
        final_test_loss_meta = nn.MSELoss()(test_pred_meta_errors, y_test_meta).item() # [cite: 11]
        # print(f'\nAsset {asset_idx}, NN Final Test Loss (predicting error): {final_test_loss_meta:.4f}')

    # Predict errors for the entire primary model's test set (for refinement)
    with torch.no_grad():
        all_pred_errors_meta = model_nn(X_meta_tensor).numpy().flatten()

    # Refine primary signals
    refined_signals = primary_signals_to_refine.copy() # [cite: 11]
    
    # Adjustment logic: signal = original_signal - predicted_error
    # Only adjust if the predicted error is significant (above threshold)
    adjustment_mask = np.abs(all_pred_errors_meta) > threshold_meta_adjustment # [cite: 11]
    
    refined_signals[adjustment_mask] = primary_signals_to_refine[adjustment_mask] - all_pred_errors_meta[adjustment_mask]
    
    # Ensure signals remain in a valid range (e.g., [0,1] if they are probabilities/weights)
    refined_signals = np.clip(refined_signals, 0, 1) # Assuming signals should be between 0 and 1

    # print(f"Asset {asset_idx}: Number of signals refined by meta-model: {np.sum(adjustment_mask)}")
    return refined_signals


# --- Main Loop for Training and Prediction (Modified from In [10]) ---
# train_ratio = 0.8 # [cite: 7]
# num_assets = len(data_list)
# n_total_overall = data_list[0].shape[0] # Assuming all have min_rows
# n_train_overall = int(n_total_overall * train_ratio) # [cite: 7]
# n_test_overall = n_total_overall - n_train_overall # [cite: 7]

# # Store results
# all_primary_signals_test = np.zeros((n_test_overall, num_assets)) # [cite: 7]
# all_refined_signals_test = np.zeros((n_test_overall, num_assets)) # [cite: 7]
# primary_model_targets_for_meta = np.zeros((n_test_overall, num_assets)) # To store y_test_pm for each asset

# for i in range(num_assets):
#     print(f"Processing Asset {i}...")
    # Generate continuous labels (targets for primary model) for this asset
    # This assumes 'label_mat' is generated correctly using the modified continuous signal generation
    # For this example, let's use a placeholder for label_mat or regenerate it simply
    # Example: Re-generating simple binmat and y_sm for target
#     current_binmat = triple_barrier(data_list[i][:, 3], days=10, use_dynamic_ptsl=True)
#     y_sm_temp = np.zeros(min_rows)
#     for k_idx in range(min_rows):
#         if k_idx == 0: y_sm_temp[k_idx] = current_binmat[k_idx]
#         else: y_sm_temp[k_idx] = 0.5 * current_binmat[k_idx] + (1 - 0.5) * y_sm_temp[k_idx-1]
#     current_continuous_labels = (y_sm_temp + 1) / 2 # Target for primary model

#     primary_preds_test, scaler_for_asset = primary_model(
#         asset_idx=i,
#         full_asset_data=data_list[i], # Pass all features for this asset [cite: 5]
#         continuous_labels_for_asset=current_continuous_labels,
#         train_ratio=train_ratio
#     )
#     all_primary_signals_test[:, i] = primary_preds_test

    # Store the true targets the primary model was trying to predict on its test set
#     primary_model_targets_for_meta[:, i] = current_continuous_labels[n_train_overall:]
    
    # Original features for the primary model's test set (needed for secondary model)
#     X_test_features_pm = data_list[i][n_train_overall:]
    # Scale them if the primary model's scaler was used and features are to be scaled for NN
    # X_test_features_pm_scaled = scaler_for_asset.transform(X_test_features_pm)


#     refined_s = secondary_model(
#         asset_idx=i,
#         X_test_features_from_primary=X_test_features_pm, # Pass unscaled or scaled based on NN's needs [cite: 6]
#         primary_model_targets_on_test_set=primary_model_targets_for_meta[:, i],
#         primary_model_predictions_on_test_set=primary_preds_test,
#         primary_signals_to_refine=primary_preds_test,
#         nn_input_dim=data_list[i].shape[1] # Original feature dim, will be +1 in function [cite: 6]
#     )
#     all_refined_signals_test[:, i] = refined_s

# print("Primary Signals (test set head):", all_primary_signals_test[:5]) # [cite: 7]
# print("Refined Signals (test set head):", all_refined_signals_test[:5]) # [cite: 7]

# not_equal = all_primary_signals_test != all_refined_signals_test # [cite: 9]
# refined_num_per_asset = np.sum(not_equal, axis=0) # [cite: 9]
# print("Number of refined signals per asset:", refined_num_per_asset) # [cite: 9]


# --- Portfolio Construction and Evaluation (Modified from In [12] - [17]) ---
# The portfolio construction (`construct_portfolio`) and Sharpe calculation (`calculate_sharpe_ratio`)
# can remain largely the same, but they will now operate on `all_primary_signals_test`
# and `all_refined_signals_test`.

# Make sure to pass the correct covariance matrix calculation (e.g., using training data returns)
# and sector mappings as in your original code.

# Example:
# Closemat_all = np.zeros((min_rows, num_assets)) # [cite: 10]
# for i in range(num_assets):
# Closemat_all[:,i] = data_list[i][:,3] # [cite: 10]

# Returns for covariance calculation (use data corresponding to the primary model's training period)
# Close_train_pm = Closemat_all[:n_train_overall, :] # [cite: 10]
# returns_train_pm = np.log(Close_train_pm[1:] / Close_train_pm[:-1]) # [cite: 10]
# if np.any(np.isinf(returns_train_pm)) or np.any(np.isnan(returns_train_pm)):
#     returns_train_pm = np.nan_to_num(returns_train_pm, nan=0.0, posinf=0.0, neginf=0.0)

# Robust covariance estimation is recommended here (e.g., shrinkage)
# cov_matrix_train_pm = np.cov(returns_train_pm, rowvar=False) # [cite: 10]
# Make sure cov_matrix is not singular or ill-conditioned. Add regularization if needed.
# cov_matrix_train_pm = cov_matrix_train_pm + np.eye(cov_matrix_train_pm.shape[0]) * 1e-6 # Small regularization


# def apply_constraints(weights, sector_mapping, cov_matrix, max_exposure=0.4, min_sectors=2, max_risk=0.01): # [cite: 10]
    # Original constraints logic [cite: 10]
    # def asset_exposure_constraint(weights, max_exposure): return all(abs(w) <= max_exposure for w in weights) # [cite: 10]
    # def sector_diversification_constraint(weights, sector_mapping, min_sectors): # [cite: 10]
    #     # Ensure weights are not all zero before checking nonzero
    #     if np.all(weights == 0): return len(np.unique(sector_mapping)) >= min_sectors # Or specific handling
    #     active_sectors = sector_mapping[np.nonzero(weights)[0]] # Get sectors for assets with non-zero weights
    #     return len(np.unique(active_sectors)) >= min_sectors
    # def risk_tolerance_constraint(weights, cov_matrix, max_risk): # [cite: 10]
    #     portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights)) # [cite: 10]
    #     return portfolio_variance <= max_risk # [cite: 10]
    
#     if not (sector_mapping.shape[0] == cov_matrix.shape[0] == len(weights)):
#         # print(f"Shape mismatch: sectors {sector_mapping.shape}, cov {cov_matrix.shape}, weights {len(weights)}")
#         return False # Or raise error

#     return (asset_exposure_constraint(weights, max_exposure) and
#             sector_diversification_constraint(weights, sector_mapping, min_sectors) and
#             risk_tolerance_constraint(weights, cov_matrix, max_risk))


# def construct_portfolio_revised(signals_matrix, cov_matrix_for_constraints, sector_mapping_for_constraints): # [cite: 11]
#     weights = signals_matrix.copy() # [cite: 11]
    # Normalize signals to sum to 1 (assuming long-only, fully invested from positive signals)
#     row_sums = weights.sum(axis=1, keepdims=True) # [cite: 11]
#     row_sums[row_sums == 0] = 1e-6 # Avoid division by zero if all signals are zero for a day
#     weights = weights / row_sums # [cite: 11]

#     constraint_violations_count = 0 # [cite: 11]
#     for i in range(len(weights)): # [cite: 11]
#         if i == 0: # For the first day, we must use the current weights or a default if they fail
#             if not apply_constraints(weights[i, :], sector_mapping_for_constraints, cov_matrix_for_constraints):
                # Fallback for first day: e.g., 1/N or skip trading
#                 weights[i, :] = np.ones(weights.shape[1]) / weights.shape[1] 
#                 constraint_violations_count +=1
#         elif not apply_constraints(weights[i, :], sector_mapping_for_constraints, cov_matrix_for_constraints): # [cite: 11]
#             weights[i, :] = weights[i-1, :] # Carry forward previous day's weights [cite: 11]
#             constraint_violations_count += 1 # [cite: 11]
    # print(f"Number of days constraints caused reversion to previous weights: {constraint_violations_count}") # [cite: 11]
#     return weights

# sector_map = np.array([1,2,3,4]) # Placeholder [cite: 10]
# final_portfolio_weights_refined = construct_portfolio_revised(all_refined_signals_test, cov_matrix_train_pm, sector_map) # [cite: 11]
# final_portfolio_weights_primary = construct_portfolio_revised(all_primary_signals_test, cov_matrix_train_pm, sector_map) # [cite: 11]

# folder_path_sharpe = "/home/yichuan/ywc/meta-labeling/cryptocurrency" # [cite: 12]
# m_sharpe = n_train_overall +1 # Data for test period [cite: 12]
# n_sharpe = min_rows # [cite: 12]

# def calculate_sharpe_ratio_revised(folder_path, portfolio_weights_ts, m, n, risk_free_rate=0.0): # [cite: 11]
    # Assumes portfolio_weights_ts is [n_test_days, n_assets]
    # The rest of the function logic from In [16] can be used. [cite: 11, 12]
    # Ensure stock_returns calculation aligns with the period of portfolio_weights_ts.
    # The original `stock_prices` load from m:n, then diff. Weights should align with these return dates.
    # The number of rows in `portfolio_weights_ts` should be `len(stock_prices[:-1])`
    
    # files = sorted(os.listdir(folder_path)) # [cite: 11]
    # stock_data_close = []
    # for file in files: # [cite: 11]
    #     if file.endswith('.csv'): # Added safety
    #         df_sharpe = pd.read_csv(os.path.join(folder_path, file)).iloc[m:n] # [cite: 11]
    #         stock_data_close.append(df_sharpe['Close'].values) # [cite: 11]
    # stock_prices_eval = np.column_stack(stock_data_close) # [cite: 11]
    
    # stock_returns_eval = np.diff(stock_prices_eval, axis=0) / stock_prices_eval[:-1, :] # [cite: 12]
    # stock_returns_eval = np.nan_to_num(stock_returns_eval, nan=0.0) # Handle potential NaNs from division

    # Ensure weights match the number of return periods
#     if portfolio_weights_ts.shape[0] != stock_returns_eval.shape[0]:
#         print(f"Warning: Sharpe calculation weight rows ({portfolio_weights_ts.shape[0]}) != return rows ({stock_returns_eval.shape[0]}). Trimming weights.")
#         # This can happen if weights are for T days, returns for T-1 days. Align.
#         # Typically, weights at start of day t (W_t) apply to returns from t to t+1 (R_t+1)
#         # So, use weights[:-1] if it has one extra day.
#         min_len = min(portfolio_weights_ts.shape[0], stock_returns_eval.shape[0])
#         current_weights = portfolio_weights_ts[:min_len,:]
#         current_returns = stock_returns_eval[:min_len,:]
#     else:
#         current_weights = portfolio_weights_ts
#         current_returns = stock_returns_eval

#     portfolio_daily_returns = np.sum(current_returns * current_weights, axis=1) # [cite: 12]
    
#     if len(portfolio_daily_returns) == 0: return 0.0 # Avoid error if no returns
    
#     mean_return_daily = np.mean(portfolio_daily_returns) # [cite: 12]
#     volatility_daily = np.std(portfolio_daily_returns) # [cite: 12]

#     if volatility_daily == 0: return 0.0 # Avoid division by zero

    # Annualize (assuming daily data, ~252 trading days, or 365 for crypto)
#     annualization_factor = 365 # [cite: 12]
#     mean_return_annual = mean_return_daily * annualization_factor
#     volatility_annual = volatility_daily * np.sqrt(annualization_factor) # [cite: 12]
    
#     if volatility_annual == 0: return 0.0
    
#     sharpe = (mean_return_annual - risk_free_rate) / volatility_annual # [cite: 12]
#     return sharpe

# sharpe_ratio_refined = calculate_sharpe_ratio_revised(folder_path_sharpe, final_portfolio_weights_refined, m_sharpe, n_sharpe) # [cite: 12]
# print(f"Sharpe Ratio of the REFINED weights: {sharpe_ratio_refined:.4f}") # [cite: 12]

# sharpe_ratio_primary_revised = calculate_sharpe_ratio_revised(folder_path_sharpe, final_portfolio_weights_primary, m_sharpe, n_sharpe) # [cite: 12]
# print(f"Sharpe Ratio of the PRIMARY portfolio: {sharpe_ratio_primary_revised:.4f}") # [cite: 12]

# OneN_weights = np.ones_like(final_portfolio_weights_primary) * (1/num_assets) # [cite: 12]
# sharpe_ratio_1_N = calculate_sharpe_ratio_revised(folder_path_sharpe, OneN_weights, m_sharpe, n_sharpe) # [cite: 12]
# print(f"Sharpe Ratio of the 1/N portfolio: {sharpe_ratio_1_N:.4f}") # [cite: 12]


# TODO:
# 1. Implement robust hyperparameter tuning for all models and parameters (e.g., using time-series cross-validation like KFold with shuffle=False, or specialized time-series splits).
# 2. Implement walk-forward validation for a more realistic backtest.
# 3. Add more evaluation metrics (Sortino, Max Drawdown, Calmar, Turnover).
# 4. Conduct statistical significance testing for Sharpe Ratios (e.g., Ledoit-Wolf test for Sharpe differences or bootstrapping).
# 5. Further refine feature engineering for both primary and secondary models.
# 6. For `y_so` in continuous signal generation, ensure a proper non-lookahead model is built.
# 7. Ensure all data scaling (StandardScaler) is fit only on training data and transformed on validation/test data.
# 8. The NN architecture, optimizers, and loss functions should be carefully tuned.
# 9. Review all array indexing, especially around train/test splits and lags, to prevent off-by-one errors or data leakage.