# AI-Lab-Project

# Identifying Hazardous Near-Earth Asteroids  

## Overview  
This project classifies near-Earth asteroids as **hazardous or non-hazardous** using **machine learning**. The dataset, sourced from **NASA’s NeoWs**, includes asteroid details like size, velocity, and orbital data.  

## Dataset  
- **Key Features**: Orbital parameters, size, velocity, and approach distance.  
- **Target**: Binary classification (Hazardous/Non-Hazardous).  

## Methodology  
1. **Preprocessing** – Cleaning and normalizing data.  
2. **Feature Selection** – Identifying important asteroid properties.  
3. **Model Training** – Using classification algorithms.  
4. **Evaluation** – Accuracy, precision, and F1-score.  

# 📘 Functions & Return Values

## 📄 team_24_data_preprocessing.ipynb
Below are the functions in `team_24_data_preprocessing.ipynb`, along with their return values:

| Function Name | Arguments | Returns |
|---------------|-----------|---------|
| `remove_outliers_iqr` | `df_train, df_valid, df_test` | `df_train_cleaned, df_valid, df_test` |
| `remove_outliers_zscore` | `df_train, df_valid, df_test, z_score_threshold = 3` | `df_train_cleaned, df_valid, df_test` |
| `remove_highly_correlated` | `df_train, df_valid, df_test, high_corr_threshold = 0.99` | `df_train_reduced, df_valid_reduced, df_test_reduced` |
| `normalize_minmax` | `df_train_orig, df_valid_orig, df_test_orig` | `df_train, df_valid, df_test` |
| `standardize_data` | `df_train_orig, df_valid_orig, df_test_orig` | `df_train, df_valid, df_test` |
| `label_encode_categorical` | `df_train, df_valid, df_test` | `df_train, df_valid, df_test` |
| `select_high_corr_features` | `df_train, df_valid, df_test, target_col="Hazardous", top_n=3` | `df_train[selected_features + [target_col]], df_valid[selected_features + [target_col]], df_test[selected_features + [target_col]]` |
| `perform_pca` | `df_train, df_valid, df_test, target_col="Hazardous", n_components=11` | `df_train_pca, df_valid_pca, df_test_pca` |
| `numeric_conversion` | `df_orig` | `df` |
| `normalize_date_features` | `df, month_cols, year_cols` | `df` |
| `feature_extraction` | `df_train, df_valid, df_test` | `df_train, df_valid, df_test` |
| `data_preprocessing` | `df` | `df_final_train, df_final_valid, df_final_test` |
| `data_gausian` | `df` | `df_train, df_valid, df_test` |
| `data_knn1` | `df` | `df_train, df_valid, df_test` |
| `data_knn2` | `df` | `df_train, df_valid, df_test` |
| `data_logistic` | `df` | `df_train, df_valid, df_test` |
| `data_perceptron` | `df` | `df_train, df_valid, df_test` |
| `data_random_forest` | `df` | `df_train, df_valid, df_test` |
| `data_random_forest1` | `df` | `df_train, df_valid, df_test` |
| `data_svc1` | `df` | `df_train, df_valid, df_test` |
| `data_svc2` | `df` | `df_train, df_valid, df_test` |
| `data_svc_poly` | `df` | `df_train, df_valid, df_test` |
| `data_svc_rbf1` | `df` | `df_train, df_valid, df_test` |
| `data_svc_rbf2` | `df` | `df_train, df_valid, df_test` |

---

## 📄 team_24_supervised_learning.ipynb
Below are the functions in `team_24_supervised_learning.ipynb`, along with their return values:

| Function Name | Arguments | Returns |
|---------------|-----------|---------|
| `set_seeds` | `seed=42` | `[Nothing returned or dynamic logic]` |
| `x_y_separation` | `df_train, df_valid, df_test, target_column="Hazardous"` | `X_train, y_train, X_valid, y_valid, X_test, y_test` |
| `__init__` | `self, input_dim` | `[Nothing returned or dynamic logic]` |
| `forward` | `self, x` | `x` |
| `__init__` | `self, input_size, hidden_layers` | `[Nothing returned or dynamic logic]` |
| `forward` | `self, x` | `self.model(x)` |
| `train_model_perceptron` | `df_train, df_valid, df_test, target_column = "Hazardous", num_epochs=2000, lr=0.0001, seed = 42` | `perceptron_model, y_valid_numpy, y_test_numpy, y_valid_pred_nn, y_test_pred_nn` |
| `apply_smote` | `X, y, seed=42` | `X_res, y_res` |
| `oversample_data` | `X, y, seed=42` | `X_res, y_res` |
| `undersample_data` | `X, y, seed=42` | `X_res, y_res` |
| `train_model_fcnn` | `df_train, df_valid, df_test, target_column="Hazardous", num_epochs=2000, lr=0.0001, hidden_layers_list = [[10],[17],[31],[64],[100]],seed=42` | `model, y_valid_numpy, y_test_numpy, y_valid_pred, y_test_pred` |
| `train_model_gaussian` | `df_train, df_valid, df_test, target_column="Hazardous"` | `naiveBayes, y_valid_nb, y_test_nb, y_valid_pred_nb, y_test_pred_nb` |
| `mahalanobis_distance` | `x, mean, inv_cov_matrix` | `mahal_dist` |
| `knn_mahalanobis` | `X_train, y_train, X_test, K` | `np.array(y_pred)` |
| `train_model_knn` | `df_train, df_valid, df_test, target_column="Hazardous", k_values = [1, 3]` | `knn_final, y_valid_knn, y_test_knn, y_valid_pred_knn, y_test_pred_knn` |
| `train_model_logistic` | `df_train, df_valid, df_test, target_column="Hazardous", solver='sag',max_iter=2000` | `logistic_reg, y_valid, y_test, y_val_pred, y_test_pred` |
| `train_model_svc` | `df_train, df_valid, df_test, target_column="Hazardous"` | `model, y_valid, y_test, y_val_pred, y_test_pred` |
| `train_model_svc_rbf` | `df_train, df_valid, df_test, target_column="Hazardous"` | `model, y_valid, y_test, y_val_pred, y_test_pred` |
| `train_model_svc_poly` | `df_train, df_valid, df_test, target_column="Hazardous"` | `model, y_valid, y_test, y_val_pred, y_test_pred` |
| `train_model_random_forest` | `df_train, df_valid, df_test, target_column="Hazardous", n_estimators=100, random_state=42` | `random_forest, y_valid_rf, y_test_rf, y_valid_pred_rf, y_test_pred_rf` |
| `evaluate_model` | `y_true, y_pred, model_name="Model"` | `accuracy, precision, recall, f1` |

---
## 📄 team_24.ipynb
Below are the functions in `team_24.ipynb`, along with their return values:

| Function Name | Arguments | Returns |
|---------------|-----------|---------|
| `data_preparation` | `df_train, df_valid, df_test, method` | `df_train, df_valid, df_test` |
| `evaluate_model` | `y_true, y_pred, y_score=None` | `metrics` |

---


## Usage  
Clone the repo and run the script:  
```bash
git clone https://github.com/PriyanshuRao-code/AI-Lab-Project.git
