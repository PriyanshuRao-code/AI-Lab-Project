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
| `select_high_corr_features` | `df_train, df_valid, df_test, target_col="Hazardous", top_n=3` | `df_train[selected_features], df_valid[selected_features], df_test[selected_features]` |
| `perform_pca` | `df_train, df_valid, df_test, n_components=3` | `df_train_pca, df_valid_pca, df_test_pca` |
| `numeric_conversion` | `df_orig, one_hot_encode_month=False` | `df` |
| `normalize_date_features` | `df, month_cols, year_cols` | `df` |
| `process_splits` | `df_train, df_valid, df_test, one_hot_encode_month=False` | `df_train, df_valid, df_test` |
| `data_preprocessing` | `df` | `df_final_train, df_final_valid, df_final_test` |

---

## 📄 team_24_supervised_learning.ipynb
Below are the functions in `team_24_supervised_learning.ipynb`, along with their return values:

| Function Name | Arguments | Returns |
|---------------|-----------|---------|
| `x_y_separation` | `df_train, df_valid, df_test, target_column="Hazardous"` | `X_train, y_train, X_valid, y_valid, X_test, y_test` |
| `train_model_perceptron` | `df_train, df_valid, df_test, target_column = "Hazardous", num_epochs=2000, lr=0.0001` | `perceptron_model, y_valid_numpy, y_test_numpy, y_valid_pred_nn, y_test_pred_nn` |
| `train_model_gaussian` | `df_train, df_valid, df_test, target_column="Hazardous"` | `naiveBayes, y_valid_nb, y_test_nb, y_valid_pred_nb, y_test_pred_nb` |
| `mahalanobis_distance` | `x, mean, inv_cov_matrix` | `mahal_dist` |
| `knn_mahalanobis` | `X_train, y_train, X_test, K` | `np.array(y_pred)` |
| `train_model_knn` | `df_train, df_valid, df_test, target_column="Hazardous", k_values = [1, 3, 5, 7, 11]` | `knn_final, y_valid_knn, y_test_knn, y_valid_pred_knn, y_test_pred_knn` |
| `evaluate_model` | `y_true, y_pred, model_name="Model"` | `accuracy, precision, recall, f1` |
| `train_model_logistic` | `df_train, df_valid, df_test, target_column="Hazardous"` | `logistic_reg, y_valid, y_test, y_val_pred, y_test_pred` |
| `train_model_svc` | `df_train, df_valid, df_test, target_column="Hazardous", method='linear'` | `model, y_valid, y_test, y_val_pred, y_test_pred` |

---



## Usage  
Clone the repo and run the script:  
```bash
git clone https://github.com/PriyanshuRao-code/AI-Lab-Project.git
