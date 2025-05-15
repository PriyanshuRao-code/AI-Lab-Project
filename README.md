# 🚀 NASA Asteroid Hazard Prediction

This project predicts whether an asteroid is hazardous or not using multiple machine learning models. The pipeline includes detailed data preprocessing, model training, and performance evaluation. Built using Jupyter Notebooks and deployed with Streamlit.

---
## 📂 Repository Structure
```plaintext
├── 📂 notebooks
│   ├── 📄 24.csv
│   ├── 📄 team_24.ipynb
│   ├── 📄 team_24_data_preprocessing.ipynb
│   └── 📄 team_24_supervised_learning.ipynb
|
└── 📂 src
    ├── 📄 24.csv
    ├── 🐍 Team24_Code_Data_Preprocessing.py
    ├── 🐍 Team24_Code_Main.py
    └── 🐍 Team24_Code_Supervised_Learning.py

├── 📂 branch_archives
│   ├── 📂 Analysis_in_main_branch
│   ├── 📂 Armaan
│   ├── 📂 Priyanshu-Rao
│   ├── 📂 Sai-Lohith
│   └── 📂 riya
│
├── 📄 24.csv
├── 📖 README.md
├── 📑 Team24.pdf
├── 📝 requirements.txt
```



## 📖 Report

* Final project write‑up available on Overleaf: [AI Lab Project Report](https://www.overleaf.com/read/chzbsnwbrbbb#86e70a)

---

## 💫 ——— Credits ——— 💫

Made with 🤍 by:

- 👨‍💻 Priyanshu Rao
- 👩‍💻 Armaan Fatima  
- 👩‍💻 Riya Chitnis  
- 👨‍💻 Chittala Venkata Sai Lohith

---

## 📂 Files Overview

### 🔧 `team_24_data_preprocessing.ipynb`

* Prepares the dataset for modeling.
* Handles null values, label encoding, normalization, PCA, correlation filtering, and outlier removal.
* Also includes preprocessing tailored for specific ML models.

### 🤖 `team_24_supervised_learning.ipynb`

* Trains a wide range of machine learning models (Perceptron, FCNN, Logistic Regression, SVC, Random Forest, Gaussian, KNN).
* Performs SMOTE, over/under-sampling.

### 🧠 `team_24.ipynb`

* Integrates preprocessing and model training into a single pipeline.
* Streamlined evaluation and visualization.
---

## 🎯 Usage

There are two ways to run this project:

### 🔹 Option 1: Run via Python Script

Clone the repo and set up the environment:
```bash
git clone https://github.com/PriyanshuRao-code/AI-Lab-Project.git
cd AI-Lab-Project
pip install -r requirements.txt
````

Now run the full pipeline using the code from the src/ folder or the unified script:

```bash
python Team24_Code_Main.py
```

> This script will execute the entire pipeline using the preprocessed data and currently enabled model(s).
> 🟢 By default, only "Random Forest" is active to reduce training time.

To enable additional models (Perceptron, Logistic Regression, etc.), uncomment the respective lines inside the models dictionary in Team24\_Code\_Main.py:

```python
models = {
    # "Perceptron": train_model_perceptron,
    # "Logistic Regression": train_model_logistic,
    # "KNN": train_model_knn,
    # "Gausian": train_model_gaussian,
    # "SVC": train_model_svc,
    "Random_Forest": train_model_random_forest,
    # "SVC_poly": train_model_svc_poly,
    # "SVC_rbf": train_model_svc_rbf,
    # "FCNN": train_model_fcnn
}
```

---

### 🔹 Option 2: Run via Jupyter Notebook

If you prefer a more interactive and visual interface, open the following notebook:

```bash
notebooks/team_24.ipynb
```

> This notebook is compatible with Google Colab and supports GPU/TPU acceleration for fast experimentation.

---

📌 **Note for Google Colab users**:
In our Colab environment, we directly set the working directory to:

```
/content/AI-Lab-Project/notebooks
```

This means:

* All imports like `from team_24_data_preprocessing import *` assume that you are inside the `notebooks/` folder.
* We do **not use or depend on files outside `notebooks/`**, unless explicitly mentioned.
* This helps avoid path errors and simplifies module imports via `import-ipynb`.

---

### 🔹 Option 3: Explore via Streamlit Dashboard

To explore the results interactively on a hosted web app:

🔗 Live App:
👉 [https://ai-dashboard-team-24.streamlit.app/](https://ai-dashboard-team-24.streamlit.app/)

🛠️ Source Code:
👉 [https://github.com/PriyanshuRao-code/Streamlit.git](https://github.com/PriyanshuRao-code/Streamlit.git)

> This dashboard visualizes model comparisons, preprocessing results, and evaluation metrics. Ideal for quick insights without running code locally.

---


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

## 🧩 Note on Preprocessing Functions
The following functions are included in both team\_24\_data\_preprocessing.ipynb and Team24\_Code\_Data\_Preprocessing.py to support individual report components and learning visualizations:

| Function Name | Arguments | Returns |
|---------------|-----------|---------|
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

These are meant to generate different stages of preprocessed data used for plots, evaluation matrices, and visual comparisons in the final report.

