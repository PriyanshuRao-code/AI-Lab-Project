from Team24_Code_Data_Preprocessing import *
from Team24_Code_Supervised_Learning import *

# Start coding from here

from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from itertools import product

df = pd.read_csv('24.csv')

df_train_orig, df_temp = train_test_split(df, test_size=0.4, random_state=42)
df_valid_orig, df_test_orig = train_test_split(df_temp, test_size=0.5, random_state=42)
df_train_orig.reset_index(drop=True, inplace=True)
df_valid_orig.reset_index(drop=True, inplace=True)
df_test_orig.reset_index(drop=True, inplace=True)

outlier_methods = [None, "iqr", "zscore"]
encoding_methods = ["label", "feature-extraction"]
scaling_methods = ["minmax", "standard"]
correlation_options = [True, False]
pca_options = [True, False]
select_high_correlat = [True, False]


data_preparation_methods = []

for outlier, encode, scale, correlation, pca, select_high_corr in product(outlier_methods, encoding_methods, scaling_methods, correlation_options, pca_options, select_high_correlat):

    # Skip the unwanted combination
    if pca and encode == "feature-extraction":
      continue

    if pca and select_high_corr:
      continue

    data_preparation_methods.append({
        "outlier": outlier,
        "encode": encode,
        "scale": scale,
        "correlation": correlation,
        "pca": pca,
        "select_corr" : select_high_corr
    })

models = {
    # "Perceptron": train_model_perceptron,
    # "Logistic Regression": train_model_logistic,
    # "KNN": train_model_knn,
    # "Gausian": train_model_gaussian,
    # "SVC": train_model_svc,
    "Random_Forest": train_model_random_forest,
    # "SVC_poly" : train_model_svc_poly,
    # "SVC_rbf" : train_model_svc_rbf,
    # "FCNN": train_model_fcnn
}

def data_preparation(df_train, df_valid, df_test, method):

    if method['outlier'] == "iqr":
        df_train, df_valid, df_test = remove_outliers_iqr(df_train, df_valid, df_test)
    elif method['outlier'] == "zscore":
        df_train, df_valid, df_test = remove_outliers_zscore(df_train, df_valid, df_test)


    if method['scale'] == "minmax":
        df_train, df_valid, df_test = normalize_minmax(df_train, df_valid, df_test)
    elif method['scale'] == "standard":
        df_train, df_valid, df_test = standardize_data(df_train, df_valid, df_test)


    if method['correlation']:
        df_train, df_valid, df_test = remove_highly_correlated(df_train, df_valid, df_test)


    if method['encode'] == "label":
        df_train, df_valid, df_test = label_encode_categorical(df_train, df_valid, df_test)
    elif method['encode'] == "feature-extraction":
        df_train, df_valid, df_test = feature_extraction(df_train, df_valid, df_test)

    if method['pca']:
        df_train, df_valid, df_test = perform_pca(df_train, df_valid, df_test)

    if method['select_corr']:
        df_train, df_valid, df_test = select_high_corr_features(df_train, df_valid, df_test)

    print("One preprocessing")

    return df_train, df_valid, df_test

def evaluate_model(y_true, y_pred, y_score=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0)
    }
    if y_score is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_score)
        except:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    return metrics

results = []

for method in data_preparation_methods:
    df_train, df_valid, df_test = data_preparation(df_train_orig.copy(), df_valid_orig.copy(), df_test_orig.copy(), method)

    for model_name, model_func in models.items():
        model, y_valid, y_test, y_valid_pred, y_test_pred = model_func(df_train, df_valid, df_test)

        # Try to get probabilities or scores for ROC-AUC
        try:
            y_valid_score = model.predict_proba(df_valid.drop("Hazardous", axis=1).values)[:, 1]
            y_test_score = model.predict_proba(df_test.drop("Hazardous", axis=1).values)[:, 1]
        except:
            try:
                y_valid_score = model.decision_function(df_valid.drop("Hazardous", axis=1).values)
                y_test_score = model.decision_function(df_test.drop("Hazardous", axis=1).values)
            except:
                y_valid_score = None
                y_test_score = None


        metrics_valid = evaluate_model(y_valid, y_valid_pred, y_valid_score)
        metrics_test = evaluate_model(y_test, y_test_pred, y_test_score)

        results.append({
            "data_preprocessing_methods": method,
            "model": model_name,
            "valid": metrics_valid,
            "test": metrics_test
        })

# Converting into dataframe
df_results = pd.json_normalize(results, sep='_')

print(df_results)

# Non useful results (Just for our testing)

# Bar Plot


# plt.figure(figsize=(12,6))
# sns.barplot(data=df_results, x='model', y='test_f1_score', hue='data_preprocessing_methods_outlier')
# plt.xticks(rotation=45)
# plt.title("Test F1 Score Comparison")
# plt.show()


# plt.figure(figsize=(12,6))
# sns.barplot(data=df_results, x='model', y='test_f1_score', hue='data_preprocessing_methods_encode')
# plt.xticks(rotation=45)
# plt.title("Test F1 Score Comparison")
# plt.show()


# plt.figure(figsize=(12,6))
# sns.barplot(data=df_results, x='model', y='test_f1_score', hue='data_preprocessing_methods_scale')
# plt.xticks(rotation=45)
# plt.title("Test F1 Score Comparison")
# plt.show()


# plt.figure(figsize=(12,6))
# sns.barplot(data=df_results, x='model', y='test_f1_score', hue='data_preprocessing_methods_correlation')
# plt.xticks(rotation=45)
# plt.title("Test F1 Score Comparison")
# plt.show()

# # Bar Plot
# plt.figure(figsize=(12,6))
# sns.barplot(data=df_results, x='model', y='test_recall', hue='data_preprocessing_methods_outlier')
# plt.xticks(rotation=45)
# plt.title("Test Recall Comparison")
# plt.show()


# plt.figure(figsize=(12,6))
# sns.barplot(data=df_results, x='model', y='test_recall', hue='data_preprocessing_methods_encode')
# plt.xticks(rotation=45)
# plt.title("Test Recall Comparison")
# plt.show()


# plt.figure(figsize=(12,6))
# sns.barplot(data=df_results, x='model', y='test_recall', hue='data_preprocessing_methods_scale')
# plt.xticks(rotation=45)
# plt.title("Test Recall Comparison")
# plt.show()



# plt.figure(figsize=(12,6))
# sns.barplot(data=df_results, x='model', y='test_recall', hue='data_preprocessing_methods_correlation')
# plt.xticks(rotation=45)
# plt.title("Test Recall Comparison")
# plt.show()

# # Heatmap

# pivot = df_results.pivot_table(index='model', columns='data_preprocessing_methods_outlier', values='valid_f1_score', aggfunc='mean')
# plt.figure(figsize=(14,7))
# sns.heatmap(pivot, annot=True, cmap='coolwarm')
# plt.title("Validation F1-Score Heatmap (data preprocessing outlier Wise)")
# plt.show()


# pivot = df_results.pivot_table(index='model', columns='data_preprocessing_methods_encode', values='valid_f1_score', aggfunc='mean')
# plt.figure(figsize=(14,7))
# sns.heatmap(pivot, annot=True, cmap='coolwarm')
# plt.title("Validation F1-Score Heatmap (data preprocessing encode Wise)")
# plt.show()

# pivot = df_results.pivot_table(index='model', columns='data_preprocessing_methods_scale', values='valid_f1_score', aggfunc='mean')
# plt.figure(figsize=(14,7))
# sns.heatmap(pivot, annot=True, cmap='coolwarm')
# plt.title("Validation F1-Score Heatmap (data preprocessing scale Wise)")
# plt.show()

# pivot = df_results.pivot_table(index='model', columns='data_preprocessing_methods_correlation', values='valid_f1_score', aggfunc='mean')
# plt.figure(figsize=(14,7))
# sns.heatmap(pivot, annot=True, cmap='coolwarm')
# plt.title("Validation F1-Score Heatmap (data preprocessing outlier Wise)")
# plt.show()

# # Heatmap

# pivot = df_results.pivot_table(index='model', columns='data_preprocessing_methods_outlier', values='test_f1_score', aggfunc='mean')
# plt.figure(figsize=(14,7))
# sns.heatmap(pivot, annot=True, cmap='coolwarm')
# plt.title("Test F1-Score Heatmap (data preprocessing outlier Wise)")
# plt.show()


# pivot = df_results.pivot_table(index='model', columns='data_preprocessing_methods_encode', values='test_f1_score', aggfunc='mean')
# plt.figure(figsize=(14,7))
# sns.heatmap(pivot, annot=True, cmap='coolwarm')
# plt.title("Test F1-Score Heatmap (data preprocessing encode Wise)")
# plt.show()

# pivot = df_results.pivot_table(index='model', columns='data_preprocessing_methods_scale', values='test_f1_score', aggfunc='mean')
# plt.figure(figsize=(14,7))
# sns.heatmap(pivot, annot=True, cmap='coolwarm')
# plt.title("Test F1-Score Heatmap (data preprocessing scale Wise)")
# plt.show()

# pivot = df_results.pivot_table(index='model', columns='data_preprocessing_methods_correlation', values='test_f1_score', aggfunc='mean')
# plt.figure(figsize=(14,7))
# sns.heatmap(pivot, annot=True, cmap='coolwarm')
# plt.title("Test F1-Score Heatmap (data preprocessing outlier Wise)")
# plt.show()

# # Scatter Plot


# plt.figure(figsize=(10,6))
# sns.scatterplot(data=df_results, x='valid_accuracy', y='valid_f1_score', hue='model', style='data_preprocessing_methods_outlier')
# plt.title("Accuracy vs F1-Score for different Pipelines & Models")
# plt.show()

