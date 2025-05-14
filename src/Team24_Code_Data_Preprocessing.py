import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from datetime import datetime
from sklearn.model_selection import train_test_split

original_dataframe = pd.read_csv('24.csv')

def remove_outliers_iqr(df_train, df_valid, df_test):

  df_train_numeric = df_train.select_dtypes(include=['number']).select_dtypes(exclude=['bool'])
  df_train_non_numeric = df_train.select_dtypes(exclude=['number'])

  Q1 = df_train_numeric.quantile(0.25)
  Q3 = df_train_numeric.quantile(0.75)
  IQR = Q3 - Q1

  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR

  # Removing outliers
  df_iqr = df_train_numeric[~((df_train_numeric < lower_bound) | (df_train_numeric > upper_bound)).any(axis=1)]
  df_train_cleaned = pd.concat([df_iqr, df_train_non_numeric.loc[df_iqr.index]], axis=1)

  return df_train_cleaned, df_valid, df_test

def remove_outliers_zscore(df_train, df_valid, df_test, z_score_threshold = 3):

  df_train_numeric = df_train.select_dtypes(include=['number']).select_dtypes(exclude=['bool'])
  df_train_non_numeric = df_train.select_dtypes(exclude=['number'])

  z_scores = df_train_numeric.apply(zscore)
  df_z = df_train_numeric[(z_scores.abs() < z_score_threshold).all(axis=1)]  # Remove rows with Z-score >  z_score_threshold in any column
  df_train_cleaned = pd.concat([df_z, df_train_non_numeric.loc[df_z.index]], axis=1)

  return df_train_cleaned, df_valid, df_test

def remove_highly_correlated(df_train, df_valid, df_test, high_corr_threshold = 0.99):

  df_train_numeric = df_train.select_dtypes(include=['number']).select_dtypes(exclude=['bool'])

  high_corr_pairs = set()
  correlation_matrix = df_train_numeric.corr()

  for i in range(len(correlation_matrix.columns)):
    for j in range(i): # Lower triangular matrix
      if abs(correlation_matrix.iloc[i, j]) >= high_corr_threshold:
        col1 = correlation_matrix.columns[i]
        col2 = correlation_matrix.columns[j]
        high_corr_pairs.add((col1, col2))

  columns_to_drop = {col2 for col1, col2 in high_corr_pairs}

  df_train_reduced = df_train.drop(columns=columns_to_drop)
  df_valid_reduced = df_valid.drop(columns=columns_to_drop)
  df_test_reduced = df_test.drop(columns=columns_to_drop)

  return df_train_reduced, df_valid_reduced, df_test_reduced

def normalize_minmax(df_train_orig, df_valid_orig, df_test_orig):
  df_train = df_train_orig.copy()
  df_valid = df_valid_orig.copy()
  df_test = df_test_orig.copy()
  scaler = MinMaxScaler()

  df_train_numeric = df_train.select_dtypes(include=['number']).select_dtypes(exclude=['bool'])
  df_test_numeric = df_test.select_dtypes(include=['number']).select_dtypes(exclude=['bool'])
  df_valid_numeric = df_valid.select_dtypes(include=['number']).select_dtypes(exclude=['bool'])

  # Normalization
  df_train[df_train_numeric.columns] = scaler.fit_transform(df_train_numeric)
  df_test[df_test_numeric.columns] = scaler.transform(df_test_numeric)
  df_valid[df_valid_numeric.columns] = scaler.transform(df_valid_numeric)

  return df_train, df_valid, df_test

def standardize_data(df_train_orig, df_valid_orig, df_test_orig):
  df_train = df_train_orig.copy()
  df_valid = df_valid_orig.copy()
  df_test = df_test_orig.copy()
  scaler = StandardScaler()

  df_train_numeric = df_train.select_dtypes(include=['number']).select_dtypes(exclude=['bool'])
  df_test_numeric = df_test.select_dtypes(include=['number']).select_dtypes(exclude=['bool'])
  df_valid_numeric = df_valid.select_dtypes(include=['number']).select_dtypes(exclude=['bool'])

  # Standardization
  df_train[df_train_numeric.columns] = scaler.fit_transform(df_train_numeric)
  df_test[df_test_numeric.columns] = scaler.transform(df_test_numeric)
  df_valid[df_valid_numeric.columns] = scaler.transform(df_valid_numeric)

  return df_train, df_valid, df_test

def label_encode_categorical(df_train, df_valid, df_test):

  categorical_cols = df_train.select_dtypes(include=['object']).columns
  label_encoders = {}

  for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    df_train[col] = label_encoders[col].fit_transform(df_train[col])

    df_valid[col] = df_valid[col].map(lambda x: label_encoders[col].transform([x])[0] if x in label_encoders[col].classes_ else -1)
    df_test[col] = df_test[col].map(lambda x: label_encoders[col].transform([x])[0] if x in label_encoders[col].classes_ else -1)

  df_valid = df_valid[df_train.columns]
  df_test = df_test[df_train.columns]

  return df_train, df_valid, df_test

def select_high_corr_features(df_train, df_valid, df_test, target_col="Hazardous", top_n=3):
    df_train_temp = df_train.copy()
    df_train_temp[target_col] = df_train_temp[target_col].astype(int)

    df_train_numeric = df_train_temp.select_dtypes(include=['number']).select_dtypes(exclude=['bool'])

    corr_values = df_train_numeric.corr()[target_col].abs().sort_values(ascending=False)

    selected_features = corr_values.drop(index=target_col).head(top_n).index.tolist()

    return df_train[selected_features + [target_col]], df_valid[selected_features + [target_col]], df_test[selected_features + [target_col]]

def perform_pca(df_train, df_valid, df_test, target_col="Hazardous", n_components=11):

  # Split features & target
  X_train = df_train.drop(columns=[target_col])
  X_valid = df_valid.drop(columns=[target_col])
  X_test  = df_test.drop(columns=[target_col])

  y_train = df_train[target_col].reset_index(drop=True)
  y_valid = df_valid[target_col].reset_index(drop=True)
  y_test  = df_test[target_col].reset_index(drop=True)

  # PCA
  pca = PCA(n_components=n_components)
  pca.fit(X_train)

  X_train_pca = pd.DataFrame(pca.transform(X_train), columns=[f'PC{i+1}' for i in range(n_components)])
  X_valid_pca = pd.DataFrame(pca.transform(X_valid), columns=[f'PC{i+1}' for i in range(n_components)])
  X_test_pca  = pd.DataFrame(pca.transform(X_test),  columns=[f'PC{i+1}' for i in range(n_components)])

  # Add target back
  df_train_pca = pd.concat([X_train_pca, y_train], axis=1)
  df_valid_pca = pd.concat([X_valid_pca, y_valid], axis=1)
  df_test_pca  = pd.concat([X_test_pca,  y_test],  axis=1)

  return df_train_pca, df_valid_pca, df_test_pca

def numeric_conversion(df_orig):
  df = df_orig.copy()

  # Dropping 'Equinox' and 'Orbiting Body'
  df.drop(columns=['Equinox', 'Orbiting Body'], errors='ignore', inplace=True)

  # Converting COLUMNS to datetime
  df['Close Approach Date'] = pd.to_datetime(df['Close Approach Date'])
  df['Close Approach Year'] = df['Close Approach Date'].dt.year
  df['Close Approach Month'] = df['Close Approach Date'].dt.month

  df['Orbit Determination Date'] = pd.to_datetime(df['Orbit Determination Date'])
  df['Orbit Determination Year'] = df['Orbit Determination Date'].dt.year
  df['Orbit Determination Month'] = df['Orbit Determination Date'].dt.month

  # Encoding 'Hazardous' column
  df['Hazardous'] = df['Hazardous'].astype(int)

  df = df.drop(columns=["Close Approach Date", "Orbit Determination Date"])
  return df

def normalize_date_features(df, month_cols, year_cols):
  for col in month_cols:
    if col in df.columns:
      df[col] = (df[col] - 1) / (12 - 1)

  for col in year_cols:
    if col in df.columns:
      df[col] = (df[col] - 1900) / (2100 - 1900)

  return df

def feature_extraction(df_train, df_valid, df_test):
  df_train = numeric_conversion(df_train)
  df_valid = numeric_conversion(df_valid)
  df_test = numeric_conversion(df_test)

  month_cols = ['Close Approach Month', 'Orbit Determination Month']
  year_cols = ['Close Approach Year', 'Orbit Determination Year']

  # Normalize date features
  df_train = normalize_date_features(df_train, month_cols, year_cols)
  df_valid = normalize_date_features(df_valid, month_cols, year_cols)
  df_test = normalize_date_features(df_test, month_cols, year_cols)

  return df_train, df_valid, df_test


# The following functions are intended solely for report generation using various stages of the data pipeline


def data_preprocessing(df):
  df_train, df_temp = train_test_split(df, test_size=0.4, random_state=42)
  df_valid, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)
  df_train.reset_index(drop=True, inplace=True)
  df_valid.reset_index(drop=True, inplace=True)
  df_test.reset_index(drop=True, inplace=True)

  no_outliers_train, no_outliers_valid, no_outliers_test = remove_outliers_zscore(df_train, df_valid, df_test)
  no_highly_correlated_train, no_highly_correlated_valid, no_highly_correlated_test = remove_highly_correlated(no_outliers_train, no_outliers_valid, no_outliers_test)
  normal_train, normal_valid, normal_test = normalize_minmax(no_highly_correlated_train, no_highly_correlated_valid, no_highly_correlated_test)
  df_final_train, df_final_valid, df_final_test = feature_extraction(normal_train, normal_valid, normal_test)
  return df_final_train, df_final_valid, df_final_test

def data_gausian(df):
  df_train, df_temp = train_test_split(df, test_size=0.4, random_state=42)
  df_valid, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)
  df_train.reset_index(drop=True, inplace=True)
  df_valid.reset_index(drop=True, inplace=True)
  df_test.reset_index(drop=True, inplace=True)


  df_train, df_valid, df_test = remove_outliers_iqr(df_train, df_valid, df_test)
  df_train, df_valid, df_test = feature_extraction(df_train, df_valid, df_test)
  df_train, df_valid, df_test = standardize_data(df_train, df_valid, df_test)
  df_train, df_valid, df_test = remove_highly_correlated(df_train, df_valid, df_test)

  return df_train, df_valid, df_test

def data_knn1(df):
  df_train, df_temp = train_test_split(df, test_size=0.4, random_state=42)
  df_valid, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)
  df_train.reset_index(drop=True, inplace=True)
  df_valid.reset_index(drop=True, inplace=True)
  df_test.reset_index(drop=True, inplace=True)

  df_train, df_valid, df_test = remove_outliers_iqr(df_train, df_valid, df_test)
  df_train, df_valid, df_test = label_encode_categorical(df_train, df_valid, df_test)
  df_train, df_valid, df_test = standardize_data(df_train, df_valid, df_test)
  df_train, df_valid, df_test = remove_highly_correlated(df_train, df_valid, df_test)
  df_train, df_valid, df_test = perform_pca(df_train, df_valid, df_test)

  return df_train, df_valid, df_test

def data_knn2(df):
  # Shows trade off between accuracy and recall
  df_train, df_temp = train_test_split(df, test_size=0.4, random_state=42)
  df_valid, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)
  df_train.reset_index(drop=True, inplace=True)
  df_valid.reset_index(drop=True, inplace=True)
  df_test.reset_index(drop=True, inplace=True)

  df_train, df_valid, df_test = feature_extraction(df_train, df_valid, df_test)
  df_train, df_valid, df_test = standardize_data(df_train, df_valid, df_test)

  return df_train, df_valid, df_test

def data_logistic(df):
  df_train, df_temp = train_test_split(df, test_size=0.4, random_state=42)
  df_valid, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)
  df_train.reset_index(drop=True, inplace=True)
  df_valid.reset_index(drop=True, inplace=True)
  df_test.reset_index(drop=True, inplace=True)

  df_train, df_valid, df_test = remove_outliers_zscore(df_train, df_valid, df_test)
  df_train, df_valid, df_test = feature_extraction(df_train, df_valid, df_test)
  df_train, df_valid, df_test = standardize_data(df_train, df_valid, df_test)
  df_train, df_valid, df_test = standardize_data(df_train, df_valid, df_test)

  return df_train, df_valid, df_test

def data_perceptron(df):
  # Not needed acctually
  df_train, df_temp = train_test_split(df, test_size=0.4, random_state=42)
  df_valid, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)
  df_train.reset_index(drop=True, inplace=True)
  df_valid.reset_index(drop=True, inplace=True)
  df_test.reset_index(drop=True, inplace=True)

  df_train, df_valid, df_test = remove_outliers_zscore(df_train, df_valid, df_test)
  df_train, df_valid, df_test = label_encode_categorical(df_train, df_valid, df_test)
  df_train, df_valid, df_test = standardize_data(df_train, df_valid, df_test)

  return df_train, df_valid, df_test

def data_random_forest(df):
  df_train, df_temp = train_test_split(df, test_size=0.4, random_state=42)
  df_valid, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)
  df_train.reset_index(drop=True, inplace=True)
  df_valid.reset_index(drop=True, inplace=True)
  df_test.reset_index(drop=True, inplace=True)

  df_train, df_valid, df_test = remove_outliers_iqr(df_train, df_valid, df_test)
  df_train, df_valid, df_test = label_encode_categorical(df_train, df_valid, df_test)
  df_train, df_valid, df_test = normalize_minmax(df_train, df_valid, df_test)
  df_train, df_valid, df_test = remove_highly_correlated(df_train, df_valid, df_test)

  return df_train, df_valid, df_test

def data_random_forest1(df):
  df_train, df_temp = train_test_split(df, test_size=0.4, random_state=42)
  df_valid, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)
  df_train.reset_index(drop=True, inplace=True)
  df_valid.reset_index(drop=True, inplace=True)
  df_test.reset_index(drop=True, inplace=True)

  df_train, df_valid, df_test = remove_outliers_iqr(df_train, df_valid, df_test)
  df_train, df_valid, df_test = label_encode_categorical(df_train, df_valid, df_test)
  df_train, df_valid, df_test = normalize_minmax(df_train, df_valid, df_test)
  df_train, df_valid, df_test = remove_highly_correlated(df_train, df_valid, df_test)
  df_train, df_valid, df_test = select_high_corr_features(df_train, df_valid, df_test)

  return df_train, df_valid, df_test

def data_svc1(df):
  df_train, df_temp = train_test_split(df, test_size=0.4, random_state=42)
  df_valid, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)
  df_train.reset_index(drop=True, inplace=True)
  df_valid.reset_index(drop=True, inplace=True)
  df_test.reset_index(drop=True, inplace=True)

  df_train, df_valid, df_test = label_encode_categorical(df_train, df_valid, df_test)
  df_train, df_valid, df_test = standardize_data(df_train, df_valid, df_test)
  return df_train, df_valid, df_test

def data_svc2(df):
  # Immune to outliers
  df_train, df_temp = train_test_split(df, test_size=0.4, random_state=42)
  df_valid, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)
  df_train.reset_index(drop=True, inplace=True)
  df_valid.reset_index(drop=True, inplace=True)
  df_test.reset_index(drop=True, inplace=True)

  df_train, df_valid, df_test = remove_outliers_zscore(df_train, df_valid, df_test)
  df_train, df_valid, df_test = label_encode_categorical(df_train, df_valid, df_test)
  df_train, df_valid, df_test = standardize_data(df_train, df_valid, df_test)
  return df_train, df_valid, df_test

def data_svc_poly(df):
  df_train, df_temp = train_test_split(df, test_size=0.4, random_state=42)
  df_valid, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)
  df_train.reset_index(drop=True, inplace=True)
  df_valid.reset_index(drop=True, inplace=True)
  df_test.reset_index(drop=True, inplace=True)

  df_train, df_valid, df_test = remove_outliers_iqr(df_train, df_valid, df_test)
  df_train, df_valid, df_test = feature_extraction(df_train, df_valid, df_test)
  df_train, df_valid, df_test = normalize_minmax(df_train, df_valid, df_test)

  return df_train, df_valid, df_test

def data_svc_rbf1(df):
  df_train, df_temp = train_test_split(df, test_size=0.4, random_state=42)
  df_valid, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)
  df_train.reset_index(drop=True, inplace=True)
  df_valid.reset_index(drop=True, inplace=True)
  df_test.reset_index(drop=True, inplace=True)

  df_train, df_valid, df_test = feature_extraction(df_train, df_valid, df_test)
  df_train, df_valid, df_test = standardize_data(df_train, df_valid, df_test)
  df_train, df_valid, df_test = remove_highly_correlated(df_train, df_valid, df_test)

  return df_train, df_valid, df_test

def data_svc_rbf2(df):
  # Increasing recall might significantly drop accuracy.
  df_train, df_temp = train_test_split(df, test_size=0.4, random_state=42)
  df_valid, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)
  df_train.reset_index(drop=True, inplace=True)
  df_valid.reset_index(drop=True, inplace=True)
  df_test.reset_index(drop=True, inplace=True)

  df_train, df_valid, df_test = remove_outliers_iqr(df_train, df_valid, df_test)
  df_train, df_valid, df_test = feature_extraction(df_train, df_valid, df_test)
  df_train, df_valid, df_test = normalize_minmax(df_train, df_valid, df_test)

  return df_train, df_valid, df_test

# # TESTING PURPOSE ONLY

print("-----------------Team24_Code_Data_Preprocessing.py is running perfectly fine.-------------------------")




# df_testing_purpose = original_dataframe.copy()

# df_train, df_temp = train_test_split(df_testing_purpose, test_size=0.4, random_state=42)
# df_valid, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)
# df_train.reset_index(drop=True, inplace=True)
# df_valid.reset_index(drop=True, inplace=True)
# df_test.reset_index(drop=True, inplace=True)

# no_outliers_train, no_outliers_valid, no_outliers_test = remove_outliers_zscore(df_train, df_valid, df_test)
# no_highly_correlated_train, no_highly_correlated_valid, no_highly_correlated_test = remove_highly_correlated(no_outliers_train, no_outliers_valid, no_outliers_test)
# normal_train, normal_valid, normal_test = normalize_minmax(no_highly_correlated_train, no_highly_correlated_valid, no_highly_correlated_test)
# df_final_train, df_final_valid, df_final_test = feature_extraction(normal_train, normal_valid, normal_test)


# print(f"Original data shape: {df_testing_purpose.shape}")
# print("\n")

# print(f"Train data shape: {df_train.shape}")
# print(f"Validation data shape: {df_valid.shape}")
# print(f"Test data shape: {df_test.shape}")
# print("\n")


# print(f"no_outliers_train shape: {no_outliers_train.shape}")
# print(f"no_outliers_valid shape: {no_outliers_valid.shape}")
# print(f"no_outliers_test shape: {no_outliers_test.shape}")
# print("\n")


# print(f"no_highly_correlated_train shape: {no_highly_correlated_train.shape}")
# print(f"no_highly_correlated_valid shape: {no_highly_correlated_valid.shape}")
# print(f"no_highly_correlated_test shape: {no_highly_correlated_test.shape}")
# print("\n")


# print(f"normal_train shape: {normal_train.shape}")
# print(f"normal_valid shape: {normal_valid.shape}")
# print(f"normal_test shape: {normal_test.shape}")
# print("\n")


# print(f"df_final_train shape: {df_final_train.shape}")
# print(f"df_final_valid shape: {df_final_valid.shape}")
# print(f"df_final_test shape: {df_final_test.shape}")
# print("\n")


# print(f" Hazardous count in no_outliers train : {no_outliers_train[no_outliers_train['Hazardous']==1].shape[0]}")
# print(f" Hazardous count in df_final_train : {df_final_train[df_final_train['Hazardous']==1].shape[0]}")