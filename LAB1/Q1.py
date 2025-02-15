import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_dataset(file_path):
    """Loads the dataset from the given file path."""
    return pd.read_csv("C:/Users/Administrator/OneDrive - Amrita vishwa vidyapeetham/SEM6_CSE/NNDL_DRUG/LAB1/Custom_CNN_Features.csv")

def dataset_info(df):
    """Displays dataset info and first few rows."""
    print("\nDataset Information:")
    print(df.info())
    print("\nFirst Few Rows:")
    print(df.head())

def check_missing_values(df):
    """Checks for missing values in the dataset."""
    missing_values = df.isnull().sum().sum()
    print(f"\nTotal Missing Values: {missing_values}")
    return missing_values

def class_balance(df):
    """Analyzes class label distribution."""
    print("\nClass Label Distribution:")
    print(df["Class Label"].value_counts())

def correlation_heatmap(df):
    """Plots correlation matrix heatmap."""
    correlation_matrix = df.iloc[:, 2:].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, cmap="coolwarm", vmax=1, vmin=-1, center=0, square=True)
    plt.title("Feature Correlation Heatmap")
    plt.show()

def compute_matrix_rank(df):
    """Computes matrix rank (dimensionality)."""
    feature_matrix = df.iloc[:, 2:].values
    matrix_rank = np.linalg.matrix_rank(feature_matrix)
    print(f"\nMatrix Rank (Intrinsic Dimensionality): {matrix_rank}")
    return matrix_rank

def feature_range_analysis(df):
    """Analyzes feature value ranges and prints summary."""
    feature_stats = pd.DataFrame({
        "Min": df.iloc[:, 2:].min(),
        "Max": df.iloc[:, 2:].max(),
        "Mean": df.iloc[:, 2:].mean()
    })
    print("\nFeature Range Summary:")
    print(feature_stats.describe())
    return feature_stats

def normalize_data(df):
    """Applies Min-Max normalization to dataset."""
    df_normalized = df.copy()
    overall_min = df.iloc[:, 2:].min().min()
    overall_max = df.iloc[:, 2:].max().max()
    df_normalized.iloc[:, 2:] = (df.iloc[:, 2:] - overall_min) / (overall_max - overall_min)
    
    print("\nDataset Normalization Applied (Min-Max Scaling).")
    return df_normalized

def main():
    file_path = "/mnt/data/Custom_CNN_Features.csv"
    df = load_dataset(file_path)

    # Perform EDA
    dataset_info(df)
    check_missing_values(df)
    class_balance(df)
    correlation_heatmap(df)
    compute_matrix_rank(df)
    feature_range_analysis(df)

    # Normalize and save dataset
    df_normalized = normalize_data(df)
    normalized_file_path = "C:/Users/Administrator/OneDrive - Amrita vishwa vidyapeetham/SEM6_CSE/NNDL_DRUG/LAB1/Custom_CNN_Features_Normalized.csv"
    df_normalized.to_csv(normalized_file_path, index=False)
    print(f"\nNormalized dataset saved to: {normalized_file_path}")

if __name__ == "__main__":
    main()
