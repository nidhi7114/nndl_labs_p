import time
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler  # <-- Add this import
from sklearn.exceptions import NotFittedError


# Load dataset
def load_dataset(file_path):
    """Loads dataset and returns features (X) and labels (y)."""
    df = pd.read_csv("C:/Users/Administrator/OneDrive - Amrita vishwa vidyapeetham/SEM6_CSE/NNDL_DRUG/LAB1/Custom_CNN_Features.csv")
    X = df.iloc[:, 2:].values  # Features (ignoring first two columns if non-feature)
    y = df.iloc[:, 1].values   # Assuming second column is class label
    return X, y

# Train-Test Split (75% train, 25% test)
def split_data(X, y):
    """Splits data into train and test sets (75:25) and applies feature scaling."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    # Standardization (Feature Scaling)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

from sklearn.exceptions import NotFittedError

def evaluate_models(X_train, X_test, y_train, y_test):
    """Trains multiple classifiers, evaluates their performance, checks for underfitting/overfitting, and measures execution time."""
    
    models = {
        "k-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Support Vector Machine": SVC(probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Naïve Bayes": GaussianNB()
    }
    
    results = []

    for name, model in models.items():
        # Measure training time
        start_train = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_train
        
        # Measure inference time
        start_test = time.time()
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        test_time = time.time() - start_test

        # Try to get probability predictions
        try:
            y_prob = model.predict_proba(X_test)
            auroc = roc_auc_score(y_test, y_prob, multi_class='ovr')
        except (AttributeError, ValueError, NotFittedError):
            y_prob = None
            auroc = "N/A"

        # Train & Test Metrics
        metrics = {}
        for y_true, y_pred, dataset in [(y_train, y_pred_train, "Train"), (y_test, y_pred_test, "Test")]:
            metrics[dataset] = {
                "Accuracy": accuracy_score(y_true, y_pred),
                "Precision": precision_score(y_true, y_pred, average='weighted'),
                "Recall": recall_score(y_true, y_pred, average='weighted'),
                "F1-Score": f1_score(y_true, y_pred, average='weighted')
            }

        # Check for Overfitting/Underfitting
        overfit_status = "Regular Fit"
        if metrics["Train"]["Accuracy"] > metrics["Test"]["Accuracy"] + 0.10:  # Large gap
            overfit_status = "Overfit"
        elif metrics["Train"]["Accuracy"] < 0.60 and metrics["Test"]["Accuracy"] < 0.60:  # Low scores
            overfit_status = "Underfit"

        # Store Results
        results.append([
            name,
            metrics["Train"]["Accuracy"], metrics["Test"]["Accuracy"],
            metrics["Train"]["Precision"], metrics["Test"]["Precision"],
            metrics["Train"]["Recall"], metrics["Test"]["Recall"],
            metrics["Train"]["F1-Score"], metrics["Test"]["F1-Score"],
            auroc, overfit_status, train_time, test_time
        ])

    # Convert to DataFrame for better visualization
    results_df = pd.DataFrame(results, columns=[
        "Model", "Train Accuracy", "Test Accuracy",
        "Train Precision", "Test Precision",
        "Train Recall", "Test Recall",
        "Train F1-Score", "Test F1-Score",
        "AUROC", "Fit Status", "Train Time (s)", "Test Time (s)"
    ])

    print("\nModel Performance Summary:\n", results_df)
    return results_df

# Main Execution
def main():
    file_path = "/mnt/data/Custom_CNN_Features.csv"
    X, y = load_dataset(file_path)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    results_df = evaluate_models(X_train, X_test, y_train, y_test)
    
    # Save results
    results_df.to_csv("C:/Users/Administrator/OneDrive - Amrita vishwa vidyapeetham/SEM6_CSE/NNDL_DRUG/LAB1/Classification_Results_1.csv", index=False)
    print("\nResults saved to: Classification_Results_1.csv")

if __name__ == "__main__":
    main()
