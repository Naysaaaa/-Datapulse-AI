# simple_ml_examples.py
"""
Simple, beginner‑friendly examples illustrating:
1️⃣ Supervised learning – classification with scikit‑learn's LogisticRegression.
2️⃣ Unsupervised learning – clustering with scikit‑learn's KMeans.
The script is self‑contained, runs without external files, and prints concise results.
"""

# ------------------------------------------------------------
# Supervised Learning Example (Classification)
# ------------------------------------------------------------
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def supervised_example():
    """Train a LogisticRegression on a synthetic binary classification dataset.
    Prints training/validation accuracy.
    """
    # Generate a simple binary classification dataset
    X, y = make_classification(
        n_samples=200,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=42,
    )
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    # Fit Logistic Regression
    model = LogisticRegression(solver="lbfgs")
    model.fit(X_train, y_train)
    # Predict and evaluate
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print("[Supervised] LogisticRegression Accuracy:")
    print(f"  Train: {train_acc:.2f}, Test: {test_acc:.2f}\n")

# ------------------------------------------------------------
# Unsupervised Learning Example (Clustering)
# ------------------------------------------------------------
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
# pyrefly: ignore [missing-import]
import numpy as np

def unsupervised_example():
    """Generate synthetic data and cluster it with KMeans.
    Prints the cluster centers and a simple silhouette‑like measure.
    """
    X, _ = make_blobs(
        n_samples=200,
        centers=3,
        n_features=2,
        cluster_std=1.0,
        random_state=42,
    )
    # Fit KMeans with 3 clusters (we know the true number)
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X)
    # Compute average intra‑cluster distance as a rough quality metric
    intra_dist = np.mean([
        np.linalg.norm(X[labels == i] - kmeans.cluster_centers_[i], axis=1).mean()
        for i in range(3)
    ])
    print("[Unsupervised] KMeans clustering results:")
    print(f"  Cluster centers:\n{kmeans.cluster_centers_}")
    print(f"  Avg. intra‑cluster distance: {intra_dist:.2f}\n")

if __name__ == "__main__":
    supervised_example()
    unsupervised_example()
