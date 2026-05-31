"""
============================================================
COMP 30043 - Machine Intelligence Systems - CW2 (Task 1)
Comparative Analysis of Decision Tree (DT) and K-Nearest
Neighbor (KNN) Algorithms on the Brain Tumor Dataset
============================================================
Dataset source: https://www.kaggle.com/datasets/miadul/brain-tumor-dataset
File expected:  brain_tumor_dataset.csv (place in same folder as this script)
============================================================
"""


import numpy as np                                                # Numerical operations
import pandas as pd                                               # Tabular data handling
import matplotlib.pyplot as plt                                   # Plotting
import seaborn as sns                                             # Statistical plots
from sklearn.model_selection import train_test_split, cross_val_score  # Splitting & CV
from sklearn.preprocessing import LabelEncoder, StandardScaler    # Encoding & scaling
from sklearn.tree import DecisionTreeClassifier, plot_tree        # DT model
from sklearn.neighbors import KNeighborsClassifier                # KNN model
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


df = pd.read_csv("brain_tumor_dataset.csv")
print("Initial shape of dataset :", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nDataset information:")
df.info()


print("\nMissing values per column:\n", df.isnull().sum())


for col in df.columns:
    if df[col].dtype in ["int64", "float64"]:
        df[col] = df[col].fillna(df[col].mean())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])


df = df.drop_duplicates()
print("\nShape after null & duplicate handling:", df.shape)


numeric_cols = df.select_dtypes(include=[np.number]).columns
Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1
outlier_mask = ~((df[numeric_cols] < (Q1 - 1.5 * IQR)) |
                 (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
df = df[outlier_mask]
print("Shape after outlier removal :", df.shape)


encoder = LabelEncoder()
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = encoder.fit_transform(df[col].astype(str))


TARGET_COL = "MRI_Result"
X = df.drop(TARGET_COL, axis=1)
y = df[TARGET_COL]
print(f"\nUsing '{TARGET_COL}' as the target column")
print("Feature matrix shape :", X.shape)
print("Class distribution   :\n", y.value_counts())


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTraining samples : {len(X_train)}  |  Testing samples : {len(X_test)}")


dt_model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=RANDOM_STATE,
)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)


k_values = list(range(1, 21, 2))                                  # 1, 3, 5 ... 19
cv_scores = []
for k in k_values:
    knn_tmp = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn_tmp, X_train, y_train, cv=5,
                            scoring="accuracy").mean()
    cv_scores.append(score)

best_k = k_values[int(np.argmax(cv_scores))]
print(f"\nBest K value found via 5-fold cross-validation : {best_k}")


knn_model = KNeighborsClassifier(n_neighbors=best_k, metric="minkowski", p=2)
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)


def evaluate(y_true, y_pred, model_name):
    """Return a dictionary of accuracy, precision, recall and F1 (weighted)."""
    return {
        "Model"     : model_name,
        "Accuracy"  : accuracy_score(y_true, y_pred),
        "Precision" : precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall"    : recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1-score"  : f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

dt_results  = evaluate(y_test, dt_pred,  "Decision Tree")
knn_results = evaluate(y_test, knn_pred, f"KNN (k={best_k})")
results_df = pd.DataFrame([dt_results, knn_results])
print("\n========== PERFORMANCE COMPARISON ==========")
print(results_df.to_string(index=False))

print("\n--- Decision Tree classification report ---")
print(classification_report(y_test, dt_pred, zero_division=0))
print("\n--- KNN classification report ---")
print(classification_report(y_test, knn_pred, zero_division=0))


metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
dt_vals  = [dt_results[m]  for m in metrics]
knn_vals = [knn_results[m] for m in metrics]

x = np.arange(len(metrics))
width = 0.35
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width/2, dt_vals,  width, label="Decision Tree", color="#1f77b4")
ax.bar(x + width/2, knn_vals, width, label=f"KNN (k={best_k})", color="#ff7f0e")
ax.set_xticks(x); ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
ax.set_title("Decision Tree vs KNN — Performance Metrics")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig("fig1_metrics_comparison.png", dpi=150); plt.close()


plt.figure(figsize=(8, 5))
plt.plot(k_values, cv_scores, marker="o", color="#2ca02c")
plt.axvline(best_k, color="red", linestyle="--", label=f"Best K = {best_k}")
plt.title("KNN — Cross-Validation Accuracy vs K")
plt.xlabel("K (number of neighbours)"); plt.ylabel("CV Accuracy")
plt.grid(alpha=0.3); plt.legend()
plt.tight_layout(); plt.savefig("fig2_knn_k_tuning.png", dpi=150); plt.close()


fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
sns.heatmap(confusion_matrix(y_test, dt_pred),  annot=True, fmt="d",
            cmap="Blues",   ax=axes[0])
axes[0].set_title("Decision Tree — Confusion Matrix")
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
sns.heatmap(confusion_matrix(y_test, knn_pred), annot=True, fmt="d",
            cmap="Oranges", ax=axes[1])
axes[1].set_title(f"KNN (k={best_k}) — Confusion Matrix")
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")
plt.tight_layout(); plt.savefig("fig3_confusion_matrices.png", dpi=150); plt.close()


plt.figure(figsize=(16, 8))
plot_tree(dt_model, feature_names=X.columns, filled=True,
          rounded=True, fontsize=8)
plt.title("Decision Tree Structure (max_depth = 5)")
plt.tight_layout(); plt.savefig("fig4_decision_tree.png", dpi=150); plt.close()

print("\nAll figures saved to current directory:")
print("  fig1_metrics_comparison.png")
print("  fig2_knn_k_tuning.png")
print("  fig3_confusion_matrices.png")
print("  fig4_decision_tree.png")


better = "Decision Tree" if dt_results["Accuracy"] >= knn_results["Accuracy"] else "KNN"
print(f"\nConclusion: {better} performed better on this dataset.")
