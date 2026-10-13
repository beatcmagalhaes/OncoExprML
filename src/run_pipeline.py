# 1. Imports

from pathlib import Path
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold

# psutil is optional. If it is installed, RAM variation is saved.
try:
    import psutil
except ImportError:
    psutil = None


# 2. Define main project paths

PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_DIR / "input"
OUTPUT_DIR = PROJECT_DIR / "output"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"


# 3. Ensure output folders exist

OUTPUT_DIR.mkdir(exist_ok=True)
TABLES_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


# 4. Define input file paths

data_path = INPUT_DIR / "data.csv"
labels_path = INPUT_DIR / "labels.csv"


# 5. Check whether input files exist

if not data_path.exists():
    raise FileNotFoundError("data.csv was not found in the input folder.")

if not labels_path.exists():
    raise FileNotFoundError("labels.csv was not found in the input folder.")


# 6. Load data

print("Loading data...")

data = pd.read_csv(data_path, index_col=0)
labels = pd.read_csv(labels_path, index_col=0)


# 7. Validate data

print("Validating data...")

if not data.index.equals(labels.index):
    raise ValueError("Samples in data.csv and labels.csv are not in the same order.")

if "Class" not in labels.columns:
    raise ValueError("labels.csv does not contain a column named 'Class'.")

missing_values_data = data.isna().sum().sum()
if missing_values_data > 0:
    raise ValueError(f"Missing values found in data.csv: {missing_values_data}")

missing_values_labels = labels.isna().sum().sum()
if missing_values_labels > 0:
    raise ValueError(f"Missing values found in labels.csv: {missing_values_labels}")


# 8. Define features and target

X = data
y = labels["Class"]


# 9. Evaluate different PCA + K-Means configurations

print("Evaluating unsupervised PCA + K-Means model...")

pca_components_list = [10, 20, 50, 100]
clustering_results = []

for n_components in pca_components_list:

    clustering_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(
            n_components=n_components,
            svd_solver="randomized",
            random_state=42
        )),
        ("kmeans", KMeans(
            n_clusters=5,
            random_state=42,
            n_init=20
        ))
    ])

    clusters = clustering_pipeline.fit_predict(X)

    ari = adjusted_rand_score(y, clusters)
    nmi = normalized_mutual_info_score(y, clusters)

    clustering_results.append({
        "PCA components": n_components,
        "ARI": ari,
        "NMI": nmi
    })

clustering_results_df = pd.DataFrame(clustering_results)
clustering_results_df.to_csv(TABLES_DIR / "kmeans_pca_comparison.csv", index=False)


# 10. Select the best unsupervised configuration

best_clustering_config = clustering_results_df.sort_values(
    by="ARI",
    ascending=False
).iloc[0]

best_pca_components = int(best_clustering_config["PCA components"])

print("Best K-Means configuration:")
print("PCA components:", best_pca_components)
print("ARI:", best_clustering_config["ARI"])
print("NMI:", best_clustering_config["NMI"])


# 11. Train and save final unsupervised model

final_clustering_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(
        n_components=best_pca_components,
        svd_solver="randomized",
        random_state=42
    )),
    ("kmeans", KMeans(
        n_clusters=5,
        random_state=42,
        n_init=20
    ))
])

final_clusters = final_clustering_pipeline.fit_predict(X)

clustering_model_path = OUTPUT_DIR / "final_model_kmeans_pca.joblib"
joblib.dump(final_clustering_pipeline, clustering_model_path)

final_cluster_assignments = pd.DataFrame({
    "Sample": X.index,
    "True class": y.values,
    "Cluster": final_clusters
})
final_cluster_assignments.to_csv(TABLES_DIR / "final_kmeans_cluster_assignments.csv", index=False)

final_cluster_class_table = pd.crosstab(
    final_cluster_assignments["Cluster"],
    final_cluster_assignments["True class"]
)
final_cluster_class_table.to_csv(TABLES_DIR / "final_kmeans_cluster_class_table.csv")

final_clustering_metrics = pd.DataFrame([{
    "Best PCA components": best_pca_components,
    "ARI": best_clustering_config["ARI"],
    "NMI": best_clustering_config["NMI"],
    "Number of clusters": 5
}])
final_clustering_metrics.to_csv(TABLES_DIR / "final_kmeans_clustering_metrics.csv", index=False)


# 12. Split data into training and test sets

print("Splitting data into training and test sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 13. Define supervised candidate models

supervised_models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=5000,
            solver="lbfgs",
            random_state=42
        ))
    ]),
    "Linear SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="linear", random_state=42))
    ]),
    "Linear SVM + PCA 50": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(
            n_components=50,
            svd_solver="randomized",
            random_state=42
        )),
        ("model", SVC(kernel="linear", random_state=42))
    ]),
    "k-NN + PCA 50": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(
            n_components=50,
            svd_solver="randomized",
            random_state=42
        )),
        ("model", KNeighborsClassifier(n_neighbors=5, n_jobs=-1))
    ])
}


# 14. Fit models and compare train/test performance

print("Evaluating supervised models...")

model_comparison = []
overfitting_comparison = []
fitted_models = {}

for model_name, model in supervised_models.items():
    model.fit(X_train, y_train)
    fitted_models[model_name] = model

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_f1 = f1_score(y_train, y_train_pred, average="macro")
    test_f1 = f1_score(y_test, y_test_pred, average="macro")

    model_comparison.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_test_pred),
        "Precision macro": precision_score(y_test, y_test_pred, average="macro"),
        "Recall macro": recall_score(y_test, y_test_pred, average="macro"),
        "F1-score macro": test_f1
    })

    overfitting_comparison.append({
        "Model": model_name,
        "Train F1-score macro": train_f1,
        "Test F1-score macro": test_f1,
        "Train-test F1 gap": train_f1 - test_f1
    })

model_comparison_df = pd.DataFrame(model_comparison)
model_comparison_df.to_csv(TABLES_DIR / "model_comparison.csv", index=False)

overfitting_comparison_df = pd.DataFrame(overfitting_comparison)
overfitting_comparison_df.to_csv(TABLES_DIR / "overfitting_train_test_comparison.csv", index=False)

# Save model comparison figure
plt.figure(figsize=(8, 4))
plt.bar(
    model_comparison_df["Model"],
    model_comparison_df["F1-score macro"],
    edgecolor="black"
)
plt.title("Supervised model comparison")
plt.xlabel("Model")
plt.ylabel("Macro F1-score")
plt.ylim(0, 1.05)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "supervised_model_f1_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

# Save train vs test overfitting figure
x = np.arange(len(overfitting_comparison_df["Model"]))
width = 0.35
plt.figure(figsize=(8, 4))
plt.bar(
    x - width / 2,
    overfitting_comparison_df["Train F1-score macro"],
    width,
    label="Train",
    edgecolor="black"
)
plt.bar(
    x + width / 2,
    overfitting_comparison_df["Test F1-score macro"],
    width,
    label="Test",
    edgecolor="black"
)
plt.title("Train vs test performance")
plt.xlabel("Model")
plt.ylabel("Macro F1-score")
plt.ylim(0, 1.05)
plt.xticks(x, overfitting_comparison_df["Model"], rotation=20, ha="right")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "train_vs_test_f1_overfitting.png", dpi=300, bbox_inches="tight")
plt.close()


# 15. Cross-validation on training data

print("Running cross-validation on training data...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    "accuracy": "accuracy",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
    "f1_macro": "f1_macro"
}

cv_results = []

for model_name, model in supervised_models.items():
    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    cv_results.append({
        "Model": model_name,
        "Accuracy mean": scores["test_accuracy"].mean(),
        "Accuracy standard deviation": scores["test_accuracy"].std(),
        "Precision macro mean": scores["test_precision_macro"].mean(),
        "Recall macro mean": scores["test_recall_macro"].mean(),
        "F1-score macro mean": scores["test_f1_macro"].mean(),
        "F1-score macro standard deviation": scores["test_f1_macro"].std()
    })

cv_results_df = pd.DataFrame(cv_results)
cv_results_df.to_csv(TABLES_DIR / "cross_validation_results.csv", index=False)

# Save cross-validation F1 comparison figure
plt.figure(figsize=(8, 4))
plt.bar(
    cv_results_df["Model"],
    cv_results_df["F1-score macro mean"],
    yerr=cv_results_df["F1-score macro standard deviation"],
    capsize=4,
    edgecolor="black"
)
plt.title("Cross-validation macro F1-score")
plt.xlabel("Model")
plt.ylabel("Mean macro F1-score")
plt.ylim(0, 1.05)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "cross_validation_f1_comparison.png", dpi=300, bbox_inches="tight")
plt.close()


# 16. Select final supervised model between SVM variants

# The final algorithm remains Linear SVM. PCA is used only if the PCA variant
# has the best cross-validation macro F1-score among the SVM variants.

svm_variant_cv = cv_results_df[cv_results_df["Model"].isin([
    "Linear SVM",
    "Linear SVM + PCA 50"
])].copy()

svm_variant_cv.to_csv(TABLES_DIR / "svm_variant_cross_validation_comparison.csv", index=False)

best_svm_variant_name = svm_variant_cv.sort_values(
    by="F1-score macro mean",
    ascending=False
).iloc[0]["Model"]

final_model = supervised_models[best_svm_variant_name]
final_model.fit(X_train, y_train)

print("Selected final supervised model:", best_svm_variant_name)


# 17. Evaluate final model on the independent test set

print("Evaluating final supervised model on the test set...")

y_pred = final_model.predict(X_test)

final_predictions_df = pd.DataFrame({
    "Sample": X_test.index,
    "True class": y_test.values,
    "Predicted class": y_pred
})
final_predictions_df.to_csv(TABLES_DIR / "final_model_test_predictions.csv", index=False)

metrics = {
    "Final supervised model": best_svm_variant_name,
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision macro": precision_score(y_test, y_pred, average="macro"),
    "Recall macro": recall_score(y_test, y_pred, average="macro"),
    "F1-score macro": f1_score(y_test, y_pred, average="macro")
}

metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv(TABLES_DIR / "final_model_metrics.csv", index=False)

report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(TABLES_DIR / "final_model_classification_report.csv")

cm = confusion_matrix(y_test, y_pred, labels=final_model.classes_)
cm_df = pd.DataFrame(
    cm,
    index=[f"True_{label}" for label in final_model.classes_],
    columns=[f"Pred_{label}" for label in final_model.classes_]
)
cm_df.to_csv(TABLES_DIR / "final_model_confusion_matrix.csv")


# 18. Save final supervised model and metadata

model_path = OUTPUT_DIR / "final_model_supervised.joblib"
legacy_model_path = OUTPUT_DIR / "final_model_linear_svm.joblib"

joblib.dump(final_model, model_path)
joblib.dump(final_model, legacy_model_path)

final_model_metadata = pd.DataFrame([{
    "Final supervised model": best_svm_variant_name,
    "Model file": model_path.name,
    "Backward-compatible file": legacy_model_path.name,
    "Selection criterion": "Best cross-validation macro F1-score among Linear SVM variants"
}])
final_model_metadata.to_csv(TABLES_DIR / "final_supervised_model_metadata.csv", index=False)

# Save learning curve for the selected final SVM approach
print("Calculating learning curve for the final supervised model...")
train_sizes, train_scores, validation_scores = learning_curve(
    final_model,
    X,
    y,
    cv=cv,
    scoring="f1_macro",
    train_sizes=np.linspace(0.2, 1.0, 5),
    n_jobs=-1
)

learning_curve_df = pd.DataFrame({
    "Training size": train_sizes,
    "Train F1-score macro mean": train_scores.mean(axis=1),
    "Train F1-score macro standard deviation": train_scores.std(axis=1),
    "Validation F1-score macro mean": validation_scores.mean(axis=1),
    "Validation F1-score macro standard deviation": validation_scores.std(axis=1)
})
learning_curve_df.to_csv(TABLES_DIR / "learning_curve_final_svm.csv", index=False)

plt.figure(figsize=(7, 4))
plt.plot(
    learning_curve_df["Training size"],
    learning_curve_df["Train F1-score macro mean"],
    marker="o",
    label="Training score"
)
plt.plot(
    learning_curve_df["Training size"],
    learning_curve_df["Validation F1-score macro mean"],
    marker="o",
    label="Validation score"
)
plt.title(f"Learning curve - {best_svm_variant_name}")
plt.xlabel("Number of training samples")
plt.ylabel("Macro F1-score")
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "learning_curve_final_svm.png", dpi=300, bbox_inches="tight")
plt.close()


# 19. Rank the most discriminative anonymous features

print("Calculating ranking of the most discriminative anonymous features...")

top_feature_pipeline = Pipeline([
    ("variance", VarianceThreshold(threshold=0.0)),
    ("select", SelectKBest(score_func=f_classif, k=100))
])

top_feature_pipeline.fit(X_train, y_train)

variance_step = top_feature_pipeline.named_steps["variance"]
select_step = top_feature_pipeline.named_steps["select"]

features_after_variance = X_train.columns[variance_step.get_support()]
selected_features = features_after_variance[select_step.get_support()]
selected_scores = select_step.scores_[select_step.get_support()]

top_features_df = pd.DataFrame({
    "Feature": selected_features,
    "ANOVA F-score": selected_scores
}).sort_values(by="ANOVA F-score", ascending=False)

top_features_df.to_csv(TABLES_DIR / "top_selected_features.csv", index=False)


# 20. Empirical temporal and spatial cost analysis

print("Running simple empirical temporal/spatial cost analysis...")

process = psutil.Process(os.getpid()) if psutil is not None else None
complexity_empirical = []

for model_name, model in supervised_models.items():
    ram_before = process.memory_info().rss / (1024 ** 2) if process is not None else None
    t0 = time.perf_counter()

    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)

    t1 = time.perf_counter()
    ram_after = process.memory_info().rss / (1024 ** 2) if process is not None else None

    complexity_empirical.append({
        "Model": model_name,
        "Execution time (s)": t1 - t0,
        "RAM before (MB)": ram_before,
        "RAM after (MB)": ram_after,
        "RAM variation (MB)": (ram_after - ram_before) if process is not None else None,
        "Accuracy": accuracy_score(y_test, y_test_pred),
        "Precision macro": precision_score(y_test, y_test_pred, average="macro"),
        "Recall macro": recall_score(y_test, y_test_pred, average="macro"),
        "F1-score macro": f1_score(y_test, y_test_pred, average="macro")
    })

complexity_empirical_df = pd.DataFrame(complexity_empirical)
complexity_empirical_df.to_csv(TABLES_DIR / "empirical_computational_cost.csv", index=False)

# Save empirical temporal/spatial cost figures
plt.figure(figsize=(8, 4))
plt.bar(
    complexity_empirical_df["Model"],
    complexity_empirical_df["Execution time (s)"],
    edgecolor="black"
)
plt.title("Execution time by model")
plt.xlabel("Model")
plt.ylabel("Execution time (seconds)")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "empirical_execution_time_by_model.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(7, 5))
plt.scatter(
    complexity_empirical_df["Execution time (s)"],
    complexity_empirical_df["F1-score macro"],
    s=100,
    edgecolor="black"
)
for _, row in complexity_empirical_df.iterrows():
    plt.text(
        row["Execution time (s)"],
        row["F1-score macro"],
        row["Model"],
        fontsize=9,
        ha="left",
        va="bottom"
    )
plt.title("Execution time vs macro F1-score")
plt.xlabel("Execution time (seconds)")
plt.ylabel("Macro F1-score")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "empirical_time_vs_f1_score.png", dpi=300, bbox_inches="tight")
plt.close()

if psutil is not None:
    plt.figure(figsize=(8, 4))
    plt.bar(
        complexity_empirical_df["Model"],
        complexity_empirical_df["RAM variation (MB)"],
        edgecolor="black"
    )
    plt.title("Approximate RAM variation by model")
    plt.xlabel("Model")
    plt.ylabel("RAM variation (MB)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "empirical_ram_variation_by_model.png", dpi=300, bbox_inches="tight")
    plt.close()


# 21. Save computational complexity table

print("Saving computational complexity analysis...")

n_samples, n_features = X.shape
n_classes = y.nunique()
n_clusters = 5
n_folds = 5

complexity_table = pd.DataFrame({
    "Step": [
        "Data validation",
        "Standardization",
        "PCA",
        "K-Means",
        "Logistic Regression",
        "Linear SVM",
        "Linear SVM + PCA",
        "k-NN training",
        "k-NN prediction",
        "SelectKBest ANOVA",
        "Cross-validation"
    ],
    "Approximate temporal complexity": [
        "O(n × p)",
        "O(n × p)",
        "O(n × p × r)",
        "O(i × n × q × r)",
        "O(i × n × p × c)",
        "Approximately O(n × p) to O(n² × p)",
        "O(n × p × r) + SVM cost in reduced space",
        "O(n_train × r)",
        "O(n_test × n_train × r)",
        "O(n × p)",
        "O(F × model cost)"
    ],
    "Approximate spatial complexity": [
        "O(n × p)",
        "O(n × p)",
        "O(n × r + p × r)",
        "O(n × r + q × r)",
        "O(p × c)",
        "O(p × c)",
        "O(n × r + p × r)",
        "O(n_train × r)",
        "O(n_train × r)",
        "O(p)",
        "O(model memory per fold)"
    ],
    "Application in this project": [
        f"Validation of the matrix with {n_samples} samples and {n_features} features.",
        "Mean and standard deviation calculated for each expression feature.",
        f"Reduction from {n_features} features to principal components.",
        f"Clustering of samples into {n_clusters} clusters in PCA space.",
        f"Multiclass supervised baseline with {n_classes} classes.",
        "SVM classification using the original high-dimensional feature space.",
        "SVM classification after PCA dimensionality reduction.",
        "Stores training samples after PCA reduction.",
        "Computes distances between test and training samples.",
        "Selects anonymous features with higher discriminative power.",
        f"Repeats training and evaluation across {n_folds} folds."
    ]
})

complexity_table.to_csv(TABLES_DIR / "complexity_analysis.csv", index=False)


# 22. Show results in terminal

print("Pipeline completed successfully.")

print("Unsupervised model saved to:", clustering_model_path)
print("Unsupervised model metrics:")
print(final_clustering_metrics)

print("Final supervised model saved to:", model_path)
print("Final supervised model:", best_svm_variant_name)
print("Final supervised model metrics:")
print(metrics_df)

print("Overfitting train/test comparison:")
print(overfitting_comparison_df)

print("Top 5 most discriminative anonymous features:")
print(top_features_df.head())
