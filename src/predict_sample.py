# 1. Imports

from pathlib import Path
import sys

import pandas as pd
import joblib


# 2. Define main project paths

PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_DIR / "input"
OUTPUT_DIR = PROJECT_DIR / "output"
TABLES_DIR = OUTPUT_DIR / "tables"

data_path = INPUT_DIR / "data.csv"
labels_path = INPUT_DIR / "labels.csv"

supervised_model_path = OUTPUT_DIR / "final_model_supervised.joblib"
legacy_svm_model_path = OUTPUT_DIR / "final_model_linear_svm.joblib"
kmeans_model_path = OUTPUT_DIR / "final_model_kmeans_pca.joblib"

metadata_path = TABLES_DIR / "final_supervised_model_metadata.csv"
cluster_table_path = TABLES_DIR / "final_kmeans_cluster_class_table.csv"


# 3. Check whether main files exist

if not data_path.exists():
    raise FileNotFoundError("data.csv was not found in the input folder.")

if not labels_path.exists():
    raise FileNotFoundError("labels.csv was not found in the input folder.")

if supervised_model_path.exists():
    model_path_to_load = supervised_model_path
elif legacy_svm_model_path.exists():
    model_path_to_load = legacy_svm_model_path
else:
    raise FileNotFoundError("The supervised model was not found. Run src/run_pipeline.py first.")


# 4. Load data and supervised model

print("Loading data and supervised model...")

data = pd.read_csv(data_path, index_col=0)
labels = pd.read_csv(labels_path, index_col=0)

supervised_model = joblib.load(model_path_to_load)


# 5. Load supervised model metadata, if available

if metadata_path.exists():
    metadata = pd.read_csv(metadata_path)
    supervised_model_name = metadata.loc[0, "Final supervised model"]
else:
    supervised_model_name = "Linear SVM"


# 6. Try to load unsupervised model

kmeans_available = kmeans_model_path.exists()

if kmeans_available:
    print("Loading unsupervised PCA + K-Means model...")
    kmeans_model = joblib.load(kmeans_model_path)
else:
    print("Warning: PCA + K-Means model was not found. Cluster assignment will be skipped.")
    kmeans_model = None


# 7. Tumour class descriptions

tumor_info = {
    "BRCA": "Breast invasive carcinoma",
    "KIRC": "Kidney renal clear cell carcinoma",
    "COAD": "Colon adenocarcinoma",
    "LUAD": "Lung adenocarcinoma",
    "PRAD": "Prostate adenocarcinoma"
}


# 8. Select sample

# If the user provides a sample ID in the terminal, that sample is used.
# Otherwise, the first sample in the dataset is used.

if len(sys.argv) > 1:
    sample_id = sys.argv[1]
else:
    sample_id = data.index[0]


# 9. Check whether sample exists

if sample_id not in data.index:
    raise ValueError("The selected sample does not exist in the dataset.")


# 10. Prepare sample

sample = data.loc[[sample_id]]


# 11. Obtain unsupervised cluster, if model exists

if kmeans_available:
    predicted_cluster = kmeans_model.predict(sample)[0]
else:
    predicted_cluster = None


# 12. Interpret cluster if cluster vs class table exists

cluster_dominant_class = None

if predicted_cluster is not None and cluster_table_path.exists():
    cluster_class_table = pd.read_csv(cluster_table_path, index_col=0)

    # The CSV index may be read as text, so the cluster ID is converted to string.
    cluster_id_as_text = str(predicted_cluster)
    cluster_class_table.index = cluster_class_table.index.astype(str)

    if cluster_id_as_text in cluster_class_table.index:
        cluster_dominant_class = cluster_class_table.loc[cluster_id_as_text].idxmax()


# 13. Obtain true class and supervised prediction

true_class = labels.loc[sample_id, "Class"]
predicted_class = supervised_model.predict(sample)[0]

if true_class == predicted_class:
    supervised_result = "correct classification"
else:
    supervised_result = "incorrect classification"


# 14. Show results in terminal

print("")
print("Sample analysis result")
print("----------------------")
print("Sample:", sample_id)

if predicted_cluster is not None:
    print("")
    print("Unsupervised model: PCA + K-Means")
    print("Assigned cluster:", predicted_cluster)

    if cluster_dominant_class is not None:
        print(
            "Most frequent class in this cluster:",
            cluster_dominant_class,
            "-",
            tumor_info[cluster_dominant_class]
        )

    print("Note: K-Means assigns clusters, not tumour classes directly.")
else:
    print("")
    print("Unsupervised model: not available.")

print("")
print("Supervised model:", supervised_model_name)
print("True class:", true_class, "-", tumor_info[true_class])
print("Predicted class:", predicted_class, "-", tumor_info[predicted_class])
print("Supervised result:", supervised_result)


# 15. Save prediction results

# Save a small CSV table so the sample prediction can also be reused in reports.
sample_prediction_df = pd.DataFrame([{
    "Sample": sample_id,
    "True class": true_class,
    "Predicted class": predicted_class,
    "Supervised result": supervised_result,
    "Supervised model": supervised_model_name,
    "Assigned cluster": predicted_cluster,
    "Cluster dominant class": cluster_dominant_class
}])

sample_prediction_df.to_csv(
    TABLES_DIR / "sample_prediction_result.csv",
    index=False
)

# 16. Save simple Markdown report

report_path = OUTPUT_DIR / "sample_prediction_report.md"

report_text = f"""# Sample Prediction Report

## Sample information

- Sample ID: `{sample_id}`
- Number of input features: {sample.shape[1]}
"""

if predicted_cluster is not None:
    report_text = report_text + f"""

## Unsupervised model: PCA + K-Means

- Assigned cluster: `{predicted_cluster}`
"""

    if cluster_dominant_class is not None:
        report_text = report_text + f"- Most frequent class in this cluster: `{cluster_dominant_class}` - {tumor_info[cluster_dominant_class]}\n"

    report_text = report_text + """

Important: K-Means assigns numerical clusters. These clusters are interpreted only after comparison with the real tumour classes.
"""

else:
    report_text = report_text + """

## Unsupervised model: PCA + K-Means

The unsupervised model was not available. Run `src/run_pipeline.py` first.
"""

report_text = report_text + f"""

## Supervised model: {supervised_model_name}

- True class: `{true_class}` - {tumor_info[true_class]}
- Predicted class: `{predicted_class}` - {tumor_info[predicted_class]}
- Result: **{supervised_result}**
"""

report_text = report_text + """

## Important note

This prediction is part of an educational bioinformatics pipeline. It should not be interpreted as a clinical diagnosis or treatment recommendation.
"""

with open(report_path, "w", encoding="utf-8") as file:
    file.write(report_text)

print("")
print("Report saved to:", report_path)
