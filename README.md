# OncoExprML

OncoExprML is a reproducible bioinformatics and machine learning project for analysing processed RNA-Seq tumour expression profiles.

The project explores whether RNA-Seq expression profiles contain tumour-type-associated structure and evaluates supervised models for classifying samples into five tumour classes.

This repository includes the written article, presentation, notebook, Python scripts, outputs, trained models and Docker configuration required to reproduce the main workflow.

---

## Note on project revisions

Following the feedback received during the presentation, the project was revised to make the methodology and implementation clearer. In particular, the explanations of PCA, K-Means, Linear SVM, model selection, validation procedures and the transition from exploratory notebook analysis to reproducible Python scripts were improved. The necessary project files were also organized inside the project folder to make the workflow easier to inspect, run and reproduce.

---

## Dataset

The project uses the **UCI Gene Expression Cancer RNA-Seq** dataset.

The input files are stored in `input/`:

```text
input/
├── data.csv
└── labels.csv
```

Dataset summary:

| Field | Information |
|---|---|
| Samples | 801 |
| Expression features | 20,531 |
| Tumour classes | BRCA, COAD, KIRC, LUAD, PRAD |
| Input format | Processed expression matrix + labels |
| Feature identifiers | Anonymous features, from `gene_0` to `gene_20530` |

Because the features are anonymous, selected features can be interpreted only as computationally discriminative variables. They cannot be directly interpreted as real genes or biomarkers without an external mapping file.

---

## Project structure

```text
OncoExprML/
├── artigo/
│   └── AnaSilva_AndreiaVieira_BeatrizMagalhaes_MarianaBarbosa_SalvadorMota_Artigo.docx
├── input/
│   ├── data.csv
│   └── labels.csv
├── notebooks/
│   └── OncroExprML.ipynb
├── output/
│   ├── figures/
│   ├── tables/
│   ├── final_model_kmeans_pca.joblib
│   ├── final_model_linear_svm.joblib
│   ├── final_model_supervised.joblib
│   └── sample_prediction_report.md
├── ppt/
│   └── AnaSilva_AndreiaVieira_BeatrizMagalhaes_MarianaBarbosa_SalvadorMota_OncoExprML.pptx
├── src/
│   ├── run_pipeline.py
│   └── predict_sample.py
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Workflow overview

The work was developed in two main phases.

First, the exploratory and experimental analysis was performed in the Jupyter Notebook:

```text
notebooks/OncroExprML.ipynb
```

The notebook includes:

- data loading and validation;
- exploratory data analysis;
- class distribution analysis;
- initial PCA visualization;
- unsupervised analysis with PCA + K-Means;
- supervised model comparison;
- Linear SVM model selection;
- cross-validation;
- feature selection;
- lightweight label-shuffling control;
- computational complexity analysis.

After the exploratory phase, the main workflow was transferred into Python scripts to improve reproducibility:

```text
src/run_pipeline.py
src/predict_sample.py
```

---

## Main methods

### Exploratory analysis and preprocessing

The initial analysis included:

- verification of data dimensions;
- verification of missing values;
- class distribution analysis;
- initial PCA visualization;
- standardization with `StandardScaler`.

Standardization was used because PCA, K-Means, SVM and k-NN are sensitive to feature scale.

### Unsupervised analysis

The unsupervised workflow used:

```text
StandardScaler → PCA → K-Means
```

Different PCA component values were tested before K-Means:

| PCA components | ARI | NMI |
|---:|---:|---:|
| 10 | 0.794 | 0.854 |
| 20 | 0.809 | 0.860 |
| 50 | 0.815 | 0.865 |
| 100 | 0.797 | 0.858 |

The best configuration used **50 PCA components**, with:

- **ARI = 0.815**
- **NMI = 0.865**

The labels were not used during clustering. They were used only afterwards to evaluate the agreement between clusters and real tumour classes.

### Supervised analysis

Three supervised approaches were compared:

| Model | Role in the project |
|---|---|
| Logistic Regression | Linear baseline |
| Linear SVM | Final selected supervised model |
| k-NN + PCA | Distance-based comparison model |

The final selected model was **Linear SVM**, implemented with a scikit-learn pipeline:

```text
StandardScaler → SVC(kernel="linear")
```

Linear SVM was selected because it achieved the best supervised performance and is suitable for high-dimensional expression datasets.

Model comparison on the test set:

| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.981 | 0.977 | 0.973 | 0.975 |
| Linear SVM | 0.994 | 0.997 | 0.993 | 0.995 |
| k-NN + PCA | 0.988 | 0.994 | 0.986 | 0.989 |

---

## Validation steps

The supervised workflow was evaluated using:

- accuracy;
- macro precision;
- macro recall;
- macro F1-score;
- confusion matrix;
- stratified cross-validation;
- feature selection;
- lightweight label-shuffling control.

Macro F1-score was used because the classes are not perfectly balanced. It gives the same weight to each tumour class.

The label-shuffling control was included as a lightweight sanity check. It is not a formal permutation test.

---

## Complexity analysis

The notebook and outputs include an approximate computational complexity analysis.

Main notation:

| Symbol | Meaning |
|---|---|
| `n` | number of samples |
| `p` | number of expression features |
| `r` | number of PCA components |
| `q` | number of clusters |
| `i` | number of iterations |
| `F` | number of cross-validation folds |

Examples:

| Step | Approximate complexity |
|---|---|
| StandardScaler | `O(n × p)` |
| PCA | `O(n × p × r)` |
| K-Means | `O(i × n × q × r)` |
| k-NN prediction | `O(n_test × n_train × r)` |
| Cross-validation | `O(F × model cost)` |

This analysis was included to explain how computational cost scales with dataset size and why dimensionality reduction is useful before methods such as K-Means and k-NN.

---

## Running locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the main pipeline:

```bash
python src/run_pipeline.py
```

Run the sample prediction script:

```bash
python src/predict_sample.py sample_0
```

The scripts read data from `input/` and write generated results to `output/`.

---

## Running with Docker

Build the Docker image:

```bash
docker build -t oncoexprml .
```

Run the full pipeline:

```bash
docker run --rm -v "$PWD/output:/app/output" oncoexprml
```

Run prediction for a sample:

```bash
docker run --rm -v "$PWD/output:/app/output" oncoexprml python src/predict_sample.py sample_0
```

On Windows PowerShell, the mounted path may need to be written explicitly, for example:

```powershell
docker run --rm -v "C:/Users/USERNAME/Desktop/OncoExprML/output:/app/output" oncoexprml
```

---

## Important outputs

Main figures are stored in:

```text
output/figures/
```

Main tables are stored in:

```text
output/tables/
```

Trained models are stored in:

```text
output/final_model_kmeans_pca.joblib
output/final_model_linear_svm.joblib
output/final_model_supervised.joblib
```

Example prediction report:

```text
output/sample_prediction_report.md
```

---

## Limitations

This project has important limitations:

- the input is a processed RNA-Seq expression matrix, not raw sequencing reads;
- features are anonymous and cannot be mapped directly to real genes;
- no external validation cohort was used;
- the project should not be interpreted as a clinical diagnostic tool;
- K-Means assigns clusters, not tumour diagnoses;
- the supervised model classifies samples only among the five known tumour classes in the dataset.

---

## Authors

Ana Silva, Andreia Vieira, Beatriz Magalhães, Mariana Barbosa and Salvador Mota.

---

## Final note

OncoExprML is an educational bioinformatics project focused on RNA-Seq expression analysis, machine learning model comparison and reproducible computational workflows.
