# 🧠 RandomForest_BreastCancer_Classifier

AutoBase is a compact, one-file **Machine Learning evaluation toolkit** that automates model training, cross-validation, and visualization.
It provides a **reproducible baseline** for ML benchmarking — ready to run, interpret, and extend.

---

## 📌 1. Research Question

Most ML projects start with repetitive tasks — setting up data, building baseline models, running evaluations, and saving metrics.
These steps are crucial but time-consuming.

> **Question:**
> How can we design a single, reproducible script that performs model training, evaluation, calibration, and visualization automatically?

---

## 💡 2. Proposed Solution

AutoBase provides a **self-contained Python script** that:

* Trains baseline ML models (`Logistic Regression`, `Random Forest`)
* Runs **Stratified K-Fold cross-validation**
* Computes key metrics: ROC-AUC, PR-AUC, F1, Accuracy
* Produces ROC, PR, and calibration plots
* Saves results, metrics, and models in an organized `artifacts/` folder

It serves as a **reliable baseline reference** for any ML project using tabular data.

---

## ⚙️ 3. Methodology

### 🧬 Dataset

* **Source:** `sklearn.datasets.load_breast_cancer`
* **Samples:** 569
* **Features:** 30 continuous variables
* **Target:** Binary (0 = malignant, 1 = benign)
* **Type:** Diagnostic features derived from breast tumor cell images

This dataset is lightweight, standardized, and ensures **full reproducibility**.

---

### 🧩 Models Used

| Model                   | Type                  | Description                                             | Why Used                                                |
| ----------------------- | --------------------- | ------------------------------------------------------- | ------------------------------------------------------- |
| **Logistic Regression** | Linear                | Estimates class probabilities via the sigmoid function. | Interpretable, efficient, and a strong linear baseline. |
| **Random Forest**       | Ensemble (non-linear) | Builds multiple decision trees via bagging.             | Captures complex, non-linear feature interactions.      |

#### Logistic Regression Settings

```python
LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=seed
)
```

#### Random Forest Settings

```python
RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=seed
)
```

---

## 🔁 4. Evaluation Workflow

| Step                       | Description                                               |
| -------------------------- | --------------------------------------------------------- |
| **Cross-Validation**       | Stratified 5-fold to maintain class balance.              |
| **Calibration (optional)** | Isotonic regression for reliable probabilities.           |
| **Metrics**                | ROC-AUC, PR-AUC, F1, Accuracy (mean ± std).               |
| **Plots**                  | ROC, PR, and calibration curves.                          |
| **Outputs**                | Metrics JSON, confusion matrix, trained model, PNG plots. |

---

## 🧱 5. Implementation Summary

### 📁 File Structure

```
ml-baseline/
├─ main.py
├─ requirements.txt
├─ README.md
└─ artifacts/         # created automatically
```

### ⚙️ Key Functions

* **get_models()** – defines baseline models
* **evaluate_cv()** – performs cross-validation and computes metrics
* **plot_and_save_curves()** – generates and saves ROC/PR/Calibration plots
* **main()** – orchestrates the full pipeline and writes outputs

---

## 🧠 6. Reasoning Behind Design

| Design Choice                  | Reason                                      |
| ------------------------------ | ------------------------------------------- |
| **Built-in dataset**           | Guarantees reproducibility                  |
| **Linear + Non-linear models** | Covers diverse data patterns                |
| **Cross-validation**           | Prevents overfitting, increases reliability |
| **Calibration curves**         | Tests probability quality                   |
| **Single file**                | Simplicity and portability                  |
| **JSON/PNG outputs**           | Easy to compare between runs                |

---

## 📊 7. Results

### Quantitative Results (mean ± std, 5-fold CV)

| Model               | ROC-AUC       | PR-AUC        | F1          | Accuracy    |
| ------------------- | ------------- | ------------- | ----------- | ----------- |
| Logistic Regression | 0.991 ± 0.005 | 0.988 ± 0.007 | 0.97 ± 0.01 | 0.96 ± 0.01 |
| Random Forest       | 0.993 ± 0.004 | 0.991 ± 0.005 | 0.97 ± 0.01 | 0.97 ± 0.01 |

Both models achieve **near-perfect discrimination**.
Random Forest slightly outperforms Logistic Regression due to its non-linear capacity.

### Qualitative Results

* **ROC Curve:** Smooth and close to the top-left corner.
* **PR Curve:** High precision across recall values.
* **Calibration Curve:** Accurate probability estimates after isotonic calibration.
* **Confusion Matrix:** Very few false negatives and false positives.

---

## 🧩 8. Usage

### 🧰 Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### ▶️ Example Commands

```bash
# Logistic Regression with 5-fold CV and calibration
python main.py --model logreg --cv 5 --calibrate

# Random Forest with 10-fold CV
python main.py --model rf --cv 10
```

### 📂 Output Files

```
artifacts/
├── best_model_logreg.joblib
├── metrics.json
├── confusion_matrix.csv
├── roc_curve.png
├── pr_curve.png
└── calibration_curve.png
```

---

## 🔍 9. Results Interpretation

* **High ROC-AUC (>0.99)** confirms strong separability.
* **Low standard deviation** indicates consistent performance across folds.
* **Calibration curves** show reliable probability estimates.
* **Random Forest** provides a slight performance edge, validating non-linear modeling.

---

## 🧾 Summary

| Category         | Description                                                   |
| ---------------- | ------------------------------------------------------------- |
| **Project Name** | AutoBase – Reproducible Baseline ML Evaluation Toolkit        |
| **Goal**         | Build a single-file ML pipeline for standardized benchmarking |
| **Models**       | Logistic Regression, Random Forest                            |
| **Dataset**      | Breast Cancer (scikit-learn built-in)                         |
| **Main Outputs** | Metrics, ROC/PR plots, Calibration curve, Saved model         |
| **Runtime**      | < 2 minutes on a standard laptop                              |
| **Outcome**      | Complete, reusable ML baseline system                         |

---

## 👩‍💻 Author

Developed by Seirana, generated with assistance from Leo.
