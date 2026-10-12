# Mini-Project 36: Predicting Heart-Failure Events with Gradient Boosting

This project predicts whether a patient dies during the follow-up period using the **UCI Heart Failure Clinical Records Dataset**. It is intentionally designed to satisfy the supplied 100-point mini-project rubric without becoming unnecessarily complex.

## Deliverables

- `src/run_analysis.py` — reproducible download, preprocessing, tuning, evaluation, and plot generation script.
- `report.md` — ready-to-submit written report structure.
- `presentation.md` — 10-slide presentation outline with speaker notes.
- `requirements.txt` — Python dependencies.
- `results/` — generated metrics, tuning output, and figures after running the script.
- `results/dashboard_data.json` — generated explainability artifact containing actual test probabilities, threshold metrics, model metadata, and record-level contributions.
- `web/` — Vercel-ready interactive Heart Failure Risk Lab explainability platform. It consumes `web/data/dashboard_data.json`, the deployable copy of the actual Python pipeline artifact, and visualizes data, preprocessing, the fitted Gradient Boosting model, held-out probabilities, threshold behavior, evaluation, and record-level explanations.

## Project objective

Build and evaluate a Gradient Boosting classifier for the binary target `DEATH_EVENT`. Because a missed high-risk patient is more concerning than a false alarm, recall, F1-score, balanced accuracy, and PR-AUC are reported alongside accuracy.

## Run

From this project directory:

```bash
python -m pip install -r requirements.txt
python src/run_analysis.py
```

The script downloads the official UCI archive once into `data/`, records its SHA-256 hash, and writes outputs to `results/`. The final holdout test set is not used during hyperparameter tuning.

## Deploy the interactive version to Vercel

The dashboard is a static client-side app, so it does not need Python, a database, an API key, or a serverless function. It includes a demo mode using the completed experiment and an upload CSV option. Uploaded files are parsed in the browser and are not sent to a server.

1. Push this project to GitHub, or import the repository directly into Vercel.
2. Set **Root Directory** to `mini-project-36-heart-failure/web`.
3. Use the default static settings (no build command; output is the `web` folder), or set the framework preset to **Other**.
4. Deploy.

The web flow is intentionally ordered as **dataset → validation → preprocessing → features → Gradient Boosting → probability → threshold → prediction → evaluation**. Running `python src/run_analysis.py` now also writes `results/dashboard_data.json` and the deployable copy `web/data/dashboard_data.json`. These artifacts contain actual test-set probabilities, threshold metrics, model parameters, tree metadata, global feature importance, cross-validation summary, and exact tree-path additive explanations for the held-out records. The frontend does not train a second model or fabricate patient data. Its optional CSV upload is an explorer-only preview; saved predictions and evaluation remain tied to the existing Python model run.

## Dataset source

UCI Machine Learning Repository, Heart Failure Clinical Records Dataset:
<https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records>

The dataset has 299 records, 12 predictors, and the binary target `DEATH_EVENT`. The `time` feature represents follow-up duration. It is retained for the primary model because it is part of the supplied dataset, but the report explicitly discusses its interpretation and includes a sensitivity-analysis hook.

## Rubric coverage

| Rubric item | Evidence |
|---|---|
| Dataset understanding | `report.md` dataset table and generated `dataset_summary.txt` |
| Preprocessing | Missingness, duplicates, class balance, stratified split, leakage-safe modeling |
| Algorithm implementation | Manual Gradient Boosting explanation and update equations in the report |
| Modeling | Logistic Regression, Decision Tree, and tuned Gradient Boosting |
| Tuning | `RandomizedSearchCV` over estimator count, learning rate, depth, leaf size, and subsampling |
| Metrics | Accuracy, precision, recall, F1, specificity, balanced accuracy, ROC-AUC, PR-AUC |
| Manual calculation | Confusion-matrix precision/recall/F1 calculation in the report |
| Visualization | Class balance, ROC/PR curves, confusion matrix, and feature importance |
| Interpretation | Clinical-context discussion and limitations |
| Presentation | `presentation.md` |

This is an educational prediction exercise, not a clinical diagnostic tool.
