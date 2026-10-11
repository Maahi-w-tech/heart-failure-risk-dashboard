# Presentation: Predicting Heart-Failure Events Using Gradient Boosting

## Slide 1 — Title

**Predicting Heart-Failure Events Using Gradient Boosting**  
UCI Heart Failure Clinical Records Dataset  
Mini-Project 36 | Name / Roll No.

**Speaker note:** This is an educational machine-learning study, not a clinical diagnostic system.

## Slide 2 — Objective and motivation

- Predict `DEATH_EVENT`: 0 = no recorded death event, 1 = death event during follow-up.
- Compare Gradient Boosting with simple baselines.
- Prioritize recall and PR-AUC because missed positive events matter.
- Research question: does tuned Gradient Boosting improve event detection without an unacceptable false-alarm trade-off?

## Slide 3 — Dataset

- UCI Heart Failure Clinical Records Dataset
- 299 patient records, 12 predictors, 1 binary target
- Clinical variables: age, ejection fraction, serum creatinine, serum sodium, platelets, CPK
- Indicators: anaemia, diabetes, blood pressure, sex, smoking
- `time`: follow-up duration; retained but discussed as a limitation
- Missingness and duplicates verified by code

## Slide 4 — Outcome distribution and preprocessing

- Positive events are the minority class.
- 80/20 stratified train/test split, `random_state=42`
- Test set held out until the end.
- Logistic Regression: standardization in a pipeline.
- Tree models: no scaling required.
- No automatic outlier deletion because unusual clinical values may be meaningful.

**Visual:** `results/class_balance.png`

## Slide 5 — How Gradient Boosting works

- Start with an initial probability based on event prevalence.
- Fit a shallow tree to current errors/residuals.
- Add a small learning-rate-scaled correction.
- Repeat for many estimators.
- Final prediction is the additive ensemble of weak learners.

\[F_m(x)=F_{m-1}(x)+\eta h_m(x)\]

## Slide 6 — Models and tuning

- Majority-class baseline
- Logistic Regression
- Constrained Decision Tree
- Tuned Gradient Boosting classifier
- Five-fold stratified cross-validation
- Randomized search over estimators, learning rate, depth, leaf size, and subsampling
- Tuning score: recall

## Slide 7 — Evaluation metrics

- Accuracy: overall correctness
- Precision: among predicted events, how many were events?
- Recall: among true events, how many were detected?
- F1: balance of precision and recall
- Specificity: correct non-event rate
- Balanced accuracy, ROC-AUC, PR-AUC

**Manual check:** `Precision = TP/(TP+FP)`, `Recall = TP/(TP+FN)`, `F1 = 2PR/(P+R)` using the final confusion matrix.

## Slide 8 — Results

Insert the generated `results/metrics.csv` values:

| Model | Recall | F1 | Balanced accuracy | PR-AUC |
|---|---:|---:|---:|---:|
| Majority baseline | 0.000 | 0.000 | 0.500 | 0.317 |
| Logistic Regression | 0.579 | 0.667 | 0.753 | 0.757 |
| Decision Tree | 0.579 | 0.688 | 0.765 | 0.776 |
| Tuned Gradient Boosting | 0.474 | 0.621 | 0.725 | 0.817 |

**Visual:** `results/confusion_matrix.png`

## Slide 9 — Curves and feature interpretation

- ROC and PR curves compare ranking quality across thresholds.
- PR-AUC helps reveal performance on the minority event class.
- Feature importance identifies variables used heavily by the fitted ensemble.
- Importance is associative, not causal.

**Visuals:** `results/roc_pr_curves.png` and `results/feature_importance.png`

## Slide 10 — Conclusion and limitations

- Gradient Boosting provides a nonlinear ensemble comparison for this medical-risk dataset.
- On the holdout set it achieved ROC-AUC 0.899 and PR-AUC 0.817, but recall was 0.474.
- Select the final model using recall/precision trade-offs, not accuracy alone.
- Small sample size means results are uncertain and split-sensitive.
- Follow-up time may create target-proxy/leakage concerns for early prediction.
- External validation and clinical review are required before any real-world use.

**Reproduce:** `python src/run_analysis.py`
