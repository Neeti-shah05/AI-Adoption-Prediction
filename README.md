# AI-Adoption-Prediction

Predicts company-level AI adoption outcomes using Random Forest, Elastic Net, and XGBoost built in R. Includes a full pipeline from data cleaning through hyperparameter tuning and evaluation.

What it does

Cleans and preprocesses data using a recipes pipeline (imputation, one-hot encoding, rare level grouping, near-zero-variance removal)
Splits data 60/20/20 (train/validation/test) with stratified sampling
Tunes Random Forest via two-stage OOB + validation AUC search; Elastic Net via 5-fold cross-validated lambda; XGBoost via early stopping
Evaluates all three models on the same held-out test set: Accuracy, ROC-AUC, PR-AUC, Log Loss, Macro F1, Micro F1
Saves all plots, confusion matrices, feature importance charts, and a unified metrics CSV to ./outputs/

Works for both binary and multiclass classification without code changes.
