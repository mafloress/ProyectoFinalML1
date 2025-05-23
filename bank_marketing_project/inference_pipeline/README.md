# Batch Inference Pipeline for Bank Marketing Prediction

## Objective
This pipeline loads a pre-trained classification model and generates batch predictions on new, unseen feature data. The predictions indicate whether a customer is likely to subscribe to a term deposit.

## Input
-   **`best_tuned_model.joblib`**: The trained and tuned scikit-learn compatible model. This file is expected to be located in the `../training_pipeline/` directory.
-   **`bank-features-selected.csv`**: A CSV file containing the feature data for which predictions are to be made. This data must have undergone the same preprocessing and feature selection steps as the data used for training the model. It is expected to be located in the `../feature_pipeline/` directory.
-   **`selected_feature_names.json`** (optional but recommended for robustness): A JSON file from `../feature_pipeline/` listing the names of the features the model was trained on. This helps ensure column consistency.

## Processing Steps
The `batch_inference.ipynb` notebook executes the following key steps:

1.  **Load Trained Model**:
    -   Loads the serialized model (e.g., `best_tuned_model.joblib`) using `joblib`.
    -   Attempts to load the list of feature names the model was trained on (e.g., from `selected_feature_names.json` or a model attribute like `feature_names_in_`).

2.  **Load Feature Data for Inference**:
    -   Loads the new batch data from `bank-features-selected.csv`.
    -   Removes any target column (e.g., 'y') if present, as it's not needed for inference.
    -   **Feature Consistency Check**: Ensures that the columns in the inference data match the features the model was trained on (name and order). It will attempt to reorder columns or subset them if `trained_model_features` are available. Warnings or errors are issued if critical discrepancies are found (e.g., missing expected features).

3.  **Generate Predictions**:
    -   Uses the loaded model's `predict()` method on the prepared feature data (X_inference) to generate class label predictions (0 or 1).
    -   If the model supports it, `predict_proba()` is used to generate prediction probabilities for the positive class.

4.  **Save Predictions**:
    -   Creates a new DataFrame containing the original features from the input batch data.
    -   Adds new columns to this DataFrame:
        -   `predicted_label`: The predicted class label.
        -   `prediction_probability_yes`: The probability of predicting 'yes' (if available).
    -   Saves this combined DataFrame to `predictions.csv` in the `bank_marketing_project/inference_pipeline/` directory.

## Output
-   **`predictions.csv`**: A CSV file containing the original input features along with the `predicted_label` and (if applicable) `prediction_probability_yes` for each instance in the batch.

## How to Run
1.  Ensure that the training pipeline has been successfully run and the `best_tuned_model.joblib` is available in the `../training_pipeline/` directory.
2.  Ensure that the feature engineering pipeline has produced `bank-features-selected.csv` (and ideally `selected_feature_names.json`) in the `../feature_pipeline/` directory. This CSV will serve as the "new" data for batch inference.
3.  Have a Python environment with the necessary libraries installed (see `bank_marketing_project/requirements.txt`). Key libraries: pandas, numpy, joblib, scikit-learn.
4.  Open and run the `batch_inference.ipynb` Jupyter notebook from within the `bank_marketing_project/inference_pipeline/` directory.
5.  The notebook will load the model and data, generate predictions, and save them to `predictions.csv`.
