# Feature Engineering Pipeline for Bank Marketing Data

## Objective
This pipeline processes the raw bank marketing dataset (`bank-full.csv`) to perform Exploratory Data Analysis (EDA), feature engineering, and feature selection. The goal is to generate a cleaned and optimized dataset ready for training a model to predict term deposit subscriptions.

## Input
- Raw `bank-full.csv` (expected to be downloaded and placed in this directory by the notebook, or already present).

## Processing Steps
The `feature_engineering.ipynb` notebook performs the following key steps:

1.  **Data Loading**:
    -   Downloads the "Bank Marketing Data Set" from the UCI Machine Learning Repository if `bank-full.csv` is not already present.
    -   Loads `bank-full.csv` (semicolon-delimited) into a pandas DataFrame.

2.  **Exploratory Data Analysis (EDA)**:
    -   Initial data inspection (shape, head, info, describe).
    -   Identification of numerical and categorical features.
    -   Visualization of the target variable ('y') distribution.
    -   Analysis of numerical features using histograms and boxplots (vs. target).
    -   Analysis of categorical features using count plots (and vs. target).
    -   Correlation analysis of numerical features using a heatmap.

3.  **Data Preprocessing**:
    -   **Handling 'unknown' values**: Replaces 'unknown' strings with NaN, then imputes these NaNs using the mode of each respective column.
    -   **Encoding Binary Categorical Features**: Converts 'default', 'housing', 'loan', and the target 'y' from 'yes'/'no' to 1/0.
    -   **One-Hot Encoding Other Categorical Features**: Applies `pd.get_dummies` (with `drop_first=True`) to other object-type categorical features.
    -   **Scaling Numerical Features**: Scales all numerical features (excluding the already encoded binary target 'y') using `StandardScaler`.
    -   The fully processed data (before feature selection) is saved to `bank-full-processed.csv`.

4.  **Feature Selection**:
    -   **Mutual Information Gain**: Calculates mutual information scores between each feature and the target variable 'y' using the preprocessed data.
    -   **Select Top K Features**: Selects the top K (K=20, or fewer if total features < 20) features based on the highest mutual information scores.
    -   A bar chart of MI scores for the top features is displayed.

## Output
1.  `bank-full-processed.csv`: Contains the fully preprocessed data (all features after cleaning, encoding, and scaling, along with the target variable). This can be useful for analyses that require all features.
2.  `bank-features-selected.csv`: Contains the final selected features (top K based on mutual information) and the target variable. **This is the primary output intended for the training pipeline.**
3.  `selected_feature_names.json`: A JSON file containing a list of the names of the selected features.

## How to Run
1.  Ensure you have a Python environment with the necessary libraries installed (see `bank_marketing_project/requirements.txt` - though this file might be populated later in the project). Key libraries used are pandas, numpy, scikit-learn, matplotlib, seaborn, and requests.
2.  Open and run the `feature_engineering.ipynb` Jupyter notebook from within the `bank_marketing_project/feature_pipeline/` directory.
3.  The notebook will download the data if not present, perform all processing steps, and save the output files (`bank-features-selected.csv`, `bank-full-processed.csv`, and `selected_feature_names.json`) in this directory.
