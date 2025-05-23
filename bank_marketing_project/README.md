# Bank Marketing Prediction Project

## 1. Project Objective
The primary goal of this project is to analyze the Bank Marketing dataset to predict whether a customer will subscribe to a term deposit. This machine learning task is structured as a 3-pipeline system (feature, training, and inference) designed for batch predictions. The project aims to demonstrate an end-to-end MLOps workflow, including data processing, model training, experiment tracking with MLflow, and batch inference.

## 2. Dataset Description
-   **Name**: Bank Marketing dataset
-   **Source**: UCI Machine Learning Repository
-   **Link**: [`https://archive.ics.uci.edu/dataset/222/bank+marketing`](https://archive.ics.uci.edu/dataset/222/bank+marketing)
-   **Details**:
    -   The project utilizes the `bank-full.csv` file, which contains a comprehensive set of examples.
    -   The dataset includes 17 input variables covering client data (e.g., age, job, marital status, education), campaign-related data (e.g., contact type, month, day of week, duration of call), and previous campaign outcomes (e.g., pdays, previous, poutcome).
    -   The target variable is 'y', indicating whether the client subscribed to a term deposit (binary: 'yes' or 'no').

## 3. Project Structure (3-Pipeline Architecture)
The project is organized into three main pipelines, each residing in its own directory:

-   **Feature Pipeline (`feature_pipeline/`)**:
    -   This pipeline is responsible for loading the raw data (`bank-full.csv`).
    -   It performs extensive Exploratory Data Analysis (EDA) to understand data characteristics and relationships.
    -   Preprocessing steps include handling 'unknown' values (imputation with mode), encoding categorical features (binary and one-hot encoding), and scaling numerical features using `StandardScaler`.
    -   Feature selection is performed using Mutual Information Gain to identify the most relevant features for the prediction task.
    -   **Output**: `bank-features-selected.csv` (containing selected features and the target) and `selected_feature_names.json` (list of selected feature names).

-   **Training Pipeline (`training_pipeline/`)**:
    -   Loads the selected features from `bank-features-selected.csv`.
    -   Optionally uses `LazyPredict` for a broad comparison of various classification models to get a performance baseline.
    -   Performs hyperparameter tuning (using `GridSearchCV`) on a chosen model (e.g., RandomForestClassifier, GradientBoostingClassifier). The choice of model and focus on F1-score/ROC AUC for tuning is justified by the class imbalance in the target variable.
    -   Logs experiments, model parameters, performance metrics, and model artifacts (including the trained model and feature importance plots) using MLflow.
    -   **Output**: `best_tuned_model.joblib` (the serialized best performing tuned model).

-   **Inference Pipeline (`inference_pipeline/`)**:
    -   Loads the `best_tuned_model.joblib` from the training pipeline.
    -   Loads new batch feature data (which should have undergone the same preprocessing as the training data; for this project, `bank-features-selected.csv` is reused as an example of batch input).
    -   Ensures consistency between the features in the input data and the features the model was trained on.
    -   Generates predictions (class labels and probabilities).
    -   **Output**: `predictions.csv` (containing the input features along with their predicted labels and probabilities).

## 4. Key Decisions in Modeling
-   **Target Variable**: 'y', a binary variable indicating subscription to a term deposit ('yes'/'no'), encoded as 1/0.
-   **Evaluation Metrics**: Due to significant class imbalance (more 'no' than 'yes'), the primary focus for model evaluation and tuning is on **F1-score** (especially for the positive class 'yes') and **ROC AUC**. Accuracy, precision, and recall are also reported for a comprehensive understanding.
-   **Handling Imbalance**: Strategies such as using `class_weight='balanced'` in scikit-learn models (e.g., RandomForestClassifier, LogisticRegression) were employed during training to give more weight to the minority class.
-   **Feature Selection**: Mutual Information Gain was used to select the top K (K=20) features from the preprocessed dataset, aiming to improve model performance and reduce complexity.
-   **Model Selection**: `LazyPredict` provided an initial broad comparison. A promising, robust classifier (e.g., RandomForestClassifier or GradientBoostingClassifier) was then chosen for detailed hyperparameter tuning and final evaluation.

## 5. Main Results and Metrics Summary
The final tuned model (e.g., a RandomForestClassifier or GradientBoostingClassifier, depending on the tuning outcome) demonstrates robust performance in predicting term deposit subscriptions. For instance, a typical outcome might be:

*"The final tuned RandomForestClassifier achieved an F1-score (positive class 'yes') of approximately 0.50-0.60 and a ROC AUC of approximately 0.85-0.90 on the held-out test set. This indicates a good balance between precision and recall for identifying potential subscribers, and a strong ability to distinguish between the two classes."* (Actual values depend on the final run of the training notebook).

-   Detailed metrics, parameters, and model artifacts (including feature importance plots and the model itself) for each training run are tracked in **MLflow** under the experiment name: **"Bank Marketing Predictions"**.
-   Users are encouraged to refer to the individual pipeline README files and the Jupyter notebooks for more detailed execution logs, specific outputs, and visualizations.

## 6. Setup and How to Run

### Prerequisites
-   Python 3.7+
-   pip (Python package installer)
-   Git
-   Docker (for containerization, see Section 8)

### Installation
1.  Clone the repository:
    ```bash
    git clone <repository_url> 
    # Replace <repository_url> with the actual URL of this project's repository
    ```
2.  Navigate to the project directory:
    ```bash
    cd bank_marketing_project
    ```
3.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt 
    # Ploomber is included in requirements.txt
    ```

### Running the Pipelines Manually
The notebooks should ideally be run in the following sequence:

1.  **Feature Pipeline**:
    -   Navigate to the feature pipeline directory: `cd feature_pipeline`
    -   Run the Jupyter Notebook: `jupyter notebook feature_engineering.ipynb` (or use JupyterLab)
    -   This will generate `bank-features-selected.csv` and `selected_feature_names.json` in the `feature_pipeline` directory, and `bank-full-processed.csv`.

2.  **Training Pipeline**:
    -   Navigate to the training pipeline directory: `cd ../training_pipeline` (if you were in `feature_pipeline`)
    -   Ensure MLflow is accessible. You can start the MLflow UI to view experiments by running:
        ```bash
        mlflow ui
        ```
        (This typically starts a server at `http://localhost:5000`)
    -   Run the Jupyter Notebook: `jupyter notebook model_training.ipynb`
    -   This will train the model, log experiments to MLflow, and save `best_tuned_model.joblib` in the `training_pipeline` directory.

3.  **Inference Pipeline**:
    -   Navigate to the inference pipeline directory: `cd ../inference_pipeline` (if you were in `training_pipeline`)
    -   Run the Jupyter Notebook: `jupyter notebook batch_inference.ipynb`
    -   This will load the trained model and generate `predictions.csv` in the `inference_pipeline` directory using `bank-features-selected.csv` as input.

For automated execution, see the "Automation with GitHub Actions" section below. For pipeline orchestration, see "Orchestration with Ploomber".

## 7. Automation with GitHub Actions

### 7.1. Objective of Automation
This project implements GitHub Actions to automate the execution of the feature engineering and model training pipelines. This automation can be triggered manually or by pushes to the main branch, ensuring that data artifacts and models are consistently regenerated.

### 7.2. Implemented Workflows
The following workflows are defined in the `.github/workflows/` directory:

-   **Feature Engineering Pipeline (`.github/workflows/feature_pipeline.yml`)**:
    -   **Purpose**: Automates the execution of the `feature_pipeline/feature_engineering.ipynb` notebook using `papermill`. It generates key data artifacts: `bank-features-selected.csv` (selected features for model training) and `selected_feature_names.json` (list of these feature names). The executed notebook (`executed_feature_engineering.ipynb`) is also saved. These generated files are then committed back to the repository.
    -   **Triggers**:
        -   Manually via `workflow_dispatch`.
        -   On push to the `main` branch if changes are detected in `feature_pipeline/**`, `requirements.txt`, or the workflow file itself (`.github/workflows/feature_pipeline.yml`).

-   **Model Training Pipeline (`.github/workflows/training_pipeline.yml`)**:
    -   **Purpose**: Automates the execution of the `training_pipeline/model_training.ipynb` notebook using `papermill`. This pipeline trains the model, generates the `best_tuned_model.joblib` artifact, and logs experiment data (parameters, metrics, artifacts) to a local MLflow tracking instance (saved in the `mlruns/` directory). The `best_tuned_model.joblib`, the executed notebook (`executed_model_training.ipynb`), and the `mlruns/` directory are then committed back to the repository.
    -   **Triggers**:
        -   Manually via `workflow_dispatch`.
        -   On push to the `main` branch if changes are detected in `training_pipeline/**`, `requirements.txt`, the workflow file itself (`.github/workflows/training_pipeline.yml`), or importantly, if the key input `feature_pipeline/bank-features-selected.csv` or `feature_pipeline/selected_feature_names.json` are updated (indicating new features are available for training).

### 7.3. Usage and Monitoring
-   **Monitoring**: The status and logs of these automated workflows can be monitored from the "Actions" tab of the GitHub repository.
-   **Manual Triggers**: Both workflows can be manually triggered from the "Actions" tab by selecting the desired workflow and clicking "Run workflow".
-   **Outputs**: When the workflows run, they commit their outputs (generated data files, model artifacts, executed notebooks, and the `mlruns` directory) back to the Git repository. This provides versioning for these artifacts directly within the repository.
-   **Notebook Execution**: `papermill` is used to execute the Jupyter notebooks in a non-interactive way, allowing for parameterization (though not heavily used in the current setup) and saving the executed notebook with its outputs for inspection.
-   **MLflow Data**: For simplicity, the `training_pipeline.yml` workflow commits the `mlruns` directory (containing local MLflow experiment data) back to the repository. For more robust, collaborative, or production environments, configuring a remote MLflow tracking server is highly recommended.

## 8. Model Containerization with Docker

### 8.1. Purpose
Containerization with Docker packages the FastAPI application along with the trained model, dependencies, and necessary configurations into a portable, self-contained unit. This ensures consistency across different environments and simplifies deployment.

### 8.2. Dockerfile
A `Dockerfile` is created in the root of the `bank_marketing_project` directory. Key steps in the Dockerfile include:
-   Starting from a Python base image (e.g., `python:3.9-slim`).
-   Setting the working directory to `/app`.
-   Copying `requirements.txt` and installing dependencies.
-   Copying the FastAPI application code (from `bank_marketing_project/app/`) into `/app/app/` within the image.
-   Copying the trained model (`training_pipeline/best_tuned_model.joblib`) to `/app/training_pipeline/` within the image.
-   Copying the selected features list (`feature_pipeline/selected_feature_names.json`) to `/app/feature_pipeline/` within the image.
-   Exposing port 8000.
-   Defining the `CMD` to run the FastAPI application using `uvicorn app.main:app --host 0.0.0.0 --port 8000`.

### 8.3. Build the Docker Image
Navigate to the `bank_marketing_project` directory in your terminal and run:
```bash
docker build -t bank-marketing-service .
```
This command builds a Docker image tagged as `bank-marketing-service`.

### 8.4. Run the Docker Container
Once the image is built, run it as a container:
```bash
docker run -d -p 8001:8000 bank-marketing-service
```
-   `-d`: Runs the container in detached mode (in the background).
-   `-p 8001:8000`: Maps port 8001 on your host machine to port 8000 inside the container (where the FastAPI app is running). You can use other host ports if 8001 is occupied, e.g., `-p 80:8000` for standard HTTP port.

### 8.5. Test the API Endpoint
After the container is running, you can test the `/predict/` endpoint.

#### Using `curl`
You'll need a sample JSON payload. First, find out some actual feature names from `bank_marketing_project/feature_pipeline/selected_feature_names.json`. Let's assume the first few are "duration", "pdays", "euribor3m", "age", "campaign". Replace these with actual features if they differ.

**Example `curl` command:**
```bash
curl -X POST "http://localhost:8001/predict/" \
-H "Content-Type: application/json" \
-d '{
  "instances": [
    {
      "duration": 300.0,
      "pdays": 999.0,
      "euribor3m": 4.857,
      "age": 35.0,
      "campaign": 1.0,
      "nr.employed": 5228.1,
      "cons.price.idx": 93.994,
      "cons.conf.idx": -36.4,
      "poutcome_success": 0.0, 
      "month_may": 1.0, 
      "contact_cellular": 1.0,
      "job_admin.": 1.0,
      "default_no": 1.0,
      "housing_yes": 1.0,
      "loan_no": 1.0
    },
    {
      "duration": 150.0,
      "pdays": 10.0,
      "euribor3m": 1.2,
      "age": 45.0,
      "campaign": 2.0,
      "nr.employed": 5099.1,
      "cons.price.idx": 92.893,
      "cons.conf.idx": -46.2,
      "poutcome_success": 1.0,
      "month_may": 0.0,
      "contact_cellular": 1.0,
      "job_admin.": 0.0,
      "default_no": 1.0,
      "housing_yes": 0.0,
      "loan_no": 1.0
    }
  ]
}'
```
*Note: The example payload above uses hypothetical feature names and values. You **must** replace them with actual feature names from your `selected_feature_names.json` and provide plausible values for each. Ensure all features listed in `selected_feature_names.json` are included in each instance dictionary in the payload.*

The API should return a JSON response with predictions, for example:
```json
{
  "predictions": [
    {"predicted_label": 0, "probability_yes": 0.123},
    {"predicted_label": 1, "probability_yes": 0.789}
  ]
}
```

### 8.6. Upload to Docker Hub (Next Step)
Once the image is tested and working, you can upload it to Docker Hub (or another container registry):
1.  Log in to Docker Hub: `docker login`
2.  Tag your image: `docker tag bank-marketing-service yourusername/bank-marketing-service:latest` (replace `yourusername`)
3.  Push the image: `docker push yourusername/bank-marketing-service:latest`

## 9. Orchestration with Ploomber

### 9.1. `pipeline.yaml` Definition
Ploomber is used to define and orchestrate the sequence of tasks in this project. The orchestration logic is captured in the `pipeline.yaml` file at the root of the `bank_marketing_project` directory:

```yaml
# Ploomber pipeline example for orchestrating the bank marketing project notebooks

# Optional: Configure meta settings if needed, like default paths for products
# meta:
#   source_loader:
#     kind: NotebookSourceLoader
#   product_default_class: File # Could be File or SQLRelation etc.

tasks:
  - source: feature_pipeline/feature_engineering.ipynb
    name: feature-engineering # Optional, Ploomber can infer from source name
    product:
      # Ploomber typically expects one main product per task for chaining,
      # but a task can produce multiple files.
      # We list them here for clarity and potential checking.
      # The primary output that might be used by a downstream task is 'data'.
      data: feature_pipeline/bank-features-selected.csv
      selected_features_json: feature_pipeline/selected_feature_names.json
      executed_notebook: feature_pipeline/executed_feature_engineering_ploomber.ipynb # Ploomber can save executed notebook

  - source: training_pipeline/model_training.ipynb
    name: model-training
    # Ploomber can also manage MLflow logging implicitly if configured,
    # or the notebook handles it as it does now.
    product:
      model: training_pipeline/best_tuned_model.joblib
      executed_notebook: training_pipeline/executed_model_training_ploomber.ipynb
    upstream:
      # This task depends on the features generated by the feature-engineering task.
      # Ploomber will pass the 'data' product from 'feature-engineering'
      # as an 'upstream' variable to this notebook if the notebook expects it.
      # The notebook 'model_training.ipynb' needs to be able to accept 'upstream["feature-engineering"]["data"]'
      # or simply know the path to 'bank-features-selected.csv'.
      # For simplicity, this example assumes the notebook knows the path.
      # If parameter passing is desired, the notebook needs a 'parameters' cell
      # and Ploomber injects 'upstream' and 'product' variables.
      - feature-engineering # Depends on the 'feature-engineering' task by name

  # Optional: Batch Inference Task
  # This task would typically run on new data, not directly chained here unless
  # the feature_engineering task is parameterized to run on different raw data inputs.
  # For now, let's include it to show how it would look.
  - source: inference_pipeline/batch_inference.ipynb
    name: batch-inference
    product:
      predictions: inference_pipeline/predictions_ploomber.csv
      executed_notebook: inference_pipeline/executed_batch_inference_ploomber.ipynb
    upstream:
      # Depends on the model from 'model-training' and potentially new features.
      # The notebook 'batch_inference.ipynb' would need to be aware of the upstream model path.
      - model-training
      # If 'feature-engineering' produced a general set of features for inference
      # (not just training split), it could be an upstream dependency too.
      # For this example, we assume the inference notebook loads features independently
      # or uses a version of features aligned with the trained model.

# To make notebooks Ploomber-aware for parameter injection (optional but powerful):
# 1. Add a cell with the tag 'parameters' to your input notebooks.
# 2. Ploomber will inject 'upstream' and 'product' variables into that cell if they exist.
# Example 'parameters' cell in model_training.ipynb:
# upstream = None # Will be injected by Ploomber
# product = None # Will be injected by Ploomber
# input_features_path = upstream['feature-engineering']['data'] if upstream else 'feature_pipeline/bank-features-selected.csv'
# output_model_path = product['model'] if product else 'training_pipeline/best_tuned_model.joblib'

# Without parameter injection, notebooks must rely on pre-defined relative paths,
# which is how they are currently set up. This pipeline.yaml will still help in
# orchestrating the order of execution.
```

### 9.2. Usage
Ploomber helps manage the execution order and dependencies of the notebooks.

-   **Installation**: Ploomber should already be included in `requirements.txt`. If installing manually:
    ```bash
    pip install ploomber
    ```
-   **Running the Pipeline**: To execute the entire pipeline as defined in `pipeline.yaml`:
    ```bash
    ploomber build
    ```
-   **Visualizing the Pipeline**: To generate a plot of the pipeline's Directed Acyclic Graph (DAG):
    ```bash
    ploomber plot
    ```
    This will create a `pipeline.html` (or `.png` if graphviz is installed) file showing the task dependencies.
-   **Checking Task Status**: To see the current status of tasks (e.g., if they need to be run):
    ```bash
    ploomber status
    ```
-   **Forcing Task Re-execution**: To force a specific task and its downstream dependencies to re-run, even if Ploomber thinks they are up-to-date:
    ```bash
    ploomber build --force <task-name>
    ```
    For example:
    ```bash
    ploomber build --force feature-engineering
    ```
-   **Parameterization Note**: The provided `pipeline.yaml` orchestrates the existing notebooks based on their current fixed-path dependencies. For more dynamic execution and parameter passing directly from Ploomber (e.g., passing `upstream['feature-engineering']['data']` to the training notebook), the notebooks can be modified by adding a cell tagged 'parameters'. Comments within `pipeline.yaml` provide hints on this.

## 10. Future Work / Next Steps
-   **CI/CD Enhancements**:
    -   Implement an automated inference pipeline workflow in GitHub Actions.
    -   Explore strategies for more advanced artifact management (e.g., using a dedicated feature store like Hopsworks, or cloud storage for data/models instead of committing directly to Git, especially for larger artifacts).
    -   Integrate a remote MLflow tracking server for more robust experiment management.
-   **Feature Store Integration**: Utilize a feature store like Hopsworks for centralized feature management, versioning, and serving, ensuring consistency between training and inference.
-   **API Deployment Enhancements**:
    -   Further refine the FastAPI application with more comprehensive error handling, logging, and potentially input validation based on dynamic feature types if necessary.
    -   Explore deployment to managed services (e.g., AWS SageMaker, Azure ML, Google Vertex AI, or Kubernetes).
-   **Workflow Orchestration**: Fully implement Ploomber or Apache Airflow to manage the dependencies and execution of the different pipeline stages. (Ploomber integration initiated in Section 9).
-   **Advanced Model Monitoring**: Implement sophisticated model monitoring for data drift, concept drift, and performance degradation over time using tools like Evidently AI or Grafana.
-   **Scalability**: Explore options for scaling data processing (e.g., using Spark) and model training/inference if the dataset size grows significantly.

---
This README provides an overview of the Bank Marketing Prediction project. For detailed information on each component, please refer to the README files and Jupyter notebooks within the respective pipeline directories.
