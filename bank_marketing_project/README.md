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
-   Docker (for containerization, see Section 9)

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
    ```

### Running the Pipelines
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

## 7. Automation with GitHub Actions (Conceptual)

### 7.1. Objective of Automation
The primary objective is to automate the execution of the feature, training, and inference pipelines. This automation can be triggered by scheduled events (e.g., daily or weekly) or specific repository events (e.g., a push to the main branch), ensuring that the data is processed, models are retrained, and predictions are generated consistently and without manual intervention.

### 7.2. Tools
-   **GitHub Actions**: For defining and running the automated workflows.

### 7.3. Conceptual Workflow Files
We would define separate YAML files in the `.github/workflows/` directory for each pipeline.

#### 7.3.1. `feature_pipeline.yml`
-   **Trigger**:
    -   Scheduled (e.g., daily or weekly via `on: schedule:`).
    -   On push to the `main` branch, specifically changes within the `feature_pipeline/` directory or related data source configurations.
-   **Jobs**:
    1.  **Setup Environment**:
        -   Checkout the repository code.
        -   Set up a Python environment (e.g., using `actions/setup-python@v4`).
        -   Install dependencies from `requirements.txt`.
    2.  **Run Feature Engineering Notebook**:
        -   Execute the `feature_pipeline/feature_engineering.ipynb` notebook. This can be done using `papermill` to parameterize the notebook if needed, or by converting the notebook to a Python script (`.py`) and running it directly.
    3.  **Persist Outputs**:
        -   The primary outputs (`bank-features-selected.csv`, `selected_feature_names.json`, `bank-full-processed.csv`) need to be persisted.
        -   **Option 1 (Simple)**: Commit the generated files back to the Git repository. Requires `GITHUB_TOKEN` with write permissions.
        -   **Option 2 (Advanced/Recommended)**: Upload the artifacts to a dedicated feature store (e.g., Hopsworks), a cloud storage solution (e.g., AWS S3, Azure Blob Storage, Google Cloud Storage), or a versioned data store. This is more robust for larger datasets and better MLOps practice.
    4.  **Trigger Downstream (Optional)**:
        -   Could be configured to trigger the `inference_pipeline.yml` if new features are generated and predictions are needed immediately.
        -   Could also trigger the `training_pipeline.yml` if the feature changes are significant enough to warrant retraining (this logic would be more complex).

#### 7.3.2. `training_pipeline.yml`
-   **Trigger**:
    -   Scheduled (e.g., weekly or monthly).
    -   Manually triggered using `workflow_dispatch` for on-demand retraining.
    -   Potentially triggered after significant updates from the feature pipeline.
-   **Jobs**:
    1.  **Setup Environment**:
        -   Checkout repository code.
        -   Set up Python environment and install dependencies.
    2.  **Configure MLflow**:
        -   If using a remote MLflow tracking server, configure environment variables for `MLFLOW_TRACKING_URI` and any necessary authentication tokens (e.g., `DATABRICKS_HOST`, `DATABRICKS_TOKEN` if using Databricks-hosted MLflow). These should be stored as GitHub Secrets.
    3.  **Run Model Training Notebook**:
        -   Execute the `training_pipeline/model_training.ipynb` notebook (again, using `papermill` or by converting to a script).
    4.  **Model Registry and Persistence**:
        -   The notebook is already designed to log the model to MLflow. Ensure the MLflow run correctly registers the model in the MLflow Model Registry, possibly promoting it to "Staging" or "Production" based on evaluation criteria (this might involve additional scripting or manual steps initially).
        -   The `best_tuned_model.joblib` file could also be versioned using Git LFS and committed to the repository, or uploaded to a dedicated model store (like S3, an MLflow artifact store, or a dedicated model registry service).

#### 7.3.3. `inference_pipeline.yml`
-   **Trigger**:
    -   Scheduled (e.g., daily, typically after the feature pipeline is expected to complete).
    -   On registration of a new "Production" model in the MLflow Model Registry (requires webhook setup or a polling mechanism, which is more advanced).
    -   Manually triggered via `workflow_dispatch`.
-   **Jobs**:
    1.  **Setup Environment**:
        -   Checkout repository code.
        -   Set up Python environment and install dependencies.
    2.  **Retrieve Data and Model**:
        -   Download the latest `bank-features-selected.csv`. This could be from the Git repository (if committed by `feature_pipeline.yml`) or from the feature store/cloud storage.
        -   Download/load the latest "Production" version of the trained model from the MLflow Model Registry (using MLflow client APIs) or other model store.
    3.  **Run Batch Inference Notebook**:
        -   Execute the `inference_pipeline/batch_inference.ipynb` notebook (using `papermill` or by converting to a script).
    4.  **Save Predictions**:
        -   The `predictions.csv` file needs to be saved.
        -   **Option 1 (Simple)**: Commit `predictions.csv` back to the repository.
        -   **Option 2 (Advanced/Recommended)**: Upload the predictions to a database, data warehouse, S3 bucket, or another system where downstream services or business intelligence tools can consume them.

### 7.4. Secrets Management
-   **`GITHUB_TOKEN`**: Automatically available in GitHub Actions. It's used for actions like checking out code. If workflows need to commit files back to the repository, the token might need `contents: write` permissions (configurable in the workflow file or repository settings).
-   **Cloud Credentials**: Any credentials for accessing external services like a feature store (Hopsworks), MLflow tracking server (if remote and secured), cloud storage (AWS S3, Azure Blob, GCS), or databases should be stored as encrypted secrets in the GitHub repository settings (`Settings -> Secrets and variables -> Actions`). These secrets are then accessed in the workflow files as environment variables (e.g., `secrets.AWS_ACCESS_KEY_ID`).

### 7.5. Converting Notebooks for Automation
While Jupyter notebooks are excellent for development and exploration, they are not always ideal for robust, unattended automation. For CI/CD pipelines:
-   **Convert to Python Scripts (`.py`)**: Notebooks can be converted to Python scripts using `jupyter nbconvert --to script my_notebook.ipynb`. This makes them easier to execute, test, and debug in an automated environment.
-   **Use `papermill`**: `papermill` allows you to execute notebooks programmatically, parameterize them (e.g., pass different input/output paths or configuration settings), and save the executed notebook with outputs for inspection. This can be a good middle ground, preserving the notebook format while enabling automation.
-   **Modularize Code**: Refactor common functions or complex logic from notebooks into Python modules (`.py` files) that can be imported and tested independently, then called from simpler script or notebook wrappers in the automation pipeline.

This conceptual outline provides a roadmap for establishing a CI/CD system for the MLOps pipelines using GitHub Actions.

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

## 9. Orchestration with Ploomber (Conceptual)

### 9.1. Introduction to Ploomber
Ploomber is a pipeline orchestration tool that helps in developing, building, and deploying data pipelines. It allows users to define their pipeline as a series of tasks (which can be notebooks, Python scripts, SQL scripts, etc.) and manages the dependencies and execution flow between them. This is particularly useful for creating reproducible and maintainable machine learning workflows.

### 9.2. `pipeline.yaml` Structure (Conceptual)
Ploomber uses a `pipeline.yaml` file at the root of the project to define the Directed Acyclic Graph (DAG) of tasks. Each task specifies its source code and the products it generates.

Conceptually, for this project, the `pipeline.yaml` would define tasks like:

-   **Feature Engineering Task (`feature_eng`)**:
    -   **Source**: `feature_pipeline/feature_engineering.ipynb` (or a converted `feature_engineering.py` script).
    -   **Product**:
        -   `feature_pipeline/bank-features-selected.csv`
        -   `feature_pipeline/selected_feature_names.json`
        -   (Optionally, `feature_pipeline/bank-full-processed.csv` if needed for other analyses).

-   **Model Training Task (`train_model`)**:
    -   **Source**: `training_pipeline/model_training.ipynb` (or `model_training.py`).
    -   **Upstream**: Depends on the `feature_eng` task (specifically, on the creation of `bank-features-selected.csv`).
    -   **Product**:
        -   `training_pipeline/best_tuned_model.joblib`
        -   (MLflow experiment tracking is a side effect of this task, not a direct Ploomber product, but crucial).

-   **Batch Inference Task (`batch_predict`)** (Optional, as Ploomber primarily orchestrates training; inference might be separate or triggered by other means):
    -   **Source**: `inference_pipeline/batch_inference.ipynb` (or `batch_inference.py`).
    -   **Upstream**:
        -   Depends on `train_model` for `best_tuned_model.joblib`.
        -   Depends on `feature_eng` for `bank-features-selected.csv` (acting as new batch data in this project's example). In a real-world scenario, the input data for inference would likely come from a different source or be a parameter.
    -   **Product**: `inference_pipeline/predictions.csv`.

-   **(Advanced) Model Deployment Task (`deploy_model`)**:
    -   This is a conceptual task that could be triggered after `train_model`.
    -   **Source**: Could be a Python script or a shell script.
    -   **Upstream**: Depends on `train_model`.
    -   **Actions (not direct Ploomber products but effects of the task)**:
        -   Build the Docker image using the `Dockerfile` (e.g., by running `docker build ...`).
        -   Push the built image to a container registry (e.g., Docker Hub, AWS ECR, Google GCR).
        -   Deploy or update the service running the FastAPI application (e.g., on Kubernetes, or by restarting a Docker container with the new image).

### 9.3. Benefits of Using Ploomber
-   **Reproducibility**: Ensures that pipeline runs are consistent and can be exactly reproduced.
-   **Clear Dependency Management**: Explicitly defines the relationships between tasks, making the pipeline structure easy to understand and maintain.
-   **Incremental Builds**: Ploomber can skip tasks whose inputs haven't changed, saving computation time.
-   **Easier Development & Testing**: Allows for developing and testing individual pipeline components (tasks) in isolation.
-   **Parameterization**: Supports parameterizing pipeline runs, allowing for different configurations without changing the code (e.g., different feature selection parameters, model hyperparameters).
-   **Integration with CI/CD**: Ploomber pipelines can be easily integrated into CI/CD systems like GitHub Actions for automated execution, testing, and deployment.
-   **Interactive Development**: `ploomber plot` can visualize the pipeline, and `ploomber build --interactive` can help debug.

### 9.4. FastAPI Serving
It's important to note that Ploomber's primary role here would be to orchestrate the *creation, training, and versioning of the model artifact* (`best_tuned_model.joblib`) that is then served by the FastAPI application. The FastAPI application (defined in `app/main.py` and containerized using the `Dockerfile`) is the actual model deployment mechanism for handling online/real-time prediction requests. Ploomber ensures that the model served by FastAPI is the result of a well-defined, reproducible, and version-controlled pipeline.

## 10. Future Work / Next Steps
-   **Full Automation with CI/CD**: Implement GitHub Actions or a similar CI/CD tool to automate the execution of the pipelines upon code changes or on a schedule. (See conceptual outline in Section 8).
-   **Feature Store Integration**: Utilize a feature store like Hopsworks for centralized feature management, versioning, and serving, ensuring consistency between training and inference.
-   **Containerization and API Deployment**:
    -   Containerize the inference pipeline using Docker. (See Section 9).
    -   Deploy the model as a REST API service using FastAPI for real-time (or micro-batch) predictions.
-   **Workflow Orchestration**: Use a workflow orchestrator like Ploomber or Apache Airflow to manage the dependencies and execution of the different pipeline stages. (Ploomber concept covered in Section 10).
-   **Advanced Model Monitoring**: Implement more sophisticated model monitoring for data drift, concept drift, and performance degradation over time.
-   **Scalability**: Explore options for scaling data processing (e.g., using Spark) and model training/inference if the dataset size grows significantly.

---
This README provides an overview of the Bank Marketing Prediction project. For detailed information on each component, please refer to the README files and Jupyter notebooks within the respective pipeline directories.
