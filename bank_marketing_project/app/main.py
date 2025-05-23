import os
import json
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
from typing import List, Dict, Any # Union for Pydantic types if needed

# --- Configuration & Model Loading ---
# Paths are relative to the location of main.py when running inside the Docker container
# The Dockerfile will place these files correctly.
MODEL_DIR = os.getenv("MODEL_DIR", ".") # Default to current dir if not set, but Dockerfile will structure it
MODEL_PATH = os.path.join(MODEL_DIR, "training_pipeline/best_tuned_model.joblib")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_pipeline/selected_feature_names.json")

model = None
selected_features = []

# --- Pydantic Models ---
# Dynamically create the Pydantic model based on selected_features
# This is advanced and might be overly complex for this stage.
# For now, we'll define a more generic input model expecting a list of dictionaries.
class FeatureDict(BaseModel):
    # This will allow any feature name as a key, with float values.
    # In a more robust scenario, you'd explicitly define fields if known and static,
    # or use create_model from pydantic.tools if features are truly dynamic AND need strict validation.
    # For this exercise, a simple Dict[str, float] per instance is practical.
    __root__: Dict[str, float] # Allows arbitrary key-value pairs where keys are strings, values are floats

class PredictionInput(BaseModel):
    instances: conlist(item_type=Dict[str, Any], min_items=1) # List of feature dictionaries

class PredictionOutput(BaseModel):
    predictions: List[Dict[str, Any]]

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Bank Marketing Prediction API",
    description="API to predict term deposit subscriptions using a pre-trained model.",
    version="0.1.0"
)

@app.on_event("startup")
async def load_model_and_features():
    global model, selected_features
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")
    if not os.path.exists(FEATURES_PATH):
        raise RuntimeError(f"Features file not found at {FEATURES_PATH}")

    try:
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise RuntimeError(f"Could not load model: {e}")

    try:
        with open(FEATURES_PATH, 'r') as f:
            selected_features = json.load(f)
        print(f"Selected features loaded successfully from {FEATURES_PATH}. ({len(selected_features)} features)")
        if not selected_features:
            raise ValueError("Selected features list is empty.")
    except Exception as e:
        print(f"Error loading features list: {e}")
        raise RuntimeError(f"Could not load features list: {e}")

# --- API Endpoints ---
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Bank Marketing Prediction API. Use the /predict endpoint to make predictions."}

@app.post("/predict/", response_model=PredictionOutput)
async def predict(payload: PredictionInput):
    global model, selected_features

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check server logs.")
    if not selected_features:
        raise HTTPException(status_code=503, detail="Feature list not loaded or empty. Please check server logs.")

    input_data_list = payload.instances
    
    try:
        # Convert list of dicts to DataFrame
        # Ensure all required features are present and in the correct order
        df_list = []
        for i, instance in enumerate(input_data_list):
            # Check for missing features
            missing_instance_features = set(selected_features) - set(instance.keys())
            if missing_instance_features:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Missing features in instance {i}: {missing_instance_features}. Expected: {selected_features}"
                )
            
            # Check for extra features (optional, but good for strictness)
            extra_instance_features = set(instance.keys()) - set(selected_features)
            if extra_instance_features:
                 # For this exercise, we'll ignore extra features if all selected_features are present.
                 # In a stricter setting, you might raise an error or log a warning.
                 # print(f"Warning: Extra features in instance {i} will be ignored: {extra_instance_features}")
                 pass

            # Ensure order and select only the required features
            ordered_instance = {feature: instance.get(feature) for feature in selected_features}
            df_list.append(ordered_instance)

        inference_df = pd.DataFrame(df_list, columns=selected_features)
        
        # Data type conversion (model expects numerical, mostly float due to scaling)
        # This assumes features in selected_features are all meant to be numeric.
        # If not, more sophisticated type handling is needed here based on feature metadata.
        try:
            inference_df = inference_df.astype(float)
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Data type conversion error. Ensure all feature values can be converted to float. Error: {e}"
            )

    except HTTPException: # Re-raise if it's already an HTTPException
        raise
    except Exception as e:
        print(f"Error processing input data: {e}") # Log for server
        raise HTTPException(status_code=400, detail=f"Error processing input data: {str(e)}")

    try:
        # Make predictions
        pred_labels = model.predict(inference_df)
        
        pred_probas = None
        if hasattr(model, "predict_proba"):
            pred_probas = model.predict_proba(inference_df)[:, 1] # Probability of class '1' (yes)
        
        results = []
        for i in range(len(pred_labels)):
            result_item = {"predicted_label": int(pred_labels[i])}
            if pred_probas is not None:
                result_item["probability_yes"] = float(pred_probas[i])
            results.append(result_item)
            
        return {"predictions": results}

    except Exception as e:
        print(f"Error during model prediction: {e}") # Log for server
        # Check for common sklearn errors e.g. feature names mismatch if not caught earlier
        if "X has a different number of features than required by the model" in str(e) or "feature_names mismatch" in str(e):
             detail_msg = (
                f"Feature mismatch during prediction. Model expected {len(selected_features)} features. "
                f"Provided data has {inference_df.shape[1]} features after processing. "
                f"Model features (first 5): {selected_features[:5]}... "
                f"Provided features (first 5): {inference_df.columns.tolist()[:5]}..."
            )
             raise HTTPException(status_code=400, detail=detail_msg)

        raise HTTPException(status_code=500, detail=f"Error during model prediction: {str(e)}")

# --- Main execution for Uvicorn (if running script directly) ---
# This part is usually not included if Docker CMD directly calls uvicorn
# but can be useful for local testing.
# if __name__ == \"__main__\":
#     import uvicorn
#     uvicorn.run(app, host=\"0.0.0.0\", port=8000)
