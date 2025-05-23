import os
import json
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
from typing import List, Dict, Any # Union para tipos Pydantic si es necesario

# --- Configuración y Carga del Modelo ---
# Las rutas son relativas a la ubicación de main.py cuando se ejecuta dentro del contenedor Docker
# El Dockerfile colocará estos archivos correctamente.
MODEL_DIR = os.getenv("MODEL_DIR", ".") # Valor por defecto al directorio actual si no está configurado, pero Dockerfile lo estructurará
MODEL_PATH = os.path.join(MODEL_DIR, "training_pipeline/best_tuned_model.joblib")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_pipeline/selected_feature_names.json")

model = None
selected_features = []

# --- Modelos Pydantic ---
# Crear dinámicamente el modelo Pydantic basado en selected_features
# Esto es avanzado y podría ser demasiado complejo para esta etapa.
# Por ahora, definiremos un modelo de entrada más genérico que espera una lista de diccionarios.
class FeatureDict(BaseModel):
    # Esto permitirá cualquier nombre de característica como clave, con valores flotantes.
    # En un escenario más robusto, definirías explícitamente los campos si son conocidos y estáticos,
    # o usarías create_model de pydantic.tools si las características son verdaderamente dinámicas Y necesitan validación estricta.
    # Para este ejercicio, un simple Dict[str, float] por instancia es práctico.
    __root__: Dict[str, float] # Permite pares clave-valor arbitrarios donde las claves son cadenas y los valores son flotantes

class PredictionInput(BaseModel):
    instances: conlist(item_type=Dict[str, Any], min_items=1) # Lista de diccionarios de características

class PredictionOutput(BaseModel):
    predictions: List[Dict[str, Any]]

# --- Inicialización de la App FastAPI ---
app = FastAPI(
    title="API de Predicción de Marketing Bancario",
    description="API para predecir suscripciones a depósitos a plazo utilizando un modelo preentrenado.",
    version="0.1.0"
)

@app.on_event("startup")
async def load_model_and_features():
    """
    Carga el modelo entrenado y la lista de características seleccionadas durante el inicio de la aplicación.
    Estos artefactos son esenciales para que el endpoint de predicción funcione correctamente.
    Lanza RuntimeError si los archivos necesarios no se encuentran o no se pueden cargar.
    """
    global model, selected_features
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Archivo de modelo no encontrado en {MODEL_PATH}")
    if not os.path.exists(FEATURES_PATH):
        raise RuntimeError(f"Archivo de características no encontrado en {FEATURES_PATH}")

    try:
        model = joblib.load(MODEL_PATH)
        print(f"Modelo cargado exitosamente desde {MODEL_PATH}")
    except Exception as e:
        print(f"Error cargando el modelo: {e}")
        raise RuntimeError(f"No se pudo cargar el modelo: {e}")

    try:
        with open(FEATURES_PATH, 'r') as f:
            selected_features = json.load(f)
        print(f"Características seleccionadas cargadas exitosamente desde {FEATURES_PATH}. ({len(selected_features)} características)")
        if not selected_features:
            raise ValueError("La lista de características seleccionadas está vacía.")
    except Exception as e:
        print(f"Error cargando la lista de características: {e}")
        raise RuntimeError(f"No se pudo cargar la lista de características: {e}")

# --- Endpoints de la API ---
@app.get("/")
async def read_root():
    """
    Endpoint raíz para verificar que la API está en funcionamiento.
    """
    return {"message": "Bienvenido a la API de Predicción de Marketing Bancario. Usa el endpoint /predict para realizar predicciones."}

@app.post("/predict/", response_model=PredictionOutput)
async def predict(payload: PredictionInput):
    """
    Endpoint para realizar predicciones.
    Recibe una lista de instancias (diccionarios de características),
    realiza predicciones utilizando el modelo cargado y devuelve
    las etiquetas predichas y las probabilidades.
    """
    global model, selected_features

    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado. Por favor, revisa los logs del servidor.")
    if not selected_features:
        raise HTTPException(status_code=503, detail="Lista de características no cargada o vacía. Por favor, revisa los logs del servidor.")

    input_data_list = payload.instances
    
    try:
        # Convertir lista de diccionarios a DataFrame
        # Asegurar que todas las características requeridas estén presentes y en el orden correcto
        df_list = []
        for i, instance in enumerate(input_data_list):
            # Verificar características faltantes
            missing_instance_features = set(selected_features) - set(instance.keys())
            if missing_instance_features:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Características faltantes en la instancia {i}: {missing_instance_features}. Esperadas: {selected_features}"
                )
            
            # Verificar características adicionales (opcional, pero bueno para la rigurosidad)
            extra_instance_features = set(instance.keys()) - set(selected_features)
            if extra_instance_features:
                 # Para este ejercicio, ignoraremos las características adicionales si todas las selected_features están presentes.
                 # En un entorno más estricto, podrías lanzar un error o registrar una advertencia.
                 # print(f"Advertencia: Las características adicionales en la instancia {i} serán ignoradas: {extra_instance_features}")
                 pass

            # Asegurar el orden y seleccionar solo las características requeridas
            ordered_instance = {feature: instance.get(feature) for feature in selected_features}
            df_list.append(ordered_instance)

        inference_df = pd.DataFrame(df_list, columns=selected_features)
        
        # Conversión de tipo de datos (el modelo espera numéricos, mayormente flotantes debido al escalado)
        # Esto asume que todas las características en selected_features deben ser numéricas.
        # Si no, se necesita un manejo de tipos más sofisticado aquí basado en metadatos de características.
        try:
            inference_df = inference_df.astype(float)
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error de conversión de tipo de datos. Asegúrate de que todos los valores de las características se puedan convertir a flotantes. Error: {e}"
            )

    except HTTPException: # Re-lanzar si ya es una HTTPException
        raise
    except Exception as e:
        print(f"Error procesando datos de entrada: {e}") # Registrar para el servidor
        raise HTTPException(status_code=400, detail=f"Error procesando datos de entrada: {str(e)}")

    try:
        # Realizar predicciones
        pred_labels = model.predict(inference_df)
        
        pred_probas = None
        if hasattr(model, "predict_proba"):
            pred_probas = model.predict_proba(inference_df)[:, 1] # Probabilidad de la clase '1' (yes)
        
        results = []
        for i in range(len(pred_labels)):
            result_item = {"predicted_label": int(pred_labels[i])}
            if pred_probas is not None:
                result_item["probability_yes"] = float(pred_probas[i])
            results.append(result_item)
            
        return {"predictions": results}

    except Exception as e:
        print(f"Error durante la predicción del modelo: {e}") # Registrar para el servidor
        # Verificar errores comunes de sklearn, ej. desajuste de nombres de características si no se detectó antes
        if "X has a different number of features than required by the model" in str(e) or "feature_names mismatch" in str(e):
             detail_msg = (
                f"Desajuste de características durante la predicción. El modelo esperaba {len(selected_features)} características. "
                f"Los datos proporcionados tienen {inference_df.shape[1]} características después del procesamiento. "
                f"Características del modelo (primeras 5): {selected_features[:5]}... "
                f"Características proporcionadas (primeras 5): {inference_df.columns.tolist()[:5]}..."
            )
             raise HTTPException(status_code=400, detail=detail_msg)

        raise HTTPException(status_code=500, detail=f"Error durante la predicción del modelo: {str(e)}")

# --- Ejecución principal para Uvicorn (si se ejecuta el script directamente) ---
# Esta parte generalmente no se incluye si Docker CMD llama directamente a uvicorn
# pero puede ser útil para pruebas locales.
# if __name__ == \"__main__\":
#     import uvicorn
#     uvicorn.run(app, host=\"0.0.0.0\", port=8000)
