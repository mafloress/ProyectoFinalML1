# Pipeline de Inferencia por Lotes para la Predicción de Marketing Bancario

## Objetivo
Este pipeline carga un modelo de clasificación preentrenado y genera predicciones por lotes sobre nuevos datos de características no vistos. Las predicciones indican si es probable que un cliente se suscriba a un depósito a plazo.

## Entrada
-   **`best_tuned_model.joblib`**: El modelo compatible con scikit-learn, entrenado y ajustado. Se espera que este archivo se encuentre en el directorio `../training_pipeline/`.
-   **`bank-features-selected.csv`**: Un archivo CSV que contiene los datos de características para los cuales se realizarán las predicciones. Estos datos deben haber pasado por los mismos pasos de preprocesamiento y selección de características que los datos utilizados para entrenar el modelo. Se espera que se encuentre en el directorio `../feature_pipeline/`.
-   **`selected_feature_names.json`** (opcional pero recomendado para robustez): Un archivo JSON de `../feature_pipeline/` que lista los nombres de las características con las que se entrenó el modelo. Esto ayuda a asegurar la consistencia de las columnas.

## Pasos de Procesamiento
El notebook `batch_inference.ipynb` ejecuta los siguientes pasos clave:

1.  **Cargar Modelo Entrenado**:
    -   Carga el modelo serializado (por ejemplo, `best_tuned_model.joblib`) usando `joblib`.
    -   Intenta cargar la lista de nombres de características con las que se entrenó el modelo (por ejemplo, desde `selected_feature_names.json` o un atributo del modelo como `feature_names_in_`).

2.  **Cargar Datos de Características para Inferencia**:
    -   Carga los nuevos datos por lotes desde `bank-features-selected.csv`.
    -   Elimina cualquier columna objetivo (por ejemplo, 'y') si está presente, ya que no es necesaria para la inferencia.
    -   **Verificación de Consistencia de Características**: Asegura que las columnas en los datos de inferencia coincidan con las características con las que se entrenó el modelo (nombre y orden). Intentará reordenar las columnas o subconjuntos de ellas si `trained_model_features` están disponibles. Se emiten advertencias o errores si se encuentran discrepancias críticas (por ejemplo, características esperadas faltantes).

3.  **Generar Predicciones**:
    -   Utiliza el método `predict()` del modelo cargado sobre los datos de características preparados (X_inference) para generar predicciones de etiquetas de clase (0 o 1).
    -   Si el modelo lo soporta, se utiliza `predict_proba()` para generar probabilidades de predicción para la clase positiva.

4.  **Guardar Predicciones**:
    -   Crea un nuevo DataFrame que contiene las características originales de los datos de entrada por lotes.
    -   Añade nuevas columnas a este DataFrame:
        -   `predicted_label`: La etiqueta de clase predicha.
        -   `prediction_probability_yes`: La probabilidad de predecir 'yes' (si está disponible).
    -   Guarda este DataFrame combinado en `predictions.csv` en el directorio `bank_marketing_project/inference_pipeline/`.

## Salida
-   **`predictions.csv`**: Un archivo CSV que contiene las características de entrada originales junto con `predicted_label` y (si aplica) `prediction_probability_yes` para cada instancia en el lote.

## Cómo Ejecutar
1.  Asegúrate de que el pipeline de entrenamiento se haya ejecutado correctamente y que `best_tuned_model.joblib` esté disponible en el directorio `../training_pipeline/`.
2.  Asegúrate de que el pipeline de ingeniería de características haya producido `bank-features-selected.csv` (e idealmente `selected_feature_names.json`) en el directorio `../feature_pipeline/`. Este CSV servirá como los datos "nuevos" para la inferencia por lotes.
3.  Ten un entorno Python con las bibliotecas necesarias instaladas (consulta `bank_marketing_project/requirements.txt`). Bibliotecas clave: pandas, numpy, joblib, scikit-learn.
4.  Abre y ejecuta el Jupyter notebook `batch_inference.ipynb` desde el directorio `bank_marketing_project/inference_pipeline/`.
5.  El notebook cargará el modelo y los datos, generará predicciones y las guardará en `predictions.csv`.
