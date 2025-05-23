# Pipeline de Entrenamiento de Modelos para la Predicción de Marketing Bancario

## Objetivo
Este pipeline entrena y evalúa varios modelos de clasificación para predecir las suscripciones de clientes a depósitos a plazo. Implica comparar múltiples modelos, realizar ajuste de hiperparámetros en un modelo seleccionado y rastrear experimentos usando MLflow. El modelo ajustado con mejor rendimiento se guarda para la inferencia.

## Entrada
-   `bank-features-selected.csv`: Un archivo CSV que contiene las características seleccionadas y la variable objetivo ('y'). Se espera que este archivo sea generado por el notebook `feature_engineering.ipynb` en el directorio `../feature_pipeline/`.
-   `selected_feature_names.json` (opcional, como referencia): Un archivo JSON de `../feature_pipeline/` que lista los nombres de las características seleccionadas. El notebook se basa principalmente en las columnas del CSV.

## Pasos de Procesamiento
El notebook `model_training.ipynb` ejecuta los siguientes pasos clave:

1.  **Cargar Datos**:
    -   Carga el conjunto de datos `bank-features-selected.csv`.
    -   Separa las características (X) y la variable objetivo (y).
    -   Realiza las comprobaciones y limpiezas necesarias en la variable objetivo (por ejemplo, manejo de NaNs, asegurar tipo entero).

2.  **División Train-Test (Entrenamiento-Prueba)**:
    -   Divide los datos en conjuntos de entrenamiento (80%) y prueba (20%), usando estratificación sobre la variable objetivo para mantener las proporciones de clase.

3.  **Comparación con LazyPredict (Opcional)**:
    -   Si `lazypredict` está instalado, ejecuta `LazyClassifier` para obtener una visión general del rendimiento base de múltiples algoritmos de clasificación sin un ajuste extensivo.
    -   Los resultados (Accuracy, F1-score, ROC AUC, etc.) se muestran para ayudar a informar la selección del modelo.

4.  **Configuración del Experimento MLflow**:
    -   Configura un experimento de MLflow (por ejemplo, "Bank Marketing Predictions"). Si el experimento no existe, se crea.

5.  **Entrenamiento Inicial del Modelo (Línea Base)**:
    -   Un modelo seleccionado (por ejemplo, RandomForestClassifier, potencialmente guiado por LazyPredict) se entrena con parámetros predeterminados o básicos.
    -   Este modelo se evalúa y se muestran sus métricas de rendimiento.
    -   El modelo entrenado inicialmente se guarda (por ejemplo, `randomforestclassifier_model.joblib`).
    -   Este modelo inicial también puede registrarse opcionalmente en MLflow para comparación.

6.  **Ajuste de Hiperparámetros**:
    -   Se selecciona un modelo para el ajuste (por ejemplo, RandomForestClassifier, GradientBoostingClassifier, LogisticRegression), potencialmente guiado por los mejores desempeños de LazyPredict.
    -   Se define una cuadrícula de parámetros (parameter grid) para el modelo seleccionado.
    -   Se utiliza `GridSearchCV` (con validación cruzada de 3 pliegues y puntuación `f1_weighted` o `roc_auc`) para encontrar los hiperparámetros óptimos en los datos de entrenamiento.

7.  **Seguimiento en MLflow para el Modelo Ajustado**:
    -   Se inicia una nueva ejecución de MLflow para el modelo ajustado.
    -   **Parámetros Registrados**: Tipo de modelo, mejores hiperparámetros de GridSearchCV y la lista de nombres de características.
    -   **Métricas Registradas**: Accuracy, F1-scores (ponderado, macro y para la clase positiva 'yes'), Precisión (ponderada y macro), Recall (ponderado y macro) y ROC AUC se calculan sobre el conjunto de prueba y se registran.
    -   **Artefactos Registrados**:
        -   El informe de clasificación (como archivo de texto).
        -   Un gráfico de la matriz de confusión (como imagen PNG).
        -   Un gráfico de la importancia de las características (si aplica para el modelo, como imagen PNG).
        -   El modelo scikit-learn entrenado (ajustado) usando `mlflow.sklearn.log_model()`.

8.  **Exportación del Modelo**:
    -   El mejor modelo obtenido del ajuste de hiperparámetros (vía `GridSearchCV.best_estimator_`) se guarda localmente como `best_tuned_model.joblib` en el directorio `training_pipeline`. Este es el artefacto principal del modelo destinado al pipeline de inferencia.

## Salida
-   **`best_tuned_model.joblib`**: El modelo de clasificación ajustado con mejor rendimiento, guardado usando `joblib`. Esta es la salida principal para el pipeline de inferencia.
-   **(Opcional) `{model_name}_model.joblib`**: El modelo entrenado inicialmente (antes del ajuste) también se guarda.
-   **Experimento MLflow**:
    -   **Nombre del Experimento**: "Bank Marketing Predictions" (o como se configure en el notebook).
    -   **Ejecuciones (Runs)**: Contiene ejecuciones para el modelo inicial (opcional) y el modelo ajustado.
    -   **Registrado para cada ejecución**: Parámetros, métricas (Accuracy, F1, Precision, Recall, ROC AUC) y artefactos (archivo del modelo, informe de clasificación, gráfico de matriz de confusión, gráfico de importancia de características).

## Cómo Ejecutar
1.  Asegúrate de que el pipeline de ingeniería de características se haya ejecutado y que `bank-features-selected.csv` esté disponible en `../feature_pipeline/`.
2.  Ten un entorno Python con las bibliotecas necesarias instaladas (consulta `bank_marketing_project/requirements.txt` - este archivo debería crearse y poblarse). Bibliotecas clave: pandas, scikit-learn, matplotlib, seaborn, joblib, mlflow y opcionalmente lazypredict.
3.  Asegúrate de que un servidor de seguimiento de MLflow esté en ejecución o que MLflow esté configurado para registrar localmente (predeterminado).
4.  Abre y ejecuta el Jupyter notebook `model_training.ipynb` desde el directorio `bank_marketing_project/training_pipeline/`.
5.  El notebook cargará datos, entrenará modelos, realizará ajustes, registrará resultados en MLflow y guardará el `best_tuned_model.joblib`. Luego puedes ver los detalles del experimento en la UI de MLflow.
