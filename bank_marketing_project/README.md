# Proyecto de Predicción de Marketing Bancario
Miguel Ángel Flores Saldívar

## 0. Uso de LLMs y Agentes IA
Se utilizaron los siguientes LLMs para poder tomar ideas para tomar la base del proyecto y buscar fuentes reales de datos para el proyecto, se hizo la comparativa entre cada respuesta de cada uno y se escogío la mas ad hoc:
- Gemini
- Copilot
- Deepseek
- ChatGPT

Se utilizó adicional para documentación e implementación en Github:
- Jules Google


## 1. Objetivo del Proyecto
El objetivo principal de este proyecto es analizar el conjunto de datos de Marketing Bancario para predecir si un cliente se suscribirá a un depósito a plazo. Esta tarea de machine learning está estructurada como un sistema de 3 pipelines (características, entrenamiento e inferencia) diseñado para predicciones por lotes. El proyecto busca demostrar un flujo de trabajo MLOps de extremo a extremo, incluyendo procesamiento de datos, entrenamiento de modelos, seguimiento de experimentos con MLflow e inferencia por lotes.

## 2. Descripción del Conjunto de Datos
-   **Nombre**: Conjunto de datos de Marketing Bancario (Bank Marketing dataset)
-   **Fuente**: UCI Machine Learning Repository
-   **Enlace**: [`https://archive.ics.uci.edu/dataset/222/bank+marketing`](https://archive.ics.uci.edu/dataset/222/bank+marketing)
-   **Detalles**:
    -   El proyecto utiliza el archivo `bank-full.csv`, que contiene un conjunto completo de ejemplos.
    -   El conjunto de datos incluye 17 variables de entrada que cubren datos del cliente (por ejemplo, edad, trabajo, estado civil, educación), datos relacionados con la campaña (por ejemplo, tipo de contacto, mes, día de la semana, duración de la llamada) y resultados de campañas anteriores (por ejemplo, pdays, previous, poutcome).
    -   La variable objetivo es 'y', que indica si el cliente se suscribió a un depósito a plazo (binario: 'yes' o 'no').

## 3. Estructura del Proyecto (Arquitectura de 3 Pipelines)
El proyecto está organizado en tres pipelines principales, cada uno residiendo en su propio directorio:

-   **Pipeline de Características (`feature_pipeline/`)**:
    -   Este pipeline es responsable de cargar los datos crudos (`bank-full.csv`).
    -   Realiza un Análisis Exploratorio de Datos (EDA) extensivo para entender las características y relaciones de los datos.
    -   Los pasos de preprocesamiento incluyen el manejo de valores 'unknown' (imputación con la moda), codificación de características categóricas (binaria y one-hot encoding) y escalado de características numéricas usando `StandardScaler`.
    -   La selección de características se realiza utilizando Ganancia de Información Mutua (Mutual Information Gain) para identificar las características más relevantes para la tarea de predicción.
    -   **Salida**: `bank-features-selected.csv` (conteniendo las características seleccionadas y el objetivo) y `selected_feature_names.json` (lista de nombres de características seleccionadas).

-   **Pipeline de Entrenamiento (`training_pipeline/`)**:
    -   Carga las características seleccionadas de `bank-features-selected.csv`.
    -   Opcionalmente usa `LazyPredict` para una comparación amplia de varios modelos de clasificación para obtener una línea base de rendimiento.
    -   Realiza ajuste de hiperparámetros (usando `GridSearchCV`) en un modelo elegido (por ejemplo, RandomForestClassifier, GradientBoostingClassifier). La elección del modelo y el enfoque en F1-score/ROC AUC para el ajuste se justifica por el desequilibrio de clases en la variable objetivo.
    -   Registra experimentos, parámetros del modelo, métricas de rendimiento y artefactos del modelo (incluyendo el modelo entrenado y gráficos de importancia de características) usando MLflow.
    -   **Salida**: `best_tuned_model.joblib` (el modelo ajustado con mejor rendimiento, serializado).

-   **Pipeline de Inferencia (`inference_pipeline/`)**:
    -   Carga el `best_tuned_model.joblib` del pipeline de entrenamiento.
    -   Carga nuevos datos de características por lotes (que deberían haber pasado por los mismos pasos de preprocesamiento que los datos de entrenamiento; para este proyecto, `bank-features-selected.csv` se reutiliza como ejemplo de entrada por lotes).
    -   Asegura la consistencia entre las características en los datos de entrada y las características con las que se entrenó el modelo.
    -   Genera predicciones (etiquetas de clase y probabilidades).
    -   **Salida**: `predictions.csv` (conteniendo las características de entrada junto con sus etiquetas y probabilidades predichas).

## 4. Decisiones Clave en el Modelado
-   **Variable Objetivo**: 'y', una variable binaria que indica la suscripción a un depósito a plazo ('yes'/'no'), codificada como 1/0.
-   **Métricas de Evaluación**: Debido al significativo desequilibrio de clases (más 'no' que 'yes'), el enfoque principal para la evaluación y ajuste del modelo es en **F1-score** (especialmente para la clase positiva 'yes') y **ROC AUC**. También se reportan exactitud (accuracy), precisión y recall para una comprensión integral.
-   **Manejo del Desequilibrio**: Se emplearon estrategias como el uso de `class_weight='balanced'` en modelos de scikit-learn (por ejemplo, RandomForestClassifier, LogisticRegression) durante el entrenamiento para dar más peso a la clase minoritaria.
-   **Selección de Características**: Se utilizó Ganancia de Información Mutua para seleccionar las K (K=20) características principales del conjunto de datos preprocesado, con el objetivo de mejorar el rendimiento del modelo y reducir la complejidad.
-   **Selección del Modelo**: `LazyPredict` proporcionó una comparación inicial amplia. Luego, se eligió un clasificador robusto y prometedor (por ejemplo, RandomForestClassifier o GradientBoostingClassifier) para un ajuste detallado de hiperparámetros y evaluación final.

## 5. Resultados Principales y Resumen de Métricas
El modelo final ajustado (por ejemplo, un RandomForestClassifier o GradientBoostingClassifier, dependiendo del resultado del ajuste) demuestra un rendimiento robusto en la predicción de suscripciones a depósitos a plazo. Por ejemplo, un resultado típico podría ser:

*"El RandomForestClassifier final ajustado alcanzó un F1-score (clase positiva 'yes') de aproximadamente 0.50-0.60 y un ROC AUC de aproximadamente 0.85-0.90 en el conjunto de prueba retenido. Esto indica un buen equilibrio entre precisión y recall para identificar suscriptores potenciales, y una fuerte capacidad para distinguir entre las dos clases."* (Los valores reales dependen de la ejecución final del notebook de entrenamiento).

-   Métricas detalladas, parámetros y artefactos del modelo (incluyendo gráficos de importancia de características y el modelo mismo) para cada ejecución de entrenamiento son rastreados en **MLflow** bajo el nombre de experimento: **"Bank Marketing Predictions"**.
-   Se anima a los usuarios a consultar los archivos README de cada pipeline y los Jupyter notebooks para obtener registros de ejecución más detallados, salidas específicas y visualizaciones.

## 6. Configuración y Cómo Ejecutar

### Prerrequisitos
-   Python 3.7+
-   pip (instalador de paquetes de Python)
-   Git
-   Docker (para la contenerización, ver Sección 8)

### Instalación
1.  Clona el repositorio:
    ```bash
    git clone <repository_url> 
    # Reemplaza <repository_url> con la URL real del repositorio de este proyecto
    ```
2.  Navega al directorio del proyecto:
    ```bash
    cd bank_marketing_project
    ```
3.  Instala los paquetes de Python requeridos:
    ```bash
    pip install -r requirements.txt 
    # Ploomber está incluido en requirements.txt
    ```

### Ejecutando los Pipelines Manualmente
Los notebooks idealmente deberían ejecutarse en la siguiente secuencia:

1.  **Pipeline de Características**:
    -   Navega al directorio del pipeline de características: `cd feature_pipeline`
    -   Ejecuta el Jupyter Notebook: `jupyter notebook feature_engineering.ipynb` (o usa JupyterLab)
    -   Esto generará `bank-features-selected.csv` y `selected_feature_names.json` en el directorio `feature_pipeline`, y `bank-full-processed.csv`.

2.  **Pipeline de Entrenamiento**:
    -   Navega al directorio del pipeline de entrenamiento: `cd ../training_pipeline` (si estabas en `feature_pipeline`)
    -   Asegúrate de que MLflow esté accesible. Puedes iniciar la UI de MLflow para ver los experimentos ejecutando:
        ```bash
        mlflow ui
        ```
        (Esto típicamente inicia un servidor en `http://localhost:5000`)
    -   Ejecuta el Jupyter Notebook: `jupyter notebook model_training.ipynb`
    -   Esto entrenará el modelo, registrará los experimentos en MLflow y guardará `best_tuned_model.joblib` en el directorio `training_pipeline`.

3.  **Pipeline de Inferencia**:
    -   Navega al directorio del pipeline de inferencia: `cd ../inference_pipeline` (si estabas en `training_pipeline`)
    -   Ejecuta el Jupyter Notebook: `jupyter notebook batch_inference.ipynb`
    -   Esto cargará el modelo entrenado y generará `predictions.csv` en el directorio `inference_pipeline` usando `bank-features-selected.csv` como entrada.

Para ejecución automatizada, consulta la sección "Automatización con GitHub Actions" más abajo. Para orquestación de pipelines, consulta "Orquestación con Ploomber".

## 7. Automatización con GitHub Actions

### 7.1. Objetivo de la Automatización
Este proyecto implementa GitHub Actions para automatizar la ejecución de los pipelines de ingeniería de características y entrenamiento de modelos. Esta automatización puede ser activada manualmente o por pushes a la rama principal, asegurando que los artefactos de datos y modelos se regeneren consistentemente.

### 7.2. Flujos de Trabajo Implementados
Los siguientes flujos de trabajo están definidos en el directorio `.github/workflows/`:

-   **Pipeline de Ingeniería de Características (`.github/workflows/feature_pipeline.yml`)**:
    -   **Propósito**: Automatiza la ejecución del notebook `feature_pipeline/feature_engineering.ipynb` usando `papermill`. Genera artefactos de datos clave: `bank-features-selected.csv` (características seleccionadas para el entrenamiento del modelo) y `selected_feature_names.json` (lista de estos nombres de características). El notebook ejecutado (`executed_feature_engineering.ipynb`) también se guarda. Estos archivos generados luego se confirman (commit) de vuelta al repositorio.
    -   **Disparadores (Triggers)**:
        -   Manualmente vía `workflow_dispatch`.
        -   En push a la rama `main` si se detectan cambios en `feature_pipeline/**`, `requirements.txt`, o el propio archivo de flujo de trabajo (`.github/workflows/feature_pipeline.yml`).

-   **Pipeline de Entrenamiento de Modelos (`.github/workflows/training_pipeline.yml`)**:
    -   **Propósito**: Automatiza la ejecución del notebook `training_pipeline/model_training.ipynb` usando `papermill`. Este pipeline entrena el modelo, genera el artefacto `best_tuned_model.joblib` y registra datos del experimento (parámetros, métricas, artefactos) en una instancia de seguimiento local de MLflow (guardada en el directorio `mlruns/`). El `best_tuned_model.joblib`, el notebook ejecutado (`executed_model_training.ipynb`) y el directorio `mlruns/` luego se confirman de vuelta al repositorio.
    -   **Disparadores (Triggers)**:
        -   Manualmente vía `workflow_dispatch`.
        -   En push a la rama `main` si se detectan cambios en `training_pipeline/**`, `requirements.txt`, el propio archivo de flujo de trabajo (`.github/workflows/training_pipeline.yml`), o importantemente, si la entrada clave `feature_pipeline/bank-features-selected.csv` o `feature_pipeline/selected_feature_names.json` se actualizan (indicando que nuevas características están disponibles para el entrenamiento).

### 7.3. Uso y Monitoreo
-   **Monitoreo**: El estado y los registros de estos flujos de trabajo automatizados pueden ser monitoreados desde la pestaña "Actions" del repositorio de GitHub.
-   **Disparadores Manuales**: Ambos flujos de trabajo pueden ser disparados manualmente desde la pestaña "Actions" seleccionando el flujo de trabajo deseado y haciendo clic en "Run workflow".
-   **Salidas**: Cuando los flujos de trabajo se ejecutan, confirman sus salidas (archivos de datos generados, artefactos de modelo, notebooks ejecutados y el directorio `mlruns`) de vuelta al repositorio de Git. Esto proporciona versionamiento para estos artefactos directamente dentro del repositorio.
-   **Ejecución de Notebooks**: Se utiliza `papermill` para ejecutar los Jupyter notebooks de manera no interactiva, permitiendo la parametrización (aunque no se usa extensivamente en la configuración actual) y guardando el notebook ejecutado con sus salidas para inspección.
-   **Datos de MLflow**: Por simplicidad, el flujo de trabajo `training_pipeline.yml` confirma el directorio `mlruns` (conteniendo datos de experimentos locales de MLflow) de vuelta al repositorio. Para entornos más robustos, colaborativos o de producción, se recomienda encarecidamente configurar un servidor de seguimiento remoto de MLflow.

## 8. Contenerización del Modelo con Docker

### 8.1. Propósito
La contenerización con Docker empaqueta la aplicación FastAPI junto con el modelo entrenado, dependencias y configuraciones necesarias en una unidad portátil y autocontenida. Esto asegura consistencia entre diferentes entornos y simplifica el despliegue.

### 8.2. Dockerfile
Se crea un `Dockerfile` en la raíz del directorio `bank_marketing_project`. Los pasos clave en el Dockerfile incluyen:
-   Comenzar desde una imagen base de Python (por ejemplo, `python:3.9-slim`).
-   Establecer el directorio de trabajo en `/app`.
-   Copiar `requirements.txt` e instalar dependencias.
-   Copiar el código de la aplicación FastAPI (de `bank_marketing_project/app/`) a `/app/app/` dentro de la imagen.
-   Copiar el modelo entrenado (`training_pipeline/best_tuned_model.joblib`) a `/app/training_pipeline/` dentro de la imagen.
-   Copiar la lista de características seleccionadas (`feature_pipeline/selected_feature_names.json`) a `/app/feature_pipeline/` dentro de la imagen.
-   Exponer el puerto 8000.
-   Definir el `CMD` para ejecutar la aplicación FastAPI usando `uvicorn app.main:app --host 0.0.0.0 --port 8000`.

### 8.3. Construir la Imagen de Docker
Navega al directorio `bank_marketing_project` en tu terminal y ejecuta:
```bash
docker build -t bank-marketing-service .
```
Este comando construye una imagen de Docker etiquetada como `bank-marketing-service`.

### 8.4. Ejecutar el Contenedor de Docker
Una vez construida la imagen, ejecútala como un contenedor:
```bash
docker run -d -p 8001:8000 bank-marketing-service
```
-   `-d`: Ejecuta el contenedor en modo detached (en segundo plano).
-   `-p 8001:8000`: Mapea el puerto 8001 en tu máquina host al puerto 8000 dentro del contenedor (donde se ejecuta la app FastAPI). Puedes usar otros puertos del host si el 8001 está ocupado, por ejemplo, `-p 80:8000` para el puerto HTTP estándar.

### 8.5. Probar el Endpoint de la API
Después de que el contenedor esté en ejecución, puedes probar el endpoint `/predict/`.

#### Usando `curl`
Necesitarás un payload JSON de muestra. Primero, averigua algunos nombres de características reales de `bank_marketing_project/feature_pipeline/selected_feature_names.json`. Supongamos que los primeros son "duration", "pdays", "euribor3m", "age", "campaign". Reemplázalos con características reales si difieren.

**Comando `curl` de ejemplo:**
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
*Nota: El payload de ejemplo anterior usa nombres de características y valores hipotéticos. **Debes** reemplazarlos con nombres de características reales de tu `selected_feature_names.json` y proporcionar valores plausibles para cada uno. Asegúrate de que todas las características listadas en `selected_feature_names.json` estén incluidas en cada diccionario de instancia en el payload.*

La API debería devolver una respuesta JSON con las predicciones, por ejemplo:
```json
{
  "predictions": [
    {"predicted_label": 0, "probability_yes": 0.123},
    {"predicted_label": 1, "probability_yes": 0.789}
  ]
}
```

### 8.6. Subir a Docker Hub (Siguiente Paso)
Una vez que la imagen esté probada y funcionando, puedes subirla a Docker Hub (u otro registro de contenedores):
1.  Inicia sesión en Docker Hub: `docker login`
2.  Etiqueta tu imagen: `docker tag bank-marketing-service yourusername/bank-marketing-service:latest` (reemplaza `yourusername`)
3.  Sube la imagen: `docker push yourusername/bank-marketing-service:latest`

## 9. Orquestación con Ploomber

### 9.1. Definición de `pipeline.yaml`
Ploomber se utiliza para definir y orquestar la secuencia de tareas en este proyecto. La lógica de orquestación se captura en el archivo `pipeline.yaml` en la raíz del directorio `bank_marketing_project`:

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

### 9.2. Uso
Ploomber ayuda a gestionar el orden de ejecución y las dependencias de los notebooks.

-   **Instalación**: Ploomber ya debería estar incluido en `requirements.txt`. Si se instala manualmente:
    ```bash
    pip install ploomber
    ```
-   **Ejecutando el Pipeline**: Para ejecutar el pipeline completo como se define en `pipeline.yaml`:
    ```bash
    ploomber build
    ```
-   **Visualizando el Pipeline**: Para generar un gráfico del Grafo Acíclico Dirigido (DAG) del pipeline:
    ```bash
    ploomber plot
    ```
    Esto creará un archivo `pipeline.html` (o `.png` si graphviz está instalado) mostrando las dependencias de las tareas.
-   **Verificando el Estado de las Tareas**: Para ver el estado actual de las tareas (por ejemplo, si necesitan ejecutarse):
    ```bash
    ploomber status
    ```
-   **Forzando la Re-ejecución de Tareas**: Para forzar la re-ejecución de una tarea específica y sus dependencias descendentes, incluso si Ploomber considera que están actualizadas:
    ```bash
    ploomber build --force <task-name>
    ```
    Por ejemplo:
    ```bash
    ploomber build --force feature-engineering
    ```
-   **Nota sobre Parametrización**: El `pipeline.yaml` proporcionado orquesta los notebooks existentes basándose en sus dependencias de ruta fija actuales. Para una ejecución más dinámica y paso de parámetros directamente desde Ploomber (por ejemplo, pasando `upstream['feature-engineering']['data']` al notebook de entrenamiento), los notebooks pueden modificarse añadiendo una celda etiquetada como 'parameters'. Los comentarios dentro de `pipeline.yaml` proporcionan pistas sobre esto.

## 10. Trabajo Futuro / Próximos Pasos
-   **Mejoras en CI/CD**:
    -   Implementar un flujo de trabajo automatizado para el pipeline de inferencia en GitHub Actions.
    -   Explorar estrategias para una gestión de artefactos más avanzada (por ejemplo, usando un feature store dedicado como Hopsworks, o almacenamiento en la nube para datos/modelos en lugar de confirmarlos directamente en Git, especialmente para artefactos más grandes).
    -   Integrar un servidor de seguimiento remoto de MLflow para una gestión de experimentos más robusta.
-   **Integración con Feature Store**: Utilizar un feature store como Hopsworks para la gestión centralizada de características, versionamiento y servicio, asegurando la consistencia entre el entrenamiento y la inferencia.
-   **Mejoras en el Despliegue de API**:
    -   Refinar aún más la aplicación FastAPI con un manejo de errores más completo, logging y, potencialmente, validación de entrada basada en tipos de características dinámicas si es necesario.
    -   Explorar el despliegue en servicios gestionados (por ejemplo, AWS SageMaker, Azure ML, Google Vertex AI, o Kubernetes).
-   **Orquestación de Flujos de Trabajo**: Implementar completamente Ploomber o Apache Airflow para gestionar las dependencias y la ejecución de las diferentes etapas del pipeline. (Concepto de Ploomber iniciado en la Sección 9).
-   **Monitoreo Avanzado de Modelos**: Implementar un monitoreo de modelos sofisticado para la deriva de datos (data drift), deriva de concepto (concept drift) y degradación del rendimiento a lo largo del tiempo usando herramientas como Evidently AI o Grafana.
-   **Escalabilidad**: Explorar opciones para escalar el procesamiento de datos (por ejemplo, usando Spark) y el entrenamiento/inferencia de modelos si el tamaño del conjunto de datos crece significativamente.

---
Este README proporciona una visión general del proyecto de Predicción de Marketing Bancario. Para información detallada sobre cada componente, por favor consulta los archivos README y los Jupyter notebooks dentro de los directorios de cada pipeline respectivo.
