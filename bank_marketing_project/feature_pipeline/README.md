# Pipeline de Ingeniería de Características para Datos de Marketing Bancario

## Objetivo
Este pipeline procesa el conjunto de datos crudos de marketing bancario (`bank-full.csv`) para realizar Análisis Exploratorio de Datos (EDA), ingeniería de características y selección de características. El objetivo es generar un conjunto de datos limpio y optimizado, listo para entrenar un modelo que prediga las suscripciones a depósitos a plazo.

## Entrada
-   Archivo crudo `bank-full.csv` (se espera que el notebook lo descargue y coloque en este directorio, o que ya esté presente).

## Pasos de Procesamiento
El notebook `feature_engineering.ipynb` realiza los siguientes pasos clave:

1.  **Carga de Datos**:
    -   Descarga el "Bank Marketing Data Set" del UCI Machine Learning Repository si `bank-full.csv` no está ya presente.
    -   Carga `bank-full.csv` (delimitado por punto y coma) en un DataFrame de pandas.

2.  **Análisis Exploratorio de Datos (EDA)**:
    -   Inspección inicial de datos (shape, head, info, describe).
    -   Identificación de características numéricas y categóricas.
    -   Visualización de la distribución de la variable objetivo ('y').
    -   Análisis de características numéricas usando histogramas y diagramas de caja (boxplots) (vs. el objetivo).
    -   Análisis de características categóricas usando gráficos de conteo (count plots) (y vs. el objetivo).
    -   Análisis de correlación de características numéricas usando un mapa de calor (heatmap).

3.  **Preprocesamiento de Datos**:
    -   **Manejo de valores 'unknown'**: Reemplaza las cadenas 'unknown' con NaN, luego imputa estos NaNs usando la moda de cada columna respectiva.
    -   **Codificación de Características Categóricas Binarias**: Convierte 'default', 'housing', 'loan' y la variable objetivo 'y' de 'yes'/'no' a 1/0.
    -   **Codificación One-Hot de Otras Características Categóricas**: Aplica `pd.get_dummies` (con `drop_first=True`) a otras características categóricas de tipo objeto.
    -   **Escalado de Características Numéricas**: Escala todas las características numéricas (excluyendo la variable objetivo 'y', ya codificada como binaria) usando `StandardScaler`.
    -   Los datos completamente preprocesados (antes de la selección de características) se guardan en `bank-full-processed.csv`.

4.  **Selección de Características**:
    -   **Ganancia de Información Mutua (Mutual Information Gain)**: Calcula las puntuaciones de información mutua entre cada característica y la variable objetivo 'y' utilizando los datos preprocesados.
    -   **Selección de las K Mejores Características**: Selecciona las K (K=20, o menos si el total de características es menor a 20) características principales basándose en las puntuaciones más altas de información mutua.
    -   Se muestra un gráfico de barras de las puntuaciones de MI para las características principales.

## Salida
1.  `bank-full-processed.csv`: Contiene los datos completamente preprocesados (todas las características después de la limpieza, codificación y escalado, junto con la variable objetivo). Esto puede ser útil para análisis que requieran todas las características.
2.  `bank-features-selected.csv`: Contiene las características finales seleccionadas (las K principales según la información mutua) y la variable objetivo. **Esta es la salida principal destinada al pipeline de entrenamiento.**
3.  `selected_feature_names.json`: Un archivo JSON que contiene una lista de los nombres de las características seleccionadas.

## Cómo Ejecutar
1.  Asegúrate de tener un entorno Python con las bibliotecas necesarias instaladas (consulta `bank_marketing_project/requirements.txt` - aunque este archivo podría poblarse más adelante en el proyecto). Las bibliotecas clave utilizadas son pandas, numpy, scikit-learn, matplotlib, seaborn y requests.
2.  Abre y ejecuta el Jupyter notebook `feature_engineering.ipynb` desde el directorio `bank_marketing_project/feature_pipeline/`.
3.  El notebook descargará los datos si no están presentes, realizará todos los pasos de procesamiento y guardará los archivos de salida (`bank-features-selected.csv`, `bank-full-processed.csv` y `selected_feature_names.json`) en este directorio.
