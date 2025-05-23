# Usar un runtime oficial de Python como imagen base
FROM python:3.9-slim

# Establecer variables de entorno
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Instalar dependencias del sistema requeridas por algunas bibliotecas de ML si es necesario
# RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
# Para este proyecto, se asume que las compilaciones delgadas de scikit-learn y pandas no necesitan herramientas de compilación extensas.

# Copiar el archivo de requerimientos al contenedor en /app
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar el contenido del directorio local 'app' al contenedor en /app/app
# Esto significa que main.py estará en /app/app/main.py
COPY app/ /app/app/

# Copiar el modelo entrenado y la lista de características al contenedor
# Ajustar rutas para asegurar que estén ubicadas correctamente para main.py
# main.py espera que MODEL_DIR sea /app, y luego rutas como training_pipeline/model.joblib
# Así que, colocamos estos directorios directamente bajo /app
COPY training_pipeline/best_tuned_model.joblib /app/training_pipeline/
COPY feature_pipeline/selected_feature_names.json /app/feature_pipeline/

# Exponer el puerto 8000 al mundo exterior
EXPOSE 8000

# Definir el comando para ejecutar tu aplicación
# Buscará main.py dentro del módulo/directorio 'app' copiado anteriormente.
# La variable de entorno MODEL_DIR en main.py tomará por defecto '.', que es /app en este contexto de WORKDIR.
# Entonces, MODEL_PATH será ./training_pipeline/best_tuned_model.joblib -> /app/training_pipeline/best_tuned_model.joblib
# Y FEATURES_PATH será ./feature_pipeline/selected_feature_names.json -> /app/feature_pipeline/selected_feature_names.json
# Esto se alinea con los comandos COPY.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
