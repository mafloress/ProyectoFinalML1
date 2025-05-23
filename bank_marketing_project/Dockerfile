# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for some ML libraries if needed
# RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
# For this project, assuming scikit-learn and pandas slim builds don't need extensive build tools.

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the local 'app' directory contents into the container at /app/app
# This means main.py will be at /app/app/main.py
COPY app/ /app/app/

# Copy the trained model and feature list into the container
# Adjust paths to ensure they are correctly located for main.py
# main.py expects MODEL_DIR to be /app, and then paths like training_pipeline/model.joblib
# So, we place these directories directly under /app
COPY training_pipeline/best_tuned_model.joblib /app/training_pipeline/
COPY feature_pipeline/selected_feature_names.json /app/feature_pipeline/

# Expose port 8000 to the outside world
EXPOSE 8000

# Define the command to run your application
# It will look for main.py inside the 'app' module/directory copied earlier.
# The MODEL_DIR environment variable in main.py will default to '.', which is /app in this WORKDIR context.
# So, MODEL_PATH will be ./training_pipeline/best_tuned_model.joblib -> /app/training_pipeline/best_tuned_model.joblib
# And FEATURES_PATH will be ./feature_pipeline/selected_feature_names.json -> /app/feature_pipeline/selected_feature_names.json
# This aligns with the COPY commands.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
