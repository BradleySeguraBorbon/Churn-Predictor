"""
Script para entrenar los modelos de predicción de churn.

Se ejecuta el pipeline completo, donde se carga el dataset, aplica limpieza,
divide en train/test, construye los preprocesadores, entrena Regresión Logística
y Random Forest con ajuste de hiperparámetros, y guardalos modelos entrenados en model/.

Se usa con el comando:
    python train_model.py
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

from src.preprocessing import clean_dataset, build_preprocessors
from src.model import (
    build_logreg_pipeline,
    build_rf_pipeline,
    tune_logreg,
    tune_rf,
)
from src.data import load_dataset

MODEL_DIR = "model"
LOGREG_PATH = os.path.join(MODEL_DIR, "logreg_model.joblib")
RF_PATH = os.path.join(MODEL_DIR, "rf_model.joblib")


def main():
    # 1. Cargar dataset
    print("Cargando dataset...")
    df = load_dataset()

    # 2. Limpiar dataset
    print("Aplicando limpieza...")
    df_clean = clean_dataset(df)

    # 3. Separar variable objetivo y predictoras
    X = df_clean.drop(columns=["Churn"])
    y = df_clean["Churn"]

    # 4. División train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 5. Construir preprocesadores
    preprocessor_scaled, preprocessor_unscaled = build_preprocessors()

    # 6. Entrenar Regresión Logística con tuning
    print("\nEntrenando Regresión Logística...")
    logreg_pipeline = build_logreg_pipeline(preprocessor_scaled)
    logreg_pipeline.fit(X_train, y_train)
    logreg_grid = tune_logreg(logreg_pipeline, X_train, y_train)
    best_logreg = logreg_grid.best_estimator_
    print(f"Mejor F1 (CV): {logreg_grid.best_score_:.4f}")

    # 7. Entrenar Random Forest con tuning
    print("\nEntrenando Random Forest...")
    rf_pipeline = build_rf_pipeline(preprocessor_unscaled)
    rf_pipeline.fit(X_train, y_train)
    rf_grid = tune_rf(rf_pipeline, X_train, y_train)
    best_rf = rf_grid.best_estimator_
    print(f"Mejor F1 (CV): {rf_grid.best_score_:.4f}")

    # 8. Guardar modelos entrenados
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_logreg, LOGREG_PATH)
    joblib.dump(best_rf, RF_PATH)
    print(f"\nModelos guardados en:\n  {LOGREG_PATH}\n  {RF_PATH}")


if __name__ == "__main__":
    main()