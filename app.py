"""
Punto de entrada de la aplicación. Carga los modelos previamente entrenados por
train_model.py desde model/, y luego construye la función de predicción usando ambos modelos,
para correr la interfaz de Gradio.

Se ejecuta con el comando:
    python app.py

Requisito previo:
    Haber ejecutado train_model.py al menos una vez para generar los
    archivos model/logreg_model.joblib y model/rf_model.joblib.
"""

import os
import joblib

from src.prediction import build_predict_churn
from src.interface import build_interface


MODEL_DIR = "model"
LOGREG_PATH = os.path.join(MODEL_DIR, "logreg_model.joblib")
RF_PATH = os.path.join(MODEL_DIR, "rf_model.joblib")


def main():
    # Primero se verifica que los modelos existan
    if not (os.path.exists(LOGREG_PATH) and os.path.exists(RF_PATH)):
        raise FileNotFoundError(
            "No se encontraron los modelos entrenados. "
            "Ejecutá primero 'python train_model.py' para generarlos."
        )

    # Se cargan modelos serializados
    print("Cargando modelos...")
    logreg_pipeline = joblib.load(LOGREG_PATH)
    rf_pipeline = joblib.load(RF_PATH)

    # Se constuye la función de predicción usando los modelos
    predict_churn = build_predict_churn(logreg_pipeline, rf_pipeline)

    # Se construye y lanza la interfaz
    demo = build_interface(predict_churn)
    demo.launch(share=True, debug=True, show_error=True)


if __name__ == "__main__":
    main()