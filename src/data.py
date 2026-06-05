import os
import pandas as pd


DATASET_URL = "https://drive.google.com/uc?id=1-QzFHC28fvOzu_lAAYS0PqlOavJhzvD7"
DATASET_LOCAL_PATH = "data/Telco-Customer-Churn.csv"


def load_dataset() -> pd.DataFrame:
    """
    Se carga el dataset desde una ruta local si existe, y en caso contrario
    se descarga desde la URL remota
    """
    if os.path.exists(DATASET_LOCAL_PATH):
        try:
            return pd.read_csv(DATASET_LOCAL_PATH)
        except Exception:
            pass

    try:
        return pd.read_csv(DATASET_URL)
    except Exception as e:
        raise RuntimeError(
            f"No se pudo cargar el dataset ni desde {DATASET_LOCAL_PATH} "
            f"ni desde la URL remota. Error: {e}"
        )