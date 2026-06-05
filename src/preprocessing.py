import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Columnas que se eliminan del dataset por no influir en la predicción
COLUMNAS_A_ELIMINAR = ["customerID", "PaperlessBilling"]

NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SeniorCitizen",
]

CATEGORICAL_FEATURES = [
    "gender",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
    "Dependents",
    "Partner",
]


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica las acciones de limpieza al dataset:
    convierte TotalCharges a numérico (los espacios en blanco se vuelven en NaN),
    mapea la variable objetivo Churn de Yes/No a 1/0, y elimina las columnas
    innecesarias definidas en COLUMNAS_A_ELIMINAR.
    """
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df = df.drop(columns=COLUMNAS_A_ELIMINAR)

    return df


def build_preprocessors():
    """
    Función para construir los dos ColumnTransformer utilizados en el proyecto.
    Uno con el escalamiento de variables numéricas (para la Regresión Logística)
    y otro sin escalamiento (para Random Forest). Ambos imputan valores
    faltantes y aplican OneHotEncoder a las variables categóricas.
    """
    numeric_preprocessor_scaled = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    numeric_preprocessor_unscaled = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_preprocessor = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor_scaled = ColumnTransformer(transformers=[
        ("num", numeric_preprocessor_scaled, NUMERIC_FEATURES),
        ("cat", categorical_preprocessor, CATEGORICAL_FEATURES),
    ])

    preprocessor_unscaled = ColumnTransformer(transformers=[
        ("num", numeric_preprocessor_unscaled, NUMERIC_FEATURES),
        ("cat", categorical_preprocessor, CATEGORICAL_FEATURES),
    ])

    return preprocessor_scaled, preprocessor_unscaled