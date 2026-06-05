from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

def build_logreg_pipeline(preprocessor):
    """
    Aquí se construye el pipeline de Regresión Logística, usando el preprocesamiento 
    con escalamiento y un clasificador LogisticRegression
    """
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ])


def build_rf_pipeline(preprocessor):
    """
    En esta función se construye el pipeline base de Random Forest, con el
    preprocesamiento sin escalamiento y un clasificador RandomForestClassifier
    """
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(random_state=42)),
    ])
    

def tune_logreg(pipeline, X_train, y_train):
    """
    Se hace el ajuste de hiperparámetros de Regresión Logística
    usando GridSearchCV con validación cruzada, optimizando F1.
    """
    param_grid = {
        "model__C": [0.001, 0.01, 0.1, 1, 10, 100],
        "model__penalty": ["l2"],
        "model__solver": ["lbfgs"],
        "model__class_weight": [None, "balanced"],
    }

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    grid.fit(X_train, y_train)
    return grid


def tune_rf(pipeline, X_train, y_train, cv_strategy=None):
    """
    Se hace el ajuste de hiperparámetros para Random Forest usando
    RandomizedSearchCV con 20 combinaciones y validación cruzada, 
    optimizando F1
    """
    param_dist = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [10, 15, 20, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2"],
        "model__class_weight": [None, "balanced", "balanced_subsample"],
    }

    if cv_strategy is None:
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=20,
        cv=cv_strategy,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
        random_state=42,
        refit=True,
    )

    grid.fit(X_train, y_train)
    return grid