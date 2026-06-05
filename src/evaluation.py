import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    RocCurveDisplay,
)

def explorar_umbral(nombre, pipeline, X_test, y_test):
    """
    Función para explorar el umbral de decisión que maximiza F1
    usando un pipeline entrenado. Compara contra el umbral por defecto (0.5),
    luego grafica la curva de F1 vs el umbral y retorna el umbral óptimo
    """
    # Probabilidades predichas para la clase 1
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Curva precision-recall: devuelve precision, recall y umbrales
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

    # Calcular F1 en cada umbral (se añade epsilon para evitar división por cero)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)

    # precision_recall_curve devuelve un valor más en precision/recall que
    # en thresholds. Por eso se descarta el último elemento de f1_scores.
    f1_scores = f1_scores[:-1]

    # Se encuentra el umbral que maximiza F1
    idx_optimo = f1_scores.argmax()
    umbral_optimo = thresholds[idx_optimo]
    f1_optimo = f1_scores[idx_optimo]

    # F1 en el umbral por defecto (0.5) para comparar
    y_pred_default = (y_proba >= 0.5).astype(int)
    f1_default = f1_score(y_test, y_pred_default)

    print("=" * 70)
    print(f"Exploración de umbral: {nombre}")
    print("=" * 70)
    print(f"  F1 con umbral por defecto (0.5): {f1_default:.4f}")
    print(f"  Umbral óptimo según F1:          {umbral_optimo:.4f}")
    print(f"  F1 con umbral óptimo:            {f1_optimo:.4f}")
    print(f"  Mejora absoluta:                 {f1_optimo - f1_default:+.4f}")

    # Gráfico: F1 vs umbral
    plt.figure(figsize=(8, 4))
    plt.plot(thresholds, f1_scores, label="F1 por umbral")
    plt.axvline(x=0.5, color="gray", linestyle="--", label="Umbral por defecto (0.5)")
    plt.axvline(x=umbral_optimo, color="red", linestyle="--",
                label=f"Umbral óptimo ({umbral_optimo:.3f})")
    plt.xlabel("Umbral de decisión")
    plt.ylabel("F1-score (clase 1)")
    plt.title(f"F1 vs Umbral - {nombre}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return umbral_optimo


def evaluate_trained_model(name, pipeline, X_test, y_test, positive_class=1, threshold=0.5):
    """
    Se encarga de evaluar un pipeline entrenado con el conjunto de prueba, 
    calculando accuracy, precision, recall, F1 y ROC-AUC. Luego
    muestra el reporte de clasificación, la matriz de confusión y la curva ROC
    """
    # Si el modelo soporta predict_proba, aplicar umbral personalizado
    if hasattr(pipeline, "predict_proba"):
        y_score = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_score >= threshold).astype(int)
        roc_auc = roc_auc_score(y_test, y_score)
    else:
        y_score = None
        y_pred = pipeline.predict(X_test)
        roc_auc = np.nan

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label=positive_class),
        "recall": recall_score(y_test, y_pred, pos_label=positive_class),
        "f1": f1_score(y_test, y_pred, pos_label=positive_class),
        "roc_auc": roc_auc,
        "threshold": threshold,
    }

    print("=" * 100)
    print(f"Evaluación del modelo: {name}")
    print(f"Umbral de decisión utilizado: {threshold:.4f}")
    print("=" * 100)

    print("\nMétricas principales:")
    for k, v in metrics.items():
        if k not in ("model", "threshold"):
            print(f"{k:>10}: {v:.4f}")

    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Matriz de confusión - {name}")
    plt.xlabel("Predicción")
    plt.ylabel("Valor real")
    plt.show()

    if y_score is not None:
        RocCurveDisplay.from_predictions(y_test, y_score)
        plt.title(f"Curva ROC - {name}")
        plt.show()

    return metrics