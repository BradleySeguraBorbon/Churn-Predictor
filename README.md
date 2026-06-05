# Churn Predictor

Sistema de predicción de abandono de clientes (Churn) para una empresa de servicios de telecomunicaciones, desarrollado como Proyecto Final del curso de Inteligencia Artificial de la Universidad Nacional, Sede Regional Brunca.

El sistema implementa un agente de clasificación supervisada que estima la probabilidad de que un cliente abandone el servicio a partir de su perfil (detalles de contrato, personal y uso del servicio). Incluye dos modelos comparables (Regresión Logística y Random Forest), una evaluación experimental con métricas formales, y una interfaz interactiva construida con la librería Gradio.

## Dataset

El proyecto utiliza el dataset público **IBM Telco Customer Churn**, disponible en Kaggle:
[https://www.kaggle.com/datasets/blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

Presenta 7043 registros y 21 columnas con información de clientes de una empresa de telecomunicaciones en California.

## Estructura del proyecto

```
Churn-Predictor/
├── app.py                    # Punto de inicio de la interfaz (carga modelos entrenados)
├── train_model.py            # Script para el entrenamiento (genera los archivos .joblib)
├── requirements.txt          # Dependencias específicas del proyecto
├── data/
│   └── Telco-Customer-Churn.csv
├── model/
│   ├── logreg_model.joblib   # Modelos entrenados generados por train_model.py
│   └── rf_model.joblib
├── src/
│   ├── data.py               # Carga del dataset
│   ├── preprocessing.py      # Limpieza y preprocesamiento
│   ├── model.py              # Pipelines y ajuste de hiperarámetros
│   ├── evaluation.py         # Métricas y exploración de umbrales
│   ├── prediction.py         # Función de predicción e indicador visual de resultado
│   └── interface.py          # Construcción de la interfaz con Gradio
└── notebook/
    └── Proyecto Final - Inteligencia Artificial.ipynb
```

## Requisitos

El proyecto se ejecuta en un entorno de Python con el stack base (`numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `jupyterlab`). Las dependencias adicionales se listan en `requirements.txt` y se instalan con:

```bash
pip install -r requirements.txt
```

## Ejecución

El proyecto puede ejecutarse de tres formas equivalentes:

### 1. Notebook en JupyterLab (local)

Iniciar el entorno virtual de Python, y desde la raíz del proyecto ejecutar:

```bash
jupyter lab
```

Se debe abrir `notebook/Proyecto Final - Inteligencia Artificial.ipynb` y ejecutar todas las celdas. El notebook entrena los modelos y ejecuta la interfaz Gradio al final.

### 2. Notebook en Google Colab

Abrir el notebook directamente desde el repositorio en GitHub mediante:

```
https://colab.research.google.com/github/BradleySeguraBorbon/Churn-Predictor/blob/main/notebook/Proyecto%20Final%20-%20Inteligencia%20Artificial.ipynb
```

La primera celda del notebook detecta automáticamente que el entorno es Google Colab y clona el repositorio si es necesario.

### 3. Scripts independientes (sin notebook)

Para entrenar los modelos:

```bash
python train_model.py
```

Esto genera los archivos de los modelos entrenados `model/logreg_model.joblib` y `model/rf_model.joblib`.

Y luego, para ejecutar la interfaz Gradio con los modelos entrenados:

```bash
python app.py
```

## Autores

- Bradley Segura Borbón
- Noemí Murillo Godínez
- Sebastián Vega Zuñiga

## Curso

Inteligencia Artificial  
Universidad Nacional, Sede Regional Brunca   
I Ciclo, 2026