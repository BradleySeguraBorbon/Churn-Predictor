"""
En este archivo se generan explicaciones en lenguaje natural sobre las predicciones
del modelo de Regresión Logística, aprovechando sus coeficientes que son interpretables. 
Se usa el modelo google/flan-t5-base para convertir las contribuciones numéricas
de cada variable en oraciones que sean legibles en inglés.
"""

import warnings
import logging

import numpy as np

from src.preprocessing import CATEGORICAL_FEATURES
logging.getLogger("transformers").setLevel(logging.ERROR)

_GENERATOR = None
_MODEL_NAME = "google/flan-t5-large"


def _get_generator():
    """
    Se hace una carga perezosa del modelo. La primera llamada lo descarga
    y lo cachea localmente. Llamadas siguientes son instantáneas.
    """
    global _GENERATOR
    if _GENERATOR is not None:
        return _GENERATOR

    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(_MODEL_NAME)
        _GENERATOR = (tokenizer, model)
        return _GENERATOR
    except ImportError:
        warnings.warn(
            "Las dependencias 'transformers' y/o 'torch' no están instaladas. "
            "La explicación por IA no estará disponible, por lo que se usará el análisis basado en reglas. "
        )
        return None
    except Exception as e:
        warnings.warn(f"No se pudo cargar el modelo {_MODEL_NAME}: {e}")
        return None


# Descripciones en lenguaje natural para cada categoría one-hot, para convertirlas en oraciones legibles
_CATEGORICAL_DESCRIPTIONS = {
    ("Contract", "Month-to-month"): "has a month-to-month contract",
    ("Contract", "One year"): "has a one-year contract",
    ("Contract", "Two year"): "has a two-year contract",
    ("InternetService", "DSL"): "uses DSL internet",
    ("InternetService", "Fiber optic"): "uses fiber optic internet",
    ("InternetService", "No"): "does not have internet service",
    ("OnlineSecurity", "Yes"): "has online security enabled",
    ("OnlineSecurity", "No"): "does not have online security",
    ("OnlineSecurity", "No internet service"): "has no internet service",
    ("OnlineBackup", "Yes"): "has online backup enabled",
    ("OnlineBackup", "No"): "does not have online backup",
    ("OnlineBackup", "No internet service"): "has no internet service",
    ("DeviceProtection", "Yes"): "has device protection",
    ("DeviceProtection", "No"): "does not have device protection",
    ("DeviceProtection", "No internet service"): "has no internet service",
    ("TechSupport", "Yes"): "has tech support enabled",
    ("TechSupport", "No"): "does not have tech support",
    ("TechSupport", "No internet service"): "has no internet service",
    ("StreamingTV", "Yes"): "uses streaming TV",
    ("StreamingTV", "No"): "does not use streaming TV",
    ("StreamingTV", "No internet service"): "has no internet service",
    ("StreamingMovies", "Yes"): "uses streaming movies",
    ("StreamingMovies", "No"): "does not use streaming movies",
    ("StreamingMovies", "No internet service"): "has no internet service",
    ("PaymentMethod", "Electronic check"): "pays by electronic check",
    ("PaymentMethod", "Mailed check"): "pays by mailed check",
    ("PaymentMethod", "Bank transfer (automatic)"): "pays by automatic bank transfer",
    ("PaymentMethod", "Credit card (automatic)"): "pays by automatic credit card",
    ("PhoneService", "Yes"): "has phone service",
    ("PhoneService", "No"): "does not have phone service",
    ("MultipleLines", "Yes"): "has multiple phone lines",
    ("MultipleLines", "No"): "has a single phone line",
    ("MultipleLines", "No phone service"): "has no phone service",
    ("gender", "Male"): "is male",
    ("gender", "Female"): "is female",
    ("Dependents", "Yes"): "has dependents at home",
    ("Dependents", "No"): "has no dependents at home",
    ("Partner", "Yes"): "has a partner",
    ("Partner", "No"): "has no partner",
}

_DOMAIN_KNOWLEDGE_RISK = {
    "uses fiber optic internet": "fiber optic users face strong competition from other providers and have higher service expectations, leading to more frequent provider switching",
    "pays by electronic check": "electronic check users tend to evaluate alternative providers more frequently and have weaker payment commitments compared to automatic methods",
}

def _describe_factor(feature_name, customer_row):
    """
    Convierte un nombre de feature en una descripción legible en inglés
    """
    if feature_name.startswith("num__"):
        col = feature_name.replace("num__", "")
        value = customer_row[col]
        descs = {
            "tenure": f"has been a customer for {int(value)} months",
            "MonthlyCharges": f"pays ${value:.2f} per month",
            "TotalCharges": f"has paid ${value:.2f} in total charges so far",
            "SeniorCitizen": "is a senior citizen" if value == 1 else "is not a senior citizen",
        }
        return descs.get(col)

    if feature_name.startswith("cat__"):
        rest = feature_name.replace("cat__", "")
        for col in CATEGORICAL_FEATURES:
            prefix = col + "_"
            if rest.startswith(prefix):
                category = rest[len(prefix):]
                if str(customer_row[col]) == category:
                    return _CATEGORICAL_DESCRIPTIONS.get((col, category))
                return None
    return None


def _generate_factor_sentence(generator, description, is_risk):
    """
    Se le pide al LLM que genere una oración corta explicando por qué un factor
    aumenta o disminuye el riesgo de churn para este cliente.
    """
    if is_risk:
        domain_hint = _DOMAIN_KNOWLEDGE_RISK.get(description)
        if domain_hint:
            prompt = (
                "Rewrite the following explanation as a single concise sentence suitable for a customer retention report.\n\n"
                f"Customer characteristic: {description}\n"
                f"Known reason this increases churn risk: {domain_hint}\n"
                "Rewritten explanation:"
            )
        else:
            prompt = (
                "Explain why each customer characteristic increases telecom churn risk.\n\n"
                "Characteristic: has a month-to-month contract\n"
                "Explanation: Month-to-month contracts have no long-term commitment, making it easy for customers to switch providers without penalties.\n\n"
                "Characteristic: pays by electronic check\n"
                "Explanation: Electronic check users are statistically less loyal and tend to evaluate alternatives more frequently than customers with automatic payments.\n\n"
                f"Characteristic: {description}\n"
                "Explanation:"
            )
    else:
        prompt = (
            "Explain why each customer characteristic decreases telecom churn risk.\n\n"
            "Characteristic: has a two-year contract\n"
            "Explanation: Long-term contracts create commitment and early-termination penalties, making customers far less likely to leave.\n\n"
            "Characteristic: has tech support enabled\n"
            "Explanation: Customers with tech support have a stronger relationship with the provider and resolve issues more quickly, reducing frustration-driven churn.\n\n"
            f"Characteristic: {description}\n"
            "Explanation:"
        )

    try:
        tokenizer, model = generator
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            num_beams=4,
            no_repeat_ngram_size=3,
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    except Exception as e:
        print(f"Error en _generate_factor_sentence('{description}'): {e}")
        return None


def _generate_recommendation(generator, prob_yes, risk_descs, protective_descs):
    """
    Genera una recomendación basada en los resultados para mejorar la retención.
    """
    prob_pct = int(round(prob_yes * 100))
    risk_summary = ", ".join(risk_descs) if risk_descs else "no major risks"
    protective_summary = ", ".join(protective_descs) if protective_descs else "no clear protective factors"

    prompt = (
        "Given a customer's churn risk and their characteristics, recommend concrete retention actions.\n\n"
        "Customer: 80% churn risk. Risks: month-to-month contract, electronic check payment. Protective: long tenure.\n"
        "Recommendation: Offer a one-year contract with a 15% discount for the first three months, and propose switching to automatic credit card payment in exchange for a small loyalty credit.\n\n"
        "Customer: 45% churn risk. Risks: no tech support, fiber optic service. Protective: has dependents.\n"
        "Recommendation: Include free tech support for six months and propose a family-plan bundle that adds value for households with dependents.\n\n"
        f"Customer: {prob_pct}% churn risk. Risks: {risk_summary}. Protective: {protective_summary}.\n"
        "Recommendation:"
    )

    try:
        tokenizer, model = generator
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            num_beams=4,
            no_repeat_ngram_size=3,
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    except Exception as e:
        print(f"Error en _generate_recommendation: {e}")
        return None

def explain_prediction(pipeline, customer_data, prob_yes, n_factors=3):
    """
    Se genera una explicación en lenguaje natural sobre la predicción del modelo,
    usando los coeficientes de la Regresión Logística
    """
    generator = _get_generator()
    if generator is None:
        return None

    try:
        print("Iniciando explain_prediction...")

        # Se extrean el preprocesador, modelo, nombres de features y los coeficientes
        preprocessor = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]
        feature_names = preprocessor.get_feature_names_out()
        coefficients = model.coef_[0]
        print(f"Features: {len(feature_names)}, Coefs: {len(coefficients)}")

        # Se transforman los datos del cliente y se calcula el aporte de cada factor
        X_transformed = preprocessor.transform(customer_data)
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()
        x_vector = X_transformed[0]
        contributions = coefficients * x_vector
        print(f"Contribuciones calculadas. Top 3 positivas: {sorted(contributions, reverse=True)[:3]}")

        # Se ordenan e identifican los factores top de riesgo y protección
        idx_desc = np.argsort(contributions)[::-1]
        IGNORE_AS_RISK = {"num__tenure", "num__TotalCharges"}
        customer_row = customer_data.iloc[0]

        risk_factors = []
        for idx in idx_desc:
            if contributions[idx] <= 0:
                break
            if feature_names[idx] in IGNORE_AS_RISK:
                continue
            description = _describe_factor(feature_names[idx], customer_row)
            if description:
                risk_factors.append((description, contributions[idx]))
            if len(risk_factors) >= n_factors:
                break
        print(f"Factores de riesgo identificados: {len(risk_factors)}")

        protective_factors = []
        for idx in idx_desc[::-1]:
            if contributions[idx] >= 0:
                break
            description = _describe_factor(feature_names[idx], customer_row)
            if description:
                protective_factors.append((description, contributions[idx]))
            if len(protective_factors) >= n_factors:
                break
        print(f"Factores protectores identificados: {len(protective_factors)}")

        # Se generan las oraciones por cada factor con el LLM
        print("Generando oraciones de riesgo con LLM...")
        risk_sentences = []
        for desc, _ in risk_factors:
            sentence = _generate_factor_sentence(generator, desc, is_risk=True)
            if sentence:
                risk_sentences.append(f"- {sentence}")
        print(f"Oraciones de factores de riesgo generadas: {len(risk_sentences)}")

        print("Generando oraciones de factores protectores con LLM...")
        protective_sentences = []
        for desc, _ in protective_factors:
            sentence = _generate_factor_sentence(generator, desc, is_risk=False)
            if sentence:
                protective_sentences.append(f"- {sentence}")
        print(f"Oraciones de factores protectores generadas: {len(protective_sentences)}")

        # Y finalmente se genera la recomendación general
        print("Generando recomendación...")
        recommendation = _generate_recommendation(
            generator,
            prob_yes,
            [d for d, _ in risk_factors],
            [d for d, _ in protective_factors],
        )
        print(f"Recomendación: {'OK' if recommendation else 'None'}")

        # Si todo falló, devolver None
        if not risk_sentences and not protective_sentences and not recommendation:
            print("[DEBUG explainer] TODO falló, devolviendo None")
            return None

        risk_block = "\n".join(risk_sentences) if risk_sentences else "- No significant risk factors identified."
        prot_block = "\n".join(protective_sentences) if protective_sentences else "- No significant protective factors identified."
        rec_block = recommendation if recommendation else "No recommendation generated."
        prob_pct = prob_yes * 100
        prob_no_pct = (1 - prob_yes) * 100

        print("[DEBUG explainer] Éxito, devolviendo análisis completo")
        return f"""### Customer profile analysis (AI-generated)

**Model:** Logistic Regression &nbsp;|&nbsp; **P(churn):** {prob_pct:.1f}% &nbsp;|&nbsp; **P(retention):** {prob_no_pct:.1f}%

---

**Risk factors detected:**
{risk_block}

**Protective factors:**
{prot_block}

---

{rec_block}
"""

    except Exception as e:
        import traceback
        print(f"EXCEPCIÓN: {e}")
        print(traceback.format_exc())
        return None