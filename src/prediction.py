import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src.explainer import explain_prediction

def make_churn_gauge(prob_yes: float):
    """
    Se usa para construira un indicador visual semicircular (gauge) que muestra la
    probabilidad de abandono del cliente, mostrando en zonas de riesgo
    bajo, medio y alto (se incluye medio para mejorar la claridad, pero solo bajo y alto son las principales).
    """
    fig, ax = plt.subplots(figsize=(6, 3.5), subplot_kw={"aspect": "equal"})
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#0F1117")

    zone_colors = ["#1DB954", "#F5A623", "#E53935"]
    zone_labels = ["Bajo", "Medio", "Alto"]
    zone_ranges = [(0, 0.33), (0.33, 0.66), (0.66, 1.0)]
    theta_start, theta_end = np.pi, 0.0

    for (lo, hi), color in zip(zone_ranges, zone_colors):
        t0 = theta_start + (theta_end - theta_start) * lo
        t1 = theta_start + (theta_end - theta_start) * hi
        theta = np.linspace(t0, t1, 100)
        outer_r, inner_r = 1.0, 0.62
        x_outer = np.cos(theta) * outer_r
        y_outer = np.sin(theta) * outer_r
        x_inner = np.cos(theta[::-1]) * inner_r
        y_inner = np.sin(theta[::-1]) * inner_r
        ax.fill(
            np.concatenate([x_outer, x_inner]),
            np.concatenate([y_outer, y_inner]),
            color=color, alpha=0.85, zorder=2,
        )

    for (lo, hi), label, color in zip(zone_ranges, zone_labels, zone_colors):
        mid = (lo + hi) / 2
        angle = theta_start + (theta_end - theta_start) * mid
        rx, ry = np.cos(angle) * 0.81, np.sin(angle) * 0.81
        ax.text(rx, ry + 0.04, label, ha="center", va="center",
                fontsize=9, fontweight="bold", color="white", zorder=5)

    needle_angle = theta_start + (theta_end - theta_start) * prob_yes
    nx = np.cos(needle_angle) * 0.78
    ny = np.sin(needle_angle) * 0.78
    ax.annotate("", xy=(nx, ny), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->,head_width=0.12,head_length=0.08",
                                color="white", lw=2.5), zorder=6)
    ax.add_patch(plt.Circle((0, 0), 0.07, color="white", zorder=7))

    risk_pct = int(round(prob_yes * 100))
    if prob_yes < 0.33:
        risk_label, risk_color = "RIESGO BAJO",  "#1DB954"
    elif prob_yes < 0.66:
        risk_label, risk_color = "RIESGO MEDIO", "#F5A623"
    else:
        risk_label, risk_color = "RIESGO ALTO",  "#E53935"

    ax.text(0, -0.22, f"{risk_pct}%", ha="center", va="center",
            fontsize=26, fontweight="bold", color="white", zorder=8)
    ax.text(0, -0.45, risk_label, ha="center", va="center",
            fontsize=11, fontweight="bold", color=risk_color, zorder=8)
    ax.text(0, -0.60, "probabilidad de abandono", ha="center", va="center",
            fontsize=8, color="#888888", zorder=8)

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-0.75, 1.15)
    ax.axis("off")
    plt.tight_layout(pad=0.3)
    return fig


def build_predict_churn(logreg_pipeline, rf_pipeline):
    """
    Aquí se construye la función predict_churn usando los pipelines enviados por parámetro.
    """
    def predict_churn(
        model_choice,
        tenure,
        MonthlyCharges,
        TotalCharges,
        Contract,
        InternetService,
        OnlineSecurity,
        OnlineBackup,
        DeviceProtection,
        TechSupport,
        StreamingTV,
        StreamingMovies,
        PaymentMethod,
        PhoneService,
        MultipleLines,
        gender,
        SeniorCitizen,
        Dependents,
        Partner,
        use_transformer,
    ):
        try:
            data = pd.DataFrame([{
                "tenure": tenure,
                "MonthlyCharges": MonthlyCharges,
                "TotalCharges": TotalCharges,
                "Contract": Contract,
                "InternetService": InternetService,
                "OnlineSecurity": OnlineSecurity,
                "OnlineBackup": OnlineBackup,
                "DeviceProtection": DeviceProtection,
                "TechSupport": TechSupport,
                "StreamingTV": StreamingTV,
                "StreamingMovies": StreamingMovies,
                "PaymentMethod": PaymentMethod,
                "PhoneService": PhoneService,
                "MultipleLines": MultipleLines,
                "gender": gender,
                "SeniorCitizen": SeniorCitizen,
                "Dependents": Dependents,
                "Partner": Partner,
            }])

            model = rf_pipeline if model_choice == "Random Forest" else logreg_pipeline
            pred     = model.predict(data)[0]
            prob     = model.predict_proba(data)[0]
            prob_no  = float(prob[0])
            prob_yes = float(prob[1])

            result = (
                "⚠️ El cliente CANCELARÁ el servicio"
                if pred == 1 else
                "✅ El cliente CONTINUARÁ con el servicio"
            )

            gauge_fig = make_churn_gauge(prob_yes)

            ai_analysis = None

            # Análisis con transformer para Regresión Logística
            if use_transformer and model_choice == "Logistic Regression":
                print("Llamando a explain_prediction...")
                ai_analysis = explain_prediction(logreg_pipeline, data, prob_yes)
                print(f"explain_prediction devolvió: {'None (fallback)' if ai_analysis is None else 'resultado válido'}")
                if ai_analysis is None:
                    print("⚠️  Transformer no disponible o falló. Usando análisis rule-based.")

            # Análisis basado en reglas (el tradicional)
            if ai_analysis is None:
                risk_factors, protective_factors = [], []

                if Contract == "Month-to-month":
                    risk_factors.append("contrato mensual sin compromiso a largo plazo")
                else:
                    protective_factors.append(f"contrato *{Contract}* que ancla al cliente")
                if InternetService == "Fiber optic":
                    risk_factors.append("fibra óptica (alta competencia y expectativas elevadas)")
                if PaymentMethod == "Electronic check":
                    risk_factors.append("cheque electrónico (históricamente asociado a mayor churn)")
                if tenure < 12:
                    risk_factors.append(f"antigüedad baja ({tenure} meses), período de mayor riesgo")
                elif tenure > 36:
                    protective_factors.append(f"alta fidelidad ({tenure} meses con el servicio)")
                if OnlineSecurity == "No" and InternetService != "No":
                    risk_factors.append("sin seguridad en línea activa")
                if TechSupport == "No" and InternetService != "No":
                    risk_factors.append("sin soporte técnico contratado")
                if OnlineSecurity == "Yes":  protective_factors.append("seguridad en línea activa")
                if TechSupport == "Yes":     protective_factors.append("soporte técnico contratado")
                if OnlineBackup == "Yes":    protective_factors.append("respaldo en línea activo")
                if Dependents == "Yes":      protective_factors.append("cliente con dependientes a cargo (mayor estabilidad)")
                if Partner == "Yes":         protective_factors.append("cliente con pareja (perfil de mayor estabilidad)")

                risk_lines = "\n".join(f"- {f}" for f in risk_factors)      or "- Ninguno identificado"
                prot_lines = "\n".join(f"- {f}" for f in protective_factors) or "- Ninguno identificado"

                if prob_yes >= 0.66:
                    recomendacion = "**Acción inmediata recomendada.** Riesgo elevado. Contacto proactivo, oferta de retención y revisión de experiencia del servicio."
                elif prob_yes >= 0.33:
                    recomendacion = "**Monitoreo activo.** Riesgo moderado. Seguimiento periódico y beneficios adicionales en la próxima facturación."
                else:
                    recomendacion = "**Cliente estable.** Riesgo bajo. Mantener calidad del servicio y programas de fidelización."

                ai_analysis = f"""### Análisis del perfil del cliente

**Modelo:** {model_choice} &nbsp;|&nbsp; **P(abandono):** {prob_yes:.1%} &nbsp;|&nbsp; **P(permanencia):** {prob_no:.1%}

---

**Factores de riesgo detectados:**
{risk_lines}

**Factores protectores:**
{prot_lines}

---

{recomendacion}
"""
            return result, f"{prob_no:.1%}", f"{prob_yes:.1%}", gauge_fig, ai_analysis

        except Exception as e:
            import traceback
            return f"Error: {e}", "—", "—", None, f"```\n{traceback.format_exc()}\n```"

    return predict_churn