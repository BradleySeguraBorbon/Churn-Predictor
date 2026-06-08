import gradio as gr


CUSTOM_CSS = """
.gradio-container { background-color: #0F1117 !important; font-family: 'Inter', sans-serif !important; }
.main-title { color:#667EEA; font-size:2rem !important; font-weight:800 !important;
              text-align:center; padding:1rem 0 0.3rem; }
.subtitle   { text-align:center; color:#888; font-size:.9rem; margin-bottom:1.2rem; }
.result-box textarea { background:#1A1D2E !important; border:2px solid #667EEA !important;
              border-radius:12px !important; color:#fff !important; font-size:1.15rem !important;
              font-weight:700 !important; text-align:center !important; padding:1rem !important; }
.prob-no  textarea { background:#0D2318 !important; border:1px solid #1DB954 !important;
              color:#1DB954 !important; font-size:1.4rem !important; font-weight:800 !important;
              text-align:center !important; border-radius:10px !important; }
.prob-yes textarea { background:#0D1A2A !important; border:1px solid #38BDF8 !important;
              color:#38BDF8 !important; font-size:1.4rem !important; font-weight:800 !important;
              text-align:center !important; border-radius:10px !important; }
.ai-box   { background:#12151F !important; border:1px solid #2A2D3E !important;
              border-radius:12px !important; padding:1.2rem !important; }
.section-lbl { color:#667EEA; font-size:.72rem; font-weight:700; text-transform:uppercase;
               letter-spacing:.12em; margin:.9rem 0 .3rem; }
#predict-btn { background:#667EEA !important; !important; border:none !important;
               color:#fff !important; font-size:1.05rem !important; font-weight:700 !important;
               padding:.75rem 2rem !important; border-radius:10px !important; }
#predict-btn:hover { background:#5A6FD6 !important; opacity:.85 !important; }
"""


def build_interface(predict_churn):
    """
    Función que construye y devuelve la interfaz de Gradio del predictor de churn,
    usando la función predict_churn obtenida como parámetro. La misma
    interfaz sirve tanto para el notebook como para correr con app.py.
    """
    with gr.Blocks(css=CUSTOM_CSS, title="Churn Predictor") as demo:

        gr.HTML("""
            <div class="main-title">Churn Predictor</div>
            <div class="subtitle">Prediccion de abandono de clientes para servicios de telecomunicaciones</div>
        """)

        with gr.Row():

            with gr.Column(scale=2):

                model_choice = gr.Radio(
                    ["Logistic Regression", "Random Forest"],
                    value="Logistic Regression", label="Modelo predictivo",
                )
                
                use_transformer = gr.Checkbox(
                    value=False,
                    label="Usar transformer para análisis (solo disponible con Logistic Regression)",
                    interactive=True, 
                )
                
                def _toggle_transformer_availability(modelo):
                    if modelo == "Logistic Regression":
                        return gr.update(interactive=True)
                    else:
                        return gr.update(interactive=False, value=False)

                model_choice.change(
                    fn=_toggle_transformer_availability,
                    inputs=[model_choice],
                    outputs=[use_transformer],
                )

                gr.HTML('<div class="section-lbl">Facturacion</div>')
                with gr.Row():
                    tenure         = gr.Slider(0, 72, value=29, step=1, label="Antiguedad (meses)")
                    MonthlyCharges = gr.Slider(18.25, 118.75, value=70.35, step=0.01, label="Cargo mensual ($)")
                TotalCharges = gr.Number(value=2000, label="Cargo total acumulado ($)")

                gr.HTML('<div class="section-lbl">Contrato e Internet</div>')
                with gr.Row():
                    Contract        = gr.Dropdown(["Month-to-month","One year","Two year"],
                                                  value="Month-to-month", label="Tipo de contrato")
                    InternetService = gr.Dropdown(["DSL","Fiber optic","No"],
                                                  value="Fiber optic", label="Servicio de Internet")

                gr.HTML('<div class="section-lbl">Servicios adicionales</div>')
                OPT = ["Yes", "No", "No internet service"]
                with gr.Row():
                    OnlineSecurity   = gr.Dropdown(OPT, value="No", label="Seguridad en linea")
                    OnlineBackup     = gr.Dropdown(OPT, value="No", label="Respaldo en linea")
                with gr.Row():
                    DeviceProtection = gr.Dropdown(OPT, value="No", label="Proteccion dispositivos")
                    TechSupport      = gr.Dropdown(OPT, value="No", label="Soporte tecnico")
                with gr.Row():
                    StreamingTV      = gr.Dropdown(OPT, value="No", label="Streaming TV")
                    StreamingMovies  = gr.Dropdown(OPT, value="No", label="Streaming Movies")

                gr.HTML('<div class="section-lbl">Pago y Telefonia</div>')
                with gr.Row():
                    PaymentMethod = gr.Dropdown(
                        ["Electronic check","Mailed check",
                         "Bank transfer (automatic)","Credit card (automatic)"],
                        value="Electronic check", label="Metodo de pago",
                    )
                    PhoneService = gr.Dropdown(["Yes","No"], value="Yes", label="Servicio telefonico")
                with gr.Row():
                    MultipleLines = gr.Dropdown(["Yes","No","No phone service"],
                                                value="No", label="Multiples lineas")

                gr.HTML('<div class="section-lbl">Perfil del cliente</div>')
                with gr.Row():
                    gender        = gr.Radio(["Male","Female"], value="Male", label="Genero")
                    SeniorCitizen = gr.Radio([0, 1], value=0, label="Adulto mayor (1 = Si)")
                with gr.Row():
                    Dependents    = gr.Radio(["Yes", "No"], value="No", label="Dependientes a cargo")
                    Partner       = gr.Radio(["Yes", "No"], value="No", label="Tiene pareja")

                btn = gr.Button("Predecir churn", elem_id="predict-btn")

            with gr.Column(scale=3):

                result = gr.Textbox(label="Resultado", interactive=False,
                                    elem_classes=["result-box"])

                with gr.Row():
                    prob_no  = gr.Textbox(label="Permanencia", interactive=False, elem_classes=["prob-no"])
                    prob_yes = gr.Textbox(label="Abandono",    interactive=False, elem_classes=["prob-yes"])

                plot = gr.Plot(label="Indicador de riesgo")

                gr.HTML('<div class="section-lbl">Análisis automático</div>')
                ai_analysis = gr.Markdown(
                    value="> El análisis aparecerá aquí tras ejecutar la prediccion.",
                    elem_classes=["ai-box"],
                )

        btn.click(
            predict_churn,
            inputs=[
                model_choice, tenure, MonthlyCharges, TotalCharges,
                Contract, InternetService,
                OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
                StreamingTV, StreamingMovies,
                PaymentMethod, PhoneService, MultipleLines,
                gender, SeniorCitizen, Dependents, Partner,
                use_transformer,
            ],
            outputs=[result, prob_no, prob_yes, plot, ai_analysis],
        )

    return demo