"""Aplicación de Minería de Datos. El entrenamiento se realiza en el notebook."""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MONTHS = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio",
          "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
LABELS = {
    "Edad": "Edad (años)",
    "Genero": "Género",
    "IMC": "Índice de masa corporal (kg/m²)",
    "Glucosa_mg_dL": "Glucosa (mg/dL)",
    "Presion_Sistolica_mmHg": "Presión sistólica (mmHg)",
    "Colesterol_mg_dL": "Colesterol (mg/dL)",
    "Actividad_Fisica": "Actividad física",
    "Fumador": "¿Fuma?",
    "Horas_Sueno": "Sueño (horas por noche)",
    "Nivel_Estres": "Nivel de estrés (1 a 10)",
    "Consultas_Medicas": "Consultas médicas en los últimos 12 meses",
    "Mes_Numero": "Mes del registro",
}


@st.cache_resource
def load_pipeline():
    return joblib.load(ROOT / "pipeline.joblib")


st.set_page_config(page_title="Nivel de riesgo en datos sintéticos", layout="wide")
report = json.loads((ROOT / "resultados.json").read_text(encoding="utf-8"))
st.title("Nivel de riesgo en datos sintéticos de salud")
st.write("Estima si un registro simulado pertenece a la categoría de riesgo bajo, medio o alto.")
st.caption("Beny Niebles · Geiser Guerrero | Minería de Datos")
st.caption("Fuente de los datos: Dataset_Sintetico_Salud_.xlsx, material del curso.")
st.info("Prototipo académico con datos sintéticos. No sustituye una evaluación profesional.")
prediction, exploration, evaluation = st.tabs(["Predicción", "Exploración", "Evaluación"])

with prediction:
    st.subheader("Datos del registro")
    st.write("Usa un caso simulado. Los rangos indicados corresponden a los datos de entrenamiento.")
    with st.form("registro"):
        columns = st.columns(3)
        values = {}
        for index, (name, spec) in enumerate(report["fields"].items()):
            with columns[index % 3]:
                if name == "Mes_Numero":
                    values[name] = st.selectbox(
                        LABELS[name], spec["options"], format_func=lambda month: MONTHS[month - 1], key=name
                    )
                elif "options" in spec:
                    values[name] = st.selectbox(LABELS[name], spec["options"], key=name)
                else:
                    values[name] = st.text_input(
                        LABELS[name], value=str(spec["default"]),
                        help=f"Rango observado: {spec['min']} a {spec['max']}.", key=name
                    )
        submitted = st.form_submit_button("Estimar nivel de riesgo", type="primary")

    if submitted:
        errors = []
        for name, spec in report["fields"].items():
            if "options" in spec:
                if values[name] not in spec["options"]:
                    errors.append(f"Selecciona una categoría válida en {LABELS[name]}.")
                continue
            try:
                value = float(values[name].strip().replace(",", "."))
                if not spec["min"] <= value <= spec["max"]:
                    raise ValueError
                if spec["integer"] and not value.is_integer():
                    raise ValueError
                values[name] = int(value) if spec["integer"] else value
            except ValueError:
                kind = "un número entero" if spec["integer"] else "un número"
                errors.append(f"{LABELS[name]} debe ser {kind} entre {spec['min']} y {spec['max']}.")
        if errors:
            for error in errors:
                st.error(error)
        else:
            pipeline = load_pipeline()
            row = pd.DataFrame([values], columns=list(report["fields"]))
            label = pipeline.predict(row)[0]
            probabilities = dict(zip(pipeline.classes_, pipeline.predict_proba(row)[0]))
            st.success(f"Nivel estimado: {label}")
            st.write("Probabilidad que el modelo asigna a cada clase")
            for column, risk in zip(st.columns(3), report["classes"]):
                column.metric(risk, f"{probabilities[risk]:.1%}")
            st.caption("Estos porcentajes describen la clasificación del registro sintético.")

with exploration:
    st.subheader("Registros por nivel de riesgo")
    st.bar_chart(pd.Series(report["distribution"], name="Registros"), sort=False)
    st.caption("El tamaño de cada grupo ayuda a interpretar el desempeño por clase.")
    st.subheader("Glucosa mediana por nivel de riesgo")
    st.bar_chart(pd.Series(report["glucose"], name="Glucosa (mg/dL)"), sort=False)
    st.subheader("Composición del riesgo por actividad física")
    st.bar_chart(pd.DataFrame.from_dict(report["activity"], orient="index"), stack=True)
    st.caption("Las barras apiladas muestran el porcentaje de cada clase dentro de cada grupo de actividad.")

with evaluation:
    st.subheader("Resultados en el conjunto de prueba")
    st.caption(f"Regresión logística · {report['test_count']} registros de prueba")
    captions = ["Exactitud", "Precisión macro", "Recall macro", "F1 macro"]
    for column, title, value in zip(st.columns(4), captions, report["metrics"].values()):
        column.metric(title, f"{value:.1%}")
    st.write(f"Validación cruzada de 5 particiones en entrenamiento: F1 macro "
             f"{report['cv_mean']:.3f} ± {report['cv_std']:.3f} (desviación estándar).")
    st.subheader("Resultados por clase")
    st.dataframe(pd.DataFrame.from_dict(report["class_metrics"], orient="index").round(3))
    st.subheader("Matriz de confusión")
    labels = report["classes"]
    st.dataframe(pd.DataFrame(report["matrix"], index=[f"Real: {x}" for x in labels],
                              columns=[f"Predicho: {x}" for x in labels]))
    high_row = report["matrix"][labels.index("Alto")]
    missed_high = sum(high_row) - high_row[labels.index("Alto")]
    st.caption(f"El modelo clasificó {missed_high} de {sum(high_row)} registros de riesgo alto "
               "como riesgo bajo o medio.")
    st.subheader("Limitaciones")
    st.write("La etiqueta de riesgo fue construida con fines didácticos a partir de varias "
             "características simuladas. El rendimiento mide cuánto reproduce ese patrón el modelo. "
             "Se desconoce su validez con registros reales y las probabilidades no se calibraron.")
