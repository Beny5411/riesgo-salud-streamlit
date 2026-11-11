import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
LABELS = {
    "Pregnancies": "Número de embarazos",
    "Glucose": "Glucosa",
    "BloodPressure": "Presión arterial",
    "SkinThickness": "Pliegue cutáneo",
    "Insulin": "Insulina",
    "BMI": "Índice de masa corporal",
    "DiabetesPedigreeFunction": "Índice de antecedentes familiares (DiabetesPedigreeFunction)",
    "Age": "Edad (años)",
}


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "pipeline.joblib")


st.set_page_config(page_title="Proyecto de diabetes · Beny y Geiser", layout="wide")
report = json.loads((ROOT / "resultados.json").read_text(encoding="utf-8"))
st.title("Proyecto de diabetes")
st.write("Clasificación de registros según la variable Outcome a partir de ocho características.")
st.caption("Beny Niebles · Geiser Guerrero | Minería de Datos")
st.caption("Fuente registrada en el segundo avance: [Diabetes Data Set, Kaggle]"
           "(https://www.kaggle.com/datasets/vikasukani/diabetes-data-set).")
st.info("Prototipo académico. No sustituye una decisión profesional ni confirma un diagnóstico.")
st.caption("Se conserva la codificación del archivo: Outcome 0 y Outcome 1. "
           "La procedencia y la composición de este conjunto limitan su interpretación clínica.")
pred_tab, eda_tab, metrics_tab = st.tabs(["Predicción", "Exploración", "Evaluación"])

with pred_tab:
    st.subheader("Datos del registro")
    st.write("Introduce valores en la escala del archivo original. Puedes dejar vacías las mediciones "
             "indicadas como opcionales. El modelo las completa con la mediana de entrenamiento.")
    st.caption("El notebook no documenta las unidades originales de todas las mediciones. "
               "Usa registros del dataset para esta demostración.")
    with st.form("registro"):
        columns = st.columns(2)
        values = {}
        for index, (name, spec) in enumerate(report["fields"].items()):
            with columns[index % 2]:
                values[name] = st.text_input(
                    LABELS[name], value=str(spec["default"]), key=name,
                    help=f"Rango observado: {spec['min']} a {spec['max']}. "
                         + ("Opcional. Vacío o cero representa medición ausente." if spec["optional"] else "Obligatorio.")
                )
        submitted = st.form_submit_button("Clasificar registro", type="primary")
    if submitted:
        errors = []
        for name, spec in report["fields"].items():
            raw = values[name].strip().replace(",", ".")
            if spec["optional"] and not raw:
                values[name] = np.nan
                continue
            try:
                value = float(raw)
                if spec["optional"] and value == 0:
                    values[name] = np.nan
                    continue
                if not spec["min"] <= value <= spec["max"]:
                    raise ValueError
                if spec["integer"] and not value.is_integer():
                    raise ValueError
                values[name] = value
            except ValueError:
                errors.append(f"Revisa {LABELS[name]}. Introduce un valor "
                              f"{'entero ' if spec['integer'] else ''}"
                              f"entre {spec['min']} y {spec['max']}.")
        if errors:
            for error in errors:
                st.error(error)
        else:
            model = load_model()
            row = pd.DataFrame([values], columns=list(report["fields"]))
            prediction = int(model.predict(row)[0])
            probs = model.predict_proba(row)[0]
            st.success(f"Clasificación estimada: Outcome {prediction}")
            for column, label, probability in zip(st.columns(2), model.classes_, probs):
                column.metric(f"Probabilidad de Outcome {label}", f"{probability:.1%}")
            st.caption("Probabilidades del clasificador, sin calibración clínica.")

with eda_tab:
    st.subheader("Distribución de Outcome")
    st.bar_chart(pd.Series(report["distribution"], name="Registros"))
    st.caption("Se muestran los 744 registros después de retirar duplicados.")
    st.subheader("Glucosa mediana por clase")
    st.bar_chart(pd.Series(report["glucose"], name="Glucosa"))
    st.caption("Comparación de las clases tras tratar los ceros como mediciones ausentes.")
    st.subheader("Mediciones ausentes")
    st.bar_chart(pd.Series(report["missing"], name="Registros con medición ausente"))
    st.caption("La imputación se ajusta únicamente con entrenamiento, dentro del pipeline.")

with metrics_tab:
    st.subheader("Evaluación del modelo publicado")
    st.write(report["model"])
    st.caption(f"{report['train_count']} registros de entrenamiento y {report['test_count']} de prueba.")
    metrics = report["metrics"]
    for col, name, value in zip(st.columns(4), ["Accuracy", "Precisión clase 1", "Recall clase 1", "F1 clase 1"],
                                [metrics["accuracy"], metrics["1"]["precision"], metrics["1"]["recall"], metrics["1"]["f1-score"]]):
        col.metric(name, f"{value:.1%}")
    st.write(f"F1 de validación cruzada en entrenamiento: {report['cv_mean']:.3f} ± "
             f"{report['cv_std']:.3f}, cinco particiones.")
    st.subheader("Matriz de confusión")
    st.dataframe(pd.DataFrame(report["matrix"], index=["Real 0", "Real 1"],
                              columns=["Predicho 0", "Predicho 1"]))
    st.write(f"Hay {report['matrix'][1][0]} falsos negativos y {report['matrix'][0][1]} falsos positivos.")
    st.subheader("Resultados por clase")
    st.dataframe(pd.DataFrame({k: metrics[k] for k in ["0", "1"]}).T.round(3))
    st.subheader("Comparación antes y después")
    before = report["original_metrics"]
    st.dataframe(pd.DataFrame([
        {"Versión": "Logística del segundo corte", "Accuracy": before["accuracy"], **before["1"]},
        {"Versión": "Modelo publicado", "Accuracy": metrics["accuracy"], **metrics["1"]},
    ]).round(3), hide_index=True)
    st.subheader("Selección con validación cruzada")
    st.dataframe(pd.DataFrame(report["comparison"]).round(3), hide_index=True)
    st.write("Se eligió el mayor F1 medio en entrenamiento. El conjunto de prueba ya se había "
             "evaluado en el segundo avance. Sus resultados son una referencia comparativa, "
             "con una validación externa aún pendiente.")
    st.subheader("Limitaciones")
    st.write("Se eliminaron 1.256 duplicados exactos. Faltan muchas mediciones de insulina y pliegue "
             "cutáneo. El archivo sustituyó al del primer avance y su procedencia necesita "
             "documentación adicional. El modelo comete errores y carece de validación clínica.")
