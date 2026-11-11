"""Revisión reproducible del modelo del segundo avance."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from preparacion import ceros_a_ausentes

ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / "diabetes-dataset.csv")
assert df.shape == (2000, 9) and df.duplicated().sum() == 1256
datos = df.drop_duplicates().reset_index(drop=True)
X = datos.drop(columns="Outcome")
y = datos["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
preparacion = [
    ("ceros_ausentes", FunctionTransformer(ceros_a_ausentes)),
    ("imputacion", SimpleImputer(strategy="median")),
    ("escala", StandardScaler()),
]
candidatos = {
    "Línea base mayoritaria": DummyClassifier(strategy="most_frequent"),
    "Logística original": LogisticRegression(max_iter=2000, random_state=42),
    "Bosque original": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
}
for c in [0.1, 1, 10]:
    candidatos[f"Logística equilibrada C={c}"] = LogisticRegression(
        C=c, class_weight="balanced", max_iter=2000, random_state=42
    )
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
comparacion = []
pipelines = {}
for nombre, modelo in candidatos.items():
    pipe = Pipeline(preparacion + [("modelo", modelo)])
    scores = cross_validate(
        clone(pipe), X_train, y_train, cv=cv,
        scoring=["accuracy", "precision", "recall", "f1"], return_train_score=True
    )
    comparacion.append({
        "Modelo": nombre,
        "F1 CV": float(scores["test_f1"].mean()),
        "Desviación F1": float(scores["test_f1"].std()),
        "Recall CV": float(scores["test_recall"].mean()),
        "Precisión CV": float(scores["test_precision"].mean()),
        "Accuracy CV": float(scores["test_accuracy"].mean()),
        "F1 entrenamiento": float(scores["train_f1"].mean()),
    })
    pipelines[nombre] = pipe
mejor = max(comparacion, key=lambda r: (r["F1 CV"], r["Recall CV"]))
final = clone(pipelines[mejor["Modelo"]]).fit(X_train, y_train)
pred = final.predict(X_test)
reporte = classification_report(y_test, pred, output_dict=True, zero_division=0)
original = clone(pipelines["Logística original"]).fit(X_train, y_train)
pred_original = original.predict(X_test)
reporte_original = classification_report(y_test, pred_original, output_dict=True, zero_division=0)
assert round(reporte_original["accuracy"], 3) == 0.765
assert round(reporte_original["1"]["f1-score"], 3) == 0.598
campos = {}
for columna in X.columns:
    serie = ceros_a_ausentes(X_train)[columna].dropna()
    campos[columna] = {
        "min": float(serie.min()), "max": float(serie.max()),
        "default": float(serie.median()),
        "integer": columna in ["Pregnancies", "Age"],
        "optional": columna in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"],
    }
indice = X_test.index[0]
registro = X_test.loc[indice].to_dict()
resultado = {
    "model": mejor["Modelo"], "comparison": comparacion,
    "metrics": reporte, "original_metrics": reporte_original,
    "matrix": confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
    "original_matrix": confusion_matrix(y_test, pred_original, labels=[0, 1]).tolist(),
    "fields": campos,
    "distribution": {str(k): int(v) for k,v in y.value_counts().sort_index().items()},
    "test_count": len(y_test), "train_count": len(y_train),
    "cv_mean": mejor["F1 CV"], "cv_std": mejor["Desviación F1"],
    "missing": ceros_a_ausentes(X).isna().sum().to_dict(),
    "glucose": ceros_a_ausentes(datos).groupby("Outcome")["Glucose"].median().to_dict(),
    "fixture": {"row": registro, "actual": int(y_test.loc[indice]),
                "prediction": int(final.predict(X_test.loc[[indice]])[0]), "index": int(indice)},
    "sha256": hashlib.sha256((ROOT / "diabetes-dataset.csv").read_bytes()).hexdigest(),
}
joblib.dump(final, ROOT / "pipeline.joblib", compress=3)
(ROOT / "resultados.json").write_text(
    json.dumps(resultado, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
)
print(pd.DataFrame(comparacion).round(3).to_string(index=False))
print("Seleccionado", resultado["model"])
print("Prueba", reporte)
print("Matriz", resultado["matrix"])
