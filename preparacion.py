import numpy as np


def ceros_a_ausentes(datos):
    """Aplica la misma regla de ausencia de medición usada en el segundo avance."""
    datos = datos.copy()
    columnas = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    datos[columnas] = datos[columnas].replace(0, np.nan)
    return datos
