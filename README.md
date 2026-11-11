# Proyecto de diabetes

**Beny Niebles y Geiser Guerrero · Minería de Datos**

Aplicación académica que continúa `Notebook_Diabetes_Avance2.ipynb`. Clasifica registros mediante Outcome 0 y Outcome 1, con las ocho variables del proyecto de diabetes. El notebook actualizado es `Modelado.ipynb`.

[Abrir la aplicación](https://riesgo-salud-beny-geiser.streamlit.app/)

## Datos y reproducción

El archivo `diabetes-dataset.csv` se recuperó de la fuente registrada en el segundo avance. Coinciden sus 2.000 filas, 9 columnas, 1.256 duplicados, 744 filas únicas y las métricas originales de regresión logística. La procedencia se atribuye al registro del grupo y a la página de descarga. El archivo sustituyó al planteado en el primer avance. [Fuente de Kaggle](https://www.kaggle.com/datasets/vikasukani/diabetes-data-set)

Con Python 3.12, instala las dependencias y ejecuta la aplicación.

```sh
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Para reproducir la revisión y guardar el pipeline completo, ejecuta `python revision.py` o las celdas de `Modelado.ipynb` en orden. La aplicación carga el archivo serializado y los resultados calculados. La regla que trata ceros como ausencias, la imputación, el escalado y el clasificador se guardan en el pipeline.

## Revisión del tercer corte

La regresión logística original tuvo recall de 0,510 y F1 de 0,598 para Outcome 1. Se conservaron la división estratificada original y la semilla 42. Se compararon una línea base mayoritaria, los dos modelos originales y regresiones logísticas con pesos de clase equilibrados y C de 0,1, 1 y 10.

La selección empleó el F1 medio de cinco particiones estratificadas del conjunto de entrenamiento. Se eligió la logística equilibrada con C=10, F1 de validación de 0,662 ± 0,068. La diferencia con las otras variantes equilibradas es pequeña y no demuestra una superioridad estadísticamente concluyente.

| Métrica en prueba | Logística original | Modelo publicado |
| --- | --- | --- |
| Accuracy | 0,765 | 0,745 |
| Precisión de Outcome 1 | 0,722 | 0,618 |
| Recall de Outcome 1 | 0,510 | 0,667 |
| F1 de Outcome 1 | 0,598 | 0,642 |
| Falsos negativos | 25 | 17 |
| Falsos positivos | 10 | 21 |

El ajuste detecta más positivos y aumenta las falsas alarmas. El F1 sigue por debajo de 0,70. La prueba ya había sido evaluada en el segundo avance, por lo que constituye una referencia comparativa y la validación externa continúa pendiente. La aplicación muestra estos resultados y limitaciones.

Se corrigió además la interpretación escrita de las matrices originales. La logística tenía 10 falsos positivos y el bosque 13, coherentes con los reportes guardados, en lugar de los 14 y 16 que decía el texto.

## Limitaciones y uso

Prototipo académico que no sustituye una decisión profesional. Las etiquetas se presentan según la codificación del dataset. Faltan detalles sobre el origen y las unidades de algunas variables. Hay numerosas mediciones ausentes de insulina y pliegue cutáneo. Las probabilidades carecen de calibración clínica. El formulario utiliza la escala del dataset y permite omitir las cinco mediciones que el notebook trata como ausentes.

## Uso de inteligencia artificial

- Herramienta utilizada, OpenAI Codex.
- Propósito, revisar el modelo del segundo corte, adaptar la aplicación, verificar los resultados y desplegarla.
- Prompt principal, «Realiza el trabajo que se solicita, pregúntame lo que necesites saber». La actualización del archivo de referencia se solicitó con «Dale con este que te acabo de enviar», acompañada de `Notebook_Diabetes_Avance2.ipynb`.
- Problemas encontrados, la primera aplicación se había construido con la actividad guiada sobre riesgo sintético. Al recibir el notebook del proyecto se adaptó al objetivo Outcome. Se detectaron conteos de falsos positivos incorrectos en la interpretación escrita y resultados insuficientes en recall y F1.
- Ajustes realizados, incorporación del tratamiento de ceros al pipeline, comparación por validación cruzada, pesos de clase, regularización, validación del formulario y corrección de los conteos.
- Verificación, reproducción de las métricas originales, comparación de matrices, ejecución de pruebas del formulario y comparación de un registro de prueba entre el notebook y el pipeline serializado. La revisión y comprensión por parte de ambos integrantes corresponde al grupo.
