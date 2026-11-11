# Nivel de riesgo en datos sintéticos de salud

**Minería de Datos · Beny Niebles y Geiser Guerrero**

Prototipo académico para clasificar un registro simulado como de riesgo bajo, medio o alto. La aplicación presenta tres visualizaciones exploratorias, valida las entradas del formulario y muestra la predicción y las probabilidades por clase. El conjunto proviene del material docente `Dataset_Sintetico_Salud_.xlsx`. Sus registros y etiquetas son sintéticos.

## Ejecutar la aplicación

Con Python 3.12, desde la carpeta del repositorio:

```sh
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

En Streamlit Community Cloud, selecciona el repositorio, la rama `main` y el archivo `app.py`. Usa Python 3.12. La aplicación carga `pipeline.joblib` y muestra las métricas y los agregados de `resultados.json`.

## Modelo y evaluación

Se conserva la regresión logística del notebook de modelado. La división original asigna el 75 % de los registros a entrenamiento y el 25 % a prueba, con estratificación por nivel de riesgo. Se retiran duplicados exactos y se excluye el identificador del paciente. La imputación, el escalado y la codificación categórica se ajustan con los datos de entrenamiento y se guardan junto con el clasificador.

La regresión logística fue seleccionada por su F1 macro, su desempeño por clase y la validación cruzada frente al árbol de decisión. En la aplicación se reporta una validación cruzada de cinco particiones calculada solo con el conjunto de entrenamiento. La matriz de confusión y las demás métricas finales corresponden al conjunto de prueba. El modelo reproduce patrones de una etiqueta sintética y carece de validación clínica.

Para regenerar los archivos, abre `Modelado.ipynb` con el Excel en la misma carpeta y ejecuta las celdas en orden, incluida la sección final **Exportación a Streamlit**.

## Pruebas locales

| Entrada | Resultado observado |
| --- | --- |
| Valores iniciales del formulario | Riesgo Bajo y probabilidades de las tres clases |
| Edad de −1 | La aplicación mostró un error de rango |
| Texto en el campo de edad | La aplicación solicitó un número entero |
| Registro de prueba con índice 314 | Clase real Medio. El notebook y la aplicación predijeron Medio |

El F1 macro en prueba fue de 0,859. En la clase Alta, el modelo detectó 30 de 33 registros. Los tres restantes quedaron en la clase Media.

## Uso de inteligencia artificial

- **Herramienta y propósito.** OpenAI Codex revisó los archivos del curso, reprodujo el clasificador y preparó la interfaz y la documentación del despliegue.
- **Prompt principal.** «Realiza el trabajo que se solicita, pregúntame lo que necesites saber», en referencia a la actividad de despliegue en Streamlit y sus recursos adjuntos.
- **Problema identificado.** El notebook de EDA se adjuntó como posible base del despliegue, pero contiene la exploración de los datos. Se localizó el notebook de modelado que contiene los nombres del grupo y la carga del Excel correspondiente. La coincidencia con la entrega oficial queda sujeta a la revisión del grupo.
- **Ajustes y verificación.** Se conservó el clasificador y se incorporaron el reporte de validación cruzada sobre entrenamiento, la exportación del pipeline y el formulario con restricciones basadas en los datos de entrenamiento. Se ejecutaron las celdas de código y se verificó el comportamiento del formulario con las pruebas indicadas arriba.
