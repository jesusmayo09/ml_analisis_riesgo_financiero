# ML para Análisis de Riesgo Financiero
## Regresión Logística Multivariable y Preprocesamiento

![Status](https://img.shields.io/badge/status-active-brightgreen)
![Python](https://img.shields.io/badge/python-3.7+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 📋 Descripción del Proyecto

Este proyecto implementa un modelo de **Regresión Logística Multivariable** para analizar y predecir el riesgo financiero en solicitudes de crédito. El modelo fue entrenado con datos de clientes colombianos y utiliza técnicas de preprocesamiento avanzado incluyendo normalización y codificación de variables categóricas.

### Objetivo Principal
Desarrollar un modelo predictivo que determine la probabilidad de que un cliente pague o incumpla con un crédito, considerando variables como:
- **Score crediticio** (300-950)
- **Ingresos** (en términos de SMMLV)
- **Monto del crédito** (en millones)
- **Edad del cliente**
- **Estrato socioeconómico**
- **Nivel educativo**

### Reto del Lab

Modifica el codigo en tu Colab para responder a las siguientes preguntas en tu informe de laboratorio:

1. Analis de Coeficientes: Segun el DataFrame 'coeficientes' impreso en el Paso 3, Cual variable penaliza mas la solicitud del credito (coeficiente negativo mas alto)? Tiene sentido financiero?

2. Prediccion Personalizada: Crea un cliente nuevo con tus propios datos (inventa el Score y el Monto). Recuerda aplicar el `scaler.transform()` a tu cliente antes de usar `modelo.predict_proba()`. Cual es la probabilidad de aprobacion?

3. Etica en IA: Vuelve al Paso 1 y elemina la variable 'Estrato' de la matriz X. Vuelve a entrenar el modelo y responde a las preguntas: Baja drasticamente el 'accuracy' del model? Si no baja mucho, deberia el banco eleminar esta variable por razones eticas?
