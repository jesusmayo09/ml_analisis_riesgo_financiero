# ML para Analisis de Riesgo Financiero
## Regresion Logistica Multivariable y Preprocesamiento

### Reto del Lab

Modifica el codigo en tu Colab para responder a las siguientes preguntas en tu informe de laboratorio:

1. Analis de Coeficientes: Segun el DataFrame 'coeficientes' impreso en el Paso 3, Cual variable penaliza mas la solicitud del credito (coeficiente negativo mas alto)? Tiene sentido financiero?

2. Prediccion Personalizada: Crea un cliente nuevo con tus propios datos (inventa el Score y el Monto). Recuerda aplicar el `scaler.transform()` a tu cliente antes de usar `modelo.predict_proba()`. Cual es la probabilidad de aprobacion?

3. Etica en IA: Vuelve al Paso 1 y elemina la variable 'Estrato' de la matriz X. Vuelve a entrenar el modelo y responde a las preguntas: Baja drasticamente el 'accuracy' del model? Si no baja mucho, deberia el banco eleminar esta variable por razones eticas?
