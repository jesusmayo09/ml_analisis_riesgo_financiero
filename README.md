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
---

## 🎯 Reto del Lab

Modifica el código en tu entorno para responder a las siguientes preguntas en tu informe de laboratorio:

### **1. Análisis de Coeficientes**
> Según el DataFrame `coeficientes` impreso en el Paso 3, **¿Cuál variable penaliza más la solicitud del crédito** (coeficiente negativo más alto)? **¿Tiene sentido financiero?**

**Respuesta esperada:**
La variable que penaliza más es **"Monto_Millones"** con un coeficiente de **-1.32**. Esto tiene sentido financiero porque un monto de crédito demasiado alto (desproporcionado respecto a los ingresos del cliente) incrementa significativamente el riesgo de mora.

### **2. Predicción Personalizada**
> Crea un cliente nuevo con tus **propios datos** (inventa el Score y el Monto). Recuerda aplicar el `scaler.transform()` a tu cliente antes de usar `modelo.predict_proba()`. **¿Cuál es la probabilidad de aprobación?**

**Pasos a seguir:**
1. Define los datos del nuevo cliente en un diccionario
2. Convierte a DataFrame
3. Aplica One-Hot Encoding a variables categóricas
4. Asegúrate de que las columnas coincidan con X_train
5. Escala los datos usando el `scaler` previamente ajustado
6. Utiliza `modelo.predict_proba()` para obtener probabilidades

**Ejemplo de implementación:**
```python
nuevo_cliente_data = {
    'Score': [750],
    'Ingresos_SMMLV': [6.0],
    'Monto_Millones': [30.0],
    'Edad': [45],
    'Estrato': [4],
    'Nivel_Educativo': ['Profesional']
}
nuevo_cliente_df = pd.DataFrame(nuevo_cliente_data)
# ... procesamiento y predicción
```

### **3. Ética en IA**
> Vuelve al Paso 1 y **elimina la variable "Estrato"** de la matriz X. Vuelve a entrenar el modelo y responde:
> 
> **a) ¿Baja drásticamente el "accuracy" del modelo?**
> 
> **b) Si no baja mucho, ¿debería el banco eliminar esta variable por razones éticas?**

**Análisis esperado:**
Al eliminar la variable "Estrato", el accuracy del modelo cambia de 84% a 85%, lo que **no representa una baja significativa**. Esto sugiere que:
- La variable "Estrato" tiene poco impacto en la capacidad predictiva del modelo
- Por razones éticas, podría ser deseable eliminarla para evitar discriminación socioeconómica
- Sin sacrificar significativamente el desempeño, el banco podría implementar una política más justa

---

## 📊 Resultados Esperados

### Métricas del Modelo Base
```
Accuracy:     ~85%
Precision:    ~85-86%
Recall:       ~72-93%
F1-Score:     ~79-88%
```

### Coeficientes Principales
| Variable | Coeficiente | Interpretación |
|----------|------------|----------------|
| Score | Positivo | Mayor score → Mayor probabilidad de pago |
| Ingresos_SMMLV | Positivo | Mayores ingresos → Más confiabilidad |
| Monto_Millones | -1.32 (Negativo) | Montos altos → Mayor riesgo |
| Estrato | Variable | Impacto limitado en predicción |

---

## ⚖️ Consideraciones Éticas

Este proyecto aborda importantes cuestiones sobre equidad en IA:

- **Sesgo algorítmico**: Variables socioeconómicas pueden perpetuar discriminación
- **Fairness**: Es importante evaluar si el modelo afecta desigualmente a grupos demográficos
- **Transparencia**: Las predicciones deben ser interpretables para justificar decisiones
- **Responsabilidad**: El banco debe considerar el impacto social de sus modelos
