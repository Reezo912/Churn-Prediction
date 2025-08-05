# Churn Prediction - Olist Dataset

> **⚠️ Proyecto en desarrollo**  
> Análisis exploratorio y modelo de predicción de churn usando el dataset de Olist.

## 📋 Descripción

Proyecto de machine learning para predecir el churn de clientes usando el dataset de [Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). El análisis incluye EDA completo, feature engineering y modelado predictivo.

## 🚧 Estado Actual

### ✅ Completado
- [x] Estructura del proyecto
- [x] Scripts de utilidades para EDA (`scripts/utils.py`)
- [x] Notebooks de preprocesamiento de datos
- [x] Análisis exploratorio inicial

### 🔄 En Progreso
- [ ] Feature engineering
- [ ] Desarrollo de modelos
- [ ] Optimización de hiperparámetros

## 📁 Estructura

```
Churn-Prediction/
├── data/
│   ├── raw/           # Dataset Olist original
│   └── processed/     # Datos procesados
├── scripts/
│   ├── utils.py       # Utilidades para EDA
│   ├── data_preprocess.ipynb
│   ├── EDA.ipynb
│   ├── preprocess_items.ipynb
│   ├── preprocess_payments.ipynb
│   └── preprocess_reviews.ipynb
└── requirements.txt
```

## 🛠️ Instalación

```bash
git clone <url-del-repositorio>
cd Churn-Prediction
pip install -r requirements.txt
```

## 📊 Utilidades Disponibles

El módulo `scripts/utils.py` incluye herramientas para EDA:

```python
from scripts.utils import (
    detectar_tipos_columnas,
    graficar_numericas,
    graficar_categoricas,
    graficar_variable_con_target,
    matriz_correlacion_numericas,
    informacion_mutua_con_target
)

# Detectar tipos automáticamente
tipos = detectar_tipos_columnas(df, target='churn')

# Visualizaciones univariadas
graficar_numericas(df, variables_numericas)

# Análisis bivariado
graficar_variable_con_target(df, 'review_score', 'churn')
```

## 🔧 Características

- **Detección automática de tipos**: Clasifica variables como ordinal, discreta o continua
- **Manejo inteligente de outliers**: Filtra automáticamente para mejor visualización
- **Visualizaciones profesionales**: Colores armoniosos y estadísticas integradas
- **Análisis de correlaciones**: Spearman, V de Cramér, Mutual Information

## 📝 Notebooks

- `data_preprocess.ipynb`: Limpieza inicial del dataset Olist
- `preprocess_items.ipynb`: Procesamiento de productos
- `preprocess_payments.ipynb`: Análisis de transacciones
- `preprocess_reviews.ipynb`: Procesamiento de reseñas
- `EDA.ipynb`: Análisis exploratorio completo

## 🚀 Próximos Pasos

1. **Feature Engineering**: Crear variables derivadas del comportamiento de compra
2. **Modelado**: Implementar algoritmos de clasificación
3. **Validación**: Cross-validation y métricas de rendimiento

---

*Dataset: [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)*
