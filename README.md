# Olist — Por qué **no** sirve para churn (y qué sí se puede hacer)

> **Decisión del proyecto:** con este dataset **no es adecuado** modelar *churn real de clientes*.  

## 📋 Descripción

Evalué el dataset público de [Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) para predecir *churn*. Conclusión clara: **no hay evento de baja**, hay **muy poca recompra** y existe **censura temporal** fuerte.  
Este repo deja un pipeline limpio (sin fuga temporal) y dos rutas útiles:

1) **Propensión a segunda compra** en H días (90/120/180) — *ejercicio académico válido*.  
2) **Problemas operativos** (retraso/cancelación/review negativa) — *valor de negocio real*.

## 🧪 Conclusión del EDA

> Tras revisar el procedimiento, este dataset no es viable para realizar una prediccion de Churn, ya que la mayoria de clientes que tenmos (97%) unicamente realiza una compra.  
> Con esto doy el estudio de churn para este dataset por cerrado

## ❌ Por qué NO aplica a churn de clientes

- **Base rate mínimo:** ≈2–4% de clientes recompran → problema ultra-desbalanceado, PR-AUC base bajísima.  
- **Censura por derecha:** muchas primeras compras quedan cerca del fin del periodo → “no recompra” ≠ “abandonó”.  
- **Sin etiqueta de baja:** no existe “churn” explícito; solo historiales de pedidos.  
- **Riesgo de fuga temporal:** fácil contaminar con fechas de entrega/reviews posteriores a la compra.  
- **Poca historia por cliente:** mayoría con 1 pedido → poca señal conductual.

## ✅ Qué SÍ se puede modelar con Olist

- **Propensión a segunda compra** (clasificación): P(recompra en H días).  
  *Métricas:* PR-AUC, Lift@k. *Uso:* priorizar campañas baratas.
- **Retraso de entrega** (clasificación) y **tiempo de entrega** (regresión).  
  *Uso:* SLA, promesas realistas, alertas.
- **Cancelación de pedido/ítem** (clasificación).  
  *Uso:* verificación de stock, doble confirmación, mitigación.
- **Review negativa (≤3)** (clasificación pre-entrega).  
  *Uso:* comunicación proactiva.
- **CLV básico** (BG/NBD + Gamma-Gamma).  
  *Uso:* ranking de valor esperado con datos transaccionales.


## 📁 Estructura

```
Churn-Prediction/
├── data/
│   ├── raw/                 # Dataset Olist original
│   └── processed/           # Datos procesados
├── scripts/
│   ├── utils.py             # Utilidades EDA
│   ├── data_preprocess.ipynb
│   ├── viability_check.ipynb           # (opcional) chequeos de viabilidad y censura
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
## 📝 Notebooks

- `data_preprocess.ipynb`: Limpieza inicial (versión original).  
- `preprocess_items.ipynb`: Procesamiento de productos.  
- `preprocess_payments.ipynb`: Análisis de pagos.  
- `preprocess_reviews.ipynb`: Procesamiento de reseñas.  
- `EDA.ipynb`: Análisis exploratorio.  
- `viability_check.ipynb` *(opcional)*: % de recompra, censura efectiva, baseline esperado, recomendación de seguir o no.

---

*Dataset: [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)*
