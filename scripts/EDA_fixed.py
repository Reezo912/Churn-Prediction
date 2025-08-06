# Corrección del error en EDA.ipynb
# El problema está en la línea donde se usa graficar_variable_con_target

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
sys.path.append('.')

from utils import graficar_variable_con_target

# Cargar datos
BASE_DIR = Path.cwd().parent
DATA_DIR = (BASE_DIR / "data").resolve()
df = pd.read_csv(DATA_DIR/"processed/processed_dataset.csv")

# Definir variables (como en el notebook)
variables_categoricas = [
    'order_status',
    'payment_type', 
    'customer_city',
    'customer_state' 
]

variables_numericas = [
    'total_payment',
    'max_installments',
    'total_reviews',
    'worst_review',
    'mean_review_score',
    'last_review',
    'total_price',
    'item_count',
    'total_freight_value'
]

# Crear variable churn (como en el notebook)
date_cols = [
    'order_purchase_timestamp', 
    'order_approved_at', 
    'order_delivered_carrier_date', 
    'order_delivered_customer_date', 
    'order_estimated_delivery_date'
]

for col in date_cols:
    df[col] = pd.to_datetime(df[col])

# Crear variable churn
last_purchase = df.groupby('customer_id')['order_purchase_timestamp'].max().reset_index()
cutoff_date = df["order_purchase_timestamp"].max() - pd.Timedelta(days=180)
last_purchase["churn"] = last_purchase["order_purchase_timestamp"] < cutoff_date
df['churn'] = last_purchase["churn"]

# CORRECCIÓN: Usar la sintaxis correcta
print("Analizando variables con el target...")
for i in list(variables_categoricas + variables_numericas): 
    try:
        graficar_variable_con_target(df=df, feature=i, target='churn')
        print(f"✅ Gráfico creado para: {i}")
    except Exception as e:
        print(f"❌ Error con {i}: {e}")

print("\n¡Análisis completado!") 