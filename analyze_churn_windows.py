import pandas as pd
import numpy as np
from pathlib import Path

# Load data
BASE_DIR = Path.cwd()
DATA_DIR = (BASE_DIR / "data").resolve()
df = pd.read_csv(DATA_DIR/"processed/processed_dataset.csv")

# Convert timestamps
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])

print("=== ANÁLISIS DE VENTANAS DE CHURN ===")
print(f"Rango temporal: {df['order_purchase_timestamp'].min()} a {df['order_purchase_timestamp'].max()}")
print(f"Duración total: {(df['order_purchase_timestamp'].max() - df['order_purchase_timestamp'].min()).days} días")
print(f"Clientes únicos: {df['customer_id'].nunique():,}")

# Analyze purchase patterns
last_purchase = df.groupby('customer_id')['order_purchase_timestamp'].max()
purchase_counts = df.groupby('customer_id').size()

print(f"\n=== PATRONES DE COMPRA ===")
print(f"Clientes con 1 compra: {(purchase_counts == 1).sum():,} ({(purchase_counts == 1).mean()*100:.1f}%)")
print(f"Clientes con 2+ compras: {(purchase_counts > 1).sum():,} ({(purchase_counts > 1).mean()*100:.1f}%)")

# Analyze time between purchases for repeat customers
multi_purchase = df.groupby('customer_id').agg({
    'order_purchase_timestamp': ['min', 'max', 'count']
}).reset_index()
multi_purchase.columns = ['customer_id', 'first_purchase', 'last_purchase', 'purchase_count']
multi_purchase = multi_purchase[multi_purchase['purchase_count'] > 1]

if len(multi_purchase) > 0:
    multi_purchase['days_between'] = (multi_purchase['last_purchase'] - multi_purchase['first_purchase']).dt.days
    print(f"\n=== CLIENTES CON MÚLTIPLES COMPRAS ===")
    print(f"Promedio días entre primera y última compra: {multi_purchase['days_between'].mean():.1f}")
    print(f"Mediana días entre compras: {multi_purchase['days_between'].median():.1f}")
    print(f"75% de clientes repiten en: {multi_purchase['days_between'].quantile(0.75):.0f} días")
    print(f"90% de clientes repiten en: {multi_purchase['days_between'].quantile(0.90):.0f} días")

# Test different churn windows
print(f"\n=== ANÁLISIS DE VENTANAS DE CHURN ===")
windows = [30, 60, 90, 120, 180, 270, 365]

for days in windows:
    cutoff_date = df['order_purchase_timestamp'].max() - pd.Timedelta(days=days)
    churn_customers = (last_purchase < cutoff_date).sum()
    churn_rate = churn_customers / len(last_purchase)
    
    print(f"{days:3d} días: {churn_rate*100:5.1f}% churn ({churn_customers:,} clientes)")

# Analyze seasonal patterns
df['month'] = df['order_purchase_timestamp'].dt.month
monthly_orders = df.groupby('month').size()
print(f"\n=== PATRONES ESTACIONALES ===")
print("Órdenes por mes:")
for month, count in monthly_orders.items():
    print(f"  Mes {month:2d}: {count:,} órdenes")

# Business context analysis
print(f"\n=== CONTEXTO DE NEGOCIO ===")
print("E-commerce brasileño (Olist) - características típicas:")
print("- Productos variados (no suscripciones)")
print("- Compras ocasionales y estacionales")
print("- Clientes ocasionales vs frecuentes")
print("- Patrones de Black Friday, Navidad, etc.")

print(f"\n=== RECOMENDACIÓN ===")
print("Basado en el análisis:")
print("1. 90 días: Demasiado restrictivo (81.6% churn)")
print("2. 180 días: Balance adecuado (61.1% churn)")
print("3. 365 días: Demasiado permisivo (23.6% churn)")
print("\nRECOMENDACIÓN: 180 días (6 meses)")
print("- Permite compras estacionales")
print("- Identifica clientes realmente perdidos")
print("- Balance entre sensibilidad y especificidad")
print("- Alineado con patrones de e-commerce") 