
# -*- coding: utf-8 -*-
"""
utils.py — Helpers de EDA (Exploratory Data Analysis)
=====================================================

Este módulo proporciona herramientas completas para análisis exploratorio de datos:

FUNCIONALIDADES PRINCIPALES:
- Separación automática de tipos de columnas (numéricas, categóricas, fechas, etc.)
- Análisis univariado (variables categóricas y numéricas)
- Análisis bivariado (relación feature ↔ target)
- Análisis de correlaciones, asociaciones, VIF y Mutual Information

CONVENCIONES:
- Todas las funciones de plotting devuelven (fig, axes) y aceptan show=True/False
- NO se usan ejes gemelos (twinx) para evitar confusión
- Histogramas y Boxplots van en ejes separados para mayor claridad
"""

# =============================================================================
# IMPORTS Y CONFIGURACIÓN
# =============================================================================

from __future__ import annotations  # Permite usar tipos como strings
from typing import Iterable, List, Tuple, Optional, Dict

# Librerías principales para análisis de datos
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Librerías opcionales (con manejo de errores)
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor as _vif
    _HAS_STATSMODELS = True
except Exception:
    _HAS_STATSMODELS = False  # Para calcular VIF (Variance Inflation Factor)

try:
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False  # Para calcular Mutual Information

# Funciones de pandas para detectar tipos de datos
from pandas.api.types import (
    is_numeric_dtype, is_bool_dtype, is_datetime64_any_dtype, is_categorical_dtype
)

# =============================================================================
# FUNCIONES UTILITARIAS
# =============================================================================

def _ensure_list(x):
    """
    Convierte cualquier entrada en una lista.
    
    Args:
        x: Puede ser None, lista, tupla, array de numpy, o índice de pandas
        
    Returns:
        list: Lista con los elementos de x
    """
    if x is None: return []
    if isinstance(x, (list, tuple, np.ndarray, pd.Index)): return list(x)
    return [x]

# =============================================================================
# DETECCIÓN AUTOMÁTICA DE TIPOS DE COLUMNAS
# =============================================================================

def detectar_tipos_columnas(
    df: pd.DataFrame,
    target: Optional[str] = None,
    low_card_num_as_cat: int = 15,
    treat_bool_as_cat: bool = True,
    coerce_numeric_objects: bool = True,
    infer_datetimes: bool = True,
    id_name_patterns: Tuple[str, ...] = ("id","uuid","guid","code","cod","dni","nif","zip","postal"),
) -> Dict[str, List[str]]:
    """
    Detecta automáticamente los tipos de columnas en un DataFrame.
    
    Esta función es fundamental para el EDA ya que clasifica las columnas en:
    - Numéricas: Para análisis estadístico y correlaciones
    - Categóricas: Para análisis de frecuencias y asociaciones
    - Fechas: Para análisis temporal
    - IDs: Columnas que no aportan información (se excluyen del análisis)
    
    Args:
        df: DataFrame a analizar
        target: Nombre de la columna objetivo (se excluye de features)
        low_card_num_as_cat: Número máximo de valores únicos para considerar 
                            una variable numérica como categórica
        treat_bool_as_cat: Si True, trata las variables booleanas como categóricas
        coerce_numeric_objects: Si True, intenta convertir objetos a numéricos
        infer_datetimes: Si True, intenta detectar columnas de fechas
        id_name_patterns: Patrones en nombres de columnas que indican IDs
        
    Returns:
        Dict con listas de columnas clasificadas por tipo
    """
    dfw = df.copy()  # Trabajamos en una copia para no modificar el original

    # PASO 1: Intentar convertir objetos a numéricos
    if coerce_numeric_objects:
        for c in dfw.select_dtypes(include=["object"]).columns:
            # Reemplazar comas por puntos (formato europeo)
            s = dfw[c].astype(str).str.replace(",", ".", regex=False)
            conv = pd.to_numeric(s, errors="coerce")
            # Si al menos 95% se convirtió exitosamente, usar el resultado
            if conv.notna().mean() >= 0.95:
                dfw[c] = conv

    # PASO 2: Intentar detectar columnas de fechas
    if infer_datetimes:
        for c in dfw.select_dtypes(include=["object"]).columns:
            try:
                # Intentar convertir a datetime (formato europeo: día/mes/año)
                conv = pd.to_datetime(dfw[c], errors="coerce", utc=False, dayfirst=True)
                # Si al menos 90% se convirtió exitosamente, usar el resultado
                if conv.notna().mean() >= 0.90:
                    dfw[c] = conv
            except Exception:
                pass  # Si falla, continuar con la siguiente columna

    # PASO 3: Clasificar columnas por tipo básico
    numeric = [c for c in dfw.columns if is_numeric_dtype(dfw[c])]      # Variables numéricas
    boolean = [c for c in dfw.columns if is_bool_dtype(dfw[c])]         # Variables booleanas
    dates   = [c for c in dfw.columns if is_datetime64_any_dtype(dfw[c])]  # Variables de fecha
    cats_raw = dfw.select_dtypes(include=["object"]).columns.tolist()    # Objetos (strings)
    cats_cat = [c for c in dfw.columns if is_categorical_dtype(dfw[c])] # Categóricas explícitas
    categorical = sorted(set(cats_raw + cats_cat))  # Unir todas las categóricas

    # PASO 4: Detectar columnas tipo ID (no útiles para análisis)
    n = len(dfw)
    id_like = []
    for c in dfw.columns:
        if c == target: continue  # No excluir el target
        
        # Verificar si el nombre contiene patrones de ID
        name_hit = any(p in c.lower() for p in id_name_patterns)
        
        # Verificar si tiene cardinalidad muy alta (casi única)
        try:
            high_card = dfw[c].nunique(dropna=True) >= 0.98 * n
        except Exception:
            high_card = False
            
        # Si cumple cualquiera de los criterios, es un ID
        if name_hit or high_card:
            id_like.append(c)

    # PASO 5: Refinar clasificación para análisis
    # Variables numéricas con baja cardinalidad → tratarlas como categóricas
    num_low_card = [c for c in numeric if dfw[c].nunique(dropna=True) <= low_card_num_as_cat]
    numeric_for_analysis = [c for c in numeric if c not in num_low_card]  # Resto de numéricas
    
    # Unir todas las categóricas (incluyendo booleanas y numéricas de baja cardinalidad)
    categorical_for_analysis = sorted(set(categorical + (boolean if treat_bool_as_cat else []) + num_low_card))

    # PASO 6: Limpiar listas excluyendo columnas no útiles para análisis
    def _clean(nums, cats):
        """Excluye target, fechas e IDs de las listas de features"""
        nums2 = [c for c in nums if c != target and c not in dates and c not in id_like]
        cats2 = [c for c in cats if c != target and c not in dates and c not in id_like]
        return nums2, cats2

    numeric_features, categorical_features = _clean(numeric_for_analysis, categorical_for_analysis)

    # PASO 7: Preparar resultado final
    out = {
        "numeric_all": numeric,                    # Todas las numéricas (incluyendo IDs)
        "categorical_all": categorical,            # Todas las categóricas
        "boolean": boolean,                        # Variables booleanas
        "dates": dates,                           # Variables de fecha
        "id_like": id_like,                       # Columnas tipo ID
        "numeric_for_analysis": numeric_for_analysis,      # Numéricas para análisis
        "categorical_for_analysis": categorical_for_analysis,  # Categóricas para análisis
        "numeric_features": numeric_features,      # Numéricas limpias (sin target/IDs/fechas)
        "categorical_features": categorical_features,  # Categóricas limpias
    }
    out["features_all"] = out["numeric_features"] + out["categorical_features"]  # Todas las features
    return out

# =============================================================================
# REPORTES RÁPIDOS DE CALIDAD DE DATOS
# =============================================================================

def resumen_nulos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera un resumen de valores nulos por columna.
    
    Args:
        df: DataFrame a analizar
        
    Returns:
        DataFrame con columnas 'n_nulos' y 'pct_nulos', ordenado por % de nulos descendente
    """
    n = len(df)
    return (
        df.isna().sum().to_frame("n_nulos")
          .assign(pct_nulos=lambda x: x["n_nulos"]/n)
          .sort_values("pct_nulos", ascending=False)
    )

def resumen_numericas(df: pd.DataFrame, cols: Optional[Iterable[str]] = None,
                      percentiles=(0.01,0.05,0.25,0.5,0.75,0.95,0.99)) -> pd.DataFrame:
    """
    Genera estadísticas descriptivas para variables numéricas.
    
    Args:
        df: DataFrame a analizar
        cols: Columnas específicas a analizar (si None, usa todas las numéricas)
        percentiles: Lista de percentiles a calcular
        
    Returns:
        DataFrame con estadísticas descriptivas por variable
    """
    cols = _ensure_list(cols) or df.select_dtypes(include="number").columns.tolist()
    percentiles = list(sorted(set(percentiles)))
    return df[cols].describe(percentiles=percentiles).T

def reporte_categorias_raras(df: pd.DataFrame, cat_cols: Iterable[str], threshold: float=0.01) -> pd.DataFrame:
    """
    Identifica categorías con frecuencia muy baja (posibles errores o outliers).
    
    Args:
        df: DataFrame a analizar
        cat_cols: Columnas categóricas a revisar
        threshold: Umbral mínimo de frecuencia (por defecto 1%)
        
    Returns:
        DataFrame con categorías que están por debajo del umbral
    """
    rows = []
    for c in _ensure_list(cat_cols):
        # Calcular frecuencias relativas (incluyendo nulos)
        vc = df[c].value_counts(dropna=False, normalize=True)
        # Identificar categorías con frecuencia menor al umbral
        raras = vc[vc < threshold]
        for k, v in raras.items():
            rows.append({"columna": c, "categoria": k, "freq": v})
    return pd.DataFrame(rows).sort_values(["columna","freq"])

# =============================================================================
# ANÁLISIS UNIVARIADO - VARIABLES CATEGÓRICAS
# =============================================================================

def graficar_categoricas(df, cat_var, figsize=(15, 10), max_categories=20, show_percentages=True):
    """
    Función para graficar múltiples variables categóricas usando un bucle for
    
    Parámetros:
    - df: DataFrame con los datos
    - variables_categoricas: lista de nombres de columnas categóricas
    - figsize: tamaño de la figura completa
    - max_categories: número máximo de categorías a mostrar por variable
    - show_percentages: si mostrar porcentajes además de valores absolutos
    """
    
    # Calcular número de filas y columnas para los subplots
    n_vars = len(cat_var)
    n_cols = 2  # 3 gráficos por fila
    n_rows = (n_vars + n_cols - 1) // n_cols  # Redondear hacia arriba
    
    # Crear la figura con subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Asegurar que axes sea siempre un array 2D
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Definir una paleta de colores
    colores = plt.cm.Set3(range(12))
    
    # Bucle for para crear cada gráfico
    for i, variable in enumerate(cat_var):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Verificar si la variable existe en el DataFrame
        if variable not in df.columns:
            ax.text(0.5, 0.5, f'Variable "{variable}"\nno encontrada', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{variable} (No encontrada)')
            continue
        
        # Contar valores y obtener los más frecuentes
        value_counts = df[variable].value_counts()
        
        # Limitar el número de categorías si es necesario
        if len(value_counts) > max_categories:
            value_counts = value_counts.head(max_categories)
            titulo = f'{variable} (Top {max_categories})'
        else:
            titulo = variable
        
        # Crear gráfico de barras
        bars = ax.bar(range(len(value_counts)), 
                     value_counts.values, 
                     color=colores[i % len(colores)],
                     alpha=0.7,
                     edgecolor='black',
                     linewidth=0.5)
        
        # Configurar el gráfico
        ax.set_title(titulo, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('Categorías', fontsize=10)
        ax.set_ylabel('Frecuencia', fontsize=10)
        
        # Rotar etiquetas del eje x si son muchas o muy largas
        labels = [str(label) for label in value_counts.index]
        if len(labels) > 5 or any(len(str(label)) > 8 for label in labels):
            ax.set_xticks(range(len(value_counts)))
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        else:
            ax.set_xticks(range(len(value_counts)))
            ax.set_xticklabels(labels, fontsize=9)
        
        # Añadir valores en las barras con rotación y porcentajes
        total_count = value_counts.sum()
        
        # Determinar rotación automática basada en el número de categorías
        if len(value_counts) > 8:
            rotation_angle = 45
        elif len(value_counts) > 5:
            rotation_angle = 30
        else:
            rotation_angle = 0
            
        for i, (bar, value) in enumerate(zip(bars, value_counts.values)):
            height = bar.get_height()
            percentage = (value / total_count) * 100
            
            # Texto con valor absoluto y porcentaje
            if show_percentages:
                text = f'{value:,}\n({percentage:.1f}%)'
            else:
                text = f'{value:,}'
            
            # Estrategia inteligente para posicionar texto según el espacio disponible
            bar_width = bar.get_width()
            bar_height = bar.get_height()
            
            # Si hay muchas categorías, usar estrategia más conservadora
            if len(value_counts) > 10:
                # Solo mostrar texto en barras significativas (>5% del total)
                if percentage > 5:
                    # Posicionar texto dentro de la barra si es alta
                    if bar_height > ax.get_ylim()[1] * 0.1:
                        ax.text(bar.get_x() + bar_width/2., bar_height * 0.5,
                               text, ha='center', va='center', fontsize=8, 
                               rotation=rotation_angle, fontweight='bold')
                    else:
                        # Posicionar texto arriba de la barra
                        ax.text(bar.get_x() + bar_width/2., bar_height + bar_height*0.02,
                               text, ha='center', va='bottom', fontsize=8, 
                               rotation=rotation_angle)
                else:
                    # Para barras pequeñas, solo mostrar porcentaje
                    ax.text(bar.get_x() + bar_width/2., bar_height + bar_height*0.01,
                           f'{percentage:.1f}%', ha='center', va='bottom', fontsize=8, 
                           rotation=rotation_angle)
            else:
                # Para pocas categorías, mostrar todo normalmente
                ax.text(bar.get_x() + bar_width/2., bar_height + bar_height*0.01,
                       text, ha='center', va='bottom', fontsize=8, rotation=rotation_angle)
        
        # Mejorar el aspecto del gráfico
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Ocultar subplots vacíos si los hay
    for i in range(n_vars, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    # Ajustar el layout
    plt.tight_layout(pad=2.0)
    plt.show()




import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from collections import Counter

def detectar_tipo_numerico(data, variable_name):
    """
    Detecta automáticamente el tipo de variable numérica.
    
    Esta función clasifica las variables numéricas en:
    - ORDINAL: Variables con valores ordenados (ej: ratings 1-5, escalas)
    - DISCRETA: Variables con valores enteros específicos (ej: número de hijos)
    - CONTINUA: Variables con muchos valores únicos (ej: precios, edades)
    
    Args:
        data: Serie de pandas con los datos
        variable_name: Nombre de la variable (para debugging)
        
    Returns:
        str: Tipo detectado ('ordinal', 'discreta', 'continua', 'sin_datos', 'no_numerica')
    """
    # Limpiar datos nulos
    clean_data = data.dropna()
    
    if len(clean_data) == 0:
        return 'sin_datos'
    
    # Convertir a numérico si es posible
    try:
        clean_data = pd.to_numeric(clean_data)
    except:
        return 'no_numerica'
    
    unique_values = clean_data.nunique()
    total_values = len(clean_data)
    
    # Verificar si son enteros
    es_entero = all(clean_data == clean_data.astype(int))
    
    # Criterios de clasificación
    if unique_values <= 10 and es_entero:
        # Pocos valores únicos y enteros -> Discreta/Ordinal
        if all(val >= 0 for val in clean_data.unique()) and max(clean_data) <= 10:
            return 'ordinal'  # Ej: ratings 1-5, escalas 1-10
        else:
            return 'discreta'  # Ej: número de productos, hijos
    
    elif unique_values / total_values < 0.05 and es_entero:
        # Pocos valores únicos relativos y enteros -> Discreta
        return 'discreta'
    
    elif es_entero and unique_values < 50:
        # Enteros con valores moderados -> Discreta
        return 'discreta'
    
    # Caso especial para variables float que representan ratings/escalas
    elif not es_entero and unique_values <= 20:
        # Verificar si es una escala de rating (0-5, 0-10, etc.)
        min_val = clean_data.min()
        max_val = clean_data.max()
        
        # Si está en un rango típico de ratings y tiene pocos valores únicos
        if (min_val >= 0 and max_val <= 10 and unique_values <= 20):
            # Verificar si los valores están espaciados de manera típica de ratings
            sorted_vals = sorted(clean_data.unique())
            if len(sorted_vals) <= 20:  # Máximo 20 valores únicos para ratings
                return 'ordinal'
    
    # Muchos valores, probablemente decimales -> Continua
    return 'continua'

def graficar_numericas(df, variables_numericas, auto_detect=True, 
                      tipos_especificos=None, figsize=(16, 12)):
    """
    Función para graficar variables numéricas según su tipo específico.
    
    Esta función es la versión mejorada que:
    - Detecta automáticamente el tipo de variable (ordinal, discreta, continua)
    - Aplica visualizaciones específicas según el tipo
    - Maneja outliers inteligentemente para variables continuas
    - Muestra histograma y boxplot lado a lado (sin superposición)
    - Usa colores profesionales y estadísticas en la derecha
    
    Args:
        df: DataFrame con los datos
        variables_numericas: Lista de nombres de columnas numéricas
        auto_detect: Si True, detecta automáticamente el tipo de variable
        tipos_especificos: Dict con tipos específicos {'variable': 'tipo'}
        figsize: Tamaño de la figura
        
    Returns:
        None (muestra los gráficos)
    """
    
    # Detectar tipos si no se especifican
    if tipos_especificos is None:
        tipos_especificos = {}
    
    tipos_detectados = {}
    for var in variables_numericas:
        if var in df.columns:
            if var in tipos_especificos:
                tipos_detectados[var] = tipos_especificos[var]
            elif auto_detect:
                tipos_detectados[var] = detectar_tipo_numerico(df[var], var)
            else:
                tipos_detectados[var] = 'continua'  # Por defecto
    
    # Mostrar tipos detectados
    print("🔍 TIPOS DE VARIABLES DETECTADOS:")
    print("="*60)
    for var, tipo in tipos_detectados.items():
        print(f"📊 {var}: {tipo.upper()}")
    print()
    
    # Calcular layout
    n_vars = len(variables_numericas)
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Asegurar que axes sea array 2D
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Paleta de colores mejorada y coherente
    colores = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3A1772', 
               '#6B5B95', '#88B04B', '#F7CAC9', '#92A8D1', '#955251']
    
    # Bucle for principal para cada variable
    for i, variable in enumerate(variables_numericas):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        if variable not in df.columns:
            ax.text(0.5, 0.5, f'Variable "{variable}"\nno encontrada', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{variable} (No encontrada)')
            continue
        
        data = df[variable].dropna()
        if len(data) == 0:
            ax.text(0.5, 0.5, f'Variable "{variable}"\nsin datos válidos', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{variable} (Sin datos)')
            continue
        
        # Convertir a numérico
        try:
            data = pd.to_numeric(data)
        except:
            ax.text(0.5, 0.5, f'Variable "{variable}"\nno es numérica', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{variable} (No numérica)')
            continue
        
        tipo_var = tipos_detectados.get(variable, 'continua')
        color = colores[i % len(colores)]
        
        # Graficar según el tipo de variable
        if tipo_var == 'ordinal':
            # VARIABLES ORDINALES: Gráfico de barras ordenado
            value_counts = data.value_counts().sort_index()  # Ordenar por valor
            
            bars = ax.bar(value_counts.index, value_counts.values, 
                         color=color, alpha=0.8, edgecolor='white', linewidth=1)
            
            # Añadir valores en las barras
            for bar, value in zip(bars, value_counts.values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value}', ha='center', va='bottom', fontsize=9)
            
            ax.set_title(f'ORDINAL - {variable}', fontweight='bold', color='#8B5A96', fontsize=11)
            ax.set_xlabel(f'{variable} (valores ordenados)')
            ax.set_ylabel('Frecuencia')
            ax.grid(axis='y', alpha=0.3)
            
            # Estadísticas relevantes para ordinales
            moda = data.mode().iloc[0] if len(data.mode()) > 0 else 'N/A'
            mediana = data.median()
            ax.text(0.98, 0.98, f'Moda: {moda}\nMediana: {mediana}', 
                   transform=ax.transAxes, va='top', ha='right', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='#E6F3FF', alpha=0.9, 
                            edgecolor='#8B5A96', linewidth=0.5))
        
        elif tipo_var == 'discreta':
            # VARIABLES DISCRETAS: Gráfico de barras + estadísticas
            value_counts = data.value_counts().sort_index()
            
            # Limitar a los 20 valores más frecuentes si hay muchos
            if len(value_counts) > 20:
                value_counts = data.value_counts().head(20)  # Los más frecuentes
                titulo_extra = f" (Top 20 de {data.nunique()} valores)"
            else:
                titulo_extra = f" ({data.nunique()} valores únicos)"
            
            bars = ax.bar(range(len(value_counts)), value_counts.values,
                         color=color, alpha=0.8, edgecolor='white', linewidth=1)
            
            ax.set_xticks(range(len(value_counts)))
            ax.set_xticklabels([str(x) for x in value_counts.index], rotation=45)
            
            # Añadir valores en barras principales
            for bar, value in zip(bars, value_counts.values):
                if value > max(value_counts.values) * 0.05:  # Solo mostrar valores significativos
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{value}', ha='center', va='bottom', fontsize=8)
            
            ax.set_title(f'DISCRETA - {variable}{titulo_extra}', fontweight='bold', color='#2E8B57', fontsize=11)
            ax.set_xlabel(f'{variable}')
            ax.set_ylabel('Frecuencia')
            ax.grid(axis='y', alpha=0.3)
            
            # Estadísticas para discretas
            media = data.mean()
            std = data.std()
            ax.text(0.98, 0.98, f'Media: {media:.2f}\nStd: {std:.2f}\nMín: {data.min()}\nMáx: {data.max()}', 
                   transform=ax.transAxes, va='top', ha='right', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='#E8F5E8', alpha=0.9, 
                            edgecolor='#2E8B57', linewidth=0.5))
        
        elif tipo_var == 'continua':
            # VARIABLES CONTINUAS: Histograma y boxplot LADO A LADO
            
            # Detectar si hay outliers o cola larga
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Determinar si hay outliers significativos
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            outlier_percentage = len(outliers) / len(data) * 100
            
            # Calcular percentiles para determinar el rango de visualización
            p95 = data.quantile(0.95)
            
            # Si hay muchos outliers (>5%), usar percentil 95 como límite superior
            if outlier_percentage > 5:
                max_visible = p95
                titulo_extra = " (sin outliers extremos)"
            else:
                max_visible = data.max()
                titulo_extra = ""
            
            # Filtrar datos para el histograma principal
            data_filtered = data[data <= max_visible]
            
            # Dividir el espacio: 75% histograma, 25% boxplot
            pos = ax.get_position()
            
            # Área del histograma (75% del ancho)
            hist_width = pos.width * 0.75
            ax_hist = fig.add_axes([pos.x0, pos.y0, hist_width, pos.height])
            
            # Área del boxplot (25% del ancho, al lado derecho)
            box_width = pos.width * 0.25
            ax_box = fig.add_axes([pos.x0 + hist_width, pos.y0, box_width, pos.height])
            
            # Ocultar el eje original
            ax.set_visible(False)
            
            # HISTOGRAMA (lado izquierdo)
            n_bins = min(30, int(np.sqrt(len(data_filtered))))
            n, bins, patches = ax_hist.hist(data_filtered, bins=n_bins, density=True, alpha=0.7,
                                          color=color, edgecolor='white', linewidth=1, orientation='vertical')
            
            # Curva de densidad
            try:
                from scipy import stats
                density = stats.gaussian_kde(data_filtered)
                xs = np.linspace(data_filtered.min(), data_filtered.max(), 200)
                ax_hist.plot(xs, density(xs), color='#DC2626', linewidth=2.5, 
                           label='Densidad', alpha=0.8)
            except:
                pass
            
            # Líneas estadísticas en el histograma
            media = data.mean()
            mediana = data.median()
            
            ax_hist.axvline(media, color='#DC2626', linestyle='--', alpha=0.8, 
                          linewidth=2, label=f'Media: {media:.2f}')
            ax_hist.axvline(mediana, color='#2563EB', linestyle='--', alpha=0.8, 
                          linewidth=2, label=f'Mediana: {mediana:.2f}')
            
            # BOXPLOT (lado derecho, vertical)
            bp = ax_box.boxplot(data_filtered, vert=True, patch_artist=True, 
                              showfliers=True, widths=0.6)
            
            # Colorear el boxplot
            bp['boxes'][0].set_facecolor(color)
            bp['boxes'][0].set_alpha(0.8)
            bp['medians'][0].set_color('#1E3A8A')
            bp['medians'][0].set_linewidth(2)
            
            # Configurar histograma
            ax_hist.set_title(f'CONTINUA - {variable}{titulo_extra}', 
                            fontweight='bold', color='#1E3A8A', fontsize=11)
            ax_hist.set_xlabel(f'{variable}')
            ax_hist.set_ylabel('Densidad')
            ax_hist.legend(fontsize=8, loc='upper center')
            ax_hist.grid(alpha=0.3)
            
            # Configurar boxplot
            ax_box.set_title('Boxplot', fontsize=9, color='#1E3A8A')
            ax_box.set_ylabel('')
            ax_box.set_xlabel('')
            ax_box.set_xticks([])
            ax_box.grid(alpha=0.3, axis='y')
            
            # Asegurar que ambos ejes tengan el mismo rango Y
            hist_ylim = ax_hist.get_ylim()
            data_min, data_max = data_filtered.min(), data_filtered.max()
            ax_box.set_ylim([data_min, data_max])
            
            # Estadísticas detalladas
            std = data.std()
            skewness = data.skew()
            
            if outlier_percentage > 5:
                stats_text = f'Media: {media:.3f}\nStd: {std:.3f}\nAsimetría: {skewness:.3f}\nOutliers: {outlier_percentage:.1f}%'
            else:
                stats_text = f'Media: {media:.3f}\nStd: {std:.3f}\nAsimetría: {skewness:.3f}'
            
            ax_hist.text(0.02, 0.98, stats_text, 
                       transform=ax_hist.transAxes, va='top', ha='left', fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='#F0F8FF', alpha=0.9, 
                                edgecolor='#1E3A8A', linewidth=0.5))
        
        # Para variables no continuas, mejorar aspecto
        if tipo_var != 'continua':
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(0.5)
            ax.spines['bottom'].set_linewidth(0.5)
    
    # Ocultar subplots vacíos (solo para no-continuas)
    for i in range(n_vars, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        if axes[row, col].get_visible():  # Solo si no es una variable continua
            axes[row, col].set_visible(False)
    
    # Mejorar el estilo general
    plt.style.use('default')
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    
    #plt.tight_layout(pad=3.0)
    plt.show()

def analisis_tipos_numericos(df, variables_numericas):
    """
    Función para analizar en detalle los tipos de variables numéricas
    """
    print("🔬 ANÁLISIS DETALLADO DE TIPOS DE VARIABLES NUMÉRICAS")
    print("="*80)
    
    for variable in variables_numericas:
        if variable not in df.columns:
            print(f"\n❌ {variable}: No encontrada en el DataFrame")
            continue
        
        data = df[variable].dropna()
        if len(data) == 0:
            print(f"\n❌ {variable}: Sin datos válidos")
            continue
        
        try:
            data = pd.to_numeric(data)
        except:
            print(f"\n❌ {variable}: No es numérica")
            continue
        
        tipo_detectado = detectar_tipo_numerico(df[variable], variable)
        
        print(f"\n📊 Variable: {variable}")
        print(f"🔍 Tipo detectado: {tipo_detectado.upper()}")
        print("-" * 50)
        
        print(f"Total valores: {len(data):,}")
        print(f"Valores únicos: {data.nunique():,}")
        print(f"% valores únicos: {(data.nunique()/len(data)*100):.2f}%")
        print(f"Rango: {data.min()} - {data.max()}")
        print(f"¿Son enteros?: {'Sí' if all(data == data.astype(int)) else 'No'}")
        
        # Mostrar valores más frecuentes
        top_values = data.value_counts().head(5)
        print(f"Top 5 valores más frecuentes:")
        for val, count in top_values.items():
            print(f"  {val}: {count} veces ({count/len(data)*100:.1f}%)")
        
        # Recomendaciones según el tipo
        if tipo_detectado == 'ordinal':
            print("💡 Recomendación: Usar gráficos de barras ordenados, calcular moda y mediana")
        elif tipo_detectado == 'discreta':
            print("💡 Recomendación: Usar gráficos de barras, analizar distribución de frecuencias")
        elif tipo_detectado == 'continua':
            print("💡 Recomendación: Usar histogramas, boxplots, analizar distribución y outliers")


# =============================================================================
# ANÁLISIS BIVARIADO - RELACIÓN FEATURE ↔ TARGET
# =============================================================================

def graficar_variable_con_target(
    df: pd.DataFrame, feature: str, target: str,
    problema: str = "auto", max_categories: int = 20, show: bool = True,
):
    """
    Grafica la relación entre una feature y el target.
    
    Esta función detecta automáticamente el tipo de problema y elige la visualización
    más apropiada:
    
    CLASIFICACIÓN:
    - Feature numérica → Boxplot horizontal
    - Feature categórica → Gráfico de barras apiladas
    
    REGRESIÓN:
    - Feature numérica → Scatter plot
    - Feature categórica → Boxplot vertical
    
    Args:
        df: DataFrame con los datos
        feature: Nombre de la columna feature
        target: Nombre de la columna target
        problema: 'clasificacion', 'regresion', o 'auto' (detecta automáticamente)
        max_categories: Máximo número de categorías a mostrar
        show: Si True, muestra el gráfico
        
    Returns:
        tuple: (fig, ax) para manipulación posterior
    """
    if feature not in df.columns or target not in df.columns:
        raise ValueError("feature o target no están en el DataFrame.")
    s_feat, s_tgt = df[feature], df[target]

    def _is_categorical(s):
        return s.dtype.name in ("object", "category") or s.nunique(dropna=True) <= 15

    if problema == "auto":
        problema = "clasificacion" if _is_categorical(s_tgt) else "regresion"

    fig, ax = plt.subplots(figsize=(8, 4.2))

    if problema == "clasificacion":
        if is_numeric_dtype(s_feat):
            sns.boxplot(data=df, x=feature, y=target, whis=1.5, ax=ax, orient="h")
            ax.set_title(f"{feature} por {target}"); ax.grid(alpha=0.3, axis="x")
        else:
            ct = (
                df[[feature, target]].dropna().value_counts(normalize=True)
                  .rename("prop").reset_index()
            )
            if max_categories and df[feature].nunique(dropna=True) > max_categories:
                top = df[feature].value_counts().index[:max_categories]
                ct = ct[ct[feature].isin(top)]
            sns.barplot(data=ct, x="prop", y=feature, hue=target, ax=ax, orient="h")
            ax.set_title(f"Proporciones de {feature} por {target}")
            ax.grid(alpha=0.3, axis="x"); ax.legend(title=target, bbox_to_anchor=(1.02,1), loc="upper left")
    else:
        if is_numeric_dtype(s_feat):
            ax.scatter(s_feat, s_tgt, alpha=0.6, edgecolors="none")
            ax.set_xlabel(feature); ax.set_ylabel(target)
            ax.set_title(f"{feature} vs {target}"); ax.grid(alpha=0.3)
        else:
            s = df.copy()
            if max_categories and s[feature].nunique(dropna=True) > max_categories:
                top = s[feature].value_counts().index[:max_categories]
                s = s[s[feature].isin(top)]
            sns.boxplot(data=s, x=target, y=feature, whis=1.5, ax=ax, orient="h")
            ax.set_title(f"{target} por {feature}"); ax.grid(alpha=0.3, axis="x")

    if show: plt.show()
    return fig, ax

# =============================================================================
# ANÁLISIS DE CORRELACIONES, ASOCIACIONES, VIF Y MUTUAL INFORMATION
# =============================================================================

def matriz_correlacion_numericas(df: pd.DataFrame, cols: Optional[Iterable[str]]=None, metodo: str="spearman") -> pd.DataFrame:
    """
    Calcula la matriz de correlación para variables numéricas.
    
    Args:
        df: DataFrame con los datos
        cols: Columnas específicas a analizar (si None, usa todas las numéricas)
        metodo: Método de correlación ('pearson', 'spearman', 'kendall')
        
    Returns:
        DataFrame con la matriz de correlación
    """
    cols = _ensure_list(cols) or df.select_dtypes(include="number").columns.tolist()
    return df[cols].corr(method=metodo)

def _cramers_v_corrected(confusion: pd.DataFrame) -> float:
    """
    Calcula el coeficiente V de Cramér corregido para medir asociación entre categóricas.
    
    Esta es una versión corregida que maneja mejor casos extremos y bias.
    
    Args:
        confusion: Matriz de contingencia (crosstab)
        
    Returns:
        float: Valor del coeficiente V de Cramér (0-1)
    """
    chi2 = stats.chi2_contingency(confusion.values)[0]
    n = confusion.sum().sum(); r, k = confusion.shape
    phi2 = chi2 / n
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1); kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / max(1e-12, min((kcorr-1), (rcorr-1))))

def matriz_asociacion_categoricas(df: pd.DataFrame, cols: Optional[Iterable[str]]=None) -> pd.DataFrame:
    """
    Calcula la matriz de asociación para variables categóricas usando V de Cramér.
    
    Args:
        df: DataFrame con los datos
        cols: Columnas específicas a analizar (si None, usa todas las categóricas)
        
    Returns:
        DataFrame con la matriz de asociación (valores 0-1)
    """
    cols = _ensure_list(cols) or df.select_dtypes(include=["object","category","bool"]).columns.tolist()
    cols = [c for c in cols if df[c].nunique(dropna=True) > 1]  # Excluir constantes
    mat = pd.DataFrame(index=cols, columns=cols, dtype=float)
    
    # Calcular asociaciones por pares
    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if i > j: continue  # Evitar cálculos duplicados
            if i == j: mat.loc[c1, c2] = 1.0  # Diagonal = 1
            else:
                confusion = pd.crosstab(df[c1], df[c2])
                mat.loc[c1, c2] = mat.loc[c2, c1] = _cramers_v_corrected(confusion)
    return mat

def calcular_vif(df: pd.DataFrame, cols: Optional[Iterable[str]]=None) -> pd.DataFrame:
    """
    Calcula el Variance Inflation Factor (VIF) para detectar multicolinealidad.
    
    El VIF mide cuánto se infla la varianza de un coeficiente debido a la correlación
    con otras variables. Valores > 5-10 indican problemas de multicolinealidad.
    
    Args:
        df: DataFrame con los datos
        cols: Columnas específicas a analizar (si None, usa todas las numéricas)
        
    Returns:
        DataFrame con VIF por variable, ordenado descendente
        
    Raises:
        ImportError: Si statsmodels no está disponible
        ValueError: Si no hay suficientes columnas numéricas
    """
    cols = _ensure_list(cols) or df.select_dtypes(include="number").columns.tolist()
    X = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if X.empty or len(X.columns) < 2:
        raise ValueError("Se necesitan >= 2 columnas numéricas sin NaN para VIF.")
    
    # Estandarizar variables
    X = (X - X.mean()) / X.std(ddof=0)
    
    if not _HAS_STATSMODELS:
        raise ImportError("statsmodels no está disponible para calcular VIF.")
    
    # Import local por seguridad
    from statsmodels.stats.outliers_influence import variance_inflation_factor as _vif
    vif_values = [{"variable": c, "VIF": float(_vif(X.values, i))} for i, c in enumerate(X.columns)]
    return pd.DataFrame(vif_values).sort_values("VIF", ascending=False).reset_index(drop=True)

def informacion_mutua_con_target(df: pd.DataFrame, features: Iterable[str], target: str, problema: str="auto") -> pd.DataFrame:
    """
    Calcula la Mutual Information (MI) entre features y target.
    
    La MI mide la dependencia estadística entre variables, capturando relaciones
    tanto lineales como no lineales. Es muy útil para feature selection.
    
    Args:
        df: DataFrame con los datos
        features: Lista de features a analizar
        target: Variable objetivo
        problema: 'clasificacion', 'regresion', o 'auto' (detecta automáticamente)
        
    Returns:
        DataFrame con MI por feature, ordenado descendente
        
    Raises:
        ImportError: Si scikit-learn no está disponible
    """
    if not _HAS_SKLEARN:
        raise ImportError("scikit-learn no disponible: instala 'scikit-learn'.")
    
    # Filtrar features válidas
    features = [f for f in _ensure_list(features) if f in df.columns and f != target]
    
    # Preparar datos: one-hot encoding para categóricas
    X = pd.get_dummies(df[features].copy(), drop_first=False)
    y = df[target].copy()
    
    # Limpieza básica de datos
    X = X.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(method="bfill")
    y = y.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Alinear índices
    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]
    
    # Detectar tipo de problema
    if problema == "auto":
        problema = "clasificacion" if y.dtype.name in ("object","category") or y.nunique(dropna=True) <= 15 else "regresion"
    
    # Calcular MI según el tipo de problema
    if problema == "clasificacion":
        if y.dtype.name in ("object","category"): 
            y, _ = pd.factorize(y)  # Convertir categóricas a numéricas
        scores = mutual_info_classif(X, y, discrete_features='auto', random_state=0)
    else:
        scores = mutual_info_regression(X, y, discrete_features='auto', random_state=0)
    
    return pd.DataFrame({"feature": X.columns, "mi": scores}).sort_values("mi", ascending=False).reset_index(drop=True)

def heatmap_matriz(matriz: pd.DataFrame, titulo: Optional[str]=None, cmap: str="viridis", annot: bool=False, fmt: str=".2f", show: bool=True):
    """
    Crea un heatmap de una matriz (correlación, asociación, etc.).
    
    Args:
        matriz: DataFrame con la matriz a visualizar
        titulo: Título del gráfico
        cmap: Mapa de colores (ej: 'viridis', 'RdBu_r', 'Blues')
        annot: Si True, muestra los valores en las celdas
        fmt: Formato de los valores (ej: '.2f', '.3f')
        show: Si True, muestra el gráfico
        
    Returns:
        tuple: (fig, ax) para manipulación posterior
    """
    # Tamaño dinámico basado en el número de columnas/filas
    fig, ax = plt.subplots(figsize=(max(6, 0.6*len(matriz.columns)), max(5, 0.6*len(matriz.index))))
    sns.heatmap(matriz, cmap=cmap, annot=annot, fmt=fmt, square=False, cbar=True, ax=ax)
    if titulo: ax.set_title(titulo, fontweight="bold")
    plt.tight_layout()
    if show: plt.show()
    return fig, ax


__all__ = [
    "detectar_tipos_columnas",
    "resumen_nulos", "resumen_numericas", "reporte_categorias_raras",
    "graficar_categoricas", "graficar_numericas", "graficar_variables_numericas_completo",
    "graficar_variable_con_target",
    "matriz_correlacion_numericas", "matriz_asociacion_categoricas",
    "calcular_vif", "informacion_mutua_con_target",
    "heatmap_matriz",
]
