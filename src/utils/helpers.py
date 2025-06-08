"""Funções auxiliares do projeto"""

import pandas as pd
import numpy as np
from pathlib import Path

def get_memory_usage(df: pd.DataFrame) -> float:
    """Calcular uso de memória do DataFrame em MB"""
    memory_usage = df.memory_usage(deep=True).sum()
    return memory_usage / 1024 / 1024

def calculate_missing_percentage(df: pd.DataFrame) -> pd.Series:
    """Calcular percentual de valores ausentes por coluna"""
    return (df.isnull().sum() / len(df)) * 100

def get_categorical_summary(df: pd.DataFrame, column: str) -> dict:
    """Obter resumo de variável categórica"""
    value_counts = df[column].value_counts()
    return {
        'unique_values': df[column].nunique(),
        'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
        'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
        'most_frequent_percentage': (value_counts.iloc[0] / len(df)) * 100 if len(value_counts) > 0 else 0
    }

def get_numerical_summary(df: pd.DataFrame, column: str) -> dict:
    """Obter resumo de variável numérica"""
    col_data = df[column].dropna()
    return {
        'mean': col_data.mean(),
        'median': col_data.median(),
        'std': col_data.std(),
        'min': col_data.min(),
        'max': col_data.max(),
        'q25': col_data.quantile(0.25),
        'q75': col_data.quantile(0.75),
        'skewness': col_data.skew(),
        'kurtosis': col_data.kurtosis()
    }

def ensure_directory(path: Path) -> Path:
    """Garantir que diretório existe"""
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_dataframe(df: pd.DataFrame, filename: str, directory: Path) -> Path:
    """Salvar DataFrame com logging"""
    ensure_directory(directory)
    filepath = directory / filename
    df.to_csv(filepath, index=False)
    return filepath