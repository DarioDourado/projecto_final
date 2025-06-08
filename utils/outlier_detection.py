import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, List
import logging

logger = logging.getLogger('ml_pipeline')

def detect_outliers_zscore(data: pd.DataFrame, columns: List[str], threshold: float = 3.0) -> pd.DataFrame:
    """
    Detecta outliers usando Z-score.
    
    Args:
        data: DataFrame com os dados
        columns: Lista de colunas para análise
        threshold: Limiar do Z-score (padrão: 3.0)
    
    Returns:
        DataFrame com outliers marcados
    """
    outliers_mask = pd.DataFrame(False, index=data.index, columns=columns)
    
    for col in columns:
        if col in data.columns:
            z_scores = np.abs(stats.zscore(data[col].dropna()))
            outliers_mask[col] = z_scores > threshold
            
            outlier_count = outliers_mask[col].sum()
            logger.info(f"Coluna '{col}': {outlier_count} outliers detectados (Z-score > {threshold})")
    
    return outliers_mask

def detect_outliers_iqr(data: pd.DataFrame, columns: List[str], factor: float = 1.5) -> pd.DataFrame:
    """
    Detecta outliers usando Interquartile Range (IQR).
    
    Args:
        data: DataFrame com os dados
        columns: Lista de colunas para análise
        factor: Fator multiplicativo do IQR (padrão: 1.5)
    
    Returns:
        DataFrame com outliers marcados
    """
    outliers_mask = pd.DataFrame(False, index=data.index, columns=columns)
    
    for col in columns:
        if col in data.columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            outliers_mask[col] = (data[col] < lower_bound) | (data[col] > upper_bound)
            
            outlier_count = outliers_mask[col].sum()
            logger.info(f"Coluna '{col}': {outlier_count} outliers detectados (IQR method)")
    
    return outliers_mask

def handle_outliers(data: pd.DataFrame, outliers_mask: pd.DataFrame, method: str = 'remove') -> pd.DataFrame:
    """
    Trata outliers detectados.
    
    Args:
        data: DataFrame original
        outliers_mask: Máscara de outliers
        method: Método de tratamento ('remove', 'cap', 'median')
    
    Returns:
        DataFrame com outliers tratados
    """
    data_clean = data.copy()
    
    if method == 'remove':
        # Remove linhas com outliers
        outlier_rows = outliers_mask.any(axis=1)
        data_clean = data_clean[~outlier_rows]
        logger.info(f"Removidas {outlier_rows.sum()} linhas com outliers")
    
    elif method == 'cap':
        # Substitui outliers pelos percentis 5% e 95%
        for col in outliers_mask.columns:
            if col in data_clean.columns:
                p5 = data_clean[col].quantile(0.05)
                p95 = data_clean[col].quantile(0.95)
                data_clean.loc[outliers_mask[col], col] = np.clip(
                    data_clean.loc[outliers_mask[col], col], p5, p95
                )
    
    elif method == 'median':
        # Substitui outliers pela mediana
        for col in outliers_mask.columns:
            if col in data_clean.columns:
                median_val = data_clean[col].median()
                data_clean.loc[outliers_mask[col], col] = median_val
    
    return data_clean