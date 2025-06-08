"""Módulo para detecção e remoção de outliers"""

import pandas as pd
import numpy as np
from scipy.stats import zscore
from typing import Tuple, List, Dict, Any
from ..utils.logger import get_logger, log_function

logger = get_logger(__name__)

class OutlierDetector:
    """Classe para detecção e remoção de outliers"""
    
    def __init__(self, method='zscore', threshold=3):
        self.method = method
        self.threshold = threshold
        self.outlier_stats = {}
    
    @log_function
    def detect_outliers_zscore(self, df: pd.DataFrame, columns: List[str]) -> pd.Series:
        """Detectar outliers usando Z-score"""
        outlier_mask = pd.Series([False] * len(df), index=df.index)
        
        for col in columns:
            if col in df.columns:
                z_scores = np.abs(zscore(df[col].dropna()))
                col_outliers = z_scores > self.threshold
                
                # Expandir máscara para incluir índices originais
                col_outlier_mask = df[col].notna() & df[col].isin(df[col].dropna().iloc[col_outliers])
                outlier_mask |= col_outlier_mask
                
                # Estatísticas
                outlier_count = col_outlier_mask.sum()
                self.outlier_stats[col] = {
                    'count': outlier_count,
                    'percentage': (outlier_count / len(df)) * 100
                }
                
                logger.info(f"📊 {col}: {outlier_count} outliers ({(outlier_count/len(df)*100):.1f}%)")
        
        return outlier_mask
    
    @log_function
    def detect_outliers_iqr(self, df: pd.DataFrame, columns: List[str]) -> pd.Series:
        """Detectar outliers usando IQR"""
        outlier_mask = pd.Series([False] * len(df), index=df.index)
        
        for col in columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                col_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                outlier_mask |= col_outliers
                
                # Estatísticas
                outlier_count = col_outliers.sum()
                self.outlier_stats[col] = {
                    'count': outlier_count,
                    'percentage': (outlier_count / len(df)) * 100,
                    'bounds': (lower_bound, upper_bound)
                }
                
                logger.info(f"📊 {col}: {outlier_count} outliers IQR ({(outlier_count/len(df)*100):.1f}%)")
        
        return outlier_mask
    
    @log_function
    def remove_outliers(self, df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Remover outliers do DataFrame"""
        logger.info(f"🔍 Detectando outliers usando método: {self.method}")
        
        if self.method == 'zscore':
            outlier_mask = self.detect_outliers_zscore(df, columns)
        elif self.method == 'iqr':
            outlier_mask = self.detect_outliers_iqr(df, columns)
        else:
            raise ValueError(f"Método não reconhecido: {self.method}")
        
        # Remover outliers
        df_clean = df[~outlier_mask].copy()
        
        total_outliers = outlier_mask.sum()
        removal_percentage = (total_outliers / len(df)) * 100
        
        stats = {
            'method': self.method,
            'threshold': self.threshold,
            'total_outliers_removed': total_outliers,
            'removal_percentage': removal_percentage,
            'original_size': len(df),
            'final_size': len(df_clean),
            'column_stats': self.outlier_stats
        }
        
        logger.info(f"🗑️ Removidos {total_outliers} outliers ({removal_percentage:.1f}%)")
        logger.info(f"📊 Dataset: {len(df)} → {len(df_clean)} registros")
        
        return df_clean, stats