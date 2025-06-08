"""Módulo para análise exploratória de dados"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ..utils.logger import get_logger, log_function
from ..visualization.modern_styles import ModernStyle
from ..config.settings import IMAGES_DIR

logger = get_logger(__name__)

class EDAAnalyzer:
    """Classe para análise exploratória de dados"""
    
    def __init__(self):
        ModernStyle.setup_matplotlib()
    
    @log_function
    def generate_all_visualizations(self, df: pd.DataFrame):
        """Gerar todas as visualizações"""
        
        # Criar diretório de imagens
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        
        # 1. Histogramas para variáveis numéricas
        self._generate_histograms(df)
        
        # 2. Gráficos de barras para variáveis categóricas
        self._generate_bar_plots(df)
        
        # 3. Matriz de correlação
        self._generate_correlation_matrix(df)
        
        # 4. Distribuição da variável target
        self._generate_target_distribution(df)
        
        logger.info("✅ Todas as visualizações foram geradas")
    
    def _generate_histograms(self, df: pd.DataFrame):
        """Gerar histogramas"""
        numerical_cols = df.select_dtypes(include=['int', 'float']).columns
        
        for col in numerical_cols:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Histograma
            df[col].hist(bins=30, ax=ax, alpha=0.7)
            
            # Aplicar estilo moderno
            ModernStyle.apply_modern_style(ax, title=f"Distribuição de {col}")
            ax.set_xlabel(col)
            ax.set_ylabel('Frequência')
            
            # Salvar
            filename = f"hist_{col}.png"
            ModernStyle.save_plot(filename)
            
            logger.info(f"📊 Histograma gerado: {filename}")
    
    def _generate_bar_plots(self, df: pd.DataFrame):
        """Gerar gráficos de barras"""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if df[col].nunique() < 20:  # Apenas se não tiver muitas categorias
                fig, ax = plt.subplots(figsize=(12, 6))
                
                value_counts = df[col].value_counts()
                value_counts.plot(kind='bar', ax=ax)
                
                ModernStyle.apply_modern_style(ax, title=f"Distribuição de {col}")
                ax.set_xlabel(col)
                ax.set_ylabel('Frequência')
                plt.xticks(rotation=45)
                
                filename = f"bar_{col}.png"
                ModernStyle.save_plot(filename)
                
                logger.info(f"📊 Gráfico de barras gerado: {filename}")
    
    def _generate_correlation_matrix(self, df: pd.DataFrame):
        """Gerar matriz de correlação"""
        numerical_cols = df.select_dtypes(include=['int', 'float']).columns
        
        if len(numerical_cols) > 1:
            fig, ax = plt.subplots(figsize=(12, 10))
            
            corr_matrix = df[numerical_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            
            ModernStyle.apply_modern_style(ax, title="Matriz de Correlação")
            
            filename = "correlation_matrix.png"
            ModernStyle.save_plot(filename)
            
            logger.info(f"📊 Matriz de correlação gerada: {filename}")
    
    def _generate_target_distribution(self, df: pd.DataFrame):
        """Gerar distribuição da variável target"""
        target_col = 'salary'
        
        if target_col in df.columns:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            df[target_col].value_counts().plot(kind='pie', ax=ax, autopct='%1.1f%%')
            
            ModernStyle.apply_modern_style(ax, title="Distribuição de Salários")
            
            filename = "target_distribution.png"
            ModernStyle.save_plot(filename)
            
            logger.info(f"📊 Distribuição target gerada: {filename}")