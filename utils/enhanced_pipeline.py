"""
Pipeline de Machine Learning com melhorias de logging, tratamento de outliers,
validação cruzada e métricas abrangentes.

Este módulo fornece uma implementação robusta para treinamento e avaliação
de modelos de machine learning com práticas recomendadas.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Imports dos módulos criados
from .logging_config import setup_logging
from .outlier_detection import detect_outliers_iqr, handle_outliers
from .cross_validation import perform_cross_validation, evaluate_model_stability
from .evaluation_metrics import comprehensive_evaluation, plot_confusion_matrix, plot_roc_curves

class EnhancedMLPipeline:
    """
    Pipeline de Machine Learning aprimorado com funcionalidades avançadas.
    
    Esta classe encapsula todo o processo de machine learning, desde o
    pré-processamento até a avaliação final, incluindo logging detalhado,
    tratamento de outliers e validação cruzada.
    
    Attributes:
        model: Modelo de machine learning
        scaler: Scaler para normalização das features
        logger: Logger para registro de eventos
        is_fitted: Flag indicando se o modelo foi treinado
    """
    
    def __init__(
        self,
        model: Any = None,
        log_level: int = logging.INFO,
        log_file: Optional[str] = None,
        random_state: int = 42
    ):
        """
        Inicializa o pipeline de machine learning.
        
        Args:
            model: Modelo de ML a ser usado (padrão: RandomForestClassifier)
            log_level: Nível de logging
            log_file: Arquivo para salvar logs (opcional)
            random_state: Seed para reprodutibilidade
        """
        self.logger = setup_logging(log_level, log_file)
        self.random_state = random_state
        self.model = model or RandomForestClassifier(random_state=random_state)
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        self.logger.info("Pipeline de ML inicializado")
    
    def preprocess_data(
        self,
        data: pd.DataFrame,
        target_column: str,
        handle_outliers_method: str = 'iqr',
        outlier_treatment: str = 'remove',
        feature_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Realiza pré-processamento completo dos dados.
        
        Args:
            data: DataFrame com os dados
            target_column: Nome da coluna target
            handle_outliers_method: Método para detectar outliers ('iqr' ou 'zscore')
            outlier_treatment: Como tratar outliers ('remove', 'cap', 'median')
            feature_columns: Colunas para usar como features (opcional)
        
        Returns:
            Tuple com features (X) e target (y) processados
        
        Raises:
            ValueError: Se target_column não existir nos dados
        """
        self.logger.info("Iniciando pré-processamento dos dados")
        
        if target_column not in data.columns:
            raise ValueError(f"Coluna target '{target_column}' não encontrada")
        
        # Separar features e target
        if feature_columns is None:
            feature_columns = [col for col in data.columns if col != target_column]
        
        X = data[feature_columns].copy()
        y = data[target_column].copy()
        
        self.logger.info(f"Dados originais: {X.shape[0]} amostras, {X.shape[1]} features")
        
        # Detectar e tratar outliers apenas em colunas numéricas
        numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_columns:
            self.logger.info("Detectando outliers...")
            outliers_mask = detect_outliers_iqr(X, numeric_columns)
            
            # Combinar dados para tratamento
            data_with_target = pd.concat([X, y], axis=1)
            data_clean = handle_outliers(data_with_target, outliers_mask, outlier_treatment)
            
            # Separar novamente
            X = data_clean[feature_columns]
            y = data_clean[target_column]
            
            self.logger.info(f"Dados após tratamento: {X.shape[0]} amostras")
        
        # Tratar valores faltantes
        if X.isnull().sum().sum() > 0:
            self.logger.warning("Valores faltantes detectados - preenchendo com mediana/moda")
            for col in X.columns:
                if X[col].dtype in ['float64', 'int64']:
                    X[col].fillna(X[col].median(), inplace=True)
                else:
                    X[col].fillna(X[col].mode()[0], inplace=True)
        
        self.logger.info("Pré-processamento concluído")
        return X, y
    
    def train_with_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        cv_folds: int = 5,
        scale_features: bool = True
    ) -> Dict[str, Any]:
        """
        Treina o modelo com validação cruzada.
        
        Args:
            X: Features
            y: Target variable
            test_size: Proporção dos dados para teste
            cv_folds: Número de folds para validação cruzada
            scale_features: Se deve escalar as features
        
        Returns:
            Dicionário com resultados da validação cruzada
        
        Raises:
            ValueError: Se os dados estão vazios
        """
        if X.empty or y.empty:
            raise ValueError("Dados de entrada estão vazios")
        
        self.logger.info("Iniciando treinamento com validação cruzada")
        
        # Split dos dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Escalar features se necessário
        if scale_features:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
            X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
            self.logger.info("Features escaladas")
        
        # Validação cruzada
        cv_results = perform_cross_validation(
            self.model, X_train, y_train, cv_folds=cv_folds
        )
        
        # Avaliar estabilidade
        stability = evaluate_model_stability(cv_results)
        
        # Treinar modelo final
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Avaliação no conjunto de teste
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test) if hasattr(self.model, 'predict_proba') else None
        
        test_metrics = comprehensive_evaluation(y_test, y_pred, y_pred_proba)
        
        # Armazenar dados de teste para visualizações
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        
        results = {
            'cv_results': cv_results,
            'stability_metrics': stability,
            'test_metrics': test_metrics,
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
        
        self.logger.info("Treinamento concluído com sucesso")
        return results
    
    def generate_visualizations(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Gera visualizações dos resultados do modelo.
        
        Args:
            save_path: Caminho para salvar as figuras (opcional)
        
        Returns:
            Dicionário com as figuras geradas
        
        Raises:
            RuntimeError: Se o modelo não foi treinado
        """
        if not self.is_fitted:
            raise RuntimeError("Modelo deve ser treinado antes de gerar visualizações")
        
        self.logger.info("Gerando visualizações...")
        
        figures = {}
        
        # Matriz de confusão
        figures['confusion_matrix'] = plot_confusion_matrix(
            self.y_test, self.y_pred
        )
        
        # Curvas ROC (se probabilidades disponíveis)
        if self.y_pred_proba is not None:
            figures['roc_curves'] = plot_roc_curves(
                self.y_test, self.y_pred_proba
            )
        
        # Salvar figuras se caminho fornecido
        if save_path:
            import os
            os.makedirs(save_path, exist_ok=True)
            
            for name, fig in figures.items():
                fig.savefig(f"{save_path}/{name}.png", dpi=300, bbox_inches='tight')
                self.logger.info(f"Figura salva: {save_path}/{name}.png")
        
        self.logger.info("Visualizações geradas com sucesso")
        return figures