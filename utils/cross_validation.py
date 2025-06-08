import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, f1_score, roc_auc_score, precision_score, recall_score
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger('ml_pipeline')

def perform_cross_validation(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = 5,
    scoring_metrics: Dict[str, Any] = None,
    random_state: int = 42
) -> Dict[str, np.ndarray]:
    """
    Realiza validação cruzada com múltiplas métricas.
    
    Args:
        model: Modelo de machine learning
        X: Features
        y: Target variable
        cv_folds: Número de folds para validação cruzada
        scoring_metrics: Dicionário com métricas de avaliação
        random_state: Seed para reprodutibilidade
    
    Returns:
        Dicionário com scores de cada métrica
    """
    if scoring_metrics is None:
        scoring_metrics = {
            'f1': make_scorer(f1_score, average='weighted'),
            'precision': make_scorer(precision_score, average='weighted'),
            'recall': make_scorer(recall_score, average='weighted'),
            'roc_auc': make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr')
        }
    
    # Configuração do cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    logger.info(f"Iniciando validação cruzada com {cv_folds} folds...")
    
    # Realiza validação cruzada
    cv_results = cross_validate(
        model, X, y,
        cv=cv,
        scoring=scoring_metrics,
        return_train_score=True,
        n_jobs=-1
    )
    
    # Log dos resultados
    for metric_name in scoring_metrics.keys():
        test_scores = cv_results[f'test_{metric_name}']
        train_scores = cv_results[f'train_{metric_name}']
        
        logger.info(f"{metric_name.upper()}:")
        logger.info(f"  Test:  {test_scores.mean():.4f} ± {test_scores.std():.4f}")
        logger.info(f"  Train: {train_scores.mean():.4f} ± {train_scores.std():.4f}")
    
    return cv_results

def evaluate_model_stability(cv_results: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Avalia a estabilidade do modelo baseado na variação dos scores.
    
    Args:
        cv_results: Resultados da validação cruzada
    
    Returns:
        Dicionário com métricas de estabilidade
    """
    stability_metrics = {}
    
    for key, scores in cv_results.items():
        if key.startswith('test_'):
            metric_name = key.replace('test_', '')
            cv_std = scores.std()
            cv_mean = scores.mean()
            
            # Coeficiente de variação
            cv_coefficient = cv_std / cv_mean if cv_mean != 0 else 0
            
            stability_metrics[f'{metric_name}_stability'] = 1 - cv_coefficient
            
            logger.info(f"Estabilidade {metric_name}: {stability_metrics[f'{metric_name}_stability']:.4f}")
    
    return stability_metrics