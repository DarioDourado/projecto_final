import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, auc
)
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger('ml_pipeline')

def comprehensive_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
    class_names: Optional[list] = None
) -> Dict[str, float]:
    """
    Realiza avaliação abrangente do modelo com múltiplas métricas.
    
    Args:
        y_true: Labels verdadeiros
        y_pred: Predições do modelo
        y_pred_proba: Probabilidades das predições (opcional)
        class_names: Nomes das classes (opcional)
    
    Returns:
        Dicionário com métricas de avaliação
    """
    metrics = {}
    
    # Relatório de classificação
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Métricas básicas
    metrics['accuracy'] = report['accuracy']
    metrics['macro_f1'] = report['macro avg']['f1-score']
    metrics['weighted_f1'] = report['weighted avg']['f1-score']
    metrics['macro_precision'] = report['macro avg']['precision']
    metrics['macro_recall'] = report['macro avg']['recall']
    
    # ROC-AUC se probabilidades estão disponíveis
    if y_pred_proba is not None:
        try:
            if len(np.unique(y_true)) == 2:
                # Classificação binária
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            else:
                # Classificação multiclasse
                metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
                metrics['roc_auc_ovo'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovo')
            
            logger.info("ROC-AUC calculado com sucesso")
        except Exception as e:
            logger.warning(f"Erro ao calcular ROC-AUC: {e}")
    
    # Log das métricas principais
    logger.info("=== MÉTRICAS DE AVALIAÇÃO ===")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    return metrics

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plota matriz de confusão.
    
    Args:
        y_true: Labels verdadeiros
        y_pred: Predições do modelo
        class_names: Nomes das classes
        figsize: Tamanho da figura
    
    Returns:
        Figura do matplotlib
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        ax=ax
    )
    ax.set_xlabel('Predições')
    ax.set_ylabel('Valores Reais')
    ax.set_title('Matriz de Confusão')
    
    logger.info("Matriz de confusão gerada")
    return fig

def plot_roc_curves(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    class_names: Optional[list] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plota curvas ROC para classificação multiclasse.
    
    Args:
        y_true: Labels verdadeiros
        y_pred_proba: Probabilidades das predições
        class_names: Nomes das classes
        figsize: Tamanho da figura
    
    Returns:
        Figura do matplotlib
    """
    n_classes = y_pred_proba.shape[1]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for i in range(n_classes):
        # Binarizar para classe atual
        y_true_binary = (y_true == i).astype(int)
        
        fpr, tpr, _ = roc_curve(y_true_binary, y_pred_proba[:, i])
        auc_score = auc(fpr, tpr)
        
        class_name = class_names[i] if class_names else f'Classe {i}'
        
        ax.plot(
            fpr, tpr,
            label=f'{class_name} (AUC = {auc_score:.3f})',
            linewidth=2
        )
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Taxa de Falsos Positivos')
    ax.set_ylabel('Taxa de Verdadeiros Positivos')
    ax.set_title('Curvas ROC por Classe')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    logger.info("Curvas ROC geradas")
    return fig