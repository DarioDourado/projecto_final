"""Métricas avançadas e KPIs para relatório académico"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, 
    roc_curve, confusion_matrix
)

class AdvancedMetrics:
    """Métricas avançadas para avaliação rigorosa"""
    
    def __init__(self):
        self.metrics_summary = {}
        self.business_kpis = {}
    
    def calculate_comprehensive_metrics(self, y_true, y_pred, y_pred_proba, model_name="Model"):
        """Calcular métricas abrangentes"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        try:
            # Ensure all arrays have the same length
            min_length = min(len(y_true), len(y_pred))
            if y_pred_proba is not None:
                min_length = min(min_length, len(y_pred_proba))
            
            # Truncate to the same length
            y_true = y_true[:min_length]
            y_pred = y_pred[:min_length]
            if y_pred_proba is not None:
                y_pred_proba = y_pred_proba[:min_length]
            
            # Convert to binary if necessary
            if hasattr(y_true, 'iloc') and isinstance(y_true.iloc[0], str):
                y_true_binary = (y_true == '>50K').astype(int)
            else:
                y_true_binary = y_true
                
            if isinstance(y_pred[0], str):
                y_pred_binary = (y_pred == '>50K').astype(int)
            else:
                y_pred_binary = y_pred
            
            # Final length validation
            if len(y_true_binary) != len(y_pred_binary):
                raise ValueError(f"Length mismatch after processing: y_true({len(y_true_binary)}) vs y_pred({len(y_pred_binary)})")
            
            metrics = {
                'model_name': model_name,
                'accuracy': accuracy_score(y_true_binary, y_pred_binary),
                'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
                'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
                'f1_score': f1_score(y_true_binary, y_pred_binary, zero_division=0)
            }
            
            # Add ROC AUC if probabilities are available
            if y_pred_proba is not None:
                try:
                    y_prob = y_pred_proba[:, 1] if y_pred_proba.ndim > 1 else y_pred_proba
                    if len(y_prob) == len(y_true_binary):
                        metrics['roc_auc'] = roc_auc_score(y_true_binary, y_prob)
                    else:
                        logging.warning(f"⚠️ Probability length mismatch for {model_name}")
                        metrics['roc_auc'] = 0.0
                except Exception as e:
                    logging.warning(f"⚠️ Could not calculate ROC AUC for {model_name}: {e}")
                    metrics['roc_auc'] = 0.0
            else:
                metrics['roc_auc'] = 0.0
            
            # Store metrics for comparison
            self.metrics_summary[model_name] = metrics
            
            logging.info(f"✅ Métricas calculadas para {model_name}")
            return metrics
            
        except Exception as e:
            logging.error(f"❌ Erro no cálculo de métricas para {model_name}: {e}")
            return {
                'model_name': model_name, 
                'accuracy': 0.0, 
                'precision': 0.0, 
                'recall': 0.0, 
                'f1_score': 0.0, 
                'roc_auc': 0.0
            }
    
    def generate_business_kpis(self, df, predictions, model_name="Model"):
        """Gerar KPIs orientados ao negócio"""
        
        y_true = df['salary']
        y_pred = predictions
        
        kpis = {
            'total_samples': len(df),
            'high_salary_rate': (y_true == '>50K').mean(),
            'prediction_accuracy': (y_pred == y_true).mean(),
            'false_positive_rate': ((y_pred == '>50K') & (y_true == '<=50K')).mean(),
            'false_negative_rate': ((y_pred == '<=50K') & (y_true == '>50K')).mean(),
        }
        
        # KPIs por segmento demográfico
        if 'sex' in df.columns:
            for sex in df['sex'].unique():
                if pd.notna(sex):
                    mask = df['sex'] == sex
                    if mask.sum() > 0:
                        kpis[f'accuracy_{sex.lower()}'] = (y_pred[mask] == y_true.loc[mask]).mean()
        
        self.business_kpis[model_name] = kpis
        
        # Salvar relatório de KPIs
        self._save_kpi_report(kpis, model_name)
        
        logging.info(f"✅ KPIs de negócio calculados para {model_name}")
        return kpis
    
    def _plot_roc_pr_curves(self, y_true, y_prob, model_name):
        """Plotar curvas ROC e Precision-Recall"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Curva ROC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('Taxa de Falsos Positivos')
        ax1.set_ylabel('Taxa de Verdadeiros Positivos')
        ax1.set_title(f'Curva ROC - {model_name}')
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)
        
        # Curva Precision-Recall
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        
        ax2.plot(recall, precision, color='darkgreen', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precisão')
        ax2.set_title(f'Curva Precision-Recall - {model_name}')
        ax2.legend(loc="lower left")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Salvar
        output_dir = Path("output/images")
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / f"roc_pr_curves_{model_name.lower().replace(' ', '_')}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_kpi_report(self, kpis, model_name):
        """Salvar relatório de KPIs"""
        output_dir = Path("output/analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report = []
        report.append(f"# RELATÓRIO DE KPIs - {model_name}\n\n")
        report.append(f"**Data:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        
        # KPIs Gerais
        report.append("## 📊 KPIs Gerais\n")
        for key, value in kpis.items():
            if isinstance(value, float):
                report.append(f"- **{key.replace('_', ' ').title()}:** {value:.3f}\n")
            else:
                report.append(f"- **{key.replace('_', ' ').title()}:** {value:,}\n")
        
        # Salvar relatório
        report_file = output_dir / f"kpi_report_{model_name.lower().replace(' ', '_')}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.writelines(report)
    
    def generate_comparison_report(self):
        """Gerar relatório comparativo entre modelos"""
        if not self.metrics_summary:
            logging.warning("⚠️ Nenhuma métrica disponível para comparação")
            return
        
        # Criar DataFrame com métricas
        df_metrics = pd.DataFrame(self.metrics_summary).T
        
        # Salvar CSV
        output_dir = Path("output/analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        df_metrics.to_csv(output_dir / "model_comparison.csv")
        
        # Gerar gráfico comparativo
        self._plot_model_comparison(df_metrics)
        
        # Gerar relatório em markdown
        self._save_comparison_report(df_metrics)
        
        logging.info("✅ Relatório comparativo gerado")
    
    def _plot_model_comparison(self, df_metrics):
        """Plotar comparação entre modelos"""
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(df_metrics))
        width = 0.15
        
        for i, metric in enumerate(metrics_to_plot):
            if metric in df_metrics.columns:
                ax.bar(x + i * width, df_metrics[metric], width, label=metric.title())
        
        ax.set_xlabel('Modelos')
        ax.set_ylabel('Score')
        ax.set_title('Comparação de Performance entre Modelos')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(df_metrics.index, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_dir = Path("output/images")
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_comparison_report(self, df_metrics):
        """Salvar relatório comparativo em markdown"""
        output_dir = Path("output/analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report = []
        report.append("# RELATÓRIO COMPARATIVO DE MODELOS\n\n")
        report.append(f"**Data:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        
        # Tabela de métricas
        report.append("## 📊 Comparação de Métricas\n\n")
        report.append("| Modelo | Accuracy | Precision | Recall | F1-Score | ROC-AUC |\n")
        report.append("|--------|----------|-----------|--------|----------|----------|\n")
        
        for model_name, row in df_metrics.iterrows():
            report.append(f"| {model_name} |")
            for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
                if metric in row:
                    report.append(f" {row[metric]:.4f} |")
                else:
                    report.append(" N/A |")
            report.append("\n")
        
        # Melhor modelo por métrica
        report.append("\n## 🏆 Melhor Modelo por Métrica\n\n")
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
            if metric in df_metrics.columns:
                best_model = df_metrics[metric].idxmax()
                best_score = df_metrics[metric].max()
                report.append(f"- **{metric.title()}**: {best_model} ({best_score:.4f})\n")
        
        # Salvar relatório
        report_file = output_dir / "model_comparison_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.writelines(report)
        
        logging.info(f"📊 Relatório comparativo salvo: {report_file}")