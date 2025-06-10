"""Pipeline de análise de performance"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_curve, precision_recall_curve, auc

logger = logging.getLogger(__name__)

class PerformancePipeline:
    """Pipeline de análise de performance"""
    
    def __init__(self):
        self.output_dir = Path("output")
        self.images_dir = self.output_dir / "images"
        self.analysis_dir = self.output_dir / "analysis"
        
        # Criar diretórios
        self.output_dir.mkdir(exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)
        self.analysis_dir.mkdir(exist_ok=True)
    
    def run(self, models, results, df):
        """Executar pipeline de performance"""
        logger.info("🎯 ANÁLISES DE PERFORMANCE")
        logger.info("="*60)
        
        # 1. Comparação de modelos
        comparison_data = self._generate_model_comparison(results)
        
        # 2. Curvas ROC/PR
        self._generate_roc_pr_curves(results)
        
        # 3. Feature importance
        self._generate_feature_importance(models)
        
        # 4. Relatório
        self._generate_performance_report(comparison_data)
        
        logger.info("✅ Análises de performance concluídas")
    
    def _generate_model_comparison(self, results):
        """Gerar comparação entre modelos"""
        comparison_data = []
        
        for model_name, result in results.items():
            if 'accuracy' in result:
                comparison_data.append({
                    'model_name': model_name,
                    'accuracy': result['accuracy'],
                    'precision': result.get('precision', 0.0),
                    'recall': result.get('recall', 0.0),
                    'f1_score': result.get('f1_score', 0.0),
                    'roc_auc': result.get('roc_auc', 0.0)
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df.to_csv(self.analysis_dir / "model_comparison.csv", index=False)
            logger.info("✅ Comparação de modelos salva")
        
        return comparison_data
    
    def _generate_roc_pr_curves(self, results):
        """Gerar curvas ROC e Precision-Recall"""
        for model_name, result in results.items():
            if all(key in result for key in ['y_test', 'y_pred', 'y_pred_proba']):
                try:
                    self._plot_roc_pr_curve(result, model_name)
                except Exception as e:
                    logger.error(f"❌ Erro curvas {model_name}: {e}")
    
    def _plot_roc_pr_curve(self, result, model_name):
        """Plotar curva ROC e PR para um modelo"""
        y_test = result['y_test']
        y_pred_proba = result['y_pred_proba']
        
        # Converter para binário
        if hasattr(y_test.iloc[0] if hasattr(y_test, 'iloc') else y_test[0], 'replace'):
            y_test_binary = (y_test == '>50K').astype(int)
        else:
            y_test_binary = y_test
        
        y_prob = y_pred_proba[:, 1] if y_pred_proba.ndim > 1 else y_pred_proba
        
        # Criar subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Curva ROC
        fpr, tpr, _ = roc_curve(y_test_binary, y_prob)
        roc_auc = auc(fpr, tpr)
        
        ax1.plot(fpr, tpr, 'darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], 'navy', lw=2, linestyle='--', label='Random')
        ax1.set_xlabel('Taxa de Falsos Positivos')
        ax1.set_ylabel('Taxa de Verdadeiros Positivos')
        ax1.set_title(f'Curva ROC - {model_name}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Curva Precision-Recall
        precision, recall, _ = precision_recall_curve(y_test_binary, y_prob)
        pr_auc = auc(recall, precision)
        
        ax2.plot(recall, precision, 'darkgreen', lw=2, label=f'PR (AUC = {pr_auc:.3f})')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precisão')
        ax2.set_title(f'Curva Precision-Recall - {model_name}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Salvar
        filename = f"roc_pr_curves_{model_name.lower().replace(' ', '_')}.png"
        plt.savefig(self.images_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Curvas geradas para {model_name}")
    
    def _generate_feature_importance(self, models):
        """Gerar análise de feature importance"""
        from src.pipelines.utils import load_real_feature_names
        
        for model_name, model in models.items():
            if hasattr(model, 'feature_importances_'):
                try:
                    importances = model.feature_importances_
                    feature_names = load_real_feature_names(len(importances))
                    
                    # DataFrame
                    importance_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance': importances
                    }).sort_values('importance', ascending=False)
                    
                    # Salvar CSV
                    csv_name = f"feature_importance_{model_name.lower().replace(' ', '_')}.csv"
                    importance_df.to_csv(self.analysis_dir / csv_name, index=False)
                    
                    # Gerar gráfico
                    self._plot_feature_importance(importance_df, model_name)
                    
                    logger.info(f"✅ Feature importance para {model_name}")
                    
                except Exception as e:
                    logger.error(f"❌ Erro feature importance {model_name}: {e}")
    
    def _plot_feature_importance(self, importance_df, model_name):
        """Plotar feature importance"""
        plt.figure(figsize=(14, 10))
        top_features = importance_df.head(20)
        
        bars = plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importância da Feature')
        plt.title(f'Top 20 Features - {model_name}')
        plt.grid(True, alpha=0.3)
        
        # Colorir barras
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Valores nas barras
        for bar, importance in zip(bars, top_features['importance']):
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                   f'{importance:.3f}', ha='left', va='center', fontsize=8)
        
        plt.tight_layout()
        
        # Salvar
        img_name = f"feature_importance_{model_name.lower().replace(' ', '_')}.png"
        plt.savefig(self.images_dir / img_name, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_performance_report(self, comparison_data):
        """Gerar relatório de performance"""
        report = []
        report.append("# RELATÓRIO DE PERFORMANCE DOS MODELOS\n\n")
        report.append(f"**Data:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if comparison_data:
            report.append("## 📊 Resumo dos Modelos\n\n")
            report.append("| Modelo | Accuracy | Precision | Recall | F1-Score | ROC-AUC |\n")
            report.append("|--------|----------|-----------|--------|----------|----------|\n")
            
            for model_data in comparison_data:
                report.append(f"| {model_data['model_name']} |")
                report.append(f" {model_data['accuracy']:.4f} |")
                report.append(f" {model_data['precision']:.4f} |")
                report.append(f" {model_data['recall']:.4f} |")
                report.append(f" {model_data['f1_score']:.4f} |")
                report.append(f" {model_data['roc_auc']:.4f} |\n")
            
            # Melhor modelo
            best_model = max(comparison_data, key=lambda x: x['accuracy'])
            report.append(f"\n## 🏆 Melhor Modelo\n\n")
            report.append(f"**{best_model['model_name']}** com accuracy de {best_model['accuracy']:.4f}\n\n")
        
        # Salvar
        with open(self.analysis_dir / "performance_report.md", 'w', encoding='utf-8') as f:
            f.writelines(report)
        
        logger.info("✅ Relatório de performance salvo")