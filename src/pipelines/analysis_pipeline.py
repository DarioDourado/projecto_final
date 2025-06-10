"""Pipeline de análises avançadas"""

import logging
import joblib
from pathlib import Path
from src.analysis.clustering import SalaryClusteringAnalysis
from src.analysis.association_rules import AssociationRulesAnalysis
from src.evaluation.advanced_metrics import AdvancedMetrics

logger = logging.getLogger(__name__)

class AnalysisPipeline:
    """Pipeline de análises avançadas"""
    
    def __init__(self):
        self.clustering = SalaryClusteringAnalysis()
        self.association = AssociationRulesAnalysis()
        self.advanced_metrics = AdvancedMetrics()
    
    def run_clustering(self, df):
        """Executar análise de clustering"""
        logger.info("🎯 ANÁLISE DE CLUSTERING")
        logger.info("="*60)
        
        preprocessor_path = Path("data/processed/preprocessor.joblib")
        best_k = None
        
        if preprocessor_path.exists():
            try:
                preprocessor = joblib.load(preprocessor_path)
                X_features = df.drop('salary', axis=1)
                X_processed = preprocessor.transform(X_features)
                
                clusters, best_k = self.clustering.perform_kmeans_analysis(X_processed)
                self.clustering.visualize_clusters_pca(X_processed, clusters, df['salary'])
                
                logger.info(f"✅ Clustering concluído: {best_k} clusters")
            except Exception as e:
                logger.error(f"❌ Erro no clustering: {e}")
        else:
            logger.warning("⚠️ Pré-processador não encontrado")
        
        return best_k
    
    def run_association_rules(self, df):
        """Executar análise de regras de associação"""
        logger.info("📋 ANÁLISE DE REGRAS DE ASSOCIAÇÃO")
        logger.info("="*60)
        
        transactions = self.association.prepare_transaction_data(df)
        rules = []
        
        if transactions:
            rules = self.association.find_association_rules(
                transactions, min_support=0.03, min_confidence=0.5
            )
            rules_count = len(rules) if hasattr(rules, '__len__') else 0
            logger.info(f"✅ {rules_count} regras de associação encontradas")
        else:
            logger.warning("⚠️ Nenhuma transação válida para análise")
        
        return rules
    
    def run_advanced_metrics(self, df, results):
        """Executar métricas avançadas"""
        logger.info("📊 MÉTRICAS AVANÇADAS")
        logger.info("="*60)
        
        for model_name, result in results.items():
            if all(key in result for key in ['y_test', 'y_pred', 'y_pred_proba']):
                try:
                    self.advanced_metrics.calculate_comprehensive_metrics(
                        result['y_test'], 
                        result['y_pred'], 
                        result['y_pred_proba'],
                        model_name
                    )
                    
                    self.advanced_metrics.generate_business_kpis(
                        df, result['y_pred'], model_name
                    )
                    
                    logger.info(f"✅ Métricas calculadas para {model_name}")
                except Exception as e:
                    logger.error(f"❌ Erro para {model_name}: {e}")
        
        # Relatório comparativo
        try:
            self.advanced_metrics.generate_comparison_report()
            logger.info("✅ Relatório comparativo gerado")
        except Exception as e:
            logger.error(f"❌ Erro ao gerar relatório: {e}")