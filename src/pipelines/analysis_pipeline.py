"""Pipeline de análises avançadas"""

import logging

logger = logging.getLogger(__name__)

class AnalysisPipeline:
    """Pipeline de análises avançadas"""
    
    def __init__(self):
        pass
    
    def run_clustering(self, df):
        """Executar análise de clustering"""
        try:
            # Import local
            import joblib
            from pathlib import Path
            from src.analysis.clustering import SalaryClusteringAnalysis
            
            logger.info("🎯 Iniciando análise de clustering...")
            
            clustering = SalaryClusteringAnalysis()
            preprocessor_path = Path("data/processed/preprocessor.joblib")
            best_k = None
            
            if preprocessor_path.exists():
                try:
                    preprocessor = joblib.load(preprocessor_path)
                    X_features = df.drop('salary', axis=1)
                    X_processed = preprocessor.transform(X_features)
                    
                    clusters, best_k = clustering.perform_kmeans_analysis(X_processed)
                    clustering.visualize_clusters_pca(X_processed, clusters, df['salary'])
                    
                    logger.info(f"✅ Clustering concluído: {best_k} clusters")
                except Exception as e:
                    logger.error(f"❌ Erro no clustering: {e}")
            else:
                logger.warning("⚠️ Pré-processador não encontrado")
            
            return best_k
            
        except Exception as e:
            logger.error(f"❌ Erro no pipeline de clustering: {e}")
            return None
    
    def run_association_rules(self, df):
        """Executar análise de regras de associação"""
        try:
            # Import local
            from src.analysis.association_rules import AssociationRulesAnalysis
            
            logger.info("📋 Iniciando análise de regras de associação...")
            
            association = AssociationRulesAnalysis()
            transactions = association.prepare_transaction_data(df)
            rules = []
            
            if transactions:
                rules = association.find_association_rules(
                    transactions, min_support=0.03, min_confidence=0.5
                )
                rules_count = len(rules) if hasattr(rules, '__len__') else 0
                logger.info(f"✅ {rules_count} regras de associação encontradas")
            else:
                logger.warning("⚠️ Nenhuma transação válida para análise")
            
            return rules
            
        except Exception as e:
            logger.error(f"❌ Erro no pipeline de regras: {e}")
            return []
    
    def run_advanced_metrics(self, df, results):
        """Executar métricas avançadas"""
        try:
            # Import local
            from src.evaluation.advanced_metrics import AdvancedMetrics
            
            logger.info("📊 Iniciando métricas avançadas...")
            
            advanced_metrics = AdvancedMetrics()
            
            for model_name, result in results.items():
                if all(key in result for key in ['y_test', 'y_pred', 'y_pred_proba']):
                    try:
                        advanced_metrics.calculate_comprehensive_metrics(
                            result['y_test'], 
                            result['y_pred'], 
                            result['y_pred_proba'],
                            model_name
                        )
                        
                        advanced_metrics.generate_business_kpis(
                            df, result['y_pred'], model_name
                        )
                        
                        logger.info(f"✅ Métricas calculadas para {model_name}")
                    except Exception as e:
                        logger.error(f"❌ Erro para {model_name}: {e}")
            
            # Relatório comparativo
            try:
                advanced_metrics.generate_comparison_report()
                logger.info("✅ Relatório comparativo gerado")
            except Exception as e:
                logger.error(f"❌ Erro ao gerar relatório: {e}")
                
        except Exception as e:
            logger.error(f"❌ Erro no pipeline de métricas: {e}")