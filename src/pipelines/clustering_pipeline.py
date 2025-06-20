import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from src.analysis.clustering import SalaryClusteringAnalysis

class ClusteringPipeline:
    """Pipeline para análise de clustering"""
    
    def __init__(self):
        self.clustering_analysis = SalaryClusteringAnalysis()
        self.logger = logging.getLogger(__name__)

    def run(self, df):
        """Método principal - executar análise completa de clustering"""
        return self.run_complete_clustering_analysis(df)

    def run_complete_clustering_analysis(self, df):
        """Executar análise completa de clustering (K-Means + DBSCAN)"""
        self.logger.info("🎯 Iniciando análise completa de clustering...")
        
        try:
            # Preparar dados
            X_processed = self._prepare_clustering_data(df)
            
            if X_processed is None:
                self.logger.error("❌ Falha na preparação dos dados")
                return None
            
            # Executar K-Means
            self.logger.info("🔄 Executando K-Means...")
            kmeans_clusters, kmeans_k = self.clustering_analysis.perform_kmeans_analysis(X_processed)
            kmeans_silhouette = max(self.clustering_analysis.silhouette_scores.get('kmeans', [0])) if self.clustering_analysis.silhouette_scores.get('kmeans') else 0
            
            # Executar DBSCAN
            self.logger.info("🔄 Executando DBSCAN...")
            dbscan_result = self.clustering_analysis.perform_dbscan_analysis(X_processed)
            
            # Construir resultados
            results = {}
            
            # Resultado K-Means
            if kmeans_clusters is not None:
                results['K-Means'] = {
                    'clusters': kmeans_clusters,
                    'n_clusters': kmeans_k,
                    'silhouette_score': kmeans_silhouette,
                    'has_noise': False,
                    'noise_percentage': 0
                }
                self.logger.info(f"✅ K-Means: {kmeans_k} clusters, Silhouette: {kmeans_silhouette:.3f}")
            
            # Resultado DBSCAN
            if dbscan_result and len(dbscan_result) == 3:
                dbscan_clusters, dbscan_k, dbscan_silhouette = dbscan_result
                if dbscan_clusters is not None:
                    n_noise = list(dbscan_clusters).count(-1)
                    results['DBSCAN'] = {
                        'clusters': dbscan_clusters,
                        'n_clusters': dbscan_k,
                        'silhouette_score': dbscan_silhouette,
                        'has_noise': True,
                        'noise_percentage': (n_noise / len(dbscan_clusters)) * 100
                    }
                    self.logger.info(f"✅ DBSCAN: {dbscan_k} clusters, Silhouette: {dbscan_silhouette:.3f}, Ruído: {results['DBSCAN']['noise_percentage']:.1f}%")
            
            self.logger.info(f"✅ Clustering concluído: {len(results)} métodos")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Erro no clustering: {e}")
            return None

    def _prepare_clustering_data(self, df: pd.DataFrame) -> np.ndarray:
        """Preparar dados para clustering"""
        try:
            self.logger.info("📊 Preparando dados para clustering...")
            
            # Selecionar colunas numéricas
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_cols:
                self.logger.error("❌ Nenhuma coluna numérica encontrada")
                return None
            
            # Preparar dados
            X = df[numeric_cols].copy()
            
            # Remover linhas com valores ausentes
            X = X.dropna()
            
            if len(X) == 0:
                self.logger.error("❌ Nenhuma linha válida após remoção de NaN")
                return None
            
            # Normalizar dados
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            self.logger.info(f"📊 Dados preparados: {X_scaled.shape[0]} amostras, {X_scaled.shape[1]} features")
            return X_scaled
            
        except Exception as e:
            self.logger.error(f"❌ Erro na preparação dos dados: {e}")
            return None