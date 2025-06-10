"""Análise de Clustering para Segmentação Salarial - VERSÃO COMPLETA"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging
from pathlib import Path

class SalaryClusteringAnalysis:
    """Análise de clustering para segmentação de perfis salariais"""
    
    def __init__(self):
        self.kmeans_model = None
        self.dbscan_model = None
        self.pca = None
        self.scaler = StandardScaler(with_mean=False)  # Para matrizes esparsas
        self.silhouette_scores = {}
    
    def perform_kmeans_analysis(self, X, max_clusters=8):
        """Análise K-Means com método do cotovelo"""
        logging.info("🔍 Iniciando análise K-Means...")
        
        # Verificar se é matriz esparsa e converter se necessário
        if hasattr(X, 'sparse') and X.sparse:
            logging.info("📊 Convertendo matriz esparsa para densa...")
            X_scaled = X.toarray()
            X_scaled = StandardScaler().fit_transform(X_scaled)
        elif hasattr(X, 'shape') and len(X.shape) == 2:
            try:
                X_scaled = StandardScaler().fit_transform(X)
            except ValueError as e:
                if "sparse" in str(e).lower():
                    logging.info("📊 Usando StandardScaler com with_mean=False...")
                    X_scaled = StandardScaler(with_mean=False).fit_transform(X)
                else:
                    raise e
        else:
            X_scaled = X
        
        inertias = []
        silhouette_scores = []
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
            
            logging.info(f"  📊 K={k}: Inércia={kmeans.inertia_:.2f}, Silhouette={silhouette_scores[-1]:.3f}")
        
        # Gráfico do cotovelo e silhouette
        self._plot_clustering_analysis(range(2, max_clusters + 1), inertias, silhouette_scores)
        
        # Escolher melhor k (maior silhouette score)
        best_k = range(2, max_clusters + 1)[np.argmax(silhouette_scores)]
        logging.info(f"✅ Melhor número de clusters: {best_k} (Silhouette: {max(silhouette_scores):.3f})")
        
        # Treinar modelo final
        self.kmeans_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        clusters = self.kmeans_model.fit_predict(X_scaled)
        
        # Salvar resultados
        self._save_clustering_results(clusters, best_k, max(silhouette_scores))
        
        return clusters, best_k
    
    def _plot_clustering_analysis(self, k_range, inertias, silhouette_scores):
        """Plotar análise do cotovelo e silhouette"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Método do cotovelo
        ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Número de Clusters')
        ax1.set_ylabel('Inércia')
        ax1.set_title('Método do Cotovelo')
        ax1.grid(True, alpha=0.3)
        
        # Análise silhouette
        ax2.plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Número de Clusters')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Análise Silhouette')
        ax2.grid(True, alpha=0.3)
        
        # Marcar melhor k
        best_k_idx = np.argmax(silhouette_scores)
        best_k = k_range[best_k_idx]
        ax2.axvline(best_k, color='green', linestyle='--', alpha=0.7, label=f'Melhor K={best_k}')
        ax2.legend()
        
        plt.tight_layout()
        
        # Salvar
        images_dir = Path("output/images")
        images_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(images_dir / "clustering_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info("📈 Gráficos de análise de clustering salvos")
    
    def visualize_clusters_pca(self, X, clusters, target=None):
        """Visualizar clusters com PCA 2D"""
        logging.info("🎨 Gerando visualizações PCA dos clusters...")
        
        # Aplicar PCA
        self.pca = PCA(n_components=2, random_state=42)
        X_pca = self.pca.fit_transform(X)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Clusters
        scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, 
                                  cmap='viridis', alpha=0.7, s=50)
        axes[0].set_title('Clusters K-Means')
        axes[0].set_xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.1%} variância)')
        axes[0].set_ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.1%} variância)')
        plt.colorbar(scatter1, ax=axes[0])
        
        # Target real (se disponível)
        if target is not None:
            target_encoded = (target == '>50K').astype(int) if hasattr(target, 'str') else target
            scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=target_encoded, 
                                      cmap='RdYlBu', alpha=0.7, s=50)
            axes[1].set_title('Classes Reais (Salário)')
            axes[1].set_xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.1%} variância)')
            axes[1].set_ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.1%} variância)')
            plt.colorbar(scatter2, ax=axes[1])
        
        plt.tight_layout()
        
        # Salvar
        images_dir = Path("output/images")
        plt.savefig(images_dir / "clusters_pca_visualization.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info("📈 Visualizações PCA dos clusters salvas")
    
    def _save_clustering_results(self, clusters, best_k, best_silhouette):
        """Salvar resultados do clustering"""
        try:
            analysis_dir = Path("output/analysis")
            analysis_dir.mkdir(parents=True, exist_ok=True)
            
            # Estatísticas dos clusters
            unique_clusters, counts = np.unique(clusters, return_counts=True)
            
            results = []
            for cluster_id, count in zip(unique_clusters, counts):
                results.append({
                    'cluster_id': int(cluster_id),
                    'size': int(count),
                    'percentage': float(count / len(clusters) * 100)
                })
            
            results_df = pd.DataFrame(results)
            results_df.to_csv(analysis_dir / "clustering_results.csv", index=False)
            
            # Sumário geral
            summary = {
                'best_k': best_k,
                'silhouette_score': best_silhouette,
                'total_samples': len(clusters),
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            summary_df = pd.DataFrame([summary])
            summary_df.to_csv(analysis_dir / "clustering_summary.csv", index=False)
            
            logging.info("✅ Resultados de clustering salvos")
            
        except Exception as e:
            logging.error(f"❌ Erro ao salvar resultados: {e}")