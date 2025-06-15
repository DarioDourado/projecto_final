"""Análise de Clustering para Segmentação Salarial - VERSÃO COMPLETA"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from pathlib import Path
import logging
import warnings
from src.analysis.clustering_comparation import ClusteringAnalises

warnings.filterwarnings('ignore')

class SalaryClusteringAnalysis:
    """Análise de clustering para segmentação de perfis salariais"""
    
    def __init__(self):
        self.kmeans_model = None
        self.dbscan_model = None
        self.pca = None
        self.scaler = StandardScaler()
        self.silhouette_scores = {}
        self.clustering_analises = ClusteringAnalises()

    def perform_kmeans_analysis(self, X, max_k=10):
        """Análise K-Means com método do cotovelo e silhouette"""
        logging.info("🔍 Iniciando análise K-Means...")
        
        # Normalizar dados
        X_scaled = self.scaler.fit_transform(X)
        
        # Análise para diferentes valores de K
        k_range = range(2, max_k + 1)
        inertias = []
        silhouette_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            
            inertias.append(kmeans.inertia_)
            sil_score = silhouette_score(X_scaled, clusters)
            silhouette_scores.append(sil_score)
            
            logging.info(f"  📊 K={k}: Silhouette Score = {sil_score:.3f}")
        
        # Encontrar melhor K
        best_k = k_range[np.argmax(silhouette_scores)]
        best_silhouette = max(silhouette_scores)
        
        # Treinar modelo final
        self.kmeans_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        final_clusters = self.kmeans_model.fit_predict(X_scaled)
        
        # Salvar scores
        self.silhouette_scores['kmeans'] = silhouette_scores
        
        logging.info(f"✅ Melhor K-Means: K={best_k}, Silhouette: {best_silhouette:.3f}")
        
        # Plotar análises
        self._plot_kmeans_analysis(k_range, inertias, silhouette_scores)
        
        return final_clusters, best_k

    def perform_dbscan_analysis(self, X, eps_range=None, min_samples_range=None):
        """Análise DBSCAN com otimização de parâmetros"""
        logging.info("🔍 Iniciando análise DBSCAN...")
        
        # Valores padrão para otimização
        if eps_range is None:
            eps_range = np.arange(0.3, 2.0, 0.1)
        if min_samples_range is None:
            min_samples_range = range(3, 10)
        
        # Verificar e preparar dados
        try:
            if isinstance(X, pd.DataFrame):
                X_work = X.values
            else:
                X_work = X
            
            # Normalizar dados
            X_scaled = StandardScaler().fit_transform(X_work)
            
        except Exception as e:
            logging.error(f"❌ Erro na preparação dos dados para DBSCAN: {e}")
            return None, 0, -1
        
        best_eps = 0.5
        best_min_samples = 5
        best_silhouette = -1
        best_n_clusters = 0
        results = []
        
        # Grid search para encontrar melhores parâmetros
        for eps in eps_range:
            for min_samples in min_samples_range:
                try:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    clusters = dbscan.fit_predict(X_scaled)
                    
                    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
                    n_noise = list(clusters).count(-1)
                    
                    if n_clusters > 1:
                        # Calcular silhouette apenas para pontos não-noise
                        if n_noise < len(clusters):
                            mask = clusters != -1
                            if mask.sum() > 1:
                                silhouette = silhouette_score(X_scaled[mask], clusters[mask])
                            else:
                                silhouette = -1
                        else:
                            silhouette = -1
                    else:
                        silhouette = -1
                    
                    results.append({
                        'eps': eps,
                        'min_samples': min_samples,
                        'n_clusters': n_clusters,
                        'n_noise': n_noise,
                        'silhouette_score': silhouette
                    })
                    
                    if silhouette > best_silhouette and n_clusters > 1:
                        best_silhouette = silhouette
                        best_eps = eps
                        best_min_samples = min_samples
                        best_n_clusters = n_clusters
                        
                    logging.info(f"  📊 eps={eps:.2f}, min_samples={min_samples}: "
                               f"{n_clusters} clusters, {n_noise} noise, "
                               f"Silhouette={silhouette:.3f}")
                               
                except Exception as e:
                    logging.warning(f"⚠️ Erro com eps={eps:.2f}, min_samples={min_samples}: {e}")
                    continue
        
        # Treinar modelo final com melhores parâmetros
        if best_silhouette > -1:
            self.dbscan_model = DBSCAN(eps=best_eps, min_samples=best_min_samples)
            final_clusters = self.dbscan_model.fit_predict(X_scaled)
            
            logging.info(f"✅ Melhor DBSCAN: eps={best_eps:.2f}, min_samples={best_min_samples}")
            logging.info(f"✅ {best_n_clusters} clusters, Silhouette: {best_silhouette:.3f}")
            
            # Plotar análise de parâmetros
            self._plot_dbscan_analysis(results)
            
            # Salvar resultados
            self._save_dbscan_results(final_clusters, best_eps, best_min_samples, best_silhouette)
            
            return final_clusters, best_n_clusters, best_silhouette
        else:
            logging.warning("⚠️ Não foi possível encontrar clustering válido com DBSCAN")
            return None, 0, -1

    def _plot_kmeans_analysis(self, k_range, inertias, silhouette_scores):
        """Plotar análise K-Means"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Método do cotovelo
        axes[0].plot(k_range, inertias, 'bo-')
        axes[0].set_xlabel('Número de Clusters (K)')
        axes[0].set_ylabel('Inércia')
        axes[0].set_title('Método do Cotovelo')
        axes[0].grid(True, alpha=0.3)
        
        # Silhouette Score
        axes[1].plot(k_range, silhouette_scores, 'ro-')
        axes[1].set_xlabel('Número de Clusters (K)')
        axes[1].set_ylabel('Silhouette Score')
        axes[1].set_title('Análise Silhouette Score')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Salvar
        images_dir = Path("output/images")
        images_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(images_dir / "kmeans_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info("📈 Gráficos de análise K-Means salvos")

    def _plot_dbscan_analysis(self, results):
        """Plotar análise de parâmetros DBSCAN"""
        results_df = pd.DataFrame(results)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Heatmap Silhouette Score
        pivot_silhouette = results_df.pivot(index='min_samples', columns='eps', values='silhouette_score')
        sns.heatmap(pivot_silhouette, annot=True, cmap='viridis', ax=axes[0,0], fmt='.3f')
        axes[0,0].set_title('Silhouette Score por Parâmetros')
        axes[0,0].set_xlabel('eps')
        axes[0,0].set_ylabel('min_samples')
        
        # 2. Heatmap Número de Clusters
        pivot_clusters = results_df.pivot(index='min_samples', columns='eps', values='n_clusters')
        sns.heatmap(pivot_clusters, annot=True, cmap='plasma', ax=axes[0,1], fmt='d')
        axes[0,1].set_title('Número de Clusters por Parâmetros')
        axes[0,1].set_xlabel('eps')
        axes[0,1].set_ylabel('min_samples')
        
        # 3. Scatter plot eps vs silhouette
        valid_results = results_df[results_df['silhouette_score'] > -1]
        if not valid_results.empty:
            scatter = axes[1,0].scatter(valid_results['eps'], valid_results['silhouette_score'], 
                                      c=valid_results['n_clusters'], cmap='viridis', s=50)
            axes[1,0].set_xlabel('eps')
            axes[1,0].set_ylabel('Silhouette Score')
            axes[1,0].set_title('eps vs Silhouette Score')
            plt.colorbar(scatter, ax=axes[1,0], label='N° Clusters')
        
        # 4. Distribuição de ruído
        axes[1,1].scatter(results_df['eps'], results_df['n_noise'], alpha=0.6)
        axes[1,1].set_xlabel('eps')
        axes[1,1].set_ylabel('Pontos de Ruído')
        axes[1,1].set_title('eps vs Pontos de Ruído')
        
        plt.tight_layout()
        
        # Salvar
        images_dir = Path("output/images")
        images_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(images_dir / "dbscan_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info("📈 Gráficos de análise DBSCAN salvos")

    def _save_dbscan_results(self, clusters, best_eps, best_min_samples, best_silhouette):
        """Salvar resultados DBSCAN"""
        try:
            analysis_dir = Path("output/analysis")
            analysis_dir.mkdir(parents=True, exist_ok=True)
            
            # Estatísticas dos clusters
            unique_clusters = np.unique(clusters)
            n_noise = list(clusters).count(-1)
            n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
            
            results = []
            for cluster_id in unique_clusters:
                if cluster_id != -1:  # Excluir ruído
                    count = list(clusters).count(cluster_id)
                    results.append({
                        'cluster_id': int(cluster_id),
                        'size': int(count),
                        'percentage': float(count / len(clusters) * 100)
                    })
            
            # Adicionar estatísticas de ruído
            if n_noise > 0:
                results.append({
                    'cluster_id': -1,
                    'size': n_noise,
                    'percentage': float(n_noise / len(clusters) * 100)
                })
            
            results_df = pd.DataFrame(results)
            results_df.to_csv(analysis_dir / "dbscan_results.csv", index=False)
            
            # Sumário geral
            summary = {
                'best_eps': best_eps,
                'best_min_samples': best_min_samples,
                'silhouette_score': best_silhouette,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'total_samples': len(clusters),
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            summary_df = pd.DataFrame([summary])
            summary_df.to_csv(analysis_dir / "dbscan_summary.csv", index=False)
            
            logging.info("✅ Resultados DBSCAN salvos")
            
        except Exception as e:
            logging.error(f"❌ Erro ao salvar resultados DBSCAN: {e}")

    def visualize_dbscan_clusters(self, X, clusters):
        """Visualizar clusters DBSCAN com PCA 2D"""
        logging.info("🎨 Gerando visualizações PCA dos clusters DBSCAN...")
        
        # Aplicar PCA
        self.pca = PCA(n_components=2, random_state=42)
        X_pca = self.pca.fit_transform(X)
        
        plt.figure(figsize=(12, 8))
        
        # Separar pontos de ruído
        unique_clusters = np.unique(clusters)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
        
        for i, cluster_id in enumerate(unique_clusters):
            mask = clusters == cluster_id
            if cluster_id == -1:
                # Pontos de ruído em preto
                plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                           c='black', marker='x', s=50, alpha=0.6, label='Ruído')
            else:
                plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                           c=[colors[i]], s=50, alpha=0.7, label=f'Cluster {cluster_id}')
        
        plt.xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.1%} variância)')
        plt.ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.1%} variância)')
        plt.title('Clusters DBSCAN - Visualização PCA')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Salvar
        images_dir = Path("output/images")
        images_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(images_dir / "dbscan_clusters_pca.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info("📈 Visualizações PCA dos clusters DBSCAN salvas")

    def compare_clustering_methods(self, X):
        """Comparar K-Means vs DBSCAN"""
        logging.info("🔍 Comparando métodos de clustering...")
        
        results = {}
        
        # K-Means
        try:
            kmeans_clusters, kmeans_k = self.perform_kmeans_analysis(X)
            kmeans_silhouette = max(self.silhouette_scores.get('kmeans', [0])) if self.silhouette_scores.get('kmeans') else 0
            results['K-Means'] = {
                'clusters': kmeans_clusters,
                'n_clusters': kmeans_k,
                'silhouette_score': kmeans_silhouette,
                'has_noise': False,
                'noise_percentage': 0
            }
            logging.info(f"✅ K-Means: {kmeans_k} clusters, Silhouette: {kmeans_silhouette:.3f}")
        except Exception as e:
            logging.error(f"❌ Erro no K-Means: {e}")
            results['K-Means'] = None
        
        # DBSCAN
        try:
            dbscan_result = self.perform_dbscan_analysis(X)
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
                    logging.info(f"✅ DBSCAN: {dbscan_k} clusters, Silhouette: {dbscan_silhouette:.3f}, Ruído: {results['DBSCAN']['noise_percentage']:.1f}%")
                else:
                    results['DBSCAN'] = None
            else:
                results['DBSCAN'] = None
        except Exception as e:
            logging.error(f"❌ Erro no DBSCAN: {e}")
            results['DBSCAN'] = None
        
        # Gerar relatório comparativo
        self._generate_clustering_comparison_report(results)
        
        # Salvar no clustering_analises para comparação
        self.clustering_analises.results = results
        
        return results

    def _generate_clustering_comparison_report(self, results):
        """Gerar relatório comparativo entre métodos"""
        try:
            comparison_data = []
            
            for method, result in results.items():
                if result is not None:
                    comparison_data.append({
                        'Método': method,
                        'N° Clusters': result['n_clusters'],
                        'Silhouette Score': result['silhouette_score'],
                        'Detecta Ruído': 'Sim' if result['has_noise'] else 'Não',
                        'Percentual de Ruído': f"{result['noise_percentage']:.1f}%"
                    })
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                
                # Salvar CSV
                analysis_dir = Path("output/analysis")
                analysis_dir.mkdir(parents=True, exist_ok=True)
                comparison_df.to_csv(analysis_dir / "clustering_comparison.csv", index=False)
                
                # Gerar relatório markdown
                report_path = analysis_dir / "clustering_comparison_report.md"
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write("# RELATÓRIO COMPARATIVO - CLUSTERING\n\n")
                    f.write(f"**Data:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write("## 📊 Comparação de Métodos\n\n")
                    f.write(comparison_df.to_markdown(index=False))
                    f.write("\n\n## 🏆 Recomendações\n\n")
                    
                    # Determinar melhor método
                    best_method = comparison_df.loc[comparison_df['Silhouette Score'].idxmax(), 'Método']
                    best_score = comparison_df['Silhouette Score'].max()
                    
                    f.write(f"**Melhor método:** {best_method} (Silhouette Score: {best_score:.3f})\n\n")
                    
                    if 'DBSCAN' in [r['Método'] for r in comparison_data]:
                        dbscan_result = next(r for r in comparison_data if r['Método'] == 'DBSCAN')
                        f.write(f"**DBSCAN:** Detectou {dbscan_result['Percentual de Ruído']} de pontos de ruído\n")
                    
                    f.write("\n### 📝 Observações:\n")
                    f.write("- **K-Means:** Melhor para clusters esféricos e balanceados\n")
                    f.write("- **DBSCAN:** Melhor para clusters de formas arbitrárias e detecção de outliers\n")
                
                logging.info("✅ Relatório comparativo de clustering gerado")
            
        except Exception as e:
            logging.error(f"❌ Erro ao gerar relatório comparativo: {e}")

    def visualize_clusters_pca(self, X, clusters, target=None):
        """Visualizar clusters com PCA 2D"""
        logging.info("🎨 Gerando visualizações PCA dos clusters...")
        
        try:
            # Aplicar PCA
            self.pca = PCA(n_components=2, random_state=42)
            if isinstance(X, pd.DataFrame):
                X_pca = self.pca.fit_transform(X.values)
            else:
                X_pca = self.pca.fit_transform(X)
            
            fig, axes = plt.subplots(1, 2 if target is not None else 1, figsize=(15, 6))
            if target is None:
                axes = [axes]
            
            # Plot 1: Clusters
            unique_clusters = np.unique(clusters)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
            
            for i, cluster_id in enumerate(unique_clusters):
                mask = clusters == cluster_id
                label = f'Cluster {cluster_id}' if cluster_id != -1 else 'Ruído'
                color = 'black' if cluster_id == -1 else colors[i]
                marker = 'x' if cluster_id == -1 else 'o'
                
                axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                               c=[color], s=50, alpha=0.7, label=label, marker=marker)
            
            axes[0].set_xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.1%} variância)')
            axes[0].set_ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.1%} variância)')
            axes[0].set_title('Clusters Identificados - Visualização PCA')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Plot 2: Target (se disponível)
            if target is not None:
                try:
                    if hasattr(target, 'map'):
                        target_encoded = target.map({'>50K': 1, '<=50K': 0})
                    else:
                        target_encoded = target
                    
                    scatter = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], 
                                            c=target_encoded, cmap='RdYlBu', alpha=0.7)
                    axes[1].set_xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.1%} variância)')
                    axes[1].set_ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.1%} variância)')
                    axes[1].set_title('Classes Reais (Target)')
                    plt.colorbar(scatter, ax=axes[1])
                    axes[1].grid(True, alpha=0.3)
                except Exception as e:
                    logging.warning(f"⚠️ Erro ao plotar target: {e}")
            
            plt.tight_layout()
            
            # Salvar
            images_dir = Path("output/images")
            images_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(images_dir / "clusters_pca_visualization.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info("📈 Visualizações PCA dos clusters salvas")
            
        except Exception as e:
            logging.error(f"❌ Erro na visualização PCA: {e}")