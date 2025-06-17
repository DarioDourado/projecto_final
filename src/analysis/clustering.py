"""
🎯 Análise de Clustering Avançada para Dados Salariais
Implementa DBSCAN e K-Means com validação robusta
"""

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class SalaryClusteringAnalysis:
    """Análise completa de clustering para dados salariais"""
    
    def __init__(self, output_dir="output/analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.scaler = StandardScaler()
        self.results = {}
        
        print("🎯 Inicializando análise de clustering...")
    
    def prepare_data(self, df):
        """Preparar dados para clustering com validação robusta"""
        print(f"📊 Preparando dados: {len(df)} registros originais")
        
        # Selecionar apenas colunas numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remover colunas ID ou target se existirem
        exclude_cols = ['id', 'index', 'salary', 'income', 'target', 'y']
        numeric_cols = [col for col in numeric_cols if col.lower() not in exclude_cols]
        
        if len(numeric_cols) == 0:
            print("❌ Nenhuma coluna numérica encontrada para clustering")
            return None, None
        
        print(f"🔍 Colunas selecionadas para clustering: {numeric_cols}")
        
        # Extrair dados numéricos
        X = df[numeric_cols].copy()
        
        # Remover linhas com valores nulos
        X_clean = X.dropna()
        
        if len(X_clean) == 0:
            print("❌ Todos os registros têm valores nulos")
            return None, None
        
        if len(X_clean) < 10:
            print(f"⚠️ Poucos registros válidos ({len(X_clean)}). Clustering pode não ser efetivo.")
        
        # Normalizar dados
        X_scaled = self.scaler.fit_transform(X_clean)
        X_scaled_df = pd.DataFrame(X_scaled, columns=numeric_cols, index=X_clean.index)
        
        print(f"✅ Dados preparados: {len(X_scaled_df)} registros, {len(numeric_cols)} features")
        
        return X_scaled_df, numeric_cols
    
    def perform_dbscan_analysis(self, X):
        """Executar análise DBSCAN com parâmetros otimizados"""
        if X is None or len(X) == 0:
            print("❌ Dados inválidos para DBSCAN")
            return None
        
        print("🎯 Executando análise DBSCAN...")
        
        # Testar diferentes valores de eps
        eps_values = [0.3, 0.5, 0.7, 1.0, 1.5]
        min_samples_values = [3, 5, 10]
        
        best_params = None
        best_score = -1
        best_clusters = None
        
        results_list = []
        
        for eps in eps_values:
            for min_samples in min_samples_values:
                try:
                    # Executar DBSCAN
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    clusters = dbscan.fit_predict(X)
                    
                    # Calcular métricas
                    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
                    n_noise = list(clusters).count(-1)
                    noise_ratio = n_noise / len(clusters)
                    
                    # Silhouette score apenas se temos clusters válidos
                    silhouette = 0
                    if n_clusters > 1 and n_noise < len(clusters):
                        try:
                            silhouette = silhouette_score(X, clusters)
                        except:
                            silhouette = 0
                    
                    result = {
                        'eps': eps,
                        'min_samples': min_samples,
                        'n_clusters': n_clusters,
                        'n_noise': n_noise,
                        'noise_ratio': noise_ratio,
                        'silhouette': silhouette,
                        'clusters': clusters
                    }
                    
                    results_list.append(result)
                    
                    # Critério de seleção: balancear clusters e ruído
                    if n_clusters > 0 and noise_ratio < 0.5:
                        score = silhouette - (noise_ratio * 0.5)  # Penalizar muito ruído
                        if score > best_score:
                            best_score = score
                            best_params = result
                            best_clusters = clusters
                    
                    print(f"   • eps={eps:.1f}, min_samples={min_samples}: "
                          f"{n_clusters} clusters, {noise_ratio:.1%} ruído, "
                          f"silhouette={silhouette:.3f}")
                    
                except Exception as e:
                    print(f"   ❌ Erro com eps={eps}, min_samples={min_samples}: {e}")
                    continue
        
        if best_params is None:
            print("❌ DBSCAN não encontrou parâmetros válidos")
            # Usar parâmetros padrão como fallback
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            best_clusters = dbscan.fit_predict(X)
            n_clusters = len(set(best_clusters)) - (1 if -1 in best_clusters else 0)
            
            best_params = {
                'eps': 0.5,
                'min_samples': 5,
                'n_clusters': n_clusters,
                'n_noise': list(best_clusters).count(-1),
                'noise_ratio': list(best_clusters).count(-1) / len(best_clusters),
                'silhouette': 0,
                'clusters': best_clusters
            }
        
        # Salvar resultados detalhados
        self._save_dbscan_results(X, best_clusters, best_params, results_list)
        
        print(f"✅ DBSCAN concluído: {best_params['n_clusters']} clusters, "
              f"{best_params['noise_ratio']:.1%} ruído")
        
        return best_clusters, best_params['n_clusters'], best_params['silhouette']
    
    def _save_dbscan_results(self, X, clusters, best_params, all_results):
        """Salvar resultados DBSCAN em arquivos CSV"""
        try:
            # 1. Resultados principais com dados originais + clusters
            results_df = X.copy()
            results_df['cluster'] = clusters
            
            # Adicionar informações adicionais
            results_df['is_noise'] = (results_df['cluster'] == -1)
            results_df['cluster_size'] = results_df.groupby('cluster')['cluster'].transform('count')
            
            # Salvar arquivo principal
            results_path = self.output_dir / "dbscan_results.csv"
            results_df.to_csv(results_path, index=True)
            print(f"💾 Resultados DBSCAN salvos: {results_path}")
            
            # 2. Resumo dos clusters
            summary_data = []
            unique_clusters = sorted(set(clusters))
            
            for cluster_id in unique_clusters:
                cluster_mask = clusters == cluster_id
                cluster_size = sum(cluster_mask)
                
                if cluster_id == -1:
                    cluster_name = "Ruído"
                else:
                    cluster_name = f"Cluster {cluster_id}"
                
                summary_data.append({
                    'cluster_id': cluster_id,
                    'cluster_name': cluster_name,
                    'size': cluster_size,
                    'percentage': (cluster_size / len(clusters)) * 100,
                    'is_noise': cluster_id == -1
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_path = self.output_dir / "dbscan_summary.csv"
            summary_df.to_csv(summary_path, index=False)
            print(f"💾 Resumo DBSCAN salvo: {summary_path}")
            
            # 3. Parâmetros testados
            params_df = pd.DataFrame(all_results)
            params_path = self.output_dir / "dbscan_parameters.csv"
            params_df.to_csv(params_path, index=False)
            print(f"💾 Parâmetros DBSCAN salvos: {params_path}")
            
            # 4. Metadados
            metadata = {
                'algorithm': 'DBSCAN',
                'best_parameters': {k: v for k, v in best_params.items() if k != 'clusters'},
                'total_records': len(clusters),
                'features_used': list(X.columns),
                'execution_timestamp': pd.Timestamp.now().isoformat()
            }
            
            import json
            metadata_path = self.output_dir / "dbscan_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Metadados DBSCAN salvos: {metadata_path}")
            
        except Exception as e:
            print(f"❌ Erro ao salvar resultados DBSCAN: {e}")
    
    def perform_kmeans_analysis(self, X):
        """Executar análise K-Means com determinação automática do melhor K"""
        if X is None or len(X) == 0:
            print("❌ Dados inválidos para K-Means")
            return None, None
        
        print("🎯 Executando análise K-Means...")
        
        # Determinar range de K baseado no tamanho dos dados
        max_k = min(10, len(X) // 2)
        k_range = range(2, max_k + 1)
        
        if max_k < 2:
            print("❌ Dados insuficientes para K-Means (necessário pelo menos 4 registros)")
            return None, None
        
        inertias = []
        silhouette_scores = []
        
        for k in k_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X)
                
                inertias.append(kmeans.inertia_)
                
                if k > 1:
                    silhouette = silhouette_score(X, clusters)
                    silhouette_scores.append(silhouette)
                else:
                    silhouette_scores.append(0)
                
                print(f"   • K={k}: inertia={kmeans.inertia_:.2f}, silhouette={silhouette:.3f}")
                
            except Exception as e:
                print(f"   ❌ Erro com K={k}: {e}")
                continue
        
        # Encontrar melhor K usando método do cotovelo + silhouette
        if len(silhouette_scores) > 0:
            best_k = k_range[np.argmax(silhouette_scores)]
        else:
            best_k = 3 
        
        # Executar K-Means final
        kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        final_clusters = kmeans_final.fit_predict(X)
        
        # Salvar resultados K-Means
        self._save_kmeans_results(X, final_clusters, best_k, inertias, silhouette_scores, k_range)
        
        print(f"✅ K-Means concluído: {best_k} clusters")
        
        return final_clusters, best_k
    
    def _save_kmeans_results(self, X, clusters, best_k, inertias, silhouette_scores, k_range):
        """Salvar resultados K-Means"""
        try:
            # Resultados principais
            results_df = X.copy()
            results_df['cluster'] = clusters
            results_df['cluster_size'] = results_df.groupby('cluster')['cluster'].transform('count')
            
            results_path = self.output_dir / "kmeans_results.csv"
            results_df.to_csv(results_path, index=True)
            print(f"💾 K-Means resultados salvos: {results_path}")
            
            # Métricas por K
            metrics_data = []
            for i, k in enumerate(k_range):
                if i < len(inertias) and i < len(silhouette_scores):
                    metrics_data.append({
                        'k': k,
                        'inertia': inertias[i],
                        'silhouette': silhouette_scores[i],
                        'is_best': k == best_k
                    })
            
            metrics_df = pd.DataFrame(metrics_data)
            metrics_path = self.output_dir / "kmeans_metrics.csv"
            metrics_df.to_csv(metrics_path, index=False)
            print(f"💾 K-Means métricas salvas: {metrics_path}")
            
        except Exception as e:
            print(f"❌ Erro ao salvar K-Means: {e}")
    
    def run_complete_analysis(self, df):
        """Executar análise completa de clustering"""
        print("🚀 Iniciando análise completa de clustering...")
        
        # Preparar dados
        X, feature_names = self.prepare_data(df)
        
        if X is None:
            print("❌ Falha na preparação dos dados")
            return {}
        
        results = {}
        
        # DBSCAN
        try:
            dbscan_result = self.perform_dbscan_analysis(X)
            if dbscan_result:
                clusters, n_clusters, silhouette = dbscan_result
                results['dbscan'] = {
                    'clusters': clusters,
                    'n_clusters': n_clusters,
                    'silhouette': silhouette,
                    'algorithm': 'DBSCAN'
                }
        except Exception as e:
            print(f"❌ Erro no DBSCAN: {e}")
        
        # K-Means
        try:
            kmeans_result = self.perform_kmeans_analysis(X)
            if kmeans_result:
                clusters, best_k = kmeans_result
                results['kmeans'] = {
                    'clusters': clusters,
                    'best_k': best_k,
                    'algorithm': 'K-Means'
                }
        except Exception as e:
            print(f"❌ Erro no K-Means: {e}")
        
        # Comparação entre métodos
        if len(results) > 1:
            try:
                comparison = self.compare_clustering_methods(results)
                results['comparison'] = comparison
            except Exception as e:
                print(f"❌ Erro na comparação: {e}")
        
        print(f"✅ Análise de clustering concluída: {len(results)} métodos executados")
        
        return results
    
    def compare_clustering_methods(self, results):
        """Comparar diferentes métodos de clustering"""
        comparison_data = []
        
        for method, result in results.items():
            if method == 'comparison':
                continue
                
            if method == 'dbscan':
                comparison_data.append({
                    'method': 'DBSCAN',
                    'n_clusters': result.get('n_clusters', 0),
                    'silhouette': result.get('silhouette', 0),
                    'has_noise': True,
                    'noise_handling': 'Identifica outliers automaticamente'
                })
            elif method == 'kmeans':
                comparison_data.append({
                    'method': 'K-Means',
                    'n_clusters': result.get('best_k', 0),
                    'silhouette': 0,
                    'has_noise': False,
                    'noise_handling': 'Não identifica outliers'
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            comparison_path = self.output_dir / "clustering_comparison.csv"
            comparison_df.to_csv(comparison_path, index=False)
            print(f"💾 Comparação salva: {comparison_path}")
            
            return comparison_df
        
        return None

def main():
    """Função de teste"""
    print("🧪 Testando SalaryClusteringAnalysis...")
    
    # Dados de teste
    np.random.seed(42)
    test_data = pd.DataFrame({
        'age': np.random.randint(20, 65, 100),
        'experience': np.random.randint(0, 40, 100),
        'education_level': np.random.randint(1, 5, 100),
        'salary': np.random.randint(30000, 120000, 100)
    })
    
    clustering = SalaryClusteringAnalysis()
    results = clustering.run_complete_analysis(test_data)
    
    print(f"✅ Teste concluído: {len(results)} resultados")

if __name__ == "__main__":
    main()