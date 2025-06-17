import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from datetime import datetime

class ClusteringAnalises:
    """Classe para análises comparativas de clustering"""
    
    def __init__(self):
        self.results = {}
        self.comparison_data = []
    
    def add_clustering_result(self, method_name, clusters, n_clusters, silhouette_score, 
                            has_noise=False, noise_percentage=0, additional_metrics=None):
        """Adicionar resultado de clustering para comparação"""
        self.results[method_name] = {
            'clusters': clusters,
            'n_clusters': n_clusters,
            'silhouette_score': silhouette_score,
            'has_noise': has_noise,
            'noise_percentage': noise_percentage,
            'additional_metrics': additional_metrics or {}
        }
        
        # Adicionar aos dados de comparação
        self.comparison_data.append({
            'Método': method_name,
            'N° Clusters': n_clusters,
            'Silhouette Score': round(silhouette_score, 3),
            'Detecta Ruído': 'Sim' if has_noise else 'Não',
            'Percentual de Ruído': f"{noise_percentage:.1f}%"
        })
        
        logging.info(f"✅ Resultado {method_name} adicionado: {n_clusters} clusters, "
                    f"Silhouette: {silhouette_score:.3f}")
    
    def generate_comparison_report(self, output_dir="output/analysis"):
        """Gerar relatório comparativo detalhado"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if not self.comparison_data:
            logging.warning("⚠️ Nenhum resultado para comparar")
            return None
        
        # Criar DataFrame
        comparison_df = pd.DataFrame(self.comparison_data)
        
        # Salvar CSV
        csv_path = output_path / "clustering_comparison.csv"
        comparison_df.to_csv(csv_path, index=False)
        
        # Gerar relatório markdown
        report_path = output_path / "clustering_comparison_report.md"
        self._create_markdown_report(comparison_df, report_path)
        
        # Gerar visualizações
        self._create_comparison_charts(comparison_df, output_path)
        
        logging.info(f"✅ Relatório comparativo salvo em: {report_path}")
        return report_path
    
    def _create_markdown_report(self, comparison_df, report_path):
        """Criar relatório markdown detalhado"""
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 🎯 RELATÓRIO COMPARATIVO - CLUSTERING\n\n")
            f.write(f"**Data de Análise:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Tabela de resultados
            f.write("## 📊 Comparação de Métodos\n\n")
            f.write(comparison_df.to_markdown(index=False))
            f.write("\n\n")
            
            # Análise detalhada
            f.write("## 🔍 Análise Detalhada\n\n")
            
            # Determinar melhor método
            best_method = comparison_df.loc[comparison_df['Silhouette Score'].idxmax(), 'Método']
            best_score = comparison_df['Silhouette Score'].max()
            
            f.write(f"### 🏆 Melhor Método: {best_method}\n")
            f.write(f"- **Silhouette Score:** {best_score:.3f}\n")
            f.write(f"- **Número de Clusters:** {comparison_df[comparison_df['Método'] == best_method]['N° Clusters'].iloc[0]}\n\n")
            
            # Análise por método
            for _, row in comparison_df.iterrows():
                method = row['Método']
                f.write(f"### 📈 {method}\n")
                f.write(f"- **Clusters:** {row['N° Clusters']}\n")
                f.write(f"- **Silhouette Score:** {row['Silhouette Score']}\n")
                f.write(f"- **Detecta Ruído:** {row['Detecta Ruído']}\n")
                if row['Percentual de Ruído'] != '0.0%':
                    f.write(f"- **Ruído:** {row['Percentual de Ruído']}\n")
                
                # Características específicas
                if method == 'K-Means':
                    f.write("- **Características:** Eficiente, clusters esféricos, requer K pré-definido\n")
                    f.write("- **Melhor para:** Dados com clusters bem separados e balanceados\n")
                elif method == 'DBSCAN':
                    f.write("- **Características:** Detecta outliers, clusters de formas arbitrárias\n")
                    f.write("- **Melhor para:** Dados com ruído e clusters de densidades variadas\n")
                
                f.write("\n")
            
            # Recomendações
            f.write("## 💡 Recomendações\n\n")
            f.write("### Quando usar cada método:\n\n")
            f.write("**K-Means:**\n")
            f.write("- Análise exploratória rápida\n")
            f.write("- Clusters esféricos e bem definidos\n")
            f.write("- Dados sem muito ruído\n")
            f.write("- Quando você tem uma ideia do número de clusters\n\n")
            
            f.write("**DBSCAN:**\n")
            f.write("- Detecção de outliers/anomalias\n")
            f.write("- Clusters de formas irregulares\n")
            f.write("- Dados com ruído significativo\n")
            f.write("- Quando o número de clusters é desconhecido\n\n")
            
            f.write("### 🎯 Conclusão:\n")
            f.write("A combinação de ambos os métodos oferece uma visão mais completa dos dados, "
                   "permitindo identificar tanto padrões estruturais quanto anomalias.\n")
    
    def _create_comparison_charts(self, comparison_df, output_path):
        """Criar gráficos comparativos"""
        try:
            # Gráfico de Silhouette Scores
            plt.figure(figsize=(12, 6))
            
            methods = comparison_df['Método']
            scores = comparison_df['Silhouette Score']
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(methods)]
            
            bars = plt.bar(methods, scores, color=colors, alpha=0.7, edgecolor='black')
            plt.title('📊 Comparação de Silhouette Scores', fontsize=16, fontweight='bold')
            plt.xlabel('Método de Clustering', fontsize=12)
            plt.ylabel('Silhouette Score', fontsize=12)
            plt.grid(axis='y', alpha=0.3)
            
            # Adicionar valores nas barras
            for bar, score in zip(bars, scores):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(output_path / "clustering_comparison_chart.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Gráfico de Número de Clusters
            plt.figure(figsize=(10, 6))
            
            n_clusters = comparison_df['N° Clusters']
            colors_clusters = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'][:len(methods)]
            
            bars = plt.bar(methods, n_clusters, color=colors_clusters, alpha=0.7, edgecolor='black')
            plt.title('🎯 Número de Clusters por Método', fontsize=16, fontweight='bold')
            plt.xlabel('Método de Clustering', fontsize=12)
            plt.ylabel('Número de Clusters', fontsize=12)
            plt.grid(axis='y', alpha=0.3)
            
            # Adicionar valores nas barras
            for bar, clusters in zip(bars, n_clusters):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        f'{clusters}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(output_path / "clustering_n_clusters_chart.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info("📈 Gráficos comparativos salvos")
            
        except Exception as e:
            logging.error(f"❌ Erro ao criar gráficos: {e}")
    
    def get_best_method(self):
        """Retornar o melhor método baseado no Silhouette Score"""
        if not self.comparison_data:
            return None
        
        comparison_df = pd.DataFrame(self.comparison_data)
        best_idx = comparison_df['Silhouette Score'].idxmax()
        best_method = comparison_df.loc[best_idx, 'Método']
        best_score = comparison_df.loc[best_idx, 'Silhouette Score']
        
        return {
            'method': best_method,
            'score': best_score,
            'details': self.results.get(best_method, {})
        }
    
    def export_results_to_csv(self, output_path="output/analysis/clustering_results_detailed.csv"):
        """Exportar resultados detalhados para CSV"""
        detailed_results = []
        
        for method, result in self.results.items():
            detailed_results.append({
                'Método': method,
                'N° Clusters': result['n_clusters'],
                'Silhouette Score': result['silhouette_score'],
                'Detecta Ruído': result['has_noise'],
                'Percentual de Ruído': result['noise_percentage'],
                'Total de Pontos': len(result['clusters']) if result['clusters'] is not None else 0,
                'Timestamp': datetime.now().isoformat()
            })
        
        df = pd.DataFrame(detailed_results)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        logging.info(f"✅ Resultados detalhados exportados para: {output_path}")
        return output_path