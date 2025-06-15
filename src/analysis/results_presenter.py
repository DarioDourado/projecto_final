"""
Sistema de apresentação de resultados com tratamento robusto
para dados DBSCAN, APRIORI, FP-GROWTH, ECLAT
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ResultsPresenter:
    """Apresentador robusto de resultados dos algoritmos"""
    
    def __init__(self, analysis_dir="output/analysis"):
        self.analysis_dir = Path(analysis_dir)
        self.results = {}
        self.comparisons = {}
        
        # Configurar plots
        plt.style.use('default')  # Usar estilo mais compatível
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
        print("🎨 Inicializando apresentador de resultados...")
        self._load_all_results()
    
    def _load_all_results(self):
        """Carregar todos os resultados com validação robusta"""
        try:
            # DBSCAN Results
            dbscan_files = {
                'results': self.analysis_dir / "dbscan_results.csv",
                'summary': self.analysis_dir / "dbscan_summary.csv",
                'parameters': self.analysis_dir / "dbscan_parameters.csv"
            }
            
            for file_type, file_path in dbscan_files.items():
                if file_path.exists():
                    try:
                        df = pd.read_csv(file_path)
                        # Verificar se não está vazio
                        if len(df) > 0:
                            self.results[f'dbscan_{file_type}'] = df
                            print(f"✅ DBSCAN {file_type} carregado: {len(df):,} registros")
                        else:
                            print(f"⚠️ DBSCAN {file_type} está vazio")
                    except Exception as e:
                        print(f"❌ Erro ao carregar DBSCAN {file_type}: {e}")
            
            # Association Rules Results
            algorithms = ['apriori', 'fp_growth', 'eclat']
            for alg in algorithms:
                file_path = self.analysis_dir / f"{alg}_rules.csv"
                if file_path.exists():
                    try:
                        df = pd.read_csv(file_path)
                        if len(df) > 0:
                            self.results[alg] = df
                            print(f"✅ {alg.upper()} carregado: {len(df):,} regras")
                        else:
                            print(f"⚠️ {alg.upper()} está vazio")
                    except Exception as e:
                        print(f"❌ Erro ao carregar {alg}: {e}")
            
            # K-Means se existir
            kmeans_files = ['kmeans_results.csv', 'kmeans_metrics.csv']
            for file_name in kmeans_files:
                file_path = self.analysis_dir / file_name
                if file_path.exists():
                    try:
                        df = pd.read_csv(file_path)
                        if len(df) > 0:
                            key = file_name.replace('.csv', '').replace('_', '_')
                            self.results[key] = df
                            print(f"✅ {key} carregado: {len(df):,} registros")
                    except Exception as e:
                        print(f"❌ Erro ao carregar {file_name}: {e}")
                        
        except Exception as e:
            print(f"❌ Erro geral ao carregar resultados: {e}")
    
    def present_dbscan_results(self):
        """Apresentar resultados detalhados do DBSCAN com validação"""
        print("\n" + "="*80)
        print("🎯 ANÁLISE DBSCAN - CLUSTERING DE SALÁRIOS")
        print("="*80)
        
        # Verificar se temos dados DBSCAN
        if 'dbscan_results' not in self.results:
            print("❌ Resultados DBSCAN não encontrados")
            print("💡 Verifique se o arquivo output/analysis/dbscan_results.csv existe")
            return
        
        df = self.results['dbscan_results']
        
        # Verificar se a coluna cluster existe
        if 'cluster' not in df.columns:
            print("❌ Coluna 'cluster' não encontrada nos resultados DBSCAN")
            print(f"📋 Colunas disponíveis: {list(df.columns)}")
            
            # Tentar usar dbscan_summary se disponível
            if 'dbscan_summary' in self.results:
                print("🔄 Usando dados do summary...")
                self._present_dbscan_from_summary()
                return
            else:
                print("❌ Sem dados alternativos disponíveis")
                return
        
        # Estatísticas básicas
        print(f"📊 ESTATÍSTICAS BÁSICAS:")
        print(f"   • Total de registros: {len(df):,}")
        
        # Análise de clusters
        cluster_counts = df['cluster'].value_counts().sort_index()
        n_clusters = len(cluster_counts) - (1 if -1 in cluster_counts.index else 0)
        noise_count = cluster_counts.get(-1, 0)
        noise_rate = (noise_count / len(df)) * 100
        
        print(f"   • Clusters únicos: {n_clusters}")
        print(f"   • Pontos de ruído (-1): {noise_count:,}")
        print(f"   • Taxa de ruído: {noise_rate:.2f}%")
        
        # Distribuição detalhada por cluster
        print(f"\n📈 DISTRIBUIÇÃO POR CLUSTER:")
        for cluster_id in sorted(cluster_counts.index):
            count = cluster_counts[cluster_id]
            percentage = (count / len(df)) * 100
            
            if cluster_id == -1:
                cluster_name = "RUÍDO/OUTLIERS"
                icon = "🔴"
            else:
                cluster_name = f"Cluster {cluster_id}"
                icon = "🔵" if cluster_id == 0 else "🟡"
            
            print(f"   {icon} {cluster_name}: {count:,} registros ({percentage:.2f}%)")
        
        # Análise de qualidade
        print(f"\n🎯 ANÁLISE DE QUALIDADE:")
        
        if n_clusters == 1:
            print("   ⚠️ Apenas 1 cluster encontrado - dados muito homogêneos")
        elif n_clusters > 10:
            print("   ⚠️ Muitos clusters encontrados - possível overfitting")
        else:
            print(f"   ✅ Número de clusters adequado: {n_clusters}")
        
        if noise_rate > 10:
            print(f"   ⚠️ Taxa de ruído alta ({noise_rate:.1f}%) - revisar parâmetros")
        elif noise_rate > 5:
            print(f"   💡 Taxa de ruído moderada ({noise_rate:.1f}%)")
        else:
            print(f"   ✅ Taxa de ruído baixa ({noise_rate:.1f}%)")
        
        # Características dos clusters principais
        self._analyze_cluster_characteristics(df)
        
        # Criar visualizações
        self._create_dbscan_plots(df)
    
    def _present_dbscan_from_summary(self):
        """Apresentar DBSCAN usando arquivo de summary"""
        summary_df = self.results['dbscan_summary']
        
        print(f"📊 ESTATÍSTICAS BÁSICAS (do summary):")
        total_records = summary_df['size'].sum()
        print(f"   • Total de registros: {total_records:,}")
        
        n_clusters = len(summary_df[summary_df['cluster_id'] != -1])
        noise_records = summary_df[summary_df['cluster_id'] == -1]['size'].sum() if -1 in summary_df['cluster_id'].values else 0
        noise_rate = (noise_records / total_records) * 100
        
        print(f"   • Clusters únicos: {n_clusters}")
        print(f"   • Pontos de ruído: {noise_records:,}")
        print(f"   • Taxa de ruído: {noise_rate:.2f}%")
        
        print(f"\n📈 DISTRIBUIÇÃO POR CLUSTER:")
        for _, row in summary_df.iterrows():
            cluster_id = row['cluster_id']
            size = row['size']
            percentage = row['percentage']
            
            if pd.isna(cluster_id):  # Linha vazia
                continue
                
            if cluster_id == -1:
                cluster_name = "RUÍDO/OUTLIERS"
                icon = "🔴"
            else:
                cluster_name = f"Cluster {int(cluster_id)}"
                icon = "🔵" if cluster_id == 0 else "🟡"
            
            print(f"   {icon} {cluster_name}: {size:,} registros ({percentage:.2f}%)")
    
    def _analyze_cluster_characteristics(self, df):
        """Analisar características dos clusters"""
        print(f"\n🔍 CARACTERÍSTICAS DOS CLUSTERS:")
        
        # Colunas numéricas para análise (excluindo cluster)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'cluster' in numeric_cols:
            numeric_cols.remove('cluster')
        
        if len(numeric_cols) == 0:
            print("   ⚠️ Nenhuma variável numérica disponível para análise")
            return
        
        # Analisar clusters principais (não ruído)
        main_clusters = df[df['cluster'] != -1]['cluster'].unique()
        
        for cluster_id in sorted(main_clusters):
            cluster_data = df[df['cluster'] == cluster_id]
            
            print(f"\n   📋 CLUSTER {cluster_id} ({len(cluster_data):,} registros):")
            
            # Top 3 características numéricas
            for col in numeric_cols[:3]:
                if col in cluster_data.columns:
                    mean_val = cluster_data[col].mean()
                    std_val = cluster_data[col].std()
                    print(f"     - {col}: μ={mean_val:.2f} ± {std_val:.2f}")
    
    def _create_dbscan_plots(self, df):
        """Criar visualizações para DBSCAN"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('🎯 ANÁLISE DBSCAN - CLUSTERING DE SALÁRIOS', fontsize=16, fontweight='bold')
            
            # 1. Distribuição de clusters
            cluster_counts = df['cluster'].value_counts().sort_index()
            colors = ['red' if x == -1 else 'skyblue' for x in cluster_counts.index]
            
            bars = axes[0,0].bar(range(len(cluster_counts)), cluster_counts.values, color=colors)
            axes[0,0].set_title('Distribuição de Clusters')
            axes[0,0].set_ylabel('Número de Registros')
            
            # Labels para as barras
            labels = ['Ruído' if x == -1 else f'C{x}' for x in cluster_counts.index]
            axes[0,0].set_xticks(range(len(cluster_counts)))
            axes[0,0].set_xticklabels(labels)
            
            # Adicionar valores nas barras
            for bar, count in zip(bars, cluster_counts.values):
                height = bar.get_height()
                axes[0,0].text(bar.get_x() + bar.get_width()/2., height,
                              f'{count:,}', ha='center', va='bottom')
            
            # 2. Taxa de ruído (Pizza)
            noise_count = (df['cluster'] == -1).sum()
            valid_count = len(df) - noise_count
            
            if noise_count > 0:
                sizes = [valid_count, noise_count]
                labels_pie = ['Clustered', 'Noise']
                colors_pie = ['lightgreen', 'lightcoral']
                
                axes[0,1].pie(sizes, labels=labels_pie, colors=colors_pie, autopct='%1.1f%%')
                axes[0,1].set_title('Taxa de Ruído vs Clustered')
            else:
                axes[0,1].text(0.5, 0.5, 'Sem Ruído\nDetectado', 
                              ha='center', va='center', fontsize=14)
                axes[0,1].set_title('Taxa de Ruído')
            
            # 3. Distribuição por tamanho do cluster
            cluster_sizes = df.groupby('cluster').size().sort_values(ascending=False)
            
            if len(cluster_sizes) > 1:
                axes[1,0].bar(range(len(cluster_sizes)), cluster_sizes.values, 
                             color=['skyblue' if x != -1 else 'red' for x in cluster_sizes.index])
                axes[1,0].set_title('Tamanho dos Clusters')
                axes[1,0].set_ylabel('Número de Registros')
                axes[1,0].set_xlabel('Clusters (ordenados por tamanho)')
                
                # Log scale se necessário
                if max(cluster_sizes.values) / min(cluster_sizes.values) > 100:
                    axes[1,0].set_yscale('log')
                    axes[1,0].set_title('Tamanho dos Clusters (escala log)')
            
            # 4. Resumo textual
            summary_text = f"""RESUMO DBSCAN:

• Total: {len(df):,} registros
• Clusters: {len(df['cluster'].unique()) - (1 if -1 in df['cluster'].values else 0)}
• Ruído: {(df['cluster'] == -1).sum():,} ({(df['cluster'] == -1).mean()*100:.1f}%)

CLUSTER PRINCIPAL:
• Cluster 0: {(df['cluster'] == 0).sum():,} registros
• Representa: {(df['cluster'] == 0).mean()*100:.1f}% dos dados

QUALIDADE:
• Homogeneidade: {'Alta' if (df['cluster'] == 0).mean() > 0.9 else 'Média'}
• Outliers: {'Muitos' if (df['cluster'] == -1).mean() > 0.05 else 'Poucos'}"""
            
            axes[1,1].text(0.05, 0.95, summary_text, transform=axes[1,1].transAxes, 
                          fontsize=10, verticalalignment='top', fontfamily='monospace',
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
            axes[1,1].set_xlim(0, 1)
            axes[1,1].set_ylim(0, 1)
            axes[1,1].axis('off')
            axes[1,1].set_title('Resumo Executivo')
            
            plt.tight_layout()
            
            # Salvar gráfico
            plot_path = self.analysis_dir / "dbscan_analysis.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📊 Gráfico DBSCAN salvo em: {plot_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"❌ Erro ao criar gráficos DBSCAN: {e}")
            import traceback
            print("Stack trace:")
            traceback.print_exc()

    def present_association_rules_results(self):
        """Apresentar resultados das regras de associação"""
        print("\n" + "="*80)
        print("📋 ANÁLISE DE REGRAS DE ASSOCIAÇÃO")
        print("="*80)
        
        algorithms = ['apriori', 'fp_growth', 'eclat']
        algorithm_names = {'apriori': 'APRIORI', 'fp_growth': 'FP-GROWTH', 'eclat': 'ECLAT'}
        
        # Resumo geral
        total_rules = 0
        algorithm_stats = {}
        
        for alg in algorithms:
            if alg in self.results:
                rules_count = len(self.results[alg])
                total_rules += rules_count
                algorithm_stats[alg] = rules_count
        
        print(f"📊 RESUMO GERAL:")
        print(f"   • Total de regras encontradas: {total_rules:,}")
        print(f"   • Algoritmos executados: {len(algorithm_stats)}")
        
        # Análise por algoritmo
        for alg in algorithms:
            if alg not in self.results:
                print(f"\n❌ {algorithm_names[alg]}: Não executado")
                continue
            
            df = self.results[alg]
            print(f"\n🔍 {algorithm_names[alg]}:")
            print(f"   • Regras encontradas: {len(df):,}")
            
            # Estatísticas das métricas
            for metric in ['confidence', 'lift', 'support']:
                if metric in df.columns:
                    mean_val = df[metric].mean()
                    max_val = df[metric].max()
                    print(f"   • {metric.capitalize()} - Média: {mean_val:.3f}, Máximo: {max_val:.3f}")
            
            # Top 5 regras por confiança
            if 'confidence' in df.columns and len(df) > 0:
                print(f"   📈 TOP 5 REGRAS (por confiança):")
                top_rules = df.nlargest(5, 'confidence')
                
                for idx, (_, row) in enumerate(top_rules.iterrows(), 1):
                    antecedent = row.get('antecedents', 'N/A')
                    consequent = row.get('consequents', 'N/A')
                    confidence = row.get('confidence', 0)
                    lift = row.get('lift', 0)
                    
                    print(f"     {idx}. {antecedent} → {consequent}")
                    print(f"        Confiança: {confidence:.3f}, Lift: {lift:.3f}")
        
        # Criar visualizações
        if algorithm_stats:
            self._create_association_plots()

    def _create_association_plots(self):
        """Criar visualizações para regras de associação"""
        try:
            algorithms = ['apriori', 'fp_growth', 'eclat']
            available_algs = [alg for alg in algorithms if alg in self.results]
            
            if not available_algs:
                print("❌ Nenhum resultado de association rules para plotar")
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('📋 ANÁLISE DE REGRAS DE ASSOCIAÇÃO', fontsize=16, fontweight='bold')
            
            # 1. Comparação de número de regras
            rule_counts = []
            alg_names = []
            for alg in available_algs:
                rule_counts.append(len(self.results[alg]))
                alg_names.append(alg.replace('_', '-').upper())
            
            colors = ['skyblue', 'lightgreen', 'lightcoral'][:len(alg_names)]
            bars = axes[0,0].bar(alg_names, rule_counts, color=colors)
            axes[0,0].set_title('Número de Regras por Algoritmo')
            axes[0,0].set_ylabel('Número de Regras')
            
            # Adicionar valores nas barras
            for bar, count in zip(bars, rule_counts):
                height = bar.get_height()
                axes[0,0].text(bar.get_x() + bar.get_width()/2., height,
                              f'{count}', ha='center', va='bottom')
            
            # 2. Distribuição de confiança
            for i, alg in enumerate(available_algs[:3]):
                if 'confidence' in self.results[alg].columns:
                    confidence_data = self.results[alg]['confidence']
                    axes[0,1].hist(confidence_data, alpha=0.6, 
                                  label=alg.replace('_', '-').upper(), 
                                  bins=20, color=colors[i])
            
            axes[0,1].set_title('Distribuição de Confiança')
            axes[0,1].set_xlabel('Confiança')
            axes[0,1].set_ylabel('Frequência')
            if available_algs:
                axes[0,1].legend()
            
            # 3. Lift vs Confidence scatter
            if available_algs and 'confidence' in self.results[available_algs[0]].columns:
                for i, alg in enumerate(available_algs[:3]):
                    df = self.results[alg]
                    if 'lift' in df.columns and 'confidence' in df.columns:
                        sample_size = min(1000, len(df))  # Limitar pontos para performance
                        sample_df = df.sample(n=sample_size) if len(df) > sample_size else df
                        
                        axes[1,0].scatter(sample_df['confidence'], sample_df['lift'], 
                                        alpha=0.6, label=alg.replace('_', '-').upper(),
                                        color=colors[i], s=20)
                
                axes[1,0].set_title('Lift vs Confiança')
                axes[1,0].set_xlabel('Confiança')
                axes[1,0].set_ylabel('Lift')
                axes[1,0].legend()
                axes[1,0].grid(True, alpha=0.3)
            
            # 4. Resumo comparativo
            summary_text = f"""RESUMO ASSOCIATION RULES:

ALGORITMOS EXECUTADOS:
"""
            
            for alg in available_algs:
                df = self.results[alg]
                name = alg.replace('_', '-').upper()
                rules_count = len(df)
                avg_conf = df['confidence'].mean() if 'confidence' in df.columns else 0
                
                summary_text += f"• {name}: {rules_count} regras (conf: {avg_conf:.3f})\n"
            
            summary_text += f"""
MÉTRICAS GERAIS:
• Total de regras: {sum(len(self.results[alg]) for alg in available_algs)}
• Algoritmos ativos: {len(available_algs)}/3
• Status: {'✅ Completo' if len(available_algs) == 3 else '⚠️ Parcial'}"""
            
            axes[1,1].text(0.05, 0.95, summary_text, transform=axes[1,1].transAxes, 
                          fontsize=10, verticalalignment='top', fontfamily='monospace',
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
            axes[1,1].set_xlim(0, 1)
            axes[1,1].set_ylim(0, 1)
            axes[1,1].axis('off')
            axes[1,1].set_title('Resumo Executivo')
            
            plt.tight_layout()
            
            # Salvar gráfico
            plot_path = self.analysis_dir / "association_rules_analysis.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📊 Gráfico Association Rules salvo em: {plot_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"❌ Erro ao criar gráficos Association Rules: {e}")
            import traceback
            traceback.print_exc()

    def generate_final_report(self):
        """Gerar relatório final consolidado"""
        print("\n" + "="*100)
        print("📋 RELATÓRIO FINAL CONSOLIDADO - TODOS OS ALGORITMOS")
        print("="*100)
        
        # Header com timestamp
        print(f"📅 Gerado em: {datetime.now().strftime('%d/%m/%Y às %H:%M:%S')}")
        print(f"📁 Localização dos resultados: {self.analysis_dir}")
        
        # Executar todas as análises
        self.present_dbscan_results()
        self.present_association_rules_results()
        
        # Criar comparação final
        self._create_final_comparison()
        
        print("\n" + "="*100)
        print("✅ RELATÓRIO FINAL CONCLUÍDO!")
        print("📊 Gráficos salvos em: output/analysis/")
        print("📋 Dados analisados com sucesso!")
        print("="*100)

    def _create_final_comparison(self):
        """Criar comparação final entre todos os algoritmos"""
        print(f"\n⚖️ COMPARAÇÃO FINAL DOS ALGORITMOS:")
        
        # Status dos algoritmos
        algorithms_status = {
            'DBSCAN': 'dbscan_results' in self.results or 'dbscan_summary' in self.results,
            'APRIORI': 'apriori' in self.results,
            'FP-GROWTH': 'fp_growth' in self.results,
            'ECLAT': 'eclat' in self.results
        }
        
        executed_count = sum(algorithms_status.values())
        total_algorithms = len(algorithms_status)
        
        print(f"📊 RESUMO EXECUTIVO:")
        print(f"   • Algoritmos executados: {executed_count}/{total_algorithms}")
        print(f"   • Taxa de sucesso: {(executed_count/total_algorithms)*100:.1f}%")
        
        print(f"\n🎯 STATUS DETALHADO:")
        for alg, status in algorithms_status.items():
            icon = "✅" if status else "❌"
            print(f"   {icon} {alg}: {'Executado com sucesso' if status else 'Não executado'}")
        
        # Salvar comparação final
        try:
            comparison_data = []
            
            # DBSCAN
            if algorithms_status['DBSCAN']:
                if 'dbscan_results' in self.results:
                    df = self.results['dbscan_results']
                    n_clusters = len(df['cluster'].unique()) - (1 if -1 in df['cluster'].values else 0)
                    noise_rate = (df['cluster'] == -1).mean() * 100
                elif 'dbscan_summary' in self.results:
                    summary_df = self.results['dbscan_summary']
                    n_clusters = len(summary_df[summary_df['cluster_id'] != -1])
                    noise_rate = summary_df[summary_df['cluster_id'] == -1]['percentage'].sum()
                
                comparison_data.append({
                    'Algorithm': 'DBSCAN',
                    'Type': 'Clustering',
                    'Status': 'Success',
                    'Results': f"{n_clusters} clusters",
                    'Quality': f"{noise_rate:.1f}% noise"
                })
            
            # Association Rules
            for alg in ['apriori', 'fp_growth', 'eclat']:
                if alg in self.results:
                    df = self.results[alg]
                    avg_conf = df['confidence'].mean() if 'confidence' in df.columns else 0
                    
                    comparison_data.append({
                        'Algorithm': alg.replace('_', '-').upper(),
                        'Type': 'Association Rules',
                        'Status': 'Success',
                        'Results': f"{len(df)} rules",
                        'Quality': f"{avg_conf:.3f} avg confidence"
                    })
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                comparison_path = self.analysis_dir / "final_algorithms_comparison.csv"
                comparison_df.to_csv(comparison_path, index=False)
                print(f"💾 Comparação final salva em: {comparison_path}")
        
        except Exception as e:
            print(f"❌ Erro ao salvar comparação: {e}")

def main():
    """Função principal para apresentar todos os resultados"""
    print("🚀 Iniciando apresentação completa de resultados...")
    
    # Verificar se diretório de análise existe
    analysis_dir = Path("output/analysis")
    if not analysis_dir.exists():
        print(f"❌ Diretório {analysis_dir} não encontrado!")
        print("💡 Execute primeiro o pipeline principal: python main.py")
        return False
    
    # Criar apresentador
    presenter = ResultsPresenter(analysis_dir)
    
    # Gerar relatório completo
    presenter.generate_final_report()
    
    print("\n🎉 Apresentação completa finalizada!")
    print("📊 Todos os gráficos e comparações foram gerados!")
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("✅ Apresentação concluída com sucesso!")
    else:
        print("❌ Falha na apresentação dos resultados")