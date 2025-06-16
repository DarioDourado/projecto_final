"""
🎓 Dashboard Ultra-Simplificado - Análise Salarial
Usando apenas Streamlit e Pandas (sem matplotlib/plotly)
Dados de output/analysis gerados pelo main.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
from datetime import datetime

# Configuração
warnings.filterwarnings('ignore')
st.set_page_config(
    page_title="🎓 Dashboard Acadêmico - Análise Salarial",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CARREGAMENTO DOS DADOS DO MAIN.PY
# =============================================================================

@st.cache_data
def load_analysis_data():
    """Carregar dados de output/analysis"""
    data = {}
    analysis_dir = Path("output/analysis")
    
    if not analysis_dir.exists():
        st.error("❌ Diretório output/analysis não encontrado!")
        st.info("💡 Execute primeiro: python main.py")
        return {}
    
    # Arquivos esperados do main.py
    expected_files = {
        'dbscan_results.csv': 'DBSCAN Clustering',
        'apriori_rules.csv': 'Regras APRIORI',
        'fp_growth_rules.csv': 'Regras FP-Growth', 
        'eclat_rules.csv': 'Regras ECLAT',
        'advanced_metrics_v2.csv': 'Métricas Avançadas',
        'clustering_results_v2.csv': 'Resultados Clustering',
        'pipeline_results.json': 'Resultados Pipeline',
        'metrics_summary.json': 'Resumo Métricas'
    }
    
    files_loaded = 0
    files_total = len(expected_files)
    
    for filename, description in expected_files.items():
        file_path = analysis_dir / filename
        
        if file_path.exists():
            try:
                if filename.endswith('.csv'):
                    data[filename.replace('.csv', '')] = pd.read_csv(file_path)
                elif filename.endswith('.json'):
                    with open(file_path, 'r') as f:
                        data[filename.replace('.json', '')] = json.load(f)
                
                files_loaded += 1
                st.sidebar.success(f"✅ {description}")
                
            except Exception as e:
                st.sidebar.error(f"❌ Erro em {filename}: {str(e)}")
        else:
            st.sidebar.warning(f"⚠️ {filename} não encontrado")
    
    # Carregar dados originais
    original_data_paths = [
        "bkp/4-Carateristicas_salario.csv",
        "data/adult.csv",
        "data/processed/adult_processed.csv"
    ]
    
    for path in original_data_paths:
        if Path(path).exists():
            try:
                data['original_dataset'] = pd.read_csv(path)
                st.sidebar.success(f"✅ Dados originais: {Path(path).name}")
                break
            except Exception as e:
                st.sidebar.warning(f"⚠️ Erro em {Path(path).name}: {str(e)}")
    
    # Status geral
    st.sidebar.markdown("---")
    st.sidebar.metric("📊 Arquivos Carregados", f"{files_loaded}/{files_total}")
    
    return data

# =============================================================================
# COMPONENTES VISUAIS SIMPLES
# =============================================================================

def create_metric_grid(metrics_dict, cols=4):
    """Criar grid de métricas"""
    metric_cols = st.columns(cols)
    
    for i, (key, value) in enumerate(metrics_dict.items()):
        col_idx = i % cols
        with metric_cols[col_idx]:
            if isinstance(value, float):
                st.metric(key, f"{value:.3f}")
            elif isinstance(value, int):
                st.metric(key, f"{value:,}")
            else:
                st.metric(key, str(value))

def show_dataframe_info(df, title=""):
    """Mostrar informações básicas do DataFrame"""
    if title:
        st.subheader(title)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Linhas", f"{len(df):,}")
    with col2:
        st.metric("📋 Colunas", len(df.columns))
    with col3:
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("❌ Missing %", f"{missing_pct:.1f}%")
    with col4:
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        st.metric("💾 Memória", f"{memory_mb:.1f} MB")

def show_simple_bar_chart(data, title="", max_items=10):
    """Criar gráfico de barras simples usando Streamlit nativo"""
    if isinstance(data, pd.Series):
        chart_data = data.head(max_items)
    else:
        chart_data = data
    
    st.subheader(title)
    st.bar_chart(chart_data)

def show_value_counts_table(df, column, title="", max_items=10):
    """Mostrar tabela de contagem de valores"""
    if title:
        st.subheader(title)
    
    if column in df.columns:
        value_counts = df[column].value_counts().head(max_items)
        
        # Criar DataFrame para melhor visualização
        result_df = pd.DataFrame({
            'Valor': value_counts.index,
            'Contagem': value_counts.values,
            'Percentual': (value_counts.values / len(df) * 100).round(2)
        })
        
        st.dataframe(result_df, use_container_width=True)
        
        # Gráfico de barras simples
        st.bar_chart(value_counts)
    else:
        st.warning(f"❌ Coluna '{column}' não encontrada")

# =============================================================================
# PÁGINAS DO DASHBOARD
# =============================================================================

def show_overview_page(data):
    """📊 Página de Visão Geral"""
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 10px; margin-bottom: 2rem; color: white;">
        <h1 style="margin: 0;">📊 Dashboard Acadêmico</h1>
        <p style="margin: 0.5rem 0 0 0;">Análise Salarial - Resultados do Pipeline Científico</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Status do pipeline
    st.subheader("🚀 Status do Pipeline Executado")
    
    pipeline_status = {
        'DBSCAN Clustering': '✅' if 'dbscan_results' in data else '❌',
        'Regras APRIORI': '✅' if 'apriori_rules' in data else '❌',
        'Regras FP-Growth': '✅' if 'fp_growth_rules' in data else '❌',
        'Regras ECLAT': '✅' if 'eclat_rules' in data else '❌',
        'Métricas Avançadas': '✅' if 'advanced_metrics_v2' in data else '❌',
        'Dados Originais': '✅' if 'original_dataset' in data else '❌'
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        for algo, status in list(pipeline_status.items())[:3]:
            st.markdown(f"**{algo}:** {status}")
    
    with col2:
        for algo, status in list(pipeline_status.items())[3:]:
            st.markdown(f"**{algo}:** {status}")
    
    # Resumo quantitativo
    st.subheader("📊 Resumo Quantitativo")
    
    summary_metrics = {}
    
    if 'dbscan_results' in data:
        dbscan_df = data['dbscan_results']
        summary_metrics['Pontos DBSCAN'] = len(dbscan_df)
        if 'cluster' in dbscan_df.columns:
            summary_metrics['Clusters DBSCAN'] = dbscan_df['cluster'].nunique()
    
    if 'apriori_rules' in data:
        summary_metrics['Regras APRIORI'] = len(data['apriori_rules'])
    
    if 'fp_growth_rules' in data:
        summary_metrics['Regras FP-Growth'] = len(data['fp_growth_rules'])
    
    if 'eclat_rules' in data:
        summary_metrics['Regras ECLAT'] = len(data['eclat_rules'])
    
    if 'original_dataset' in data:
        summary_metrics['Registros Dataset'] = len(data['original_dataset'])
    
    if summary_metrics:
        create_metric_grid(summary_metrics, cols=3)
    
    # Pipeline Results (se disponível)
    if 'pipeline_results' in data:
        st.subheader("🎯 Resultados do Pipeline")
        
        results = data['pipeline_results']
        
        if isinstance(results, dict):
            st.json(results)
        else:
            st.write(results)

def show_clustering_page(data):
    """🎯 Página de Clustering"""
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 10px; margin-bottom: 2rem; color: white;">
        <h1 style="margin: 0;">🎯 Análise de Clustering</h1>
        <p style="margin: 0.5rem 0 0 0;">Resultados do DBSCAN e segmentação de dados</p>
    </div>
    """, unsafe_allow_html=True)
    
    # DBSCAN Results
    if 'dbscan_results' in data:
        dbscan_df = data['dbscan_results']
        
        show_dataframe_info(dbscan_df, "📊 Informações do DBSCAN")
        
        # Análise dos clusters
        if 'cluster' in dbscan_df.columns:
            st.subheader("🎯 Distribuição dos Clusters")
            
            cluster_counts = dbscan_df['cluster'].value_counts().sort_index()
            
            # Mostrar como tabela
            cluster_df = pd.DataFrame({
                'Cluster': cluster_counts.index,
                'Pontos': cluster_counts.values,
                'Percentual': (cluster_counts.values / len(dbscan_df) * 100).round(2)
            })
            
            st.dataframe(cluster_df, use_container_width=True)
            
            # Gráfico de barras
            st.subheader("📈 Visualização dos Clusters")
            st.bar_chart(cluster_counts)
            
            # Análise de clusters específicos
            st.subheader("🔍 Análise Detalhada por Cluster")
            
            selected_cluster = st.selectbox(
                "Selecione um cluster:",
                options=sorted(cluster_counts.index)
            )
            
            cluster_data = dbscan_df[dbscan_df['cluster'] == selected_cluster]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("👥 Pontos no Cluster", len(cluster_data))
                st.metric("📊 % do Total", f"{(len(cluster_data)/len(dbscan_df)*100):.1f}%")
            
            with col2:
                if selected_cluster == -1:
                    st.warning("🔴 Este é o cluster de ruído (outliers)")
                else:
                    st.success(f"🎯 Cluster {selected_cluster} - Grupo válido")
            
            # Preview dos dados do cluster
            st.subheader(f"👀 Preview - Cluster {selected_cluster}")
            st.dataframe(cluster_data.head(10), use_container_width=True)
        
        else:
            st.warning("❌ Coluna 'cluster' não encontrada nos resultados DBSCAN")
    
    # Clustering Results v2 (se disponível)
    if 'clustering_results_v2' in data:
        st.subheader("🎯 Resultados Adicionais de Clustering")
        
        clustering_df = data['clustering_results_v2']
        show_dataframe_info(clustering_df, "")
        
        st.dataframe(clustering_df.head(10), use_container_width=True)
    
    if 'dbscan_results' not in data and 'clustering_results_v2' not in data:
        st.warning("❌ Nenhum resultado de clustering encontrado")
        st.info("💡 Execute: python main.py para gerar os resultados")

def show_association_rules_page(data):
    """📋 Página de Regras de Associação"""
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 10px; margin-bottom: 2rem; color: white;">
        <h1 style="margin: 0;">📋 Regras de Associação</h1>
        <p style="margin: 0.5rem 0 0 0;">APRIORI, FP-Growth e ECLAT</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Verificar quais algoritmos estão disponíveis
    algorithms = {
        'apriori_rules': ('🎯 APRIORI', 'Algoritmo clássico de mineração'),
        'fp_growth_rules': ('🌳 FP-Growth', 'Algoritmo baseado em árvore'),
        'eclat_rules': ('⚡ ECLAT', 'Algoritmo de busca vertical')
    }
    
    available_algorithms = [key for key in algorithms.keys() if key in data]
    
    if not available_algorithms:
        st.warning("❌ Nenhuma regra de associação encontrada")
        st.info("💡 Execute: python main.py para gerar as regras")
        return
    
    # Tabs para cada algoritmo
    tab_names = [algorithms[alg][0] for alg in available_algorithms]
    tabs = st.tabs(tab_names + ["📊 Comparação"])
    
    # Mostrar cada algoritmo
    for i, algorithm_key in enumerate(available_algorithms):
        with tabs[i]:
            algorithm_name, algorithm_desc = algorithms[algorithm_key]
            
            st.subheader(f"{algorithm_name}")
            st.markdown(f"*{algorithm_desc}*")
            
            rules_df = data[algorithm_key]
            
            # Informações básicas
            show_dataframe_info(rules_df, "")
            
            # Métricas das regras
            metrics = {}
            
            if 'confidence' in rules_df.columns:
                metrics['Confidence Média'] = rules_df['confidence'].mean()
                metrics['Confidence Máxima'] = rules_df['confidence'].max()
            
            if 'support' in rules_df.columns:
                metrics['Support Médio'] = rules_df['support'].mean()
                metrics['Support Máximo'] = rules_df['support'].max()
            
            if 'lift' in rules_df.columns:
                metrics['Lift Médio'] = rules_df['lift'].mean()
                metrics['Lift Máximo'] = rules_df['lift'].max()
            
            if metrics:
                create_metric_grid(metrics, cols=3)
            
            # Top 10 regras
            st.subheader("🏆 Top 10 Regras")
            
            if 'confidence' in rules_df.columns:
                top_rules = rules_df.nlargest(10, 'confidence')
            else:
                top_rules = rules_df.head(10)
            
            st.dataframe(top_rules, use_container_width=True)
            
            # Distribuição de métricas
            if 'confidence' in rules_df.columns:
                st.subheader("📊 Distribuição de Confidence")
                confidence_ranges = pd.cut(rules_df['confidence'], bins=5).value_counts()
                st.bar_chart(confidence_ranges)
    
    # Tab de comparação
    if len(available_algorithms) > 1:
        with tabs[-1]:
            st.subheader("📊 Comparação dos Algoritmos")
            
            comparison_data = []
            
            for algorithm_key in available_algorithms:
                algorithm_name = algorithms[algorithm_key][0]
                rules_df = data[algorithm_key]
                
                row = {
                    'Algoritmo': algorithm_name,
                    'Total Regras': len(rules_df)
                }
                
                if 'confidence' in rules_df.columns:
                    row['Confidence Média'] = f"{rules_df['confidence'].mean():.3f}"
                    row['Confidence Máxima'] = f"{rules_df['confidence'].max():.3f}"
                
                if 'support' in rules_df.columns:
                    row['Support Médio'] = f"{rules_df['support'].mean():.3f}"
                
                if 'lift' in rules_df.columns:
                    row['Lift Médio'] = f"{rules_df['lift'].mean():.3f}"
                
                comparison_data.append(row)
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Gráfico de comparação simples
            st.subheader("📈 Total de Regras por Algoritmo")
            
            rules_count = pd.Series({
                algorithms[alg][0]: len(data[alg]) 
                for alg in available_algorithms
            })
            
            st.bar_chart(rules_count)

def show_metrics_page(data):
    """📈 Página de Métricas"""
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 10px; margin-bottom: 2rem; color: white;">
        <h1 style="margin: 0;">📈 Métricas e Performance</h1>
        <p style="margin: 0.5rem 0 0 0;">Análise detalhada dos resultados</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Advanced Metrics
    if 'advanced_metrics_v2' in data:
        st.subheader("📊 Métricas Avançadas")
        
        metrics_df = data['advanced_metrics_v2']
        show_dataframe_info(metrics_df, "")
        
        st.dataframe(metrics_df, use_container_width=True)
        
        # Análise das métricas numéricas
        numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            st.subheader("📈 Estatísticas das Métricas Numéricas")
            st.dataframe(metrics_df[numeric_cols].describe(), use_container_width=True)
    
    # Metrics Summary
    if 'metrics_summary' in data:
        st.subheader("📋 Resumo das Métricas")
        
        summary = data['metrics_summary']
        
        if isinstance(summary, dict):
            # Mostrar como métricas se forem valores simples
            simple_metrics = {k: v for k, v in summary.items() if isinstance(v, (int, float, str))}
            
            if simple_metrics:
                create_metric_grid(simple_metrics, cols=4)
            
            # Mostrar JSON completo
            st.json(summary)
        else:
            st.write(summary)
    
    # Pipeline Results
    if 'pipeline_results' in data:
        st.subheader("🚀 Resultados do Pipeline")
        
        results = data['pipeline_results']
        
        if isinstance(results, dict):
            st.json(results)
        else:
            st.write(results)
    
    if not any(key in data for key in ['advanced_metrics_v2', 'metrics_summary', 'pipeline_results']):
        st.warning("❌ Nenhuma métrica encontrada")
        st.info("💡 Execute: python main.py para gerar as métricas")

def show_original_data_page(data):
    """📊 Página dos Dados Originais"""
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 10px; margin-bottom: 2rem; color: white;">
        <h1 style="margin: 0;">📊 Dados Originais</h1>
        <p style="margin: 0.5rem 0 0 0;">Análise exploratória do dataset base</p>
    </div>
    """, unsafe_allow_html=True)
    
    if 'original_dataset' in data:
        df = data['original_dataset']
        
        # Informações gerais
        show_dataframe_info(df, "📊 Informações Gerais")
        
        # Preview dos dados
        st.subheader("👀 Preview dos Dados")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Análise de colunas
        st.subheader("📋 Análise das Colunas")
        
        col_info = []
        for col in df.columns:
            col_info.append({
                'Coluna': col,
                'Tipo': str(df[col].dtype),
                'Não-Nulos': df[col].count(),
                'Nulos': df[col].isnull().sum(),
                'Únicos': df[col].nunique()
            })
        
        col_df = pd.DataFrame(col_info)
        st.dataframe(col_df, use_container_width=True)
        
        # Análise de variáveis categóricas
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if categorical_cols:
            st.subheader("🏷️ Análise de Variáveis Categóricas")
            
            selected_cat = st.selectbox("Selecione uma variável:", categorical_cols)
            
            show_value_counts_table(df, selected_cat, f"Distribuição de {selected_cat}")
        
        # Análise de variáveis numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            st.subheader("📊 Estatísticas das Variáveis Numéricas")
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
            
            # Análise individual
            selected_num = st.selectbox("Selecione uma variável numérica:", numeric_cols)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Média", f"{df[selected_num].mean():.2f}")
                st.metric("Mediana", f"{df[selected_num].median():.2f}")
            
            with col2:
                st.metric("Desvio Padrão", f"{df[selected_num].std():.2f}")
                st.metric("Amplitude", f"{df[selected_num].max() - df[selected_num].min():.2f}")
            
            # Histograma usando line_chart
            st.subheader(f"📈 Distribuição de {selected_num}")
            hist_data = df[selected_num].value_counts().sort_index()
            st.line_chart(hist_data)
    
    else:
        st.warning("❌ Dados originais não encontrados")
        st.info("💡 Certifique-se de que existe um arquivo CSV em bkp/ ou data/")

def show_reports_page(data):
    """📁 Página de Relatórios"""
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 10px; margin-bottom: 2rem; color: white;">
        <h1 style="margin: 0;">📁 Relatórios e Exportações</h1>
        <p style="margin: 0.5rem 0 0 0;">Downloads e documentação</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Arquivos disponíveis
    st.subheader("📊 Arquivos Disponíveis")
    
    analysis_dir = Path("output/analysis")
    if analysis_dir.exists():
        files_info = []
        
        for file in analysis_dir.iterdir():
            if file.is_file():
                files_info.append({
                    'Arquivo': file.name,
                    'Tipo': file.suffix.upper(),
                    'Tamanho (KB)': f"{file.stat().st_size / 1024:.1f}",
                    'Modificado': datetime.fromtimestamp(file.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
                })
        
        if files_info:
            files_df = pd.DataFrame(files_info)
            st.dataframe(files_df, use_container_width=True)
        else:
            st.warning("❌ Nenhum arquivo encontrado em output/analysis")
    else:
        st.warning("❌ Diretório output/analysis não existe")
    
    # Resumo dos algoritmos
    resumo_file = Path("output/resumo_algoritmos.txt")
    if resumo_file.exists():
        st.subheader("📋 Resumo dos Algoritmos")
        
        try:
            with open(resumo_file, 'r', encoding='utf-8') as f:
                resumo_content = f.read()
            
            st.text_area(
                "Conteúdo:",
                resumo_content,
                height=300
            )
            
            st.download_button(
                label="📥 Download Resumo",
                data=resumo_content,
                file_name="resumo_algoritmos.txt",
                mime="text/plain"
            )
        except Exception as e:
            st.error(f"❌ Erro ao ler resumo: {e}")
    
    # Exportar dados
    st.subheader("💾 Exportar Dados")
    
    if data:
        # Listar datasets disponíveis
        dataset_options = {k: v for k, v in data.items() if isinstance(v, pd.DataFrame)}
        
        if dataset_options:
            selected_dataset = st.selectbox(
                "Selecione dataset para exportar:",
                list(dataset_options.keys())
            )
            
            if st.button("📥 Gerar CSV"):
                df_to_export = dataset_options[selected_dataset]
                csv = df_to_export.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"{selected_dataset}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
                
                st.success("✅ CSV gerado com sucesso!")
        else:
            st.warning("❌ Nenhum dataset disponível para exportação")
    
    # Relatório consolidado
    st.subheader("📊 Relatório Consolidado")
    
    if st.button("📋 Gerar Relatório Completo"):
        report_content = f"""
# Relatório de Análise Salarial
Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Datasets Carregados
"""
        
        for key, value in data.items():
            if isinstance(value, pd.DataFrame):
                report_content += f"- {key}: {len(value)} registros, {len(value.columns)} colunas\n"
            else:
                report_content += f"- {key}: {type(value).__name__}\n"
        
        report_content += f"""

## Algoritmos Executados
- DBSCAN Clustering: {'✅' if 'dbscan_results' in data else '❌'}
- Regras APRIORI: {'✅' if 'apriori_rules' in data else '❌'}
- Regras FP-Growth: {'✅' if 'fp_growth_rules' in data else '❌'}
- Regras ECLAT: {'✅' if 'eclat_rules' in data else '❌'}
- Métricas Avançadas: {'✅' if 'advanced_metrics_v2' in data else '❌'}

## Resumo Quantitativo
"""
        
        if 'dbscan_results' in data:
            dbscan_df = data['dbscan_results']
            report_content += f"- Pontos DBSCAN: {len(dbscan_df)}\n"
            if 'cluster' in dbscan_df.columns:
                report_content += f"- Clusters encontrados: {dbscan_df['cluster'].nunique()}\n"
        
        for alg in ['apriori_rules', 'fp_growth_rules', 'eclat_rules']:
            if alg in data:
                report_content += f"- Regras {alg.replace('_rules', '').upper()}: {len(data[alg])}\n"
        
        st.download_button(
            label="📥 Download Relatório",
            data=report_content,
            file_name=f"relatorio_analise_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown"
        )
        
        st.success("✅ Relatório gerado!")

# =============================================================================
# NAVEGAÇÃO PRINCIPAL
# =============================================================================

def main():
    """Função principal"""
    
    # CSS customizado
    st.markdown("""
    <style>
    .stApp > header {
        background-color: transparent;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .nav-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.75rem;
        border-radius: 5px;
        margin: 0.25rem 0;
    }
    
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 0.75rem;
        border-radius: 5px;
        margin: 0.25rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Carregar dados
    data = load_analysis_data()
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="nav-header">
            <h2>🎓 Dashboard</h2>
            <p>Análise Salarial</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🧭 Navegação")
        
        # Inicializar estado
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 'overview'
        
        # Páginas
        pages = {
            'overview': '📊 Visão Geral',
            'clustering': '🎯 Clustering',
            'association': '📋 Regras Associação',
            'metrics': '📈 Métricas',
            'original_data': '📊 Dados Originais',
            'reports': '📁 Relatórios'
        }
        
        for page_key, page_name in pages.items():
            if st.button(page_name, key=f"nav_{page_key}", use_container_width=True):
                st.session_state.current_page = page_key
                st.rerun()
        
        st.markdown("---")
        st.markdown(f"### ℹ️ Sistema")
        st.markdown(f"**📅 Atualizado:** {datetime.now().strftime('%H:%M:%S')}")
        st.markdown(f"**📊 Datasets:** {len([k for k, v in data.items() if isinstance(v, pd.DataFrame)])}")
    
    # Conteúdo principal
    if not data:
        st.error("❌ Nenhum dado carregado!")
        st.info("💡 Execute: python main.py")
        return
    
    current_page = st.session_state.current_page
    
    if current_page == 'overview':
        show_overview_page(data)
    elif current_page == 'clustering':
        show_clustering_page(data)
    elif current_page == 'association':
        show_association_rules_page(data)
    elif current_page == 'metrics':
        show_metrics_page(data)
    elif current_page == 'original_data':
        show_original_data_page(data)
    elif current_page == 'reports':
        show_reports_page(data)

if __name__ == "__main__":
    main()
