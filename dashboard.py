"""
🎓 Dashboard Científico Acadêmico - Análise Salarial
Sistema Totalmente Integrado com main.py e Relatório Acadêmico
Versão: 3.0 - Completamente Refatorado e Corrigido
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from pathlib import Path
import json
import logging
from datetime import datetime
import sys
import subprocess
import time
import warnings
warnings.filterwarnings('ignore')

# Configurações da página
st.set_page_config(
    page_title="🎓 Dashboard Científico - Análise Salarial Acadêmica",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# SISTEMA DE DADOS CORRIGIDO
# =============================================================================

class ScientificDataLoader:
    """Carregador de dados científico com paths corretos"""
    
    def __init__(self):
        # ✅ PATHS CORRETOS baseados no workspace real
        self.data_paths = {
            'main_dataset': [
                'bkp/4-Carateristicas_salario.csv',  # ✅ Confirmado no workspace
                'data/raw/4-Carateristicas_salario.csv',
                '4-Carateristicas_salario.csv'
            ],
            'outputs': {
                'dbscan_results': 'output/analysis/dbscan_results.csv',
                'apriori_rules': 'output/analysis/apriori_rules.csv',
                'fp_growth_rules': 'output/analysis/fp_growth_rules.csv',
                'eclat_rules': 'output/analysis/eclat_rules.csv',
                'academic_report': 'output/analysis/relatorio_academico_v2.md',
                'pipeline_summary': 'output/resumo_algoritmos.txt'
            },
            'models': {
                'random_forest': 'data/processed/random_forest_model_v2.joblib',
                'logistic_regression': 'data/processed/logistic_regression_model_v2.joblib',
                'preprocessor': 'data/processed/preprocessor_v2.joblib'
            }
        }
        
        self.logger = logging.getLogger(__name__)
    
    def load_main_dataset(self):
        """Carregar dataset principal com paths corretos"""
        for path in self.data_paths['main_dataset']:
            if Path(path).exists():
                try:
                    df = pd.read_csv(path)
                    self.logger.info(f"✅ Dataset carregado: {path} ({len(df):,} registros)")
                    return df, f"✅ Dataset: {len(df):,} registros de {path}"
                except Exception as e:
                    self.logger.warning(f"⚠️ Erro ao carregar {path}: {e}")
                    continue
        
        # Fallback para dados sintéticos
        return self._create_sample_data(), "⚠️ Usando dados de demonstração (arquivo não encontrado)"
    
    def check_pipeline_execution(self):
        """Verificar se o pipeline main.py foi executado"""
        results = {
            'executed': False,
            'algorithms': {},
            'models': {},
            'files_found': [],
            'summary': None
        }
        
        # Verificar arquivos de output
        for algo, path in self.data_paths['outputs'].items():
            if Path(path).exists():
                results['algorithms'][algo] = True
                results['files_found'].append(path)
                
                # Carregar dados se possível
                if path.endswith('.csv'):
                    try:
                        df = pd.read_csv(path)
                        results['algorithms'][f"{algo}_records"] = len(df)
                    except:
                        pass
            else:
                results['algorithms'][algo] = False
        
        # Verificar modelos
        for model, path in self.data_paths['models'].items():
            results['models'][model] = Path(path).exists()
        
        # Verificar resumo do pipeline
        summary_path = self.data_paths['outputs']['pipeline_summary']
        if Path(summary_path).exists():
            try:
                with open(summary_path, 'r', encoding='utf-8') as f:
                    results['summary'] = f.read()
                results['executed'] = True
            except:
                pass
        
        # Status geral
        algorithms_ok = sum(1 for v in results['algorithms'].values() if isinstance(v, bool) and v)
        models_ok = sum(results['models'].values())
        
        results['execution_score'] = (algorithms_ok + models_ok) / (len(self.data_paths['outputs']) + len(self.data_paths['models']))
        results['executed'] = results['execution_score'] > 0.5
        
        return results
    
    def load_algorithm_results(self, algorithm):
        """Carregar resultados de algoritmo específico"""
        if algorithm in self.data_paths['outputs']:
            path = self.data_paths['outputs'][algorithm]
            if Path(path).exists():
                try:
                    if path.endswith('.csv'):
                        return pd.read_csv(path)
                    elif path.endswith('.md'):
                        with open(path, 'r', encoding='utf-8') as f:
                            return f.read()
                    elif path.endswith('.txt'):
                        with open(path, 'r', encoding='utf-8') as f:
                            return f.read()
                except Exception as e:
                    self.logger.error(f"Erro ao carregar {algorithm}: {e}")
        return None
    
    def _create_sample_data(self):
        """Criar dados de amostra baseados no dataset original"""
        np.random.seed(42)
        n_samples = 1000
        
        # Dados baseados nas características do dataset real
        ages = np.random.normal(39, 13, n_samples).astype(int)
        ages = np.clip(ages, 17, 90)
        
        education_nums = np.random.randint(1, 17, n_samples)
        hours_per_week = np.random.normal(40, 12, n_samples).astype(int)
        hours_per_week = np.clip(hours_per_week, 1, 99)
        
        return pd.DataFrame({
            'age': ages,
            'workclass': np.random.choice(['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov'], n_samples),
            'fnlwgt': np.random.randint(12285, 1484705, n_samples),
            'education': np.random.choice(['Bachelors', 'HS-grad', 'Some-college', 'Masters'], n_samples),
            'education-num': education_nums,
            'marital-status': np.random.choice(['Married-civ-spouse', 'Never-married', 'Divorced'], n_samples),
            'occupation': np.random.choice(['Prof-specialty', 'Craft-repair', 'Exec-managerial'], n_samples),
            'relationship': np.random.choice(['Husband', 'Not-in-family', 'Wife'], n_samples),
            'race': np.random.choice(['White', 'Black', 'Asian-Pac-Islander'], n_samples),
            'sex': np.random.choice(['Male', 'Female'], n_samples),
            'capital-gain': np.random.exponential(1077, n_samples).astype(int),
            'capital-loss': np.random.exponential(87, n_samples).astype(int),
            'hours-per-week': hours_per_week,
            'native-country': np.random.choice(['United-States', 'Mexico', 'Philippines'], n_samples),
            'salary': np.random.choice(['<=50K', '>50K'], n_samples, p=[0.76, 0.24])
        })

# =============================================================================
# SISTEMA DE AUTENTICAÇÃO ACADÊMICA
# =============================================================================

class AcademicAuthSystem:
    """Sistema de autenticação para ambiente acadêmico"""
    
    def __init__(self):
        self.users = {
            "professor": {"password": "academic2025", "role": "admin", "name": "Professor", "access": "full"},
            "aluno": {"password": "student2025", "role": "student", "name": "Estudante", "access": "read"},
            "revisor": {"password": "reviewer2025", "role": "reviewer", "name": "Revisor Científico", "access": "analysis"},
            "demo": {"password": "demo", "role": "guest", "name": "Demonstração", "access": "limited"}
        }
    
    def authenticate(self, username, password):
        if username in self.users and self.users[username]["password"] == password:
            return True, self.users[username]
        return False, None
    
    def show_login_page(self):
        """Interface de login acadêmica"""
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 15px; color: white; margin-bottom: 2rem;">
            <h1>🎓 Sistema Acadêmico de Análise Salarial</h1>
            <h3>Dashboard Científico Integrado</h3>
            <p><i>Implementação Completa: DBSCAN • APRIORI • FP-GROWTH • ECLAT</i></p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("### 🔐 Acesso Acadêmico")
            
            username = st.selectbox(
                "👤 Perfil de Usuário:",
                ["", "professor", "aluno", "revisor", "demo"],
                help="Selecione seu perfil acadêmico"
            )
            
            password = st.text_input("🔑 Senha:", type="password")
            
            if st.button("🚀 Entrar no Sistema", use_container_width=True, type="primary"):
                if username and password:
                    success, user_data = self.authenticate(username, password)
                    if success:
                        st.session_state.authenticated = True
                        st.session_state.user = user_data
                        st.session_state.username = username
                        st.success(f"✅ Bem-vindo(a), {user_data['name']}!")
                        st.balloons()
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Credenciais inválidas!")
                else:
                    st.warning("⚠️ Preencha todos os campos!")
            
            # Credenciais de demonstração
            with st.expander("ℹ️ Credenciais de Demonstração"):
                st.markdown("""
                | Usuário | Senha | Acesso |
                |---------|--------|---------|
                | **professor** | academic2024 | Completo |
                | **aluno** | student2024 | Leitura |
                | **revisor** | reviewer2024 | Análise |
                | **demo** | demo | Limitado |
                """)

# =============================================================================
# PÁGINAS DO DASHBOARD CIENTÍFICO
# =============================================================================

class ScientificDashboardPages:
    """Páginas do dashboard científico"""
    
    def __init__(self, data_loader):
        self.data_loader = data_loader
    
    def show_overview_page(self, df, pipeline_status):
        """Página de visão geral científica"""
        st.title("📊 Visão Geral - Análise Científica Salarial")
        st.markdown("---")
        
        # Header com informações do projeto
        st.markdown("""
        <div style="background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); 
                    padding: 2rem; border-radius: 15px; color: white; margin-bottom: 2rem;">
            <h2>🎓 Projeto Acadêmico - Análise Salarial</h2>
            <p><strong>Algoritmos Implementados:</strong> DBSCAN, APRIORI, FP-GROWTH, ECLAT, Random Forest, Logistic Regression</p>
            <p><strong>Dataset:</strong> Adult Income Dataset (32.561 registros originais)</p>
            <p><strong>Objetivo:</strong> Análise preditiva e descritiva de fatores salariais</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "📊 Registros Dataset",
                f"{len(df):,}",
                help="Total de registros carregados"
            )
        
        with col2:
            algorithms_completed = sum(1 for k, v in pipeline_status['algorithms'].items() 
                                     if isinstance(v, bool) and v)
            total_algorithms = len([k for k, v in pipeline_status['algorithms'].items() 
                                  if isinstance(v, bool)])
            st.metric(
                "🤖 Algoritmos",
                f"{algorithms_completed}/{total_algorithms}",
                help="Algoritmos executados com sucesso"
            )
        
        with col3:
            models_ready = sum(pipeline_status['models'].values())
            st.metric(
                "🎯 Modelos ML",
                f"{models_ready}/3",
                help="Modelos treinados e disponíveis"
            )
        
        with col4:
            if 'salary' in df.columns:
                high_salary_rate = (df['salary'] == '>50K').mean()
                st.metric(
                    "💰 Taxa >50K",
                    f"{high_salary_rate:.1%}",
                    help="Percentual de salários altos"
                )
        
        # Status detalhado do pipeline
        st.subheader("🔬 Status dos Algoritmos Científicos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Clustering & Análise Não-Supervisionada")
            
            dbscan_status = "✅ Completo" if pipeline_status['algorithms'].get('dbscan_results', False) else "❌ Pendente"
            dbscan_records = pipeline_status['algorithms'].get('dbscan_results_records', 0)
            
            st.markdown(f"""
            **DBSCAN Clustering:**
            - Status: {dbscan_status}
            - Registros: {dbscan_records:,} pontos analisados
            - Objetivo: Identificar grupos baseados em densidade
            """)
            
            st.markdown("#### 🤖 Modelos Supervisionados")
            models = pipeline_status['models']
            
            rf_status = "✅ Treinado" if models.get('random_forest', False) else "❌ Ausente"
            lr_status = "✅ Treinado" if models.get('logistic_regression', False) else "❌ Ausente"
            
            st.markdown(f"""
            **Random Forest:** {rf_status}
            **Logistic Regression:** {lr_status}
            **Preprocessor:** {'✅ Disponível' if models.get('preprocessor', False) else '❌ Ausente'}
            """)
        
        with col2:
            st.markdown("#### 📋 Regras de Associação")
            
            apriori_status = "✅ Executado" if pipeline_status['algorithms'].get('apriori_rules', False) else "❌ Pendente"
            fp_status = "✅ Executado" if pipeline_status['algorithms'].get('fp_growth_rules', False) else "❌ Pendente"
            eclat_status = "✅ Executado" if pipeline_status['algorithms'].get('eclat_rules', False) else "❌ Pendente"
            
            apriori_records = pipeline_status['algorithms'].get('apriori_rules_records', 0)
            fp_records = pipeline_status['algorithms'].get('fp_growth_rules_records', 0)
            eclat_records = pipeline_status['algorithms'].get('eclat_rules_records', 0)
            
            st.markdown(f"""
            **APRIORI:** {apriori_status} ({apriori_records:,} regras)
            **FP-GROWTH:** {fp_status} ({fp_records:,} regras)
            **ECLAT:** {eclat_status} ({eclat_records:,} regras)
            
            *Objetivo: Descobrir padrões frequentes entre variáveis categóricas*
            """)
        
        # Executar pipeline se necessário
        if not pipeline_status['executed']:
            st.warning("⚠️ Pipeline principal não foi executado completamente")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.info("""
                💡 **Para obter resultados completos:**
                1. Execute `python main.py` no terminal
                2. Aguarde a conclusão (1-2 minutos)
                3. Recarregue esta página
                """)
            
            with col2:
                if st.button("🚀 Executar Pipeline", type="primary"):
                    self._execute_main_pipeline()
        
        # Resumo do pipeline (se disponível)
        if pipeline_status['summary']:
            st.subheader("📄 Resumo da Execução")
            
            with st.expander("Ver Resumo Completo", expanded=False):
                st.text(pipeline_status['summary'])
        
        # Visualizações de overview
        self._show_overview_visualizations(df)
    
    def show_clustering_page(self, df, pipeline_status):
        """Página de análise de clustering DBSCAN"""
        st.title("🎯 Análise de Clustering - DBSCAN")
        st.markdown("---")
        
        # Explicação teórica
        with st.expander("📚 Sobre o Algoritmo DBSCAN", expanded=False):
            st.markdown("""
            **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**
            
            - **Tipo:** Clustering baseado em densidade
            - **Vantagens:** Identifica clusters de formas arbitrárias, detecta outliers automaticamente
            - **Variáveis utilizadas:** age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week
            - **Parâmetros:** eps (distância máxima), min_samples (pontos mínimos por cluster)
            
            **Justificativa da Escolha das Variáveis:**
            1. **age** - Experiência profissional correlaciona com idade
            2. **education-num** - Anos de estudo = principal preditor salarial  
            3. **hours-per-week** - Carga horária reflete dedicação/disponibilidade
            4. **capital-gain/loss** - Indicadores de classe socioeconômica
            5. **fnlwgt** - Peso populacional para representatividade
            """)
        
        # Resultados do DBSCAN
        if pipeline_status['algorithms'].get('dbscan_results', False):
            dbscan_data = self.data_loader.load_algorithm_results('dbscan_results')
            
            if dbscan_data is not None and not dbscan_data.empty:
                st.success(f"✅ Resultados DBSCAN carregados: {len(dbscan_data):,} pontos analisados")
                
                # Métricas de clustering
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if 'cluster' in dbscan_data.columns:
                        n_clusters = len(dbscan_data['cluster'].unique()) - (1 if -1 in dbscan_data['cluster'].unique() else 0)
                        st.metric("🎯 Clusters Identificados", n_clusters)
                
                with col2:
                    if 'cluster' in dbscan_data.columns:
                        noise_points = (dbscan_data['cluster'] == -1).sum()
                        st.metric("🔊 Pontos de Ruído", noise_points)
                
                with col3:
                    if 'cluster' in dbscan_data.columns:
                        noise_percentage = (dbscan_data['cluster'] == -1).mean()
                        st.metric("📊 % Ruído", f"{noise_percentage:.1%}")
                
                with col4:
                    # Tentar calcular silhouette score se disponível
                    try:
                        from sklearn.metrics import silhouette_score
                        from sklearn.preprocessing import StandardScaler
                        
                        # Preparar dados para silhouette
                        numeric_cols = ['age', 'education-num', 'hours-per-week', 'capital-gain', 'capital-loss', 'fnlwgt']
                        available_cols = [col for col in numeric_cols if col in dbscan_data.columns]
                        
                        if len(available_cols) > 1 and 'cluster' in dbscan_data.columns:
                            X = dbscan_data[available_cols].fillna(0)
                            labels = dbscan_data['cluster']
                            
                            # Apenas calcular se há clusters válidos (não apenas ruído)
                            if len(labels.unique()) > 1 and not all(labels == -1):
                                scaler = StandardScaler()
                                X_scaled = scaler.fit_transform(X)
                                silhouette = silhouette_score(X_scaled, labels)
                                st.metric("📈 Silhouette Score", f"{silhouette:.3f}")
                            else:
                                st.metric("📈 Silhouette Score", "N/A")
                        else:
                            st.metric("📈 Silhouette Score", "N/A")
                    except:
                        st.metric("📈 Silhouette Score", "N/A")
                
                # Visualizações dos clusters
                st.subheader("📊 Visualização dos Clusters")
                
                self._plot_dbscan_results(dbscan_data)
                
                # Análise dos perfis dos clusters
                st.subheader("👥 Perfis dos Clusters")
                
                self._analyze_cluster_profiles(dbscan_data)
                
                # Dados tabulares
                st.subheader("📋 Dados dos Clusters")
                
                # Filtros
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'cluster' in dbscan_data.columns:
                        cluster_options = ['Todos'] + sorted(dbscan_data['cluster'].unique().tolist())
                        selected_cluster = st.selectbox("Filtrar por Cluster:", cluster_options)
                
                with col2:
                    show_noise = st.checkbox("Incluir Pontos de Ruído", value=True)
                
                # Aplicar filtros
                filtered_data = dbscan_data.copy()
                
                if selected_cluster != 'Todos':
                    filtered_data = filtered_data[filtered_data['cluster'] == selected_cluster]
                
                if not show_noise and 'cluster' in filtered_data.columns:
                    filtered_data = filtered_data[filtered_data['cluster'] != -1]
                
                st.dataframe(
                    filtered_data,
                    use_container_width=True,
                    height=400
                )
                
            else:
                st.error("❌ Erro ao carregar dados do DBSCAN")
        else:
            st.info("ℹ️ Execute o pipeline principal (main.py) para gerar os resultados do DBSCAN")
            
            # Mostrar explicação conceitual
            st.subheader("🎓 Conceito Teórico - DBSCAN")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Processo do DBSCAN:**
                1. Define pontos core (≥ min_samples vizinhos)
                2. Expande clusters a partir dos pontos core
                3. Marca pontos isolados como ruído (-1)
                4. Agrupa pontos conectados por densidade
                """)
            
            with col2:
                st.markdown("""
                **Parâmetros Típicos:**
                - **eps:** 0.5 (distância máxima entre pontos)
                - **min_samples:** 5 (pontos mínimos por cluster)
                - **Normalização:** StandardScaler aplicado
                """)
    
    def show_association_rules_page(self, df, pipeline_status):
        """Página de regras de associação"""
        st.title("📋 Regras de Associação")
        st.markdown("---")
        
        # Seleção do algoritmo
        col1, col2 = st.columns([1, 3])
        
        with col1:
            algorithm = st.selectbox(
                "🔍 Algoritmo:",
                ["apriori", "fp_growth", "eclat"],
                help="Selecione o algoritmo de mineração de regras"
            )
        
        with col2:
            # Explicação do algoritmo selecionado
            algorithm_info = {
                "apriori": "**APRIORI**: Algoritmo clássico que gera candidatos iterativamente",
                "fp_growth": "**FP-GROWTH**: Usa árvore FP para mineração eficiente sem candidatos",
                "eclat": "**ECLAT**: Baseado em intersecção de conjuntos, eficiente para datasets densos"
            }
            st.info(algorithm_info[algorithm])
        
        # Carregar regras do algoritmo selecionado
        rules_key = f"{algorithm}_rules"
        
        if pipeline_status['algorithms'].get(rules_key, False):
            rules_data = self.data_loader.load_algorithm_results(rules_key)
            
            if rules_data is not None and not rules_data.empty:
                st.success(f"✅ {len(rules_data):,} regras {algorithm.upper()} carregadas")
                
                # Métricas das regras
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_support = rules_data['support'].mean() if 'support' in rules_data.columns else 0
                    st.metric("📊 Support Médio", f"{avg_support:.3f}")
                
                with col2:
                    avg_confidence = rules_data['confidence'].mean() if 'confidence' in rules_data.columns else 0
                    st.metric("🎯 Confidence Médio", f"{avg_confidence:.3f}")
                
                with col3:
                    if 'lift' in rules_data.columns:
                        avg_lift = rules_data['lift'].mean()
                        st.metric("📈 Lift Médio", f"{avg_lift:.3f}")
                    else:
                        st.metric("📈 Lift Médio", "N/A")
                
                with col4:
                    strong_rules = 0
                    if 'confidence' in rules_data.columns and 'lift' in rules_data.columns:
                        strong_rules = len(rules_data[(rules_data['confidence'] > 0.7) & (rules_data['lift'] > 1.2)])
                    st.metric("💪 Regras Fortes", strong_rules)
                
                # Filtros interativos
                st.subheader("🔍 Filtros de Análise")
                
                col1, col2, col3 = st.columns(3)
                
                filtered_rules = rules_data.copy()
                
                with col1:
                    if 'support' in rules_data.columns:
                        min_support = st.slider(
                            "Support mínimo:", 
                            0.0, 
                            float(rules_data['support'].max()), 
                            0.01, 
                            0.01,
                            help="Frequência mínima do conjunto de itens"
                        )
                        filtered_rules = filtered_rules[filtered_rules['support'] >= min_support]
                
                with col2:
                    if 'confidence' in rules_data.columns:
                        min_confidence = st.slider(
                            "Confidence mínimo:", 
                            0.0, 
                            1.0, 
                            0.5, 
                            0.01,
                            help="Probabilidade condicional da regra"
                        )
                        filtered_rules = filtered_rules[filtered_rules['confidence'] >= min_confidence]
                
                with col3:
                    if 'lift' in rules_data.columns:
                        min_lift = st.slider(
                            "Lift mínimo:", 
                            1.0, 
                            float(rules_data['lift'].max()) if rules_data['lift'].max() > 1 else 2.0, 
                            1.0, 
                            0.1,
                            help="Força da associação (>1 = positiva)"
                        )
                        filtered_rules = filtered_rules[filtered_rules['lift'] >= min_lift]
                
                # Resultados filtrados
                st.subheader(f"📋 Regras {algorithm.upper()} Filtradas ({len(filtered_rules)} de {len(rules_data)})")
                
                if not filtered_rules.empty:
                    # Preparar dados para exibição
                    display_rules = filtered_rules.copy()
                    
                    # Arredondar valores numéricos
                    numeric_cols = display_rules.select_dtypes(include=[np.number]).columns
                    display_rules[numeric_cols] = display_rules[numeric_cols].round(4)
                    
                    # Ordenar por lift decrescente
                    if 'lift' in display_rules.columns:
                        display_rules = display_rules.sort_values('lift', ascending=False)
                    
                    st.dataframe(
                        display_rules,
                        use_container_width=True,
                        height=400
                    )
                    
                    # Visualizações das regras
                    st.subheader("📊 Visualização das Regras")
                    
                    self._plot_association_rules(filtered_rules, algorithm)
                    
                    # Top regras
                    st.subheader("🏆 Top 10 Regras Mais Relevantes")
                    
                    if 'lift' in filtered_rules.columns:
                        top_rules = filtered_rules.nlargest(10, 'lift')
                        
                        for idx, rule in top_rules.iterrows():
                            with st.container():
                                col1, col2, col3 = st.columns([2, 1, 1])
                                
                                with col1:
                                    antecedents = rule.get('antecedents', 'N/A')
                                    consequents = rule.get('consequents', 'N/A')
                                    st.write(f"**Se** {antecedents} **→ Então** {consequents}")
                                
                                with col2:
                                    confidence = rule.get('confidence', 0)
                                    st.write(f"Conf: {confidence:.3f}")
                                
                                with col3:
                                    lift = rule.get('lift', 0)
                                    st.write(f"Lift: {lift:.3f}")
                
                else:
                    st.info("ℹ️ Nenhuma regra atende aos critérios selecionados. Ajuste os filtros.")
            else:
                st.error(f"❌ Erro ao carregar regras {algorithm.upper()}")
        else:
            st.info(f"ℹ️ Execute o pipeline principal para gerar regras {algorithm.upper()}")
            
            # Explicação conceitual
            st.subheader(f"🎓 Conceito - {algorithm.upper()}")
            
            concepts = {
                "apriori": {
                    "description": "Algoritmo clássico de mineração de regras de associação",
                    "steps": [
                        "1. Gera conjuntos frequentes de tamanho 1",
                        "2. Combina conjuntos para gerar candidatos maiores", 
                        "3. Poda candidatos infrequentes",
                        "4. Repete até não haver mais conjuntos frequentes",
                        "5. Gera regras a partir dos conjuntos frequentes"
                    ],
                    "complexity": "O(2^n) no pior caso"
                },
                "fp_growth": {
                    "description": "Algoritmo eficiente que usa estrutura de árvore FP",
                    "steps": [
                        "1. Constrói árvore FP compacta",
                        "2. Minera padrões por projeção condicional",
                        "3. Não gera candidatos explicitamente",
                        "4. Mais eficiente que APRIORI",
                        "5. Especialmente bom para datasets densos"
                    ],
                    "complexity": "Mais eficiente que APRIORI"
                },
                "eclat": {
                    "description": "Baseado em intersecção de conjuntos (vertical)",
                    "steps": [
                        "1. Representa dados em formato vertical",
                        "2. Usa intersecção de TID-sets",
                        "3. Breadth-first ou depth-first search",
                        "4. Eficiente para datasets densos",
                        "5. Menos uso de memória"
                    ],
                    "complexity": "Eficiente para dados densos"
                }
            }
            
            concept = concepts[algorithm]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**{concept['description']}**")
                st.markdown("**Passos do Algoritmo:**")
                for step in concept['steps']:
                    st.markdown(f"- {step}")
            
            with col2:
                st.markdown(f"**Complexidade:** {concept['complexity']}")
                st.markdown("**Métricas Importantes:**")
                st.markdown("""
                - **Support**: Freq(A∪B) / N
                - **Confidence**: Freq(A∪B) / Freq(A)  
                - **Lift**: Confidence(A→B) / Support(B)
                """)
    
    def show_ml_models_page(self, df, pipeline_status):
        """Página de modelos de Machine Learning"""
        st.title("🤖 Modelos de Machine Learning")
        st.markdown("---")
        
        models_status = pipeline_status['models']
        
        # Header explicativo
        st.markdown("""
        <div style="background: #f0f2f6; padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">
            <h4>🎯 Objetivo: Predição de Faixa Salarial (>50K vs ≤50K)</h4>
            <p><strong>Problema:</strong> Classificação binária supervisionada</p>
            <p><strong>Features:</strong> age, education-num, hours-per-week, capital-gain, capital-loss, fnlwgt + variáveis categóricas</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Status dos modelos
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌳 Random Forest")
            
            if models_status.get('random_forest', False):
                st.success("✅ Modelo treinado e disponível")
                
                # Métricas baseadas no relatório acadêmico
                st.markdown("""
                **Performance (baseada no relatório):**
                - **Acurácia:** ~84.08% - 86.32%
                - **Tipo:** Ensemble (múltiplas árvores)
                - **Vantagens:** Robustez, lida com overfitting
                - **Interpretabilidade:** Feature importance disponível
                """)
                
                # Métricas em cards
                met_col1, met_col2 = st.columns(2)
                with met_col1:
                    st.metric("🎯 Acurácia", "84.08%", "Alta performance")
                with met_col2:
                    st.metric("🌲 Árvores", "100", "Ensemble robusto")
                
            else:
                st.error("❌ Modelo não encontrado")
                st.info("Execute `python main.py` para treinar")
        
        with col2:
            st.subheader("📈 Logistic Regression")
            
            if models_status.get('logistic_regression', False):
                st.success("✅ Modelo treinado e disponível")
                
                st.markdown("""
                **Performance (baseada no relatório):**
                - **Acurácia:** ~81.85% - 85.87%
                - **Tipo:** Linear (baseline)
                - **Vantagens:** Interpretabilidade máxima
                - **Coeficientes:** Facilmente explicáveis
                """)
                
                # Métricas em cards
                met_col1, met_col2 = st.columns(2)
                with met_col1:
                    st.metric("🎯 Acurácia", "81.85%", "Baseline sólido")
                with met_col2:
                    st.metric("🔍 Interpretabilidade", "Máxima", "Linear")
                
            else:
                st.error("❌ Modelo não encontrado")
                st.info("Execute `python main.py` para treinar")
        
        # Comparação de modelos
        st.subheader("⚖️ Comparação de Performance")
        
        if models_status.get('random_forest', False) and models_status.get('logistic_regression', False):
            # Dados baseados no relatório acadêmico
            comparison_data = {
                'Modelo': ['Random Forest', 'Logistic Regression'],
                'Acurácia': [0.8408, 0.8185],  # Valores do relatório
                'Precisão': [0.8592, 0.8531],
                'Recall': [0.8632, 0.8587],
                'F1-Score': [0.8532, 0.8539],
                'Interpretabilidade': ['Moderada', 'Alta'],
                'Tipo': ['Ensemble', 'Linear']
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Gráfico de comparação
            fig = px.bar(
                comparison_df, 
                x='Modelo', 
                y='Acurácia',
                title="🏆 Comparação de Acurácia dos Modelos",
                color='Modelo',
                text='Acurácia',
                color_discrete_sequence=['#1f77b4', '#ff7f0e']
            )
            fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
            fig.update_layout(showlegend=False, yaxis_tickformat='.0%')
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabela detalhada
            st.subheader("📊 Métricas Detalhadas")
            
            # Preparar dados para exibição
            display_df = comparison_df.copy()
            
            # Formatar percentuais
            for col in ['Acurácia', 'Precisão', 'Recall', 'F1-Score']:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}")
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Interpretação dos resultados
            st.subheader("🧠 Interpretação dos Resultados")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Random Forest (Vencedor):**
                - ✅ Melhor acurácia geral (84.08%)
                - ✅ Captura relações não-lineares
                - ✅ Resistente a overfitting
                - ⚠️ Menos interpretável ("black box")
                - 🎯 **Recomendado para predição**
                """)
            
            with col2:
                st.markdown("""
                **Logistic Regression (Baseline):**
                - ✅ Altamente interpretável
                - ✅ Rápido para treinar/prever
                - ✅ Coeficientes explicáveis
                - ⚠️ Assume relações lineares
                - 🎯 **Recomendado para explicação**
                """)
        
        else:
            st.info("ℹ️ Execute o pipeline principal para ver a comparação completa")
            
            # Placeholder com dados teóricos
            st.markdown("""
            **Comparação Esperada (baseada na literatura):**
            - Random Forest geralmente supera modelos lineares
            - Diferença típica: 2-5% de acurácia
            - Trade-off: Performance vs Interpretabilidade
            """)
        
        # Simulador de predição
        st.subheader("🔮 Simulador de Predição")
        
        st.markdown("*Simule uma predição com características de entrada:*")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.slider("Idade", 18, 80, 35, help="Idade da pessoa")
            education_num = st.slider("Anos de Educação", 1, 16, 10, help="Anos de escolaridade formal")
        
        with col2:
            hours_per_week = st.slider("Horas por Semana", 1, 100, 40, help="Horas trabalhadas por semana")
            capital_gain = st.number_input("Ganhos de Capital", 0, 100000, 0, help="Capital gains anuais")
        
        with col3:
            capital_loss = st.number_input("Perdas de Capital", 0, 10000, 0, help="Capital losses anuais")
            sex = st.selectbox("Sexo", ["Male", "Female"])
        
        if st.button("🔮 Fazer Predição Simulada", type="primary"):
            # Simulação baseada em heurísticas do dataset
            
            # Calcular score baseado nas variáveis
            score = 0
            
            # Idade (experiência)
            if age > 40:
                score += 0.3
            elif age > 30:
                score += 0.1
            
            # Educação (principal fator)
            if education_num >= 13:  # Bachelor+
                score += 0.4
            elif education_num >= 10:  # Some college
                score += 0.2
            
            # Horas (dedicação)
            if hours_per_week > 50:
                score += 0.2
            elif hours_per_week > 40:
                score += 0.1
            
            # Capital (riqueza)
            if capital_gain > 1000:
                score += 0.3
            
            # Adicionar ruído para realismo
            import random
            score += random.uniform(-0.1, 0.1)
            
            # Normalizar
            probability = min(max(score, 0), 1)
            prediction = ">50K" if probability > 0.5 else "≤50K"
            
            # Mostrar resultado
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                if prediction == ">50K":
                    st.success(f"**Predição: {prediction}** 💰")
                else:
                    st.info(f"**Predição: {prediction}**")
            
            with result_col2:
                st.metric("Probabilidade >50K", f"{probability:.1%}")
            
            # Explicação
            st.markdown("""
            **⚠️ Nota:** Esta é uma simulação baseada em heurísticas.
            Para predições reais, execute o pipeline principal e use os modelos treinados.
            """)
    
    def show_academic_report_page(self, pipeline_status):
        """Página do relatório acadêmico"""
        st.title("📚 Relatório Acadêmico")
        st.markdown("---")
        
        # Verificar se relatório existe
        if pipeline_status['algorithms'].get('academic_report', False):
            report_content = self.data_loader.load_algorithm_results('academic_report')
            
            if report_content:
                st.success("✅ Relatório acadêmico disponível")
                
                # Mostrar conteúdo do relatório
                st.markdown(report_content)
                
                # Download do relatório
                st.download_button(
                    "📥 Download Relatório Completo",
                    report_content,
                    "relatorio_academico_v2.md",
                    "text/markdown",
                    help="Baixar relatório em formato Markdown"
                )
            else:
                st.error("❌ Erro ao carregar o relatório")
        else:
            st.info("ℹ️ Execute o pipeline principal (main.py) para gerar o relatório acadêmico")
            
            # Mostrar estrutura esperada do relatório
            st.subheader("📋 Estrutura do Relatório Acadêmico")
            
            st.markdown("""
            O relatório acadêmico completo incluirá:
            
            ## 📖 **1. Introdução e Objetivos**
            - Contextualização do problema
            - Objetivos específicos e gerais
            - Justificativa científica
            
            ## 🔬 **2. Metodologia Científica**
            - Descrição dos algoritmos implementados
            - Fundamentação teórica
            - Pipeline de análise
            
            ## 📊 **3. Análise Exploratória dos Dados**
            - Caracterização do dataset (32.561 registros)
            - Distribuições das variáveis
            - Correlações e insights preliminares
            
            ## 🤖 **4. Resultados dos Algoritmos**
            
            ### **4.1 DBSCAN Clustering**
            - Parâmetros otimizados
            - Clusters identificados
            - Perfis socioeconômicos
            
            ### **4.2 Regras de Associação**
            - **APRIORI**: Regras clássicas
            - **FP-GROWTH**: Mineração eficiente
            - **ECLAT**: Intersecção de conjuntos
            
            ### **4.3 Modelos Supervisionados**
            - **Random Forest**: 84.08% acurácia
            - **Logistic Regression**: 81.85% acurácia
            - Comparação e validação
            
            ## 💡 **5. Discussão e Interpretação**
            - Insights principais
            - Implicações práticas
            - Limitações identificadas
            
            ## 🎯 **6. Conclusões e Trabalhos Futuros**
            - Síntese dos resultados
            - Recomendações para organizações
            - Direções para pesquisas futuras
            
            ## 📚 **7. Referências Bibliográficas**
            - Literatura científica consultada
            - Fundamentação teórica
            """)
            
            # Botão para executar pipeline
            if st.button("🚀 Gerar Relatório (Executar Pipeline)", type="primary"):
                self._execute_main_pipeline()
    
    def _execute_main_pipeline(self):
        """Executar pipeline principal"""
        try:
            with st.spinner("🔄 Executando pipeline principal... (pode demorar 1-2 minutos)"):
                result = subprocess.run(
                    [sys.executable, 'main.py'], 
                    capture_output=True, 
                    text=True, 
                    timeout=300
                )
                
                if result.returncode == 0:
                    st.success("✅ Pipeline executado com sucesso!")
                    st.balloons()
                    
                    # Mostrar saída se houver
                    if result.stdout:
                        with st.expander("📋 Log de Execução"):
                            st.text(result.stdout)
                    
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error(f"❌ Erro na execução do pipeline:")
                    if result.stderr:
                        st.code(result.stderr)
                    
        except subprocess.TimeoutExpired:
            st.warning("⏰ Timeout - Pipeline ainda em execução. Aguarde e recarregue a página.")
        except FileNotFoundError:
            st.error("❌ Arquivo main.py não encontrado. Verifique se está no diretório correto.")
        except Exception as e:
            st.error(f"❌ Erro inesperado: {e}")
    
    def _show_overview_visualizations(self, df):
        """Visualizações da página de overview"""
        st.subheader("📊 Análise Exploratória do Dataset")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribuição de salários
            if 'salary' in df.columns:
                salary_counts = df['salary'].value_counts()
                
                fig = px.pie(
                    values=salary_counts.values,
                    names=salary_counts.index,
                    title="💰 Distribuição de Faixas Salariais",
                    color_discrete_sequence=['#ff7f0e', '#1f77b4']
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Distribuição por sexo
            if 'sex' in df.columns:
                sex_counts = df['sex'].value_counts()
                
                fig = px.bar(
                    x=sex_counts.index,
                    y=sex_counts.values,
                    title="👥 Distribuição por Gênero",
                    color=sex_counts.index,
                    color_discrete_sequence=['#2ca02c', '#d62728']
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        # Distribuição de idade e educação
        col1, col2 = st.columns(2)
        
        with col1:
            if 'age' in df.columns:
                fig = px.histogram(
                    df, 
                    x='age', 
                    title="📈 Distribuição de Idades",
                    nbins=20,
                    color_discrete_sequence=['#9467bd']
                )
                fig.update_layout(yaxis_title="Frequência")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'education-num' in df.columns:
                fig = px.histogram(
                    df, 
                    x='education-num', 
                    title="🎓 Distribuição de Anos de Educação",
                    nbins=16,
                    color_discrete_sequence=['#8c564b']
                )
                fig.update_layout(yaxis_title="Frequência")
                st.plotly_chart(fig, use_container_width=True)
    
    def _plot_dbscan_results(self, dbscan_data):
        """Plotar resultados do DBSCAN"""
        if 'cluster' in dbscan_data.columns:
            # Preparar dados para visualização
            plot_cols = []
            
            # Tentar usar as variáveis principais do clustering
            preferred_cols = ['age', 'education-num', 'hours-per-week', 'capital-gain']
            
            for col in preferred_cols:
                if col in dbscan_data.columns:
                    plot_cols.append(col)
                if len(plot_cols) >= 2:
                    break
            
            # Fallback para qualquer coluna numérica
            if len(plot_cols) < 2:
                numeric_cols = dbscan_data.select_dtypes(include=[np.number]).columns
                numeric_cols = [col for col in numeric_cols if col != 'cluster']
                plot_cols = list(numeric_cols[:2])
            
            if len(plot_cols) >= 2:
                # Scatter plot dos clusters
                fig = px.scatter(
                    dbscan_data,
                    x=plot_cols[0],
                    y=plot_cols[1],
                    color='cluster',
                    title=f"🎯 Clusters DBSCAN - {plot_cols[0]} vs {plot_cols[1]}",
                    labels={'cluster': 'Cluster ID'},
                    hover_data=['cluster']
                )
                
                # Destacar ruído com cor especial
                fig.update_traces(
                    marker=dict(size=6, opacity=0.7)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Distribuição dos clusters
                if len(dbscan_data['cluster'].unique()) > 1:
                    cluster_counts = dbscan_data['cluster'].value_counts().sort_index()
                    
                    # Renomear cluster -1 para "Ruído"
                    cluster_names = []
                    for cluster_id in cluster_counts.index:
                        if cluster_id == -1:
                            cluster_names.append("Ruído")
                        else:
                            cluster_names.append(f"Cluster {cluster_id}")
                    
                    fig2 = px.bar(
                        x=cluster_names,
                        y=cluster_counts.values,
                        title="📊 Distribuição dos Clusters",
                        color=cluster_names
                    )
                    fig2.update_layout(showlegend=False, xaxis_title="Cluster", yaxis_title="Número de Pontos")
                    st.plotly_chart(fig2, use_container_width=True)
            else:
                st.warning("⚠️ Dados insuficientes para visualização dos clusters")
    
    def _analyze_cluster_profiles(self, dbscan_data):
        """Analisar perfis dos clusters"""
        if 'cluster' in dbscan_data.columns:
            # Análise por cluster
            clusters = sorted(dbscan_data['cluster'].unique())
            
            # Estatísticas por cluster
            numeric_cols = dbscan_data.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col != 'cluster']
            
            if len(numeric_cols) > 0:
                st.subheader("📈 Estatísticas por Cluster")
                
                profiles = []
                
                for cluster_id in clusters:
                    cluster_data = dbscan_data[dbscan_data['cluster'] == cluster_id]
                    
                    cluster_name = "Ruído" if cluster_id == -1 else f"Cluster {cluster_id}"
                    
                    profile = {
                        'Cluster': cluster_name,
                        'Tamanho': len(cluster_data),
                        'Percentual': f"{len(cluster_data)/len(dbscan_data)*100:.1f}%"
                    }
                    
                    # Adicionar médias das variáveis numéricas principais
                    for col in numeric_cols[:4]:  # Limitar para não ficar muito largo
                        if col in cluster_data.columns:
                            profile[f'{col}_média'] = f"{cluster_data[col].mean():.1f}"
                    
                    profiles.append(profile)
                
                profiles_df = pd.DataFrame(profiles)
                st.dataframe(profiles_df, use_container_width=True, hide_index=True)
                
                # Interpretação dos clusters
                st.subheader("🧠 Interpretação dos Perfis")
                
                if len(clusters) > 1:
                    for cluster_id in clusters:
                        if cluster_id != -1:  # Ignorar ruído na interpretação
                            cluster_data = dbscan_data[dbscan_data['cluster'] == cluster_id]
                            
                            with st.container():
                                st.markdown(f"**Cluster {cluster_id}** ({len(cluster_data)} pessoas)")
                                
                                # Características principais
                                characteristics = []
                                
                                if 'age' in cluster_data.columns:
                                    avg_age = cluster_data['age'].mean()
                                    if avg_age < 30:
                                        characteristics.append("👶 Jovens")
                                    elif avg_age > 50:
                                        characteristics.append("👴 Experientes")
                                    else:
                                        characteristics.append("👨 Adultos")
                                
                                if 'education-num' in cluster_data.columns:
                                    avg_edu = cluster_data['education-num'].mean()
                                    if avg_edu >= 13:
                                        characteristics.append("🎓 Alta escolaridade")
                                    elif avg_edu <= 9:
                                        characteristics.append("📚 Baixa escolaridade")
                                    else:
                                        characteristics.append("🏫 Escolaridade média")
                                
                                if 'hours-per-week' in cluster_data.columns:
                                    avg_hours = cluster_data['hours-per-week'].mean()
                                    if avg_hours > 50:
                                        characteristics.append("⏰ Carga horária alta")
                                    elif avg_hours < 35:
                                        characteristics.append("🕐 Meio período")
                                
                                st.markdown(f"- {' • '.join(characteristics)}")
                                st.markdown("---")
    
    def _plot_association_rules(self, rules_data, algorithm):
        """Plotar visualizações das regras de associação"""
        if not rules_data.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Scatter plot Support vs Confidence
                if 'support' in rules_data.columns and 'confidence' in rules_data.columns:
                    size_col = 'lift' if 'lift' in rules_data.columns else None
                    
                    fig = px.scatter(
                        rules_data,
                        x='support',
                        y='confidence',
                        size=size_col,
                        title=f"📊 {algorithm.upper()}: Support vs Confidence",
                        hover_data=['lift'] if 'lift' in rules_data.columns else None,
                        labels={
                            'support': 'Support (Frequência)',
                            'confidence': 'Confidence (Confiança)',
                            'lift': 'Lift (Força da Associação)'
                        }
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Histograma das métricas
                if 'lift' in rules_data.columns:
                    fig = px.histogram(
                        rules_data,
                        x='lift',
                        title=f"📈 {algorithm.upper()}: Distribuição do Lift",
                        nbins=20,
                        labels={'lift': 'Lift (Força da Associação)'}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# APLICAÇÃO PRINCIPAL
# =============================================================================

def main():
    """Aplicação principal do dashboard científico"""
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Inicializar sistemas
    auth_system = AcademicAuthSystem()
    
    # Verificar autenticação
    if not st.session_state.get('authenticated', False):
        auth_system.show_login_page()
        return
    
    # Header do usuário autenticado
    user_data = st.session_state.get('user', {})
    st.sidebar.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 1rem; border-radius: 10px; color: white; margin-bottom: 1rem;">
        <h4>👤 {user_data.get('name', 'Usuário')}</h4>
        <p><strong>Perfil:</strong> {user_data.get('role', 'N/A').title()}</p>
        <p><strong>Acesso:</strong> {user_data.get('access', 'N/A').title()}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Botão de logout
    if st.sidebar.button("🚪 Sair do Sistema", use_container_width=True):
        st.session_state.clear()
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Menu de navegação
    st.sidebar.markdown("### 📚 Navegação Acadêmica")
    
    page = st.sidebar.selectbox(
        "Selecione a página:",
        [
            "📊 Visão Geral",
            "🎯 Clustering (DBSCAN)", 
            "📋 Regras de Associação",
            "🤖 Modelos ML",
            "📚 Relatório Acadêmico"
        ],
        help="Navegue pelas diferentes análises científicas"
    )
    
    # Informações do projeto na sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 🎓 Sobre o Projeto
    
    **Algoritmos Implementados:**
    - 🎯 DBSCAN (Clustering)
    - 📋 APRIORI (Regras)
    - 🌲 FP-GROWTH (Regras)
    - 🔍 ECLAT (Regras)
    - 🤖 Random Forest
    - 📈 Logistic Regression
    
    **Dataset:** Adult Income (UCI)
    **Objetivo:** Análise Salarial
    """)
    
    # Carregar dados e status
    data_loader = ScientificDataLoader()
    
    with st.spinner("🔄 Carregando dados..."):
        df, load_message = data_loader.load_main_dataset()
        pipeline_status = data_loader.check_pipeline_execution()
    
    # Mostrar status de carregamento na sidebar
    if "demonstração" in load_message:
        st.sidebar.warning(load_message)
    else:
        st.sidebar.success(load_message)
    
    # Status do pipeline na sidebar
    execution_score = pipeline_status.get('execution_score', 0)
    
    if execution_score >= 0.8:
        st.sidebar.success(f"✅ Pipeline: {execution_score:.0%} completo")
    elif execution_score >= 0.5:
        st.sidebar.warning(f"⚠️ Pipeline: {execution_score:.0%} completo")
    else:
        st.sidebar.error(f"❌ Pipeline: {execution_score:.0%} completo")
    
    # Inicializar páginas
    pages = ScientificDashboardPages(data_loader)
    
    # Roteamento de páginas baseado na seleção
    try:
        if page == "📊 Visão Geral":
            pages.show_overview_page(df, pipeline_status)
        elif page == "🎯 Clustering (DBSCAN)":
            pages.show_clustering_page(df, pipeline_status)
        elif page == "📋 Regras de Associação":
            pages.show_association_rules_page(df, pipeline_status)
        elif page == "🤖 Modelos ML":
            pages.show_ml_models_page(df, pipeline_status)
        elif page == "📚 Relatório Acadêmico":
            pages.show_academic_report_page(pipeline_status)
    except Exception as e:
        st.error(f"❌ Erro ao carregar página: {e}")
        st.info("🔄 Tente recarregar a página ou contate o administrador")
        
        # Debug info para desenvolvimento
        if user_data.get('role') == 'admin':
            with st.expander("🐛 Debug Info (Admin)"):
                st.code(str(e))
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🎓 <strong>Dashboard Científico Acadêmico</strong> | 
        Desenvolvido para Análise Salarial | 
        <i>Versão 3.0 - Totalmente Integrado</i></p>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# CSS PERSONALIZADO PARA TEMA ACADÊMICO
# =============================================================================

def apply_academic_theme():
    """Aplicar tema visual acadêmico"""
    st.markdown("""
    <style>
    /* Tema acadêmico */
    .main > div {
        padding-top: 2rem;
    }
    
    /* Cards de métricas */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Header acadêmico */
    .academic-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
    }
    
    /* Status cards */
    .status-card {
        border: 2px solid #e0e0e0;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .status-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .status-success {
        border-color: #4caf50;
        background: linear-gradient(135deg, #f1f8e9 0%, #e8f5e8 100%);
    }
    
    .status-warning {
        border-color: #ff9800;
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
    }
    
    .status-error {
        border-color: #f44336;
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
    }
    
    /* Botões personalizados */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar personalizada */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Tabelas */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Métricas do Streamlit */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border: 1px solid #e9ecef;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        font-weight: 600;
    }
    
    /* Alerts personalizados */
    .stAlert {
        border-radius: 10px;
        border: none;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Títulos */
    h1 {
        color: #1e3c72;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    h2 {
        color: #2a5298;
        font-weight: 600;
    }
    
    h3 {
        color: #667eea;
        font-weight: 600;
    }
    
    /* Loading spinner */
    .stSpinner {
        text-align: center;
    }
    
    /* Plotly charts */
    .js-plotly-plot {
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Footer */
    .footer {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-top: 2rem;
    }
    
    /* Animações */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fadeIn {
        animation: fadeIn 0.6s ease-out;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .metric-card {
            padding: 1rem;
        }
        
        .academic-header {
            padding: 1.5rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# PONTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    # Aplicar tema visual
    apply_academic_theme()
    
    # Executar aplicação principal
    main()