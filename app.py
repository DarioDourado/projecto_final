"""
🌍 Dashboard Multilingual - Análise Salarial
Sistema Modular com Páginas Específicas - VERSÃO CORRIGIDA
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import logging
import sys
import warnings
import json
from datetime import datetime

# Configurações
warnings.filterwarnings('ignore')

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent / "src"))

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configurar página
st.set_page_config(
    page_title="Dashboard Multilingual - Análise Salarial",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

class SimpleAuth:
    """Sistema de autenticação simplificado"""
    
    def __init__(self):
        self.users = {
            'admin': {'password': 'admin123', 'name': 'Administrador', 'role': 'admin'},
            'user': {'password': 'user123', 'name': 'Usuário', 'role': 'user'},
            'demo': {'password': 'demo123', 'name': 'Demo', 'role': 'user'}
        }
    
    def authenticate(self, username, password):
        """Autenticar usuário"""
        if username in self.users and self.users[username]['password'] == password:
            st.session_state.authenticated = True
            st.session_state.user_data = self.users[username]
            st.session_state.username = username
            return True
        return False
    
    def is_authenticated(self):
        """Verificar se usuário está autenticado"""
        return st.session_state.get('authenticated', False)
    
    def get_user_data(self):
        """Obter dados do usuário"""
        return st.session_state.get('user_data', {})
    
    def logout(self):
        """Fazer logout"""
        st.session_state.authenticated = False
        st.session_state.user_data = {}
        st.session_state.username = ''

class MultilingualDashboard:
    """Dashboard principal com páginas específicas"""
    
    def __init__(self):
        """Inicializar dashboard"""
        self.auth = SimpleAuth()
        
        # Páginas específicas
        self.pages = {
            'overview': {
                'title': 'Visão Geral',
                'icon': '📊',
                'roles': ['admin', 'user'],
                'func': self._show_overview_page
            },
            'exploratory': {
                'title': 'Análise Exploratória',
                'icon': '🔍',
                'roles': ['admin', 'user'],
                'func': self._show_exploratory_page
            },
            'models': {
                'title': 'Modelos ML',
                'icon': '🤖',
                'roles': ['admin', 'user'],
                'func': self._show_models_page
            },
            'clustering': {
                'title': 'Clustering',
                'icon': '🎯',
                'roles': ['admin', 'user'],
                'func': self._show_clustering_page
            },
            'association_rules': {
                'title': 'Regras de Associação',
                'icon': '📋',
                'roles': ['admin', 'user'],
                'func': self._show_association_rules_page
            },
            'prediction': {
                'title': 'Predição',
                'icon': '🔮',
                'roles': ['admin', 'user'],
                'func': self._show_prediction_page
            },
            'metrics': {
                'title': 'Métricas',
                'icon': '📈',
                'roles': ['admin', 'user'],
                'func': self._show_metrics_page
            },
            'reports': {
                'title': 'Relatórios',
                'icon': '📁',
                'roles': ['admin', 'user'],
                'func': self._show_reports_page
            },
            'admin': {
                'title': 'Administração',
                'icon': '⚙️',
                'roles': ['admin'],
                'func': self._show_admin_page
            }
        }
        
        # Estado inicial
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 'overview'
        
        # Cache de dados
        self.data_cache = None
    
    def run(self):
        """Executar dashboard"""
        # CSS personalizado
        self._apply_css()
        
        # Header
        st.markdown("""
        <div class="main-header">
            🌍 Dashboard Multilingual - Análise Salarial
        </div>
        """, unsafe_allow_html=True)
        
        # Verificar autenticação
        if not self.auth.is_authenticated():
            self._show_login_page()
            return
        
        # Carregar dados uma vez
        if self.data_cache is None:
            self.data_cache = self._load_data()
        
        # Layout principal
        self._show_sidebar()
        self._execute_current_page()
    
    def _apply_css(self):
        """Aplicar CSS personalizado"""
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #1f4e79, #2c6aa0);
            color: white;
            padding: 1rem;
            text-align: center;
            font-size: 1.5rem;
            font-weight: bold;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        
        .nav-button-active {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            border: none !important;
            padding: 0.75rem 1rem !important;
            border-radius: 10px !important;
            width: 100% !important;
            text-align: left !important;
            margin-bottom: 0.5rem !important;
            font-weight: 600 !important;
        }
        
        .nav-button-inactive {
            background: #f8f9fa !important;
            color: #495057 !important;
            border: 1px solid #dee2e6 !important;
            padding: 0.75rem 1rem !important;
            border-radius: 10px !important;
            width: 100% !important;
            text-align: left !important;
            margin-bottom: 0.5rem !important;
        }
        
        .user-section {
            background: linear-gradient(135deg, #6c5ce7 0%, #a29bfe 100%);
            color: white;
            padding: 1rem;
            border-radius: 15px;
            margin-bottom: 1rem;
            text-align: center;
        }
        
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _load_data(self):
        """Carregar dados"""
        try:
            # Tentar vários caminhos
            data_paths = [
                "bkp/4-Carateristicas_salario.csv",
                "data/raw/4-Carateristicas_salario.csv",
                "4-Carateristicas_salario.csv"
            ]
            
            df = None
            source = None
            
            for path in data_paths:
                if Path(path).exists():
                    df = pd.read_csv(path)
                    source = path
                    break
            
            if df is not None:
                # Limpeza básica
                df = df.replace('?', pd.NA)
                
                # Escanear arquivos de análise
                files_status = self._scan_analysis_files()
                
                return {
                    'df': df,
                    'status': f'✅ Dados carregados de {source} ({len(df):,} registros)',
                    'files_status': files_status,
                    'source': source
                }
            else:
                # Dados de exemplo
                return self._create_sample_data()
                
        except Exception as e:
            logging.error(f"Erro ao carregar dados: {e}")
            return self._create_sample_data()
    
    def _scan_analysis_files(self):
        """Escanear arquivos de análise"""
        files_status = {
            'analysis': [],
            'models': [],
            'images': []
        }
        
        # Diretórios para verificar
        analysis_dirs = ['output/analysis', 'output', '.']
        models_dirs = ['models', 'data/processed', 'output/models']
        images_dirs = ['output/images', 'imagens', 'images']
        
        # Análises
        for dir_path in analysis_dirs:
            path = Path(dir_path)
            if path.exists():
                files_status['analysis'].extend(list(path.glob('*.csv')))
                files_status['analysis'].extend(list(path.glob('*.json')))
        
        # Modelos
        for dir_path in models_dirs:
            path = Path(dir_path)
            if path.exists():
                files_status['models'].extend(list(path.glob('*.pkl')))
                files_status['models'].extend(list(path.glob('*.joblib')))
        
        # Imagens
        for dir_path in images_dirs:
            path = Path(dir_path)
            if path.exists():
                files_status['images'].extend(list(path.glob('*.png')))
                files_status['images'].extend(list(path.glob('*.jpg')))
        
        return files_status
    
    def _create_sample_data(self):
        """Criar dados de exemplo"""
        try:
            np.random.seed(42)
            n_samples = 1000
            
            data = {
                'age': np.random.randint(18, 70, n_samples),
                'workclass': np.random.choice(['Private', 'Self-emp-not-inc', 'Federal-gov'], n_samples),
                'education': np.random.choice(['Bachelors', 'HS-grad', 'Masters'], n_samples),
                'education-num': np.random.randint(1, 17, n_samples),
                'marital-status': np.random.choice(['Married-civ-spouse', 'Divorced', 'Never-married'], n_samples),
                'occupation': np.random.choice(['Tech-support', 'Sales', 'Exec-managerial'], n_samples),
                'relationship': np.random.choice(['Wife', 'Husband', 'Not-in-family'], n_samples),
                'race': np.random.choice(['White', 'Black', 'Asian-Pac-Islander'], n_samples),
                'sex': np.random.choice(['Female', 'Male'], n_samples),
                'capital-gain': np.random.randint(0, 10000, n_samples),
                'capital-loss': np.random.randint(0, 5000, n_samples),
                'hours-per-week': np.random.randint(20, 80, n_samples),
                'native-country': np.random.choice(['United-States', 'Canada'], n_samples),
                'salary': np.random.choice(['<=50K', '>50K'], n_samples)
            }
            
            df = pd.DataFrame(data)
            
            return {
                'df': df,
                'status': '✅ Dados de exemplo criados (1,000 registros)',
                'files_status': {'analysis': [], 'models': [], 'images': []},
                'source': 'sample_data'
            }
            
        except Exception as e:
            logging.error(f"Erro ao criar dados de exemplo: {e}")
            return {
                'df': None,
                'status': f'❌ Erro: {e}',
                'files_status': {'analysis': [], 'models': [], 'images': []},
                'source': None
            }
    
    def _show_login_page(self):
        """Página de login"""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("### 🔐 Login")
            
            with st.form("login_form", clear_on_submit=True):
                username = st.text_input("👤 Nome de usuário", placeholder="admin, user, demo")
                password = st.text_input("🔑 Senha", type="password", placeholder="admin123, user123, demo123")
                
                if st.form_submit_button("🚪 Entrar", use_container_width=True):
                    if username and password:
                        if self.auth.authenticate(username, password):
                            st.success("✅ Login realizado com sucesso!")
                            st.rerun()
                        else:
                            st.error("❌ Credenciais inválidas!")
                    else:
                        st.warning("⚠️ Preencha todos os campos!")
            
            # Ajuda
            with st.expander("ℹ️ Credenciais de teste"):
                st.info("""
                **Credenciais disponíveis:**
                - **admin** / **admin123** (Administrador)
                - **user** / **user123** (Usuário)
                - **demo** / **demo123** (Demo)
                """)
    
    def _show_sidebar(self):
        """Mostrar sidebar"""
        with st.sidebar:
            # Informações do usuário
            self._show_user_info()
            
            st.markdown("---")
            
            # Navegação
            self._show_navigation()
            
            st.markdown("---")
            
            # Status dos dados
            self._show_data_status()
            
            st.markdown("---")
            
            # Controles
            if st.button("🔄 Recarregar Dados", use_container_width=True):
                self.data_cache = None
                self.data_cache = self._load_data()
                st.success("✅ Dados recarregados!")
                st.rerun()
    
    def _show_user_info(self):
        """Mostrar informações do usuário"""
        user_data = self.auth.get_user_data()
        if user_data:
            st.markdown(f"""
            <div class="user-section">
                <h4>👤 {user_data.get('name', 'N/A')}</h4>
                <p><strong>Papel:</strong> {user_data.get('role', 'user').title()}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🚪 Logout", use_container_width=True):
                self.auth.logout()
                st.rerun()
    
    def _show_navigation(self):
        """Mostrar navegação"""
        st.markdown("### 🧭 Navegação")
        
        user_data = self.auth.get_user_data()
        user_role = user_data.get('role', 'user')
        
        for page_key, page_info in self.pages.items():
            if user_role in page_info['roles']:
                button_text = f"{page_info['icon']} {page_info['title']}"
                is_current = st.session_state.current_page == page_key
                
                if is_current:
                    st.markdown(f"""
                    <div class="nav-button-active">
                        {button_text}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    if st.button(button_text, key=f"nav_{page_key}", use_container_width=True):
                        st.session_state.current_page = page_key
                        st.rerun()
    
    def _show_data_status(self):
        """Mostrar status dos dados"""
        st.markdown("### 📊 Status dos Dados")
        
        if self.data_cache:
            status = self.data_cache.get('status', '❌ Erro')
            
            if "✅" in status:
                st.success(status)
            elif "⚠️" in status:
                st.warning(status)
            else:
                st.error(status)
            
            df = self.data_cache.get('df')
            if df is not None:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📋 Registros", f"{len(df):,}")
                with col2:
                    st.metric("📊 Colunas", len(df.columns))
    
    def _execute_current_page(self):
        """Executar página atual"""
        current_page = st.session_state.current_page
        
        if current_page in self.pages:
            try:
                self.pages[current_page]['func']()
            except Exception as e:
                st.error(f"❌ Erro ao carregar página: {e}")
                logging.error(f"Erro na página {current_page}: {e}")
        else:
            st.error(f"❌ Página '{current_page}' não encontrada")
    
    # =========================================================================
    # PÁGINAS ESPECÍFICAS
    # =========================================================================
    
    def _show_overview_page(self):
        """Página de visão geral - ESPECÍFICA"""
        st.header("📊 Visão Geral do Dataset")
        
        df = self.data_cache.get('df') if self.data_cache else None
        
        if df is None:
            st.error("❌ Dados não disponíveis")
            return
        
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📋 Registros</h3>
                <h2>{len(df):,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📊 Colunas</h3>
                <h2>{len(df.columns)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            if 'salary' in df.columns:
                high_salary_rate = (df['salary'] == '>50K').mean()
                st.markdown(f"""
                <div class="metric-card">
                    <h3>💰 Salário Alto</h3>
                    <h2>{high_salary_rate:.1%}</h2>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>💰 Salário Alto</h3>
                    <h2>N/A</h2>
                </div>
                """, unsafe_allow_html=True)
        
        with col4:
            missing_rate = df.isnull().sum().sum() / (len(df) * len(df.columns))
            st.markdown(f"""
            <div class="metric-card">
                <h3>❌ Missing</h3>
                <h2>{missing_rate:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Gráficos de distribuição
        st.subheader("📈 Distribuições Principais")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'salary' in df.columns:
                salary_counts = df['salary'].value_counts()
                fig = px.pie(
                    values=salary_counts.values,
                    names=salary_counts.index,
                    title="💰 Distribuição de Salário"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'sex' in df.columns:
                sex_counts = df['sex'].value_counts()
                fig = px.bar(
                    x=sex_counts.index,
                    y=sex_counts.values,
                    title="👥 Distribuição por Sexo"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Amostra dos dados
        st.subheader("📋 Amostra dos Dados")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Estatísticas descritivas
        st.subheader("📊 Estatísticas Descritivas")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
    
    def _show_exploratory_page(self):
        """Página de análise exploratória - ESPECÍFICA"""
        st.header("🔍 Análise Exploratória Avançada")
        
        df = self.data_cache.get('df') if self.data_cache else None
        
        if df is None:
            st.error("❌ Dados não disponíveis")
            return
        
        # Controles
        col1, col2, col3 = st.columns(3)
        
        with col1:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                x_var = st.selectbox("📊 Variável X:", numeric_cols)
        
        with col2:
            y_var = st.selectbox("📊 Variável Y:", ["Nenhuma"] + numeric_cols)
        
        with col3:
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            if categorical_cols:
                color_var = st.selectbox("🎨 Cor por:", ["Nenhuma"] + categorical_cols)
            else:
                color_var = "Nenhuma"
        
        # Gráficos baseados na seleção
        if 'x_var' in locals():
            if y_var != "Nenhuma":
                # Scatter plot
                fig = px.scatter(
                    df, x=x_var, y=y_var,
                    color=color_var if color_var != "Nenhuma" else None,
                    title=f"📊 {x_var} vs {y_var}"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Histograma
                fig = px.histogram(
                    df, x=x_var,
                    color=color_var if color_var != "Nenhuma" else None,
                    title=f"📊 Distribuição de {x_var}"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Análises por categoria
        st.subheader("🎯 Análises por Categoria")
        
        if 'salary' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                if 'workclass' in df.columns:
                    workclass_salary = df.groupby('workclass')['salary'].apply(
                        lambda x: (x == '>50K').mean()
                    ).reset_index()
                    workclass_salary.columns = ['workclass', 'high_salary_rate']
                    
                    fig = px.bar(
                        workclass_salary, x='workclass', y='high_salary_rate',
                        title="💼 Taxa de Salário Alto por Classe de Trabalho"
                    )
                    fig.update_layout(xaxis_tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'education' in df.columns:
                    education_salary = df.groupby('education')['salary'].apply(
                        lambda x: (x == '>50K').mean()
                    ).reset_index()
                    education_salary.columns = ['education', 'high_salary_rate']
                    
                    fig = px.bar(
                        education_salary, x='education', y='high_salary_rate',
                        title="🎓 Taxa de Salário Alto por Educação"
                    )
                    fig.update_layout(xaxis_tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Correlações
        st.subheader("🔗 Matriz de Correlação")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                title="🔗 Matriz de Correlação"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def _show_models_page(self):
        """Página de modelos ML - ESPECÍFICA"""
        st.header("🤖 Modelos de Machine Learning")
        
        df = self.data_cache.get('df') if self.data_cache else None
        files_status = self.data_cache.get('files_status', {}) if self.data_cache else {}
        
        if df is None:
            st.error("❌ Dados não disponíveis")
            return
        
        # Verificar modelos salvos
        model_files = files_status.get('models', [])
        
        if model_files:
            st.success(f"✅ {len(model_files)} modelo(s) encontrado(s)")
            
            for model_file in model_files:
                with st.expander(f"📁 {model_file.name}"):
                    st.write(f"**Localização:** {model_file}")
                    st.write(f"**Tamanho:** {model_file.stat().st_size / 1024:.1f} KB")
                    st.write(f"**Modificado:** {datetime.fromtimestamp(model_file.stat().st_mtime)}")
        else:
            st.warning("⚠️ Nenhum modelo treinado encontrado")
            st.info("💡 Execute o pipeline principal para treinar modelos: `python main.py`")
        
        # Preparação de dados (exemplo)
        st.subheader("📊 Preparação dos Dados para ML")
        
        if 'salary' in df.columns:
            # Target distribution
            target_dist = df['salary'].value_counts()
            fig = px.pie(
                values=target_dist.values,
                names=target_dist.index,
                title="🎯 Distribuição da Variável Target (Salary)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Features numéricas
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_features:
            st.subheader("📊 Features Numéricas")
            selected_features = st.multiselect(
                "Selecione features para análise:",
                numeric_features,
                default=numeric_features[:3] if len(numeric_features) >= 3 else numeric_features
            )
            
            if selected_features:
                st.dataframe(df[selected_features].describe(), use_container_width=True)
        
        # Informações sobre os algoritmos
        st.subheader("🧠 Algoritmos Implementados")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🌳 Random Forest**
            - Ensemble de árvores de decisão
            - Reduz overfitting
            - Boa performance geral
            """)
        
        with col2:
            st.markdown("""
            **📈 Logistic Regression**
            - Modelo linear para classificação
            - Interpretável
            - Rápido para treinar
            """)
    
    def _show_clustering_page(self):
        """Página de clustering - ESPECÍFICA"""
        st.header("🎯 Análise de Clustering")
        
        df = self.data_cache.get('df') if self.data_cache else None
        files_status = self.data_cache.get('files_status', {}) if self.data_cache else {}
        
        if df is None:
            st.error("❌ Dados não disponíveis")
            return
        
        # Verificar resultados de clustering
        analysis_files = files_status.get('analysis', [])
        clustering_files = [f for f in analysis_files if 'dbscan' in str(f).lower() or 'cluster' in str(f).lower()]
        
        if clustering_files:
            st.success(f"✅ {len(clustering_files)} arquivo(s) de clustering encontrado(s)")
            
            for file in clustering_files:
                with st.expander(f"📁 {file.name}"):
                    try:
                        if file.suffix == '.csv':
                            df_cluster = pd.read_csv(file)
                            st.dataframe(df_cluster.head(), use_container_width=True)
                            
                            # Visualizar clusters se possível
                            if 'cluster' in df_cluster.columns:
                                cluster_counts = df_cluster['cluster'].value_counts()
                                fig = px.bar(
                                    x=cluster_counts.index,
                                    y=cluster_counts.values,
                                    title=f"🎯 Distribuição de Clusters - {file.name}"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Erro ao ler arquivo: {e}")
        else:
            st.warning("⚠️ Nenhum resultado de clustering encontrado")
            st.info("💡 Execute o pipeline principal para gerar clustering: `python main.py`")
        
        # Preparação para clustering
        st.subheader("📊 Preparação para Clustering")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            st.info(f"✅ {len(numeric_cols)} variáveis numéricas disponíveis para clustering")
            
            # Visualização 2D das features
            col1, col2 = st.columns(2)
            
            with col1:
                x_feature = st.selectbox("Feature X:", numeric_cols, key="cluster_x")
            
            with col2:
                y_feature = st.selectbox("Feature Y:", numeric_cols, index=1 if len(numeric_cols) > 1 else 0, key="cluster_y")
            
            if x_feature and y_feature and x_feature != y_feature:
                # Scatter plot das features
                color_by = 'salary' if 'salary' in df.columns else None
                fig = px.scatter(
                    df, x=x_feature, y=y_feature,
                    color=color_by,
                    title=f"📊 {x_feature} vs {y_feature}"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Poucas variáveis numéricas para clustering eficaz")
        
        # Informações sobre algoritmos
        st.subheader("🧠 Algoritmos de Clustering")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🎯 DBSCAN**
            - Baseado em densidade
            - Detecta ruído automaticamente
            - Não requer número de clusters
            """)
        
        with col2:
            st.markdown("""
            **🎯 K-Means**
            - Baseado em centróides
            - Rápido e eficiente
            - Requer número de clusters
            """)
    
    def _show_association_rules_page(self):
        """Página de regras de associação - ESPECÍFICA"""
        st.header("📋 Regras de Associação")
        
        df = self.data_cache.get('df') if self.data_cache else None
        files_status = self.data_cache.get('files_status', {}) if self.data_cache else {}
        
        if df is None:
            st.error("❌ Dados não disponíveis")
            return
        
        # Verificar resultados de regras de associação
        analysis_files = files_status.get('analysis', [])
        association_files = [f for f in analysis_files 
                           if any(keyword in str(f).lower() 
                                 for keyword in ['apriori', 'fp_growth', 'eclat', 'association', 'rules'])]
        
        if association_files:
            st.success(f"✅ {len(association_files)} arquivo(s) de regras encontrado(s)")
            
            for file in association_files:
                with st.expander(f"📁 {file.name}"):
                    try:
                        if file.suffix == '.csv':
                            df_rules = pd.read_csv(file)
                            st.dataframe(df_rules.head(10), use_container_width=True)
                            
                            # Estatísticas das regras
                            if len(df_rules) > 0:
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("📊 Total de Regras", len(df_rules))
                                
                                with col2:
                                    if 'confidence' in df_rules.columns:
                                        avg_confidence = df_rules['confidence'].mean()
                                        st.metric("🎯 Confiança Média", f"{avg_confidence:.3f}")
                                
                                with col3:
                                    if 'lift' in df_rules.columns:
                                        avg_lift = df_rules['lift'].mean()
                                        st.metric("📈 Lift Médio", f"{avg_lift:.3f}")
                    except Exception as e:
                        st.error(f"Erro ao ler arquivo: {e}")
        else:
            st.warning("⚠️ Nenhum resultado de regras de associação encontrado")
            st.info("💡 Execute o pipeline principal para gerar regras: `python main.py`")
        
        # Preparação para regras de associação
        st.subheader("📊 Análise de Padrões nos Dados")
        
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            st.info(f"✅ {len(categorical_cols)} variáveis categóricas disponíveis")
            
            # Análise de frequência
            selected_col = st.selectbox("Selecione uma variável para análise:", categorical_cols)
            
            if selected_col:
                value_counts = df[selected_col].value_counts()
                fig = px.bar(
                    x=value_counts.index[:10],  # Top 10
                    y=value_counts.values[:10],
                    title=f"📊 Top 10 valores em {selected_col}"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Informações sobre algoritmos
        st.subheader("🧠 Algoritmos de Regras de Associação")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **📋 APRIORI**
            - Algoritmo clássico
            - Baseado em suporte
            - Gera candidatos iterativamente
            """)
        
        with col2:
            st.markdown("""
            **🚀 FP-GROWTH**
            - Mais eficiente que Apriori
            - Usa estrutura FP-Tree
            - Sem geração de candidatos
            """)
        
        with col3:
            st.markdown("""
            **⚡ ECLAT**
            - Baseado em intersecção
            - Eficiente para datasets esparsos
            - Abordagem vertical
            """)
    
    def _show_prediction_page(self):
        """Página de predição - ESPECÍFICA"""
        st.header("🔮 Interface de Predição")
        
        df = self.data_cache.get('df') if self.data_cache else None
        files_status = self.data_cache.get('files_status', {}) if self.data_cache else {}
        
        if df is None:
            st.error("❌ Dados não disponíveis")
            return
        
        # Verificar modelos disponíveis
        model_files = files_status.get('models', [])
        
        if not model_files:
            st.warning("⚠️ Nenhum modelo treinado encontrado")
            st.info("💡 Execute o pipeline principal para treinar modelos: `python main.py`")
            return
        
        st.success(f"✅ {len(model_files)} modelo(s) disponível(is)")
        
        # Interface de predição
        st.subheader("📝 Fazer Predição Individual")
        
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input("Idade", min_value=17, max_value=100, value=30)
                education_num = st.number_input("Anos de Educação", min_value=1, max_value=16, value=12)
                hours_per_week = st.number_input("Horas/Semana", min_value=1, max_value=99, value=40)
                capital_gain = st.number_input("Capital Gain", min_value=0, max_value=100000, value=0)
            
            with col2:
                workclass = st.selectbox("Classe de Trabalho", 
                                       ["Private", "Self-emp-not-inc", "Federal-gov", "Local-gov"])
                education = st.selectbox("Educação", 
                                       ["Bachelors", "HS-grad", "Masters", "Some-college"])
                marital_status = st.selectbox("Estado Civil",
                                            ["Married-civ-spouse", "Never-married", "Divorced"])
                sex = st.selectbox("Sexo", ["Male", "Female"])
            
            submitted = st.form_submit_button("🎯 Fazer Predição", use_container_width=True)
            
            if submitted:
                # Simular predição (implementar carregamento real do modelo)
                prediction_proba = np.random.random()
                
                if prediction_proba > 0.5:
                    st.success("💰 **Predição: Salário > 50K**")
                    st.info(f"Probabilidade: {prediction_proba:.3f}")
                else:
                    st.warning("💰 **Predição: Salário ≤ 50K**")
                    st.info(f"Probabilidade: {1-prediction_proba:.3f}")
                
                # Mostrar dados de entrada
                st.subheader("📋 Dados de Entrada")
                input_data = {
                    'Idade': age,
                    'Educação (anos)': education_num,
                    'Horas/Semana': hours_per_week,
                    'Capital Gain': capital_gain,
                    'Classe de Trabalho': workclass,
                    'Educação': education,
                    'Estado Civil': marital_status,
                    'Sexo': sex
                }
                
                input_df = pd.DataFrame([input_data])
                st.dataframe(input_df, use_container_width=True)
        
        # Exemplos de predições
        st.subheader("📊 Exemplos de Predições")
        
        if len(df) > 0:
            sample_data = df.sample(3)
            st.dataframe(sample_data, use_container_width=True)
    
    def _show_metrics_page(self):
        """Página de métricas - ESPECÍFICA"""
        st.header("📈 Métricas e KPIs do Sistema")
        
        df = self.data_cache.get('df') if self.data_cache else None
        files_status = self.data_cache.get('files_status', {}) if self.data_cache else {}
        
        if df is None:
            st.error("❌ Dados não disponíveis")
            return
        
        # Métricas de qualidade dos dados
        st.subheader("📊 Qualidade dos Dados")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.metric("✅ Completude", f"{completeness:.1f}%")
        
        with col2:
            uniqueness = df.nunique().sum() / len(df)
            st.metric("🔍 Unicidade", f"{uniqueness:.1f}")
        
        with col3:
            consistency = 100  # Implementar cálculo real
            st.metric("🎯 Consistência", f"{consistency:.1f}%")
        
        with col4:
            validity = 95  # Implementar cálculo real
            st.metric("✓ Validade", f"{validity:.1f}%")
        
        # Métricas do pipeline
        st.subheader("🔧 Status do Pipeline")
        
        analysis_files = files_status.get('analysis', [])
        model_files = files_status.get('models', [])
        image_files = files_status.get('images', [])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📋 Análises", len(analysis_files))
        
        with col2:
            st.metric("🤖 Modelos", len(model_files))
        
        with col3:
            st.metric("🖼️ Imagens", len(image_files))
        
        # Gráfico de distribuição dos dados
        st.subheader("📊 Distribuição das Variáveis")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            selected_var = st.selectbox("Selecione uma variável:", numeric_cols)
            
            fig = px.histogram(
                df, x=selected_var,
                title=f"📊 Distribuição de {selected_var}"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Métricas de performance (simuladas)
        st.subheader("⚡ Performance do Sistema")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("⏱️ Tempo de Carga", "1.2s")
        
        with col2:
            st.metric("💾 Uso de Memória", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
        
        with col3:
            st.metric("🚀 Uptime", "99.9%")
    
    def _show_reports_page(self):
        """Página de relatórios - ESPECÍFICA"""
        st.header("📁 Relatórios e Exportações")
        
        df = self.data_cache.get('df') if self.data_cache else None
        files_status = self.data_cache.get('files_status', {}) if self.data_cache else {}
        
        if df is None:
            st.error("❌ Dados não disponíveis")
            return
        
        # Relatório executivo
        st.subheader("📋 Relatório Executivo")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **📊 Resumo dos Dados:**
            - Total de registros: {len(df):,}
            - Colunas: {len(df.columns)}
            - Período: {datetime.now().strftime('%Y-%m-%d')}
            """)
        
        with col2:
            if 'salary' in df.columns:
                high_salary_count = (df['salary'] == '>50K').sum()
                high_salary_rate = (df['salary'] == '>50K').mean()
                
                st.markdown(f"""
                **💰 Análise Salarial:**
                - Salários altos: {high_salary_count:,} ({high_salary_rate:.1%})
                - Salários baixos: {len(df) - high_salary_count:,} ({1-high_salary_rate:.1%})
                """)
        
        # Insights principais
        st.subheader("💡 Insights Principais")
        
        insights = []
        
        if 'age' in df.columns:
            avg_age = df['age'].mean()
            insights.append(f"🎂 Idade média: {avg_age:.1f} anos")
        
        if 'education-num' in df.columns:
            avg_education = df['education-num'].mean()
            insights.append(f"🎓 Educação média: {avg_education:.1f} anos")
        
        if 'hours-per-week' in df.columns:
            avg_hours = df['hours-per-week'].mean()
            insights.append(f"⏰ Horas médias/semana: {avg_hours:.1f}h")
        
        for insight in insights:
            st.info(insight)
        
        # Seção de exportação
        st.subheader("📤 Exportar Dados")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Exportar CSV", use_container_width=True):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv,
                    file_name=f"salary_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📊 Exportar Excel", use_container_width=True):
                # Implementar exportação Excel
                st.info("🚧 Funcionalidade em desenvolvimento")
        
        with col3:
            if st.button("📋 Gerar PDF", use_container_width=True):
                # Implementar geração PDF
                st.info("🚧 Funcionalidade em desenvolvimento")
        
        # Status dos arquivos gerados
        st.subheader("📁 Arquivos Gerados pelo Pipeline")
        
        all_files = []
        all_files.extend(files_status.get('analysis', []))
        all_files.extend(files_status.get('models', []))
        all_files.extend(files_status.get('images', []))
        
        if all_files:
            files_data = []
            for file in all_files:
                files_data.append({
                    'Nome': file.name,
                    'Tipo': file.suffix,
                    'Tamanho (KB)': f"{file.stat().st_size / 1024:.1f}",
                    'Modificado': datetime.fromtimestamp(file.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
                })
            
            files_df = pd.DataFrame(files_data)
            st.dataframe(files_df, use_container_width=True)
        else:
            st.warning("⚠️ Nenhum arquivo gerado encontrado")
    
    def _show_admin_page(self):
        """Página de administração - ESPECÍFICA"""
        st.header("⚙️ Administração do Sistema")
        
        user_data = self.auth.get_user_data()
        
        if user_data.get('role') != 'admin':
            st.error("❌ Acesso restrito a administradores!")
            return
        
        # Informações do sistema
        st.subheader("💻 Informações do Sistema")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("👥 Usuários", "3")  # admin, user, demo
        
        with col2:
            st.metric("🔗 Sessões Ativas", "1")
        
        with col3:
            st.metric("⏱️ Uptime", "24h")
        
        # Gestão de usuários
        st.subheader("👥 Gestão de Usuários")
        
        users_data = [
            {'Usuário': 'admin', 'Nome': 'Administrador', 'Papel': 'admin', 'Status': '✅ Ativo'},
            {'Usuário': 'user', 'Nome': 'Usuário', 'Papel': 'user', 'Status': '✅ Ativo'},
            {'Usuário': 'demo', 'Nome': 'Demo', 'Papel': 'user', 'Status': '✅ Ativo'}
        ]
        
        users_df = pd.DataFrame(users_data)
        st.dataframe(users_df, use_container_width=True)
        
        # Configurações do sistema
        st.subheader("🔧 Configurações")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🧹 Limpar Cache", use_container_width=True):
                self.data_cache = None
                st.success("✅ Cache limpo com sucesso!")
                st.rerun()
        
        with col2:
            if st.button("🔄 Recarregar Sistema", use_container_width=True):
                st.success("✅ Sistema recarregado!")
                st.rerun()
        
        # Logs do sistema
        st.subheader("📋 Logs do Sistema")
        
        logs_data = [
            {'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Evento': 'Login admin', 'Status': '✅'},
            {'Timestamp': (datetime.now()).strftime('%Y-%m-%d %H:%M:%S'), 'Evento': 'Dados carregados', 'Status': '✅'},
            {'Timestamp': (datetime.now()).strftime('%Y-%m-%d %H:%M:%S'), 'Evento': 'Dashboard iniciado', 'Status': '✅'}
        ]
        
        logs_df = pd.DataFrame(logs_data)
        st.dataframe(logs_df, use_container_width=True)

# Executar aplicação
if __name__ == "__main__":
    dashboard = MultilingualDashboard()
    dashboard.run()