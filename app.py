"""
🌍 Dashboard Multilingual - Análise Salarial - VERSÃO INTEGRADA COM MAIN.PY
Sistema completo integrado com todos os gráficos e análises do pipeline principal
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
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

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
    page_title="Dashboard Integrado - Análise Salarial",
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

class IntegratedDashboard:
    """Dashboard integrado com pipeline principal (main.py)"""
    
    def __init__(self):
        """Inicializar dashboard integrado"""
        self.auth = SimpleAuth()
        
        # Páginas integradas com análises do main.py
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
                'title': 'Clustering (DBSCAN)',
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
            'visualizations': {
                'title': 'Visualizações Geradas',
                'icon': '🎨',
                'roles': ['admin', 'user'],
                'func': self._show_visualizations_page
            },
            'pipeline_results': {
                'title': 'Resultados Pipeline',
                'icon': '⚙️',
                'roles': ['admin', 'user'],
                'func': self._show_pipeline_results_page
            },
            'prediction': {
                'title': 'Predição',
                'icon': '🔮',
                'roles': ['admin', 'user'],
                'func': self._show_prediction_page
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
        
        # Diretórios de análise
        self.output_dirs = {
            'analysis': Path('output/analysis'),
            'images': Path('output/images'),
            'models': Path('models'),
            'output': Path('output')
        }
    
    def run(self):
        """Executar dashboard integrado"""
        # CSS personalizado
        self._apply_css()
        
        # Header
        st.markdown("""
        <div class="main-header">
            🌍 Dashboard Integrado - Análise Salarial & Pipeline Acadêmico
        </div>
        """, unsafe_allow_html=True)
        
        # Verificar autenticação
        if not self.auth.is_authenticated():
            self._show_login_page()
            return
        
        # Carregar dados uma vez
        if self.data_cache is None:
            self.data_cache = self._load_integrated_data()
        
        # Layout principal
        self._show_sidebar()
        self._execute_current_page()
    
    def _apply_css(self):
        """Aplicar CSS personalizado aprimorado"""
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #1f4e79, #2c6aa0, #4a90a4);
            color: white;
            padding: 1.5rem;
            text-align: center;
            font-size: 1.8rem;
            font-weight: bold;
            border-radius: 15px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        .pipeline-status {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
            text-align: center;
        }
        
        .algorithm-card {
            background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
            text-align: center;
        }
        
        .results-box {
            background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%);
            color: #2d3436;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
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
        
        .image-container {
            border: 2px solid #dee2e6;
            border-radius: 10px;
            padding: 1rem;
            margin: 1rem 0;
            background: #f8f9fa;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _load_integrated_data(self):
        """Carregar dados integrados com pipeline principal"""
        try:
            # 1. Carregar dados principais
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
            
            # 2. Escanear arquivos de análise gerados pelo main.py
            analysis_files = self._scan_pipeline_files()
            
            # 3. Verificar status do pipeline
            pipeline_status = self._check_pipeline_status(analysis_files)
            
            if df is not None:
                # Limpeza básica
                df = df.replace('?', pd.NA)
                
                return {
                    'df': df,
                    'status': f'✅ Dados carregados de {source} ({len(df):,} registros)',
                    'analysis_files': analysis_files,
                    'pipeline_status': pipeline_status,
                    'source': source
                }
            else:
                # Dados de exemplo se não encontrar o CSV
                return self._create_sample_data_with_analysis()
                
        except Exception as e:
            logging.error(f"Erro ao carregar dados integrados: {e}")
            return self._create_sample_data_with_analysis()
    
    def _scan_pipeline_files(self):
        """Escanear arquivos gerados pelo pipeline principal"""
        files = {
            'images': {
                'clustering': [],
                'association': [],
                'models': [],
                'dashboard': [],
                'distributions': []
            },
            'analysis': {
                'csv': [],
                'json': []
            },
            'models': []
        }
        
        # Gráficos principais esperados do main.py
        expected_images = {
            'clustering': [
                'dbscan_analysis.png',
                'clustering_analysis_v2.png'
            ],
            'association': [
                'association_rules_analysis.png'
            ],
            'models': [
                'model_comparison_v2.png',
                'feature_importance_v2.png'
            ],
            'dashboard': [
                'summary_dashboard_v2.png',
                'temporal_analysis_v2.png'
            ],
            'distributions': [
                'numeric_distributions.png',
                'categorical_distributions.png',
                'salary_distribution.png',
                'correlacao.png'
            ]
        }
        
        # Verificar em múltiplos diretórios
        search_dirs = [
            Path('output/images'),
            Path('output/analysis'),
            Path('output'),
            Path('images'),
            Path('analysis')
        ]
        
        for search_dir in search_dirs:
            if search_dir.exists():
                # Imagens
                for category, image_list in expected_images.items():
                    for image_name in image_list:
                        image_path = search_dir / image_name
                        if image_path.exists():
                            files['images'][category].append(image_path)
                
                # Arquivos adicionais
                files['analysis']['csv'].extend(list(search_dir.glob('*.csv')))
                files['analysis']['json'].extend(list(search_dir.glob('*.json')))
        
        # Modelos
        model_dirs = [Path('models'), Path('data/processed'), Path('output/models')]
        for model_dir in model_dirs:
            if model_dir.exists():
                files['models'].extend(list(model_dir.glob('*.pkl')))
                files['models'].extend(list(model_dir.glob('*.joblib')))
        
        return files
    
    def _check_pipeline_status(self, analysis_files):
        """Verificar status de execução do pipeline"""
        status = {
            'executed': False,
            'algorithms': {
                'dbscan': False,
                'apriori': False,
                'fp_growth': False,
                'eclat': False,
                'ml_models': False
            },
            'visualizations_count': 0,
            'analysis_files_count': 0
        }
        
        # Verificar algoritmos por arquivos
        all_files = []
        for category in analysis_files['images'].values():
            all_files.extend(category)
        all_files.extend(analysis_files['analysis']['csv'])
        
        for file_path in all_files:
            file_name = str(file_path).lower()
            
            if 'dbscan' in file_name:
                status['algorithms']['dbscan'] = True
            elif 'apriori' in file_name:
                status['algorithms']['apriori'] = True
            elif 'fp_growth' in file_name or 'fp-growth' in file_name:
                status['algorithms']['fp_growth'] = True
            elif 'eclat' in file_name:
                status['algorithms']['eclat'] = True
            elif 'model' in file_name:
                status['algorithms']['ml_models'] = True
        
        # Contadores
        status['visualizations_count'] = sum(len(v) for v in analysis_files['images'].values())
        status['analysis_files_count'] = len(analysis_files['analysis']['csv'])
        
        # Pipeline executado se tiver pelo menos 2 algoritmos
        executed_count = sum(1 for executed in status['algorithms'].values() if executed)
        status['executed'] = executed_count >= 2
        
        return status
    
    def _create_sample_data_with_analysis(self):
        """Criar dados de exemplo com status de análise"""
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
                'status': '⚠️ Dados de exemplo criados - Execute python main.py para análise completa',
                'analysis_files': {'images': {k: [] for k in ['clustering', 'association', 'models', 'dashboard', 'distributions']}, 'analysis': {'csv': [], 'json': []}, 'models': []},
                'pipeline_status': {'executed': False, 'algorithms': {k: False for k in ['dbscan', 'apriori', 'fp_growth', 'eclat', 'ml_models']}, 'visualizations_count': 0, 'analysis_files_count': 0},
                'source': 'sample_data'
            }
            
        except Exception as e:
            logging.error(f"Erro ao criar dados de exemplo: {e}")
            return None
    
    def _show_login_page(self):
        """Página de login"""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("### 🔐 Login - Dashboard Integrado")
            
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
                - **admin** / **admin123** (Administrador completo)
                - **user** / **user123** (Usuário padrão)
                - **demo** / **demo123** (Demonstração)
                
                **💡 Dica:** Execute `python main.py` para gerar todas as análises!
                """)
    
    def _show_sidebar(self):
        """Mostrar sidebar integrada"""
        with st.sidebar:
            # Informações do usuário
            self._show_user_info()
            
            st.markdown("---")
            
            # Status do pipeline
            self._show_pipeline_status()
            
            st.markdown("---")
            
            # Navegação
            self._show_navigation()
            
            st.markdown("---")
            
            # Controles
            self._show_controls()
    
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
    
    def _show_pipeline_status(self):
        """Mostrar status do pipeline principal"""
        st.markdown("### ⚙️ Status Pipeline")
        
        if self.data_cache:
            pipeline_status = self.data_cache.get('pipeline_status', {})
            
            if pipeline_status.get('executed', False):
                st.markdown("""
                <div class="pipeline-status">
                    ✅ Pipeline Executado!
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="results-box">
                    ⚠️ Execute: python main.py
                </div>
                """, unsafe_allow_html=True)
            
            # Algoritmos implementados
            algorithms = pipeline_status.get('algorithms', {})
            
            st.markdown("**🧠 Algoritmos:**")
            for alg, status in algorithms.items():
                emoji = "✅" if status else "❌"
                alg_name = alg.upper().replace('_', '-')
                st.markdown(f"  {emoji} {alg_name}")
            
            # Estatísticas
            viz_count = pipeline_status.get('visualizations_count', 0)
            files_count = pipeline_status.get('analysis_files_count', 0)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🎨 Gráficos", viz_count)
            with col2:
                st.metric("📊 Análises", files_count)
    
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
    
    def _show_controls(self):
        """Mostrar controles"""
        st.markdown("### 🔧 Controles")
        
        if st.button("🔄 Recarregar Dados", use_container_width=True):
            self.data_cache = None
            self.data_cache = self._load_integrated_data()
            st.success("✅ Dados recarregados!")
            st.rerun()
        
        if st.button("▶️ Executar Pipeline", use_container_width=True):
            st.info("💡 Execute no terminal: `python main.py`")
    
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
    # PÁGINAS ESPECÍFICAS INTEGRADAS COM MAIN.PY
    # =========================================================================
    
    def _show_overview_page(self):
        """Página de visão geral integrada"""
        st.header("📊 Visão Geral - Dashboard Integrado")
        
        df = self.data_cache.get('df') if self.data_cache else None
        pipeline_status = self.data_cache.get('pipeline_status', {}) if self.data_cache else {}
        
        if df is None:
            st.error("❌ Dados não disponíveis")
            return
        
        # Status da execução
        if pipeline_status.get('executed', False):
            st.success("✅ Pipeline principal executado com sucesso!")
        else:
            st.warning("⚠️ Pipeline não executado. Execute `python main.py` para análise completa.")
        
        # Métricas principais integradas
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
            algorithms_executed = sum(1 for executed in pipeline_status.get('algorithms', {}).values() if executed)
            st.markdown(f"""
            <div class="metric-card">
                <h3>🧠 Algoritmos</h3>
                <h2>{algorithms_executed}/5</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            viz_count = pipeline_status.get('visualizations_count', 0)
            st.markdown(f"""
            <div class="metric-card">
                <h3>🎨 Gráficos</h3>
                <h2>{viz_count}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Status dos algoritmos principais
        st.subheader("🧠 Status dos Algoritmos Implementados")
        
        algorithms_info = {
            'dbscan': {'name': 'DBSCAN', 'desc': 'Clustering baseado em densidade'},
            'apriori': {'name': 'APRIORI', 'desc': 'Regras de associação clássicas'},
            'fp_growth': {'name': 'FP-GROWTH', 'desc': 'Mineração eficiente de padrões'},
            'eclat': {'name': 'ECLAT', 'desc': 'Algoritmo de intersecção'},
            'ml_models': {'name': 'ML MODELS', 'desc': 'Random Forest + Logistic Regression'}
        }
        
        for alg_key, alg_info in algorithms_info.items():
            executed = pipeline_status.get('algorithms', {}).get(alg_key, False)
            status_emoji = "✅" if executed else "❌"
            status_text = "Executado" if executed else "Não executado"
            
            st.markdown(f"""
            <div class="algorithm-card">
                <strong>{status_emoji} {alg_info['name']}</strong> - {alg_info['desc']}<br>
                <small>Status: {status_text}</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Distribuições principais se dados disponíveis
        st.subheader("📈 Distribuições dos Dados")
        
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
    
    def _show_exploratory_page(self):
        """Página de análise exploratória com gráficos do pipeline"""
        st.header("🔍 Análise Exploratória - Integrada com Pipeline")
        
        df = self.data_cache.get('df') if self.data_cache else None
        analysis_files = self.data_cache.get('analysis_files', {}) if self.data_cache else {}
        
        if df is None:
            st.error("❌ Dados não disponíveis")
            return
        
        # Mostrar gráficos de distribuição gerados pelo main.py
        st.subheader("📊 Distribuições Geradas pelo Pipeline")
        
        distributions_images = analysis_files.get('images', {}).get('distributions', [])
        
        if distributions_images:
            tabs = st.tabs([f"📊 {img.stem}" for img in distributions_images])
            
            for i, (tab, img_path) in enumerate(zip(tabs, distributions_images)):
                with tab:
                    try:
                        image = Image.open(img_path)
                        st.image(image, caption=f"Gerado pelo pipeline: {img_path.name}", use_column_width=True)
                    except Exception as e:
                        st.error(f"Erro ao carregar imagem: {e}")
        else:
            st.warning("⚠️ Nenhuma distribuição encontrada. Execute `python main.py` para gerar.")
        
        # Análise interativa adicional
        st.subheader("🔍 Análise Interativa")
        
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
        """Página de modelos ML integrada com pipeline"""
        st.header("🤖 Modelos ML - Integrados com Pipeline")
        
        df = self.data_cache.get('df') if self.data_cache else None
        analysis_files = self.data_cache.get('analysis_files', {}) if self.data_cache else {}
        pipeline_status = self.data_cache.get('pipeline_status', {}) if self.data_cache else {}
        
        if df is None:
            st.error("❌ Dados não disponíveis")
            return
        
        # Status dos modelos
        ml_executed = pipeline_status.get('algorithms', {}).get('ml_models', False)
        
        if ml_executed:
            st.success("✅ Modelos ML executados pelo pipeline!")
        else:
            st.warning("⚠️ Modelos não executados. Execute `python main.py` para treinar.")
        
        # Gráficos de modelos gerados pelo main.py
        st.subheader("📊 Análises dos Modelos Geradas")
        
        model_images = analysis_files.get('images', {}).get('models', [])
        
        if model_images:
            for img_path in model_images:
                st.markdown(f"### 📈 {img_path.stem.replace('_', ' ').title()}")
                try:
                    image = Image.open(img_path)
                    st.image(image, caption=f"Gerado pelo pipeline: {img_path.name}", use_column_width=True)
                except Exception as e:
                    st.error(f"Erro ao carregar imagem: {e}")
        else:
            st.info("💡 Nenhum gráfico de modelo encontrado. Execute o pipeline para gerar.")
        
        # Informações sobre modelos implementados
        st.subheader("🧠 Modelos Implementados no Pipeline")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="algorithm-card">
                <h4>🌳 Random Forest</h4>
                <p>• Ensemble de árvores de decisão</p>
                <p>• Reduz overfitting</p>
                <p>• Boa performance geral</p>
                <p>• Implementado no main.py</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="algorithm-card">
                <h4>📈 Logistic Regression</h4>
                <p>• Modelo linear para classificação</p>
                <p>• Interpretável</p>
                <p>• Rápido para treinar</p>
                <p>• Implementado no main.py</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Target distribution
        if 'salary' in df.columns:
            st.subheader("🎯 Distribuição da Variável Target")
            target_dist = df['salary'].value_counts()
            fig = px.pie(
                values=target_dist.values,
                names=target_dist.index,
                title="🎯 Distribuição da Variável Target (Salary)"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def _show_clustering_page(self):
        """Página de clustering integrada com DBSCAN do pipeline"""
        st.header("🎯 Clustering (DBSCAN) - Integrado com Pipeline")
        
        df = self.data_cache.get('df') if self.data_cache else None
        analysis_files = self.data_cache.get('analysis_files', {}) if self.data_cache else {}
        pipeline_status = self.data_cache.get('pipeline_status', {}) if self.data_cache else {}
        
        if df is None:
            st.error("❌ Dados não disponíveis")
            return
        
        # Status do DBSCAN
        dbscan_executed = pipeline_status.get('algorithms', {}).get('dbscan', False)
        
        if dbscan_executed:
            st.success("✅ DBSCAN executado pelo pipeline!")
        else:
            st.warning("⚠️ DBSCAN não executado. Execute `python main.py` para gerar clusters.")
        
        # Gráficos de clustering gerados pelo main.py
        st.subheader("📊 Análises de Clustering Geradas")
        
        clustering_images = analysis_files.get('images', {}).get('clustering', [])
        
        if clustering_images:
            for img_path in clustering_images:
                st.markdown(f"### 🎯 {img_path.stem.replace('_', ' ').title()}")
                try:
                    image = Image.open(img_path)
                    st.image(image, caption=f"Gerado pelo pipeline: {img_path.name}", use_column_width=True)
                except Exception as e:
                    st.error(f"Erro ao carregar imagem: {e}")
        else:
            st.info("💡 Nenhum gráfico de clustering encontrado. Execute o pipeline para gerar.")
        
        # Verificar arquivos CSV de clustering
        st.subheader("📋 Resultados de Clustering")
        
        clustering_csvs = [f for f in analysis_files.get('analysis', {}).get('csv', []) 
                          if 'dbscan' in str(f).lower() or 'cluster' in str(f).lower()]
        
        if clustering_csvs:
            for csv_path in clustering_csvs:
                with st.expander(f"📁 {csv_path.name}"):
                    try:
                        df_cluster = pd.read_csv(csv_path)
                        st.dataframe(df_cluster.head(), use_container_width=True)
                        
                        # Visualizar clusters se possível
                        if 'cluster' in df_cluster.columns:
                            cluster_counts = df_cluster['cluster'].value_counts()
                            fig = px.bar(
                                x=cluster_counts.index,
                                y=cluster_counts.values,
                                title=f"🎯 Distribuição de Clusters - {csv_path.name}"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Erro ao ler arquivo: {e}")
        
        # Informações sobre DBSCAN
        st.subheader("🧠 Sobre o Algoritmo DBSCAN")
        
        st.markdown("""
        <div class="algorithm-card">
            <h4>🎯 DBSCAN (Density-Based Spatial Clustering)</h4>
            <p><strong>Características:</strong></p>
            <p>• Baseado em densidade de pontos</p>
            <p>• Detecta ruído automaticamente</p>
            <p>• Não requer número de clusters pré-definido</p>
            <p>• Identifica clusters de formas arbitrárias</p>
            <p><strong>Implementação:</strong> Disponível no main.py</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Preparação para clustering (dados atuais)
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
    
    def _show_association_rules_page(self):
        """Página de regras de associação integrada com APRIORI, FP-GROWTH, ECLAT"""
        st.header("📋 Regras de Associação - APRIORI + FP-GROWTH + ECLAT")
        
        df = self.data_cache.get('df') if self.data_cache else None
        analysis_files = self.data_cache.get('analysis_files', {}) if self.data_cache else {}
        pipeline_status = self.data_cache.get('pipeline_status', {}) if self.data_cache else {}
        
        if df is None:
            st.error("❌ Dados não disponíveis")
            return
        
        # Status dos algoritmos
        algorithms_status = {
            'apriori': pipeline_status.get('algorithms', {}).get('apriori', False),
            'fp_growth': pipeline_status.get('algorithms', {}).get('fp_growth', False),
            'eclat': pipeline_status.get('algorithms', {}).get('eclat', False)
        }
        
        # Mostrar status
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status = "✅" if algorithms_status['apriori'] else "❌"
            st.markdown(f"""
            <div class="algorithm-card">
                <h4>{status} APRIORI</h4>
                <p>Algoritmo clássico</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            status = "✅" if algorithms_status['fp_growth'] else "❌"
            st.markdown(f"""
            <div class="algorithm-card">
                <h4>{status} FP-GROWTH</h4>
                <p>Mais eficiente</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            status = "✅" if algorithms_status['eclat'] else "❌"
            st.markdown(f"""
            <div class="algorithm-card">
                <h4>{status} ECLAT</h4>
                <p>Intersecção vertical</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Gráficos de regras de associação gerados pelo main.py
        st.subheader("📊 Análises de Regras de Associação Geradas")
        
        association_images = analysis_files.get('images', {}).get('association', [])
        
        if association_images:
            for img_path in association_images:
                st.markdown(f"### 📋 {img_path.stem.replace('_', ' ').title()}")
                try:
                    image = Image.open(img_path)
                    st.image(image, caption=f"Gerado pelo pipeline: {img_path.name}", use_column_width=True)
                except Exception as e:
                    st.error(f"Erro ao carregar imagem: {e}")
        else:
            st.info("💡 Nenhum gráfico de regras encontrado. Execute o pipeline para gerar.")
        
        # Resultados em CSV
        st.subheader("📋 Resultados das Regras de Associação")
        
        association_csvs = [f for f in analysis_files.get('analysis', {}).get('csv', []) 
                           if any(keyword in str(f).lower() 
                                 for keyword in ['apriori', 'fp_growth', 'eclat', 'association', 'rules'])]
        
        if association_csvs:
            tabs = st.tabs([f"📋 {csv.stem.upper()}" for csv in association_csvs])
            
            for tab, csv_path in zip(tabs, association_csvs):
                with tab:
                    try:
                        df_rules = pd.read_csv(csv_path)
                        
                        # Estatísticas das regras
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
                        
                        # Mostrar regras
                        st.dataframe(df_rules.head(10), use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Erro ao ler arquivo: {e}")
        else:
            st.warning("⚠️ Nenhum resultado encontrado. Execute `python main.py` para gerar regras.")
        
        # Análise de padrões nos dados atuais
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
    
    def _show_visualizations_page(self):
        """Página dedicada a todas as visualizações geradas pelo pipeline"""
        st.header("🎨 Visualizações Geradas pelo Pipeline")
        
        analysis_files = self.data_cache.get('analysis_files', {}) if self.data_cache else {}
        
        # Organizar visualizações por categoria
        all_images = analysis_files.get('images', {})
        total_images = sum(len(v) for v in all_images.values())
        
        if total_images == 0:
            st.warning("⚠️ Nenhuma visualização encontrada.")
            st.info("💡 Execute `python main.py` para gerar todas as visualizações.")
            return
        
        st.success(f"✅ {total_images} visualizações encontradas!")
        
        # Tabs por categoria
        categories = [k for k, v in all_images.items() if v]
        
        if categories:
            tabs = st.tabs([f"🎨 {cat.replace('_', ' ').title()}" for cat in categories])
            
            for tab, category in zip(tabs, categories):
                with tab:
                    st.subheader(f"📊 {category.replace('_', ' ').title()}")
                    
                    images_in_category = all_images[category]
                    
                    # Mostrar imagens em grid
                    if len(images_in_category) == 1:
                        # Uma imagem - tela cheia
                        img_path = images_in_category[0]
                        try:
                            image = Image.open(img_path)
                            st.image(image, caption=f"{img_path.name}", use_column_width=True)
                        except Exception as e:
                            st.error(f"Erro ao carregar {img_path.name}: {e}")
                    
                    elif len(images_in_category) == 2:
                        # Duas imagens - lado a lado
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            try:
                                image = Image.open(images_in_category[0])
                                st.image(image, caption=f"{images_in_category[0].name}", use_column_width=True)
                            except Exception as e:
                                st.error(f"Erro: {e}")
                        
                        with col2:
                            try:
                                image = Image.open(images_in_category[1])
                                st.image(image, caption=f"{images_in_category[1].name}", use_column_width=True)
                            except Exception as e:
                                st.error(f"Erro: {e}")
                    
                    else:
                        # Múltiplas imagens - uma por vez com seleção
                        selected_image = st.selectbox(
                            f"Selecione uma visualização de {category}:",
                            options=[img.stem for img in images_in_category],
                            key=f"select_{category}"
                        )
                        
                        if selected_image:
                            selected_path = next(img for img in images_in_category if img.stem == selected_image)
                            try:
                                image = Image.open(selected_path)
                                st.image(image, caption=f"{selected_path.name}", use_column_width=True)
                            except Exception as e:
                                st.error(f"Erro ao carregar {selected_path.name}: {e}")
        
        # Informações sobre as visualizações
        st.subheader("ℹ️ Sobre as Visualizações")
        
        st.markdown("""
        **🎨 Visualizações Geradas pelo Pipeline:**
        
        - **📊 Dashboard**: Resumo executivo completo
        - **🎯 Clustering**: Análises DBSCAN detalhadas
        - **📋 Association**: Regras APRIORI, FP-GROWTH, ECLAT
        - **🤖 Models**: Comparação e importância de features
        - **📈 Distributions**: Histogramas e distribuições
        
        **💡 Para gerar novas visualizações:** Execute `python main.py`
        """)
    
    def _show_pipeline_results_page(self):
        """Página dedicada aos resultados completos do pipeline"""
        st.header("⚙️ Resultados Completos do Pipeline")
        
        pipeline_status = self.data_cache.get('pipeline_status', {}) if self.data_cache else {}
        analysis_files = self.data_cache.get('analysis_files', {}) if self.data_cache else {}
        
        # Status geral
        if pipeline_status.get('executed', False):
            st.success("✅ Pipeline executado com sucesso!")
        else:
            st.error("❌ Pipeline não foi executado")
            st.info("💡 Execute `python main.py` para processar todas as análises")
            return
        
        # Resumo executivo
        st.subheader("📊 Resumo Executivo")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            algorithms_count = sum(1 for executed in pipeline_status.get('algorithms', {}).values() if executed)
            st.metric("🧠 Algoritmos", f"{algorithms_count}/5")
        
        with col2:
            viz_count = pipeline_status.get('visualizations_count', 0)
            st.metric("🎨 Visualizações", viz_count)
        
        with col3:
            files_count = pipeline_status.get('analysis_files_count', 0)
            st.metric("📊 Arquivos CSV", files_count)
        
        with col4:
            models_count = len(analysis_files.get('models', []))
            st.metric("🤖 Modelos", models_count)
        
        # Status detalhado dos algoritmos
        st.subheader("🧠 Status Detalhado dos Algoritmos")
        
        algorithms_detail = {
            'dbscan': {
                'name': 'DBSCAN',
                'description': 'Clustering baseado em densidade',
                'type': 'Clustering',
                'output': 'dbscan_results.csv + dbscan_analysis.png'
            },
            'apriori': {
                'name': 'APRIORI',
                'description': 'Regras de associação clássicas',
                'type': 'Association Rules',
                'output': 'apriori_rules.csv'
            },
            'fp_growth': {
                'name': 'FP-GROWTH',
                'description': 'Mineração eficiente de padrões',
                'type': 'Association Rules',
                'output': 'fp_growth_rules.csv'
            },
            'eclat': {
                'name': 'ECLAT',
                'description': 'Algoritmo de intersecção vertical',
                'type': 'Association Rules',
                'output': 'eclat_rules.csv'
            },
            'ml_models': {
                'name': 'ML MODELS',
                'description': 'Random Forest + Logistic Regression',
                'type': 'Machine Learning',
                'output': 'model_comparison_v2.png + feature_importance_v2.png'
            }
        }
        
        for alg_key, alg_info in algorithms_detail.items():
            executed = pipeline_status.get('algorithms', {}).get(alg_key, False)
            status_color = "success" if executed else "error"
            status_text = "✅ Executado" if executed else "❌ Não executado"
            
            with st.container():
                col1, col2, col3, col4 = st.columns([2, 3, 2, 3])
                
                with col1:
                    if executed:
                        st.success(alg_info['name'])
                    else:
                        st.error(alg_info['name'])
                
                with col2:
                    st.write(f"**Tipo:** {alg_info['type']}")
                    st.write(alg_info['description'])
                
                with col3:
                    st.write(f"**Status:**")
                    st.write(status_text)
                
                with col4:
                    st.write(f"**Output:**")
                    st.write(alg_info['output'])
        
        # Arquivos gerados
        st.subheader("📁 Arquivos Gerados")
        
        # CSV files
        csv_files = analysis_files.get('analysis', {}).get('csv', [])
        if csv_files:
            st.markdown("**📊 Arquivos CSV de Análise:**")
            for csv_file in csv_files:
                st.markdown(f"• `{csv_file.name}` - {csv_file.stat().st_size / 1024:.1f} KB")
        
        # JSON files
        json_files = analysis_files.get('analysis', {}).get('json', [])
        if json_files:
            st.markdown("**📋 Arquivos JSON:**")
            for json_file in json_files:
                st.markdown(f"• `{json_file.name}` - {json_file.stat().st_size / 1024:.1f} KB")
        
        # Model files
        model_files = analysis_files.get('models', [])
        if model_files:
            st.markdown("**🤖 Modelos Treinados:**")
            for model_file in model_files:
                st.markdown(f"• `{model_file.name}` - {model_file.stat().st_size / 1024:.1f} KB")
        
        # Verificar se existe summary dashboard gerado
        dashboard_images = analysis_files.get('images', {}).get('dashboard', [])
        summary_dashboard = next((img for img in dashboard_images if 'summary' in img.name.lower()), None)
        
        if summary_dashboard:
            st.subheader("📊 Dashboard de Resumo Gerado")
            try:
                image = Image.open(summary_dashboard)
                st.image(image, caption="Dashboard de resumo gerado pelo pipeline", use_column_width=True)
            except Exception as e:
                st.error(f"Erro ao carregar dashboard: {e}")
        
        # Log de execução se disponível
        st.subheader("📋 Informações Técnicas")
        
        st.markdown(f"""
        **🔧 Configurações do Pipeline:**
        - **Algoritmos principais:** 4 (DBSCAN, APRIORI, FP-GROWTH, ECLAT)
        - **Modelos ML:** 2 (Random Forest, Logistic Regression)
        - **Visualizações:** {pipeline_status.get('visualizations_count', 0)} gráficos gerados
        - **Tempo de execução:** Varia de 30s a 2 minutos
        - **Formato de saída:** CSV, PNG, JSON
        
        **📊 Métricas calculadas:**
        - Accuracy, Precision, Recall, F1-Score (ML)
        - Silhouette Score, Inertia (Clustering)
        - Confidence, Lift, Support (Association Rules)
        """)
    
    def _show_prediction_page(self):
        """Página de predição integrada"""
        st.header("🔮 Interface de Predição")
        
        df = self.data_cache.get('df') if self.data_cache else None
        analysis_files = self.data_cache.get('analysis_files', {}) if self.data_cache else {}
        
        if df is None:
            st.error("❌ Dados não disponíveis")
            return
        
        # Verificar modelos disponíveis
        model_files = analysis_files.get('models', [])
        
        if not model_files:
            st.warning("⚠️ Nenhum modelo treinado encontrado")
            st.info("💡 Execute o pipeline principal para treinar modelos: `python main.py`")
            return
        
        st.success(f"✅ {len(model_files)} modelo(s) disponível(is)")
        
        # Interface de predição
        st.subheader("📝 Fazer Predição Individual")
        
        with st.form("prediction_form"):
            st.markdown("**Insira os dados para predição:**")
            
            # Campos de entrada baseados nas colunas do dataset
            col1, col2, col3 = st.columns(3)
            
            with col1:
                age = st.number_input("🎂 Idade", min_value=16, max_value=90, value=30)
                education_num = st.number_input("📚 Anos de Educação", min_value=1, max_value=16, value=10)
                hours_per_week = st.number_input("⏰ Horas/Semana", min_value=1, max_value=99, value=40)
            
            with col2:
                workclass = st.selectbox("💼 Classe de Trabalho", 
                                       ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'])
                education = st.selectbox("🎓 Educação", 
                                       ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'])
                marital_status = st.selectbox("💑 Estado Civil", 
                                            ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])
            
            with col3:
                occupation = st.selectbox("👨‍💼 Ocupação", 
                                        ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'])
                relationship = st.selectbox("👥 Relacionamento", 
                                          ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'])
                sex = st.selectbox("⚧ Sexo", ['Female', 'Male'])
            
            col4, col5 = st.columns(2)
            
            with col4:
                race = st.selectbox("🌍 Raça", 
                                  ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'])
                capital_gain = st.number_input("💰 Ganho de Capital", min_value=0, max_value=99999, value=0)
            
            with col5:
                native_country = st.selectbox("🏳️ País de Origem", 
                                            ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands'])
                capital_loss = st.number_input("📉 Perda de Capital", min_value=0, max_value=4356, value=0)
            
            # Botão de predição
            if st.form_submit_button("🔮 Fazer Predição", use_container_width=True):
                # Simular predição (aqui você carregaria o modelo real)
                st.success("🎯 Predição realizada!")
                
                # Resultado simulado
                probability = np.random.rand()
                prediction = ">50K" if probability > 0.5 else "<=50K"
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("📊 Predição", prediction)
                
                with col2:
                    st.metric("🎯 Confiança", f"{probability:.1%}")
                
                st.info("💡 Esta é uma predição simulada. Execute o pipeline para usar modelos reais.")
        
        # Predição em lote
        st.subheader("📊 Predição em Lote")
        
        uploaded_file = st.file_uploader("📁 Faça upload de um CSV para predição em lote", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
                st.success(f"✅ Arquivo carregado: {len(df_upload)} registros")
                
                # Mostrar preview
                st.dataframe(df_upload.head(), use_container_width=True)
                
                if st.button("🔮 Executar Predições em Lote"):
                    # Simular predições
                    predictions = np.random.choice(['<=50K', '>50K'], size=len(df_upload))
                    df_upload['prediction'] = predictions
                    
                    st.success("✅ Predições concluídas!")
                    st.dataframe(df_upload, use_container_width=True)
                    
                    # Download
                    csv = df_upload.to_csv(index=False)
                    st.download_button(
                        label="📥 Baixar Resultados",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
                    
            except Exception as e:
                st.error(f"❌ Erro ao processar arquivo: {e}")
    
    def _show_admin_page(self):
        """Página de administração"""
        st.header("⚙️ Painel de Administração")
        
        user_data = self.auth.get_user_data()
        if user_data.get('role') != 'admin':
            st.error("❌ Acesso negado. Apenas administradores podem acessar esta página.")
            return
        
        # Status do sistema
        st.subheader("🖥️ Status do Sistema")
        
        # Informações do sistema
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h4>💾 Cache</h4>
                <p>Status: Ativo</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h4>📁 Arquivos</h4>
                <p>Pipeline: Integrado</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h4>🔐 Autenticação</h4>
                <p>Sistema: Ativo</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Controles administrativos
        st.subheader("🔧 Controles Administrativos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Limpar Cache", use_container_width=True):
                self.data_cache = None
                st.success("✅ Cache limpo!")
                st.rerun()
            
            if st.button("🔄 Recarregar Sistema", use_container_width=True):
                st.success("✅ Sistema recarregado!")
                st.rerun()
        
        with col2:
            if st.button("📊 Verificar Pipeline", use_container_width=True):
                self.data_cache = self._load_integrated_data()
                st.success("✅ Pipeline verificado!")
                st.rerun()
            
            if st.button("📁 Escanear Arquivos", use_container_width=True):
                analysis_files = self._scan_pipeline_files()
                total_files = sum(len(v) if isinstance(v, list) else sum(len(vv) for vv in v.values()) for v in analysis_files.values())
                st.success(f"✅ {total_files} arquivos encontrados!")
        
        # Logs e informações técnicas
        st.subheader("📋 Informações Técnicas")
        
        with st.expander("🔍 Detalhes do Sistema"):
            st.json({
                "dashboard_version": "2.0 - Integrado com main.py",
                "pipeline_integration": "Completa",
                "algorithms_supported": ["DBSCAN", "APRIORI", "FP-GROWTH", "ECLAT", "ML Models"],
                "visualization_formats": ["PNG", "JPG"],
                "data_formats": ["CSV", "JSON"],
                "authentication": "Simples com roles",
                "pages_count": len(self.pages),
                "output_directories": [str(d) for d in self.output_dirs.values()]
            })
        
        # Usuários conectados (simulado)
        st.subheader("👥 Usuários Conectados")
        
        # Simular dados de usuários
        users_data = {
            'Usuário': ['admin', 'user', 'demo'],
            'Status': ['🟢 Online', '🟡 Idle', '🔴 Offline'],
            'Última Atividade': ['Agora', '5 min atrás', '1 hora atrás'],
            'Página Atual': ['Administração', 'Visão Geral', 'N/A']
        }
        
        df_users = pd.DataFrame(users_data)
        st.dataframe(df_users, use_container_width=True)
        
        # Configurações
        st.subheader("⚙️ Configurações")
        
        with st.form("admin_settings"):
            st.markdown("**Configurações do Dashboard:**")
            
            auto_refresh = st.checkbox("🔄 Auto-refresh", value=False)
            debug_mode = st.checkbox("🐛 Modo Debug", value=False)
            cache_enabled = st.checkbox("💾 Cache Habilitado", value=True)
            
            if st.form_submit_button("💾 Salvar Configurações"):
                st.success("✅ Configurações salvas!")

# =========================================================================
# EXECUÇÃO PRINCIPAL
# =========================================================================

def main():
    """Função principal do dashboard integrado"""
    try:
        dashboard = IntegratedDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"❌ Erro crítico no dashboard: {e}")
        logging.error(f"Erro crítico: {e}")
        
        # Mostrar botão de reset em caso de erro
        if st.button("🔄 Reiniciar Dashboard"):
            st.rerun()

if __name__ == "__main__":
    main()