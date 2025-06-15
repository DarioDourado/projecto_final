"""
🌍 Dashboard Multilingual - Análise Salarial
Sistema Modular com Navegação por Botões - VERSÃO DEFINITIVA
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
import warnings

# Configurações
warnings.filterwarnings('ignore')

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent / "src"))

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Imports seguros
try:
    from src.utils.i18n import I18nSystem
    from src.auth.authentication import AuthenticationSystem
    from src.data.loader import load_data
except ImportError as e:
    logging.warning(f"Imports não disponíveis: {e}")
    # Fallbacks serão implementados

# Imports das páginas (com fallbacks)
try:
    from src.pages.overview import show_overview_page
    from src.pages.exploratory import show_exploratory_page
    from src.pages.models import show_models_page
    from src.pages.clustering import show_clustering_page
    from src.pages.association_rules import show_association_rules_page
    from src.pages.prediction import show_prediction_page
    from src.pages.metrics import show_metrics_page
    from src.pages.reports import show_reports_page
    from src.pages.admin import show_admin_page
    PAGES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Páginas não disponíveis: {e}")
    PAGES_AVAILABLE = False

class MultilingualDashboard:
    """Dashboard principal multilingual com navegação por botões"""
    
    def __init__(self):
        """Inicializar dashboard"""
        # Inicializar sistemas
        try:
            self.i18n = I18nSystem()
            self.auth = AuthenticationSystem(self.i18n)
        except:
            # Fallback para sistema básico
            self.i18n = self._create_fallback_i18n()
            self.auth = self._create_fallback_auth()
        
        # Configurar páginas
        self.pages = {
            'overview': {
                'title': 'Visão Geral',
                'icon': '📊',
                'roles': ['admin', 'user', 'analyst'],
                'func': self._show_overview
            },
            'exploratory': {
                'title': 'Análise Exploratória',
                'icon': '🔍',
                'roles': ['admin', 'user', 'analyst'],
                'func': self._show_exploratory
            },
            'models': {
                'title': 'Modelos ML',
                'icon': '🤖',
                'roles': ['admin', 'user', 'analyst'],
                'func': self._show_models
            },
            'clustering': {
                'title': 'Clustering',
                'icon': '🎯',
                'roles': ['admin', 'user', 'analyst'],
                'func': self._show_clustering
            },
            'association_rules': {
                'title': 'Regras de Associação',
                'icon': '📋',
                'roles': ['admin', 'user', 'analyst'],
                'func': self._show_association_rules
            },
            'prediction': {
                'title': 'Predição',
                'icon': '🔮',
                'roles': ['admin', 'user', 'analyst'],
                'func': self._show_prediction
            },
            'metrics': {
                'title': 'Métricas',
                'icon': '📈',
                'roles': ['admin', 'user', 'analyst'],
                'func': self._show_metrics
            },
            'reports': {
                'title': 'Relatórios',
                'icon': '📁',
                'roles': ['admin', 'user', 'analyst'],
                'func': self._show_reports
            },
            'admin': {
                'title': 'Administração',
                'icon': '⚙️',
                'roles': ['admin'],
                'func': self._show_admin
            }
        }
        
        # Estado da sessão
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 'overview'
        
        # Cache de dados
        self.data_cache = None

    def _create_fallback_i18n(self):
        """Criar sistema i18n básico como fallback"""
        class FallbackI18n:
            def t(self, key, default=None):
                return default or key.split('.')[-1].replace('_', ' ').title()
            
            def get_language(self):
                return 'pt'
            
            def set_language(self, lang):
                pass
        
        return FallbackI18n()

    def _create_fallback_auth(self):
        """Criar sistema auth básico como fallback"""
        class FallbackAuth:
            def __init__(self):
                self.users = {
                    'admin': {'password': 'admin123', 'role': 'admin', 'name': 'Admin'},
                    'user': {'password': 'user123', 'role': 'user', 'name': 'User'},
                    'demo': {'password': 'demo123', 'role': 'user', 'name': 'Demo'}
                }
            
            def is_authenticated(self):
                return st.session_state.get('authenticated', False)
            
            def authenticate(self, username, password):
                if username in self.users and self.users[username]['password'] == password:
                    st.session_state.authenticated = True
                    st.session_state.user_data = self.users[username]
                    st.session_state.username = username
                    return True
                return False
            
            def logout(self):
                st.session_state.authenticated = False
                st.session_state.user_data = {}
                st.session_state.username = None
            
            def get_user_data(self):
                return st.session_state.get('user_data', {})
        
        return FallbackAuth()

    def run(self):
        """Executar dashboard"""
        # Configurar página
        st.set_page_config(
            page_title="Dashboard Multilingual - Análise Salarial",
            page_icon="🌍",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # CSS personalizado
        self._apply_custom_css()
        
        # Header principal
        st.markdown("""
        <div class="main-header">
            🌍 Dashboard Multilingual - Análise Salarial
        </div>
        """, unsafe_allow_html=True)
        
        # Verificar autenticação
        if not self.auth.is_authenticated():
            self._show_login_page()
            return
        
        # Carregar dados (uma vez)
        if self.data_cache is None:
            self.data_cache = self._safe_load_data()
        
        # Layout principal
        self._show_sidebar()
        self._execute_current_page()

    def _apply_custom_css(self):
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
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2) !important;
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
            transition: all 0.3s ease !important;
        }
        
        .nav-button-inactive:hover {
            background: #e9ecef !important;
            border-color: #667eea !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1) !important;
        }
        
        .user-section {
            background: linear-gradient(135deg, #6c5ce7 0%, #a29bfe 100%);
            color: white;
            padding: 1rem;
            border-radius: 15px;
            margin-bottom: 1rem;
            text-align: center;
        }
        
        .data-status {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid #28a745;
        }
        
        .data-status.error {
            border-left-color: #dc3545;
        }
        
        .data-status.warning {
            border-left-color: #ffc107;
        }
        </style>
        """, unsafe_allow_html=True)

    def _safe_load_data(self):
        """Carregar dados de forma segura"""
        try:
            # Tentar carregar usando o sistema modular
            if 'load_data' in globals():
                raw_data = load_data()
                
                # Garantir retorno consistente
                if isinstance(raw_data, dict):
                    return raw_data
                elif isinstance(raw_data, tuple) and len(raw_data) >= 2:
                    return {
                        'df': raw_data[0],
                        'status': raw_data[1] if len(raw_data) > 1 else '✅ Dados carregados',
                        'files_status': raw_data[2] if len(raw_data) > 2 else {'analysis': [], 'models': [], 'images': []},
                        'source': raw_data[3] if len(raw_data) > 3 else None
                    }
            
            # Fallback: tentar carregar diretamente
            return self._fallback_load_data()
            
        except Exception as e:
            logging.error(f"Erro ao carregar dados: {e}")
            return self._fallback_load_data()

    def _fallback_load_data(self):
        """Fallback para carregamento de dados"""
        try:
            # Tentar caminhos conhecidos
            csv_paths = [
                "data/raw/4-Carateristicas_salario.csv",
                "4-Carateristicas_salario.csv",
                "bkp/4-Carateristicas_salario.csv"
            ]
            
            for csv_path in csv_paths:
                if Path(csv_path).exists():
                    df = pd.read_csv(csv_path)
                    return {
                        'df': df,
                        'status': f'✅ Dados carregados de {csv_path}',
                        'files_status': self._scan_files(),
                        'source': csv_path
                    }
            
            # Se não encontrou, criar dados de exemplo
            return self._create_sample_data()
            
        except Exception as e:
            logging.error(f"Erro no fallback: {e}")
            return {
                'df': None,
                'status': f'❌ Erro ao carregar dados: {e}',
                'files_status': {'analysis': [], 'models': [], 'images': []},
                'source': None
            }

    def _scan_files(self):
        """Escanear arquivos disponíveis"""
        files_status = {
            'analysis': [],
            'models': [],
            'images': []
        }
        
        # Diretórios para buscar
        dirs = {
            'analysis': ['output/analysis', 'output', '.'],
            'models': ['models', 'data/processed', 'output/models'],
            'images': ['output/images', 'imagens', 'images']
        }
        
        for category, paths in dirs.items():
            for path_str in paths:
                path = Path(path_str)
                if path.exists():
                    if category == 'analysis':
                        files_status[category].extend(path.glob('*.csv'))
                        files_status[category].extend(path.glob('*.md'))
                    elif category == 'models':
                        files_status[category].extend(path.glob('*.pkl'))
                        files_status[category].extend(path.glob('*.joblib'))
                    elif category == 'images':
                        files_status[category].extend(path.glob('*.png'))
                        files_status[category].extend(path.glob('*.jpg'))
        
        return files_status

    def _create_sample_data(self):
        """Criar dados de exemplo"""
        try:
            np.random.seed(42)
            
            n_samples = 1000
            data = {
                'age': np.random.randint(18, 70, n_samples),
                'workclass': np.random.choice(['Private', 'Self-emp-not-inc', 'Federal-gov'], n_samples),
                'education': np.random.choice(['Bachelors', 'HS-grad', 'Masters', 'Some-college'], n_samples),
                'education-num': np.random.randint(1, 17, n_samples),
                'marital-status': np.random.choice(['Married-civ-spouse', 'Divorced', 'Never-married'], n_samples),
                'occupation': np.random.choice(['Tech-support', 'Craft-repair', 'Sales', 'Exec-managerial'], n_samples),
                'relationship': np.random.choice(['Wife', 'Husband', 'Not-in-family', 'Own-child'], n_samples),
                'race': np.random.choice(['White', 'Black', 'Asian-Pac-Islander'], n_samples),
                'sex': np.random.choice(['Female', 'Male'], n_samples),
                'capital-gain': np.random.randint(0, 10000, n_samples),
                'capital-loss': np.random.randint(0, 5000, n_samples),
                'hours-per-week': np.random.randint(20, 80, n_samples),
                'native-country': np.random.choice(['United-States', 'Canada', 'Mexico'], n_samples),
                'salary': np.random.choice(['<=50K', '>50K'], n_samples)
            }
            
            df = pd.DataFrame(data)
            
            return {
                'df': df,
                'status': '✅ Dados de exemplo criados',
                'files_status': {'analysis': [], 'models': [], 'images': []},
                'source': 'sample_data'
            }
            
        except Exception as e:
            return {
                'df': None,
                'status': f'❌ Erro ao criar dados de exemplo: {e}',
                'files_status': {'analysis': [], 'models': [], 'images': []},
                'source': None
            }

    def _show_login_page(self):
        """Mostrar página de login"""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("### 🔐 Login")
            
            with st.form("login_form", clear_on_submit=True):
                username = st.text_input(
                    "👤 Nome de usuário",
                    placeholder="admin, user, demo"
                )
                password = st.text_input(
                    "🔑 Senha",
                    type="password",
                    placeholder="admin123, user123, demo123"
                )
                
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
        """Mostrar sidebar com navegação"""
        with st.sidebar:
            # Informações do usuário
            self._show_user_info()
            
            st.markdown("---")
            
            # Navegação por botões
            self._show_navigation_buttons()
            
            st.markdown("---")
            
            # Status dos dados
            self._show_data_status()
            
            st.markdown("---")
            
            # Controles extras
            self._show_extra_controls()

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
            
            if st.button("🚪 Logout", use_container_width=True, type="secondary"):
                self.auth.logout()
                st.rerun()

    def _show_navigation_buttons(self):
        """Mostrar navegação com botões"""
        st.markdown("### 🧭 Navegação")
        
        # Obter papel do usuário
        user_data = self.auth.get_user_data()
        user_role = user_data.get('role', 'user')
        
        # Filtrar páginas disponíveis
        available_pages = []
        for page_key, page_info in self.pages.items():
            if user_role in page_info['roles']:
                available_pages.append((page_key, page_info))
        
        # Criar botões de navegação
        for page_key, page_info in available_pages:
            button_text = f"{page_info['icon']} {page_info['title']}"
            
            # Página atual
            is_current = st.session_state.current_page == page_key
            
            if is_current:
                # Botão ativo (visual apenas)
                st.markdown(f"""
                <div class="nav-button-active">
                    {button_text}
                </div>
                """, unsafe_allow_html=True)
            else:
                # Botão clicável
                if st.button(
                    button_text,
                    key=f"nav_{page_key}",
                    use_container_width=True,
                    type="secondary"
                ):
                    st.session_state.current_page = page_key
                    st.rerun()

    def _show_data_status(self):
        """Mostrar status dos dados"""
        st.markdown("### 📊 Status dos Dados")
        
        if self.data_cache and isinstance(self.data_cache, dict):
            status = self.data_cache.get('status', '❌ Não carregado')
            
            # Determinar tipo de status
            status_type = "error"
            if "✅" in status:
                status_type = "success"
            elif "⚠️" in status:
                status_type = "warning"
            
            st.markdown(f"""
            <div class="data-status {status_type}">
                {status}
            </div>
            """, unsafe_allow_html=True)
            
            # Métricas
            df = self.data_cache.get('df')
            if df is not None:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📋 Registros", f"{len(df):,}")
                with col2:
                    st.metric("📊 Colunas", len(df.columns))
                
                if 'salary' in df.columns:
                    high_salary_rate = (df['salary'] == '>50K').mean()
                    st.metric("💰 Salário Alto", f"{high_salary_rate:.1%}")
        else:
            st.error("❌ Erro no carregamento dos dados")

    def _show_extra_controls(self):
        """Mostrar controles extras"""
        st.markdown("### ⚙️ Controles")
        
        # Recarregar dados
        if st.button("🔄 Recarregar Dados", use_container_width=True):
            self.data_cache = None
            self.data_cache = self._safe_load_data()
            st.success("✅ Dados recarregados!")
            st.rerun()

    def _execute_current_page(self):
        """Executar página atual"""
        try:
            current_page = st.session_state.current_page
            
            if current_page in self.pages:
                page_info = self.pages[current_page]
                
                # Verificar permissões
                user_data = self.auth.get_user_data()
                user_role = user_data.get('role', 'user')
                
                if user_role not in page_info['roles']:
                    st.error("❌ Acesso negado!")
                    return
                
                # Executar página
                page_info['func']()
                
            else:
                st.error(f"❌ Página '{current_page}' não encontrada")
                
        except Exception as e:
            st.error(f"❌ Erro ao executar página: {e}")
            logging.error(f"Erro na página {current_page}: {e}")

    # Métodos das páginas (com fallbacks)
    def _show_overview(self):
        """Página de visão geral"""
        if PAGES_AVAILABLE:
            try:
                df = self.data_cache.get('df') if self.data_cache else None
                files_status = self.data_cache.get('files_status', {}) if self.data_cache else {}
                show_overview_page(df, files_status, self.i18n)
                return
            except:
                pass
        
        # Fallback
        self._show_fallback_page("📊 Visão Geral")

    def _show_exploratory(self):
        """Página de análise exploratória"""
        if PAGES_AVAILABLE:
            try:
                df = self.data_cache.get('df') if self.data_cache else None
                files_status = self.data_cache.get('files_status', {}) if self.data_cache else {}
                show_exploratory_page(df, files_status, self.i18n)
                return
            except:
                pass
        
        self._show_fallback_page("🔍 Análise Exploratória")

    def _show_models(self):
        """Página de modelos ML"""
        if PAGES_AVAILABLE:
            try:
                df = self.data_cache.get('df') if self.data_cache else None
                files_status = self.data_cache.get('files_status', {}) if self.data_cache else {}
                show_models_page(df, files_status, self.i18n)
                return
            except:
                pass
        
        self._show_fallback_page("🤖 Modelos ML")

    def _show_clustering(self):
        """Página de clustering"""
        if PAGES_AVAILABLE:
            try:
                df = self.data_cache.get('df') if self.data_cache else None
                files_status = self.data_cache.get('files_status', {}) if self.data_cache else {}
                show_clustering_page(df, files_status, self.i18n)
                return
            except:
                pass
        
        self._show_fallback_page("🎯 Clustering")

    def _show_association_rules(self):
        """Página de regras de associação"""
        if PAGES_AVAILABLE:
            try:
                df = self.data_cache.get('df') if self.data_cache else None
                files_status = self.data_cache.get('files_status', {}) if self.data_cache else {}
                show_association_rules_page(df, files_status, self.i18n)
                return
            except:
                pass
        
        self._show_association_fallback()

    def _show_prediction(self):
        """Página de predição"""
        if PAGES_AVAILABLE:
            try:
                df = self.data_cache.get('df') if self.data_cache else None
                files_status = self.data_cache.get('files_status', {}) if self.data_cache else {}
                show_prediction_page(df, files_status, self.i18n)
                return
            except:
                pass
        
        self._show_fallback_page("🔮 Predição")

    def _show_metrics(self):
        """Página de métricas"""
        if PAGES_AVAILABLE:
            try:
                df = self.data_cache.get('df') if self.data_cache else None
                files_status = self.data_cache.get('files_status', {}) if self.data_cache else {}
                show_metrics_page(df, files_status, self.i18n)
                return
            except:
                pass
        
        self._show_fallback_page("📈 Métricas")

    def _show_reports(self):
        """Página de relatórios"""
        if PAGES_AVAILABLE:
            try:
                df = self.data_cache.get('df') if self.data_cache else None
                files_status = self.data_cache.get('files_status', {}) if self.data_cache else {}
                show_reports_page(df, files_status, self.i18n)
                return
            except:
                pass
        
        self._show_fallback_page("📁 Relatórios")

    def _show_admin(self):
        """Página de administração"""
        if PAGES_AVAILABLE:
            try:
                df = self.data_cache.get('df') if self.data_cache else None
                files_status = self.data_cache.get('files_status', {}) if self.data_cache else {}
                show_admin_page(df, files_status, self.i18n)
                return
            except:
                pass
        
        self._show_fallback_page("⚙️ Administração")

    def _show_fallback_page(self, title):
        """Página fallback quando a original não existe"""
        st.header(title)
        st.info("🚧 Esta página está em desenvolvimento")
        
        df = self.data_cache.get('df') if self.data_cache else None
        
        if df is not None:
            st.markdown("### 📊 Visualização dos Dados")
            st.dataframe(df.head(), use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📋 Total de Registros", f"{len(df):,}")
            with col2:
                st.metric("📊 Colunas", len(df.columns))
            with col3:
                if 'salary' in df.columns:
                    high_salary = (df['salary'] == '>50K').sum()
                    st.metric("💰 Salários Altos", f"{high_salary:,}")
        else:
            st.warning("❌ Dados não disponíveis")

    def _show_association_fallback(self):
        """Fallback específico para regras de associação"""
        st.header("📋 Regras de Associação")
        
        # Procurar arquivos
        association_files = []
        if self.data_cache and 'files_status' in self.data_cache:
            files_status = self.data_cache['files_status']
            if 'analysis' in files_status:
                association_files = [f for f in files_status['analysis'] 
                                   if any(keyword in str(f).lower() 
                                         for keyword in ['apriori', 'fp_growth', 'eclat', 'association'])]
        
        if association_files:
            st.success(f"✅ {len(association_files)} arquivo(s) encontrado(s)")
            
            for file in association_files:
                with st.expander(f"📁 {file.name}"):
                    try:
                        if file.suffix == '.csv':
                            df_rules = pd.read_csv(file)
                            st.dataframe(df_rules.head(10), use_container_width=True)
                        else:
                            st.text(f"Arquivo: {file.name}")
                    except Exception as e:
                        st.error(f"Erro ao ler arquivo: {e}")
        else:
            st.warning("⚠️ Nenhum resultado de regras de associação encontrado")
            st.info("💡 Execute o pipeline principal para gerar as regras: `python main.py`")

# Executar aplicação
if __name__ == "__main__":
    dashboard = MultilingualDashboard()
    dashboard.run()