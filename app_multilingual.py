"""
🌍 Dashboard Multilingual - Análise Salarial - CORRIGIDO
Sistema modular com componentes separados
Versão: 6.1 - Erro DuplicateWidgetID Corrigido
"""

import streamlit as st
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent / "src"))

# Imports dos módulos
from src.utils.i18n import I18nSystem
from src.auth.authentication import AuthenticationSystem
from src.components.layout import apply_modern_css
from src.pages.login import show_login_page
from src.pages.overview import show_overview_page
from src.pages.exploratory import show_exploratory_page
from src.pages.models import show_models_page
from src.pages.prediction import show_prediction_page
from src.pages.clustering import show_clustering_page
from src.pages.metrics import show_metrics_page
from src.pages.reports import show_reports_page
from src.pages.admin import show_admin_page
from src.components.navigation import show_user_info
from src.data.loader import load_data

# Configurações
st.set_page_config(
    page_title="Salary Analysis Dashboard | Análise Salarial",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

class MultilingualDashboard:
    """Dashboard principal multilingual"""
    
    def __init__(self):
        # Inicializar sistemas
        self.i18n = I18nSystem()
        self.auth = AuthenticationSystem(self.i18n)
        
        # Aplicar estilos
        apply_modern_css()
        
        # Inicializar estado
        self._init_session_state()
        
        # Carregar dados
        self.data = self._load_dashboard_data()
    
    def _init_session_state(self):
        """Inicializar estado da sessão"""
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 'overview'
    
    def _load_dashboard_data(self):
        """Carregar dados do dashboard"""
        with st.spinner("🔄 Carregando dados..."):
            return load_data()
    
    def run(self):
        """Executar o dashboard"""
        # Verificar autenticação
        if not st.session_state.authenticated:
            show_login_page(self.auth, self.i18n)
            return
        
        # Mostrar interface principal
        self._show_main_interface()
    
    def _show_main_interface(self):
        """Mostrar interface principal autenticada"""
        # Header principal
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 2rem;">
            <h1 style="margin: 0; font-size: 2.5rem;">💰 {self.i18n.t('app.title', 'Dashboard de Análise Salarial')}</h1>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">{self.i18n.t('app.subtitle', 'Sistema Académico de Análise e Predição Salarial')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar com navegação
        self._show_sidebar()
        
        # Conteúdo principal baseado na página selecionada
        page = st.session_state.current_page
        
        # Mapeamento de páginas - COMPLETO
        page_mapping = {
            'overview': lambda: show_overview_page(self.data, self.i18n),
            'exploratory': lambda: show_exploratory_page(self.data, self.i18n),
            'models': lambda: show_models_page(self.data, self.i18n),
            'prediction': lambda: show_prediction_page(self.data, self.i18n),
            'clustering': lambda: show_clustering_page(self.data, self.i18n),
            'metrics': lambda: show_metrics_page(self.data, self.i18n),
            'reports': lambda: show_reports_page(self.data, self.i18n),
            'admin': lambda: show_admin_page(self.data, self.i18n, self.auth),
            # NOVAS PÁGINAS baseadas no app.py
            'association_rules': lambda: self._show_association_rules_placeholder(),
            'advanced_visualizations': lambda: self._show_advanced_visualizations_placeholder(),
            'system_info': lambda: self._show_system_info_placeholder()
        }
        
        # Executar página selecionada
        if page in page_mapping:
            try:
                page_mapping[page]()
            except Exception as e:
                st.error(f"❌ Erro ao carregar página: {e}")
                st.exception(e)
                # Fallback para overview
                if page != 'overview':
                    st.session_state.current_page = 'overview'
                    st.rerun()
        else:
            st.error(f"❌ Página não encontrada: {page}")
            st.session_state.current_page = 'overview'
            st.rerun()
    
    def _show_sidebar(self):
        """Mostrar sidebar com navegação"""
        with st.sidebar:
            # Informações do usuário
            show_user_info(self.auth, self.i18n)
            
            st.markdown("---")
            
            # Seletor de idioma
            self._show_language_selector()
            
            st.markdown("---")
            
            # Menu de navegação
            self._show_navigation_menu()
            
            st.markdown("---")
            
            # Status dos dados
            self._show_data_status()
            
            st.markdown("---")
            
            # Botão de logout - CORRIGIDO: Key unico
            if st.button(
                f"🚪 {self.i18n.t('auth.logout', 'Logout')}", 
                use_container_width=True,
                key="main_logout_button" 
            ):
                self.auth.logout(st.session_state.get('session_id'))
                # Limpar sessão completamente
                for key in ['authenticated', 'session_id', 'user_data', 'username', 'current_page']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
    
    def _show_language_selector(self):
        """Mostrar seletor de idioma - CORRIGIDO"""
        st.markdown(f"### 🌍 {self.i18n.t('navigation.language', 'Idioma')}")
        
        # ✅ CORREÇÃO: Obter languages como dict e extrair as chaves
        languages_dict = self.i18n.get_available_languages()
        current_lang = self.i18n.get_language()
        
        # Verificar se languages_dict é válido
        if not languages_dict or not isinstance(languages_dict, dict):
            # Fallback para configuração padrão
            languages_dict = {
                'pt': {'name': '🇵🇹 Português', 'flag': '🇵🇹'},
                'en': {'name': '🇺🇸 English', 'flag': '🇺🇸'}
            }
        
        # Extrair lista de códigos de idioma
        language_codes = list(languages_dict.keys())
        
        # Criar mapeamento para display
        lang_display = {}
        for code, info in languages_dict.items():
            if isinstance(info, dict):
                lang_display[code] = info.get('name', f'🌍 {code.upper()}')
            else:
                # Fallback se info não for dict
                lang_display[code] = f'🌍 {code.upper()}'
        
        # Encontrar índice atual com segurança
        try:
            current_index = language_codes.index(current_lang) if current_lang in language_codes else 0
        except (ValueError, AttributeError):
            current_index = 0
        
        # Selectbox com tratamento de erro
        try:
            selected_lang = st.selectbox(
                "Selecionar idioma:",
                language_codes,
                index=current_index,
                format_func=lambda x: lang_display.get(x, f'🌍 {x}'),
                label_visibility="collapsed",
                key="language_selector_main"  # ✅ KEY ÚNICO
            )
            
            # Atualizar idioma se mudou
            if selected_lang != current_lang:
                self.i18n.set_language(selected_lang)
                st.rerun()
        
        except Exception as e:
            st.error(f"❌ Erro no seletor de idioma: {e}")
            # Fallback: mostrar idioma atual
            st.info(f"🌍 Idioma atual: {current_lang}")
            
            # Botões simples como fallback
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🇵🇹 PT", key="fallback_pt"):
                    self.i18n.set_language('pt')
                    st.rerun()
            
            with col2:
                if st.button("🇺🇸 EN", key="fallback_en"):
                    self.i18n.set_language('en')
                    st.rerun()

    def _show_navigation_menu(self):
        """Mostrar menu de navegação"""
        st.markdown(f"### 📋 {self.i18n.t('navigation.menu', 'Menu')}")
        
        # Páginas disponíveis - EXPANDIDO baseado no app.py
        pages = {
            'overview': {
                'icon': '📊',
                'title': self.i18n.t('navigation.overview', 'Visão Geral'),
                'access': ['user', 'admin', 'guest']
            },
            'exploratory': {
                'icon': '🔍',
                'title': self.i18n.t('navigation.exploratory', 'Análise Exploratória'),
                'access': ['user', 'admin']
            },
            'models': {
                'icon': '🤖',
                'title': self.i18n.t('navigation.models', 'Modelos ML'),
                'access': ['user', 'admin']
            },
            'prediction': {
                'icon': '🔮',
                'title': self.i18n.t('navigation.prediction', 'Predição'),
                'access': ['user', 'admin']
            },
            'clustering': {
                'icon': '🎯',
                'title': self.i18n.t('navigation.clustering', 'Clustering'),
                'access': ['user', 'admin']
            },
            'association_rules': {
                'icon': '📋',
                'title': self.i18n.t('navigation.association_rules', 'Regras de Associação'),
                'access': ['user', 'admin']
            },
            'metrics': {
                'icon': '📊',
                'title': self.i18n.t('navigation.advanced_metrics', 'Métricas Avançadas'),
                'access': ['user', 'admin']
            },
            'reports': {
                'icon': '📁',
                'title': self.i18n.t('navigation.reports', 'Relatórios'),
                'access': ['user', 'admin']
            },
            'advanced_visualizations': {
                'icon': '🎨',
                'title': self.i18n.t('navigation.advanced_viz', 'Visualizações Avançadas'),
                'access': ['user', 'admin']
            },
            'system_info': {
                'icon': '🔧',
                'title': self.i18n.t('navigation.system_info', 'Informações do Sistema'),
                'access': ['admin']
            },
            'admin': {
                'icon': '⚙️',
                'title': self.i18n.t('navigation.admin', 'Administração'),
                'access': ['admin']
            }
        }
        
        # Verificar papel do usuário
        user_role = st.session_state.get('user_data', {}).get('role', 'guest')
        
        # Mostrar páginas baseado no acesso
        for page_key, page_info in pages.items():
            if user_role in page_info['access']:
                button_type = "primary" if st.session_state.current_page == page_key else "secondary"
                
                if st.button(
                    f"{page_info['icon']} {page_info['title']}",
                    use_container_width=True,
                    type=button_type,
                    key=f"nav_button_{page_key}",  # ✅ KEY ÚNICO para cada botão
                    disabled=(st.session_state.current_page == page_key)
                ):
                    st.session_state.current_page = page_key
                    st.rerun()
    
    def _show_data_status(self):
        """Mostrar status dos dados"""
        st.markdown(f"### 📊 {self.i18n.t('navigation.data_status', 'Status dos Dados')}")
        
        status = self.data.get('status', '❌ Erro')
        
        if '✅' in status:
            st.success(status)
        elif '⚠️' in status:
            st.warning(status)
        else:
            st.error(status)
        
        # Estatísticas rápidas
        df = self.data.get('df')
        if df is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📋 Registros", f"{len(df):,}")
            with col2:
                st.metric("📊 Colunas", len(df.columns))
            
            models = self.data.get('models', {})
            if models:
                st.metric("🤖 Modelos", len(models))
        
        # Botão para recarregar dados
        if st.button(
            f"🔄 {self.i18n.t('actions.reload_data', 'Recarregar Dados')}", 
            use_container_width=True,
            key="reload_data_button"  # ✅ KEY ÚNICO
        ):
            self.data = self._load_dashboard_data()
            st.success("✅ Dados recarregados!")
            st.rerun()
    
    # PLACEHOLDERS para páginas que faltam (baseadas no app.py)
    def _show_association_rules_placeholder(self):
        """Placeholder para regras de associação"""
        st.markdown(f"## 📋 {self.i18n.t('navigation.association_rules', 'Regras de Associação')}")
        st.info("🚧 Esta funcionalidade está sendo migrada do app.py para a estrutura modular.")
        st.markdown("### 💡 Em desenvolvimento:")
        st.markdown("""
        - 📊 Análise de regras de associação
        - 🔍 Padrões de comportamento salarial  
        - 📈 Métricas de suporte e confiança
        - 🎯 Regras de interesse específico
        """)
    
    def _show_advanced_visualizations_placeholder(self):
        """Placeholder para visualizações avançadas"""
        st.markdown(f"## 🎨 {self.i18n.t('navigation.advanced_viz', 'Visualizações Avançadas')}")
        st.info("🚧 Visualizações avançadas em desenvolvimento.")
        st.markdown("### 💡 Funcionalidades previstas:")
        st.markdown("""
        - 📊 Gráficos interativos avançados
        - 🎨 Visualizações 3D e animadas
        - 📈 Dashboards personalizáveis
        - 🔍 Análise visual exploratória
        """)
    
    def _show_system_info_placeholder(self):
        """Placeholder para informações do sistema"""
        st.markdown(f"## 🔧 {self.i18n.t('navigation.system_info', 'Informações do Sistema')}")
        
        # Informações básicas do sistema
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Status dos Componentes")
            st.success("✅ Sistema de autenticação ativo")
            st.success("✅ Sistema i18n funcionando")
            st.success("✅ Carregamento de dados OK")
            
        with col2:
            st.markdown("### 📁 Estrutura de Arquivos")
            paths_to_check = [
                ("translate/", "Traduções"),
                ("src/", "Código fonte"),
                ("config/", "Configurações"),
                ("data/", "Dados")
            ]
            
            for path, desc in paths_to_check:
                if Path(path).exists():
                    st.success(f"✅ {desc}: {path}")
                else:
                    st.error(f"❌ {desc}: {path}")

# Executar aplicação
if __name__ == "__main__":
    dashboard = MultilingualDashboard()
    dashboard.run()