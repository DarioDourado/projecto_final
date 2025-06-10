"""
🎨 Componente de Sidebar Multilingual
Navegação principal com autenticação e configurações
"""

import streamlit as st
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class SidebarComponent:
    """Componente de sidebar com navegação e configurações"""
    
    def __init__(self, i18n_system, auth_system):
        self.i18n = i18n_system
        self.auth = auth_system
    
    def render(self, user_data: Optional[Dict[str, Any]] = None) -> str:
        """Renderizar sidebar completo"""
        with st.sidebar:
            # Header do sistema
            self._render_header()
            
            # Informações do usuário
            if user_data:
                self._render_user_info(user_data)
            
            # Seletor de idioma
            self.i18n.show_language_selector()
            
            # Menu de navegação
            selected_page = self._render_navigation(user_data)
            
            # Configurações e logout
            if user_data:
                self._render_user_actions(user_data)
            
            return selected_page
    
    def _render_header(self):
        """Renderizar header do sistema"""
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h1 style="font-size: 1.5rem; margin: 0;">
                💰 Dashboard
            </h1>
            <p style="font-size: 0.8rem; color: #666; margin: 0;">
                v4.0 Multilingual
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
    
    def _render_user_info(self, user_data: Dict[str, Any]):
        """Renderizar informações do usuário"""
        user_name = self.auth.get_user_display_name(user_data)
        user_role = self.auth.get_user_role(user_data)
        
        # Card do usuário
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 1rem;
        ">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-size: 1.5rem; margin-right: 0.5rem;">👤</span>
                <strong>{user_name}</strong>
            </div>
            <div style="font-size: 0.8rem; opacity: 0.9;">
                🎯 {self.i18n.t('auth.role', 'Role')}: {user_role.title()}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_navigation(self, user_data: Optional[Dict[str, Any]]) -> str:
        """Renderizar menu de navegação baseado em permissões"""
        st.markdown(f"### {self.i18n.t('navigation.title', '🧭 Navegação')}")
        
        # Definir páginas disponíveis
        pages = self._get_available_pages(user_data)
        
        # Renderizar opções
        selected_page = None
        
        for page_key, page_info in pages.items():
            icon = page_info.get('icon', '📄')
            label = self.i18n.t(f'navigation.{page_key}', page_info.get('label', page_key))
            
            if st.button(f"{icon} {label}", key=f"nav_{page_key}", use_container_width=True):
                selected_page = page_key
        
        return selected_page or 'overview'
    
    def _get_available_pages(self, user_data: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Obter páginas disponíveis baseado em permissões"""
        all_pages = {
            'overview': {
                'icon': '📊',
                'label': 'Visão Geral',
                'permission': 'read'
            },
            'exploratory': {
                'icon': '📈',
                'label': 'Análise Exploratória',
                'permission': 'read'
            },
            'models': {
                'icon': '🤖',
                'label': 'Modelos ML',
                'permission': 'analyze'
            },
            'clustering': {
                'icon': '🎯',
                'label': 'Clustering',
                'permission': 'analyze'
            },
            'metrics': {
                'icon': '📊',
                'label': 'Métricas Avançadas',
                'permission': 'analyze'
            },
            'prediction': {
                'icon': '🔮',
                'label': 'Predição',
                'permission': 'predict'
            },
            'reports': {
                'icon': '📁',
                'label': 'Relatórios',
                'permission': 'read'
            },
            'settings': {
                'icon': '⚙️',
                'label': 'Configurações',
                'permission': 'read'
            }
        }
        
        # Adicionar página admin se usuário for admin
        if user_data and self.auth.is_admin(user_data):
            all_pages['admin'] = {
                'icon': '👨‍💼',
                'label': 'Administração',
                'permission': 'admin'
            }
        
        # Filtrar páginas baseado em permissões
        if not user_data:
            # Usuário não logado - apenas páginas públicas
            return {k: v for k, v in all_pages.items() if k in ['overview']}
        
        available_pages = {}
        for page_key, page_info in all_pages.items():
            required_permission = page_info.get('permission', 'read')
            
            if (self.auth.has_permission(user_data, required_permission) or
                self.auth.has_permission(user_data, 'all')):
                available_pages[page_key] = page_info
        
        return available_pages
    
    def _render_user_actions(self, user_data: Dict[str, Any]):
        """Renderizar ações do usuário"""
        st.markdown("---")
        
        # Configurações rápidas
        with st.expander(f"⚙️ {self.i18n.t('settings.title', 'Configurações')}"):
            # Theme toggle (placeholder)
            theme = st.toggle(
                self.i18n.t('settings.dark_theme', 'Tema Escuro'),
                key='dark_theme'
            )
            
            # Notifications
            notifications = st.toggle(
                self.i18n.t('settings.notifications', 'Notificações'),
                value=True,
                key='notifications'
            )
        
        # Logout
        if st.button(
            f"🚪 {self.i18n.t('auth.logout', 'Logout')}",
            key='logout_btn',
            use_container_width=True,
            type='secondary'
        ):
            session_id = st.session_state.get('session_id')
            if session_id:
                self.auth.logout(session_id)
            
            # Limpar sessão do Streamlit
            for key in list(st.session_state.keys()):
                if key not in ['language']:  # Manter idioma
                    del st.session_state[key]
            
            st.success(self.i18n.t('auth.logout_success', 'Logout realizado!'))
            st.rerun()
    
    def render_login_sidebar(self):
        """Renderizar sidebar para usuários não logados"""
        with st.sidebar:
            self._render_header()
            
            st.markdown(f"### {self.i18n.t('auth.login_title', '🔓 Acesso')}")
            
            # Informações do sistema
            st.info(self.i18n.t('auth.welcome', 'Bem-vindo ao sistema'))
            
            # Seletor de idioma
            self.i18n.show_language_selector()
            
            # Credenciais padrão (para demo)
            with st.expander("👥 Contas de Demonstração"):
                st.markdown("""
                **Admin:** admin / admin123  
                **Demo:** demo / demo123  
                **Guest:** guest / guest123
                """)