"""
🎨 Componente de Header Multilingual
Header principal com breadcrumbs e informações do sistema
"""

import streamlit as st
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class HeaderComponent:
    """Componente de header com breadcrumbs e status"""
    
    def __init__(self, i18n_system):
        self.i18n = i18n_system
    
    def render(self, 
               current_page: str = 'overview',
               user_data: Optional[Dict[str, Any]] = None,
               breadcrumbs: Optional[List[str]] = None,
               show_system_status: bool = True):
        """Renderizar header completo"""
        
        # Header principal
        self._render_main_header(current_page)
        
        # Breadcrumbs
        if breadcrumbs:
            self._render_breadcrumbs(breadcrumbs)
        
        # Status do sistema
        if show_system_status:
            self._render_system_status(user_data)
        
        # Separador
        st.markdown("---")
    
    def _render_main_header(self, current_page: str):
        """Renderizar header principal"""
        col1, col2, col3 = st.columns([3, 2, 1])
        
        with col1:
            # Título da página atual
            page_title = self.i18n.t(f'navigation.{current_page}', current_page.title())
            st.markdown(f"# {page_title}")
        
        with col2:
            # Status em tempo real
            current_time = datetime.now().strftime("%H:%M:%S")
            st.markdown(f"🕐 {current_time}")
        
        with col3:
            # Indicador de status
            st.markdown("🟢 Online")
    
    def _render_breadcrumbs(self, breadcrumbs: List[str]):
        """Renderizar breadcrumbs de navegação"""
        breadcrumb_html = []
        
        for i, crumb in enumerate(breadcrumbs):
            translated_crumb = self.i18n.t(f'navigation.{crumb}', crumb.title())
            
            if i == len(breadcrumbs) - 1:
                # Último item (página atual)
                breadcrumb_html.append(f"<strong>{translated_crumb}</strong>")
            else:
                # Itens anteriores
                breadcrumb_html.append(translated_crumb)
        
        breadcrumb_str = " > ".join(breadcrumb_html)
        
        st.markdown(f"""
        <div style="
            background-color: #f0f2f6;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            margin-bottom: 1rem;
            font-size: 0.9rem;
        ">
            🏠 {breadcrumb_str}
        </div>
        """, unsafe_allow_html=True)
    
    def _render_system_status(self, user_data: Optional[Dict[str, Any]]):
        """Renderizar status do sistema"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Status dos dados
            self._render_status_card(
                "💾",
                self.i18n.t('data.status', 'Dados'),
                "✅ Carregados",
                "success"
            )
        
        with col2:
            # Status dos modelos
            self._render_status_card(
                "🤖",
                self.i18n.t('models.status', 'Modelos'),
                "⚠️ Treinar",
                "warning"
            )
        
        with col3:
            # Idioma atual
            current_lang = self.i18n.get_language().upper()
            self._render_status_card(
                "🌍",
                self.i18n.t('settings.language', 'Idioma'),
                current_lang,
                "info"
            )
        
        with col4:
            # Usuário atual
            if user_data:
                username = user_data.get('username', 'Unknown')
                role = user_data.get('role', 'guest')
                self._render_status_card(
                    "👤",
                    self.i18n.t('auth.user', 'Usuário'),
                    f"{username} ({role})",
                    "info"
                )
            else:
                self._render_status_card(
                    "👤",
                    self.i18n.t('auth.user', 'Usuário'),
                    self.i18n.t('auth.not_logged', 'Não logado'),
                    "warning"
                )
    
    def _render_status_card(self, icon: str, title: str, value: str, status_type: str):
        """Renderizar card de status"""
        
        # Cores baseadas no tipo de status
        colors = {
            "success": "#28a745",
            "warning": "#ffc107", 
            "error": "#dc3545",
            "info": "#17a2b8"
        }
        
        color = colors.get(status_type, "#6c757d")
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {color}15 0%, {color}25 100%);
            border-left: 4px solid {color};
            padding: 0.75rem;
            border-radius: 5px;
            margin-bottom: 0.5rem;
        ">
            <div style="display: flex; align-items: center; margin-bottom: 0.25rem;">
                <span style="font-size: 1.2rem; margin-right: 0.5rem;">{icon}</span>
                <strong style="font-size: 0.8rem; color: #666;">{title}</strong>
            </div>
            <div style="font-size: 0.9rem; font-weight: 600; color: {color};">
                {value}
            </div>
        </div>
        """, unsafe_allow_html=True)

    def render_page_actions(self, actions: List[Dict[str, Any]]):
        """Renderizar ações da página no header"""
        if not actions:
            return
        
        st.markdown("### ⚡ Ações Rápidas")
        
        cols = st.columns(len(actions))
        
        for i, action in enumerate(actions):
            with cols[i]:
                if st.button(
                    f"{action.get('icon', '⚡')} {action.get('label', 'Ação')}",
                    key=f"action_{i}",
                    help=action.get('help', ''),
                    type=action.get('type', 'secondary')
                ):
                    # Executar callback se fornecido
                    if 'callback' in action:
                        action['callback']()
                    
                    # Retornar ação clicada
                    return action.get('key', f'action_{i}')
        
        return None

    def render_alerts(self, alerts: List[Dict[str, Any]]):
        """Renderizar alertas no header"""
        if not alerts:
            return
        
        for alert in alerts:
            alert_type = alert.get('type', 'info')
            message = alert.get('message', '')
            dismissible = alert.get('dismissible', True)
            
            # Mapear tipos para funções do Streamlit
            if alert_type == 'success':
                st.success(message)
            elif alert_type == 'warning':
                st.warning(message)
            elif alert_type == 'error':
                st.error(message)
            else:
                st.info(message)

    def render_metrics_bar(self, metrics: Dict[str, Any]):
        """Renderizar barra de métricas no header"""
        if not metrics:
            return
        
        cols = st.columns(len(metrics))
        
        for i, (key, metric_data) in enumerate(metrics.items()):
            with cols[i]:
                label = metric_data.get('label', key)
                value = metric_data.get('value', 'N/A')
                delta = metric_data.get('delta', None)
                help_text = metric_data.get('help', None)
                
                st.metric(
                    label=label,
                    value=value,
                    delta=delta,
                    help=help_text
                )