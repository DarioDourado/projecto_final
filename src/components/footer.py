"""
🦶 Componente de Footer Multilingual
Footer com informações do sistema e estatísticas
"""

import streamlit as st
from typing import Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class FooterComponent:
    """Componente de footer com informações do sistema"""
    
    def __init__(self, i18n_system):
        self.i18n = i18n_system
    
    def render(self, 
               show_system_info: bool = True,
               show_performance: bool = True,
               custom_content: Optional[str] = None):
        """Renderizar footer completo"""
        
        st.markdown("---")
        
        # Container principal do footer
        footer_container = st.container()
        
        with footer_container:
            # Conteúdo personalizado
            if custom_content:
                st.markdown(custom_content)
            
            # Informações do sistema
            if show_system_info:
                self._render_system_info()
            
            # Métricas de performance
            if show_performance:
                self._render_performance_metrics()
            
            # Footer final
            self._render_main_footer()
    
    def _render_system_info(self):
        """Renderizar informações do sistema"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            **📊 Sistema**
            - Versão: 4.0
            - Framework: Streamlit
            - Python: 3.9+
            """)
        
        with col2:
            current_lang = self.i18n.get_language().upper()
            available_langs = len(self.i18n.get_available_languages())
            
            st.markdown(f"""
            **🌍 Internacionalização**
            - Idioma: {current_lang}
            - Disponíveis: {available_langs}
            - Cache: {'✅' if self.i18n.config.get('cache_enabled') else '❌'}
            """)
        
        with col3:
            # Estatísticas da sessão
            session_keys = len(st.session_state.keys())
            
            st.markdown(f"""
            **💾 Sessão**
            - Chaves: {session_keys}
            - Idioma: {st.session_state.get('language', 'pt')}
            - Autenticado: {'✅' if st.session_state.get('authenticated') else '❌'}
            """)
        
        with col4:
            # Timestamp
            current_time = datetime.now()
            
            st.markdown(f"""
            **🕐 Timestamp**
            - Data: {current_time.strftime('%d/%m/%Y')}
            - Hora: {current_time.strftime('%H:%M:%S')}
            - Timezone: UTC
            """)
    
    def _render_performance_metrics(self):
        """Renderizar métricas de performance"""
        st.markdown("### 📈 Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Cache do i18n
            cache_size = len(self.i18n.cache) if hasattr(self.i18n, 'cache') else 0
            st.metric("🗄️ Cache i18n", f"{cache_size} itens")
        
        with col2:
            # Memória da sessão (simulada)
            session_size = len(str(st.session_state))
            st.metric("💾 Sessão", f"{session_size // 1024}KB")
        
        with col3:
            # Páginas carregadas (simulada)
            page_loads = st.session_state.get('page_loads', 0) + 1
            st.session_state.page_loads = page_loads
            st.metric("📄 Páginas", f"{page_loads}")
        
        with col4:
            # Uptime da sessão
            if 'session_start' not in st.session_state:
                st.session_state.session_start = datetime.now()
            
            uptime = datetime.now() - st.session_state.session_start
            uptime_minutes = int(uptime.total_seconds() // 60)
            st.metric("⏱️ Sessão", f"{uptime_minutes}min")
    
    def _render_main_footer(self):
        """Renderizar footer principal"""
        st.markdown("---")
        
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col1:
            st.markdown(f"""
            <div style="text-align: left; color: #666;">
                <small>
                    © 2024 {self.i18n.t('app.title', 'Dashboard de Análise Salarial')}<br>
                    {self.i18n.t('app.subtitle', 'Sistema Académico de Análise')}
                </small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="text-align: center; color: #666;">
                <small>🚀 v4.0</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="text-align: right; color: #666;">
                <small>
                    {self.i18n.t('footer.powered_by', 'Powered by')} Streamlit<br>
                    {self.i18n.t('footer.made_with', 'Feito com')} ❤️ {self.i18n.t('footer.for_education', 'para educação')}
                </small>
            </div>
            """, unsafe_allow_html=True)

    def render_debug_info(self, debug_data: Dict[str, Any]):
        """Renderizar informações de debug"""
        if not debug_data:
            return
        
        with st.expander("🐛 Debug Info"):
            for key, value in debug_data.items():
                st.text(f"{key}: {value}")