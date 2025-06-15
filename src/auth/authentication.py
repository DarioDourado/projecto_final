"""
Sistema de autenticação com suporte a i18n - CORRIGIDO
"""

import streamlit as st
import hashlib
import logging
from typing import Dict, Optional, Any
from pathlib import Path
import json

class AuthenticationSystem:
    """Sistema de autenticação para o dashboard"""
    
    def __init__(self, i18n_system):
        self.i18n = i18n_system
        self.logger = logging.getLogger(__name__)
        
        # Usuários pré-definidos (ESTRUTURA CORRIGIDA)
        self.users = {
            'admin': {
                'password': 'admin123',  # Senha em texto simples para hash
                'role': 'admin',
                'name': 'Administrador',
                'email': 'admin@dashboard.com'
            },
            'user': {
                'password': 'user123',
                'role': 'user', 
                'name': 'Usuário',
                'email': 'user@dashboard.com'
            },
            'demo': {
                'password': 'demo123',
                'role': 'user',
                'name': 'Demo User',
                'email': 'demo@dashboard.com'
            },
            'analyst': {
                'password': 'analyst123',
                'role': 'analyst',
                'name': 'Analista',
                'email': 'analyst@dashboard.com'
            }
        }
        
        # Inicializar sessão
        self._initialize_session()

    def _initialize_session(self):
        """Inicializar variáveis de sessão"""
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'user_data' not in st.session_state:
            st.session_state.user_data = {}
        if 'username' not in st.session_state:
            st.session_state.username = None
        if 'login_attempts' not in st.session_state:
            st.session_state.login_attempts = 0

    def _hash_password(self, password: str) -> str:
        """Hash da senha usando SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()

    def authenticate(self, username: str, password: str) -> bool:
        """Autenticar usuário - MÉTODO CORRIGIDO"""
        try:
            self.logger.info(f"Tentativa de autenticação para usuário: {username}")
            
            if username not in self.users:
                self.logger.warning(f"Usuário '{username}' não encontrado")
                return False
            
            user_data = self.users[username]
            
            # Verificar senha - COMPARAÇÃO CORRIGIDA
            stored_password = user_data.get('password')
            
            if stored_password == password:  # Comparação direta (em produção use hash)
                # Autenticação bem-sucedida
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.user_data = {
                    'username': username,
                    'role': user_data.get('role', 'user'),
                    'name': user_data.get('name', username),
                    'email': user_data.get('email', f'{username}@dashboard.com')
                }
                st.session_state.login_attempts = 0
                
                self.logger.info(f"Autenticação bem-sucedida para: {username}")
                return True
            else:
                st.session_state.login_attempts += 1
                self.logger.warning(f"Senha incorreta para: {username}")
                return False
                
        except Exception as e:
            self.logger.error(f"Erro na autenticação: {e}")
            return False

    def logout(self):
        """Fazer logout do usuário"""
        st.session_state.authenticated = False
        st.session_state.user_data = {}
        st.session_state.username = None
        st.session_state.login_attempts = 0
        self.logger.info("Logout realizado")

    def is_authenticated(self) -> bool:
        """Verificar se usuário está autenticado"""
        return st.session_state.get('authenticated', False)

    def get_user_data(self) -> Dict[str, Any]:
        """Obter dados do usuário atual"""
        return st.session_state.get('user_data', {})

    def is_admin(self) -> bool:
        """Verificar se usuário atual é administrador"""
        user_data = self.get_user_data()
        return user_data.get('role') == 'admin'

    def get_login_attempts(self) -> int:
        """Obter número de tentativas de login"""
        return st.session_state.get('login_attempts', 0)

    def is_locked(self) -> bool:
        """Verificar se conta está bloqueada por muitas tentativas"""
        return self.get_login_attempts() >= 5

    def reset_login_attempts(self):
        """Reset tentativas de login"""
        st.session_state.login_attempts = 0

    def show_login_form(self):
        """Mostrar formulário de login - MÉTODO ATUALIZADO"""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("### 🔐 Login")
            
            # Formulário de login
            with st.form("login_form", clear_on_submit=True):
                username = st.text_input(
                    f"👤 {self.i18n.t('auth.username', 'Nome de usuário')}",
                    placeholder="admin, user, demo ou analyst"
                )
                password = st.text_input(
                    f"🔑 {self.i18n.t('auth.password', 'Senha')}",
                    type="password",
                    placeholder="admin123, user123, demo123 ou analyst123"
                )
                
                login_button = st.form_submit_button(
                    f"🚪 {self.i18n.t('auth.login_button', 'Entrar')}",
                    use_container_width=True
                )
                
                if login_button and username and password:
                    if self.is_locked():
                        st.error(f"🔒 {self.i18n.t('auth.account_locked', 'Conta bloqueada por muitas tentativas')}")
                    elif self.authenticate(username, password):
                        st.success(f"✅ {self.i18n.t('auth.login_success', 'Login realizado com sucesso!')}")
                        st.rerun()
                    else:
                        attempts = self.get_login_attempts()
                        remaining = 5 - attempts
                        
                        if remaining > 0:
                            st.error(f"❌ {self.i18n.t('auth.invalid_credentials', 'Credenciais inválidas')} ({remaining} tentativas restantes)")
                        else:
                            st.error(f"🔒 {self.i18n.t('auth.account_locked', 'Conta bloqueada por muitas tentativas')}")
        
        # Informações de ajuda
        with st.expander(f"ℹ️ {self.i18n.t('auth.login_help', 'Ajuda de Login')}"):
            st.markdown("""
            **👥 Usuários disponíveis:**
            - **admin** / admin123 (Administrador)
            - **user** / user123 (Usuário) 
            - **demo** / demo123 (Demonstração)
            - **analyst** / analyst123 (Analista)
            
            **🔧 Problemas de login?**
            - Verifique as credenciais
            - Aguarde se houver muitas tentativas
            - Recarregue a página se necessário
            """)

    def show_user_info(self):
        """Mostrar informações do usuário na sidebar"""
        if self.is_authenticated():
            user_data = self.get_user_data()
            
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 👤 Usuário Logado")
            st.sidebar.markdown(f"**Nome:** {user_data.get('name', 'N/A')}")
            st.sidebar.markdown(f"**Papel:** {user_data.get('role', 'N/A')}")
            
            if st.sidebar.button(f"🚪 {self.i18n.t('auth.logout', 'Sair')}"):
                self.logout()
                st.rerun()

    def require_authentication(self):
        """Middleware para páginas que requerem autenticação"""
        if not self.is_authenticated():
            st.warning(self.i18n.t('auth.login_required', 'Login necessário para acessar esta página'))
            return False
        return True

    def require_admin(self):
        """Middleware para páginas que requerem acesso de admin"""
        if not self.require_authentication():
            return False
        
        if not self.is_admin():
            st.error(self.i18n.t('auth.admin_required', 'Acesso restrito a administradores'))
            return False
        
        return True

    def create_session(self, username: str, user_data: dict) -> str:
        """Criar sessão (para compatibilidade)"""
        # Para o sistema atual, a sessão é gerenciada pelo Streamlit
        return f"session_{username}"

    def create_user(self, username: str, password: str, role: str = 'user', name: str = None):
        """Criar novo usuário (função admin)"""
        if username in self.users:
            return False, self.i18n.t('auth.user_exists', 'Usuário já existe')
        
        self.users[username] = {
            'password': password,  # Em produção, use hash
            'role': role,
            'name': name or username,
            'email': f'{username}@dashboard.com'
        }
        
        self.logger.info(f"Novo usuário criado: {username}")
        return True, self.i18n.t('auth.user_created', 'Usuário criado com sucesso')

    def change_password(self, username: str, old_password: str, new_password: str):
        """Alterar senha do usuário"""
        if username not in self.users:
            return False, self.i18n.t('auth.user_not_found', 'Usuário não encontrado')
        
        # Verificar senha atual
        if self.users[username]['password'] != old_password:
            return False, self.i18n.t('auth.invalid_old_password', 'Senha atual incorreta')
        
        # Alterar senha
        self.users[username]['password'] = new_password
        
        self.logger.info(f"Senha alterada para usuário: {username}")
        return True, self.i18n.t('auth.password_changed', 'Senha alterada com sucesso')