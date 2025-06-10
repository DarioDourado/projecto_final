"""
🛡️ Sistema de Permissões
Controle de acesso baseado em papéis (RBAC)
"""

from typing import List, Dict, Any, Optional
from enum import Enum
import streamlit as st
from src.utils.i18n import i18n

class Role(Enum):
    """Papéis do sistema"""
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"

class Permission(Enum):
    """Permissões do sistema"""
    # Visualização
    VIEW_DASHBOARD = "view_dashboard"
    VIEW_DATA = "view_data"
    VIEW_ANALYSIS = "view_analysis"
    VIEW_MODELS = "view_models"
    VIEW_REPORTS = "view_reports"
    
    # Execução
    EXECUTE_PIPELINE = "execute_pipeline"
    TRAIN_MODELS = "train_models"
    RUN_ANALYSIS = "run_analysis"
    
    # Administração
    MANAGE_USERS = "manage_users"
    MANAGE_SYSTEM = "manage_system"
    VIEW_LOGS = "view_logs"
    
    # Dados
    DOWNLOAD_DATA = "download_data"
    UPLOAD_DATA = "upload_data"
    DELETE_DATA = "delete_data"

class PermissionManager:
    """Gerenciador de permissões"""
    
    def __init__(self):
        # Mapeamento de papéis para permissões
        self.role_permissions = {
            Role.ADMIN: [
                Permission.VIEW_DASHBOARD,
                Permission.VIEW_DATA,
                Permission.VIEW_ANALYSIS,
                Permission.VIEW_MODELS,
                Permission.VIEW_REPORTS,
                Permission.EXECUTE_PIPELINE,
                Permission.TRAIN_MODELS,
                Permission.RUN_ANALYSIS,
                Permission.MANAGE_USERS,
                Permission.MANAGE_SYSTEM,
                Permission.VIEW_LOGS,
                Permission.DOWNLOAD_DATA,
                Permission.UPLOAD_DATA,
                Permission.DELETE_DATA
            ],
            Role.USER: [
                Permission.VIEW_DASHBOARD,
                Permission.VIEW_DATA,
                Permission.VIEW_ANALYSIS,
                Permission.VIEW_MODELS,
                Permission.VIEW_REPORTS,
                Permission.EXECUTE_PIPELINE,
                Permission.TRAIN_MODELS,
                Permission.RUN_ANALYSIS,
                Permission.DOWNLOAD_DATA
            ],
            Role.GUEST: [
                Permission.VIEW_DASHBOARD,
                Permission.VIEW_DATA,
                Permission.VIEW_ANALYSIS,
                Permission.VIEW_REPORTS
            ]
        }
        
        # Páginas do sistema e permissões necessárias
        self.page_permissions = {
            "📊 Visão Geral": [Permission.VIEW_DASHBOARD],
            "📈 Análise Exploratória": [Permission.VIEW_ANALYSIS],
            "🤖 Modelos ML": [Permission.VIEW_MODELS],
            "🎯 Clustering": [Permission.RUN_ANALYSIS],
            "📋 Regras de Associação": [Permission.RUN_ANALYSIS],
            "📊 Métricas Avançadas": [Permission.VIEW_MODELS],
            "🔮 Predição": [Permission.TRAIN_MODELS],
            "📁 Relatórios": [Permission.VIEW_REPORTS],
            "⚙️ Configurações": [Permission.MANAGE_SYSTEM]
        }
    
    def get_user_role(self) -> Optional[Role]:
        """Obter papel do usuário atual"""
        if 'user_data' in st.session_state and st.session_state.user_data:
            role_str = st.session_state.user_data.get('role', 'guest')
            try:
                return Role(role_str)
            except ValueError:
                return Role.GUEST
        return None
    
    def has_permission(self, permission: Permission) -> bool:
        """Verificar se usuário tem permissão específica"""
        user_role = self.get_user_role()
        
        if user_role is None:
            return False
        
        return permission in self.role_permissions.get(user_role, [])
    
    def has_any_permission(self, permissions: List[Permission]) -> bool:
        """Verificar se usuário tem qualquer uma das permissões"""
        return any(self.has_permission(perm) for perm in permissions)
    
    def has_all_permissions(self, permissions: List[Permission]) -> bool:
        """Verificar se usuário tem todas as permissões"""
        return all(self.has_permission(perm) for perm in permissions)
    
    def can_access_page(self, page_name: str) -> bool:
        """Verificar se usuário pode acessar página"""
        required_permissions = self.page_permissions.get(page_name, [])
        
        if not required_permissions:
            return True  # Página sem restrições
        
        return self.has_any_permission(required_permissions)
    
    def get_accessible_pages(self) -> List[str]:
        """Obter lista de páginas acessíveis ao usuário"""
        accessible_pages = []
        
        for page_name in self.page_permissions.keys():
            if self.can_access_page(page_name):
                accessible_pages.append(page_name)
        
        return accessible_pages
    
    def require_permission(self, permission: Permission, 
                          error_message: Optional[str] = None) -> bool:
        """
        Decorador/função para exigir permissão
        
        Args:
            permission: Permissão necessária
            error_message: Mensagem de erro personalizada
        
        Returns:
            True se tem permissão, False caso contrário
        """
        if not self.has_permission(permission):
            if error_message is None:
                error_message = i18n.t('auth.access_denied')
            
            st.error(error_message)
            st.stop()
            return False
        
        return True
    
    def require_role(self, required_role: Role, 
                    error_message: Optional[str] = None) -> bool:
        """
        Exigir papel específico
        
        Args:
            required_role: Papel necessário
            error_message: Mensagem de erro personalizada
        
        Returns:
            True se tem papel, False caso contrário
        """
        user_role = self.get_user_role()
        
        if user_role != required_role:
            if error_message is None:
                role_name = i18n.t(f'auth.role_{required_role.value}')
                error_message = i18n.t('auth.role_required', role=role_name)
            
            st.error(error_message)
            st.stop()
            return False
        
        return True
    
    def show_permission_denied(self, page_name: str):
        """Mostrar página de acesso negado"""
        st.error(i18n.t('auth.access_denied'))
        
        user_role = self.get_user_role()
        if user_role:
            st.info(
                i18n.t('auth.current_role_info', 
                      role=i18n.t(f'auth.role_{user_role.value}'))
            )
        
        # Sugerir páginas acessíveis
        accessible_pages = self.get_accessible_pages()
        if accessible_pages:
            st.info(i18n.t('auth.accessible_pages'))
            for page in accessible_pages[:5]:  # Mostrar apenas primeiras 5
                st.write(f"• {page}")
    
    def create_role_badge(self, role: Role) -> str:
        """Criar badge visual para papel"""
        badges = {
            Role.ADMIN: "🛡️ Admin",
            Role.USER: "👤 User", 
            Role.GUEST: "👁️ Guest"
        }
        return badges.get(role, "❓ Unknown")
    
    def get_permission_description(self, permission: Permission) -> str:
        """Obter descrição da permissão"""
        descriptions = {
            Permission.VIEW_DASHBOARD: i18n.t('permissions.view_dashboard'),
            Permission.VIEW_DATA: i18n.t('permissions.view_data'),
            Permission.VIEW_ANALYSIS: i18n.t('permissions.view_analysis'),
            Permission.VIEW_MODELS: i18n.t('permissions.view_models'),
            Permission.VIEW_REPORTS: i18n.t('permissions.view_reports'),
            Permission.EXECUTE_PIPELINE: i18n.t('permissions.execute_pipeline'),
            Permission.TRAIN_MODELS: i18n.t('permissions.train_models'),
            Permission.RUN_ANALYSIS: i18n.t('permissions.run_analysis'),
            Permission.MANAGE_USERS: i18n.t('permissions.manage_users'),
            Permission.MANAGE_SYSTEM: i18n.t('permissions.manage_system'),
            Permission.VIEW_LOGS: i18n.t('permissions.view_logs'),
            Permission.DOWNLOAD_DATA: i18n.t('permissions.download_data'),
            Permission.UPLOAD_DATA: i18n.t('permissions.upload_data'),
            Permission.DELETE_DATA: i18n.t('permissions.delete_data')
        }
        
        return descriptions.get(permission, permission.value)

# Instância global
permissions = PermissionManager()

# Decorador de conveniência
def require_permission(permission: Permission):
    """Decorador para exigir permissão"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if permissions.require_permission(permission):
                return func(*args, **kwargs)
            return None
        return wrapper
    return decorator

def require_role(role: Role):
    """Decorador para exigir papel"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if permissions.require_role(role):
                return func(*args, **kwargs)
            return None
        return wrapper
    return decorator