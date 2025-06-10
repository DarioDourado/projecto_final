"""
🚀 Projeto Final - Sistema Modular
Dashboard Multilingual de Análise Salarial

Estrutura:
- auth/: Sistema de autenticação
- components/: Componentes de interface
- data/: Gestão de dados
- pages/: Páginas do dashboard  
- utils/: Utilitários e i18n
"""

__version__ = "5.0.0"
__author__ = "Dashboard Team"
__description__ = "Sistema Modular de Análise Salarial Multilingual"

# Imports principais
from .auth import AuthenticationSystem
from .utils import I18nSystem
from .data import load_data

__all__ = [
    'AuthenticationSystem',
    'I18nSystem', 
    'load_data'
]