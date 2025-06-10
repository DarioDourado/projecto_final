"""
📱 Páginas do Dashboard
Módulo contendo todas as páginas do sistema
"""

from .login import show_login_page
from .overview import show_overview_page
from .exploratory import show_exploratory_page
from .models import show_models_page
from .prediction import show_prediction_page

__all__ = [
    'show_login_page',
    'show_overview_page', 
    'show_exploratory_page',
    'show_models_page',
    'show_prediction_page'
]