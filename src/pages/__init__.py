"""
Módulo de páginas do dashboard
"""

from .overview import show_overview_page
from .exploratory import show_exploratory_page
from .models import show_models_page
from .admin import show_admin_page
from .prediction import show_prediction_page 

__all__ = [
    'show_overview_page',
    'show_exploratory_page', 
    'show_models_page',
    'show_admin_page',
    'show_prediction_page' 
]