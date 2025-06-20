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

# Versão do sistema
__version__ = "2.0.0"

# Configurações básicas
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Imports essenciais (removendo imports problemáticos)
try:
    from .pipelines.hybrid_pipeline import HybridPipelineSQL
    from .pipelines.hybrid_academic_pipeline import HybridAcademicPipeline
except ImportError as e:
    print(f"⚠️ Aviso: Alguns módulos não puderam ser importados: {e}")

# Configurações do sistema
DEFAULT_CONFIG = {
    'data_sources': ['sql', 'csv'],
    'algorithms': ['dbscan', 'apriori', 'fp_growth', 'eclat'],
    'output_dir': 'output',
    'log_level': 'INFO'
}

# Exportar principais classes
__all__ = [
    'HybridPipelineSQL',
    'HybridAcademicPipeline',
    'DEFAULT_CONFIG'
]