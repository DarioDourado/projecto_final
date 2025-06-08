"""Estilos modernos para visualizações"""

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ..config.settings import MODERN_COLORS, IMAGES_DIR
from ..utils.logger import get_logger

logger = get_logger(__name__)

class ModernStyle:
    """Classe para aplicar estilos modernos às visualizações"""
    
    @staticmethod
    def setup_matplotlib():
        """Configurar matplotlib com estilo moderno"""
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Configuração global
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'figure.dpi': 100,
            'figure.facecolor': 'white',
            'axes.facecolor': '#f8f9fa',
            'axes.edgecolor': '#dee2e6',
            'font.family': ['DejaVu Sans', 'Arial', 'sans-serif'],
            'font.size': 11,
            'text.color': '#343a40',
            'grid.color': '#e9ecef',
            'grid.alpha': 0.7,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight'
        })
        
        logger.info("✅ Estilo matplotlib configurado")
    
    @staticmethod
    def apply_modern_style(ax, title="", remove_spines=True):
        """Aplicar estilo moderno a um eixo"""
        if title:
            ax.set_title(title, fontsize=16, fontweight='bold',
                        color=MODERN_COLORS['dark'], pad=20)
        
        if remove_spines:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color(MODERN_COLORS['secondary'])
            ax.spines['bottom'].set_color(MODERN_COLORS['secondary'])
        
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_facecolor('#fafbfc')
        
        return ax
    
    @staticmethod
    def save_plot(filename, dpi=300, transparent=False):
        """Salvar gráfico com configurações modernas"""
        try:
            IMAGES_DIR.mkdir(parents=True, exist_ok=True)
            filepath = IMAGES_DIR / filename
            
            plt.tight_layout()
            plt.savefig(filepath, 
                       dpi=dpi, 
                       transparent=transparent,
                       bbox_inches='tight',
                       facecolor='white',
                       edgecolor='none',
                       pad_inches=0.2)
            
            logger.info(f"✅ Gráfico salvo: {filename}")
            plt.close()
            
        except Exception as e:
            logger.error(f"❌ Erro ao salvar {filename}: {e}")
            plt.close()