"""Estilos de visualização modernos - Extraído do projeto_salario.py"""

import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# Paleta de cores moderna (extraída do projeto_salario.py)
MODERN_COLORS = {
    'primary': '#007bff',
    'secondary': '#6c757d', 
    'success': '#28a745',
    'danger': '#dc3545',
    'warning': '#ffc107',
    'info': '#17a2b8',
    'light': '#f8f9fa',
    'dark': '#343a40',
    'gradient_blue': ['#667eea', '#764ba2'],
    'gradient_sunset': ['#ff9a9e', '#fecfef'],
    'gradient_ocean': ['#2196f3', '#21cbf3'],
    'categorical': ['#007bff', '#28a745', '#ffc107', '#dc3545', '#17a2b8', '#6f42c1', '#fd7e14']
}

def setup_matplotlib_style():
    """Configurar estilo global do matplotlib"""
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Configuração global (extraída do projeto_salario.py)
    plt.rcParams.update({
        # Figura
        'figure.figsize': (12, 8),
        'figure.dpi': 100,
        'figure.facecolor': 'white',
        'figure.edgecolor': 'none',
        
        # Eixos
        'axes.facecolor': '#f8f9fa',
        'axes.edgecolor': '#dee2e6',
        'axes.linewidth': 1.2,
        'axes.grid': True,
        'axes.axisbelow': True,
        'axes.labelsize': 12,
        'axes.titlesize': 16,
        'axes.titleweight': 'bold',
        'axes.titlepad': 20,
        
        # Grid
        'grid.color': '#e9ecef',
        'grid.linestyle': '-',
        'grid.linewidth': 0.8,
        'grid.alpha': 0.7,
        
        # Texto - SEM FONTES PROBLEMÁTICAS
        'font.family': ['DejaVu Sans', 'Arial', 'sans-serif'],
        'font.size': 11,
        'text.color': '#343a40',
        
        # Legendas
        'legend.fontsize': 10,
        'legend.frameon': True,
        'legend.fancybox': True,
        'legend.shadow': True,
        'legend.framealpha': 0.9,
        'legend.facecolor': 'white',
        'legend.edgecolor': '#dee2e6',
        
        # Ticks
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'xtick.color': '#6c757d',
        'ytick.color': '#6c757d',
        
        # Cores
        'patch.linewidth': 0.5,
        'patch.facecolor': '#007bff',
        'patch.edgecolor': '#0056b3',
        
        # Salvamento
        'savefig.dpi': 300,
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'none',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.2
    })

def apply_modern_style(ax, title="", subtitle="", remove_spines=True):
    """Aplicar estilo moderno consistente a um gráfico"""
    # Título principal (SEM EMOJIS PROBLEMÁTICOS)
    if title:
        ax.set_title(title, fontsize=16, fontweight='bold', 
                    color=MODERN_COLORS['dark'], pad=20)
    
    # Subtítulo
    if subtitle:
        ax.text(0.5, 0.95, subtitle, transform=ax.transAxes, 
               fontsize=11, color=MODERN_COLORS['secondary'],
               ha='center', style='italic')
    
    # Remover spines desnecessários
    if remove_spines:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(MODERN_COLORS['secondary'])
        ax.spines['bottom'].set_color(MODERN_COLORS['secondary'])
    
    # Grid sutil
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Cor de fundo
    ax.set_facecolor('#fafbfc')
    
    return ax

def save_modern_plot(filename, dpi=300, transparent=False):
    """Salvar gráfico com configurações modernas na estrutura correta"""
    try:
        # Usar estrutura output/images/
        project_root = Path(__file__).parent.parent.parent
        images_dir = project_root / "output" / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        plt.tight_layout()
        plt.savefig(images_dir / filename, 
                    dpi=dpi, 
                    transparent=transparent,
                    bbox_inches='tight',
                    facecolor='white',
                    edgecolor='none',
                    pad_inches=0.2)
        print(f"✓ Gráfico salvo: {images_dir / filename}")
        plt.close()
    except Exception as e:
        print(f"✗ Erro ao salvar {filename}: {e}")
        plt.close()